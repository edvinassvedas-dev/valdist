"""Quote retrieval is confined to `prices/`."""

from __future__ import annotations

import ast
import datetime as _dt
import pathlib

import pytest
from _scan import python_files

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PRICES_ROOT = REPO_ROOT / "prices"

#: The repo's own Python packages. Tests are excluded deliberately - a test may
#: legitimately reference a forbidden module name in a string or an assertion.
OWN_PACKAGES = ("valdist", "gui", "prices")

#: Same list applies to `valdist/`. `socket` is omitted for the same reason
#: `gui/` omits it: serving a local page is not retrieval, and banning
#: it would ban the viewer's own stdlib server while stopping nothing.
NETWORK_MODULES = {
    "requests",
    "httpx",
    "urllib.request",
    "aiohttp",
    "yfinance",
    "pandas_datareader",
    "alpha_vantage",
}


def _imported_modules(py_file: pathlib.Path) -> set[str]:
    """Every module named by an `import x` / `from x import ...` in one file."""
    tree = ast.parse(py_file.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _python_files(package: str):
    """Every module in one of the repo's own packages, never an empty list."""
    return python_files(REPO_ROOT / package, f"{package}/")


# --------------------------------------------------------------------------- #
# 1. Retrieval lives in exactly one directory
# --------------------------------------------------------------------------- #


def test_network_imports_appear_only_under_prices() -> None:
    """`prices/` may fetch. Nothing else in the repo may."""
    offenders: dict[str, list[str]] = {}
    for package in OWN_PACKAGES:
        if package == "prices":
            continue
        for py_file in _python_files(package):
            hits = {
                m
                for m in _imported_modules(py_file)
                if m in NETWORK_MODULES or m.split(".")[0] in NETWORK_MODULES
            }
            if hits:
                offenders[str(py_file.relative_to(REPO_ROOT))] = sorted(hits)

    assert offenders == {}, (
        f"network/data-retrieval imports outside prices/: {offenders}. "
        "Retrieval is confined to prices/; everything else consumes its output."
    )


# --------------------------------------------------------------------------- #
# 2. Nothing can reach the fetcher (the load-bearing half)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("package", ["valdist", "gui"])
def test_engine_and_viewer_never_import_prices(package: str) -> None:
    """The viewer reads the artefact; it cannot trigger a fetch."""
    offenders = {}
    for py_file in _python_files(package):
        hits = {m for m in _imported_modules(py_file) if m == "prices" or m.startswith("prices.")}
        if hits:
            offenders[str(py_file.relative_to(REPO_ROOT))] = sorted(hits)

    assert offenders == {}, (
        f"{package}/ imports the price fetcher: {offenders}. "
        f"{package}/ may read prices/'s output file; it may not call it."
    )


# --------------------------------------------------------------------------- #
# 3. The fetcher is a producer, not a library the rest of the repo builds on
# --------------------------------------------------------------------------- #


# Coverage and dead-key checks for the symbol table are not here: they are
# claims about private data, so they live in `tests/test_symbols_table.py` and
# skip when no record is present. Table shape validation
# (`<TICKER>.<EXCHANGE>`, no overlap between the maps) lives in
# `prices.fetch.load_symbols` itself, so it holds for any table anyone loads.
#
# What remains in this file is the half that is always about code: the import
# bans below.


@pytest.mark.skipif(not (REPO_ROOT / "prices").is_dir(), reason="prices/ not present")
def test_prices_does_not_import_the_engine() -> None:
    """`prices/` produces an artefact; it does not depend on valdist."""
    offenders = {}
    for py_file in _python_files("prices"):
        hits = {m for m in _imported_modules(py_file) if m == "valdist" or m.startswith("valdist.")}
        if hits:
            offenders[str(py_file.relative_to(REPO_ROOT))] = sorted(hits)

    assert offenders == {}, (
        f"prices/ imports valdist: {offenders}. It is a standalone producer - "
        "keeping it independent is what lets it be replaced or run on its own."
    )


# ---------------------------------------------------------------------------
# Quote selection: the most recent settled close, never a live intraday tick.
#
# `fetch_one` used to read only the daily `close` array and take the last
# non-null entry. Yahoo does not always consolidate the newest daily bar
# immediately, and when it has not, the *same payload* carries the finished
# close in `meta.regularMarketPrice`. The result was a silently stale quote.
#
# Measured on A45, 2026-07-27: the 24 July bar was null while meta held that
# day's 39.00 close, so `prices.json` recorded 35.98 as of 23 July - the price
# from before the 23 July earnings release and the 8.4% move it caused. Feeding
# the shipped A45 spec 35.98 instead of 39.00 moves `p_undervalued` from 0.2274
# to 0.3271. A one-day-stale price is usually harmless; this one straddled the
# event, which is exactly when it is not.
#
# The fix must not cost the property the docstring promises: the value returned
# is a close, not a live tick, so two runs on the same day agree. `meta` is
# therefore trusted only when the session its trade belongs to has finished -
# either because the trade predates the current session, or because that session
# has already ended (`now >= regular.end`, the same discriminator `_is_live_bar`
# uses on the daily bar).
#
# That second branch is the 2026-07-31 correction. The rule used to be "refuse
# anything dated on the session day, whether the session is open, not yet open,
# or just shut" - deliberate, and harmless while the bar carries the same number,
# but it bites in exactly the shape this section exists for: the A45 case after
# the bell, where the newest bar is null and meta holds that day's settled close.
# The old rule threw that close away and dated the quote to the previous session,
# which is the very defect the meta fallback was written to fix.
# ---------------------------------------------------------------------------

US_OPEN_UTC = _dt.time(13, 30)
US_CLOSE_UTC = _dt.time(20, 0)


def _epoch(day: str, at: _dt.time) -> int:
    d = _dt.date.fromisoformat(day)
    return int(_dt.datetime.combine(d, at, tzinfo=_dt.UTC).timestamp())


def _chart(
    bars,
    *,
    meta_price=None,
    meta_day=None,
    meta_at=US_CLOSE_UTC,
    session_day="2026-07-27",
    session_end=True,
    gmtoffset=None,
):
    """A Yahoo chart `result[0]`, shaped like the real one."""
    meta = {"currency": "USD"}
    if gmtoffset is not None:
        meta["gmtoffset"] = gmtoffset
    if meta_price is not None:
        meta["regularMarketPrice"] = meta_price
        meta["regularMarketTime"] = _epoch(meta_day, meta_at)
    if session_day is not None:
        regular = {"start": _epoch(session_day, US_OPEN_UTC)}
        if session_end:
            regular["end"] = _epoch(session_day, US_CLOSE_UTC)
        meta["currentTradingPeriod"] = {"regular": regular}
    return {
        "meta": meta,
        "timestamp": [_epoch(day, US_OPEN_UTC) for day, _ in bars],
        "indicators": {"quote": [{"close": [close for _, close in bars]}]},
    }


def test_unconsolidated_final_bar_falls_back_to_the_settled_meta_close() -> None:
    """The A45 case: newest bar is null, meta holds that day's finished close."""
    from prices.fetch import select_quote

    res = _chart(
        [("2026-07-22", 36.77), ("2026-07-23", 35.98), ("2026-07-24", None)],
        meta_price=39.0,
        meta_day="2026-07-24",
    )
    quote = select_quote(res)
    assert quote is not None
    assert quote["price"] == 39.0, (
        "took the stale 23 July bar instead of the settled 24 July close sitting "
        "in meta of the same payload - this is the shipped A45 defect."
    )
    assert quote["as_of"] == "2026-07-24"


def test_same_day_meta_close_is_taken_once_the_session_has_ended() -> None:
    """The A45 case after the bell: null final bar, meta holds today's close."""
    from prices.fetch import select_quote

    res = _chart(
        [("2026-07-24", 35.98), ("2026-07-27", None)],
        meta_price=39.0,
        meta_day="2026-07-27",
        session_day="2026-07-27",
    )
    quote = select_quote(res, now=_epoch("2026-07-27", _dt.time(21, 0)))
    assert quote is not None
    assert quote["price"] == 39.0, (
        "refused a settled same-day close an hour after the bell and fell back "
        "to the previous session's bar - the stale quote this fallback exists "
        "to prevent."
    )
    assert quote["as_of"] == "2026-07-27"
    assert quote["live"] is False, "a finished session's close is not moving"


def test_same_day_meta_close_needs_a_session_end_to_prove_settlement() -> None:
    """A trading period with a start but no end proves nothing, so refuse."""
    from prices.fetch import select_quote

    res = _chart(
        [("2026-07-24", 35.98), ("2026-07-27", None)],
        meta_price=39.0,
        meta_day="2026-07-27",
        session_day="2026-07-27",
        session_end=False,
    )
    quote = select_quote(res, now=_epoch("2026-07-27", _dt.time(21, 0)))
    assert quote is not None
    assert quote["price"] == 35.98, "trusted meta without being able to prove the session ended"
    assert quote["as_of"] == "2026-07-24"


def test_live_intraday_tick_is_never_used() -> None:
    """Reproducibility: mid-session, meta is a moving tick and must be ignored."""
    from prices.fetch import select_quote

    res = _chart(
        [("2026-07-23", 35.98), ("2026-07-24", 39.0), ("2026-07-27", None)],
        meta_price=40.5,
        meta_day="2026-07-27",
        meta_at=_dt.time(17, 0),  # inside the 2026-07-27 regular session
    )
    quote = select_quote(res, now=_epoch("2026-07-27", _dt.time(17, 0)))
    assert quote is not None
    assert quote["price"] == 39.0, (
        "used a live intraday tick. Two runs on the same day would then "
        "disagree, which is the property fetch_one's docstring promises."
    )
    assert quote["as_of"] == "2026-07-24"


def test_meta_stamped_before_the_opening_bell_is_not_trusted() -> None:
    """A trade dated on the session's own day is refused even if it predates
    the open.
    """
    from prices.fetch import select_quote

    res = _chart(
        [("2026-07-24", 63.0), ("2026-07-27", None)],
        meta_price=63.0,
        meta_day="2026-07-27",
        meta_at=_dt.time(6, 55),  # before the 2026-07-27 open
    )
    quote = select_quote(res, now=_epoch("2026-07-27", _dt.time(6, 55)))
    assert quote is not None
    assert quote["as_of"] == "2026-07-24", (
        "dated a pre-open tick as the current day, overstating freshness"
    )
    assert quote["price"] == 63.0


def test_consolidated_bar_wins_over_meta_of_the_same_day() -> None:
    """No regression: when the newest bar exists, it is the authority."""
    from prices.fetch import select_quote

    res = _chart(
        [("2026-07-23", 63.4), ("2026-07-24", 63.0)],
        meta_price=62.5,
        meta_day="2026-07-24",
    )
    quote = select_quote(res)
    assert quote is not None
    assert quote["price"] == 63.0
    assert quote["as_of"] == "2026-07-24"


def test_meta_older_than_the_newest_bar_is_ignored() -> None:
    from prices.fetch import select_quote

    res = _chart(
        [("2026-07-23", 35.98), ("2026-07-24", 39.0)],
        meta_price=35.98,
        meta_day="2026-07-23",
    )
    quote = select_quote(res)
    assert quote is not None
    assert quote["price"] == 39.0
    assert quote["as_of"] == "2026-07-24"


def test_meta_without_a_trading_period_is_not_trusted() -> None:
    """Cannot prove the session is finished, so do not guess: keep the bar."""
    from prices.fetch import select_quote

    res = _chart(
        [("2026-07-23", 35.98), ("2026-07-24", None)],
        meta_price=39.0,
        meta_day="2026-07-24",
        session_day=None,
    )
    quote = select_quote(res)
    assert quote is not None
    assert quote["price"] == 35.98
    assert quote["as_of"] == "2026-07-23"


def test_no_usable_close_anywhere_returns_none() -> None:
    from prices.fetch import select_quote

    assert select_quote(_chart([("2026-07-23", None)])) is None


# ---------------------------------------------------------------------------
# A live daily bar is flagged, not guessed at.
#
# `fetch_one`'s docstring used to record, as an accepted limit, that it "cannot
# tell a live daily bar from a finished one - the payload does not distinguish
# them". Measured across six markets in three session states on 2026-07-30, the
# payload distinguishes them exactly: `currentTradingPeriod.regular.end` plus the
# current time separates a moving bar from a settled one, and it is the `.end`
# the old code never read. The three states, all observed live:
#   VICI / A25.TO  mid-session        bar dated the session day, now < end  LIVE
#   A47.L / A03.BR  minutes after the  bar dated the session day, now >= end SETTLED
#                    bell
#   7203.T / BHP.AX  next session queued bar older than the session day        SETTLED
# The middle row is why the meta rule could not simply be reused: it refuses
# anything dated on the session day, which would have thrown away every European
# close each evening and left the fetcher a day stale.
#
# Measured blast radius at 17:27 UTC on 2026-07-30: 16 of 28 rows in the shipped
# `prices.json` were moving values written as though settled, the widest 6.38%
# from the close they were about to be compared against (A40 10.159 vs 9.550).
#
# The flag rather than a refusal is a deliberate choice: the artefact keeps the
# intraday price, because that is what `prices/` exists to provide, and says so.
# ---------------------------------------------------------------------------


def test_a_live_intraday_bar_is_flagged() -> None:
    """Mid-session, the newest bar is still moving and must say so."""
    from prices.fetch import select_quote

    res = _chart([("2026-07-24", 39.0), ("2026-07-27", 40.5)], session_day="2026-07-27")
    quote = select_quote(res, now=_epoch("2026-07-27", _dt.time(17, 0)))
    assert quote is not None
    assert quote["price"] == 40.5, "the intraday price is kept - see the header above"
    assert quote["live"] is True, (
        "wrote a moving intraday bar as though it were a settled close. This is "
        "the 16-of-28 defect measured on 2026-07-30."
    )


def test_a_bar_from_a_finished_session_is_not_flagged() -> None:
    """Minutes after the bell the same bar is settled. The A47.L / A03.BR row."""
    from prices.fetch import select_quote

    res = _chart([("2026-07-24", 39.0), ("2026-07-27", 40.5)], session_day="2026-07-27")
    quote = select_quote(res, now=_epoch("2026-07-27", _dt.time(21, 0)))
    assert quote is not None
    assert quote["price"] == 40.5
    assert quote["live"] is False, "marked a finished session's close as live"


def test_a_bar_from_a_previous_session_is_not_flagged() -> None:
    """The next session is queued and the newest bar predates it."""
    from prices.fetch import select_quote

    res = _chart([("2026-07-23", 35.98), ("2026-07-24", 39.0)], session_day="2026-07-27")
    quote = select_quote(res, now=_epoch("2026-07-27", _dt.time(12, 0)))
    assert quote is not None
    assert quote["live"] is False


def test_liveness_needs_a_session_end_to_prove_it() -> None:
    """A trading period with a start but no end proves nothing, so claim nothing."""
    from prices.fetch import select_quote

    res = _chart([("2026-07-27", 40.5)], session_day="2026-07-27", session_end=False)
    quote = select_quote(res, now=_epoch("2026-07-27", _dt.time(17, 0)))
    assert quote is not None
    assert quote["live"] is False, "claimed liveness from a trading period with no end"


def test_liveness_needs_a_trading_period_at_all() -> None:
    """No trading period, no proof - the conservative branch, and the one that
    keeps the old behaviour when the payload shape is unfamiliar."""
    from prices.fetch import select_quote

    res = _chart([("2026-07-27", 40.5)], session_day=None)
    quote = select_quote(res, now=_epoch("2026-07-27", _dt.time(17, 0)))
    assert quote is not None
    assert quote["live"] is False


def test_a_quote_taken_from_meta_is_never_live() -> None:
    """`_settled_meta_close` only ever returns a finished session's close, so a
    quote sourced from it is settled by construction."""
    from prices.fetch import select_quote

    res = _chart(
        [("2026-07-23", 35.98), ("2026-07-24", None)],
        meta_price=39.0,
        meta_day="2026-07-24",
        session_day="2026-07-27",
    )
    quote = select_quote(res, now=_epoch("2026-07-27", _dt.time(17, 0)))
    assert quote is not None
    assert quote["price"] == 39.0
    assert quote["live"] is False


def test_fetch_one_propagates_the_live_flag() -> None:
    """The flag has to reach the artefact, not just the pure function."""
    import contextlib
    import io
    import json as _json

    from prices import fetch as fetch_mod

    payload = {
        "chart": {
            "result": [_chart([("2026-07-24", 39.0), ("2026-07-27", 40.5)])],
        }
    }

    @contextlib.contextmanager
    def fake_urlopen(req, timeout=None):
        yield io.BytesIO(_json.dumps(payload).encode())

    original = fetch_mod.urllib.request.urlopen
    fetch_mod.urllib.request.urlopen = fake_urlopen
    try:
        # Explicit table: this case tests quote selection, not classification, and
        # the symbol map moved to the private record in 2026-09-04's batch 3b.
        row = fetch_mod.fetch_one("A01", symbols=({}, {"A01"}))
    finally:
        fetch_mod.urllib.request.urlopen = original

    assert "live" in row, "fetch_one dropped the live flag on the way to the artefact"


def test_fetch_one_delegates_to_select_quote() -> None:
    """The pure function must be on the real path, not beside it."""
    import contextlib
    import io
    import json as _json

    from prices import fetch as fetch_mod

    payload = {
        "chart": {
            "result": [
                _chart(
                    [("2026-07-23", 35.98), ("2026-07-24", None)],
                    meta_price=39.0,
                    meta_day="2026-07-24",
                )
            ]
        }
    }

    @contextlib.contextmanager
    def fake_urlopen(req, timeout=None):
        yield io.BytesIO(_json.dumps(payload).encode())

    original = fetch_mod.urllib.request.urlopen
    fetch_mod.urllib.request.urlopen = fake_urlopen
    try:
        row = fetch_mod.fetch_one("A45", symbols=({}, {"A45"}))
    finally:
        fetch_mod.urllib.request.urlopen = original

    assert row["price"] == 39.0, "fetch_one is not routing through select_quote"
    assert row["as_of"] == "2026-07-24"
    assert row["ticker"] == "A45"
    assert row["currency"] == "USD"


# ---------------------------------------------------------------------------
# A session that spans UTC midnight is still one trading day.
#
# Both rules above key on a date: the bar (or meta trade) belongs to the current
# trading period when its date equals `regular.start`'s. Taken in UTC, that
# silently assumes an exchange whose regular session opens and closes inside one
# UTC day - true of most western venues (US, TSX, LSE, Euronext, Madrid,
# Amsterdam) and false of any venue far enough east.
#
# Measured live on 2026-07-31, FPH.NZ (NZX, `gmtoffset` 43200):
#
#   regular.start  2026-07-30T22:00:00Z   = 10:00 NZST on the 31 July session
#   regular.end    2026-07-31T05:00:00Z   = 17:00 NZST on the same session
#   bars           2026-07-29T22:00Z 40.77   (the 30 July NZ session)
#                  2026-07-30T22:00Z None    (the 31 July session, unconsolidated)
#                  2026-07-31T05:00:10Z 40.69 (the last trade of that session)
#
# So one NZ trading day carries two different UTC dates, and the rules read the
# session as 07-30 while reading its own bars as 07-29 and 07-31. A mid-session
# NZ bar is therefore never flagged live, the meta fallback is refused outright
# and can never fire, and a settled bar is dated to the UTC day its session
# opened on rather than the day it traded.
#
# The fix compares dates in the exchange's own day, using the `gmtoffset` Yahoo
# already sends (verified present: FPH.NZ 43200, VICI -14400, A47.L 3600).
#
# Rejected: interval containment on raw timestamps (`start <= ts <= end`),
# which needs no timezone arithmetic at all. It fails on the very payload that
# motivates the change - that 05:00:10Z bar is stamped ten seconds past a
# 05:00:00Z end - so it would need an arbitrary grace window, and a tolerance
# invented to make a check pass is one built not to fire (2026-07-25).
#
# Nothing in the shipped record is affected today. That is why these fixtures
# are synthetic where the section above uses live shapes: the defect is real
# and measured, but nothing shipped exercises it.
# ---------------------------------------------------------------------------

NZ_GMTOFFSET = 12 * 3600
NZ_SESSION = ("2026-07-30T22:00:00", "2026-07-31T05:00:00")


def _utc(stamp: str) -> int:
    return int(_dt.datetime.fromisoformat(stamp).replace(tzinfo=_dt.UTC).timestamp())


def _straddling_chart(bars, *, meta=None, session=NZ_SESSION):
    """A chart whose regular session spans UTC midnight, shaped from FPH.NZ."""
    m = {"currency": "NZD", "gmtoffset": NZ_GMTOFFSET}
    if meta is not None:
        m["regularMarketPrice"], m["regularMarketTime"] = meta[0], _utc(meta[1])
    m["currentTradingPeriod"] = {"regular": {"start": _utc(session[0]), "end": _utc(session[1])}}
    return {
        "meta": m,
        "timestamp": [_utc(stamp) for stamp, _ in bars],
        "indicators": {"quote": [{"close": [close for _, close in bars]}]},
    }


def test_a_live_bar_is_flagged_when_the_session_spans_utc_midnight() -> None:
    """Mid-session on the NZX: the running bar's UTC date is the day after the
    session's own, so a UTC comparison reads it as belonging to an earlier
    period and calls a moving price settled."""
    from prices.fetch import select_quote

    res = _straddling_chart(
        [("2026-07-29T22:00:00", 40.77), ("2026-07-31T02:00:00", 40.69)],
    )
    quote = select_quote(res, now=_utc("2026-07-31T02:30:00"))
    assert quote is not None
    assert quote["price"] == 40.69
    assert quote["live"] is True, (
        "called a mid-session bar settled because the session opened on the "
        "previous UTC day - the 2026-07-30 defect, reopened by timezone"
    )


def test_settled_meta_close_is_taken_when_the_session_spans_utc_midnight() -> None:
    """The A45 shape on a straddling venue: the meta fallback could never fire."""
    from prices.fetch import select_quote

    res = _straddling_chart(
        [("2026-07-29T22:00:00", 40.77), ("2026-07-30T22:00:00", None)],
        meta=(40.69, "2026-07-31T05:00:10"),
    )
    quote = select_quote(res, now=_utc("2026-07-31T06:00:00"))
    assert quote is not None
    assert quote["price"] == 40.69, (
        "refused a settled close an hour after the NZX bell and fell back to "
        "the previous session's bar"
    )
    assert quote["as_of"] == "2026-07-31"
    assert quote["live"] is False


def test_a_quote_is_dated_by_the_exchanges_trading_day() -> None:
    """A bar stamped 22:00Z belongs to the next day's session, and says so."""
    from prices.fetch import select_quote

    res = _straddling_chart([("2026-07-29T22:00:00", 40.77)])
    quote = select_quote(res, now=_utc("2026-07-31T06:00:00"))
    assert quote is not None
    assert quote["as_of"] == "2026-07-30", (
        "dated the quote by the UTC day its session opened on rather than the "
        "exchange's own trading day"
    )


def test_an_offset_that_does_not_cross_midnight_changes_nothing() -> None:
    """The regression guard: every shipped venue keeps its current answers."""
    from prices.fetch import select_quote

    bars = [("2026-07-24", 39.0), ("2026-07-27", 40.5)]
    live = select_quote(
        _chart(bars, session_day="2026-07-27", gmtoffset=-4 * 3600),
        now=_epoch("2026-07-27", _dt.time(17, 0)),
    )
    settled = select_quote(
        _chart(bars, session_day="2026-07-27", gmtoffset=-4 * 3600),
        now=_epoch("2026-07-27", _dt.time(21, 0)),
    )
    assert live == {"price": 40.5, "as_of": "2026-07-27", "currency": "USD", "live": True}
    assert settled == {"price": 40.5, "as_of": "2026-07-27", "currency": "USD", "live": False}
