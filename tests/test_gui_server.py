"""Behavioural tests for `gui/server.py`, as distinct from the static AST scan."""

from __future__ import annotations

import datetime as dt
import pathlib
import re
import time

import pytest

from gui.server import _ticker_of

# --------------------------------------------------------------------------- #
# A directory name is not a ticker until it looks like one
# --------------------------------------------------------------------------- #
#
# The pattern used to capture `(.+)` after the date prefix and the result was
# formatted straight into a quote URL. Not remotely exploitable - the
# trigger is a directory you named yourself and the host is a fixed https literal
# - but it is an unvalidated string reaching a URL, and the `?` case silently
# truncated the intended query parameters, so a legitimately odd symbol would
# have failed confusingly rather than loudly.


@pytest.mark.parametrize(
    ("label", "name"),
    [
        ("path traversal", "2026-07-25-../etc/passwd"),
        ("query injection", "2026-07-25-A?x=1"),
        ("fragment", "2026-07-25-A#frag"),
        ("whitespace", "2026-07-25-A B"),
        ("slash", "2026-07-25-a/b"),
        ("empty ticker", "2026-07-25-"),
        ("too long", "2026-07-25-" + "A" * 13),
        ("not date-prefixed", "notes"),
        ("bad date", "26-07-25-A12"),
    ],
)
def test_a_name_that_is_not_a_ticker_yields_no_ticker(label: str, name: str) -> None:
    """Anything unrecognised is simply "not a ticker", and gets no quote."""
    assert _ticker_of(name) is None, f"{label}: {name!r} should not parse as a ticker"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("2026-07-25-A12", "A12"),
        ("2026-07-26-A51", "A51"),
        ("2026-07-13-A01", "A01"),
        ("2026-07-24-A22", "A22"),
        ("2026-07-25-BRK.B", "BRK.B"),  # dotted classes are real symbols
        ("2026-07-25-RDS-A", "RDS-A"),  # so are hyphenated ones
    ],
)
def test_a_real_analysis_folder_still_yields_its_ticker(name: str, expected: str) -> None:
    """The allowlist must not break the shipped set or plausible future names."""
    assert _ticker_of(name) == expected


# --------------------------------------------------------------------------- #
# The report schema is a contract, not a footer
# --------------------------------------------------------------------------- #
#
# `report_schema_version` was emitted by the engine, asserted to exist, and
# rendered in a footer by the viewer - and compared against nothing anywhere.
# A guarantee with nothing watching it is not yet a guarantee. Without a
# comparison, a bumped version would render missing or renamed fields as
# blanks rather than refusing a payload it does not understand.


def test_the_viewer_declares_the_schema_it_was_written_against() -> None:
    """And it must equal the engine's today."""
    from gui.server import SUPPORTED_REPORT_SCHEMA
    from valdist.report import REPORT_SCHEMA_VERSION

    assert SUPPORTED_REPORT_SCHEMA == REPORT_SCHEMA_VERSION, (
        "the engine's report schema moved and the viewer was not revisited. "
        "Check gui/index.html still reads every field it renders, then update "
        "SUPPORTED_REPORT_SCHEMA."
    )


def test_a_matching_payload_is_reported_as_supported() -> None:
    from gui.server import SUPPORTED_REPORT_SCHEMA, check_report_schema

    status = check_report_schema({"report_schema_version": SUPPORTED_REPORT_SCHEMA})
    assert status["supported"] is True
    assert status["expected"] == SUPPORTED_REPORT_SCHEMA
    assert status["got"] == SUPPORTED_REPORT_SCHEMA


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        ("newer major", {"report_schema_version": "2.0"}),
        ("older", {"report_schema_version": "0.9"}),
        ("field absent", {}),
        ("field null", {"report_schema_version": None}),
        ("not a string", {"report_schema_version": 1.0}),
    ],
)
def test_a_payload_the_viewer_does_not_understand_is_flagged(label: str, payload: dict) -> None:
    """Including the absent case: "no version" is not "fine"."""
    from gui.server import check_report_schema

    status = check_report_schema(payload)
    assert status["supported"] is False, f"{label} should not be reported as supported"


def test_do_run_carries_the_schema_check_to_the_client(tmp_path) -> None:
    """The check has to reach the UI, or it is another gate with no caller."""
    from gui.server import ANALYSES_DIR, SUPPORTED_REPORT_SCHEMA, do_run

    name = sorted(p.name for p in ANALYSES_DIR.iterdir() if p.is_dir())[0]
    out = do_run(name)

    assert out["schema"]["supported"] is True
    assert out["schema"]["expected"] == SUPPORTED_REPORT_SCHEMA
    assert out["payload"]["report_schema_version"] == SUPPORTED_REPORT_SCHEMA


def test_an_engine_bump_is_seen_by_do_run(monkeypatch, tmp_path) -> None:
    """End to end: move the engine's version and the response says unsupported."""
    import valdist.report as vr
    from gui.server import ANALYSES_DIR, do_run

    monkeypatch.setattr(vr, "REPORT_SCHEMA_VERSION", "99.0")
    name = sorted(p.name for p in ANALYSES_DIR.iterdir() if p.is_dir())[0]
    out = do_run(name)

    assert out["schema"]["supported"] is False
    assert out["schema"]["got"] == "99.0"


# --------------------------------------------------------------------------- #
# The portfolio cache is bounded, and cold misses are not computed twice
# --------------------------------------------------------------------------- #
#
# Two related fixes, together because they are the same few lines. The
# cache key includes the quoted price, so every refresh at a new price added a
# full set of rows and nothing was ever evicted (measured: 0 -> 13 -> 26 -> 39 ->
# 52 across four price variants). And the lookup was check-then-compute with no
# lock, so three simultaneous cold requests each ran the whole set: 26.5s / 26.5s
# / 27.0s against ~9s for one.


def test_the_cache_evicts_instead_of_growing_without_bound() -> None:
    """Least-recently-used, so a long-lived viewer cannot grow forever."""
    from gui.server import _PORTFOLIO_CACHE_MAX, _cached_row, clear_portfolio_cache

    clear_portfolio_cache()
    for i in range(_PORTFOLIO_CACHE_MAX + 25):
        _cached_row(("k", i), lambda i=i: {"row": i})

    from gui.server import _PORTFOLIO_CACHE

    assert len(_PORTFOLIO_CACHE) == _PORTFOLIO_CACHE_MAX
    assert ("k", 0) not in _PORTFOLIO_CACHE, "the oldest entry should have been evicted"
    assert ("k", _PORTFOLIO_CACHE_MAX + 24) in _PORTFOLIO_CACHE, "the newest must survive"


def test_a_hit_is_refreshed_so_eviction_is_least_recently_used() -> None:
    """Otherwise the row you look at every visit is the one that gets dropped."""
    from gui.server import (
        _PORTFOLIO_CACHE,
        _PORTFOLIO_CACHE_MAX,
        _cached_row,
        clear_portfolio_cache,
    )

    clear_portfolio_cache()
    for i in range(_PORTFOLIO_CACHE_MAX):
        _cached_row(("k", i), lambda i=i: {"row": i})

    _cached_row(("k", 0), lambda: {"row": "recomputed"})  # a hit, not a miss
    _cached_row(("k", "new"), lambda: {"row": "new"})

    assert ("k", 0) in _PORTFOLIO_CACHE, "a recently used key must not be the eviction victim"
    assert ("k", 1) not in _PORTFOLIO_CACHE


def test_a_cached_key_is_not_recomputed() -> None:
    from gui.server import _cached_row, clear_portfolio_cache

    clear_portfolio_cache()
    calls = []

    def compute():
        calls.append(1)
        return {"row": 1}

    assert _cached_row(("k", 1), compute) == {"row": 1}
    assert _cached_row(("k", 1), compute) == {"row": 1}
    assert len(calls) == 1


def test_concurrent_cold_misses_compute_once() -> None:
    """Three threads, one cold key, one computation."""
    import threading

    from gui.server import _cached_row, clear_portfolio_cache

    clear_portfolio_cache()
    calls, started = [], threading.Event()

    def compute():
        calls.append(1)
        started.set()
        time.sleep(0.3)  # long enough that the others are certainly waiting
        return {"row": "once"}

    results, threads = [], []
    for _ in range(3):
        t = threading.Thread(target=lambda: results.append(_cached_row(("k", "cold"), compute)))
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(calls) == 1, f"computed {len(calls)} times, expected once"
    assert results == [{"row": "once"}] * 3, "every caller must get the same row"


def test_a_failed_computation_does_not_wedge_the_key() -> None:
    """The in-flight marker has to be released on the error path too."""
    from gui.server import _cached_row, clear_portfolio_cache

    clear_portfolio_cache()

    def boom():
        raise RuntimeError("spec exploded")

    with pytest.raises(RuntimeError):
        _cached_row(("k", "bad"), boom)

    assert _cached_row(("k", "bad"), lambda: {"row": "recovered"}) == {"row": "recovered"}


def test_the_listing_carries_each_analysis_age() -> None:
    """The sidebar cannot flag staleness it is never told about."""
    from gui.server import list_analyses

    entries = sorted(list_analyses(), key=lambda e: e["name"])
    assert entries, "no entries found - this test would pass vacuously"
    for e in entries:
        assert e["as_of"] == e["name"][:10], f"{e['name']}: as_of not carried"
        assert isinstance(e["age_days"], int) and e["age_days"] >= 0, (
            f"{e['name']}: age_days is {e['age_days']!r}"
        )


def test_a_row_carries_what_it_has_to_declare_not_just_how_much() -> None:
    """Floored draws and degenerate results in the ranking, per flag rather than as a count."""
    from conftest import skip_unless_fixture_corpus

    skip_unless_fixture_corpus()

    from gui.server import _one_row

    row = _one_row("2026-04-20-FLOORCO", "FLOORCO", "2026-04-20", 26, None, False)
    assert row["status"] == "ok", row.get("error")

    flags = {d["name"]: d for d in row["diagnostics"]}
    assert "affo_terminal_spread_floored" in flags, f"expected a floored flag, got {flags}"
    fired = flags["affo_terminal_spread_floored"]
    assert fired["count"] == 53
    assert fired["rate_text"] == "0.27%"
    assert 0 < fired["rate"] < 0.01
    assert row["degenerate"] is False


def test_a_clean_row_declares_nothing() -> None:
    """No false positives, the property the golden spec is required to keep.
    An empty list, not a null: "nothing to declare" and "not computed" must not
    render the same."""
    from conftest import skip_unless_fixture_corpus

    skip_unless_fixture_corpus()

    from gui.server import _one_row

    row = _one_row("2026-01-05-ACME", "ACME", "2026-01-05", 21, None, False)
    assert row["status"] == "ok", row.get("error")
    assert row["diagnostics"] == []
    assert row["degenerate"] is False
    assert row["warnings"] == 0


def test_no_analysis_in_the_sweep_is_degenerate() -> None:
    """Runs against whichever corpus is in use, and that is the point."""
    from gui.server import do_portfolio

    rows = [r for r in do_portfolio(repriced=False)["portfolio"] if r.get("status") == "ok"]
    assert rows, "the sweep found no rows at all - it would satisfy every assertion below"
    degenerate = [r["name"] for r in rows if r["degenerate"]]
    assert degenerate == [], (
        f"{len(degenerate)} of {len(rows)} row(s) reach p_undervalued of exactly "
        f"0 or 1: {degenerate}. The degenerate badge is planted in "
        "tests/test_gui_render.py precisely because no row reaches it."
    )


def test_the_corpus_composition_is_exactly_as_built() -> None:
    """The sweep over the corpus is a weaker claim than the old one - by design."""
    from conftest import skip_unless_fixture_corpus

    skip_unless_fixture_corpus()

    from gui.server import do_portfolio

    rows = [r for r in do_portfolio(repriced=False)["portfolio"] if r.get("status") == "ok"]
    assert len(rows) == 6, f"the corpus is 6 fixtures, found {len(rows)} - update this count"
    floored = sorted(r["name"] for r in rows if r["diagnostics"])
    assert floored == ["2026-04-20-FLOORCO"], (
        f"expected exactly the floored fixture, got {floored}. A fixture that "
        "floors unintentionally is a defect in the fixture, not a reason to "
        "widen this assertion."
    )


def test_the_age_is_computed_in_exactly_one_place() -> None:
    """`_dated()` owns the clock, and this is the case that actually pins it."""
    import gui.server as gs

    src = pathlib.Path(gs.__file__).read_text(encoding="utf-8")
    clocks = re.findall(r"_dt\.date\.today\(\)", src)
    assert len(clocks) == 1, (
        f"gui/server.py reads the clock in {len(clocks)} places; age belongs to "
        "_dated() alone, so no two surfaces can disagree about how old a row is"
    )
    subtractions = re.findall(r"-\s*_dt\.date\.fromisoformat", src)
    assert len(subtractions) == 1, f"{len(subtractions)} age subtractions; expected 1"


def test_the_listing_and_the_portfolio_agree_on_age() -> None:
    """The behavioural half: the portfolio does not derive an age of its own."""
    from gui.server import do_portfolio, list_analyses

    listed = {e["name"]: e["age_days"] for e in list_analyses()}
    rows = [r for r in do_portfolio(repriced=False)["portfolio"] if r.get("status") == "ok"]
    assert rows, "no runnable rows - this comparison would be vacuous"
    disagree = {
        r["name"]: (listed[r["name"]], r["age_days"])
        for r in rows
        if listed[r["name"]] != r["age_days"]
    }
    assert disagree == {}, f"listing and portfolio disagree on age: {disagree}"


def test_a_folder_without_a_date_gets_no_age_rather_than_a_wrong_one(monkeypatch, tmp_path) -> None:
    """`None`, not zero and not today. A folder this scheme cannot date is not
    fresh - it is undated, and the two must not render the same."""
    import gui.server as gs

    (tmp_path / "notes").mkdir()
    (tmp_path / "2026-01-01-AAA").mkdir()
    monkeypatch.setattr(gs, "ANALYSES_DIR", tmp_path)
    by_name = {e["name"]: e for e in gs.list_analyses()}

    assert by_name["notes"]["as_of"] is None
    assert by_name["notes"]["age_days"] is None

    # The absolute claim, computed here rather than read back: every case around
    # this one asserts the four surfaces agree, which a uniformly wrong age
    # satisfies perfectly. Calendar days since the analysis date, no rounding and
    # no business-day cleverness.
    expected = (dt.date.today() - dt.date(2026, 1, 1)).days
    assert by_name["2026-01-01-AAA"]["age_days"] == expected


def test_every_shipped_analysis_folder_parses() -> None:
    """The guard is checked against the real record, not only fixtures."""
    from gui.server import ANALYSES_DIR

    folders = sorted(p.name for p in ANALYSES_DIR.iterdir() if p.is_dir())
    assert folders, "no entries found - this test would pass vacuously"
    unparsed = [name for name in folders if _ticker_of(name) is None]
    assert unparsed == [], f"shipped analysis folders no longer parse as tickers: {unparsed}"


# --------------------------------------------------------------------------- #
# The price ladder can be read at any margin-of-safety anchor
# --------------------------------------------------------------------------- #
#
# `margin_of_safety()` returns P10, P50 and P90 from the run the ladder has
# already done, and `do_ladder` was keeping only P50 and throwing the other two
# away. That is what made the ladder the one view answering "at what price does
# this become attractive?" on the median - the anchor the rest of the viewer
# argues against. The portfolio defaults to P10 and says why in its own copy:
# margin of safety is a question about the downside, and a name can look healthy
# on its median while carrying a severe bad case.
#
# The cost of the omission is that switching anchors would otherwise mean
# re-running: the ladder is the most expensive view in the viewer at one full
# model run per rung (~850ms x 7), so a re-anchor that re-ran would take six
# seconds to change one column. All three come back with the same run.
#
# Nothing here can test the rendering - no headless browser is wired up -
# so these pin the payload the renderer reads, which is the half that can be.


@pytest.fixture(scope="module")
def ladder():
    """One `do_ladder` call, shared across the cases below."""
    import gui.server as gs

    original = gs.LADDER_STEPS
    gs.LADDER_STEPS = (0.9, 1.0, 1.1)
    try:
        name = sorted(p.name for p in gs.ANALYSES_DIR.iterdir() if p.is_dir())[0]
        return gs.do_ladder(name)
    finally:
        gs.LADDER_STEPS = original


def test_every_rung_carries_all_three_margin_of_safety_anchors(ladder) -> None:
    """P50 alone is not enough to answer the question the ladder is for."""
    rows = ladder["ladder"]
    assert rows, "no rungs"
    for row in rows:
        missing = [k for k in ("mos_p10", "mos_p50", "mos_p90") if k not in row]
        assert not missing, (
            f"rung {row['price']} is missing {missing} - the viewer cannot offer "
            "an anchor the endpoint does not send, and re-running to get it "
            "would cost a full model run per rung"
        )
        assert row["mos_p10"] <= row["mos_p50"] <= row["mos_p90"], (
            f"margins out of order at {row['price']}: value quantiles are "
            "ordered, so their margins must be too"
        )


def test_the_three_anchors_describe_one_value_distribution(ladder) -> None:
    """The anchors are free precisely because price never enters the valuation."""
    rows = ladder["ladder"]
    for key in ("mos_p10", "mos_p50", "mos_p90"):
        implied = [(1 + r[key]) * r["price"] for r in rows]
        assert max(implied) - min(implied) < 1e-6 * max(abs(v) for v in implied), (
            f"{key} implies a different value per rung ({implied}) - the rungs "
            "are supposed to differ only in price"
        )


def test_margin_of_safety_falls_as_the_rung_price_rises(ladder) -> None:
    """The ladder's whole point: a lower price buys a wider margin."""
    rows = sorted(ladder["ladder"], key=lambda r: r["price"])
    for key in ("mos_p10", "mos_p50", "mos_p90"):
        series = [r[key] for r in rows]
        assert series == sorted(series, reverse=True), f"{key} is not monotone: {series}"


# --------------------------------------------------------------------------- #
# Compare reports the whole distribution it already computed
# --------------------------------------------------------------------------- #
#
# Same species as the ladder above, one view over, and arguably worse here: a
# comparison of two vintages of the same name (the two A12 runs this repo keeps
# side by side) is a question about what moved, and the view answered it on the
# median alone. `do_compare` sent `value_p10`/`value_p50`/`value_p90` - the whole
# range - beside a single `mos_p50`, so the payload already disagreed with
# itself about how much of the distribution matters.
#
# Nothing is re-run for this: both margins come out of `margin_of_safety()` on
# the run each side has already done, exactly as in `_one_row` and `do_ladder`.


@pytest.fixture(scope="module")
def compared():
    """One `do_compare` over the first two shipped analyses."""
    from gui.server import ANALYSES_DIR, do_compare

    names = sorted(p.name for p in ANALYSES_DIR.iterdir() if p.is_dir())[:2]
    return do_compare(*names)


def test_compare_carries_the_whole_margin_range(compared) -> None:
    """Both sides report P10/P50/P90, so a diff can show what the downside did."""
    for side in ("a", "b"):
        row = compared[side]
        missing = [k for k in ("mos_p10", "mos_p50", "mos_p90") if k not in row]
        assert not missing, (
            f"side {side} ({row['name']}) is missing {missing} while sending the "
            "full value range beside it - the payload disagrees with itself"
        )
        assert row["mos_p10"] <= row["mos_p50"] <= row["mos_p90"], (
            f"margins out of order on side {side}: {row['name']}"
        )


def test_compare_margins_agree_with_the_values_beside_them(compared) -> None:
    """Each margin is its own quantile's, not another's."""
    for side in ("a", "b"):
        row = compared[side]
        for q in ("p10", "p50", "p90"):
            implied = (1 + row[f"mos_{q}"]) * row["price"]
            value = row[f"value_{q}"]
            assert abs(implied - value) < 1e-6 * abs(value), (
                f"side {side} ({row['name']}): mos_{q} implies {implied:.6f} "
                f"against a reported value_{q} of {value:.6f}"
            )


# --------------------------------------------------------------------------- #
# A refreshed quote is not served from the cache
# --------------------------------------------------------------------------- #
#
# The portfolio cache key is (version, name, spec mtime, use_current, price) -
# everything that changes the run. But `_one_row` also baked four fields that
# come from the quote into the cached row: `current_price`, `current_as_of`,
# `current_live` and `currency`. None of them is in the key, so a fresh
# `python -m prices.fetch` did not reach a row already cached.
#
# Measured before the fix, on the real record: quoting an analysis, then
# moving the quote artefact to a different price/date/live-flag, still
# returned the stale cached values from the same view.
#
# The half that reached the screen is the live flag, and the case is ordinary
# rather than contrived: the last intraday tick before the bell is the settled
# close, so `live` flips true -> false at an unchanged price, which leaves the
# key identical. The row then keeps the dagger and "LIVE intraday, still moving"
# against a settled close - and in the other order, "settled close" against a
# price that is moving, which is the direction the 2026-07-30 work exists to
# prevent.
#
# The fix is not to put the quote in the key. That would invalidate every row on
# every fetch - 29 rows x a full 50,000-draw run - to refresh metadata that had
# no effect on the run, since `p_undervalued` at the spec price does not depend
# on the quote at all. Cached is what was computed; the observed quote is
# overlaid after the lookup. The third case below is what holds that line.


@pytest.fixture
def one_analysis(monkeypatch):
    """The portfolio narrowed to a single real analysis, with a synthetic quote."""
    import gui.server as gs

    entry = next(e for e in gs.list_analyses() if e["has_yaml"] and gs._ticker_of(e["name"]))
    name, ticker = entry["name"], gs._ticker_of(entry["name"])
    monkeypatch.setattr(gs, "list_analyses", lambda: [entry])

    def quoted(price, as_of, live):
        monkeypatch.setattr(
            gs,
            "read_current_prices",
            lambda: {
                "available": True,
                "as_of": as_of,
                "prices": {
                    ticker: {"price": price, "as_of": as_of, "live": live, "currency": "USD"}
                },
                "failures": {},
                "malformed": [],
            },
        )

    def row(repriced=False):
        return gs._portfolio_rows(repriced)["portfolio"][0]

    return name, quoted, row


def test_a_refreshed_quote_reaches_an_already_cached_row(one_analysis) -> None:
    """The measured defect: a new fetch did not reach a warm row."""
    _, quoted, row = one_analysis

    quoted(100.0, "2026-07-30", False)
    row()  # warms the cache
    quoted(105.0, "2026-07-31", True)
    after = row()

    assert (after["current_price"], after["current_as_of"], after["current_live"]) == (
        105.0,
        "2026-07-31",
        True,
    ), "served the previous fetch's quote from the cache"


def test_the_live_flag_turns_over_at_an_unchanged_price(one_analysis) -> None:
    """The half that reaches the screen, and the ordinary case."""
    _, quoted, row = one_analysis

    quoted(100.0, "2026-07-30", True)
    assert row(repriced=True)["current_live"] is True
    quoted(100.0, "2026-07-30", False)
    assert row(repriced=True)["current_live"] is False, (
        "kept 'LIVE intraday, still moving' on a price that has settled - and in "
        "the other order this reads 'settled close' against a moving price"
    )


def test_re_reading_a_quote_does_not_cost_a_model_run(one_analysis) -> None:
    """Overlaid, not re-keyed - the reason this fix is not one line in the key."""
    import gui.server as gs

    _, quoted, row = one_analysis

    quoted(100.0, "2026-07-30", True)
    row()
    before = len(gs._PORTFOLIO_CACHE)
    quoted(105.0, "2026-07-31", False)
    row()

    assert len(gs._PORTFOLIO_CACHE) == before, (
        "a changed quote added a cache entry, so the row was recomputed - that "
        "is a full 50,000-draw run per analysis on every price refresh"
    )


# --------------------------------------------------------------------------- #
#
# The default price basis, which is a tri-state and not a boolean.
#
# `repriced` used to default to False: the portfolio opened at every spec's own
# price, and the current quote was one click away. That is backwards for the
# question the view exists to answer - "what is cheapest right now?" - because
# the prices it opened with are the ones the analyst typed on the analysis date,
# up to weeks ago.
#
# The default cannot be a plain True, though, and that is the whole reason the
# parameter is now three-valued. `prices/prices.json` is untracked and absent
# until someone runs the fetcher, and a hardcoded True against no quotes would
# fall back per row and flag every row "no quote" - a screen full of warnings
# about a file the reader may have deliberately never created.
#
# So: absent means "current if there are quotes, as-analysed if not", and an
# explicit value from the toggle is obeyed either way. The server decides the
# absent case because only the server knows whether the artefact exists.


def test_the_portfolio_defaults_to_the_current_quote(one_analysis) -> None:
    """No basis given, quotes present - re-price, and say that is what happened."""
    import gui.server as gs

    _, quoted, _ = one_analysis
    quoted(123.45, "2026-08-08", False)
    out = gs._portfolio_rows(None)

    assert out["repriced"] is True, "opened at the spec price while a quote was available"
    row = out["portfolio"][0]
    assert row["repriced"] is True
    assert row["price"] == 123.45, "reported the quote as the basis but ran the spec's price"


def test_the_default_falls_back_to_as_analysed_when_there_are_no_quotes(
    one_analysis, monkeypatch
) -> None:
    """No basis given, no quotes - the spec's price, with nothing to flag."""
    import gui.server as gs

    monkeypatch.setattr(gs, "read_current_prices", lambda: {"available": False, "prices": {}})
    out = gs._portfolio_rows(None)

    assert out["repriced"] is False, (
        "claimed a re-priced basis with no quotes to re-price with, so every row "
        "comes back flagged 'no quote'"
    )
    assert out["portfolio"][0]["repriced"] is False


def test_an_explicit_basis_is_obeyed_in_both_directions(one_analysis) -> None:
    """The toggle still wins. A default is a starting point, not a policy."""
    import gui.server as gs

    _, quoted, _ = one_analysis
    quoted(123.45, "2026-08-08", False)

    assert gs._portfolio_rows(False)["repriced"] is False, (
        "'As analysed' could not be selected once quotes existed"
    )
    assert gs._portfolio_rows(True)["repriced"] is True


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ({}, None),
        ({"repriced": [""]}, None),
        ({"repriced": ["1"]}, True),
        ({"repriced": ["true"]}, True),
        ({"repriced": ["0"]}, False),
        ({"repriced": ["false"]}, False),
    ],
)
def test_the_basis_parameter_distinguishes_absent_from_off(query, expected) -> None:
    """Absent and `0` are different answers, and a boolean parser cannot say so."""
    from gui.server import _tristate

    assert _tristate(query, "repriced") is expected
