"""Fetch current quotes for the shipped analyses and write `prices.json`.

Run from the repo root:

    python -m prices.fetch            # every ticker under the record
    python -m prices.fetch ACME WIDGET # just these, merged into what is there

Stdlib only, so this adds no dependency to the project.

What this package may and may not do: retrieval is confined here - it fetches
a price, writes an artefact, and stops. It must never grow into fetching
history or returns; a price is a comparison input, a return series is raw
material for a kind of fitting this project forbids.

The written file is deliberately not tracked: market data is dated, and a repo
carrying stale quotes invites someone to trust them.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import re
import sys
import tempfile
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Root directory this fetcher reads tickers from. `VALDIST_ANALYSES` overrides
#: it; unset or empty falls back to a local default - empty matters because
#: `pathlib.Path("")` is `Path(".")`, so `VALDIST_ANALYSES=` would silently
#: point at the current working directory instead.
#:
#: Duplicated in `gui/server.py`; keep the two definitions in sync.
ANALYSES_DIR = pathlib.Path(os.environ.get("VALDIST_ANALYSES") or REPO_ROOT / "analyses")
OUT_PATH = pathlib.Path(__file__).resolve().parent / "prices.json"

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=5d&interval=1d"
USER_AGENT = "Mozilla/5.0 (valdist price fetcher)"

#: Symbol table filename, kept beside the tickers it describes rather than in
#: this module: together its two maps enumerate every ticker in one place, and
#: a copy here would duplicate that index in a module whose job is exchange
#: suffixes, not bookkeeping.
SYMBOLS_FILENAME = "symbols.json"


def load_symbols(root: pathlib.Path | None = None) -> tuple[dict[str, str], set[str]]:
    """Read `symbols.json` from the record. Returns `(overrides, us_listed)`.

    A missing file yields empty maps rather than an error - safe only because
    `fetch_one` refuses a ticker it can't classify: a checkout with no record
    has nothing to classify, and one with a record but no table refuses every
    ticker by name instead of guessing.

    A malformed file raises. Skipping what won't parse would silently
    reinstate the bare lookup this table exists to prevent, and a bare lookup
    doesn't fail - it just returns a real price for the wrong company.
    """
    root = ANALYSES_DIR if root is None else pathlib.Path(root)
    path = root / SYMBOLS_FILENAME
    if not path.is_file():
        return {}, set()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: unreadable symbol table: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(doc).__name__}")

    raw_ov = doc.get("overrides", {})
    raw_us = doc.get("us_listed", {})
    if not isinstance(raw_ov, dict) or not isinstance(raw_us, (dict, list)):
        raise ValueError(f"{path}: 'overrides' must be an object and 'us_listed' an object or list")

    overrides: dict[str, str] = {}
    for key, entry in raw_ov.items():
        symbol = entry.get("symbol") if isinstance(entry, dict) else entry
        if not isinstance(symbol, str) or not symbol:
            raise ValueError(f"{path}: override {key!r} has no symbol")
        # Same shape rule the tracked table was checked against: a value whose
        # base symbol doesn't match the key quotes a different company, cleanly.
        base, _, suffix = symbol.rpartition(".")
        if base != key or not suffix:
            raise ValueError(
                f"{path}: override {key!r} -> {symbol!r} must be '<TICKER>.<EXCHANGE>' "
                "with the ticker unchanged; a different base symbol is a different company"
            )
        overrides[key] = symbol

    us_listed = set(raw_us)
    overlap = sorted(set(overrides) & us_listed)
    if overlap:
        raise ValueError(
            f"{path}: {overlap} declared both non-US and US-listed. The two maps "
            "answer the same question and must not disagree."
        )
    return overrides, us_listed


def tickers_from_analyses() -> list[str]:
    """Every distinct ticker found, from each entry's `<date>-<TICKER>` name."""
    if not ANALYSES_DIR.is_dir():
        return []
    found = set()
    for d in ANALYSES_DIR.iterdir():
        m = re.fullmatch(r"\d{4}-\d{2}-\d{2}-(.+)", d.name) if d.is_dir() else None
        if m:
            found.add(m.group(1))
    return sorted(found)


def _utc_date(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.UTC).date().isoformat()


def _exchange_date(ts: float, meta: dict) -> str:
    """The date *ts* falls on in the exchange's day, not in UTC."""
    return _utc_date(ts + (meta.get("gmtoffset") or 0))


def _settled_meta_close(meta: dict, now: float) -> tuple[float, str] | None:
    """`meta`'s last trade, but only when it's a finished session's close."""
    price = meta.get("regularMarketPrice")
    ts = meta.get("regularMarketTime")
    reg = (meta.get("currentTradingPeriod") or {}).get("regular") or {}
    session_start, session_end = reg.get("start"), reg.get("end")
    if price is None or ts is None or session_start is None:
        return None
    day = _exchange_date(ts, meta)
    session_day = _exchange_date(session_start, meta)
    if day < session_day:
        return float(price), day
    if day == session_day and session_end is not None and now >= session_end:
        return float(price), day
    return None


def _is_live_bar(ts: float, meta: dict, now: float) -> bool:
    """True when *ts* is the current trading period's bar and that period is open."""
    reg = (meta.get("currentTradingPeriod") or {}).get("regular") or {}
    start, end = reg.get("start"), reg.get("end")
    if start is None or end is None:
        return False
    return _exchange_date(ts, meta) == _exchange_date(start, meta) and now < end


def select_quote(res: dict, now: float | None = None) -> dict | None:
    """The most recent close in a Yahoo chart result, or None."""
    now = _dt.datetime.now(_dt.UTC).timestamp() if now is None else now
    stamps = res.get("timestamp") or []
    closes = ((res.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
    meta = res.get("meta") or {}
    currency = meta.get("currency")

    bar = None
    for ts, close in zip(reversed(stamps), reversed(closes)):
        if close is not None:
            bar = (round(float(close), 4), _exchange_date(ts, meta), _is_live_bar(ts, meta, now))
            break

    settled = _settled_meta_close(meta, now)
    if settled is not None and (bar is None or settled[1] > bar[1]):
        bar = (round(settled[0], 4), settled[1], False)

    if bar is None:
        return None
    return {"price": bar[0], "as_of": bar[1], "currency": currency, "live": bar[2]}


def fetch_one(
    ticker: str,
    timeout: float = 20.0,
    symbols: tuple[dict[str, str], set[str]] | None = None,
) -> dict:
    """Latest available price for one ticker, with the date it belongs to.

    Not always a settled close - the row says which. Yahoo's daily bar array
    includes the current session's bar while that session is open, and its
    value moves during the day. The returned row carries a `live` flag so a
    caller can tell a moving intraday print from a settled close.

    Raises `ValueError` for a ticker in neither map of `symbols.json`. Guessing
    isn't safe: a bare ticker that happens to exist on a US exchange returns a
    real price for the wrong company, cleanly and silently.
    """
    overrides, us_listed = load_symbols() if symbols is None else symbols
    # Classification is mandatory and enforced here, not by a test. It used to
    # be a test's job: `SYMBOL_OVERRIDES` was consulted with a bare
    # `.get(ticker, ticker)` fallback, and `US_LISTED` was purely declarative -
    # nothing in this module read it. Enforcing it at runtime means it holds
    # for every caller, not only where a test happens to run.
    if ticker in overrides:
        symbol = overrides[ticker]
    elif ticker in us_listed:
        symbol = ticker
    else:
        raise ValueError(
            f"unclassified ticker {ticker!r}: add it to {SYMBOLS_FILENAME} under "
            "'overrides' (with its exchange suffix) or 'us_listed' (when the bare "
            "ticker IS the quote symbol). Guessing is not safe here - a bare "
            "ticker that happens to exist on a US exchange returns a real price "
            "for the wrong company, cleanly and silently."
        )
    # Checked before the network, not after. The symbol is formatted straight
    # into CHART_URL, so `A?x=1` used to truncate the intended query parameters
    # and produce a confusing failure instead of a clear refusal. `main()`
    # records this ValueError in the artefact's `failures` block, so a bad symbol
    # surfaces by name. Dot and hyphen are allowed: `A22.PA` and `RDS-A` are real.
    if not re.fullmatch(r"[A-Za-z0-9.-]{1,12}", symbol):
        raise ValueError(f"refusing malformed quote symbol {symbol!r}")
    req = urllib.request.Request(
        CHART_URL.format(symbol=symbol), headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
        payload = json.load(resp)

    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        raise ValueError(f"no chart data for {symbol!r}")
    quote = select_quote(result[0])
    if quote is None:
        raise ValueError(f"no non-null close for {symbol!r}")
    return {"ticker": ticker, "symbol": symbol, **quote}


def _previous(path: pathlib.Path) -> dict:
    """The artefact an earlier run left, or an empty one if it is unusable."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _atomic_write(path: pathlib.Path, text: str) -> None:
    """Write *text* to *path* all-or-nothing, via a sibling temp file."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp = pathlib.Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Leaving a `.tmp` sibling behind would be its own quiet mess, and the
        # next run would have no way to tell it from a real artefact.
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "tickers",
        nargs="*",
        help="default: every known ticker (rewrites the file). Naming "
        "tickers updates only those rows and keeps the rest.",
    )
    parser.add_argument("--out", type=pathlib.Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    targeted = bool(args.tickers)
    tickers = args.tickers or tickers_from_analyses()
    if not tickers:
        print("no tickers found", file=sys.stderr)
        return 1

    now = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")

    # A targeted run answers only about the tickers it was given, so it must
    # carry the others forward. A full run's answer is complete by construction,
    # and merging it would resurrect tickers whose analysis has been deleted.
    prices: dict[str, dict] = {}
    failures: dict[str, str] = {}
    previous_as_of = None
    if targeted:
        prev = _previous(args.out)
        previous_as_of = prev.get("as_of")
        prices = dict(prev["prices"]) if isinstance(prev.get("prices"), dict) else {}
        failures = dict(prev["failures"]) if isinstance(prev.get("failures"), dict) else {}
        for row in prices.values():
            # Rows written before `fetched_at` existed. Dating them to the old
            # document keeps the oldest-row rule below computable, and errs old.
            if isinstance(row, dict):
                row.setdefault("fetched_at", previous_as_of or row.get("as_of"))

    # Loaded once, not per ticker: re-reading inside the loop would let an
    # edit halfway through a 60-ticker run change how the rest resolve, and
    # the artefact would carry two different tables' answers under one `as_of`.
    symbols = load_symbols()

    fetched = 0
    for ticker in tickers:
        try:
            row = fetch_one(ticker, symbols=symbols)
            row["fetched_at"] = now
            prices[ticker] = row
            failures.pop(ticker, None)
            fetched += 1
            ccy = row.get("currency") or ""
            print(f"  {ticker:<6} {row['price']:>10.2f} {ccy:<4} {row['as_of']}")
        except (urllib.error.URLError, ValueError, KeyError, TimeoutError) as exc:
            failures[ticker] = f"{type(exc).__name__}: {exc}"
            # A price and a failure for the same ticker would contradict each
            # other, and the viewer reads the price. Dropping it is the honest
            # half: a fetch that just failed must not still render a quote.
            prices.pop(ticker, None)
            print(f"  {ticker:<6} FAILED - {exc}", file=sys.stderr)

    # Failures are recorded in the artefact rather than dropped. A ticker that
    # silently vanished would read downstream as "no current price available",
    # which is indistinguishable from "never asked for" - and the viewer would
    # then quietly fall back to the stale spec price with nothing said.
    #
    # `as_of` is the oldest row, not this run's timestamp. The viewer renders
    # one timestamp for the whole table, so after a merge the only honest
    # single value is the oldest one - it may understate freshness, never
    # overstate it.
    stamps = [
        r["fetched_at"] for r in prices.values() if isinstance(r, dict) and r.get("fetched_at")
    ]
    doc = {
        "as_of": min(stamps) if stamps else now,
        "source": "Yahoo Finance chart API (daily close)",
        "prices": prices,
        "failures": failures,
    }
    _atomic_write(args.out, json.dumps(doc, indent=2) + "\n")
    kept = f", {len(prices) - fetched} kept" if targeted else ""
    print(
        f"\nwrote {args.out} - {fetched} fetched{kept}, "
        f"{len(prices)} price(s), {len(failures)} failure(s)"
    )
    # Reports this run, not the file's accumulated contents: a preserved row
    # must not let a run that fetched nothing report success.
    return 0 if fetched else 1


if __name__ == "__main__":
    raise SystemExit(main())
