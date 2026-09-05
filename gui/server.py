"""Read-only local viewer for shipped valdist analyses."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import pathlib
import re
import threading
import traceback
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as _np

from valdist import SpecError

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Root directory the viewer reads specs from. `VALDIST_ANALYSES` overrides it;
#: unset or empty falls back to a local default - empty matters because
#: `pathlib.Path("")` is `Path(".")`, so `VALDIST_ANALYSES=` would silently
#: point at the current working directory instead.
#:
#: Duplicated in `prices/fetch.py`; keep the two definitions in sync.
ANALYSES_DIR = pathlib.Path(os.environ.get("VALDIST_ANALYSES") or REPO_ROOT / "analyses")
INDEX_HTML = pathlib.Path(__file__).resolve().parent / "index.html"

#: The `report.to_json()` contract this viewer was written against. Emitting the
#: version was only half the guarantee - it was rendered in a footer and compared
#: against nothing, so a bumped schema would have shown missing or renamed fields
#: as blanks instead of refusing a payload the UI does not understand. Bumping
#: `REPORT_SCHEMA_VERSION` in the engine now reddens `tests/test_gui_server.py`,
#: which is what makes this a contract rather than a decoration.
SUPPORTED_REPORT_SCHEMA = "1.0"


def check_report_schema(payload: dict) -> dict:
    """Whether *payload* is a report contract this viewer can render."""
    got = payload.get("report_schema_version")
    return {
        "expected": SUPPORTED_REPORT_SCHEMA,
        "got": got,
        "supported": isinstance(got, str) and got == SUPPORTED_REPORT_SCHEMA,
    }


# --------------------------------------------------------------------------- #
# Data access - read-only, and traversal-proof by construction
# --------------------------------------------------------------------------- #


def list_analyses() -> list[dict]:
    """Every analysis folder, newest first (the names sort as ISO dates)."""
    if not ANALYSES_DIR.is_dir():
        return []
    out = []
    for d in sorted(ANALYSES_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        as_of, age = _dated(d.name)
        out.append(
            {
                "name": d.name,
                "has_md": (d / f"{d.name}.md").is_file(),
                "has_yaml": (d / f"{d.name}.yaml").is_file(),
                "as_of": as_of,
                "age_days": age,
            }
        )
    return out


def _resolve(name: str) -> pathlib.Path:
    """Map a requested name to its folder, or raise."""
    if name not in {a["name"] for a in list_analyses()}:
        raise KeyError(f"unknown analysis: {name!r}")
    return ANALYSES_DIR / name


def read_analysis(name: str) -> dict:
    folder = _resolve(name)
    md, yaml_ = folder / f"{name}.md", folder / f"{name}.yaml"
    ticker = _ticker_of(name)
    quote = read_current_prices()["prices"].get(ticker) if ticker else None
    as_of, age_days = _dated(name)
    return {
        "name": name,
        "md": md.read_text(encoding="utf-8") if md.is_file() else None,
        "yaml": yaml_.read_text(encoding="utf-8") if yaml_.is_file() else None,
        # The spec's own price, so the UI can default its input to it. Parsed
        # rather than regexed out of the YAML, and tolerant: a spec too broken
        # to load still lists and still renders its text, it just has no default.
        "spec_price": _spec_price(name),
        # The current quote, if `prices/prices.json` has one, so a single page
        # can offer the same re-pricing the portfolio does. The age rides along
        # because re-pricing only refreshes the price - the value view is still
        # as of the analysis date, and the reader needs both numbers to judge
        # whether that pairing is safe.
        "ticker": ticker,
        "as_of": as_of,
        "age_days": age_days,
        "current_price": quote["price"] if quote else None,
        "current_as_of": quote.get("as_of") if quote else None,
        # True when the quote is a moving intraday bar rather than a settled
        # close. `as_of` alone can't say so - a live tick is dated today and so
        # reads as the freshest possible quote, which is exactly backwards.
        "current_live": bool(quote.get("live")) if quote else False,
        "currency": quote.get("currency") if quote else None,
    }


def _spec_price(name: str) -> float | None:
    from valdist.spec.loader import load_spec

    try:
        return load_spec(_spec_path(name)).price
    except Exception:
        return None


def _spec_path(name: str) -> pathlib.Path:
    path = _resolve(name) / f"{name}.yaml"
    if not path.is_file():
        raise KeyError(f"{name} has no spec file")
    return path


# --------------------------------------------------------------------------- #
# Engine calls - the same entry points the CLI uses, nothing bespoke
# --------------------------------------------------------------------------- #


def do_validate(name: str) -> dict:
    """Mirror of the `valdist validate` command, as data instead of text."""
    from valdist.spec.loader import load_spec
    from valdist.spec.validate import check_marginal_fit
    from valdist.spec.validate import validate as _validate

    spec = load_spec(_spec_path(name))
    errors = _validate(spec)
    notices = check_marginal_fit(spec)
    return {
        "ok": not errors,
        "errors": [{"code": e.code, "message": e.message} for e in errors],
        "notices": [{"code": n.code, "message": n.message} for n in notices],
    }


def _load_and_hydrate(name: str):
    """Shared prologue: parse the spec and build the model."""
    from valdist.spec.loader import hydrate, load_spec

    spec = load_spec(_spec_path(name))
    return spec, hydrate(spec)


def do_run(name: str, price: float | None = None) -> dict:
    """Run the Monte Carlo and return both the JSON payload and the text block."""
    from valdist import report

    spec, model = _load_and_hydrate(name)
    used = spec.price if price is None else price
    result = model.run(n=spec.n, price=used, seed=spec.seed, nu=spec.nu)
    payload = json.loads(report.to_json(result))
    return {
        "payload": payload,
        # Carried to the client, not just computed: a check the UI never receives
        # is a gate with no caller, the shape that once let `validate()` judge a
        # spec invalid and the run still print numbers anyway.
        "schema": check_report_schema(payload),
        "text": report.to_text(result),
        "spec_price": spec.price,
        "price_used": used,
        "overridden": price is not None and price != spec.price,
        "histogram": _histogram(result.value, used),
        # What each bar in the tornado is worth, and the band that produced it.
        # The bar is a rank correlation and says nothing about magnitude: on the
        # golden spec `div_growth` is the fourth bar and moves the value least of
        # the six. Sent alongside the payload rather than inside it, because a
        # swing is a property of the spec rather than of the run - putting it in
        # `to_json()` would bump the report schema for something no report needs.
        "sensitivity": _sensitivity(spec, model),
    }


def _sensitivity(spec, model) -> dict:
    """Per-driver swing, stated band, and how far the two can be trusted together."""
    out = {}
    for swing in model.driver_swings():
        driver = spec.drivers[swing["name"]]
        m = driver.marginal
        row = _marginal_row(swing["name"], m.p10, m.p50, m.p90, m.family)
        out[swing["name"]] = {
            **swing,
            "p10": m.p10,
            "p50": m.p50,
            "p90": m.p90,
            "family": m.family,
            "drift": row["drift"],
            "drift_flagged": row["drift"] > MARGINAL_DRIFT_FLAG,
        }
    return out


def _histogram(values, price: float | None, bins: int = 56) -> dict:
    """Bin the value draws for display."""
    import numpy as np

    v = np.asarray(values, dtype=float)
    lo, hi = (float(x) for x in np.quantile(v, [0.005, 0.995]))
    if not (hi > lo):  # degenerate distribution: everything on one value
        lo, hi = float(v.min()), float(v.max()) or 1.0
        if not (hi > lo):
            hi = lo + 1.0
    counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    return {
        "lo": lo,
        "hi": hi,
        "counts": [int(c) for c in counts],
        "edges": [float(e) for e in edges],
        "below": int((v < lo).sum()),
        "above": int((v > hi).sum()),
        "n": int(v.size),
        "price": price,
    }


def do_convergence(name: str) -> dict:
    """N vs 2N drift — the engine's own answer to "are 50,000 draws enough?"."""
    spec, model = _load_and_hydrate(name)
    result = model.run(n=spec.n, price=spec.price, seed=spec.seed, nu=spec.nu)
    return {"n": spec.n, "convergence": result.convergence()}


#: Multipliers for the price ladder, centred on 1.0 (the reference price).
LADDER_STEPS = (0.75, 0.85, 0.92, 1.0, 1.08, 1.18, 1.30)


def do_ladder(name: str, price: float | None = None) -> dict:
    """p_undervalued and margin of safety across a range of prices."""
    spec, model = _load_and_hydrate(name)
    centre = spec.price if price is None else price

    rungs = sorted({round(centre * f, 2) for f in LADDER_STEPS})
    rows = []
    for p in rungs:
        r = model.run(n=spec.n, price=p, seed=spec.seed, nu=spec.nu)
        mos = r.margin_of_safety(ANCHOR_QS)
        rows.append(
            {
                "price": p,
                "p_undervalued": r.p_undervalued,
                "mos_p10": mos[0.1],
                "mos_p25": mos[0.25],
                "mos_p50": mos[0.5],
                "mos_p90": mos[0.9],
                "is_centre": abs(p - centre) < 1e-9,
                "is_spec": abs(p - spec.price) < 1e-9,
            }
        )
    return {"ladder": rows, "spec_price": spec.price, "centre": centre}


def do_worlds(name: str) -> dict:
    """Percentile scenarios — the driver values behind P10 / P50 / P90."""
    spec, model = _load_and_hydrate(name)
    result = model.run(n=spec.n, price=spec.price, seed=spec.seed, nu=spec.nu)
    marginals = {
        driver_name: {
            "p10": d.marginal.p10,
            "p50": d.marginal.p50,
            "p90": d.marginal.p90,
            "family": d.marginal.family,
        }
        for driver_name, d in spec.drivers.items()
    }
    return {"worlds": result.worlds(), "marginals": marginals}


def _name(query: dict) -> str:
    """Pull the required `name` parameter, or fail with a message that says so."""
    values = query.get("name") or []
    if not values or not values[0]:
        raise ValueError("missing required query parameter: name")
    return values[0]


#: Portfolio rows keyed by (row-shape version, name, spec mtime). Running the
#: full set is slow enough to be irritating on every visit. Keying on mtime
#: means editing a spec invalidates just that row; the
#: version prefix invalidates every row when the shape below changes, so a cache
#: filled by an older build can never serve a row missing new fields. Reading a
#: mtime is a stat, not a write - the viewer stays read-only.
_PORTFOLIO_ROW_VERSION = 2

#: Bounded, LRU. The key includes the quoted price, so every refresh at a new
#: price added a full set of rows and nothing was ever evicted - measured at
#: 0 -> 13 -> 26 -> 39 -> 52 across four price variants. Entries are small dicts,
#: so this was slow rather than urgent, but "nothing is ever evicted" is a
#: property and it does not improve on its own.
_PORTFOLIO_CACHE_MAX = 512
_PORTFOLIO_CACHE: OrderedDict[tuple, dict] = OrderedDict()

#: Guards the cache dict itself, and the in-flight table below.
_CACHE_LOCK = threading.Lock()

#: One lock per key being computed. The lookup used to be check-then-compute with
#: nothing in between, so three simultaneous cold requests each ran the whole
#: portfolio: 26.5s / 26.5s / 27.0s against ~9s for one. A browser issues one
#: request at a time, so the real trigger is a double-click or a second tab.
_INFLIGHT: dict[tuple, threading.Lock] = {}


def clear_portfolio_cache() -> None:
    """Drop every cached row. Used by the tests; harmless at runtime."""
    with _CACHE_LOCK:
        _PORTFOLIO_CACHE.clear()
        _INFLIGHT.clear()


def _cached_row(key: tuple, compute) -> dict:
    """Return the cached row for *key*, computing it at most once."""
    with _CACHE_LOCK:
        hit = _PORTFOLIO_CACHE.get(key)
        if hit is not None:
            _PORTFOLIO_CACHE.move_to_end(key)
            return hit
        lock = _INFLIGHT.get(key)
        if lock is None:
            lock = _INFLIGHT[key] = threading.Lock()

    with lock:
        with _CACHE_LOCK:
            hit = _PORTFOLIO_CACHE.get(key)
            if hit is not None:
                _PORTFOLIO_CACHE.move_to_end(key)
                return hit
        try:
            row = compute()
        finally:
            # Released on the error path too. A spec that fails to validate comes
            # back as an error row rather than sinking the view, so a raising
            # `compute` is ordinary traffic - leaving its marker behind would
            # wedge that key for the life of the process.
            with _CACHE_LOCK:
                _INFLIGHT.pop(key, None)
        with _CACHE_LOCK:
            _PORTFOLIO_CACHE[key] = row
            _PORTFOLIO_CACHE.move_to_end(key)
            while len(_PORTFOLIO_CACHE) > _PORTFOLIO_CACHE_MAX:
                _PORTFOLIO_CACHE.popitem(last=False)
        return row


#: Written by `python -m prices.fetch`. Read-only, and never imported from -
#: `gui/` does not import `prices`, so the viewer consumes the artefact and
#: cannot trigger a fetch. Absent file is normal, not an error.
#: Quantiles every endpoint reports, so the viewer's anchor switch has the same
#: options everywhere. P25 is the conservative value anchor (see
#: `Result.summary`), not a fourth decoration: value does not depend on price,
#: so P25 is itself a buy-below price.
ANCHOR_QS = (0.10, 0.25, 0.50, 0.90)

PRICES_PATH = REPO_ROOT / "prices" / "prices.json"


def read_current_prices() -> dict:
    """Load the quote artefact, dropping anything malformed."""
    if not PRICES_PATH.is_file():
        return {"available": False, "prices": {}}
    try:
        doc = json.loads(PRICES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "prices": {}}
    if not isinstance(doc, dict):
        return {"available": False, "error": "prices.json is not an object", "prices": {}}

    raw = doc.get("prices")
    prices, dropped = {}, []
    for ticker, quote in raw.items() if isinstance(raw, dict) else []:
        price = quote.get("price") if isinstance(quote, dict) else None
        if isinstance(price, bool) or not isinstance(price, (int, float)):
            dropped.append(ticker)
        elif not math.isfinite(price) or price <= 0:
            dropped.append(ticker)
        else:
            prices[ticker] = quote

    return {
        "available": True,
        "as_of": doc.get("as_of"),
        "source": doc.get("source"),
        "prices": prices,
        "failures": doc.get("failures", {}) if isinstance(doc.get("failures"), dict) else {},
        # Surfaced, not silent: a dropped ticker is indistinguishable from one
        # never fetched unless it is reported.
        "malformed": sorted(dropped),
    }


#: What a ticker is allowed to look like. Letters, digits, dot and hyphen cover
#: every real symbol shape (`A12`, `BRK.B`, `RDS-A`) and nothing that means
#: something to a URL.
_TICKER_PATTERN = r"[A-Za-z0-9.-]{1,12}"


def _ticker_of(name: str) -> str | None:
    """`2026-07-25-A12` -> `A12`. None if the folder is not a ticker folder."""
    m = re.fullmatch(rf"\d{{4}}-\d{{2}}-\d{{2}}-({_TICKER_PATTERN})", name)
    return m.group(1) if m else None


def _as_of_of(name: str) -> str | None:
    return name[:10] if re.match(r"\d{4}-\d{2}-\d{2}-", name) else None


def _dated(name: str) -> tuple[str | None, int | None]:
    """`(as_of, age in days)` for an analysis folder. One clock, one subtraction."""
    as_of = _as_of_of(name)
    if as_of is None:
        return None, None
    return as_of, (_dt.date.today() - _dt.date.fromisoformat(as_of)).days


def do_portfolio(repriced: bool | None = None) -> dict:
    """Every analysis, run, as one comparable table."""
    return _portfolio_rows(repriced)


def _portfolio_rows(repriced: bool | None) -> dict:
    """Every analysis, run, as one comparable table."""
    current = read_current_prices()
    # Resolved against the quotes themselves rather than `available`, which is
    # true for a file that parsed and says nothing about whether it holds any
    # usable rows - a `prices.json` whose every quote was dropped as malformed is
    # available and empty, and would open a view that re-prices nothing.
    if repriced is None:
        repriced = bool(current["prices"])
    rows = []
    for entry in list_analyses():
        name = entry["name"]
        if not entry["has_yaml"]:
            rows.append({"name": name, "status": "no spec"})
            continue

        ticker = _ticker_of(name)
        quote = current["prices"].get(ticker) if ticker else None
        use_current = bool(repriced and quote)
        # Read, not re-derived. The sidebar shows the same number from the same
        # listing, and two subtractions against two `date.today()` calls is one
        # midnight away from disagreeing.
        as_of, age = entry["as_of"], entry["age_days"]

        price = quote["price"] if use_current else None
        row = _cached_row(
            _row_key(name, price, use_current),
            lambda: _one_row(name, ticker, as_of, age, price, use_current),
        )
        rows.append(_with_quote(row, quote))
    return {"portfolio": rows, "current": current, "repriced": repriced}


def _with_quote(row: dict, quote: dict | None) -> dict:
    """A copy of *row* carrying the quote as it stands now, not as it was cached."""
    if row.get("status") != "ok":
        return row
    return {
        **row,
        "current_price": quote["price"] if quote else None,
        "current_as_of": quote.get("as_of") if quote else None,
        "current_live": bool(quote.get("live")) if quote else False,
        "currency": quote.get("currency") if quote else None,
    }


def _row_key(name: str, price: float | None, repriced: bool) -> tuple:
    """Everything that changes the run, and nothing that doesn't."""
    return (_PORTFOLIO_ROW_VERSION, name, _spec_path(name).stat().st_mtime, price, repriced)


def _one_row(name, ticker, as_of, age, price, repriced) -> dict:
    """One computed row. Cheap to call, expensive to run."""
    try:
        spec, model = _load_and_hydrate(name)
        price = spec.price if price is None else price
        result = model.run(n=spec.n, price=price, seed=spec.seed, nu=spec.nu)
        q, mos = result.value_quantiles(ANCHOR_QS), result.margin_of_safety(ANCHOR_QS)
        return {
            "name": name,
            "status": "ok",
            "adapter": spec.valuation,
            "price": price,
            "spec_price": spec.price,
            "ticker": ticker,
            "as_of": as_of,
            "age_days": age,
            "repriced": repriced,
            # The quote's own fields aren't set here - `_with_quote` overlays
            # them after the cache lookup, because they can change without
            # anything in the key changing. See that function.
            "p_undervalued": result.p_undervalued,
            "stderr": result.p_undervalued_stderr,
            "value_p10": q[0.1],
            "value_p25": q[0.25],
            "value_p50": q[0.5],
            "value_p90": q[0.9],
            # All three margins, so the UI can re-anchor without re-running.
            # P10 is the one a margin-of-safety framework exists to protect:
            # a name can look fine on the median and still carry a brutal
            # downside.
            "mos_p10": mos[0.1],
            "mos_p25": mos[0.25],
            "mos_p50": mos[0.5],
            "mos_p90": mos[0.9],
            "warnings": len(result.warnings()),
            # What the row has to declare, not just how much: these are not
            # results, and the ranking is where that has to be legible - a
            # count of 2 cannot distinguish "a meaningless tail" from "this
            # whole number is an artefact".
            #
            # Every field comes off the result, the rate text in particular,
            # because a spec that floors draws often fires on well under 5 of
            # 50,000 and a consumer formatting the rate itself would print
            # "0.00%" for them. That is the disclosure saying nothing.
            "diagnostics": [
                {
                    "name": flag_name,
                    "count": count,
                    "rate": result.diagnostic_rate(flag_name),
                    "rate_text": result.diagnostic_rate_text(flag_name),
                }
                for flag_name, count in sorted(result.diagnostics.items())
            ],
            "degenerate": result.degenerate(),
        }
    except Exception as exc:
        # Cached like any other row: a broken spec is a stable answer until the
        # file changes, and the mtime is already in the key.
        return {"name": name, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def do_compare(name_a: str, name_b: str) -> dict:
    """Two analyses side by side: drivers, constants and headline results."""
    out = {}
    for side, name in (("a", name_a), ("b", name_b)):
        spec, model = _load_and_hydrate(name)
        result = model.run(n=spec.n, price=spec.price, seed=spec.seed, nu=spec.nu)
        q, mos = result.value_quantiles(ANCHOR_QS), result.margin_of_safety(ANCHOR_QS)
        out[side] = {
            "name": name,
            "adapter": spec.valuation,
            "price": spec.price,
            "p_undervalued": result.p_undervalued,
            "value_p10": q[0.1],
            "value_p25": q[0.25],
            "value_p50": q[0.5],
            "value_p90": q[0.9],
            # All three margins, beside the value range they belong to. A diff
            # of two vintages is a question about what moved, and a single
            # anchor can't say how much. Measured on the A12 pair this view
            # was built for (2026-07-19 against 2026-07-25), the margin moves
            # +6.1 points at P10, +7.6 at P50 and +9.0 at P90 - one number,
            # whichever it is, reports the wrong magnitude for the other two.
            "mos_p10": mos[0.1],
            "mos_p25": mos[0.25],
            "mos_p50": mos[0.5],
            "mos_p90": mos[0.9],
            "warnings": len(result.warnings()),
            "constants": dict(spec.constants),
            "drivers": {
                k: {
                    "p10": d.marginal.p10,
                    "p50": d.marginal.p50,
                    "p90": d.marginal.p90,
                    "family": d.marginal.family,
                }
                for k, d in spec.drivers.items()
            },
            "factors": list(spec.factors),
        }
    return out


#: Fields whose value is the same whatever price the spec was run at. Named
#: rather than inlined because that invariance is exactly what the timeline's
#: shared axis rests on, and a field added to one list and not the other would
#: put a price-dependent number into a column labelled price-independent.
_PRICE_FREE = ("value_p10", "value_p25", "value_p50", "value_p90")


def do_timeline(ticker: str) -> dict:
    """Every run of one company, chronologically: Compare generalised to N."""
    names = sorted(
        e["name"] for e in list_analyses() if e["has_yaml"] and _ticker_of(e["name"]) == ticker
    )
    if not names:
        raise KeyError(f"no analysis for ticker: {ticker!r}")

    # The newest run's own price. Read tolerantly - a newest spec too broken to
    # load must not deny the healthy runs their common column.
    common = _spec_price(names[-1])
    runs = []
    for name in names:
        as_of, age = _dated(name)
        own = _cached_row(
            _row_key(name, None, False), lambda: _one_row(name, ticker, as_of, age, None, False)
        )
        runs.append({**own, **_common_columns(name, ticker, as_of, age, own, common)})

    return {"ticker": ticker, "common_price": common, "runs": runs}


def _common_columns(name, ticker, as_of, age, own: dict, common: float | None) -> dict:
    """The same spec re-run at *common*, reduced to the columns that differ."""
    if own.get("status") != "ok" or common is None:
        return {}
    if own["spec_price"] == common:
        return {
            "p_undervalued_common": own["p_undervalued"],
            **{f"{k}_common": own[k] for k in _PRICE_FREE},
        }
    at_common = _cached_row(
        _row_key(name, common, False),
        lambda: _one_row(name, ticker, as_of, age, common, False),
    )
    if at_common.get("status") != "ok":
        return {}
    return {
        "p_undervalued_common": at_common["p_undervalued"],
        **{f"{k}_common": at_common[k] for k in _PRICE_FREE},
    }


#: How far a family's realized band may sit from the stated one, in units of the
#: stated p10-p90 width, before the panel emphasises it. Emphasis only: the drift
#: is reported for every driver regardless, because a number printed only above a
#: threshold makes "fine" and "unmeasured" look the same - which is how a 4.89%
#: tail error passed `validate`'s own thresholded notice in silence once already.
#: 0.10 emphasises cases worth a reader's attention without turning the panel
#: into a wall of flags.
MARGINAL_DRIFT_FLAG = 0.10


def _marginal_row(name: str, p10: float, p50: float, p90: float, family: str) -> dict:
    """One driver's stated band beside the band the engine actually draws."""
    from valdist.core.marginals import Marginal, quantile_fit_notice, tail_regime

    m = Marginal(name, p10=p10, p50=p50, p90=p90, family=family)
    realized = [float(v) for v in m.ppf(_np.array([0.1, 0.5, 0.9]))]
    # A degenerate band (p10 == p50 == p90) is a point mass, and schema-legal.
    # Its width is 0, so it gets 1.0 rather than a division by zero - the drift
    # is 0 either way, and this keeps that true instead of NaN.
    span = abs(p90 - p10) or 1.0
    drift = max(abs(r - s) for r, s in zip(realized, (p10, p50, p90), strict=True)) / span
    return {
        "name": name,
        "family": family,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "r10": realized[0],
        "r50": realized[1],
        "r90": realized[2],
        "drift": drift,
        "flagged": drift > MARGINAL_DRIFT_FLAG,
        # The family's own tail regime, from the engine rather than a
        # lookup table here.
        "tails": tail_regime(family),
        # `validate`'s own words, carried so they sit beside the band they are
        # about instead of in a different view.
        "notice": quantile_fit_notice(family, p10, p50, p90),
    }


def do_model(name: str) -> dict:
    """The whole sampling model: what each driver can be, and how they co-move."""
    from valdist.spec.loader import factor_model

    spec, _model = _load_and_hydrate(name)
    names = list(spec.drivers)
    fm = factor_model(spec)
    lam, psi = fm.loadings_matrix(names), fm.psi(names)

    return {
        "name": name,
        "factors": list(spec.factors),
        "drivers": [
            {
                "name": driver,
                # Dense and ordered, aligned to `factors`. A spec lists only its
                # non-zero loadings, and a sparse payload would leave the
                # renderer to decide what a missing entry means - putting "this
                # factor does not move this driver", which is a judgment somebody
                # made, on screen as "this cell was not filled in".
                "loadings": [float(v) for v in lam[i]],
                "sum_sq": float(lam[i] @ lam[i]),
                "psi": float(psi[i]),
            }
            for i, driver in enumerate(names)
        ],
        # Declared and never loaded on. Not an error and not caught by
        # `validate()` - `unknown_factor` catches the opposite direction - because
        # it is arithmetically inert: a zero column in Λ drops out of ΛΛ'
        # entirely. It is still the shape a mis-typed driver key leaves behind,
        # and that spec validates clean.
        "unloaded_factors": [
            f
            for j, f in enumerate(spec.factors)
            if not any(abs(lam[i][j]) > 0 for i in range(len(names)))
        ],
        # The marginals half. Same order as the loadings rows above, because the
        # two panels sit one under the other and a reader scans them together -
        # different orders would make that read wrong without looking wrong.
        "marginals": [
            _marginal_row(
                driver,
                spec.drivers[driver].marginal.p10,
                spec.drivers[driver].marginal.p50,
                spec.drivers[driver].marginal.p90,
                spec.drivers[driver].marginal.family,
            )
            for driver in names
        ],
    }


def _price(query: dict) -> float | None:
    """Optional `price` override, validated here rather than trusted."""
    values = query.get("price") or []
    if not values or not values[0].strip():
        return None
    raw = values[0].strip()
    try:
        price = float(raw)
    except ValueError:
        raise ValueError(f"price must be a number, got {raw!r}") from None
    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"price must be a finite number greater than 0, got {raw!r}")
    return price


def _tristate(query: dict, key: str) -> bool | None:
    """A flag that can also be absent, which is a third answer and not a False."""
    values = query.get(key) or []
    if not values or not values[0].strip():
        return None
    return values[0].strip().lower() in ("1", "true", "yes", "on")


def _named(query: dict, key: str) -> str:
    values = query.get(key) or []
    if not values or not values[0].strip():
        raise ValueError(f"missing required query parameter: {key}")
    return values[0]


ROUTES = {
    "/api/analyses": lambda q: {"analyses": list_analyses()},
    "/api/analysis": lambda q: read_analysis(_name(q)),
    "/api/validate": lambda q: do_validate(_name(q)),
    "/api/run": lambda q: do_run(_name(q), _price(q)),
    "/api/ladder": lambda q: do_ladder(_name(q), _price(q)),
    "/api/model": lambda q: do_model(_name(q)),
    "/api/timeline": lambda q: do_timeline(_named(q, "ticker")),
    "/api/worlds": lambda q: do_worlds(_name(q)),
    "/api/convergence": lambda q: do_convergence(_name(q)),
    "/api/portfolio": lambda q: do_portfolio(_tristate(q, "repriced")),
    "/api/compare": lambda q: do_compare(_named(q, "a"), _named(q, "b")),
}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


class Handler(BaseHTTPRequestHandler):
    server_version = "valdist-viewer"

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # No external assets are referenced; lock that down so a future edit
        # cannot quietly introduce a CDN (and with it, an outbound request).
        self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj, indent=2).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's contract
        parsed = urlparse(self.path)
        path, query = parsed.path, parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self._send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
            return

        handler = ROUTES.get(path)
        if handler is None:
            self._send_json(404, {"error": f"no such endpoint: {path}"})
            return

        try:
            self._send_json(200, handler(query))
        except KeyError as exc:
            # exc.args[0], not str(exc): KeyError's __str__ is a repr, so it
            # would render the message wrapped in its own quotes.
            self._send_json(404, {"error": exc.args[0] if exc.args else "not found"})
        except SpecError as exc:
            # A spec the validator rejects is a legitimate answer to "run this",
            # not a server fault: hand back the whole digest, never just the
            # first error. Caught before ValueError because it subclasses it.
            self._send_json(
                400,
                {
                    "error": str(exc),
                    "errors": [{"code": e.code, "message": e.message} for e in exc.errors],
                },
            )
        except ValueError as exc:
            # A malformed request - a missing parameter - is the caller's fault.
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            self._send_json(
                500,
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )

    def log_message(self, fmt: str, *args) -> None:
        """One tidy line per request instead of BaseHTTPRequestHandler's format."""
        print(f"  {self.address_string()} - {fmt % args}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8756)
    # Binds loopback by default and there is no flag to widen it: this serves
    # local files and runs local computation, and has no authentication.
    parser.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    found = list_analyses()
    print(f"valdist viewer – {len(found)} analyses under {ANALYSES_DIR}")
    print(f"  http://{args.host}:{args.port}/")
    print("  read-only: no spec is edited and nothing is written back to disk")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
