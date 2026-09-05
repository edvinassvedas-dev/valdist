"""The private symbol table: its loader, its guards, and what moved into them.

`SYMBOL_OVERRIDES` and `US_LISTED` used to be dict literals in `prices/fetch.py`.
Together they enumerated the research record, so a tracked copy was a complete
index of private work in a module whose stated job is exchange suffixes. They
now live in `symbols.json` beside the record.

Two checks left with them, and only one came back the same way.

* The shape rules - an override must read `<TICKER>.<EXCHANGE>` with the ticker
  unchanged, and the two maps must not disagree - were assertions about a table,
  not about private data. They moved into `load_symbols()`, so they now hold for
  any table anyone loads rather than only for the one this repo shipped. The
  cases below drive them through crafted fixtures and are fully public.
* Coverage - every ticker under the record is classified - is irreducibly a claim
  about private data. It runs only where the record exists, and skips rather than
  passes when it does not, so it never reads as a check that was made.

And one check got stronger by moving: classification is now enforced in
`fetch_one` itself, not by a test. Previously the code did `.get(ticker, ticker)`
and nothing read `US_LISTED` at all - a test was the only thing standing between
a new listing and a silent bare lookup. When an enforcement mechanism is
lost, the response is to replace it, or say plainly it is gone.
"""

from __future__ import annotations

import json
import pathlib

import pytest

import prices.fetch as pf

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _write(root: pathlib.Path, doc) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / pf.SYMBOLS_FILENAME
    path.write_text(json.dumps(doc) if not isinstance(doc, str) else doc, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Loading


def test_an_absent_table_is_empty_rather_than_an_error(tmp_path) -> None:
    """Safe only because `fetch_one` refuses what it cannot classify.

    A public checkout has no record, so there are no tickers to classify and
    nothing to refuse. A checkout with a record and no table refuses every
    ticker by name - which is the loud failure - instead of quietly bare-looking
    up one at a time, which is the silent one.
    """
    assert pf.load_symbols(tmp_path) == ({}, set())


def test_a_malformed_table_raises_rather_than_being_skipped(tmp_path) -> None:
    """Skipping what will not parse reinstates the bare lookup, silently."""
    _write(tmp_path, "{not json at all")
    with pytest.raises(ValueError, match="unreadable symbol table"):
        pf.load_symbols(tmp_path)


def test_a_json_scalar_is_not_a_table(tmp_path) -> None:
    _write(tmp_path, [1, 2, 3])
    with pytest.raises(ValueError, match="expected a JSON object"):
        pf.load_symbols(tmp_path)


def test_an_override_whose_base_symbol_is_not_its_key_is_refused(tmp_path) -> None:
    """The wrong-company shape, and the reason it cannot be a warning.

    `{"BBB": "CCC.PA"}` does not error at fetch time and does not return
    nothing. It returns a real, current, correct price - for whatever CCC.PA
    is - under the key BBB, and every comparison downstream is then honestly
    computed against the wrong company.
    """
    _write(tmp_path, {"overrides": {"BBB": {"symbol": "CCC.PA"}}})
    with pytest.raises(ValueError, match="different company"):
        pf.load_symbols(tmp_path)


def test_an_override_without_an_exchange_suffix_is_refused(tmp_path) -> None:
    _write(tmp_path, {"overrides": {"BBB": {"symbol": "BBB"}}})
    with pytest.raises(ValueError, match="TICKER.*EXCHANGE"):
        pf.load_symbols(tmp_path)


def test_an_override_with_no_symbol_at_all_is_refused(tmp_path) -> None:
    _write(tmp_path, {"overrides": {"BBB": {"note": "meant to fill this in"}}})
    with pytest.raises(ValueError, match="no symbol"):
        pf.load_symbols(tmp_path)


def test_a_ticker_in_both_maps_is_refused(tmp_path) -> None:
    """The two maps answer the same question and must not disagree."""
    _write(
        tmp_path,
        {"overrides": {"AAA": {"symbol": "AAA.L"}}, "us_listed": {"AAA": {"note": ""}}},
    )
    with pytest.raises(ValueError, match="both non-US and US-listed"):
        pf.load_symbols(tmp_path)


def test_notes_are_carried_by_the_file_and_ignored_by_the_loader(tmp_path) -> None:
    """The reasoning is the table's most valuable content, so it must survive.

    ~500 lines of it recorded which bare symbols were checked and what they
    returned. The loader does not need it; the next person to add a ticker does.
    """
    doc = {
        "overrides": {"AAA": {"symbol": "AAA.L", "note": "bare AAA is a degenerate row"}},
        "us_listed": {"BBB": {"note": "four letters, no collision"}},
    }
    path = _write(tmp_path, doc)
    overrides, us_listed = pf.load_symbols(tmp_path)
    assert overrides == {"AAA": "AAA.L"} and us_listed == {"BBB"}
    round_trip = json.loads(path.read_text())
    assert round_trip["overrides"]["AAA"]["note"]
    assert round_trip["us_listed"]["BBB"]["note"]


def test_a_bare_string_override_is_still_accepted(tmp_path) -> None:
    """`{"AAA": "AAA.L"}` without the note wrapper stays legal - the notes are a
    convenience for the author, not a schema the loader may hold hostage."""
    _write(tmp_path, {"overrides": {"AAA": "AAA.L"}})
    assert pf.load_symbols(tmp_path) == ({"AAA": "AAA.L"}, set())


# --------------------------------------------------------------------------- #
# The guard that moved from a test into the runtime


def _no_network(monkeypatch):
    def never(*a, **k):
        raise AssertionError("a request was made for a ticker that should have been refused")

    monkeypatch.setattr(pf.urllib.request, "urlopen", never)


def test_an_unclassified_ticker_is_refused_before_any_request(monkeypatch) -> None:
    _no_network(monkeypatch)
    with pytest.raises(ValueError, match="unclassified ticker"):
        pf.fetch_one("ZZZZ", symbols=({}, set()))


def test_a_record_with_no_table_refuses_every_ticker_by_name(monkeypatch, tmp_path) -> None:
    """The pairing that makes an absent table safe rather than silent."""
    _no_network(monkeypatch)
    monkeypatch.setattr(pf, "ANALYSES_DIR", tmp_path)
    with pytest.raises(ValueError, match="unclassified ticker 'CCC'"):
        pf.fetch_one("CCC")


@pytest.mark.parametrize(
    "ticker,symbols,expected",
    [
        ("CCC", ({"CCC": "CCC.PA"}, set()), "CCC.PA"),
        ("BBB", ({}, {"BBB"}), "BBB"),
    ],
)
def test_the_resolved_symbol_reaches_the_url(monkeypatch, ticker, symbols, expected) -> None:
    """An override keeps its suffix; a US listing stays bare. Both, or neither."""
    seen = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        return _Resp()

    monkeypatch.setattr(pf.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(pf.json, "load", lambda _resp: {"chart": {"result": []}})
    with pytest.raises(ValueError, match="no chart data"):
        pf.fetch_one(ticker, symbols=symbols)
    assert f"/chart/{expected}?" in seen["url"], seen["url"]


# --------------------------------------------------------------------------- #
# Coverage: a claim about private data, so conditional by necessity


def _record_or_skip() -> tuple[pathlib.Path, list[str]]:
    root = pf.ANALYSES_DIR
    if not root.is_dir():
        pytest.skip(f"no research record at {root} - coverage is a claim about private data")
    tickers = pf.tickers_from_analyses()
    if not tickers:
        pytest.skip(f"record at {root} holds no analyses")
    return root, tickers


def test_every_ticker_in_the_record_is_classified() -> None:
    root, tickers = _record_or_skip()
    overrides, us_listed = pf.load_symbols(root)
    unclassified = sorted(t for t in tickers if t not in overrides and t not in us_listed)
    assert unclassified == [], (
        f"{len(unclassified)} ticker(s) in the record have no declared listing venue. "
        f"Add each to {root / pf.SYMBOLS_FILENAME}. `fetch_one` refuses them, so this "
        "is a loud failure rather than a wrong price - but it is still a failure."
    )


def test_the_table_names_no_ticker_the_record_does_not_have() -> None:
    """A dead key leaves the real ticker falling through - to a refusal now,
    rather than to a bare lookup, but a stale table still misleads its reader."""
    root, tickers = _record_or_skip()
    overrides, us_listed = pf.load_symbols(root)
    known = set(tickers)
    dead = sorted((set(overrides) | us_listed) - known)
    assert dead == [], f"{root / pf.SYMBOLS_FILENAME} names {len(dead)} ticker(s) with no analysis"
