"""How `prices/fetch.py` writes its artefact, and what it must not lose."""

from __future__ import annotations

import json
import pathlib

import pytest

from prices import fetch as pf

DAY = "2026-07-26"


def _quote(ticker: str, price: float, as_of: str = DAY) -> dict:
    return {
        "ticker": ticker,
        "symbol": ticker,
        "price": price,
        "as_of": as_of,
        "currency": "USD",
    }


def _existing(path: pathlib.Path, prices: dict, failures: dict | None = None) -> None:
    """An artefact written by an earlier run."""
    path.write_text(
        json.dumps(
            {
                "as_of": "2026-07-20T09:00:00+00:00",
                "source": "Yahoo Finance chart API (daily close)",
                "prices": prices,
                "failures": failures or {},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _read(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def out(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "prices.json"


@pytest.fixture
def fake_fetch(monkeypatch: pytest.MonkeyPatch):
    """Install a fetcher over a `{ticker: price-or-exception}` table."""

    def install(table: dict):
        def fetch_one(
            ticker: str,
            timeout: float = 20.0,
            symbols: tuple[dict, set] | None = None,
        ) -> dict:
            value = table[ticker]
            if isinstance(value, Exception):
                raise value
            return _quote(ticker, value)

        monkeypatch.setattr(pf, "fetch_one", fetch_one)

    return install


# --------------------------------------------------------------------------- #
# A targeted run updates; it does not replace
# --------------------------------------------------------------------------- #


def test_targeted_run_keeps_the_tickers_it_was_not_asked_about(out, fake_fetch) -> None:
    """The regression: `python -m prices.fetch A51` kept only A51."""
    _existing(out, {"A12": _quote("A12", 74.90), "A01": _quote("A01", 512.0)})
    fake_fetch({"A51": 41.5})

    assert pf.main(["A51", "--out", str(out)]) == 0

    prices = _read(out)["prices"]
    assert sorted(prices) == ["A01", "A12", "A51"]
    assert prices["A12"]["price"] == 74.90
    assert prices["A01"]["price"] == 512.0


def test_targeted_run_refreshes_the_ticker_it_was_asked_about(out, fake_fetch) -> None:
    _existing(out, {"A12": _quote("A12", 74.90)})
    fake_fetch({"A12": 80.25})

    pf.main(["A12", "--out", str(out)])

    assert _read(out)["prices"]["A12"]["price"] == 80.25


def test_a_ticker_that_now_fails_loses_its_stale_price(out, fake_fetch) -> None:
    """A price and a failure for the same ticker would contradict each other."""
    _existing(out, {"A12": _quote("A12", 74.90), "A01": _quote("A01", 512.0)})
    fake_fetch({"A12": ValueError("no chart data for 'A12'")})

    pf.main(["A12", "--out", str(out)])

    doc = _read(out)
    assert "A12" not in doc["prices"]
    assert "A12" in doc["failures"]
    assert "A01" in doc["prices"], "an unrelated ticker must be untouched"


def test_a_ticker_that_now_succeeds_clears_its_stale_failure(out, fake_fetch) -> None:
    _existing(out, {"A01": _quote("A01", 512.0)}, failures={"A12": "URLError: timed out"})
    fake_fetch({"A12": 74.90})

    pf.main(["A12", "--out", str(out)])

    doc = _read(out)
    assert doc["failures"] == {}
    assert doc["prices"]["A12"]["price"] == 74.90


def test_unrelated_failures_survive_a_targeted_run(out, fake_fetch) -> None:
    _existing(out, {}, failures={"A01": "URLError: timed out"})
    fake_fetch({"A12": 74.90})

    pf.main(["A12", "--out", str(out)])

    assert "A01" in _read(out)["failures"]


# --------------------------------------------------------------------------- #
# A full run still replaces
# --------------------------------------------------------------------------- #


def test_full_run_replaces_so_a_deleted_analysis_does_not_linger(
    out, fake_fetch, monkeypatch
) -> None:
    """Merging every run would resurrect tickers whose analysis is gone."""
    _existing(out, {"OLD": _quote("OLD", 1.0), "A12": _quote("A12", 74.90)})
    monkeypatch.setattr(pf, "tickers_from_analyses", lambda: ["A12"])
    fake_fetch({"A12": 80.25})

    pf.main(["--out", str(out)])

    assert sorted(_read(out)["prices"]) == ["A12"]


# --------------------------------------------------------------------------- #
# The document's `as_of` must not overstate freshness
# --------------------------------------------------------------------------- #


def test_as_of_reports_the_oldest_row_not_the_newest(out, fake_fetch) -> None:
    """The viewer renders one `as_of` for the whole table (`gui/index.html`)."""
    _existing(out, {"A12": _quote("A12", 74.90)})
    previous = _read(out)["as_of"]
    fake_fetch({"A51": 41.5})

    pf.main(["A51", "--out", str(out)])

    doc = _read(out)
    assert doc["as_of"] == previous, "a merged document is only as fresh as its oldest row"
    assert doc["prices"]["A51"]["fetched_at"] > previous
    assert doc["prices"]["A12"]["fetched_at"] == previous


def test_every_written_row_records_when_it_was_fetched(out, fake_fetch) -> None:
    """Per-row provenance is what makes the oldest-row rule computable."""
    fake_fetch({"A12": 74.90})
    pf.main(["A12", "--out", str(out)])

    row = _read(out)["prices"]["A12"]
    assert row["as_of"] == DAY
    assert row["fetched_at"].startswith("20")


def test_full_run_as_of_is_the_run_time(out, fake_fetch, monkeypatch) -> None:
    """No merge, no old rows - so the oldest row is this run."""
    monkeypatch.setattr(pf, "tickers_from_analyses", lambda: ["A12"])
    fake_fetch({"A12": 74.90})

    pf.main(["--out", str(out)])

    doc = _read(out)
    assert doc["as_of"] == doc["prices"]["A12"]["fetched_at"]


# --------------------------------------------------------------------------- #
# Merging into nothing, or into something broken
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("label", "content"),
    [
        ("truncated json", '{"prices": {"A12":'),
        ("top-level list", "[]"),
        ("prices is not an object", '{"as_of": "x", "prices": []}'),
    ],
)
def test_a_malformed_existing_file_does_not_lose_the_new_fetch(
    out, fake_fetch, label: str, content: str
) -> None:
    """An unreadable file is treated as no file, not as a reason to fail."""
    out.write_text(content, encoding="utf-8")
    fake_fetch({"A12": 74.90})

    assert pf.main(["A12", "--out", str(out)]) == 0
    assert _read(out)["prices"]["A12"]["price"] == 74.90


def test_targeted_run_with_no_existing_file_just_writes(out, fake_fetch) -> None:
    fake_fetch({"A12": 74.90})

    assert pf.main(["A12", "--out", str(out)]) == 0
    assert sorted(_read(out)["prices"]) == ["A12"]


def test_a_run_where_everything_fails_returns_nonzero(out, fake_fetch) -> None:
    """Exit code reports *this run*, not the file's accumulated contents."""
    _existing(out, {"A01": _quote("A01", 512.0)})
    fake_fetch({"A12": ValueError("no chart data for 'A12'")})

    assert pf.main(["A12", "--out", str(out)]) == 1
    assert "A01" in _read(out)["prices"], "the failed run must still not lose data"


# --------------------------------------------------------------------------- #
# The symbol is checked before it reaches a URL
# --------------------------------------------------------------------------- #
#
# The fetcher half. `fetch_one` formats its symbol straight into `CHART_URL`,
# so an odd string produced a malformed request rather than a clear refusal.
# The check is here as well as in the viewer because the two cannot share
# one: `gui/` does not import `prices`, so the duplicated pattern is
# deliberate.


@pytest.mark.parametrize(
    ("label", "ticker"),
    [
        ("path traversal", "../etc/passwd"),
        ("query injection", "A?x=1"),
        ("empty", ""),
        ("whitespace", "A B"),
        ("too long", "A" * 13),
    ],
)
def test_a_malformed_symbol_is_refused_without_a_request(
    label: str, ticker: str, monkeypatch
) -> None:
    """Refused before the network, not after - and loudly, as a ValueError."""

    def never(*a, **k):
        raise AssertionError("a request was made for a symbol that should have been refused")

    monkeypatch.setattr(pf.urllib.request, "urlopen", never)

    with pytest.raises(ValueError, match="symbol"):
        pf.fetch_one(ticker)


# `test_a_real_symbol_including_an_exchange_suffix_is_accepted` used to sit here.
# It iterated `pf.SYMBOL_OVERRIDES` and `pf.US_LISTED` to prove the guard admits
# a real suffixed symbol - which made it a test over the private record, and it
# left with the tables (2026-09-04). The property it
# defended is now `test_the_resolved_symbol_reaches_the_url` in
# `tests/test_symbols_table.py`, driven through crafted fixtures instead: an
# override keeps its suffix, a US listing stays bare, and neither reaches the
# network by a path the guard did not approve.


# --------------------------------------------------------------------------- #
# The write itself is all-or-nothing
# --------------------------------------------------------------------------- #
#
# A truncated write was once rated low severity because the file is gitignored
# scratch that an argument-less re-run repairs. Merging changed the stakes that
# rating was set against: the document being overwritten now carries twelve rows
# this run did not fetch, so a crash mid-write loses all of them, and re-fetching
# is the only way back. Cheap enough to close rather than keep rating.


def test_an_interrupted_write_leaves_the_previous_file_intact(out, fake_fetch, monkeypatch) -> None:
    """The point of the whole section: a failed write loses nothing."""
    _existing(out, {"A12": _quote("A12", 74.90), "A01": _quote("A01", 512.0)})
    before = out.read_text(encoding="utf-8")
    fake_fetch({"A51": 41.5})

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(pf.os, "replace", boom)

    with pytest.raises(OSError):
        pf.main(["A51", "--out", str(out)])

    assert out.read_text(encoding="utf-8") == before


def test_an_interrupted_write_leaves_no_stray_temp_file(out, fake_fetch, monkeypatch) -> None:
    """A litter of `.tmp` siblings would be its own quiet mess."""
    _existing(out, {"A12": _quote("A12", 74.90)})
    fake_fetch({"A51": 41.5})

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(pf.os, "replace", boom)

    with pytest.raises(OSError):
        pf.main(["A51", "--out", str(out)])

    assert sorted(p.name for p in out.parent.iterdir()) == ["prices.json"]


def test_the_temp_file_is_a_sibling_of_the_destination(out, fake_fetch, monkeypatch) -> None:
    """`os.replace` is atomic only *within* a filesystem."""
    seen = {}
    real_replace = pf.os.replace

    def recording(src, dst):
        seen["src"] = pathlib.Path(src)
        real_replace(src, dst)

    monkeypatch.setattr(pf.os, "replace", recording)
    fake_fetch({"A12": 74.90})

    pf.main(["A12", "--out", str(out)])

    assert seen["src"].parent == out.parent


def test_a_successful_write_leaves_only_the_artefact(out, fake_fetch) -> None:
    fake_fetch({"A12": 74.90})
    pf.main(["A12", "--out", str(out)])

    assert sorted(p.name for p in out.parent.iterdir()) == ["prices.json"]
    assert _read(out)["prices"]["A12"]["price"] == 74.90
