"""The ticker timeline: `/api/timeline` [gui/, no anchor]."""

from __future__ import annotations

import pytest

import gui.server as gs


def _multi_run_tickers() -> dict[str, list[str]]:
    by_ticker: dict[str, list[str]] = {}
    for entry in gs.list_analyses():
        if not entry["has_yaml"]:
            continue
        ticker = gs._ticker_of(entry["name"])
        if ticker:
            by_ticker.setdefault(ticker, []).append(entry["name"])
    return {t: sorted(names) for t, names in by_ticker.items() if len(names) > 1}


@pytest.fixture(scope="module")
def multi_run() -> dict[str, list[str]]:
    found = _multi_run_tickers()
    assert found, (
        "no ticker has more than one run, so every case below "
        "would assert about a timeline of length 1 - which is the degenerate "
        "case, not the subject"
    )
    return found


def test_runs_come_back_oldest_first(multi_run) -> None:
    """A timeline reads forward. Newest-first is the sidebar's order, not this one."""
    for ticker in multi_run:
        d = gs.do_timeline(ticker)
        dates = [r["as_of"] for r in d["runs"]]
        assert dates == sorted(dates), f"{ticker}: {dates} is not chronological"


def test_the_value_view_does_not_depend_on_the_price_it_was_run_at(multi_run) -> None:
    """What licenses a shared axis across the strips."""
    ticker = next(iter(multi_run))
    d = gs.do_timeline(ticker)
    for r in d["runs"]:
        assert r["value_p50"] == r["value_p50_common"], (
            "the same spec produced a different median at a different price - "
            "the shared axis is not comparable"
        )
        assert r["value_p10"] == r["value_p10_common"]


def test_every_run_reports_both_p_undervalued_figures(multi_run) -> None:
    """Saved, and re-priced to one common price. Both, always, not a toggle."""
    for ticker, names in multi_run.items():
        d = gs.do_timeline(ticker)
        assert len(d["runs"]) == len(names)
        for r in d["runs"]:
            assert isinstance(r["p_undervalued"], float)
            assert isinstance(r["p_undervalued_common"], float)


def test_the_common_price_is_the_newest_specs_own(multi_run) -> None:
    """Named, stable and reproducible - and the newest run is its own control."""
    from valdist.spec.loader import load_spec

    for ticker, names in multi_run.items():
        d = gs.do_timeline(ticker)
        newest = load_spec(gs._spec_path(names[-1])).price

        assert d["common_price"] == newest
        last = d["runs"][-1]
        assert last["name"] == names[-1]
        assert last["p_undervalued"] == last["p_undervalued_common"], (
            f"{ticker}: the newest run is priced at the common price by "
            "definition, so its two columns cannot differ"
        )


def test_a_ticker_with_one_run_is_a_valid_timeline(multi_run) -> None:
    """Degenerate, not an error - and it has to say it cannot show drift."""
    singles = [
        gs._ticker_of(e["name"])
        for e in gs.list_analyses()
        if e["has_yaml"] and gs._ticker_of(e["name"]) and gs._ticker_of(e["name"]) not in multi_run
    ]
    assert singles, "every ticker has multiple runs - this case cannot be exercised"

    d = gs.do_timeline(singles[0])
    assert len(d["runs"]) == 1
    assert d["common_price"] == d["runs"][0]["spec_price"]


def test_an_unknown_ticker_is_refused_rather_than_answered_empty() -> None:
    """An empty timeline and a mistyped ticker must not look the same."""
    with pytest.raises(KeyError):
        gs.do_timeline("NOTATICKER")


def test_the_canonical_decomposition_is_pinned_exactly() -> None:
    """The canonical case, pinned exactly - the engine is deterministic."""
    from conftest import resolve_code, skip_unless_private_record

    skip_unless_private_record()
    ticker = resolve_code("A12")

    if ticker not in _multi_run_tickers():
        pytest.fail(
            "the A12 analyses are the canonical evidence for the timeline's "
            "second column and they are gone - see this test's docstring"
        )

    d = gs.do_timeline(ticker)
    assert len(d["runs"]) == 3, (
        "this case pins the three A12 runs; the count changed, so the numbers "
        "below are about a set that no longer exists - see the docstring"
    )
    first, mid, last = d["runs"]

    # The common price is the newest spec's own, so adding a run re-bases every
    # earlier row's second column. That is why `first` moved from 0.3823 (at the
    # old common price of 74.90) to 0.4045 here without either spec changing.
    assert d["common_price"] == pytest.approx(73.97)
    assert first["p_undervalued"] == pytest.approx(0.2862, abs=1e-4)
    assert first["p_undervalued_common"] == pytest.approx(0.4045, abs=1e-4)
    assert mid["p_undervalued"] == pytest.approx(0.4272, abs=1e-4)
    assert mid["p_undervalued_common"] == pytest.approx(0.4518, abs=1e-4)
    assert last["p_undervalued"] == pytest.approx(0.5075, abs=1e-4)
    # The newest run IS the common price, so its two columns must coincide.
    assert last["p_undervalued_common"] == pytest.approx(last["p_undervalued"], abs=1e-9)

    saved_swing = last["p_undervalued"] - first["p_undervalued"]
    view_swing = last["p_undervalued_common"] - first["p_undervalued_common"]
    assert saved_swing == pytest.approx(0.2213, abs=1e-4)
    assert view_swing == pytest.approx(0.1030, abs=1e-4)
    assert saved_swing > 2 * view_swing, (
        "the saved column reports over twice the movement the view actually "
        "made - if this stops being true the header's claim must change with it"
    )


def test_a_broken_spec_does_not_sink_the_whole_timeline(tmp_path, monkeypatch) -> None:
    """Same rule the portfolio follows: a bad row comes back AS a row."""
    import shutil

    ticker = next(iter(_multi_run_tickers()))
    names = _multi_run_tickers()[ticker]
    for name in names:
        shutil.copytree(gs.ANALYSES_DIR / name, tmp_path / name)
    spec = tmp_path / names[0] / f"{names[0]}.yaml"
    spec.write_text(spec.read_text(encoding="utf-8").replace("price:", "price: -1  #", 1))
    monkeypatch.setattr(gs, "ANALYSES_DIR", tmp_path)

    d = gs.do_timeline(ticker)
    assert d["runs"][0]["status"] == "error"
    assert d["runs"][0]["error"]
    assert d["runs"][-1]["status"] == "ok", "one broken spec took the healthy runs with it"
