"""The marginals panel: tail regimes + the `/api/model` payload [gui/, no anchor]."""

from __future__ import annotations

import numpy as np
import pytest

from valdist.core.marginals import FAMILIES, TAIL_REGIME, tail_regime

# --------------------------------------------------------------------------- #
# Two tail regimes, made explicit and made checkable
# --------------------------------------------------------------------------- #
#
# Tails are handled in one of three ways and never a fourth: (a) censored at
# the stated band, (b) extrapolated by families that define tails, (c) floored
# and flagged. (a) and (b) are properties of the marginal families, covered by
# tests/test_smoke.py, so which family is which was knowledge with no home
# until this table existed. The panel
# needs to render it, and the alternative was a lookup table in `gui/`, i.e. a
# second statement of an engine fact free to drift from the engine.


def test_every_family_declares_a_tail_regime() -> None:
    """Exhaustive over FAMILIES, so a new family cannot arrive unclassified."""
    assert set(TAIL_REGIME) == set(FAMILIES), (
        f"classified {sorted(TAIL_REGIME)} against families {sorted(FAMILIES)}"
    )
    assert set(TAIL_REGIME.values()) <= {"censored", "extrapolated"}


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_the_declared_regime_is_what_the_family_actually_does(family: str) -> None:
    """Measured, not declared - the classification is a testable property."""
    from valdist.core.marginals import Marginal

    m = Marginal("x", p10=10.0, p50=20.0, p90=40.0, family=family)
    far = m.ppf(np.array([1e-9, 1.0 - 1e-9]))
    outside = bool(far[0] < m.p10 - 1e-9 or far[1] > m.p90 + 1e-9)

    if tail_regime(family) == "censored":
        assert not outside, (
            f"{family} is classified censored but draws outside its stated band "
            f"({far[0]:g}, {far[1]:g}) vs [{m.p10}, {m.p90}]"
        )
    else:
        assert outside, (
            f"{family} is classified extrapolated but never leaves its stated "
            f"band - so the band is a bound, which counts as censored"
        )


def test_an_unknown_family_is_refused_rather_than_defaulted() -> None:
    """A default would silently label an unknown family as one of the two."""
    with pytest.raises(KeyError):
        tail_regime("not_a_family")


# --------------------------------------------------------------------------- #
# The payload
# --------------------------------------------------------------------------- #


@pytest.fixture
def transplanted(monkeypatch, tmp_path):
    """A real analysis copied into a temp directory, with one marginal edited."""
    import re as _re
    import shutil

    import gui.server as gs

    src = sorted(p for p in gs.ANALYSES_DIR.iterdir() if p.is_dir())[0]
    dst = tmp_path / src.name
    shutil.copytree(src, dst)
    monkeypatch.setattr(gs, "ANALYSES_DIR", tmp_path)

    def plant(count: int = 1) -> str:
        """Rewrite the first *count* marginal lines to a triple that trips the notice."""
        spec = dst / f"{src.name}.yaml"
        text = spec.read_text(encoding="utf-8")
        planted, n = _re.subn(
            r"marginal: \{[^}]*\}",
            "marginal: {family: lognormal, p10: 1.0, p50: 5.0, p90: 10.0}",
            text,
            count=count,
        )
        assert n == count, f"planted {n} marginals, wanted {count} - the plant did not take"
        spec.write_text(planted, encoding="utf-8")
        return src.name

    return plant


@pytest.fixture(scope="module")
def model_payload():
    """A real shipped analysis, guaranteed to have something to sweep."""
    from gui.server import ANALYSES_DIR, do_model

    name = sorted(p.name for p in ANALYSES_DIR.iterdir() if p.is_dir())[0]
    payload = do_model(name)
    assert payload["marginals"], f"{name} produced no marginals - every sweep below would pass"
    assert payload["drivers"], f"{name} produced no driver rows - same"
    return payload


def test_the_model_payload_still_carries_the_loadings(model_payload) -> None:
    """The tab holds both halves of the sampling model; neither may drop out."""
    assert model_payload["factors"] is not None
    assert model_payload["drivers"], "the loadings half of the model tab is gone"
    assert model_payload["marginals"], "the marginals half is missing"


def test_marginals_are_in_spec_order_and_pair_with_the_loadings(model_payload) -> None:
    """Same order as everything else - tornado, worlds, the YAML, the Λ rows."""
    assert [m["name"] for m in model_payload["marginals"]] == [
        d["name"] for d in model_payload["drivers"]
    ]


def test_realized_quantiles_are_the_ppf_of_the_stated_triple(model_payload) -> None:
    """Exact, from the same object the sampler draws through."""
    from valdist.core.marginals import Marginal

    for m in model_payload["marginals"]:
        mm = Marginal("x", p10=m["p10"], p50=m["p50"], p90=m["p90"], family=m["family"])
        expected = mm.ppf(np.array([0.1, 0.5, 0.9]))
        assert (m["r10"], m["r50"], m["r90"]) == pytest.approx(tuple(expected))


def test_every_marginal_reports_its_drift_not_only_the_flagged_ones(model_payload) -> None:
    """The lesson from `quantile_mismatch`, applied."""
    for m in model_payload["marginals"]:
        assert isinstance(m["drift"], float)
        assert m["drift"] >= 0.0
        assert isinstance(m["flagged"], bool)


def test_drift_is_measured_against_the_bands_own_width() -> None:
    """Not as a percentage of the value - that instrument is a known error mode."""
    from gui.server import _marginal_row

    tiny = _marginal_row("x", 1.0, 7.0, 11.0, "normal")
    assert tiny["drift"] == pytest.approx(0.1, abs=1e-9), (
        "a 1-unit miss on a 10-wide band is 0.1 of a band, whatever it is as a "
        "percentage of the endpoint"
    )


def test_a_bounds_family_reports_censored_tails_and_a_large_drift() -> None:
    """The A50 case, which is the one a reader most needs told."""
    from gui.server import _marginal_row

    row = _marginal_row("nav_other", -1709.0, -1530.0, 119.6, "pert")

    assert row["tails"] == "censored"
    assert row["r90"] == pytest.approx(-866.5, abs=0.5)
    assert row["drift"] > 0.5
    assert row["flagged"] is True


def test_lognormal3_reproduces_its_stated_triple() -> None:
    """The control: a family that does hit its numbers must report ~zero drift."""
    from gui.server import _marginal_row

    row = _marginal_row("cap_rate", 5.5, 6.4, 7.5, "lognormal3")

    assert row["tails"] == "extrapolated"
    assert row["drift"] == pytest.approx(0.0, abs=1e-9)
    assert row["flagged"] is False


def test_a_fit_notice_actually_reaches_the_payload() -> None:
    """Planted, because the sweep below cannot see this and was green without it."""
    from gui.server import _marginal_row
    from valdist.core.marginals import quantile_fit_notice

    with pytest.warns(UserWarning):
        row = _marginal_row("cap_rate", 1.0, 5.0, 10.0, "lognormal")

    assert row["notice"], "a marginal the validator would notice carries no notice"
    assert row["notice"] == quantile_fit_notice("lognormal", 1.0, 5.0, 10.0)
    assert "geometric mean" in row["notice"]


def test_a_planted_notice_survives_the_whole_endpoint(transplanted) -> None:
    """And it reaches `/api/model`, not merely `_marginal_row`."""
    from gui.server import do_model

    name = transplanted(count=1)
    with pytest.warns(UserWarning):
        d = do_model(name)

    assert any(m["notice"] for m in d["marginals"]), (
        "no marginal in the transplanted spec carries a notice - the plant did "
        "not take, so this case proves nothing"
    )


def test_a_clean_marginal_carries_no_notice(model_payload) -> None:
    """The other direction, and the reason the sweep above stays."""
    from valdist.core.marginals import quantile_fit_notice

    for m in model_payload["marginals"]:
        assert m["notice"] is None
        assert quantile_fit_notice(m["family"], m["p10"], m["p50"], m["p90"]) is None


def test_a_degenerate_band_does_not_divide_by_zero() -> None:
    """p10 == p50 == p90 is schema-legal: a point mass, not a distribution."""
    from gui.server import _marginal_row

    row = _marginal_row("shares", 5.0, 5.0, 5.0, "normal")

    assert np.isfinite(row["drift"])
    assert row["drift"] == pytest.approx(0.0)
