"""Per-driver value swings [sensitivity contract, no anchor]."""

from __future__ import annotations

import pytest

from valdist import load_spec, run
from valdist.core import Marginal, Model
from valdist.core.factors import FactorModel
from valdist.spec.loader import hydrate


@pytest.fixture(scope="module")
def vici():
    from conftest import VICI_SPEC_PATH

    spec = load_spec(VICI_SPEC_PATH)
    return spec, hydrate(spec)


def test_the_golden_spec_swings_are_exactly_these(vici) -> None:
    """Deterministic, so knowable exactly - a band around them would be a check
    built not to fire. 1e-4 leaves room for a scipy patch
    difference in the quantile path and nothing else."""
    _, model = vici
    swings = {s["name"]: s["swing"] for s in model.driver_swings()}

    assert swings["cap_rate"] == pytest.approx(7.252072, abs=1e-4)
    assert swings["cost_of_equity"] == pytest.approx(4.038902, abs=1e-4)
    assert swings["affo_growth"] == pytest.approx(3.572761, abs=1e-4)
    assert swings["affo_terminal"] == pytest.approx(1.146481, abs=1e-4)
    assert swings["div_terminal"] == pytest.approx(0.465999, abs=1e-4)
    assert swings["div_growth"] == pytest.approx(0.376006, abs=1e-4)


def test_the_swing_ranking_disagrees_with_the_tornado(vici) -> None:
    """The finding this feature exists for, pinned so it cannot quietly stop
    being true. If the golden spec is ever edited such that the two rankings
    agree, this feature has lost its motivating example and that should be a
    conversation, not a silent pass."""
    spec, model = vici
    by_bar = [name for name, _ in run(spec).tornado()]
    by_swing = [s["name"] for s in sorted(model.driver_swings(), key=lambda s: -s["swing"])]

    assert by_bar != by_swing, "the two rankings agree - the motivating case is gone"
    assert by_bar.index("div_growth") == 3, "div_growth is no longer the fourth bar"
    assert by_swing[-1] == "div_growth", "div_growth is no longer the smallest swing"
    assert by_swing.index("affo_terminal") < by_swing.index("div_growth"), (
        "affo_terminal no longer outranks div_growth on swing, which is the inversion"
    )


def test_the_endpoints_are_what_the_model_draws_not_what_was_typed() -> None:
    """`pert` reads p10/p90 as hard bounds, so its realized decile is nowhere near
    the typed one. A swing across the typed band would be a confident number about
    draws the model never makes.
    """
    fm = FactorModel([], {})
    m = Marginal("x", p10=0.0, p50=5.0, p90=10.0, family="pert")
    model = Model([m], fm, valuation=lambda v: v["x"])

    (swing,) = model.driver_swings()
    lo, hi = float(m.ppf([0.1])[0]), float(m.ppf([0.9])[0])

    assert swing["input_lo"] == pytest.approx(lo)
    assert swing["input_hi"] == pytest.approx(hi)
    assert (swing["input_lo"], swing["input_hi"]) != (0.0, 10.0), (
        "used the typed bounds, which pert does not draw as its deciles"
    )
    # The valuation is the identity, so the swing IS the realized band width.
    assert swing["swing"] == pytest.approx(hi - lo)


def test_constants_reach_the_valuation() -> None:
    """The swing is evaluated through `_evaluate`, the same path `run()` uses, so
    a valuation that reads a constant works here exactly as it does in a run."""
    fm = FactorModel([], {})
    model = Model(
        [Marginal("x", p10=1.0, p50=2.0, p90=3.0, family="normal")],
        fm,
        valuation=lambda v: v["x"] * v["scale"],
        constants={"scale": 10.0},
    )
    (swing,) = model.driver_swings()
    assert swing["swing"] == pytest.approx(10.0 * (swing["input_hi"] - swing["input_lo"]))


def test_a_driver_the_valuation_ignores_swings_nothing() -> None:
    """Zero, not absent. A driver that moves the answer not at all is a finding -
    most likely a spec that samples something the adapter never reads - and it
    must be visible rather than missing from the list."""
    fm = FactorModel([], {})
    model = Model(
        [
            Marginal("used", p10=1.0, p50=2.0, p90=3.0, family="normal"),
            Marginal("ignored", p10=1.0, p50=2.0, p90=3.0, family="normal"),
        ],
        fm,
        valuation=lambda v: v["used"],
    )
    swings = {s["name"]: s["swing"] for s in model.driver_swings()}
    assert set(swings) == {"used", "ignored"}
    assert swings["ignored"] == 0.0
    assert swings["used"] > 0.0


def test_it_needs_no_run_and_repeats_exactly(vici) -> None:
    """No RNG anywhere in it: same model, same numbers, every time."""
    _, model = vici
    assert model.driver_swings() == model.driver_swings()


def test_a_model_with_no_drivers_returns_nothing_rather_than_failing() -> None:
    """`FactorModel([], {})` with no drivers is schema-legal - a valuation of
    pure constants - and must not divide by an empty band."""
    model = Model([], FactorModel([], {}), valuation=lambda v: 1.0)
    assert model.driver_swings() == []
