"""Direct coverage of adapters/floors.py: previously only exercised
indirectly through test_reit.py/test_equity.py's adapter-level tests.
floored() is the one place the floor-and-flag pairing is implemented, so it
deserves tests that don't depend on a full adapter call.
"""

from __future__ import annotations

import math

from valdist.adapters.floors import MIN_DIVISOR, floored
from valdist.core.diagnostics import collect_diagnostics


def test_value_above_floor_passes_through_unchanged_no_flag():
    with collect_diagnostics() as counts:
        result = floored(0.05, "test_flag")
    assert result == 0.05
    assert "test_flag" not in counts


def test_value_below_floor_is_floored_and_flags():
    with collect_diagnostics() as counts:
        result = floored(-0.01, "test_flag")
    assert result == MIN_DIVISOR
    assert counts["test_flag"] == 1


def test_flag_fires_once_per_floored_call():
    with collect_diagnostics() as counts:
        floored(-0.01, "test_flag")
        floored(-0.02, "test_flag")
        floored(0.5, "test_flag")  # above floor, should not add to the count
    assert counts["test_flag"] == 2


def test_custom_floor_parameter():
    with collect_diagnostics() as counts:
        result = floored(0.5, "test_flag", floor=1.0)
    assert result == 1.0
    assert counts["test_flag"] == 1


def test_custom_floor_above_value_does_not_bind_when_value_already_higher():
    with collect_diagnostics() as counts:
        result = floored(5.0, "test_flag", floor=1.0)
    assert result == 5.0
    assert "test_flag" not in counts


def test_exact_floor_value_does_not_fire_flag():
    """The floor didn't change anything - value == floor - so per floored()'s
    own contract ("fires only when the floor actually bound") no flag fires."""
    with collect_diagnostics() as counts:
        result = floored(MIN_DIVISOR, "test_flag")
    assert result == MIN_DIVISOR
    assert "test_flag" not in counts


def test_positive_infinity_passes_through_unchanged():
    with collect_diagnostics() as counts:
        result = floored(math.inf, "test_flag")
    assert result == math.inf
    assert "test_flag" not in counts


def test_negative_infinity_is_floored_and_flags():
    with collect_diagnostics() as counts:
        result = floored(-math.inf, "test_flag")
    assert result == MIN_DIVISOR
    assert counts["test_flag"] == 1


def test_nan_is_floored_and_flags():
    """A NaN divisor must be floored AND flagged, not passed through."""
    with collect_diagnostics() as counts:
        result = floored(math.nan, "test_flag")
    assert result == MIN_DIVISOR, f"NaN escaped the floor: got {result}"
    assert counts["test_flag"] == 1, "NaN was floored but not flagged"


def test_nan_does_not_poison_a_full_run():
    """End-to-end: a valuation that produces a NaN divisor must not turn the
    whole result into NaN. It must floor, count, and warn."""
    import numpy as np

    from valdist.core import Marginal, Model
    from valdist.core.factors import FactorModel

    def valuation(v: dict) -> float:
        # A NaN divisor reaching floored() - e.g. from an upstream 0/0.
        return 100.0 / floored(math.nan, "nan_divisor_floored")

    model = Model(
        [Marginal("a", p10=1.0, p50=2.0, p90=3.0, family="normal")],
        FactorModel([], {}),
        valuation=valuation,
    )
    result = model.run(n=500, price=1.0, seed=0)

    assert np.isfinite(result.value).all(), "NaN poisoned the value array"
    assert result.diagnostics["nan_divisor_floored"] == 500
    assert any("nan_divisor_floored" in w for w in result.warnings())


def test_floored_is_a_no_op_outside_a_collector():
    """floored() must not raise when called with no collect_diagnostics()
    bound - adapters are ordinary callables usable outside Model.run()."""
    result = floored(-0.01, "test_flag")
    assert result == MIN_DIVISOR
