"""Preferred adapter (preferred_v1) correctness."""

from __future__ import annotations

import numpy as np
import pytest

HORIZONS = (0, 1, 2, 5, 7, 10, 30)


# --------------------------------------------------------------------------- #
# Anchor 21, identity 1: the perpetuity limit at call_probability = 0.
# --------------------------------------------------------------------------- #


def test_perpetuity_limit_when_never_called():
    """At p_call = 0 the value is the plain perpetuity, for any horizon."""
    from valdist.adapters.preferred import _preferred_value

    par, coupon, y = 25.0, 0.06, 0.072
    annual_div = par * coupon
    expected = annual_div / y

    for years in HORIZONS:
        v = _preferred_value(
            par=par,
            coupon_rate=coupon,
            years_to_call=years,
            arrears=0.0,
            required_yield=y,
            call_probability=0.0,
        )
        assert abs(v - expected) < 1e-10, (
            f"years={years}: value={v:.12f} perpetuity={expected:.12f} "
            f"delta={abs(v - expected):.2e}"
        )


def test_perpetuity_limit_holds_at_several_yields():
    """The identity is in the algebra, not in one lucky yield."""
    from valdist.adapters.preferred import _preferred_value

    par, coupon = 25.0, 0.06
    for y in (0.04, 0.06, 0.072, 0.095, 0.15):
        expected = par * coupon / y
        for years in (0, 3, 12):
            v = _preferred_value(par, coupon, years, 0.0, y, 0.0)
            assert abs(v - expected) < 1e-10, f"y={y} years={years}: {v} != {expected}"


# --------------------------------------------------------------------------- #
# Anchor 21, identity 2: the par identity at call_probability = 100.
# --------------------------------------------------------------------------- #


def test_par_identity_when_certainly_called_at_coupon_yield():
    """At p_call = 100 and required_yield == coupon_rate, value == par exactly."""
    from valdist.adapters.preferred import _preferred_value

    par, coupon = 25.0, 0.06
    for years in HORIZONS:
        v = _preferred_value(par, coupon, years, 0.0, coupon, 100.0)
        assert abs(v - par) < 1e-10, f"years={years}: value={v:.12f} par={par} "


def test_certainly_called_is_a_bullet_bond():
    """At p_call = 100 the value is the textbook bullet price, any y and T."""
    from valdist.adapters.preferred import _preferred_value

    par, coupon = 25.0, 0.06
    annual_div = par * coupon
    for y in (0.04, 0.06, 0.09):
        for years in (1, 5, 10):
            expected = sum(annual_div / (1 + y) ** t for t in range(1, years + 1))
            expected += par / (1 + y) ** years
            v = _preferred_value(par, coupon, years, 0.0, y, 100.0)
            assert abs(v - expected) < 1e-10, f"y={y} T={years}: {v} != {expected}"


def test_discount_to_par_iff_yield_above_coupon_when_called():
    """Sign check on identity 2: above the coupon the bullet trades below par."""
    from valdist.adapters.preferred import _preferred_value

    par, coupon = 25.0, 0.06
    assert _preferred_value(par, coupon, 5, 0.0, 0.09, 100.0) < par
    assert _preferred_value(par, coupon, 5, 0.0, 0.03, 100.0) > par


# --------------------------------------------------------------------------- #
# The call blend itself: p interpolates between the two pinned limits.
# --------------------------------------------------------------------------- #


def test_call_probability_interpolates_linearly_between_the_limits():
    """The terminal is p*par + (1-p)*perpetuity, so value is linear in p."""
    from valdist.adapters.preferred import _preferred_value

    par, coupon, y, years = 25.0, 0.06, 0.072, 5
    lo = _preferred_value(par, coupon, years, 0.0, y, 0.0)
    hi = _preferred_value(par, coupon, years, 0.0, y, 100.0)
    mid = _preferred_value(par, coupon, years, 0.0, y, 50.0)
    assert abs(mid - 0.5 * (lo + hi)) < 1e-10


def test_value_is_decreasing_in_required_yield():
    """No-arbitrage sanity: a higher required yield is a lower price."""
    from valdist.adapters.preferred import _preferred_value

    prev = float("inf")
    for y in (0.04, 0.05, 0.06, 0.07, 0.08, 0.12, 0.20):
        v = _preferred_value(25.0, 0.06, 5, 0.0, y, 30.0)
        assert v < prev, f"value rose at y={y}"
        prev = v


def test_arrears_add_straight_through():
    """Cumulative arrears are owed in full, not discounted at the credit yield."""
    from valdist.adapters.preferred import _preferred_value

    base = _preferred_value(25.0, 0.06, 5, 0.0, 0.072, 30.0)
    with_arrears = _preferred_value(25.0, 0.06, 5, 1.875, 0.072, 30.0)
    assert abs((with_arrears - base) - 1.875) < 1e-12


# --------------------------------------------------------------------------- #
# Floors and clamps are counted and surfaced, never silent.
# --------------------------------------------------------------------------- #


def test_nonpositive_required_yield_is_floored_and_flagged():
    """y <= 0 makes the perpetuity infinite or negative – incoherent, not extreme."""
    from valdist.adapters.preferred import _preferred_value
    from valdist.core.diagnostics import collect_diagnostics

    for y in (0.0, -0.02):
        with collect_diagnostics() as counts:
            v = _preferred_value(25.0, 0.06, 5, 0.0, y, 0.0)
        assert np.isfinite(v), f"y={y} produced {v}"
        assert counts["preferred_required_yield_floored"] == 1, (
            f"y={y} was not flagged; counts={dict(counts)}"
        )


def test_a_sane_yield_floors_nothing():
    """No false positives: an ordinary draw must leave the counter empty."""
    from valdist.adapters.preferred import _preferred_value
    from valdist.core.diagnostics import collect_diagnostics

    with collect_diagnostics() as counts:
        _preferred_value(25.0, 0.06, 5, 0.0, 0.072, 30.0)
    assert dict(counts) == {}


def test_call_probability_outside_the_unit_interval_is_clamped_and_flagged():
    """`call_probability` is a probability, and `normal` can draw outside [0,100]."""
    from valdist.adapters.preferred import _preferred_value
    from valdist.core.diagnostics import collect_diagnostics

    at_zero = _preferred_value(25.0, 0.06, 5, 0.0, 0.072, 0.0)
    at_full = _preferred_value(25.0, 0.06, 5, 0.0, 0.072, 100.0)

    with collect_diagnostics() as low:
        v_low = _preferred_value(25.0, 0.06, 5, 0.0, 0.072, -30.0)
    with collect_diagnostics() as high:
        v_high = _preferred_value(25.0, 0.06, 5, 0.0, 0.072, 160.0)

    assert abs(v_low - at_zero) < 1e-12, "a negative probability must clamp to 0"
    assert abs(v_high - at_full) < 1e-12, "a probability above 100 must clamp to 1"
    assert low["preferred_call_probability_clamped"] == 1, dict(low)
    assert high["preferred_call_probability_clamped"] == 1, dict(high)


def test_probability_at_the_bounds_does_not_flag():
    """Exactly 0 and exactly 100 are legal, not clamped – no false positives."""
    from valdist.adapters.preferred import _preferred_value
    from valdist.core.diagnostics import collect_diagnostics

    for p in (0.0, 100.0):
        with collect_diagnostics() as counts:
            _preferred_value(25.0, 0.06, 5, 0.0, 0.072, p)
        assert "preferred_call_probability_clamped" not in counts, f"p={p}"


def test_nan_inputs_are_floored_and_clamped_not_propagated():
    """A NaN must not silently poison every downstream statistic (floors.py)."""
    from valdist.adapters.preferred import _preferred_value
    from valdist.core.diagnostics import collect_diagnostics

    with collect_diagnostics() as counts:
        v = _preferred_value(25.0, 0.06, 5, 0.0, float("nan"), float("nan"))
    assert np.isfinite(v), f"NaN propagated into the value: {v}"
    assert counts["preferred_required_yield_floored"] == 1, dict(counts)
    assert counts["preferred_call_probability_clamped"] == 1, dict(counts)

    # NaN must take the same branch floored() takes: the lower bound. Compared
    # against an explicit 0.0 with the same NaN yield, so the floored-yield leg
    # is identical between the two and only the clamp direction is under test.
    at_lo = _preferred_value(25.0, 0.06, 5, 0.0, float("nan"), 0.0)
    assert v == at_lo, (
        f"NaN call_probability clamped to the wrong bound: got {v}, expected "
        f"{at_lo} (the value at p=0). NaN must clamp to lo, as floored() floors it."
    )


# --------------------------------------------------------------------------- #
# years_to_call: 0 is meaningful here, unlike a projection horizon.
# --------------------------------------------------------------------------- #


def test_zero_years_to_call_is_legal_and_is_the_terminal_alone():
    """A seasoned preferred is already callable; T=0 is a real spec, not an error."""
    from valdist.adapters.preferred import _preferred_value

    par, coupon, y, p = 25.0, 0.06, 0.072, 0.4
    expected = p * par + (1 - p) * (par * coupon / y)
    v = _preferred_value(par, coupon, 0, 0.0, y, 40.0)
    assert abs(v - expected) < 1e-12


def test_negative_years_to_call_raises_loudly():
    """Negative is nonsense and there is no distribution to keep alive."""
    from valdist.adapters.preferred import _preferred_value

    with pytest.raises(ValueError, match="years_to_call"):
        _preferred_value(25.0, 0.06, -1, 0.0, 0.072, 40.0)


def test_require_nonnegative_years_accepts_zero_and_rejects_negative():
    """The new floors.py guard, tested directly – it is a sibling of
    require_positive_years, not a loosening of it."""
    from valdist.adapters.floors import require_nonnegative_years, require_positive_years

    assert require_nonnegative_years(0, "years_to_call", "preferred_v1") == 0
    assert require_nonnegative_years(7, "years_to_call", "preferred_v1") == 7
    with pytest.raises(ValueError, match="years_to_call"):
        require_nonnegative_years(-1, "years_to_call", "preferred_v1")

    # The original guard is unchanged: 0 is still rejected there.
    with pytest.raises(ValueError, match="affo_years"):
        require_positive_years(0, "affo_years", "reit_v2")


# --------------------------------------------------------------------------- #
# The registered adapter: percent convention and registry declarations.
# --------------------------------------------------------------------------- #


def test_preferred_v1_converts_percent_inputs():
    """Drivers arrive in percent, same convention as reit_v2/equity_v2."""
    from valdist.adapters.preferred import _preferred_value, preferred_v1

    v = {
        "par": 25.0,
        "coupon_rate": 6.0,
        "years_to_call": 5,
        "arrears": 0.0,
        "required_yield": 7.2,
        "call_probability": 30.0,
    }
    expected = _preferred_value(25.0, 0.06, 5, 0.0, 0.072, 30.0)
    assert abs(preferred_v1(v) - expected) < 1e-12


def test_preferred_v1_is_registered_with_its_declarations():
    """The registry channel is how validate() stays domain-agnostic."""
    import valdist  # noqa: F401  – import side effect registers the adapters
    from valdist.adapters.registry import (
        blend_weight_inputs,
        get_valuation,
        is_registered,
        positive_year_inputs,
        required_inputs,
    )

    assert is_registered("preferred_v1")
    assert get_valuation("preferred_v1") is not None
    assert required_inputs("preferred_v1") == frozenset(
        {"par", "coupon_rate", "years_to_call", "arrears", "required_yield", "call_probability"}
    )
    # `years_to_call` admits 0, so it must NOT be declared as a positive-years
    # input – that declaration means ">= 1" and would reject a legal spec.
    assert positive_year_inputs("preferred_v1") is None
    # A single method, so there is nothing to blend and nothing to declare.
    assert blend_weight_inputs("preferred_v1") is None


def test_the_implied_yield_of_a_perpetuity_is_coupon_over_price():
    """The implied-expectations test this adapter exists to make possible."""
    from valdist.adapters.preferred import _preferred_value

    par, coupon, price = 25.0, 0.06, 20.86
    implied = par * coupon / price
    assert abs(implied - 0.0719079) < 1e-6
    v = _preferred_value(par, coupon, 5, 0.0, implied, 0.0)
    assert abs(v - price) < 1e-10, "the perpetuity must round-trip to the price"
