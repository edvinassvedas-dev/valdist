"""Preferred-share valuation: a fixed-coupon perpetual with an issuer call option."""

from __future__ import annotations

from valdist.adapters.floors import clamped, floored, require_nonnegative_years
from valdist.adapters.registry import register


def _preferred_value(
    par: float,
    coupon_rate: float,
    years_to_call: int,
    arrears: float,
    required_yield: float,
    call_probability: float,
) -> float:
    """Value one preferred share, probability-weighting call against perpetuity."""
    years = require_nonnegative_years(years_to_call, "years_to_call", "preferred_v1")
    # Floor the rate before dividing by it. Unlike the DCF adapters there's no
    # spread to floor instead - the perpetuity divides by the yield itself, so
    # this is the only guard between a non-positive draw and a ~10,000x
    # inflated terminal.
    y = floored(required_yield, "preferred_required_yield_floored")
    p = clamped(call_probability, 0.0, 100.0, "preferred_call_probability_clamped") / 100.0

    annual_div = par * coupon_rate
    pv_divs = sum(annual_div / (1.0 + y) ** t for t in range(1, years + 1))
    perp_at_call = annual_div / y
    terminal = p * par + (1.0 - p) * perp_at_call
    return pv_divs + terminal / (1.0 + y) ** years + arrears


@register(
    "preferred_v1",
    requires={
        # constants
        "par",
        "coupon_rate",
        "years_to_call",
        "arrears",
        # drivers (sampled)
        "required_yield",
        "call_probability",
    },
    # `years_to_call` is deliberately NOT declared as a positive-years input:
    # that declaration means ">= 1", and zero is a legal, common spec here (a
    # seasoned preferred whose call protection has expired is callable today).
    # require_nonnegative_years guards it in the adapter instead. See floors.py
    # for why the two guards are siblings rather than one loosened rule.
    #
    # No blend_weights either: this is a single method, not a weighted blend of
    # several, so there is nothing to sum to 100 and declaring an empty set
    # would assert a constraint that does not exist.
)
def preferred_v1(v: dict) -> float:
    """Fixed-coupon callable preferred: perpetuity blended against call-at-par."""
    # --- constants -----------------------------------------------------------
    par = v["par"]
    years_to_call = int(v["years_to_call"])
    arrears = v["arrears"]

    # --- percent to decimal --------------------------------------------------
    coupon_rate = v["coupon_rate"] / 100.0
    required_yield = v["required_yield"] / 100.0

    # `call_probability` stays in percent: _preferred_value clamps it against
    # the 0-100 bounds the analyst actually typed, so the flag fires on the
    # number in the spec rather than on a rescaled one.
    return _preferred_value(
        par=par,
        coupon_rate=coupon_rate,
        years_to_call=years_to_call,
        arrears=arrears,
        required_yield=required_yield,
        call_probability=v["call_probability"],
    )
