"""Clinical-stage developer: net cash - burn to approval + p * asset, plus a raise."""

from __future__ import annotations

import math

from valdist.adapters.floors import clamped, floored
from valdist.adapters.registry import register
from valdist.core.diagnostics import flag


def _clinical_value(
    cash: float,
    debt: float,
    other_liabilities: float,
    shares: float,
    years_to_approval: float,
    annual_burn: float,
    discount_rate: float,
    approval_probability: float,
    asset_value: float,
    raise_amount: float,
    raise_price: float,
    years_to_readout: float | None = None,
) -> float:
    # asset_value is at the approval date, net of launch costs; no wind-down cost on failure.
    years = floored(years_to_approval, "clinical_years_to_approval_floored", floor=0.0)
    r = floored(discount_rate, "clinical_discount_rate_floored")
    p = clamped(approval_probability, 0.0, 100.0, "clinical_approval_probability_clamped") / 100.0

    # Falls back to T, not 0: clamped() would send a NaN to 0.
    readout = years if years_to_readout is None else years_to_readout
    if not (0.0 <= readout <= years):
        flag("clinical_readout_clamped")
        readout = years

    # Whole years with readout at T reproduce the plain annuity, bit for bit.
    ends = sorted({float(k) for k in range(1, math.ceil(years))} | {years, readout} - {0.0})
    pv_burn = 0.0
    start = 0.0
    for end in ends:
        paid = 1.0 if end <= readout else p
        pv_burn += annual_burn * (end - start) * paid / (1.0 + r) ** end
        start = end
    # An approved asset too small to launch is licensed or shelved, never worth less than nothing.
    asset = floored(asset_value, "clinical_asset_value_floored", floor=0.0)
    pv_asset = p * asset / (1.0 + r) ** years
    equity = cash + raise_amount - debt - other_liabilities - pv_burn + pv_asset

    new_shares = raise_amount / floored(raise_price, "clinical_raise_price_floored")
    return equity / floored(shares + new_shares, "clinical_shares_floored")


@register(
    "clinical_v1",
    requires={
        "cash",
        "debt",
        "other_liabilities",
        "shares",
        "years_to_approval",
        "annual_burn",
        "discount_rate",
        "approval_probability",
        "asset_value",
        "raise_amount",
        "raise_price",
    },
    # not positive_years: years_to_approval = 0 is legal
)
def clinical_v1(v: dict) -> float:
    # approval_probability stays in percent; the clamp works on the typed value
    return _clinical_value(
        cash=v["cash"],
        debt=v["debt"],
        other_liabilities=v["other_liabilities"],
        shares=v["shares"],
        years_to_approval=v["years_to_approval"],
        annual_burn=v["annual_burn"],
        discount_rate=v["discount_rate"] / 100.0,
        approval_probability=v["approval_probability"],
        asset_value=v["asset_value"],
        raise_amount=v["raise_amount"],
        raise_price=v["raise_price"],
        years_to_readout=v.get("years_to_readout"),
    )
