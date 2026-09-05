"""Equity valuation blend: DCF (FCFF equity bridge) + EPV + relative (cross-check)."""

from __future__ import annotations

from valdist.adapters.floors import floored, require_positive_years
from valdist.adapters.registry import register


def _dcf_equity(
    fcff: float,
    shares: float,
    years: int,
    growth: float,
    wacc: float,
    terminal: float,
    net_debt: float,
) -> float:
    """Multi-year FCFF DCF with Gordon-growth terminal value, bridged to equity."""
    years = require_positive_years(years, "years", "equity_v2")
    # Floor the rate before deriving the spread from it, not just the spread.
    # A spread floor only fires when wacc <= terminal; with a negative terminal
    # growth (an ordinary declining-business band), a negative wacc leaves the
    # spread positive. Nothing fired, and the negative discount rate inflated
    # every cash flow - 22x, silently. See floors.py and the tests.
    wacc = floored(wacc, "dcf_wacc_floored")
    spread = floored(wacc - terminal, "dcf_terminal_spread_floored")
    projected = [fcff * (1.0 + growth) ** i for i in range(1, years + 1)]
    discounted = [v / (1.0 + wacc) ** i for i, v in enumerate(projected, 1)]
    tv = projected[-1] * (1.0 + terminal) / spread
    dtv = tv / (1.0 + wacc) ** years
    equity_value = sum(discounted) + dtv - net_debt
    return equity_value / shares


def _epv_equity(
    normalized_earnings: float,
    wacc: float,
    net_debt: float,
    shares: float,
) -> float:
    """Earnings Power Value: zero-growth perpetuity of normalized earnings."""
    equity_value = normalized_earnings / floored(wacc, "epv_wacc_floored") - net_debt
    return equity_value / shares


@register(
    "equity_v2",
    requires={
        # constants
        "shares",
        "debt",
        "cash",
        "fcff",
        "years",
        "normalized_earnings",
        "relative_value_per_share",
        # weights
        "w_dcf",
        "w_epv",
        "w_relative",
        # drivers (sampled)
        "fcf_growth",
        "terminal_growth",
        "wacc",
    },
    positive_years={"years"},
    blend_weights={"w_dcf", "w_epv", "w_relative"},
)
def equity_v2(v: dict) -> float:
    """Equity blend: DCF(FCFF) + EPV + relative-valuation cross-check."""
    # --- constants -----------------------------------------------------------
    shares = v["shares"]
    net_debt = v["debt"] - v["cash"]
    fcff = v["fcff"]
    years = int(v["years"])
    normalized_earnings = v["normalized_earnings"]
    relative_value = v["relative_value_per_share"]
    w_dcf = v["w_dcf"]
    w_epv = v["w_epv"]
    w_relative = v["w_relative"]

    # --- driver values (percent → decimal) -----------------------------------
    growth = v["fcf_growth"] / 100.0
    terminal = v["terminal_growth"] / 100.0
    wacc = v["wacc"] / 100.0

    # --- three valuation methods ---------------------------------------------
    dcf = _dcf_equity(fcff, shares, years, growth, wacc, terminal, net_debt)
    epv = _epv_equity(normalized_earnings, wacc, net_debt, shares)

    # --- weighted blend --------------------------------------------------------
    total_w = w_dcf + w_epv + w_relative
    if total_w <= 0:
        raise ValueError(
            f"equity_v2: weights must sum to a positive number, got "
            f"w_dcf={w_dcf}, w_epv={w_epv}, w_relative={w_relative} (sum={total_w})"
        )
    return (dcf * w_dcf + epv * w_epv + relative_value * w_relative) / total_w
