"""REIT v2 valuation blend: DDM + AFFO-DCF (FCFE) + NAV-from-cap-rate."""

from __future__ import annotations

from valdist.adapters.floors import floored, require_positive_years
from valdist.adapters.registry import register


def _ddm_two_stage(
    dps: float,
    growth: float,
    years: int,
    terminal: float,
    rate: float,
) -> float:
    """Two-stage dividend discount model."""
    years = require_positive_years(years, "ddm_stage1_years", "reit_v2")
    # Floor the rate first: the spread floor alone can't see a negative
    # discount rate whose terminal growth is even lower (the spread stays
    # positive). reit_v2 has no EPV leg to catch that incidentally, so it used
    # to be fully silent: 532.69/share vs. 31.03, no flags at all. See floors.py.
    rate = floored(rate, "ddm_rate_floored")
    pv, d = 0.0, dps
    for t in range(1, years + 1):
        d *= 1.0 + growth
        pv += d / (1.0 + rate) ** t
    spread = floored(rate - terminal, "ddm_terminal_spread_floored")
    tv = d * (1.0 + terminal) / spread
    return pv + tv / (1.0 + rate) ** years


def _affo_dcf_equity(
    affo: float,
    shares: float,
    years: int,
    growth: float,
    coe: float,
    terminal: float,
) -> float:
    """AFFO-DCF as an equity (FCFE) flow — no enterprise-value bridge."""
    years = require_positive_years(years, "affo_years", "reit_v2")
    coe = floored(coe, "affo_coe_floored")  # floor the rate before deriving the spread
    spread = floored(coe - terminal, "affo_terminal_spread_floored")
    projected = [affo * (1.0 + growth) ** i for i in range(1, years + 1)]
    discounted = [v / (1.0 + coe) ** i for i, v in enumerate(projected, 1)]
    tv = projected[-1] * (1.0 + terminal) / spread
    dtv = tv / (1.0 + coe) ** years
    return (sum(discounted) + dtv) / shares


@register(
    "reit_v2",
    requires={
        # constants
        "shares",
        "dps",
        "ddm_stage1_years",
        "affo",
        "affo_years",
        "noi",
        "nav_debt",
        "nav_other",
        # weights
        "w_ddm",
        "w_affo",
        "w_nav",
        # drivers (sampled)
        "cost_of_equity",
        "div_growth",
        "div_terminal",
        "affo_growth",
        "affo_terminal",
        "cap_rate",
    },
    positive_years={"ddm_stage1_years", "affo_years"},
    blend_weights={"w_ddm", "w_affo", "w_nav"},
)
def reit_v2(v: dict) -> float:
    """Corrected v2 REIT blend: DDM + AFFO-DCF(FCFE) + NAV-from-cap-rate."""
    # --- constants -----------------------------------------------------------
    shares = v["shares"]
    dps = v["dps"]
    ddm_years = int(v["ddm_stage1_years"])
    affo = v["affo"]
    affo_years = int(v["affo_years"])
    noi = v["noi"]
    nav_debt = v["nav_debt"]
    nav_other = v["nav_other"]
    w_ddm = v["w_ddm"]
    w_affo = v["w_affo"]
    w_nav = v["w_nav"]

    # --- driver values (percent → decimal) -----------------------------------
    coe = v["cost_of_equity"] / 100.0
    div_g = v["div_growth"] / 100.0
    div_t = v["div_terminal"] / 100.0
    affo_g = v["affo_growth"] / 100.0
    affo_t = v["affo_terminal"] / 100.0
    cap_r = v["cap_rate"] / 100.0

    # --- three valuation methods ---------------------------------------------
    ddm = _ddm_two_stage(dps, div_g, ddm_years, div_t, coe)
    af = _affo_dcf_equity(affo, shares, affo_years, affo_g, coe, affo_t)

    # NAV is dynamic: cap_rate is sampled, so GAV (and hence NAV) varies per
    # draw. A cap rate at or below zero is incoherent (it implies an infinite
    # or negative GAV), so it's floored and flagged rather than divided
    # through silently.
    gav = noi / floored(cap_r, "nav_cap_rate_floored")
    nav = (gav - nav_debt - nav_other) / shares

    # --- weighted blend ------------------------------------------------------
    total_w = w_ddm + w_affo + w_nav
    if total_w <= 0:
        raise ValueError(
            f"reit_v2: weights must sum to a positive number, got "
            f"w_ddm={w_ddm}, w_affo={w_affo}, w_nav={w_nav} (sum={total_w})"
        )
    return (ddm * w_ddm + af * w_affo + nav * w_nav) / total_w
