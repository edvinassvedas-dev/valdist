from __future__ import annotations

import numpy as np
import pytest
from conftest import VICI_SPEC_PATH

# --------------------------------------------------------------------------- #
# Anchor #4: AFFO-DCF FCFE vs legacy bridge formula at zero leverage
# --------------------------------------------------------------------------- #


def test_affo_dcf_fcfe_equivalence():
    """v2 AFFO-DCF == legacy bridge at zero leverage (float precision)."""
    from valdist.adapters.reit import _affo_dcf_equity

    # VICI base parameters (values in decimal, not percent)
    affo, shares, years = 2680.0, 1090.0, 10
    growth, coe, terminal = 0.035, 0.09, 0.02

    v2 = _affo_dcf_equity(affo, shares, years, growth, coe, terminal)

    # Legacy bridge at zero leverage: (EV - 0 + 0) / shares
    projected = [affo * (1 + growth) ** i for i in range(1, years + 1)]
    discounted = [v / (1 + coe) ** i for i, v in enumerate(projected, 1)]
    tv = projected[-1] * (1 + terminal) / (coe - terminal)
    dtv = tv / (1 + coe) ** years
    legacy_zero_leverage = (sum(discounted) + dtv) / shares

    assert abs(v2 - legacy_zero_leverage) < 1e-10, (
        f"v2={v2:.8f}  legacy={legacy_zero_leverage:.8f}  "
        f"delta={abs(v2 - legacy_zero_leverage):.2e}"
    )


def test_affo_dcf_terminal_spread_floor():
    """_affo_dcf_equity clamps spread to 1e-4 so coe~=terminal doesn't blow up."""
    from valdist.adapters.reit import _affo_dcf_equity

    # Near-zero spread: coe ≈ terminal - should not blow up
    result = _affo_dcf_equity(1000.0, 100.0, 5, 0.02, 0.021, 0.02)
    assert np.isfinite(result), f"got {result}"
    assert result > 0


# --------------------------------------------------------------------------- #
# _ddm_two_stage missing the spread floor its siblings have
# --------------------------------------------------------------------------- #


def test_ddm_two_stage_terminal_spread_floor():
    """_ddm_two_stage must clamp (rate - terminal) to 1e-4, same as
    _affo_dcf_equity / _dcf_equity - cost_of_equity and div_terminal are
    sampled marginals, so overlapping bands can draw rate <= terminal.
    """
    from valdist.adapters.reit import _ddm_two_stage

    result = _ddm_two_stage(dps=2.0, growth=0.02, years=5, terminal=0.021, rate=0.02)
    assert np.isfinite(result), f"got {result}"
    assert result > 0


def test_ddm_two_stage_zero_spread_no_zerodivision():
    """rate == terminal must not raise ZeroDivisionError."""
    from valdist.adapters.reit import _ddm_two_stage

    result = _ddm_two_stage(dps=2.0, growth=0.02, years=5, terminal=0.02, rate=0.02)
    assert np.isfinite(result)
    assert result > 0


# --------------------------------------------------------------------------- #
# The discount RATE must be floored too, not just the terminal spread.
#
# reit_v2 is the worst case for this: unlike equity_v2 it has no EPV leg whose
# own `epv_wacc_floored` would incidentally fire on the same draw, so a negative
# cost_of_equity is COMPLETELY silent. Verified pre-fix:
#   reit_v2(cost_of_equity=-0.5%, terminals=-1%) -> 532.69/share   flags: NONE
#   reit_v2(sane: coe=9%, terminals=2%)          ->  31.03/share   flags: NONE
# a 17x inflation, reported as if it were computed - the worst failure mode.
# --------------------------------------------------------------------------- #


def test_ddm_negative_rate_with_negative_terminal_is_floored_and_flagged():
    """Negative rate, positive spread -> the spread floor cannot see it."""
    from valdist.adapters.reit import _ddm_two_stage
    from valdist.core.diagnostics import collect_diagnostics

    with collect_diagnostics() as counts:
        v = _ddm_two_stage(dps=1.8, growth=0.02, years=5, terminal=-0.01, rate=-0.005)

    assert np.isfinite(v)
    assert counts["ddm_rate_floored"] == 1, f"negative rate unflagged; counts={dict(counts)}"


def test_affo_negative_coe_with_negative_terminal_is_floored_and_flagged():
    from valdist.adapters.reit import _affo_dcf_equity
    from valdist.core.diagnostics import collect_diagnostics

    with collect_diagnostics() as counts:
        v = _affo_dcf_equity(
            affo=2680.0,
            shares=1090.0,
            years=10,
            growth=0.02,
            coe=-0.005,
            terminal=-0.01,
        )

    assert np.isfinite(v)
    assert counts["affo_coe_floored"] == 1, f"negative coe unflagged; counts={dict(counts)}"


def test_reit_v2_negative_cost_of_equity_is_not_silent():
    """End-to-end: the exact repro. A negative cost_of_equity with terminals
    below it must no longer produce a 17x-inflated number with zero flags."""
    from valdist.adapters.reit import reit_v2
    from valdist.core.diagnostics import collect_diagnostics

    sample = dict(
        shares=1090.0,
        dps=1.8,
        ddm_stage1_years=5,
        affo=2680.0,
        affo_years=10,
        noi=3300.0,
        nav_debt=17100.0,
        nav_other=420.0,
        w_ddm=50.0,
        w_affo=50.0,
        w_nav=0.0,
        cost_of_equity=-0.5,
        div_growth=2.0,
        div_terminal=-1.0,
        affo_growth=2.0,
        affo_terminal=-1.0,
        cap_rate=6.5,
    )
    with collect_diagnostics() as counts:
        v = reit_v2(sample)

    assert np.isfinite(v)
    assert counts, "reit_v2 produced a 17x-inflated value with NO diagnostics at all"
    assert counts["ddm_rate_floored"] == 1
    assert counts["affo_coe_floored"] == 1


def test_sane_reit_rates_are_never_floored():
    """No false positives - the golden spec's rates must not trip the new floors."""
    from valdist.adapters.reit import _affo_dcf_equity, _ddm_two_stage
    from valdist.core.diagnostics import collect_diagnostics

    with collect_diagnostics() as counts:
        _ddm_two_stage(dps=1.8, growth=0.035, years=5, terminal=0.02, rate=0.09)
        _affo_dcf_equity(
            affo=2680.0, shares=1090.0, years=10, growth=0.035, coe=0.09, terminal=0.02
        )

    assert "ddm_rate_floored" not in counts
    assert "affo_coe_floored" not in counts


# --------------------------------------------------------------------------- #
# Anchor #5: VICI end-to-end regression
# --------------------------------------------------------------------------- #


def test_vici_base_blend():
    """Deterministic base blend (all drivers at p50) reproduces 33.46."""
    from valdist.adapters.reit import reit_v2

    # Build sample dict with p50 driver values (in %) and VICI constants
    v = {
        # drivers at p50
        "div_growth": 3.5,
        "div_terminal": 2.0,
        "cost_of_equity": 9.0,
        "affo_growth": 3.5,
        "affo_terminal": 2.0,
        "cap_rate": 6.5,
        # constants from vici.yaml
        "shares": 1090.0,
        "dps": 1.80,
        "ddm_stage1_years": 5.0,
        "affo": 2680.0,
        "affo_years": 10.0,
        "noi": 3300.0,
        "nav_debt": 17100.0,
        "nav_other": 420.0,
        "w_ddm": 15.0,
        "w_affo": 35.0,
        "w_nav": 50.0,
    }
    blend = reit_v2(v)
    assert abs(blend - 33.46) < 0.05, f"base blend = {blend:.4f}, expected ~33.46"


def test_vici_regression():
    """VICI spec at seed=0, n=50000 reproduces the golden reference."""
    from valdist.spec.loader import hydrate, load_spec
    from valdist.spec.validate import validate

    spec = load_spec(VICI_SPEC_PATH)
    errors = validate(spec)
    assert errors == [], f"VICI spec has validation errors: {errors}"

    model = hydrate(spec)
    result = model.run(n=spec.n, price=spec.price, seed=spec.seed)

    s = result.summary()
    p10 = s["value_p10"]
    p50 = s["value_p50"]
    p90 = s["value_p90"]

    tol = 1e-4
    assert abs(p10 - 28.009594) < tol, f"P10={p10:.6f}, expected 28.009594"
    assert abs(p50 - 33.695849) < tol, f"P50={p50:.6f}, expected 33.695849"
    assert abs(p90 - 40.997838) < tol, f"P90={p90:.6f}, expected 40.997838"

    pu = result.p_undervalued
    assert abs(pu - 0.931240) < 1e-5, f"p_undervalued={pu:.6f}, expected 0.931240"

    # Sanity: standard error is reported and small at n=50000
    assert result.p_undervalued_stderr < 0.003


def test_ddm_two_stage_rejects_nonpositive_years():
    """years=0 does not crash _ddm_two_stage - it silently computes a
    DIFFERENT formula. The stage-1 loop never runs, so the terminal value is
    built off dps (the t=0 dividend) instead of the year-1 dividend.
    Verified: 26.2286 at years=0 vs 26.6143 at years=1 on the same inputs.
    """
    from valdist.adapters.reit import _ddm_two_stage

    with pytest.raises(ValueError, match="years"):
        _ddm_two_stage(dps=1.8, growth=0.035, years=0, terminal=0.02, rate=0.09)
    with pytest.raises(ValueError, match="years"):
        _ddm_two_stage(dps=1.8, growth=0.035, years=-3, terminal=0.02, rate=0.09)


def test_affo_dcf_equity_rejects_nonpositive_years():
    """years=0 makes `projected` empty, so `projected[-1]` raises a bare
    IndexError from deep inside the adapter. Reject with a clear message."""
    from valdist.adapters.reit import _affo_dcf_equity

    with pytest.raises(ValueError, match="years"):
        _affo_dcf_equity(affo=2680.0, shares=1090.0, years=0, growth=0.035, coe=0.09, terminal=0.02)


def test_reit_v2_zero_total_weight_raises_value_error():
    """w_ddm/w_affo/w_nav all zero must raise a clear ValueError, not a bare
    ZeroDivisionError - the validator normally guarantees weights sum to 100,
    but reit_v2 is a public callable a caller can invoke directly (as these
    tests do) without going through validate()."""
    from valdist.adapters.reit import reit_v2

    v = {
        "div_growth": 3.5,
        "div_terminal": 2.0,
        "cost_of_equity": 9.0,
        "affo_growth": 3.5,
        "affo_terminal": 2.0,
        "cap_rate": 6.5,
        "shares": 1090.0,
        "dps": 1.80,
        "ddm_stage1_years": 5.0,
        "affo": 2680.0,
        "affo_years": 10.0,
        "noi": 3300.0,
        "nav_debt": 17100.0,
        "nav_other": 420.0,
        "w_ddm": 0.0,
        "w_affo": 0.0,
        "w_nav": 0.0,
    }
    with pytest.raises(ValueError, match="weight"):
        reit_v2(v)
