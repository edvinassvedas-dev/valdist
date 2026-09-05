"""Equity adapter (equity_v2) correctness anchors."""

from __future__ import annotations

import numpy as np
import pytest


def test_dcf_epv_equivalence_at_zero_growth():
    from valdist.adapters.equity import _dcf_equity, _epv_equity

    fcff = 500.0
    shares = 200.0
    net_debt = 1200.0
    wacc = 0.085

    for years in (5, 7, 10):
        dcf = _dcf_equity(fcff, shares, years, 0.0, wacc, 0.0, net_debt)
        epv = _epv_equity(fcff, wacc, net_debt, shares)
        assert abs(dcf - epv) < 1e-10, (
            f"years={years}: dcf={dcf:.8f} epv={epv:.8f} delta={abs(dcf - epv):.2e}"
        )


def test_dcf_terminal_spread_floor():
    """_dcf_equity clamps spread to 1e-4 so wacc ~= terminal doesn't blow up."""
    from valdist.adapters.equity import _dcf_equity

    result = _dcf_equity(500.0, 200.0, 7, 0.04, 0.041, 0.04, 1200.0)
    assert np.isfinite(result)


def test_equity_v2_base_blend():
    """Deterministic blend wiring: DCF/EPV/relative combine per w_dcf/w_epv/w_relative."""
    from valdist.adapters.equity import _dcf_equity, _epv_equity, equity_v2

    v = {
        "fcf_growth": 4.0,
        "terminal_growth": 2.5,
        "wacc": 8.5,
        "shares": 200.0,
        "debt": 1500.0,
        "cash": 300.0,
        "fcff": 500.0,
        "normalized_earnings": 480.0,
        "years": 7.0,
        "relative_value_per_share": 30.0,
        "w_dcf": 45.0,
        "w_epv": 35.0,
        "w_relative": 20.0,
    }
    net_debt = v["debt"] - v["cash"]
    dcf = _dcf_equity(v["fcff"], v["shares"], 7, 0.04, 0.085, 0.025, net_debt)
    epv = _epv_equity(v["normalized_earnings"], 0.085, net_debt, v["shares"])
    expected = (dcf * 45.0 + epv * 35.0 + 30.0 * 20.0) / 100.0

    blend = equity_v2(v)
    assert abs(blend - expected) < 1e-8, f"blend={blend:.6f}, expected={expected:.6f}"


def test_equity_v2_zero_total_weight_raises_value_error():
    """w_dcf/w_epv/w_relative all zero must raise a clear ValueError, not a
    bare ZeroDivisionError - the validator normally guarantees weights sum to
    100, but equity_v2 is a public callable a caller can invoke directly
    (as these tests do) without going through validate()."""
    from valdist.adapters.equity import equity_v2

    v = {
        "fcf_growth": 4.0,
        "terminal_growth": 2.5,
        "wacc": 8.5,
        "shares": 200.0,
        "debt": 1500.0,
        "cash": 300.0,
        "fcff": 500.0,
        "normalized_earnings": 480.0,
        "years": 7.0,
        "relative_value_per_share": 30.0,
        "w_dcf": 0.0,
        "w_epv": 0.0,
        "w_relative": 0.0,
    }
    with pytest.raises(ValueError, match="weight"):
        equity_v2(v)


# --------------------------------------------------------------------------- #
# The discount rate itself must be floored, not just the terminal spread.
#
# The spread floor `floored(wacc - terminal)` only fires when wacc <= terminal.
# Give a spec a negative terminal-growth band - an ordinary declining-business
# assumption - and a negative wacc draw leaves the spread comfortably positive.
# Nothing floors; nothing flags. And discounting at a negative rate *inflates*
# every cash flow (dividing by 0.995**i), so the draw is not merely extreme, it
# is garbage reported as a result - the stated worst failure mode.
#
# _epv_equity already floors this exact quantity (epv_wacc_floored). floors.py's
# own docstring warned: "the condition is written once: there is no second copy
# to drift out of step with the first." It drifted.
# --------------------------------------------------------------------------- #


def test_dcf_equity_negative_wacc_with_negative_terminal_is_floored_and_flagged():
    """The bug: negative rate, positive spread -> previously silent."""
    from valdist.adapters.equity import _dcf_equity
    from valdist.core.diagnostics import collect_diagnostics

    with collect_diagnostics() as counts:
        v = _dcf_equity(7000.0, 450.0, 10, 0.04, -0.005, -0.01, -2000.0)

    assert np.isfinite(v)
    assert counts["dcf_wacc_floored"] == 1, (
        f"negative wacc was not flagged; counts={dict(counts)}. The spread stayed "
        "positive, so the terminal-spread floor could not catch it."
    )
    # The spread must be derived from the floored rate, not the raw one.
    sane = _dcf_equity(7000.0, 450.0, 10, 0.04, 0.09, -0.01, -2000.0)
    assert v > sane, "sanity: a floored near-zero rate still gives a high value"


def test_dcf_equity_wacc_zero_is_floored_and_flagged():
    """wacc=0 does not hit the terminal-spread floor (0 - terminal is positive
    for a positive terminal growth) but a 0% discount rate is still incoherent:
    it discounts nothing. Floored and flagged like any other bad divisor."""
    from valdist.adapters.equity import _dcf_equity
    from valdist.core.diagnostics import collect_diagnostics

    with collect_diagnostics() as counts:
        result = _dcf_equity(500.0, 200.0, 7, 0.04, 0.0, 0.02, 1200.0)

    assert np.isfinite(result)
    assert counts["dcf_wacc_floored"] == 1


def test_dcf_equity_wacc_negative_one_no_longer_crashes():
    """wacc=-100% used to raise ZeroDivisionError from (1+wacc)**i, and a test
    pinned that crash as documented behavior. One bad draw
    must not abort a 50,000-draw run - so the rate is floored and flagged, and
    the run survives. Upgrading a pin is what pins are for."""
    from valdist.adapters.equity import _dcf_equity
    from valdist.core.diagnostics import collect_diagnostics

    with collect_diagnostics() as counts:
        result = _dcf_equity(500.0, 200.0, 7, 0.04, -1.0, 0.02, 1200.0)

    assert np.isfinite(result)
    assert counts["dcf_wacc_floored"] == 1


def test_sane_wacc_is_never_floored():
    """No false positives: an ordinary discount rate must not trip the new floor."""
    from valdist.adapters.equity import _dcf_equity
    from valdist.core.diagnostics import collect_diagnostics

    with collect_diagnostics() as counts:
        _dcf_equity(500.0, 200.0, 7, 0.04, 0.085, 0.02, 1200.0)

    assert "dcf_wacc_floored" not in counts


def test_dcf_equity_zero_fcff_returns_negative_net_debt_per_share():
    """fcff=0 is not a crash - every projected cash flow is 0, so equity
    value collapses to -net_debt, exactly what you'd expect for a firm that
    generates no cash flow at all."""
    from valdist.adapters.equity import _dcf_equity

    result = _dcf_equity(0.0, 200.0, 7, 0.04, 0.085, 0.02, 1200.0)
    assert result == pytest.approx(-1200.0 / 200.0)


def test_dcf_equity_rejects_nonpositive_years():
    """years=0 makes `projected` empty, so `projected[-1]` raises a bare
    IndexError from deep inside the adapter. Reject with a clear message
    naming the offending input instead."""
    from valdist.adapters.equity import _dcf_equity

    with pytest.raises(ValueError, match="years"):
        _dcf_equity(500.0, 200.0, 0, 0.04, 0.085, 0.02, 1200.0)
    with pytest.raises(ValueError, match="years"):
        _dcf_equity(500.0, 200.0, -3, 0.04, 0.085, 0.02, 1200.0)


def test_dcf_equity_zero_shares_raises_zero_division():
    """shares=0 is nonsensical (no real company has zero shares outstanding)
    and unguarded - pinned as current behavior, not a claim that it's ideal."""
    from valdist.adapters.equity import _dcf_equity

    with pytest.raises(ZeroDivisionError):
        _dcf_equity(500.0, 0.0, 7, 0.04, 0.085, 0.02, 1200.0)


# --------------------------------------------------------------------------- #
# Full spec regression - parity with
# test_reit.py::test_vici_regression, but there is no equity equivalent of
# examples/vici.yaml (a real, user-authored golden spec) to pin numbers
# against, so this checks the full validate()/hydrate()/run() pipeline wires
# together cleanly and produces a sane distribution, not specific figures.
# --------------------------------------------------------------------------- #


def _equity_spec_dict() -> dict:
    return {
        "schema_version": "1.0",
        "name": "equity-regression-test",
        "valuation": "equity_v2",
        "price": 25.0,
        "seed": 0,
        "n": 20_000,
        "factors": ["macro"],
        "drivers": {
            "fcf_growth": {
                "marginal": {"family": "normal", "p10": 2.0, "p50": 4.0, "p90": 6.0},
                "loadings": {"macro": 0.4},
            },
            "terminal_growth": {
                "marginal": {"family": "normal", "p10": 1.0, "p50": 1.5, "p90": 2.0},
                "loadings": {"macro": 0.2},
            },
            "wacc": {
                "marginal": {"family": "normal", "p10": 7.0, "p50": 8.5, "p90": 10.0},
                "loadings": {"macro": -0.3},
            },
        },
        "constants": {
            "shares": 200.0,
            "debt": 1500.0,
            "cash": 300.0,
            "fcff": 500.0,
            "normalized_earnings": 480.0,
            "years": 7,
            "relative_value_per_share": 30.0,
        },
        "weights": {"w_dcf": 45.0, "w_epv": 35.0, "w_relative": 20.0},
    }


def test_equity_v2_full_spec_regression():
    from valdist.spec.loader import hydrate
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    spec = SpecModel.model_validate(_equity_spec_dict())
    errors = validate(spec)
    assert errors == [], f"equity regression spec has validation errors: {errors}"

    model = hydrate(spec)
    result = model.run(n=spec.n, price=spec.price, seed=spec.seed, nu=spec.nu)

    # Bands don't cross by construction, so nothing should floor or warn.
    assert result.diagnostics == {}, f"unexpected floored draws: {result.diagnostics}"
    assert result.warnings() == []

    s = result.summary()
    assert s["value_p10"] <= s["value_p50"] <= s["value_p90"]
    assert 0.0 < result.p_undervalued < 1.0
    assert result.p_undervalued_stderr > 0.0
