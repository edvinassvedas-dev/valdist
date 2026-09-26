"""clinical_v1 – §7 anchor 23."""

from __future__ import annotations

import numpy as np

HORIZONS = (0, 1, 2, 3, 5, 10)
RATES = (0.04, 0.10, 0.15, 0.30)

BASE = dict(
    cash=57.4,
    debt=0.0,
    other_liabilities=11.3,
    shares=98.1,
    years_to_approval=3,
    annual_burn=30.0,
    discount_rate=0.15,
    approval_probability=25.0,
    asset_value=800.0,
    raise_amount=0.0,
    raise_price=0.40,
)


def value(**overrides) -> float:
    from valdist.adapters.clinical import _clinical_value

    return _clinical_value(**{**BASE, **overrides})


def annuity(burn: float, r: float, years: int) -> float:
    return burn * (1.0 - (1.0 + r) ** -years) / r


def test_failure_limit_is_net_cash_less_the_burn_annuity():
    net_cash = BASE["cash"] - BASE["debt"] - BASE["other_liabilities"]
    for r in RATES:
        for years in HORIZONS:
            expected = (net_cash - annuity(30.0, r, years)) / BASE["shares"]
            v = value(approval_probability=0.0, discount_rate=r, years_to_approval=years)
            assert abs(v - expected) < 1e-12, f"r={r} T={years}: {v} != {expected}"


# catches the burn leg and the asset leg being discounted over different periods
def test_breakeven_asset_value_returns_net_cash():
    net_cash = BASE["cash"] - BASE["debt"] - BASE["other_liabilities"]
    burn = 30.0
    for r in RATES:
        for years in HORIZONS:
            compounded = burn * ((1.0 + r) ** years - 1.0) / r
            v = value(
                approval_probability=100.0,
                asset_value=compounded,
                annual_burn=burn,
                discount_rate=r,
                years_to_approval=years,
            )
            assert abs(v - net_cash / BASE["shares"]) < 1e-12, f"r={r} T={years}: {v}"


def test_raise_at_intrinsic_value_is_neutral():
    for years in HORIZONS:
        v0 = value(years_to_approval=years, annual_burn=10.0)
        assert v0 > 0  # a raise at a negative price means nothing
        for amount in (10.0, 50.0, 250.0):
            v = value(
                years_to_approval=years, annual_burn=10.0, raise_amount=amount, raise_price=v0
            )
            assert abs(v - v0) < 1e-12, f"T={years} raise={amount}: {v} != {v0}"


def test_raise_below_value_dilutes_and_above_accretes():
    v0 = value()
    assert value(raise_amount=50.0, raise_price=0.5 * v0) < v0
    assert value(raise_amount=50.0, raise_price=2.0 * v0) > v0


def test_raise_adds_proceeds_to_cash_and_shares_to_the_count():
    v = value(raise_amount=40.0, raise_price=0.25)
    no_raise_equity = value() * BASE["shares"]
    expected = (no_raise_equity + 40.0) / (BASE["shares"] + 40.0 / 0.25)
    assert abs(v - expected) < 1e-12


def test_value_is_linear_in_approval_probability():
    lo, hi, mid = (value(approval_probability=p) for p in (0.0, 100.0, 50.0))
    assert abs(mid - 0.5 * (lo + hi)) < 1e-12


def test_zero_horizon_is_legal_and_undiscounted():
    net_cash = BASE["cash"] - BASE["debt"] - BASE["other_liabilities"]
    v = value(years_to_approval=0)
    assert abs(v - (net_cash + 0.25 * 800.0) / BASE["shares"]) < 1e-12


def burn_pv(burn: float, r: float, years: float, readout: float | None = None, p: float = 0.0):
    readout = years if readout is None else readout
    ends = sorted({float(k) for k in range(1, int(np.ceil(years)))} | {years, readout} - {0.0})
    total, start = 0.0, 0.0
    for end in ends:
        total += burn * (end - start) * (1.0 if end <= readout else p) / (1.0 + r) ** end
        start = end
    return total


FRACTIONAL = (0.16, 0.5, 2.5, 6.75)


def test_failure_limit_holds_for_fractional_horizons():
    net_cash = BASE["cash"] - BASE["debt"] - BASE["other_liabilities"]
    for r in RATES:
        for years in FRACTIONAL:
            expected = (net_cash - burn_pv(30.0, r, years)) / BASE["shares"]
            v = value(approval_probability=0.0, discount_rate=r, years_to_approval=years)
            assert abs(v - expected) < 1e-12, f"r={r} T={years}: {v} != {expected}"


def test_breakeven_asset_value_returns_net_cash_for_fractional_horizons():
    net_cash = BASE["cash"] - BASE["debt"] - BASE["other_liabilities"]
    for r in RATES:
        for years in FRACTIONAL:
            compounded = burn_pv(30.0, r, years) * (1.0 + r) ** years
            v = value(
                approval_probability=100.0,
                asset_value=compounded,
                discount_rate=r,
                years_to_approval=years,
            )
            assert abs(v - net_cash / BASE["shares"]) < 1e-12, f"r={r} T={years}: {v}"


def test_value_is_continuous_in_the_horizon():
    for years in (1, 3, 5):
        assert abs(value(years_to_approval=years - 1e-9) - value(years_to_approval=years)) < 1e-6
    lo, mid, hi = (value(approval_probability=0.0, years_to_approval=t) for t in (3, 3.5, 4))
    assert hi < mid < lo


def test_failure_limit_depends_on_the_readout_not_the_approval_date():
    for r in RATES:
        for readout in (0.5, 1, 2.5, 3):
            ref = value(approval_probability=0.0, discount_rate=r, years_to_approval=readout)
            for years in (readout + 0.3, readout + 1, readout + 4.25):
                v = value(
                    approval_probability=0.0,
                    discount_rate=r,
                    years_to_approval=years,
                    years_to_readout=readout,
                )
                assert abs(v - ref) < 1e-12, f"r={r} R={readout} T={years}: {v} != {ref}"


# holds on the payment grid; an off-grid readout adds a payment date at itself
def test_readout_is_irrelevant_when_approval_is_certain():
    for readout in (0, 1, 3, 5):
        assert (
            abs(
                value(approval_probability=100.0, years_to_approval=5, years_to_readout=readout)
                - value(approval_probability=100.0, years_to_approval=5)
            )
            < 1e-12
        )


def test_post_readout_burn_is_weighted_by_the_approval_probability():
    net_cash = BASE["cash"] - BASE["debt"] - BASE["other_liabilities"]
    r, years, readout, p = 0.15, 6.5, 2.5, 0.25
    expected = (
        net_cash - burn_pv(30.0, r, years, readout, p) + p * 800.0 / (1.0 + r) ** years
    ) / BASE["shares"]
    v = value(years_to_approval=years, years_to_readout=readout)
    assert abs(v - expected) < 1e-12
    lo, hi, mid = (
        value(years_to_approval=years, years_to_readout=readout, approval_probability=q)
        for q in (0.0, 100.0, 50.0)
    )
    assert abs(mid - 0.5 * (lo + hi)) < 1e-12


def test_omitted_readout_means_readout_at_approval():
    for years in HORIZONS + FRACTIONAL:
        assert value(years_to_approval=years, years_to_readout=years) == value(
            years_to_approval=years
        )


def test_readout_outside_zero_to_approval_is_set_to_approval_and_flagged():
    from valdist.core.diagnostics import collect_diagnostics

    for bad in (-1.0, 4.0, float("nan")):
        with collect_diagnostics() as counts:
            v = value(years_to_approval=3, years_to_readout=bad)
        assert v == value(years_to_approval=3), bad
        assert counts["clinical_readout_clamped"] == 1, (bad, dict(counts))


# one bad sampled draw must not abort a 50,000-draw run (CLAUDE.md 2.8)
def test_negative_horizon_is_clamped_to_zero_and_flagged():
    from valdist.core.diagnostics import collect_diagnostics

    for bad in (-1.0, -0.2, float("nan")):
        with collect_diagnostics() as counts:
            v = value(years_to_approval=bad)
        assert v == value(years_to_approval=0), bad
        assert counts["clinical_years_to_approval_floored"] == 1, (bad, dict(counts))


def test_nonpositive_discount_rate_is_floored_and_flagged():
    from valdist.core.diagnostics import collect_diagnostics

    for r in (0.0, -0.05):
        with collect_diagnostics() as counts:
            v = value(discount_rate=r)
        assert np.isfinite(v)
        assert counts["clinical_discount_rate_floored"] == 1, dict(counts)


def test_probability_outside_the_unit_interval_is_clamped_and_flagged():
    from valdist.core.diagnostics import collect_diagnostics

    with collect_diagnostics() as low:
        v_low = value(approval_probability=-20.0)
    with collect_diagnostics() as high:
        v_high = value(approval_probability=130.0)
    assert v_low == value(approval_probability=0.0)
    assert v_high == value(approval_probability=100.0)
    assert low["clinical_approval_probability_clamped"] == 1
    assert high["clinical_approval_probability_clamped"] == 1


def test_nonpositive_raise_price_is_floored_and_flagged():
    from valdist.core.diagnostics import collect_diagnostics

    with collect_diagnostics() as counts:
        v = value(raise_amount=20.0, raise_price=0.0)
    assert np.isfinite(v)
    assert counts["clinical_raise_price_floored"] == 1, dict(counts)


def test_nonpositive_share_count_is_floored_and_flagged():
    from valdist.core.diagnostics import collect_diagnostics

    with collect_diagnostics() as counts:
        v = value(shares=0.0)
    assert np.isfinite(v)
    assert counts["clinical_shares_floored"] == 1, dict(counts)


def test_negative_asset_value_is_floored_at_zero_and_flagged():
    # lognormal3 with a negative shift draws below zero (ATYR 2026-09-26: 2.79%).
    from valdist.core.diagnostics import collect_diagnostics

    for bad in (-92.6, -1e-9, float("nan")):
        with collect_diagnostics() as counts:
            v = value(asset_value=bad, approval_probability=100.0)
        assert v == value(asset_value=0.0, approval_probability=100.0), bad
        assert counts["clinical_asset_value_floored"] == 1, (bad, dict(counts))
    with collect_diagnostics() as counts:
        value(asset_value=0.0)
    assert dict(counts) == {}


def test_nan_inputs_are_floored_and_clamped_to_the_lower_bound():
    from valdist.core.diagnostics import collect_diagnostics

    nan = float("nan")
    with collect_diagnostics() as counts:
        v = value(discount_rate=nan, approval_probability=nan)
    assert np.isfinite(v)
    assert counts["clinical_discount_rate_floored"] == 1
    assert counts["clinical_approval_probability_clamped"] == 1
    assert v == value(discount_rate=nan, approval_probability=0.0)


def test_an_ordinary_draw_flags_nothing():
    from valdist.core.diagnostics import collect_diagnostics

    for p in (0.0, 100.0):
        with collect_diagnostics() as counts:
            value(approval_probability=p, raise_amount=30.0)
            value(approval_probability=p, years_to_approval=4.5, years_to_readout=2.25)
            value(approval_probability=p, years_to_approval=0, years_to_readout=0)
        assert dict(counts) == {}


def test_clinical_v1_converts_percent_inputs():
    from valdist.adapters.clinical import clinical_v1

    v = {**BASE, "discount_rate": 15.0}
    assert abs(clinical_v1(v) - value()) < 1e-12


def test_clinical_v1_reads_the_optional_readout_and_a_fractional_horizon():
    from valdist.adapters.clinical import clinical_v1

    v = {**BASE, "discount_rate": 15.0, "years_to_approval": 6.5, "years_to_readout": 4.25}
    expected = value(years_to_approval=6.5, years_to_readout=4.25)
    assert abs(clinical_v1(v) - expected) < 1e-12


def test_clinical_v1_is_registered_with_its_declarations():
    import valdist  # noqa: F401
    from valdist.adapters.registry import (
        blend_weight_inputs,
        positive_year_inputs,
        required_inputs,
    )

    assert required_inputs("clinical_v1") == frozenset(BASE)
    assert positive_year_inputs("clinical_v1") is None
    assert blend_weight_inputs("clinical_v1") is None


def test_a_real_shaped_spec_validates_and_runs_clean():
    import valdist
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import check_marginal_fit

    # post 1:10 split, else raise_price < 1 trips the percent/fraction notice
    constants = {
        **{k: BASE[k] for k in ("cash", "debt", "other_liabilities", "years_to_approval")},
        "shares": 9.81,
    }
    spec = SpecModel.model_validate(
        {
            "schema_version": "1.0",
            "name": "2026-01-01-BIOCO",
            "valuation": "clinical_v1",
            "price": 4.0,
            "seed": 0,
            "n": 5000,
            "factors": ["clinical", "funding"],
            "drivers": {
                "approval_probability": {
                    "marginal": {"family": "normal", "p10": 18, "p50": 25, "p90": 32},
                    "loadings": {"clinical": 0.8, "funding": 0.0},
                },
                "asset_value": {
                    "marginal": {"family": "lognormal", "p10": 400, "p50": 800, "p90": 1600},
                    "loadings": {"clinical": 0.3, "funding": 0.0},
                },
                "annual_burn": {
                    "marginal": {"family": "normal", "p10": 25, "p50": 30, "p90": 35},
                    "loadings": {"clinical": 0.0, "funding": 0.3},
                },
                "discount_rate": {
                    "marginal": {"family": "normal", "p10": 12, "p50": 15, "p90": 18},
                    "loadings": {"clinical": 0.0, "funding": 0.5},
                },
                "raise_amount": {
                    "marginal": {"family": "normal", "p10": 60, "p50": 80, "p90": 100},
                    "loadings": {"clinical": 0.0, "funding": 0.3},
                },
                "raise_price": {
                    "marginal": {"family": "lognormal", "p10": 2.5, "p50": 4.0, "p90": 6.4},
                    "loadings": {"clinical": 0.4, "funding": -0.6},
                },
            },
            "constants": constants,
        }
    )
    assert valdist.validate(spec) == []
    assert check_marginal_fit(spec) == []
    result = valdist.run(spec)
    assert np.isfinite(result.value).all()
    assert result.diagnostics == {}
    assert result.warnings() == []

    sampled = spec.model_copy(deep=True)
    del sampled.constants["years_to_approval"]
    sampled.constants["years_to_readout"] = 2.0
    sampled.drivers["years_to_approval"] = sampled.drivers["annual_burn"].model_copy(
        update={
            "marginal": sampled.drivers["annual_burn"].marginal.model_copy(
                update={"family": "pert", "p10": 2.5, "p50": 3.0, "p90": 4.5}
            )
        }
    )
    assert valdist.validate(sampled) == []
    assert valdist.run(sampled).diagnostics == {}
