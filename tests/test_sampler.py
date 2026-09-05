"""Latin Hypercube Sampling."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from valdist.core.factors import FactorModel
from valdist.core.marginals import Marginal
from valdist.core.model import Model


def test_lhs_normal_helper_shape_and_determinism():
    from valdist.core.sampler import lhs_normal

    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    a = lhs_normal(500, 3, rng1)
    b = lhs_normal(500, 3, rng2)
    assert a.shape == (500, 3)
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)


def test_lhs_normal_stratification():
    """For d=1, each of the n LHS points falls in a distinct 1/n-quantile
    bin of the standard normal - true by construction, not a statistical-
    power check."""
    from valdist.core.sampler import lhs_normal

    n = 200
    rng = np.random.default_rng(1)
    draws = lhs_normal(n, 1, rng)[:, 0]
    u = stats.norm.cdf(draws)
    bins = np.floor(u * n).astype(int)
    bins = np.clip(bins, 0, n - 1)
    assert len(set(bins.tolist())) == n  # one point per stratum


def test_factor_model_draw_z_default_unchanged():
    """sampling="mc" (the default) is byte-identical to pre-LHS behavior."""
    fm = FactorModel(["rate"], {"a": {"rate": 0.5}, "b": {"rate": -0.3}})
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    z_default = fm.draw_z(["a", "b"], 100, rng1)
    z_explicit_mc = fm.draw_z(["a", "b"], 100, rng2, sampling="mc")
    np.testing.assert_array_equal(z_default, z_explicit_mc)


def test_factor_model_draw_z_lhs_deterministic():
    fm = FactorModel(["rate", "fundamentals"], {"a": {"rate": 0.6, "fundamentals": 0.3}})
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    z1 = fm.draw_z(["a"], 300, rng1, sampling="lhs")
    z2 = fm.draw_z(["a"], 300, rng2, sampling="lhs")
    np.testing.assert_array_equal(z1, z2)
    assert np.isfinite(z1).all()


def test_correlated_draw_z_lhs_option():
    """LHS through a CORRELATED factor model (not just a single driver)."""
    fm = FactorModel(["f"], {"a": {"f": 0.7}, "b": {"f": 0.7}})
    rng = np.random.default_rng(3)
    z = fm.draw_z(["a", "b"], 200, rng, sampling="lhs")
    assert z.shape == (200, 2)
    assert np.isfinite(z).all()
    assert np.corrcoef(z[:, 0], z[:, 1])[0, 1] > 0.2  # the shared factor survives LHS


def test_model_run_sampling_lhs_smoke():
    """Model.run(sampling='lhs') runs end-to-end and stays deterministic."""
    drivers = [
        Marginal("a", p10=1.0, p50=2.0, p90=3.0, family="normal"),
        Marginal("b", p10=10.0, p50=20.0, p90=30.0, family="lognormal"),
    ]
    model = Model(drivers, FactorModel([], {}), valuation=lambda v: v["a"] + v["b"])
    r1 = model.run(n=500, seed=5, sampling="lhs")
    r2 = model.run(n=500, seed=5, sampling="lhs")
    np.testing.assert_array_equal(r1.value, r2.value)


def test_model_run_sampling_default_matches_mc():
    """Model.run() with no sampling arg matches sampling='mc' exactly."""
    drivers = [Marginal("x", p10=1.0, p50=5.0, p90=10.0, family="lognormal")]
    model = Model(drivers, FactorModel([], {}), valuation=lambda v: v["x"])
    r_default = model.run(n=500, seed=9)
    r_mc = model.run(n=500, seed=9, sampling="mc")
    np.testing.assert_array_equal(r_default.value, r_mc.value)


def test_vici_regression_unaffected_by_lhs_addition():
    """Sanity: the golden VICI regression still passes - the default sampling
    path must be untouched by adding the LHS option.
    """
    from conftest import VICI_SPEC_PATH

    from valdist.spec.loader import hydrate, load_spec

    spec = load_spec(VICI_SPEC_PATH)
    model = hydrate(spec)
    result = model.run(n=spec.n, price=spec.price, seed=spec.seed)
    s = result.summary()
    assert abs(s["value_p50"] - 33.695849) < 1e-4
    assert abs(result.p_undervalued - 0.931240) < 1e-5


def test_invalid_sampling_raises():
    fm = FactorModel(["rate"], {"a": {"rate": 0.5}})
    with pytest.raises(ValueError, match="sampling"):
        fm.draw_z(["a"], 10, np.random.default_rng(0), sampling="bogus")
