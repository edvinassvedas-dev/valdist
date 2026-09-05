from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import VICI_FIXTURE_PATH
from scipy import stats

from valdist.adapters.reit import _affo_dcf_equity, _ddm_two_stage, reit_v2
from valdist.calibrate.selftest import ComonotonicModel, ScenarioDriver
from valdist.core import FactorModel, Marginal, Model

# --------------------------------------------------------------------------- #
# VICI valuation functions - the anchor validates the real REIT adapter
# (valdist.adapters.reit), imported above rather than reimplemented, so this
# anchor cannot silently drift from the production formulas it exists to
# check (see test_anchor_uses_real_adapter_functions below).
# --------------------------------------------------------------------------- #


def _load_vici() -> dict:
    with VICI_FIXTURE_PATH.open() as fh:
        return json.load(fh)


def _make_vici_blend(d: dict):
    """Return the VICI valuation callable with NAV pinned at base GAV."""
    # Drivers arrive here as FRACTIONS (see _vici_scenarios); reit_v2's contract
    # is PERCENT and it divides by 100 internally.
    pct = 100.0
    noi, gav = float(d["noi"]), float(d["gav"])
    pinned_cap_rate = pct * noi / gav  # ⇒ reit_v2's gav = noi/cap_rate == base gav

    constants = {
        "shares": float(d["shares"]),
        "dps": float(d["dps"]),
        "ddm_stage1_years": int(d["ddm_stage1_years"]),
        "affo": float(d["affo"]),
        "affo_years": int(d["affo_years"]),
        "noi": noi,
        "nav_debt": float(d["nav_debt"]),
        "nav_other": float(d["nav_other"]),
        "w_ddm": float(d["w_ddm"]),
        "w_affo": float(d["w_affo"]),
        "w_nav": float(d["w_nav"]),
    }

    def blend(v: dict) -> float:
        return reit_v2(
            {
                **constants,
                "cost_of_equity": v["cost_of_equity"] * pct,
                "div_growth": v["div_growth"] * pct,
                "div_terminal": v["div_terminal"] * pct,
                "affo_growth": v["affo_growth"] * pct,
                "affo_terminal": v["affo_terminal"] * pct,
                "cap_rate": pinned_cap_rate,
            }
        )

    return blend


def _pct(d: dict, key: str) -> float:
    return float(d[key]) / 100.0


def _vici_scenarios(d: dict) -> dict[str, tuple[float, float, float]]:
    """Return (worst, base, best) tuples per driver (as fractions)."""
    p = {k: _pct(d, k) for k in d if any(tok in k for tok in ("growth", "terminal", "rate"))}
    return {
        "div_growth": (
            p["ddm_worst_growth"],
            p["ddm_base_growth"],
            p["ddm_best_growth"],
        ),
        "div_terminal": (
            p["ddm_worst_terminal"],
            p["ddm_base_terminal"],
            p["ddm_best_terminal"],
        ),
        "cost_of_equity": (p["ddm_worst_rate"], p["ddm_base_rate"], p["ddm_best_rate"]),
        "affo_growth": (
            p["affo_worst_growth"],
            p["affo_base_growth"],
            p["affo_best_growth"],
        ),
        "affo_terminal": (
            p["affo_worst_terminal"],
            p["affo_base_terminal"],
            p["affo_best_terminal"],
        ),
    }


def _make_oracle() -> ComonotonicModel:
    """Build the ComonotonicModel oracle on VICI data (NAV pinned)."""
    d = _load_vici()
    sc = _vici_scenarios(d)
    drivers = [ScenarioDriver(name, *wbb) for name, wbb in sc.items()]
    return ComonotonicModel(drivers, _make_vici_blend(d))


# --------------------------------------------------------------------------- #
# Anchor #1: comonotonic limit
# --------------------------------------------------------------------------- #


def test_oracle_scenario_blend():
    """ComonotonicModel on VICI (NAV pinned) reproduces documented blends."""
    oracle = _make_oracle()
    sb = oracle.scenario_blend()
    assert abs(sb["worst"] - 29.69) < 0.05, f"worst: {sb['worst']:.4f}"
    assert abs(sb["base"] - 33.46) < 0.05, f"base: {sb['base']:.4f}"
    assert abs(sb["best"] - 39.12) < 0.05, f"best: {sb['best']:.4f}"


def test_comonotonic_limit_deterministic():
    """FactorModel at comonotonic limit reproduces oracle scenario_blend exactly."""
    d = _load_vici()
    blend = _make_vici_blend(d)
    sc = _vici_scenarios(d)

    # Marginals: sorted ascending (p10 ≤ p50 ≤ p90), normal family.
    # cost_of_equity is inverted: p10 = best_rate (lowest rate = good for value).
    # Symmetric normal specs → ppf(0.1) = p10 and ppf(0.9) = p90 exactly.
    def _normal_marginal(name: str, worst: float, base: float, best: float) -> Marginal:
        lo, hi = min(worst, best), max(worst, best)
        return Marginal(name, p10=lo, p50=base, p90=hi, family="normal")

    marginals = [_normal_marginal(name, *wbb) for name, wbb in sc.items()]
    names = [m.name for m in marginals]

    # 1 factor, unit |loadings| with correct value-direction signs, ψ = 0
    fm = FactorModel(
        factors=["market"],
        loadings={
            "div_growth": {"market": 1.0},
            "div_terminal": {"market": 1.0},
            "cost_of_equity": {"market": -1.0},  # rising rate = bad for value
            "affo_growth": {"market": 1.0},
            "affo_terminal": {"market": 1.0},
        },
    )

    # Derive loading vector and evaluate blend at f = -Z90, 0, +Z90
    lam_vec = np.array([fm.loadings[n].get("market", 0.0) for n in names])
    Z90 = stats.norm.ppf(0.90)

    def eval_at_factor(f_val: float) -> float:
        z = lam_vec * f_val
        u = stats.norm.cdf(z)
        drivers = {m.name: float(m.ppf(np.array([ui]))[0]) for m, ui in zip(marginals, u)}
        return blend(drivers)

    fm_worst = eval_at_factor(-Z90)
    fm_base = eval_at_factor(0.0)
    fm_best = eval_at_factor(+Z90)

    oracle_sb = _make_oracle().scenario_blend()

    # Deterministic comparison: must match to float precision (< 1e-8 $ tolerance)
    assert abs(fm_worst - oracle_sb["worst"]) < 1e-8, (
        f"worst: FactorModel={fm_worst:.6f}  oracle={oracle_sb['worst']:.6f}"
    )
    assert abs(fm_base - oracle_sb["base"]) < 1e-8, (
        f"base: FactorModel={fm_base:.6f}  oracle={oracle_sb['base']:.6f}"
    )
    assert abs(fm_best - oracle_sb["best"]) < 1e-8, (
        f"best: FactorModel={fm_best:.6f}  oracle={oracle_sb['best']:.6f}"
    )


def test_comonotonic_limit_sigma():
    """FactorModel(1 factor, |λ|=1, ψ=0) gives all off-diagonal correlations ±1."""
    fm = FactorModel(
        factors=["f"],
        loadings={
            "a": {"f": 1.0},
            "b": {"f": 1.0},
            "c": {"f": -1.0},
        },
    )
    Sigma = fm.matrix(["a", "b", "c"])
    # a–b: both load +1 → corr = +1*+1 = +1
    assert abs(Sigma[0, 1] - 1.0) < 1e-12
    # a–c: +1 * -1 = -1
    assert abs(Sigma[0, 2] - (-1.0)) < 1e-12
    # diagonal = 1
    np.testing.assert_allclose(np.diag(Sigma), 1.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# Anchor #2: independent limit
# --------------------------------------------------------------------------- #


def test_independent_limit_sigma():
    """All λ = 0 ⇒ Σ = I exactly."""
    fm = FactorModel(factors=["f1", "f2"], loadings={})
    Sigma = fm.matrix(["x", "y", "z"])
    np.testing.assert_array_equal(Sigma, np.eye(3))


def test_independent_limit_mc():
    """At the independent limit, driver samples are uncorrelated."""
    rng = np.random.default_rng(0)
    fm = FactorModel(factors=["f"], loadings={})
    n = 100_000
    z = fm.draw_z(["x", "y"], n, rng)

    rho_s = stats.spearmanr(z[:, 0], z[:, 1]).statistic
    assert abs(rho_s) < 0.01, f"Spearman rank-corr = {rho_s:.4f}, expected ~0"

    rho_p = np.corrcoef(z[:, 0], z[:, 1])[0, 1]
    assert abs(rho_p) < 0.01, f"Pearson corr = {rho_p:.4f}, expected ~0"


# --------------------------------------------------------------------------- #
# Anchor #3: PSD by construction
# --------------------------------------------------------------------------- #


def test_psd_by_construction():
    """Σ = ΛΛ' + diag(ψ²) is PSD for any valid loadings, without projection."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        k = rng.integers(1, 4)  # 1–3 factors
        d = rng.integers(2, 8)  # 2–7 drivers
        # Random loadings satisfying Σλ² ≤ 1
        Lambda = rng.standard_normal((d, k))
        row_sumsq = np.sum(Lambda**2, axis=1, keepdims=True)
        max_sq = row_sumsq.max()
        if max_sq > 1.0:
            Lambda /= np.sqrt(max_sq) * 1.01  # scale down
        psi_sq = np.maximum(0.0, 1.0 - np.sum(Lambda**2, axis=1))
        Sigma = Lambda @ Lambda.T + np.diag(psi_sq)
        eigvals = np.linalg.eigvalsh(Sigma)
        assert eigvals.min() >= -1e-10, f"Σ has negative eigenvalue {eigvals.min():.2e} - not PSD"


def test_psd_via_factor_model_matrix():
    """FactorModel.matrix() returns a PSD matrix for valid loadings."""
    fm = FactorModel(
        factors=["rate", "fundamentals"],
        loadings={
            "cost_of_equity": {"rate": 0.85, "fundamentals": -0.10},
            "affo_growth": {"rate": -0.20, "fundamentals": 0.75},
            "cap_rate": {"rate": 0.80, "fundamentals": -0.20},
        },
    )
    Sigma = fm.matrix(["cost_of_equity", "affo_growth", "cap_rate"])
    eigvals = np.linalg.eigvalsh(Sigma)
    assert eigvals.min() >= -1e-10


def test_loadings_exceed_one_raises():
    """FactorModel rejects loadings where Σλ² > 1."""
    with pytest.raises(ValueError, match=r"sum.*squared.*loading|sum\(λ²\)|Σλ²"):
        FactorModel(
            factors=["f"],
            loadings={"bad_driver": {"f": 1.2}},  # 1.2² = 1.44 > 1
        )


def test_unknown_factor_in_loadings_raises():
    """FactorModel rejects loadings that reference an undeclared factor."""
    with pytest.raises(ValueError, match="undeclared factor"):
        FactorModel(
            factors=["rate"],
            loadings={"driver": {"rate": 0.5, "ghost": 0.3}},
        )


# --------------------------------------------------------------------------- #
# Anchor #6: determinism with FactorModel
# --------------------------------------------------------------------------- #


def test_determinism_factor_model():
    """Same spec + seed ⇒ identical percentiles when using FactorModel."""
    fm = FactorModel(
        factors=["f"],
        loadings={"a": {"f": 0.7}, "b": {"f": -0.5}},
    )
    drivers = [
        Marginal("a", p10=1.0, p50=3.0, p90=5.0, family="normal"),
        Marginal("b", p10=0.5, p50=1.0, p90=2.0, family="lognormal"),
    ]
    model = Model(drivers, fm, valuation=lambda v: v["a"] * v["b"])

    r1 = model.run(n=5000, seed=99)
    r2 = model.run(n=5000, seed=99)

    np.testing.assert_array_equal(r1.value, r2.value)
    assert r1.summary() == r2.summary()


def test_different_seeds_differ():
    """Different seeds produce different outputs (basic sanity)."""
    fm = FactorModel(factors=["f"], loadings={"x": {"f": 0.8}})
    m = Marginal("x", p10=1.0, p50=2.0, p90=3.0, family="normal")
    model = Model([m], fm, valuation=lambda v: v["x"])
    r1 = model.run(n=1000, seed=0)
    r2 = model.run(n=1000, seed=1)
    assert not np.array_equal(r1.value, r2.value)


def test_anchor_uses_real_adapter_functions():
    """Anchor 1 must validate the real REIT adapter, not a hand-copied stand-in."""
    from valdist.adapters import reit

    assert _ddm_two_stage is reit._ddm_two_stage, (
        "test_limits._ddm_two_stage has drifted from valdist.adapters.reit._ddm_two_stage"
    )
    assert _affo_dcf_equity is reit._affo_dcf_equity, (
        "test_limits._affo_dcf_equity has drifted from valdist.adapters.reit._affo_dcf_equity"
    )
    assert reit_v2 is reit.reit_v2, "test_limits.reit_v2 has drifted from adapters.reit.reit_v2"


def test_oracle_blend_routes_through_the_real_adapter():
    """The oracle's blend must BE reit_v2 (with NAV pinned), not a copy of it."""
    d = _load_vici()
    blend = _make_vici_blend(d)

    drivers = {  # fractions, as _vici_scenarios produces
        "cost_of_equity": 0.09,
        "div_growth": 0.035,
        "div_terminal": 0.02,
        "affo_growth": 0.035,
        "affo_terminal": 0.02,
    }

    # Independently call reit_v2 with the same pinned cap rate, in PERCENT.
    expected = reit_v2(
        {
            "shares": float(d["shares"]),
            "dps": float(d["dps"]),
            "ddm_stage1_years": int(d["ddm_stage1_years"]),
            "affo": float(d["affo"]),
            "affo_years": int(d["affo_years"]),
            "noi": float(d["noi"]),
            "nav_debt": float(d["nav_debt"]),
            "nav_other": float(d["nav_other"]),
            "w_ddm": float(d["w_ddm"]),
            "w_affo": float(d["w_affo"]),
            "w_nav": float(d["w_nav"]),
            "cost_of_equity": 9.0,
            "div_growth": 3.5,
            "div_terminal": 2.0,
            "affo_growth": 3.5,
            "affo_terminal": 2.0,
            "cap_rate": 100.0 * float(d["noi"]) / float(d["gav"]),
        }
    )
    assert blend(drivers) == pytest.approx(expected, abs=1e-12)


def test_pinned_cap_rate_reproduces_the_fixture_base_nav():
    """The pin is only legitimate if noi/cap_rate really does recover base GAV --
    otherwise the oracle would be validating a DIFFERENT NAV than the v1 model,
    and the golden 29.69/33.46/39.12 numbers would be coincidence."""
    d = _load_vici()
    noi, gav = float(d["noi"]), float(d["gav"])
    pinned_cap_rate_fraction = (100.0 * noi / gav) / 100.0

    assert noi / pinned_cap_rate_fraction == pytest.approx(gav, rel=1e-12)
