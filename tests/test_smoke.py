"""Phase-0 smoke test: package imports, marginals, and a trivial Model run."""

import warnings

import numpy as np
import pytest
from scipy import stats

from valdist.core import Marginal, Model
from valdist.core.factors import FactorModel


def test_lognormal_marginal_quantiles() -> None:
    m = Marginal("cap_rate", p10=5.0, p50=8.0, p90=12.0, family="lognormal")
    vals = m.ppf(np.array([0.1, 0.5, 0.9]))
    assert np.isfinite(vals).all()
    assert vals[0] < vals[1] < vals[2]
    assert vals[0] > 0  # lognormal is strictly positive
    # Median is exact by construction (exp(mu) = p50).
    # p10/p90 are not exactly reproduced unless p50 == sqrt(p10 * p90);
    # the 2-parameter fit anchors the centre at p50 and the log-span at
    # (ln(p90) - ln(p10)) / (2 * Z90).
    assert abs(vals[1] - 8.0) < 1e-9
    # Use a log-symmetric spec to verify exact p10/p90 reproduction.
    m2 = Marginal("x", p10=4.0, p50=8.0, p90=16.0, family="lognormal")
    v2 = m2.ppf(np.array([0.1, 0.5, 0.9]))
    assert abs(v2[0] - 4.0) < 0.01
    assert abs(v2[1] - 8.0) < 1e-9
    assert abs(v2[2] - 16.0) < 0.01


def test_lognormal3_marginal_exact_quantiles() -> None:
    """lognormal3 (shifted lognormal) reproduces all three quantiles exactly,
    unlike plain lognormal - but only when p50 sits closer (in log-ratio
    terms) to p10 than to p90, i.e. the "usual" right-skew shape. It does
    not rescue the opposite, common real-world pattern (worst case pinned
    near zero, base much closer to best) - see
    test_lognormal3_infeasible_for_worst_near_zero below, which needs a
    warning instead (test_lognormal_asymmetric_quantile_warns).
    """
    m = Marginal("x", p10=2.0, p50=3.0, p90=10.0, family="lognormal3")
    vals = m.ppf(np.array([0.1, 0.5, 0.9]))
    assert np.isfinite(vals).all()
    assert vals[0] < vals[1] < vals[2]
    assert abs(vals[0] - 2.0) < 1e-6
    assert abs(vals[1] - 3.0) < 1e-6
    assert abs(vals[2] - 10.0) < 1e-6

    # Log-symmetric case still works (shift comes out ~0).
    m2 = Marginal("y", p10=4.0, p50=8.0, p90=16.0, family="lognormal3")
    v2 = m2.ppf(np.array([0.1, 0.5, 0.9]))
    assert abs(v2[0] - 4.0) < 1e-6
    assert abs(v2[1] - 8.0) < 1e-6
    assert abs(v2[2] - 16.0) < 1e-6


def test_lognormal3_infeasible_for_worst_near_zero() -> None:
    """The common real pattern (worst pinned near zero, base close to best)
    has no feasible standard shifted-lognormal fit:
    - (0.5, 2.5, 4.5): p50 lands exactly on the *arithmetic* mean of p10/p90,
      so the shift equation has no solution at all (0 == nonzero).
    - (0.5, 6.0, 11.0): a shift exists but exceeds p90, which would require an
      upper-bounded, unbounded-below mirror distribution - a nonsensical shape
      for a growth-rate driver, so this is treated as infeasible rather than
      silently flipped.
    """
    with pytest.raises(ValueError, match="lognormal3"):
        Marginal("style_a", p10=0.5, p50=2.5, p90=4.5, family="lognormal3")
    with pytest.raises(ValueError, match="lognormal3"):
        Marginal("style_b", p10=0.5, p50=6.0, p90=11.0, family="lognormal3")


def test_lognormal_asymmetric_quantile_warns() -> None:
    """Plain lognormal warns when p50 is far from the geometric mean of
    p10/p90 - this pattern silently produces a much-fatter-than-intended tail.
    """
    with pytest.warns(UserWarning, match="realized p10/p90"):
        Marginal("fcf_growth", p10=0.5, p50=6.0, p90=11.0, family="lognormal")

    # A mild, VICI-like mismatch should not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Marginal("cap_rate", p10=5.5, p50=6.5, p90=7.5, family="lognormal")


def test_normal_marginal_quantiles() -> None:
    m = Marginal("rate", p10=1.0, p50=2.0, p90=3.0, family="normal")
    vals = m.ppf(np.array([0.1, 0.5, 0.9]))
    assert abs(vals[0] - 1.0) < 0.01
    assert abs(vals[1] - 2.0) < 0.01
    assert abs(vals[2] - 3.0) < 0.01


def test_normal_asymmetric_quantile_warns() -> None:
    """normal's fit ignores p50's position entirely (mu=p50, sigma from
    p90-p10 alone) - an off-center p50 is silently unreproduced at p10/p90.
    """
    with pytest.warns(UserWarning, match="not the midpoint"):
        Marginal("terminal_growth", p10=1.5, p50=2.5, p90=3.0, family="normal")

    # Exactly centered (VICI-like) should not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Marginal("cost_of_equity", p10=8.25, p50=9.0, p90=9.75, family="normal")


def test_triangular_marginal() -> None:
    m = Marginal("x", p10=0.0, p50=5.0, p90=10.0, family="triangular")
    vals = m.ppf(np.array([0.0, 0.5, 1.0]))
    assert vals[0] == pytest.approx(0.0)
    assert vals[-1] == pytest.approx(10.0)


def test_pert_marginal_monotone() -> None:
    m = Marginal("y", p10=1.0, p50=3.0, p90=9.0, family="pert")
    u = np.linspace(0.01, 0.99, 50)
    vals = m.ppf(u)
    assert np.all(np.diff(vals) > 0)


# (1, 3, 9) is chosen so the PERT shape parameters come out as exact integers:
#   span  = c - a = 8
#   alpha = 1 + 4*(b - a)/span = 1 + 4*2/8 = 2
#   beta  = 1 + 4*(c - b)/span = 1 + 4*6/8 = 4
# so the reference below is hand-derived, not a copy of the implementation.
# a != 0 matters: at a == 0 a `span = c + a` slip is invisible.
_PERT_LO, _PERT_MODE, _PERT_HI = 1.0, 3.0, 9.0


def test_pert_marginal_matches_the_closed_form_beta() -> None:
    """Monotonicity alone is far too weak a check: a wrong-but-still-monotone
    transform sails through it (a mutant changing `span = c - a` to `c + a`
    survived exactly that way). Pin the actual values against Beta(2, 4)
    rescaled onto [1, 9], which is what a PERT with mode 3 is.
    """
    m = Marginal("y", p10=_PERT_LO, p50=_PERT_MODE, p90=_PERT_HI, family="pert")
    u = np.array([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

    expected = _PERT_LO + 8.0 * stats.beta.ppf(u, 2.0, 4.0)
    assert m.ppf(u) == pytest.approx(expected, rel=1e-12)


def test_pert_marginal_mean_is_the_textbook_identity() -> None:
    """A second, implementation-independent reference: the PERT's mean is the
    classic (a + 4b + c)/6 = 3.6667. Recover it from the ppf alone, by
    integrating it over u - E[X] = the integral of the quantile function on
    [0, 1] - so nothing here knows how the ppf is built.
    """
    m = Marginal("y", p10=_PERT_LO, p50=_PERT_MODE, p90=_PERT_HI, family="pert")
    u = np.linspace(0.0, 1.0, 200_001)

    mean = float(np.trapezoid(m.ppf(u), u))
    textbook = (_PERT_LO + 4.0 * _PERT_MODE + _PERT_HI) / 6.0
    assert mean == pytest.approx(textbook, abs=1e-4)


def test_degenerate_triangular_marginal() -> None:
    """p10 == p50 == p90 is schema-legal (p10 <= p50 <= p90); triangular must
    return the point mass, not NaN (pert already handles this via
    _pert_ppf's span == 0 branch - triangular should too).
    """
    m = Marginal("x", p10=5.0, p50=5.0, p90=5.0, family="triangular")
    vals = m.ppf(np.array([0.0, 0.1, 0.5, 0.9, 1.0]))
    assert np.isfinite(vals).all(), f"got {vals}"
    assert np.all(vals == 5.0)


@pytest.mark.parametrize("family", ["normal", "lognormal"])
def test_degenerate_scale_marginal_is_a_point_mass(family: str) -> None:
    """p10 == p50 == p90 gives sigma = 0. `mu + 0 * norm.ppf(0)` is `0 * -inf`,
    which is NaN - so ppf() returned [nan, 5.0, nan] at the endpoints, plus a
    numpy RuntimeWarning. (lognormal is affected too: exp(mu + 0*-inf) = exp(nan).)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a numpy RuntimeWarning fails the test
        m = Marginal("x", p10=5.0, p50=5.0, p90=5.0, family=family)
        vals = m.ppf(np.array([0.0, 0.1, 0.5, 0.9, 1.0]))

    assert np.isfinite(vals).all(), f"{family} degenerate ppf gave {vals}"
    assert np.all(vals == 5.0), f"{family} degenerate ppf is not a point mass: {vals}"


def test_quantile_fit_notice_no_nan_for_nonpositive_lognormal_quantiles() -> None:
    """quantile_fit_notice('lognormal', ...) is called directly on raw spec
    values by check_marginal_fit() - unlike Marginal.__post_init__, it has no
    upstream guard against p10 <= 0. It must not emit a NaN-garbage message
    or raise a numpy RuntimeWarning; that case is already a hard error caught
    elsewhere (validate()'s lognormal_nonpositive_p10 check).
    """
    from valdist.core.marginals import quantile_fit_notice

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning fails the test
        msg = quantile_fit_notice("lognormal", p10=-1.0, p50=2.0, p90=5.0)
    assert msg is None, f"expected no notice for invalid p10, got: {msg!r}"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        msg = quantile_fit_notice("lognormal", p10=0.0, p50=2.0, p90=5.0)
    assert msg is None, f"expected no notice for p10=0, got: {msg!r}"


def test_marginal_mismatch_warning_points_at_caller() -> None:
    """The mismatched-quantile warning must be attributed to the code that
    constructed the Marginal, not to a frame inside marginals.py.
    """
    import inspect

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Read the line number from the live frame rather than re-opening
        # __file__ and grepping for a marker comment: __file__ can be a .pyc,
        # a zip entry, or absent entirely (--pyargs), and none of those can be
        # scanned for source text.
        expected_lineno = inspect.currentframe().f_lineno + 1
        Marginal("x", p10=1.0, p50=5.0, p90=10.0, family="lognormal")

    assert len(caught) == 1
    assert caught[0].lineno == expected_lineno, (
        f"warning attributed to line {caught[0].lineno} in {caught[0].filename}, "
        f"expected the construction site at line {expected_lineno}"
    )


def test_marginal_bad_order_raises() -> None:
    with pytest.raises(ValueError, match="p10 <= p50"):
        Marginal("bad", p10=5.0, p50=3.0, p90=7.0)


def test_marginal_lognormal_nonpositive_raises() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        Marginal("bad", p10=-1.0, p50=2.0, p90=5.0, family="lognormal")


def test_from_scenarios_sorts_worst_base_best_into_p10_p50_p90() -> None:
    """worst/base/best are labels, not an ordering guarantee - from_scenarios
    must sort them regardless of the order the caller happened to state them
    in (e.g. a rate driver where a higher value is worse)."""
    m = Marginal.from_scenarios("rate", worst=12.0, base=9.0, best=6.0, family="normal")
    assert (m.p10, m.p50, m.p90) == (6.0, 9.0, 12.0)


def test_from_scenarios_base_already_median_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        m = Marginal.from_scenarios("x", worst=1.0, base=5.0, best=10.0, family="normal")
    assert (m.p10, m.p50, m.p90) == (1.0, 5.0, 10.0)


def test_from_scenarios_base_not_median_warns_and_still_sorts() -> None:
    """base is a joint 'most likely scenario' state, not necessarily this
    driver's own median once independently sorted - warn, but still build a
    usable marginal from the sorted quantiles.
    """
    with pytest.warns(UserWarning, match="not the median"):
        m = Marginal.from_scenarios("x", worst=10.0, base=0.0, best=5.0, family="normal")
    assert (m.p10, m.p50, m.p90) == (0.0, 5.0, 10.0)


def test_model_independent_smoke() -> None:
    """Full round-trip: two drivers, identity copula, trivial valuation."""
    drivers = [
        Marginal("a", p10=1.0, p50=2.0, p90=3.0, family="normal"),
        Marginal("b", p10=10.0, p50=20.0, p90=30.0, family="lognormal"),
    ]
    model = Model(drivers, FactorModel([], {}), valuation=lambda v: v["a"] + v["b"])
    result = model.run(n=2000, price=22.0, seed=42)

    assert result.n == 2000
    assert 0.0 <= result.p_undervalued <= 1.0
    assert result.p_undervalued_stderr >= 0.0

    tornado = result.tornado()
    assert len(tornado) == 2
    names = [t[0] for t in tornado]
    assert set(names) == {"a", "b"}


def test_model_determinism() -> None:
    """Same seed produces identical output."""
    m = Marginal("x", p10=1.0, p50=5.0, p90=10.0, family="lognormal")
    model = Model([m], FactorModel([], {}), valuation=lambda v: v["x"])
    r1 = model.run(n=500, seed=7)
    r2 = model.run(n=500, seed=7)
    np.testing.assert_array_equal(r1.value, r2.value)


def test_model_constants_passed() -> None:
    """Constants are merged into every valuation call."""
    m = Marginal("growth", p10=0.02, p50=0.04, p90=0.06, family="lognormal")
    model = Model(
        [m],
        FactorModel([], {}),
        valuation=lambda v: v["growth"] * v["multiplier"],
        constants={"multiplier": 100.0},
    )
    result = model.run(n=500, seed=0)
    assert np.all(np.isfinite(result.value))


def test_model_duplicate_names_raises() -> None:
    m1 = Marginal("x", p10=1.0, p50=2.0, p90=3.0, family="normal")
    m2 = Marginal("x", p10=4.0, p50=5.0, p90=6.0, family="normal")
    with pytest.raises(ValueError, match="unique"):
        Model([m1, m2], FactorModel([], {}), valuation=lambda v: 0.0)


def test_model_run_rejects_nonpositive_n() -> None:
    """The schema enforces n > 0, but Model.run() is public API and was
    reachable with n=0 - which died as `IndexError: index -1 is out of
    bounds for axis 0 with size 0` deep inside Result, far from the cause."""
    m = Marginal("a", p10=1.0, p50=2.0, p90=3.0, family="normal")
    model = Model([m], FactorModel([], {}), valuation=lambda v: v["a"])

    with pytest.raises(ValueError, match="n must be"):
        model.run(n=0, price=2.0, seed=0)
    with pytest.raises(ValueError, match="n must be"):
        model.run(n=-10, price=2.0, seed=0)


def test_model_rejects_correlation_source_missing_matrix() -> None:
    """A corr object missing .matrix() must fail at Model construction, not
    silently later whenever something happens to call .matrix()."""

    class DrawZOnly:
        def draw_z(self, names, n, rng, sampling="mc"):
            return np.zeros((n, len(names)))

    m = Marginal("x", p10=1.0, p50=2.0, p90=3.0, family="normal")
    with pytest.raises(TypeError, match="matrix"):
        Model([m], DrawZOnly(), valuation=lambda v: v["x"])


def test_model_rejects_correlation_source_missing_draw_z() -> None:
    """A corr object missing .draw_z() must fail at Model construction, not
    deep inside _sample_inputs() at .run() time."""

    class MatrixOnly:
        def matrix(self, names):
            return np.eye(len(names))

    m = Marginal("x", p10=1.0, p50=2.0, p90=3.0, family="normal")
    with pytest.raises(TypeError, match="draw_z"):
        Model([m], MatrixOnly(), valuation=lambda v: v["x"])
