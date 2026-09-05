"""nu/sampling param-threading bugs."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

_NU_SPEC_YAML = textwrap.dedent("""
    schema_version: "1.0"
    name: "nu-threading-test"
    valuation: _test_param_threading_passthrough
    price: 10.0
    seed: 0
    n: 20000
    nu: 3.0
    factors: [f]
    drivers:
      a:
        marginal: {family: normal, p10: -3.0, p50: 0.0, p90: 3.0}
        loadings: {f: 0.8}
      b:
        marginal: {family: normal, p10: -3.0, p50: 0.0, p90: 3.0}
        loadings: {f: 0.8}
    constants: {}
    """)


@pytest.fixture(autouse=True)
def _register_passthrough():
    from valdist.adapters.registry import _REGISTRY

    _REGISTRY["_test_param_threading_passthrough"] = lambda v: v["a"] + v["b"]
    yield
    _REGISTRY.pop("_test_param_threading_passthrough", None)


# --------------------------------------------------------------------------- #
# 1. valdist.run() must thread spec.nu
# --------------------------------------------------------------------------- #


def test_top_level_run_threads_spec_nu(tmp_path: Path) -> None:
    """valdist.run(path) must match Model.run(nu=spec.nu), not nu=None."""
    import valdist
    from valdist.spec.loader import hydrate, load_spec

    spec_path = tmp_path / "nu_spec.yaml"
    spec_path.write_text(_NU_SPEC_YAML)

    result = valdist.run(spec_path)

    spec = load_spec(spec_path)
    model = hydrate(spec)
    expected = model.run(n=spec.n, price=spec.price, seed=spec.seed, nu=spec.nu)
    wrong_default = model.run(n=spec.n, price=spec.price, seed=spec.seed, nu=None)

    np.testing.assert_array_equal(result.value, expected.value)
    assert not np.array_equal(result.value, wrong_default.value), (
        "valdist.run() produced the same draws as nu=None - spec.nu is being dropped"
    )


def test_top_level_run_threads_spec_nu_via_specmodel(tmp_path: Path) -> None:
    """Same check when a SpecModel (not a path) is passed to valdist.run()."""
    import valdist
    from valdist.spec.loader import hydrate, load_spec

    spec_path = tmp_path / "nu_spec.yaml"
    spec_path.write_text(_NU_SPEC_YAML)
    spec = load_spec(spec_path)

    result = valdist.run(spec)
    model = hydrate(spec)
    expected = model.run(n=spec.n, price=spec.price, seed=spec.seed, nu=spec.nu)
    np.testing.assert_array_equal(result.value, expected.value)


# --------------------------------------------------------------------------- #
# 2. check_coverage() must thread spec.nu
# --------------------------------------------------------------------------- #


def test_check_coverage_threads_spec_nu(tmp_path: Path) -> None:
    """A coverage backtest over a nu-bearing spec must use that nu, not
    silently downgrade to the Gaussian copula it exists to widen the tails
    against.
    """
    from valdist.calibrate.coverage import check_coverage
    from valdist.spec.loader import hydrate, load_spec

    spec_path = tmp_path / "nu_spec.yaml"
    spec_path.write_text(_NU_SPEC_YAML)
    (tmp_path / "nu_spec.real").write_text("5.0")

    result = check_coverage(tmp_path, n=20000, seed=0)
    detail = result.details[0]

    spec = load_spec(spec_path)
    model = hydrate(spec)
    with_nu = model.run(n=20000, price=None, seed=0, nu=spec.nu)
    without_nu = model.run(n=20000, price=None, seed=0, nu=None)

    expected_p10 = float(np.quantile(with_nu.value, 0.10))
    wrong_p10 = float(np.quantile(without_nu.value, 0.10))

    assert detail["p10"] == pytest.approx(expected_p10)
    assert detail["p10"] != pytest.approx(wrong_p10), (
        "check_coverage() matched the nu=None run - spec.nu is being dropped"
    )


# --------------------------------------------------------------------------- #
# 3. Result.convergence() must thread the run's own sampling/nu
# --------------------------------------------------------------------------- #


def test_result_stores_sampling_and_nu(two_driver_model) -> None:
    model = two_driver_model()
    result = model.run(n=1000, seed=7, sampling="lhs", nu=3.0)
    assert result.sampling == "lhs"
    assert result.nu == 3.0


def test_convergence_reruns_with_same_sampling_and_nu(two_driver_model) -> None:
    """The 2N drift check must re-run with the SAME sampling/nu as the
    original - otherwise it's comparing two different estimators, not
    checking convergence of one.
    """
    model = two_driver_model(loading=0.8)
    result = model.run(n=1000, seed=7, sampling="lhs", nu=3.0)

    conv = result.convergence()

    manual_2n = model.run(n=2000, seed=7, sampling="lhs", nu=3.0)
    wrong_2n = model.run(n=2000, seed=7, sampling="mc", nu=None)

    assert conv["value_p50_2N"] == pytest.approx(float(np.median(manual_2n.value)))
    assert conv["value_p50_2N"] != pytest.approx(float(np.median(wrong_2n.value))), (
        "convergence() matched the mc/nu=None re-run - sampling/nu are being dropped"
    )
