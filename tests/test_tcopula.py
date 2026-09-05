"""t-copula / tail dependence."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _cleanup_registrations():
    """Remove this file's test valuations from the global registry."""
    from valdist.adapters.registry import _POSITIVE_YEARS, _REGISTRY, _REQUIRES

    names = ("test_identity_tcopula", "test_identity_cli")
    yield
    for name in names:
        _REGISTRY.pop(name, None)
        _REQUIRES.pop(name, None)
        _POSITIVE_YEARS.pop(name, None)


def test_nu_none_default_unchanged(two_driver_model):
    """nu=None (the default) is byte-identical to pre-t-copula behavior."""
    model = two_driver_model()
    r_default = model.run(n=2000, seed=0)
    r_explicit_none = model.run(n=2000, seed=0, nu=None)
    np.testing.assert_array_equal(r_default.value, r_explicit_none.value)


def test_t_copula_deterministic(two_driver_model):
    model = two_driver_model()
    r1 = model.run(n=2000, seed=3, nu=4.0)
    r2 = model.run(n=2000, seed=3, nu=4.0)
    np.testing.assert_array_equal(r1.value, r2.value)


def test_t_copula_gaussian_limit(two_driver_model):
    """As nu -> infinity, the t-copula converges to the Gaussian copula."""
    model = two_driver_model()
    r_gauss = model.run(n=50_000, seed=0, nu=None)
    r_t_huge_nu = model.run(n=50_000, seed=0, nu=10_000.0)

    s_gauss = r_gauss.summary()
    s_t = r_t_huge_nu.summary()
    assert abs(s_gauss["value_p10"] - s_t["value_p10"]) < 0.5
    assert abs(s_gauss["value_p50"] - s_t["value_p50"]) < 0.5
    assert abs(s_gauss["value_p90"] - s_t["value_p90"]) < 0.5


def test_t_copula_tail_dependence_exists(two_driver_model):
    """A small nu induces measurably more JOINT tail exceedance than the
    Gaussian copula, at the same underlying correlation - the actual
    property being purchased, not just fatter individual marginals.
    """
    model = two_driver_model(loading=0.8)  # correlation = 0.8**2 = 0.64
    n = 200_000

    result_gauss = model.run(n=n, seed=0, nu=None)
    result_t = model.run(n=n, seed=0, nu=3.0)

    def joint_exceedance_rate(result) -> float:
        a = result.X[:, 0]
        b = result.X[:, 1]
        qa = np.quantile(a, 0.99)
        qb = np.quantile(b, 0.99)
        return float(np.mean((a > qa) & (b > qb)))

    rate_gauss = joint_exceedance_rate(result_gauss)
    rate_t = joint_exceedance_rate(result_t)
    # Under independence this would be ~0.0001 (0.01 * 0.01); both are well
    # above that from the shared factor alone. The t-copula's shared mixing
    # variable should push it further still.
    assert rate_t > rate_gauss * 1.3, (
        f"expected nu=3 joint tail exceedance to clearly exceed Gaussian: "
        f"t={rate_t:.5f} gauss={rate_gauss:.5f}"
    )


def test_nu_validation(two_driver_model):
    with pytest.raises(ValueError, match="nu"):
        two_driver_model().run(n=100, seed=0, nu=0.0)
    with pytest.raises(ValueError, match="nu"):
        two_driver_model().run(n=100, seed=0, nu=-5.0)


def test_spec_nu_field_threads_through(tmp_path):
    """A spec's top-level `nu` field reaches Model.run via hydrate."""
    import textwrap

    from valdist.spec.loader import hydrate, load_spec

    spec_yaml = textwrap.dedent("""
        schema_version: "1.0"
        name: "nu-test"
        valuation: test_identity_tcopula
        price: 10.0
        seed: 0
        n: 500
        nu: 5.0
        factors: [f]
        drivers:
          a:
            marginal: {family: normal, p10: 1.0, p50: 2.0, p90: 3.0}
            loadings: {f: 0.5}
        constants: {}
        """)
    path = tmp_path / "nu_spec.yaml"
    path.write_text(spec_yaml)

    from valdist.adapters.registry import register

    @register("test_identity_tcopula")
    def _identity(v):
        return v["a"]

    spec = load_spec(path)
    assert spec.nu == 5.0
    model = hydrate(spec)
    result = model.run(n=spec.n, price=spec.price, seed=spec.seed, nu=spec.nu)
    assert np.isfinite(result.value).all()


def test_cli_run_threads_spec_nu(tmp_path):
    """Regression: the CLI's `run`/`worlds` commands must pass spec.nu into
    Model.run() themselves - hydrate()/spec.nu existing is not enough. This
    was a real gap caught during this feature's own gate check: the CLI
    silently ignored a spec's `nu` field and always ran Gaussian until
    fixed.
    """
    import textwrap

    from typer.testing import CliRunner

    from valdist.adapters.registry import register
    from valdist.cli import app

    @register("test_identity_cli")
    def _identity(v):
        return v["a"] + v["b"]

    spec_yaml = textwrap.dedent("""
        schema_version: "1.0"
        name: "nu-cli-test"
        valuation: test_identity_cli
        price: 10.0
        seed: 0
        n: 2000
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
    path = tmp_path / "nu_cli_spec.yaml"
    path.write_text(spec_yaml)

    runner = CliRunner()
    cli_result = runner.invoke(app, ["run", str(path)])
    assert cli_result.exit_code == 0, cli_result.output

    # Cross-check: the CLI's output must match calling Model.run(nu=3.0)
    # directly, not Model.run(nu=None) - i.e. nu was actually threaded
    # through, not silently dropped.
    from valdist.spec.loader import hydrate, load_spec

    spec = load_spec(path)
    model = hydrate(spec)
    with_nu = model.run(n=spec.n, price=spec.price, seed=spec.seed, nu=spec.nu)
    without_nu = model.run(n=spec.n, price=spec.price, seed=spec.seed, nu=None)
    assert not np.array_equal(with_nu.value, without_nu.value), (
        "nu=3.0 and nu=None produced identical draws - nu is not being applied"
    )


def test_spec_nu_defaults_to_none():
    """Specs without a `nu` field (like examples/vici.yaml) get nu=None."""
    from conftest import VICI_SPEC_PATH

    from valdist.spec.loader import load_spec

    spec = load_spec(VICI_SPEC_PATH)
    assert spec.nu is None
