from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import VICI_SPEC_PATH
from typer.testing import CliRunner

from valdist.cli import app

runner = CliRunner()

# --------------------------------------------------------------------------- #
# Shared fixture: small fast model (avoids slow VICI n=50000 in unit tests)
# --------------------------------------------------------------------------- #

_SIMPLE_SPEC_YAML = """\
schema_version: "1.0"
name: fast-test
valuation: test_sum_fa
price: 7.0
seed: 7
n: 2000
factors:
  - f1
  - f2
drivers:
  a:
    marginal: {family: normal, p10: 1.0, p50: 2.0, p90: 3.0}
    loadings: {f1: 0.90, f2: 0.05}
  b:
    marginal: {family: normal, p10: 1.0, p50: 2.0, p90: 3.0}
    loadings: {f1: 0.05, f2: 0.90}
weights:
  w_a: 60.0
  w_b: 40.0
"""

# valuation is 3*a + b so value is mostly driven by a, which loads heavily on f1


@pytest.fixture(scope="session", autouse=True)
def _register_test_sum_fa():
    """Register `test_sum_fa` ONCE, with exactly one formula, and clean up."""
    from valdist.adapters.registry import _REGISTRY

    _REGISTRY["test_sum_fa"] = lambda v: 3.0 * v["a"] + v["b"]
    yield
    _REGISTRY.pop("test_sum_fa", None)


@pytest.fixture(scope="session")
def fast_result(tmp_path_factory):
    """The small fast model, run exactly once for the whole session."""
    from valdist.spec.loader import hydrate, load_spec

    path = tmp_path_factory.mktemp("outputs") / "simple.yaml"
    path.write_text(_SIMPLE_SPEC_YAML)

    spec = load_spec(path)
    return hydrate(spec).run(n=spec.n, price=spec.price, seed=spec.seed)


# --------------------------------------------------------------------------- #
# factor_attribution()
# --------------------------------------------------------------------------- #


def test_factor_attribution_returns_all_factors_plus_idio(fast_result):
    """factor_attribution() returns a key per factor + 'idiosyncratic', sums to 1."""
    result = fast_result
    attr = result.factor_attribution()

    assert "f1" in attr
    assert "f2" in attr
    assert "idiosyncratic" in attr
    total = sum(attr.values())
    assert abs(total - 1.0) < 1e-6, f"shares sum to {total}"
    assert all(v >= 0 for v in attr.values()), f"negative share: {attr}"


def test_factor_attribution_f1_dominates(fast_result):
    """f1 dominates because it drives the high-weight driver a (3a+b valuation)."""
    result = fast_result
    attr = result.factor_attribution()
    assert attr["f1"] > attr["f2"], f"expected f1>f2 but got {attr}"


def test_factor_attribution_vici_rate_dominates():
    """VICI: rate factor dominates the attribution (rates bet per vici.yaml header)."""
    from valdist.spec.loader import hydrate, load_spec

    spec = load_spec(VICI_SPEC_PATH)
    model = hydrate(spec)
    result = model.run(n=spec.n, price=spec.price, seed=spec.seed, nu=spec.nu)

    attr = result.factor_attribution()
    assert "rate" in attr and "fundamentals" in attr and "idiosyncratic" in attr
    assert attr["rate"] > attr["fundamentals"], f"expected rate>fundamentals but got {attr}"


def test_factor_attribution_requires_factor_model():
    """factor_attribution raises TypeError for a non-FactorModel dependence source."""
    import numpy as np

    from valdist.core.marginals import Marginal
    from valdist.core.model import Model

    class _IndependentSource:
        """Minimal correlation source with no factor structure at all."""

        def matrix(self, names):
            return np.eye(len(names))

        def draw_z(self, names, n, rng, sampling="mc"):
            return rng.standard_normal((n, len(names)))

    m = Marginal("x", p10=1.0, p50=2.0, p90=3.0, family="normal")
    model = Model([m], _IndependentSource(), valuation=lambda v: v["x"])
    result = model.run(n=500, price=2.0, seed=0)
    with pytest.raises(TypeError):
        result.factor_attribution()


# --------------------------------------------------------------------------- #
# worlds()
# --------------------------------------------------------------------------- #


def test_worlds_returns_three_scenarios(fast_result):
    """worlds() at default qs returns a list of 3 dicts."""
    result = fast_result
    w = result.worlds()
    assert len(w) == 3


def test_worlds_contains_driver_names_and_value(fast_result):
    """Each world dict has all driver names + 'value' + 'quantile'."""
    result = fast_result
    w = result.worlds()
    for scenario in w:
        assert "a" in scenario
        assert "b" in scenario
        assert "value" in scenario
        assert "quantile" in scenario


def test_worlds_ordered_by_value(fast_result):
    """P10 world has a STRICTLY lower value than P50, which is strictly lower
    than P90.
    """
    result = fast_result
    w = result.worlds()
    assert w[0]["value"] < w[1]["value"] < w[2]["value"], (
        f"worlds not strictly ordered: {[x['value'] for x in w]}"
    )


def test_worlds_custom_quantiles(fast_result):
    """worlds() respects custom qs parameter."""
    result = fast_result
    w = result.worlds(qs=(0.25, 0.75))
    assert len(w) == 2
    assert w[0]["quantile"] == 0.25
    assert w[1]["quantile"] == 0.75


# --------------------------------------------------------------------------- #
# report module
# --------------------------------------------------------------------------- #


def test_report_to_text_nonempty(fast_result):
    """report.to_text() returns a non-empty string for a VICI-like result."""
    from valdist import report

    result = fast_result
    text = report.to_text(result)
    assert isinstance(text, str)
    assert len(text) > 50
    assert "p_undervalued" in text.lower() or "undervalued" in text.lower()


def test_report_to_text_structure(fast_result):
    """Structural check: to_text() must carry the value/price/tornado sections
    it claims to, not just some text containing the word "undervalued" - the
    fixture's spec has a price set and drivers a/b, so every section applies.
    """
    from valdist import report

    result = fast_result
    s = result.summary()
    text = report.to_text(result)
    lines = text.splitlines()

    assert lines[0] == lines[2] == lines[-1], "missing the top/bottom box border"
    assert "Value distribution" in text
    for label, value in (
        ("P10", s["value_p10"]),
        ("P50", s["value_p50"]),
        ("P90", s["value_p90"]),
    ):
        assert f"{label}  = {value:>8.2f}" in text, f"missing {label} line for {value}"

    # price is set on this fixture's spec, so the price/MOS sections must appear.
    assert f"Price = {s['price']:>8.2f}" in text
    assert f"{s['p_undervalued']:.4f}" in text
    assert "Margin of safety" in text

    # Tornado must list both drivers from the fixture spec (a, b).
    tornado_idx = next(i for i, line in enumerate(lines) if "Tornado" in line)
    tornado_block = "\n".join(lines[tornado_idx:])
    for name in result.model.names:
        assert name in tornado_block, f"driver {name!r} missing from tornado section"

    # This fixture spec's bands don't cross and isn't a degenerate p_undervalued
    # case, so no WARNINGS section should appear - no false positives.
    assert result.warnings() == []
    assert "WARNINGS" not in text


def test_report_to_json_valid(fast_result):
    """report.to_json() carries the FULL result surface, not just three keys."""
    from valdist import report

    result = fast_result
    d = json.loads(report.to_json(result))
    s = result.summary()

    # Everything summary() reports must survive the trip through JSON.
    for key, expected in s.items():
        assert key in d, f"summary key {key!r} missing from to_json payload"
        assert d[key] == pytest.approx(expected), f"{key}: json={d[key]} summary={expected}"

    # Price is set on this fixture, so the price-dependent keys must be present.
    for key in (
        "price",
        "p_undervalued",
        "p_undervalued_stderr",
        "mos_p10",
        "mos_p50",
        "mos_p90",
    ):
        assert key in d, f"{key!r} missing"

    # The honesty channels and the sensitivity table.
    assert "warnings" in d and isinstance(d["warnings"], list)
    assert "diagnostics" in d and isinstance(d["diagnostics"], dict)
    assert "tornado" in d, "tornado missing from to_json payload"
    assert {row["driver"] for row in d["tornado"]} == set(result.model.names)
    assert all("rank_corr" in row for row in d["tornado"])


# --------------------------------------------------------------------------- #
# CLI: validate
# --------------------------------------------------------------------------- #


def test_cli_validate_clean_spec(tmp_path: Path):
    """CLI validate exits 0 on a clean spec."""

    spec_file = tmp_path / "clean.yaml"
    spec_file.write_text(_SIMPLE_SPEC_YAML)

    result = runner.invoke(app, ["validate", str(spec_file)])
    assert result.exit_code == 0, result.output


def test_cli_validate_bad_spec(tmp_path: Path):
    """CLI validate exits 1 and reports errors when spec is invalid."""
    bad = """\
schema_version: "1.0"
name: bad
valuation: test_sum_fa
price: 1.0
factors: [rate]
drivers:
  overloaded:
    marginal: {family: normal, p10: 1.0, p50: 2.0, p90: 3.0}
    loadings: {rate: 1.1}
  ghost_ref:
    marginal: {family: lognormal, p10: 1.0, p50: 2.0, p90: 4.0}
    loadings: {ghost: 0.5}
weights: {w_a: 60.0, w_b: 60.0}
"""
    spec_file = tmp_path / "bad.yaml"
    spec_file.write_text(bad)

    result = runner.invoke(app, ["validate", str(spec_file)])
    assert result.exit_code == 1, result.output
    # Assert the SPECIFIC codes. The old assertion was
    #   "loadings_exceed_one" in output or "error" in output.lower()
    # and the second branch is a tautology - the CLI's own header reads
    # "INVALID – '...' has N error(s):", which contains "error". It would have
    # passed with every error code stripped out of the digest.
    for code in ("loadings_exceed_one", "unknown_factor", "weights_not_100"):
        assert code in result.output, f"{code!r} missing from digest:\n{result.output}"


# --------------------------------------------------------------------------- #
# CLI: run
# --------------------------------------------------------------------------- #


def test_cli_run_simple_spec(tmp_path: Path):
    """CLI run exits 0 and prints a report for a valid spec."""

    spec_file = tmp_path / "simple.yaml"
    spec_file.write_text(_SIMPLE_SPEC_YAML)

    result = runner.invoke(app, ["run", str(spec_file)])
    assert result.exit_code == 0, result.output
    # Report should mention key stats
    assert "p_undervalued" in result.output.lower() or "undervalued" in result.output.lower()
    assert "p50" in result.output.lower() or "median" in result.output.lower()


def test_cli_run_vici():
    """Gate test: `valdist run examples/vici.yaml` exits 0 and prints the golden
    numbers - the documented ones, on their own labelled lines.
    """
    result = runner.invoke(app, ["run", str(VICI_SPEC_PATH)])
    assert result.exit_code == 0, result.output
    out = result.output

    # Golden VICI (examples/vici.yaml header): 28.01 / 33.70 / 41.00
    assert "P10  =    28.01" in out, f"P10 line missing/wrong:\n{out}"
    assert "P50  =    33.70" in out, f"P50 line missing/wrong:\n{out}"
    assert "P90  =    41.00" in out, f"P90 line missing/wrong:\n{out}"
    # p_undervalued = 0.9312, asserted against its own label, not loose in the text
    assert "P(undervalued)  = 0.9312" in out, f"p_undervalued line missing/wrong:\n{out}"


# --------------------------------------------------------------------------- #
# CLI: worlds
# --------------------------------------------------------------------------- #


def test_cli_worlds_simple_spec(tmp_path: Path):
    """CLI worlds exits 0 and prints a worlds table."""

    spec_file = tmp_path / "simple.yaml"
    spec_file.write_text(_SIMPLE_SPEC_YAML)

    result = runner.invoke(app, ["worlds", str(spec_file)])
    assert result.exit_code == 0, result.output
    # Should contain quantile labels
    assert "p10" in result.output.lower() or "0.1" in result.output
