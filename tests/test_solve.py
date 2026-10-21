"""Model.value_at / Model.implied and `valdist solve`."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from valdist.cli import app
from valdist.core import FactorModel, Marginal, Model

ROOT = Path(__file__).resolve().parents[1]
VICI = str(ROOT / "examples" / "vici.yaml")
runner = CliRunner()


def _model(valuation, *drivers, constants=None) -> Model:
    return Model(list(drivers), FactorModel([], {}), valuation, constants or {})


def test_value_at_uses_the_typed_p50_not_the_fitted_median():
    # For pert the typed p50 is the mode: Stage 1's base, not ppf(0.5).
    x = Marginal("x", p10=4.4, p50=20.4, p90=38.4, family="pert")
    assert abs(float(x.ppf(0.5)) - 20.4) > 0.1
    m = _model(lambda v: v["x"] + v["c"], x, constants={"c": 1.0})
    assert m.value_at() == 21.4


def test_value_at_overrides_a_driver_a_constant_or_a_new_key():
    m = _model(
        lambda v: v["x"] * v["c"] + v.get("extra", 0.0),
        Marginal("x", p10=1.0, p50=2.0, p90=3.0),
        constants={"c": 10.0},
    )
    assert m.value_at({"x": 5.0}) == 50.0
    assert m.value_at({"c": 3.0}) == 6.0
    assert m.value_at({"extra": 1.5}) == 21.5


def test_implied_finds_the_exact_root_of_a_linear_valuation():
    m = _model(
        lambda v: 2.0 * v["x"] + v["y"],
        Marginal("x", p10=1.0, p50=2.0, p90=3.0),
        Marginal("y", p10=0.0, p50=1.0, p90=2.0, family="normal"),
    )
    implied = {i["name"]: i for i in m.implied(price=9.0)}
    assert implied["x"]["roots"] == [pytest.approx(4.0, abs=1e-9)]
    assert implied["y"]["roots"] == [pytest.approx(5.0, abs=1e-9)]


def test_implied_reports_every_root_of_a_non_monotone_valuation():
    m = _model(lambda v: (v["x"] - 2.0) ** 2, Marginal("x", p10=1.0, p50=2.0, p90=3.0))
    (only,) = m.implied(price=1.0)
    assert only["roots"] == [pytest.approx(1.0, abs=1e-9), pytest.approx(3.0, abs=1e-9)]


def test_implied_is_empty_when_the_price_is_out_of_reach_and_says_where_it_looked():
    m = _model(
        lambda v: min(v["x"], 1.0), Marginal("x", p10=0.0, p50=0.5, p90=1.0, family="normal")
    )
    (only,) = m.implied(price=5.0)
    assert only["roots"] == []
    assert only["lo"] < 0.0 < 1.0 < only["hi"]


def test_implied_holds_the_other_drivers_and_the_overrides_fixed():
    m = _model(
        lambda v: v["x"] + v["y"],
        Marginal("x", p10=1.0, p50=2.0, p90=3.0),
        Marginal("y", p10=0.0, p50=1.0, p90=2.0, family="normal"),
    )
    implied = {i["name"]: i for i in m.implied(price=10.0, overrides={"y": 4.0})}
    assert implied["x"]["roots"] == [pytest.approx(6.0, abs=1e-9)]
    assert "y" not in implied


def test_implied_marks_a_root_that_only_exists_through_a_floor():
    from valdist.core.diagnostics import flag

    def valuation(v):
        x = v["x"]
        if x < 0.0:
            flag("x_floored")
            x = 0.0
        return x

    m = _model(valuation, Marginal("x", p10=1.0, p50=2.0, p90=3.0))
    (only,) = m.implied(price=0.0)
    assert only["roots"] and only["flags"] == [["x_floored"]]


def test_cli_solve_prints_the_base_value_and_every_implied_input():
    from valdist.spec.loader import hydrate, load_spec

    spec = load_spec(VICI)
    model = hydrate(spec)
    result = runner.invoke(app, ["solve", VICI])
    assert result.exit_code == 0, result.output
    assert f"{model.value_at():.4f}" in result.output
    for name in model.names:
        assert name in result.output.split("Implied")[1]


def test_cli_solve_applies_set_overrides():
    from valdist.spec.loader import hydrate, load_spec

    model = hydrate(load_spec(VICI))
    name = model.names[0]
    result = runner.invoke(app, ["solve", VICI, "--set", f"{name}=1.5"])
    assert result.exit_code == 0, result.output
    assert f"{model.value_at({name: 1.5}):.4f}" in result.output
    assert name not in result.output.split("Implied")[1]


def test_cli_solve_rejects_a_malformed_set():
    result = runner.invoke(app, ["solve", VICI, "--set", "no_equals_sign"])
    assert result.exit_code == 1


@pytest.mark.parametrize("name", ["price", "no_such_input"])
def test_cli_solve_rejects_a_set_name_the_model_never_reads(name):
    # `--set price=33.4` printed "with price = 33.4" and solved at the spec's price.
    result = runner.invoke(app, ["solve", VICI, "--set", f"{name}=1.0"])
    assert result.exit_code == 1
    assert name in result.output


def test_cli_solve_accepts_a_constant():
    from valdist.spec.loader import hydrate, load_spec

    model = hydrate(load_spec(VICI))
    name = sorted(model.constants)[0]
    result = runner.invoke(app, ["solve", VICI, "--set", f"{name}=1.0"])
    assert result.exit_code == 0, result.output
    assert f"{model.value_at({name: 1.0}):.4f}" in result.output


CLINICAL = """\
schema_version: "1.0"
name: clinical-probe
valuation: clinical_v1
price: 0.5
seed: 0
n: 1000
factors: []
drivers:
  discount_rate:
    marginal: {family: normal, p10: 12.0, p50: 15.0, p90: 18.0}
  approval_probability:
    marginal: {family: normal, p10: 20.0, p50: 25.0, p90: 30.0}
constants:
  cash: 57.4
  debt: 0.0
  other_liabilities: 11.3
  shares: 98.1
  years_to_approval: 3
  annual_burn: 30.0
  asset_value: 800.0
  raise_amount: 0.0
  raise_price: 0.40
"""


def test_cli_solve_rejects_a_constant_the_adapter_never_reads(tmp_path):
    spec = tmp_path / "clinical.yaml"
    spec.write_text(CLINICAL + "  unread: 1.0\n")
    result = runner.invoke(app, ["solve", str(spec), "--set", "unread=2.0"])
    assert result.exit_code == 1
    assert "unread" in result.output


def test_cli_solve_accepts_an_optional_input_the_spec_omits(tmp_path):
    spec = tmp_path / "clinical.yaml"
    spec.write_text(CLINICAL)
    from valdist.spec.loader import hydrate, load_spec

    model = hydrate(load_spec(spec))
    result = runner.invoke(app, ["solve", str(spec), "--set", "years_to_readout=1"])
    assert result.exit_code == 0, result.output
    held = model.value_at({"years_to_readout": 1.0})
    assert held != model.value_at()
    assert f"{held:.4f}" in result.output


def test_cli_solve_falls_back_to_spec_inputs_when_the_adapter_declares_none(tmp_path):
    from valdist.adapters.registry import _REGISTRY, register

    @register("_solve_probe")
    def probe(v):
        return v["x"] + v["c"]

    spec = tmp_path / "probe.yaml"
    spec.write_text(
        'schema_version: "1.0"\nname: probe\nvaluation: _solve_probe\nprice: 5.0\n'
        "seed: 0\nn: 100\nfactors: []\n"
        "drivers:\n  x:\n    marginal: {family: normal, p10: 1.0, p50: 2.0, p90: 3.0}\n"
        "constants:\n  c: 1.0\n"
    )
    try:
        held = runner.invoke(app, ["solve", str(spec), "--set", "c=4.0"])
        bad = runner.invoke(app, ["solve", str(spec), "--set", "price=4.0"])
    finally:
        _REGISTRY.pop("_solve_probe", None)
    assert held.exit_code == 0, held.output
    assert "6.0000" in held.output
    assert bad.exit_code == 1
