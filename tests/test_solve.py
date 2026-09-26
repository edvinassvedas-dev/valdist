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
