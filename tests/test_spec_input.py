"""Anchor 7: a spec's text is never silently changed or discarded on the way in."""

from __future__ import annotations

import json

import pydantic
import pytest
import yaml
from conftest import VICI_SPEC_PATH
from typer.testing import CliRunner

import valdist
from valdist.cli import app
from valdist.spec.loader import load_spec
from valdist.spec.validate import validate

runner = CliRunner()

VICI_TEXT = VICI_SPEC_PATH.read_text(encoding="utf-8")
DRIVER = "div_growth"


def _vici() -> dict:
    return yaml.safe_load(VICI_TEXT)


def _dump(tmp_path, spec: dict):
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return path


def _text(tmp_path, text: str, name: str = "spec.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _codes(path) -> list[str]:
    return [e.code for e in validate(load_spec(path))]


def _misspell_loadings(d):
    driver = d["drivers"][DRIVER]
    driver["loading"] = driver.pop("loadings")


def _top_level_typo(d):
    d["seeed"] = 1


def _marginal_typo(d):
    d["drivers"][DRIVER]["marginal"]["p95"] = 1.0


@pytest.mark.parametrize(
    ("plant", "where"),
    [
        (_top_level_typo, "seeed"),
        (_misspell_loadings, f"drivers.{DRIVER}.loading"),
        (_marginal_typo, f"drivers.{DRIVER}.marginal.p95"),
    ],
)
def test_unknown_key_is_reported_at_every_level(tmp_path, plant, where):
    d = _vici()
    plant(d)
    errors = validate(load_spec(_dump(tmp_path, d)))
    assert [e.code for e in errors] == ["unknown_key"]
    assert where in errors[0].message


def test_every_unknown_key_lands_in_one_digest_beside_other_errors(tmp_path):
    d = _vici()
    _top_level_typo(d)
    _misspell_loadings(d)
    _marginal_typo(d)
    d["constants"]["w_nav"] -= 10
    codes = _codes(_dump(tmp_path, d))
    assert codes.count("unknown_key") == 3
    assert "weights_not_100" in codes


def test_unknown_key_refuses_to_run(tmp_path):
    from valdist.spec.validate import SpecError

    d = _vici()
    _misspell_loadings(d)
    with pytest.raises(SpecError, match="unknown_key"):
        valdist.run(str(_dump(tmp_path, d)))


def test_cli_prints_the_unknown_key_in_the_digest(tmp_path):
    d = _vici()
    _misspell_loadings(d)
    result = runner.invoke(app, ["validate", str(_dump(tmp_path, d))])
    assert result.exit_code == 1
    assert f"drivers.{DRIVER}.loading" in result.output


def test_repeated_yaml_key_is_reported_with_its_line(tmp_path):
    path = _text(tmp_path, VICI_TEXT + "\nprice: 999\n")
    errors = [e for e in validate(load_spec(path)) if e.code == "duplicate_key"]
    assert len(errors) == 1
    assert "'price'" in errors[0].message
    assert f"line {len(VICI_TEXT.splitlines()) + 2}" in errors[0].message


def test_every_repeated_key_is_reported_not_the_last(tmp_path):
    path = _text(tmp_path, VICI_TEXT + "\nprice: 999\nname: x\n")
    assert _codes(path).count("duplicate_key") == 2


def test_repeated_yaml_key_inside_a_nested_mapping_is_reported(tmp_path):
    d = _vici()
    d["constants"] = {"shares": 1.0}
    text = yaml.safe_dump(d, sort_keys=False).replace("shares: 1.0", "shares: 1.0\n  shares: 2.0")
    assert "duplicate_key" in _codes(_text(tmp_path, text))


def test_repeated_json_key_is_reported(tmp_path):
    body = json.dumps(_vici())
    text = body[:-1] + ', "price": 999}'
    errors = [e for e in validate(load_spec(_text(tmp_path, text, "s.json")))]
    assert ["duplicate_key"] == [e.code for e in errors]
    assert "'price'" in errors[0].message


def test_repeated_key_refuses_to_run(tmp_path):
    from valdist.spec.validate import SpecError

    with pytest.raises(SpecError, match="duplicate_key"):
        valdist.run(str(_text(tmp_path, VICI_TEXT + "\nprice: 999\n")))


def test_yaml_merges_and_their_overrides_are_not_repeated_keys(tmp_path):
    d = _vici()
    first, second = list(d["drivers"])[:2]
    text = yaml.safe_dump(d, sort_keys=False)
    spec = load_spec(
        _text(
            tmp_path,
            text.replace(f"{first}:\n", f"{first}: &base\n", 1).replace(
                f"{second}:\n", f"{second}:\n    <<: *base\n    <<: *base\n", 1
            ),
        )
    )
    assert "duplicate_key" not in [e.code for e in validate(spec)]


def _price(d, value):
    d["price"] = value


def _n(d, value):
    d["n"] = value


def _constant(d, value):
    d["constants"]["shares"] = value


def _band(d, value):
    d["drivers"][DRIVER]["marginal"]["p50"] = value


def _loading(d, value):
    d["drivers"][DRIVER]["loadings"] = {"rate": value}


@pytest.mark.parametrize(
    ("plant", "value"),
    [
        (_price, True),
        (_price, "27.5"),
        (_n, True),
        (_n, 50000.0),
        (_constant, True),
        (_constant, "1000"),
        (_band, "4.0"),
        (_loading, True),
    ],
)
def test_a_numeric_field_refuses_what_only_coerces_to_a_number(tmp_path, plant, value):
    d = _vici()
    plant(d, value)
    with pytest.raises(pydantic.ValidationError):
        load_spec(_dump(tmp_path, d))


# PyYAML 1.1 reads an exponent with no sign as a string.
def test_an_unsigned_yaml_exponent_is_refused_rather_than_coerced(tmp_path):
    text = VICI_TEXT.replace("price:", "price: 1.5e8  #", 1)
    with pytest.raises(pydantic.ValidationError):
        load_spec(_text(tmp_path, text))


def test_a_yaml_integer_is_still_a_number(tmp_path):
    d = _vici()
    _price(d, 27)
    _band(d, 4)
    assert validate(load_spec(_dump(tmp_path, d))) == []
