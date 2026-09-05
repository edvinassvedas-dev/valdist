"""Direct coverage of spec/schema.py's pydantic validators: _check_order
and _check_nu were previously exercised only
indirectly (through full spec YAML fixtures via test_spec.py), and an invalid
`family` value was never tested at the schema level at all.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from valdist.spec.schema import MarginalSpec, SpecModel

_ONE_DRIVER = {
    "a": {
        "marginal": {"family": "normal", "p10": 1.0, "p50": 2.0, "p90": 3.0},
        "loadings": {},
    }
}


def _base_spec(**overrides) -> dict:
    spec = {
        "schema_version": "1.0",
        "name": "schema-test",
        "valuation": "some_valuation",
        "price": 10.0,
        "drivers": _ONE_DRIVER,
    }
    spec.update(overrides)
    return spec


# --------------------------------------------------------------------------- #
# MarginalSpec._check_order
# --------------------------------------------------------------------------- #


def test_marginal_spec_accepts_ascending_quantiles():
    m = MarginalSpec(family="normal", p10=1.0, p50=2.0, p90=3.0)
    assert (m.p10, m.p50, m.p90) == (1.0, 2.0, 3.0)


def test_marginal_spec_accepts_equal_quantiles():
    """p10 <= p50 <= p90 allows equality, not just strict ascending order."""
    m = MarginalSpec(family="normal", p10=2.0, p50=2.0, p90=2.0)
    assert (m.p10, m.p50, m.p90) == (2.0, 2.0, 2.0)


def test_marginal_spec_rejects_p10_greater_than_p50():
    with pytest.raises(ValidationError, match="p10 ≤ p50 ≤ p90"):
        MarginalSpec(family="normal", p10=5.0, p50=2.0, p90=8.0)


def test_marginal_spec_rejects_p50_greater_than_p90():
    with pytest.raises(ValidationError, match="p10 ≤ p50 ≤ p90"):
        MarginalSpec(family="normal", p10=1.0, p50=9.0, p90=8.0)


def test_marginal_spec_rejects_unknown_family():
    """family is a Literal[...] -- pydantic's own type validation, not a
    custom validator, but it must still reject junk at the schema level."""
    with pytest.raises(ValidationError):
        MarginalSpec(family="uniform", p10=1.0, p50=2.0, p90=3.0)


# --------------------------------------------------------------------------- #
# SpecModel._check_nu
# --------------------------------------------------------------------------- #


def test_spec_model_nu_defaults_to_none():
    spec = SpecModel.model_validate(_base_spec())
    assert spec.nu is None


def test_spec_model_accepts_positive_nu():
    spec = SpecModel.model_validate(_base_spec(nu=4.0))
    assert spec.nu == 4.0


def test_spec_model_rejects_zero_nu():
    with pytest.raises(ValidationError, match="nu must be > 0"):
        SpecModel.model_validate(_base_spec(nu=0.0))


def test_spec_model_rejects_negative_nu():
    with pytest.raises(ValidationError, match="nu must be > 0"):
        SpecModel.model_validate(_base_spec(nu=-3.0))
