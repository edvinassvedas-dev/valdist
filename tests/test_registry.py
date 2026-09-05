"""Direct coverage of the adapter registry: unregistered-name lookup,
required_inputs() with/without a declared `requires=`, and re-registration
behavior.
"""

from __future__ import annotations

import pytest

from valdist.adapters.registry import (
    _BLEND_WEIGHTS,
    _POSITIVE_YEARS,
    _REGISTRY,
    _REQUIRES,
    get_valuation,
    is_registered,
    known_valuations,
    positive_year_inputs,
    register,
    required_inputs,
)

_NAME = "_test_registry_probe"


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    _REGISTRY.pop(_NAME, None)
    _REQUIRES.pop(_NAME, None)
    _POSITIVE_YEARS.pop(_NAME, None)
    _BLEND_WEIGHTS.pop(_NAME, None)


def test_get_valuation_unregistered_name_raises_key_error():
    with pytest.raises(KeyError, match="not registered"):
        get_valuation("_definitely_not_a_registered_valuation_name")


def test_is_registered_false_for_unknown_name():
    assert is_registered("_definitely_not_a_registered_valuation_name") is False


def test_register_and_get_valuation_roundtrip():
    @register(_NAME)
    def fn(v):
        return v["x"]

    assert is_registered(_NAME) is True
    assert get_valuation(_NAME) is fn
    assert _NAME in known_valuations()


def test_required_inputs_none_when_requires_omitted():
    """An adapter that doesn't declare `requires=` skips the check entirely
    (module docstring) - required_inputs() must return None, not an empty set,
    so callers can tell "no requirement declared" apart from "declared empty"."""

    @register(_NAME)
    def fn(v):
        return v["x"]

    assert required_inputs(_NAME) is None


def test_required_inputs_returns_declared_frozenset():
    @register(_NAME, requires={"shares", "wacc"})
    def fn(v):
        return v["shares"] * v["wacc"]

    result = required_inputs(_NAME)
    assert result == frozenset({"shares", "wacc"})
    assert isinstance(result, frozenset)


def test_required_inputs_none_for_never_registered_name():
    assert required_inputs("_never_registered_at_all") is None


def test_positive_year_inputs_declared_by_adapter():
    """Which constants are year counts is DOMAIN knowledge, so the adapter
    declares it rather than the validator hardcoding names - same channel as
    `requires=`. This is what lets spec/validate.py catch a bad year count
    while staying domain-agnostic."""
    from valdist.adapters.registry import positive_year_inputs

    @register(_NAME, requires={"n_years"}, positive_years={"n_years"})
    def fn(v):
        return float(v["n_years"])

    assert positive_year_inputs(_NAME) == frozenset({"n_years"})


def test_positive_year_inputs_none_when_not_declared():
    @register(_NAME, requires={"x"})
    def fn(v):
        return v["x"]

    assert positive_year_inputs(_NAME) is None


def test_shipped_adapters_declare_their_year_counts():
    """The two shipped adapters must actually use the channel, or the
    validator has nothing to check and #8 silently reopens."""
    from valdist.adapters.registry import positive_year_inputs

    assert positive_year_inputs("reit_v2") == frozenset({"ddm_stage1_years", "affo_years"})
    assert positive_year_inputs("equity_v2") == frozenset({"years"})


def test_blend_weight_inputs_declared_by_adapter():
    """Which constants are blend weights is DOMAIN knowledge, so the adapter
    declares it - the same channel, and the same reasoning, as `requires=` and
    `positive_years=`. This is what gives weights_not_100 reach over specs that
    keep their weights in `constants:` (i.e. all of them) without spec/ ever
    hardcoding a name like "w_ddm"."""
    from valdist.adapters.registry import blend_weight_inputs

    @register(_NAME, requires={"w_a", "w_b"}, blend_weights={"w_a", "w_b"})
    def fn(v):
        return v["w_a"] + v["w_b"]

    assert blend_weight_inputs(_NAME) == frozenset({"w_a", "w_b"})


def test_blend_weight_inputs_none_when_not_declared():
    from valdist.adapters.registry import blend_weight_inputs

    @register(_NAME, requires={"x"})
    def fn(v):
        return v["x"]

    assert blend_weight_inputs(_NAME) is None


def test_shipped_adapters_declare_their_blend_weights():
    """Both shipped adapters must actually use the channel, or weights_not_100
    has nothing to check and silently reopens - which is exactly how it came
    to be unreachable in the first place."""
    from valdist.adapters.registry import blend_weight_inputs

    assert blend_weight_inputs("reit_v2") == frozenset({"w_ddm", "w_affo", "w_nav"})
    assert blend_weight_inputs("equity_v2") == frozenset({"w_dcf", "w_epv", "w_relative"})


def test_reregistration_last_one_wins():
    """Registering the same name twice overwrites the callable and its
    requires - there is no error, no merging, just last-write-wins."""

    @register(_NAME, requires={"a"})
    def first(v):
        return 1.0

    @register(_NAME, requires={"b"})
    def second(v):
        return 2.0

    assert get_valuation(_NAME) is second
    assert get_valuation(_NAME) is not first
    assert required_inputs(_NAME) == frozenset({"b"})


def test_reregistration_without_requires_clears_the_stale_entry():
    """Re-registering REPLACES the adapter, declarations included."""

    from valdist.adapters.registry import blend_weight_inputs

    @register(_NAME, requires={"a"}, positive_years={"a"}, blend_weights={"a"})
    def first(v):
        return 1.0

    @register(_NAME)
    def second(v):
        return 2.0

    assert get_valuation(_NAME) is second
    assert required_inputs(_NAME) is None, "stale requires survived re-registration"
    assert positive_year_inputs(_NAME) is None, "stale positive_years survived re-registration"
    assert blend_weight_inputs(_NAME) is None, "stale blend_weights survived re-registration"
