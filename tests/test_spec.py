from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _register_test_identity():
    """Register the test valuation these specs name."""
    from valdist.adapters.registry import _REGISTRY

    sentinel = object()
    previous = _REGISTRY.get("test_identity", sentinel)
    _REGISTRY["test_identity"] = lambda v: v["a"] + v["b"]
    yield
    if previous is sentinel:
        _REGISTRY.pop("test_identity", None)
    else:
        _REGISTRY["test_identity"] = previous


# --------------------------------------------------------------------------- #
# Anchor #7: aggregated validation digest
# --------------------------------------------------------------------------- #


def _three_error_spec() -> dict:
    """Return a raw spec dict with exactly three distinct planted errors."""
    return {
        "schema_version": "1.0",
        "name": "test-three-errors",
        "valuation": "test_identity",
        "price": 10.0,
        "factors": ["rate"],  # "ghost" is NOT declared here
        "drivers": {
            "overloaded": {
                # Error 1: 1.1² = 1.21 > 1  →  loadings_exceed_one
                "marginal": {"family": "normal", "p10": 1.0, "p50": 2.0, "p90": 3.0},
                "loadings": {"rate": 1.1},
            },
            "ghost_ref": {
                # Error 2: "ghost" not in factors  →  unknown_factor
                "marginal": {"family": "lognormal", "p10": 1.0, "p50": 2.0, "p90": 4.0},
                "loadings": {"ghost": 0.5},
            },
        },
        # Error 3: 60 + 60 = 120 ≠ 100  →  weights_not_100
        "weights": {"w_a": 60.0, "w_b": 60.0},
    }


def test_validation_digest_three_errors():
    """Three planted spec errors all surface in one digest."""
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    spec = SpecModel.model_validate(_three_error_spec())
    errors = validate(spec)

    codes = {e.code for e in errors}
    assert "loadings_exceed_one" in codes, f"missing loadings_exceed_one; got {codes}"
    assert "unknown_factor" in codes, f"missing unknown_factor; got {codes}"
    assert "weights_not_100" in codes, f"missing weights_not_100; got {codes}"
    assert len(errors) >= 3, f"expected ≥3 errors, got {len(errors)}: {errors}"


def test_validation_clean_spec_has_no_errors():
    """A well-formed spec produces an empty digest."""
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    raw = {
        "schema_version": "1.0",
        "name": "clean-spec",
        "valuation": "test_identity",
        "price": 10.0,
        "factors": ["f"],
        "drivers": {
            "a": {
                "marginal": {"family": "normal", "p10": 1.0, "p50": 2.0, "p90": 3.0},
                "loadings": {"f": 0.7},
            },
            "b": {
                "marginal": {"family": "lognormal", "p10": 1.0, "p50": 2.0, "p90": 4.0},
                "loadings": {"f": -0.5},
            },
        },
        "weights": {"w_x": 40.0, "w_y": 60.0},  # sums to 100
        "constants": {"shares": 1000},
    }

    spec = SpecModel.model_validate(raw)
    errors = validate(spec)
    assert errors == [], f"unexpected errors: {errors}"


def test_check_marginal_fit_catches_lognormal_and_normal_mismatch():
    """check_marginal_fit surfaces quantile-fit mismatches via `validate`'s
    own aggregated-digest path - not just a hydrate()-time warning that
    `valdist validate` would otherwise never see.
    """
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import check_marginal_fit, validate

    raw = {
        "schema_version": "1.0",
        "name": "mismatch-spec",
        "valuation": "test_identity",
        "price": 10.0,
        "factors": ["f"],
        "drivers": {
            "fcf_growth": {
                "marginal": {
                    "family": "lognormal",
                    "p10": 0.5,
                    "p50": 6.0,
                    "p90": 11.0,
                },
                "loadings": {"f": 0.5},
            },
            "terminal_growth": {
                "marginal": {"family": "normal", "p10": 1.5, "p50": 2.5, "p90": 3.0},
                "loadings": {"f": 0.3},
            },
            "wacc": {
                # symmetric normal - should not appear in notices
                "marginal": {"family": "normal", "p10": 9.4, "p50": 10.4, "p90": 11.4},
                "loadings": {"f": -0.5},
            },
        },
    }
    spec = SpecModel.model_validate(raw)

    # This is a quality notice, not a spec violation - validate() itself
    # stays clean (Σλ²≤1, no unknown factors, no weights field).
    assert validate(spec) == []

    notices = check_marginal_fit(spec)
    codes_and_drivers = {(n.code, "fcf_growth" in n.message) for n in notices}
    assert ("quantile_mismatch", True) in codes_and_drivers
    driver_names_flagged = {
        name
        for name in ("fcf_growth", "terminal_growth", "wacc")
        for n in notices
        if name in n.message
    }
    assert "fcf_growth" in driver_names_flagged
    assert "terminal_growth" in driver_names_flagged
    assert "wacc" not in driver_names_flagged


def test_check_marginal_fit_flags_possible_percent_fraction_mixup():
    """Adapters expect driver values in PERCENT
    (8.0 for 8%) and divide by 100 internally; nothing previously caught a
    spec author entering 0.08 instead. A driver whose entire p10/p50/p90
    band sits inside (-1, 1) is flagged as a non-fatal notice - not a hard
    error, since a genuinely small percent value (e.g. terminal growth =
    0.5) is indistinguishable from a fraction mixup at this boundary; no
    fixed threshold resolves that ambiguity, so this is
    surfaced for the analyst to judge rather than silently rejected or
    silently accepted.
    """
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import check_marginal_fit, validate

    raw = {
        "schema_version": "1.0",
        "name": "percent-fraction-mixup",
        "valuation": "test_identity",
        "price": 10.0,
        "factors": ["f"],
        "drivers": {
            "cost_of_equity": {
                # meant 8.0/9.0/10.0 (percent) but entered as a fraction
                "marginal": {"family": "normal", "p10": 0.08, "p50": 0.09, "p90": 0.10},
                "loadings": {"f": 0.5},
            },
        },
    }
    spec = SpecModel.model_validate(raw)

    # Not a spec violation - still internally consistent and runnable.
    assert validate(spec) == []

    notices = check_marginal_fit(spec)
    codes = {n.code for n in notices}
    assert "possible_percent_fraction_mixup" in codes
    assert any("cost_of_equity" in n.message for n in notices)


def test_check_marginal_fit_no_mixup_notice_for_ordinary_percent_band():
    """A driver whose band spans outside (-1, 1) - an ordinary percent
    value - must not trigger the notice."""
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import check_marginal_fit

    raw = {
        "schema_version": "1.0",
        "name": "ordinary-percent",
        "valuation": "test_identity",
        "price": 10.0,
        "factors": ["f"],
        "drivers": {
            "cost_of_equity": {
                "marginal": {"family": "normal", "p10": 7.0, "p50": 8.5, "p90": 10.0},
                "loadings": {"f": 0.5},
            },
        },
    }
    spec = SpecModel.model_validate(raw)
    codes = {n.code for n in check_marginal_fit(spec)}
    assert "possible_percent_fraction_mixup" not in codes


def test_check_marginal_fit_mixup_notice_boundary_is_exclusive():
    """A band that just touches +/-1.0 (not strictly inside it) must not
    trigger the notice -- 1.0 itself is already a plausible percent value."""
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import check_marginal_fit

    raw = {
        "schema_version": "1.0",
        "name": "boundary-case",
        "valuation": "test_identity",
        "price": 10.0,
        "factors": ["f"],
        "drivers": {
            "rate": {
                "marginal": {"family": "normal", "p10": 0.5, "p50": 0.75, "p90": 1.0},
                "loadings": {"f": 0.5},
            },
        },
    }
    spec = SpecModel.model_validate(raw)
    codes = {n.code for n in check_marginal_fit(spec)}
    assert "possible_percent_fraction_mixup" not in codes


# --------------------------------------------------------------------------- #
# validate() aggregation gaps: these must ALL be caught by validate() and NOT
# require hydrate() to fail-fast with a lone ValueError, per the same
# "aggregated digest" discipline as above.
# --------------------------------------------------------------------------- #


def test_validate_catches_lognormal_nonpositive_p10():
    """A lognormal driver with p10 <= 0 currently passes validate() clean and
    only blows up as a lone ValueError at hydrate() - must be caught here.
    """
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    raw = {
        "schema_version": "1.0",
        "name": "lognormal-nonpositive",
        "valuation": "test_identity",
        "price": 1.0,
        "factors": [],
        "drivers": {
            "a": {
                "marginal": {
                    "family": "lognormal",
                    "p10": -1.0,
                    "p50": 2.0,
                    "p90": 5.0,
                },
                "loadings": {},
            },
        },
    }
    spec = SpecModel.model_validate(raw)
    errors = validate(spec)
    codes = {e.code for e in errors}
    assert "lognormal_nonpositive_p10" in codes, f"missing lognormal_nonpositive_p10; got {codes}"


def test_validate_catches_infeasible_lognormal3():
    """An infeasible lognormal3 triple (degenerate shift equation) must be
    caught by validate(), not just raised as a lone ValueError at hydrate().
    """
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    raw = {
        "schema_version": "1.0",
        "name": "lognormal3-infeasible",
        "valuation": "test_identity",
        "price": 1.0,
        "factors": [],
        "drivers": {
            # A11-style: p50 exactly the arithmetic mean of p10/p90.
            "a": {
                "marginal": {
                    "family": "lognormal3",
                    "p10": 0.5,
                    "p50": 2.5,
                    "p90": 4.5,
                },
                "loadings": {},
            },
        },
    }
    spec = SpecModel.model_validate(raw)
    errors = validate(spec)
    codes = {e.code for e in errors}
    assert "lognormal3_infeasible" in codes, f"missing lognormal3_infeasible; got {codes}"


def test_validate_catches_duplicate_factor_names():
    """Duplicate names in factors: currently pass validate() clean and only
    blow up as a lone ValueError inside FactorModel construction at hydrate().
    """
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    raw = {
        "schema_version": "1.0",
        "name": "dup-factors",
        "valuation": "test_identity",
        "price": 1.0,
        "factors": ["f", "f"],
        "drivers": {
            "a": {
                "marginal": {"family": "normal", "p10": 1.0, "p50": 2.0, "p90": 3.0},
                "loadings": {},
            },
        },
    }
    spec = SpecModel.model_validate(raw)
    errors = validate(spec)
    codes = {e.code for e in errors}
    assert "duplicate_factor" in codes, f"missing duplicate_factor; got {codes}"


def test_validate_catches_driver_constant_name_collision():
    """A driver named like a constant silently shadows it (dict-merge order
    in Model._evaluate); validate() must flag this instead of staying quiet.
    """
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    raw = {
        "schema_version": "1.0",
        "name": "collision",
        "valuation": "test_identity",
        "price": 1.0,
        "factors": [],
        "drivers": {
            "wacc": {
                "marginal": {"family": "normal", "p10": 8.0, "p50": 9.0, "p90": 10.0},
                "loadings": {},
            },
        },
        "constants": {"wacc": 9.0},
    }
    spec = SpecModel.model_validate(raw)
    errors = validate(spec)
    codes = {e.code for e in errors}
    assert "name_collision" in codes, f"missing name_collision; got {codes}"


def test_validate_catches_driver_weight_and_constant_weight_collisions():
    """Weight names collide the same way: weights are merged into `constants`
    at hydrate() (spec.constants | spec.weights), so either a driver or a
    constant colliding with a weight name silently shadows it.
    """
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    raw = {
        "schema_version": "1.0",
        "name": "weight-collision",
        "valuation": "test_identity",
        "price": 1.0,
        "factors": [],
        "drivers": {
            "w_a": {
                "marginal": {"family": "normal", "p10": 1.0, "p50": 2.0, "p90": 3.0},
                "loadings": {},
            },
        },
        "constants": {"w_b": 1.0},
        "weights": {"w_a": 40.0, "w_b": 60.0},
    }
    spec = SpecModel.model_validate(raw)
    errors = validate(spec)
    codes = {e.code for e in errors}
    assert "name_collision" in codes, f"missing name_collision; got {codes}"
    # Both collisions (driver/weight and constant/weight) should be reported.
    assert sum(1 for e in errors if e.code == "name_collision") >= 2


def test_validate_no_false_positives_on_clean_spec():
    """None of the new Batch-C checks should fire on the existing clean spec."""
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    raw = {
        "schema_version": "1.0",
        "name": "clean-spec-2",
        "valuation": "test_identity",
        "price": 10.0,
        "factors": ["f"],
        "drivers": {
            "a": {
                "marginal": {"family": "normal", "p10": 1.0, "p50": 2.0, "p90": 3.0},
                "loadings": {"f": 0.7},
            },
            "b": {
                "marginal": {"family": "lognormal", "p10": 1.0, "p50": 2.0, "p90": 4.0},
                "loadings": {"f": -0.5},
            },
        },
        "weights": {"w_x": 40.0, "w_y": 60.0},
        "constants": {"shares": 1000},
    }
    spec = SpecModel.model_validate(raw)
    assert validate(spec) == []


# --------------------------------------------------------------------------- #
# The last few aggregation gaps.
# Each currently passes validate() clean and then fails fast somewhere else.
# --------------------------------------------------------------------------- #


def test_validate_catches_unknown_valuation():
    """An unregistered `valuation:` key passes validate() and only raises
    KeyError at hydrate() (item 6).
    """
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    raw = {
        "schema_version": "1.0",
        "name": "unknown-valuation",
        "valuation": "no_such_adapter",
        "price": 1.0,
        "factors": [],
        "drivers": {
            "a": {
                "marginal": {"family": "normal", "p10": 1.0, "p50": 2.0, "p90": 3.0},
                "loadings": {},
            },
        },
    }
    spec = SpecModel.model_validate(raw)
    errors = validate(spec)
    codes = {e.code for e in errors}
    assert "unknown_valuation" in codes, f"missing unknown_valuation; got {codes}"
    # The message should name the registered adapters, so the fix is obvious.
    msg = next(e.message for e in errors if e.code == "unknown_valuation")
    assert "reit_v2" in msg and "equity_v2" in msg


def test_validate_catches_missing_adapter_required_input():
    """The worst gap (item 7): a reit_v2 spec missing a required constant
    survives validate() AND hydrate(), then dies with a bare KeyError in the
    middle of the Monte Carlo run. It must be caught in the digest instead.
    """
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    raw = {
        "schema_version": "1.0",
        "name": "missing-constant",
        "valuation": "reit_v2",
        "price": 30.0,
        "n": 100,
        "factors": [],
        "drivers": {
            name: {
                "marginal": {"family": "normal", "p10": 1.0, "p50": 2.0, "p90": 3.0},
                "loadings": {},
            }
            for name in (
                "cap_rate",
                "cost_of_equity",
                "div_growth",
                "div_terminal",
                "affo_growth",
                "affo_terminal",
            )
        },
        "constants": {
            "shares": 1090.0,
            "dps": 1.7,
            "ddm_stage1_years": 10,
            "affo": 2680.0,
            "affo_years": 10,
            "noi": 2400.0,
            "nav_debt": 17000.0,
            # nav_other deliberately omitted
        },
        "weights": {"w_ddm": 40.0, "w_affo": 40.0, "w_nav": 20.0},
    }
    spec = SpecModel.model_validate(raw)
    errors = validate(spec)
    codes = {e.code for e in errors}
    assert "missing_required_input" in codes, f"missing_required_input absent; got {codes}"
    msg = next(e.message for e in errors if e.code == "missing_required_input")
    assert "nav_other" in msg


def test_validate_required_inputs_satisfied_by_any_namespace():
    """The valuation is handed drivers ∪ constants ∪ weights as one flat dict,
    so a required name may legitimately be supplied by any of the three. The
    check must not care which one (e.g. the weights live in `weights:`).
    """
    # The golden spec supplies reit_v2's 17 required names across all three
    # namespaces and must stay clean.
    from conftest import VICI_SPEC_PATH

    from valdist.spec.loader import load_spec
    from valdist.spec.validate import validate

    spec = load_spec(VICI_SPEC_PATH)
    errors = [e for e in validate(spec) if e.code == "missing_required_input"]
    assert errors == [], f"golden spec wrongly flagged: {errors}"


def test_validate_catches_unsupported_schema_version():
    """schema_version is currently accepted as anything and read by nothing."""
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    raw = {
        "schema_version": "9.9",
        "name": "bad-schema-version",
        "valuation": "test_identity",
        "price": 1.0,
        "factors": [],
        "drivers": {
            "a": {
                "marginal": {"family": "normal", "p10": 1.0, "p50": 2.0, "p90": 3.0},
                "loadings": {},
            },
        },
    }
    spec = SpecModel.model_validate(raw)
    errors = validate(spec)
    codes = {e.code for e in errors}
    assert "unsupported_schema_version" in codes, f"missing unsupported_schema_version; got {codes}"


def test_validate_still_aggregates_everything_in_one_digest():
    """The new checks must join the digest, not fail-fast. A spec with
    an unknown valuation AND a bad schema_version AND weights≠100 reports all
    three at once.
    """
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    raw = {
        "schema_version": "9.9",
        "name": "many-errors",
        "valuation": "no_such_adapter",
        "price": 1.0,
        "factors": [],
        "drivers": {
            "a": {
                "marginal": {"family": "normal", "p10": 1.0, "p50": 2.0, "p90": 3.0},
                "loadings": {},
            },
        },
        "weights": {"w_a": 60.0, "w_b": 60.0},
    }
    spec = SpecModel.model_validate(raw)
    codes = {e.code for e in validate(spec)}
    assert {
        "unsupported_schema_version",
        "unknown_valuation",
        "weights_not_100",
    } <= codes


def test_golden_vici_spec_still_validates_clean():
    """Regression guard for the whole batch: the user's golden artifact must
    not acquire a single new error.
    """
    from conftest import VICI_SPEC_PATH

    from valdist.spec.loader import load_spec
    from valdist.spec.validate import validate

    assert validate(load_spec(VICI_SPEC_PATH)) == []


# --------------------------------------------------------------------------- #
# weights_not_100 must have REACH (see 2026-07-13)
#
# The check read `spec.weights` -- the top-level `weights:` block. NO spec in
# this repo populates that block: every one of them puts the blend weights in
# `constants:` as w_ddm/w_affo/w_nav, which is where the adapters read them
# from. So the check was green in tests (which hand-built synthetic specs WITH
# a weights: block) and unreachable on every spec a human actually writes.
#
# Measured on the golden spec with w_nav 50 -> 40 (weights summing to 90):
# validate() said "valid" and the run reported P50 34.04 / p_undervalued 0.9467
# against the true 33.70 / 0.9312, with zero warnings. The adapters normalise
# by total_w, so a non-100 sum does not crash - it silently renormalises the
# analyst's mix (15/35/40 becomes 16.7/38.9/44.4). A wrong number that looks
# right, which is the failure mode this codebase fears most.
# --------------------------------------------------------------------------- #


def _vici_raw_with_weights(w_ddm: float, w_affo: float, w_nav: float) -> dict:
    """The golden spec, real-shaped (weights in `constants:`), reweighted."""
    import yaml
    from conftest import VICI_SPEC_PATH

    raw = yaml.safe_load(VICI_SPEC_PATH.read_text())
    raw["constants"] |= {"w_ddm": w_ddm, "w_affo": w_affo, "w_nav": w_nav}
    return raw


def test_weights_in_constants_that_miss_100_are_caught():
    """The real spec shape: weights live in `constants:`, not in `weights:`."""
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    spec = SpecModel.model_validate(_vici_raw_with_weights(15, 35, 40))  # sums to 90
    codes = {e.code for e in validate(spec)}
    assert "weights_not_100" in codes, f"a 90-sum blend validated clean; got {codes}"


def test_weights_in_constants_that_hit_100_are_clean():
    """No false positive: the golden weights (15/35/50) must stay silent."""
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    spec = SpecModel.model_validate(_vici_raw_with_weights(15, 35, 50))
    assert [e.code for e in validate(spec)] == []


def test_run_refuses_a_spec_whose_blend_weights_miss_100(tmp_path):
    """The gate must inherit the fix: it is not enough for validate() to
    know -- `run()` must refuse, or the silent-renormalisation path stays open
    exactly as it was.
    """
    import yaml

    import valdist
    from valdist.spec.loader import load_spec
    from valdist.spec.validate import SpecError

    p = tmp_path / "bad_weights.yaml"
    p.write_text(yaml.safe_dump(_vici_raw_with_weights(15, 35, 40)))

    with pytest.raises(SpecError, match="weights_not_100"):
        valdist.run(load_spec(p))


def test_marginal_ordering_enforced():
    """Schema rejects p10 > p90."""
    from pydantic import ValidationError

    from valdist.spec.schema import SpecModel

    raw = {
        "schema_version": "1.0",
        "name": "bad-order",
        "valuation": "test_identity",
        "price": 1.0,
        "factors": [],
        "drivers": {
            "d": {
                "marginal": {"family": "normal", "p10": 9.0, "p50": 5.0, "p90": 1.0},
                "loadings": {},
            }
        },
    }
    with pytest.raises(ValidationError):
        SpecModel.model_validate(raw)


# --------------------------------------------------------------------------- #
# Phase-2 gate: load → validate → hydrate → run (trivial test spec)
# --------------------------------------------------------------------------- #


GATE_SPEC_YAML = textwrap.dedent("""\
    schema_version: "1.0"
    name: gate-test
    valuation: test_identity
    price: 5.0
    seed: 42
    n: 1000
    factors:
      - f
    drivers:
      a:
        marginal: {family: normal, p10: 1.0, p50: 2.0, p90: 3.0}
        loadings: {f: 0.7}
      b:
        marginal: {family: lognormal, p10: 1.0, p50: 2.0, p90: 4.0}
        loadings: {f: -0.5}
    weights:
      w_a: 40.0
      w_b: 60.0
    constants:
      scale: 1.0
""")


def test_errors_and_notices_are_distinguishable_by_type():
    """Severity must be expressible in the type, not inferable only from which
    function you happened to call. validate() returns ValidationError (the spec
    is broken); check_marginal_fit() returns ValidationNotice (the spec runs,
    but may not mean what the analyst thinks). They used to be the same class,
    so a caller holding a list could not tell a fatal error from a hint.
    """
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import (
        ValidationError,
        ValidationNotice,
        check_marginal_fit,
        validate,
    )

    raw = {
        "schema_version": "1.0",
        "name": "mixed",
        "valuation": "test_identity",
        "price": 10.0,
        "factors": ["f"],
        "drivers": {
            # a fatal error (Σλ² > 1) AND a non-fatal notice (fraction-looking band)
            "a": {
                "marginal": {"family": "normal", "p10": 0.08, "p50": 0.09, "p90": 0.10},
                "loadings": {"f": 1.5},
            },
        },
    }
    spec = SpecModel.model_validate(raw)

    errors = validate(spec)
    notices = check_marginal_fit(spec)

    assert errors and all(isinstance(e, ValidationError) for e in errors)
    assert notices and all(isinstance(n, ValidationNotice) for n in notices)
    # Not merely a subclass relation - a notice must never read as an error.
    assert not any(isinstance(n, ValidationError) for n in notices)


def test_validate_catches_a_sampled_projection_horizon():
    """A projection horizon supplied as a DRIVER is a category error."""
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    raw = {
        "schema_version": "1.0",
        "name": "sampled-years",
        "valuation": "equity_v2",
        "price": 25.0,
        "factors": [],
        "drivers": {
            "fcf_growth": {
                "marginal": {"family": "normal", "p10": 2.0, "p50": 4.0, "p90": 6.0},
                "loadings": {},
            },
            "terminal_growth": {
                "marginal": {"family": "normal", "p10": 1.0, "p50": 1.5, "p90": 2.0},
                "loadings": {},
            },
            "wacc": {
                "marginal": {"family": "normal", "p10": 7.0, "p50": 8.5, "p90": 10.0},
                "loadings": {},
            },
            # the planted error: a SAMPLED projection horizon
            "years": {
                "marginal": {"family": "normal", "p10": 6.0, "p50": 7.0, "p90": 8.0},
                "loadings": {},
            },
        },
        "constants": {
            "shares": 200.0,
            "debt": 1500.0,
            "cash": 300.0,
            "fcff": 500.0,
            "normalized_earnings": 480.0,
            "relative_value_per_share": 30.0,
        },
        "weights": {"w_dcf": 45.0, "w_epv": 35.0, "w_relative": 20.0},
    }
    errors = validate(SpecModel.model_validate(raw))
    codes = {e.code for e in errors}
    assert "sampled_years" in codes, f"a sampled horizon validated clean; got {codes}"
    assert any("years" in e.message for e in errors if e.code == "sampled_years")


def test_validate_checks_year_counts_supplied_as_weights_too():
    """The valuation is handed drivers ∪ constants ∪ weights merged FLAT, so a
    year count may legitimately arrive from any namespace - missing_required_input
    already searches all three. The years check only looked at spec.constants, so
    a horizon supplied anywhere else skipped validation entirely."""
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    raw = {
        "schema_version": "1.0",
        "name": "years-in-weights",
        "valuation": "equity_v2",
        "price": 25.0,
        "factors": [],
        "drivers": {
            "fcf_growth": {
                "marginal": {"family": "normal", "p10": 2.0, "p50": 4.0, "p90": 6.0},
                "loadings": {},
            },
            "terminal_growth": {
                "marginal": {"family": "normal", "p10": 1.0, "p50": 1.5, "p90": 2.0},
                "loadings": {},
            },
            "wacc": {
                "marginal": {"family": "normal", "p10": 7.0, "p50": 8.5, "p90": 10.0},
                "loadings": {},
            },
        },
        "constants": {
            "shares": 200.0,
            "debt": 1500.0,
            "cash": 300.0,
            "fcff": 500.0,
            "normalized_earnings": 480.0,
            "relative_value_per_share": 30.0,
        },
        # `years` lives here, and is invalid. Weights still sum to 100 without it
        # mattering to that check, so nothing else would have caught this.
        "weights": {"w_dcf": 45.0, "w_epv": 35.0, "w_relative": 20.0, "years": -3.0},
    }
    codes = {e.code for e in validate(SpecModel.model_validate(raw))}
    assert "nonpositive_years" in codes, f"a bad year count in weights was missed; got {codes}"


def test_validate_catches_nonpositive_year_count():
    """A year-count constant < 1 must land in the aggregated digest."""
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    raw = {
        "schema_version": "1.0",
        "name": "bad-years",
        "valuation": "equity_v2",
        "price": 25.0,
        "factors": [],
        "drivers": {
            "fcf_growth": {
                "marginal": {"family": "normal", "p10": 2.0, "p50": 4.0, "p90": 6.0},
                "loadings": {},
            },
            "terminal_growth": {
                "marginal": {"family": "normal", "p10": 1.0, "p50": 1.5, "p90": 2.0},
                "loadings": {},
            },
            "wacc": {
                "marginal": {"family": "normal", "p10": 7.0, "p50": 8.5, "p90": 10.0},
                "loadings": {},
            },
        },
        "constants": {
            "shares": 200.0,
            "debt": 1500.0,
            "cash": 300.0,
            "fcff": 500.0,
            "normalized_earnings": 480.0,
            "years": -5.0,  # <-- the planted error
            "relative_value_per_share": 30.0,
        },
        "weights": {"w_dcf": 45.0, "w_epv": 35.0, "w_relative": 20.0},
    }
    codes = {e.code for e in validate(SpecModel.model_validate(raw))}
    assert "nonpositive_years" in codes, f"got {codes}"

    raw["constants"]["years"] = 0
    codes = {e.code for e in validate(SpecModel.model_validate(raw))}
    assert "nonpositive_years" in codes, f"years=0 not caught; got {codes}"

    raw["constants"]["years"] = 7
    codes = {e.code for e in validate(SpecModel.model_validate(raw))}
    assert "nonpositive_years" not in codes, f"false positive on years=7; got {codes}"


def test_validate_catches_nonpositive_price():
    """price <= 0 produces meaningless margin-of-safety percentages."""
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    raw = {
        "schema_version": "1.0",
        "name": "bad-price",
        "valuation": "test_identity",
        "price": -50.0,
        "factors": [],
        "drivers": {
            "a": {
                "marginal": {"family": "normal", "p10": 1.0, "p50": 2.0, "p90": 3.0},
                "loadings": {},
            }
        },
    }
    codes = {e.code for e in validate(SpecModel.model_validate(raw))}
    assert "nonpositive_price" in codes, f"got {codes}"

    raw["price"] = 0.0
    codes = {e.code for e in validate(SpecModel.model_validate(raw))}
    assert "nonpositive_price" in codes, f"price=0 not caught; got {codes}"


def test_validate_catches_nonfinite_values():
    """inf / nan anywhere in the spec is meaningless and must not reach the
    engine. Verified pre-fix: `shares: .inf` and `debt: .nan` passed pydantic
    and validate() untouched."""
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    raw = {
        "schema_version": "1.0",
        "name": "nonfinite",
        "valuation": "test_identity",
        "price": 10.0,
        "factors": ["f"],
        "drivers": {
            "a": {
                "marginal": {
                    "family": "normal",
                    "p10": 1.0,
                    "p50": 2.0,
                    "p90": float("inf"),
                },
                "loadings": {"f": float("nan")},
            }
        },
        "constants": {"shares": float("inf"), "debt": float("nan")},
    }
    errors = validate(SpecModel.model_validate(raw))
    codes = {e.code for e in errors}
    assert "nonfinite_value" in codes, f"got {codes}"

    # Aggregated, not fail-fast - every offending name is named.
    blob = " ".join(e.message for e in errors if e.code == "nonfinite_value")
    for name in ("shares", "debt", "a"):
        assert name in blob, f"{name!r} not reported in: {blob}"


def test_load_spec_rejects_unknown_extension(tmp_path: Path):
    """load_spec used to treat EVERY non-YAML extension as JSON, so a .txt
    (or .csv, or a typo'd .yml) fell through to json.loads() and died with a
    cryptic `JSONDecodeError: Expecting value: line 1 column 1` that says
    nothing about the real problem."""
    from valdist.spec.loader import load_spec

    path = tmp_path / "spec.txt"
    path.write_text("schema_version: '1.0'\nname: t\n")

    with pytest.raises(ValueError, match=r"\.txt|extension|unsupported"):
        load_spec(path)


def test_load_spec_still_accepts_yaml_yml_and_json(tmp_path: Path):
    """No false positives: the three supported extensions keep working."""
    import json as _json

    from valdist.spec.loader import load_spec

    raw = {
        "schema_version": "1.0",
        "name": "ext-test",
        "valuation": "test_identity",
        "price": 10.0,
        "drivers": {
            "a": {
                "marginal": {"family": "normal", "p10": 1.0, "p50": 2.0, "p90": 3.0},
                "loadings": {},
            }
        },
    }
    for suffix, text in (
        (".yaml", yaml_dump(raw)),
        (".yml", yaml_dump(raw)),
        (".json", _json.dumps(raw)),
    ):
        path = tmp_path / f"spec{suffix}"
        path.write_text(text)
        assert load_spec(path).name == "ext-test", f"{suffix} failed to load"


def yaml_dump(obj: dict) -> str:
    import yaml

    return yaml.safe_dump(obj)


def test_e2e_load_validate_run(tmp_path: Path):
    """Phase-2 gate: valid spec loads from YAML, validates clean, and runs."""
    from valdist.adapters.registry import register
    from valdist.spec.loader import hydrate, load_spec
    from valdist.spec.validate import validate

    # Register a trivial test valuation (idempotent if already registered)
    register("test_identity")(lambda v: v["a"] + v["b"])

    spec_file = tmp_path / "gate.yaml"
    spec_file.write_text(GATE_SPEC_YAML)

    spec = load_spec(spec_file)
    assert spec.name == "gate-test"
    assert spec.valuation == "test_identity"

    errors = validate(spec)
    assert errors == [], f"unexpected errors: {errors}"

    model = hydrate(spec)
    result = model.run(n=spec.n, price=spec.price, seed=spec.seed)

    assert result.value.shape == (1000,)
    assert result.p_undervalued is not None
    assert result.summary() is not None
