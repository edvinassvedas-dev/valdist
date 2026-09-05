"""A `requires` key is location-agnostic: driver or constant, either works."""

from __future__ import annotations

import copy

import pytest
import yaml

from valdist.spec.loader import hydrate
from valdist.spec.schema import SpecModel
from valdist.spec.validate import validate

GOLDEN = "examples/vici.yaml"

# `nav_other` is a real reit_v2 `requires` key that the golden spec supplies as a
# constant. It is also exactly the key the A07 analysis needed to sample.
PROMOTED = "nav_other"
# `cap_rate` is a real reit_v2 `requires` key the golden spec supplies as a driver.
DEMOTED = "cap_rate"


@pytest.fixture
def golden() -> dict:
    """The golden spec as a raw dict, so a test can move a key between blocks
    without touching examples/vici.yaml on disk (it is the user's artifact)."""
    with open(GOLDEN) as fh:
        return yaml.safe_load(fh)


def _codes(spec: SpecModel) -> list[str]:
    return [e.code for e in validate(spec)]


def test_golden_spec_supplies_nav_other_as_a_constant(golden):
    """Guard the premise. If the golden spec ever stops carrying `nav_other` as a
    constant, the two promotion tests below would be exercising nothing."""
    assert PROMOTED in golden["constants"]
    assert PROMOTED not in golden["drivers"]
    assert DEMOTED in golden["drivers"]


def test_constant_promoted_to_driver_validates_and_runs(golden):
    """The anchor. Move a `requires` key from `constants:` to `drivers:` and the
    spec must still validate and run - this is what lets a spec author sample a
    judgment instead of freezing it."""
    raw = copy.deepcopy(golden)
    const = raw["constants"].pop(PROMOTED)
    raw["drivers"][PROMOTED] = {
        "marginal": {
            "family": "normal",
            "p50": const,
            "p10": const - 300,
            "p90": const + 300,
        },
        "loadings": {"rate": 0.2, "fundamentals": -0.3},
    }

    spec = SpecModel.model_validate(raw)
    assert _codes(spec) == [], "promoting a requires key to a driver must not error"

    result = hydrate(spec).run(n=2000, seed=0)
    assert result.value.std() > 0, "the promoted key must actually vary the output"


def test_promoted_constant_is_sampled_and_visible_in_the_tornado(golden):
    """Promotion is only useful if the analyst can see the promoted key's
    influence. It must land in the model's driver list and hence in tornado()."""
    raw = copy.deepcopy(golden)
    const = raw["constants"].pop(PROMOTED)
    raw["drivers"][PROMOTED] = {
        "marginal": {
            "family": "normal",
            "p50": const,
            "p10": const - 300,
            "p90": const + 300,
        },
        "loadings": {"rate": 0.2, "fundamentals": -0.3},
    }

    model = hydrate(SpecModel.model_validate(raw))
    assert PROMOTED in model.names, "promoted key must be a sampled driver"
    assert PROMOTED not in model.constants, "promoted key must not also be a constant"

    names = [name for name, _ in model.run(n=2000, seed=0).tornado()]
    assert PROMOTED in names, "a promoted judgment the analyst cannot see is no better than frozen"


def test_driver_demoted_to_constant_validates_and_runs(golden):
    """The contract is symmetric: a driver may equally be pinned as a constant
    (an analyst who is genuinely certain of a cap rate). It then stops being
    sampled - and stops appearing in the tornado."""
    raw = copy.deepcopy(golden)
    p50 = raw["drivers"].pop(DEMOTED)["marginal"]["p50"]
    raw["constants"][DEMOTED] = p50

    spec = SpecModel.model_validate(raw)
    assert _codes(spec) == [], "demoting a driver to a constant must not error"

    model = hydrate(spec)
    assert DEMOTED not in model.names
    assert model.constants[DEMOTED] == p50

    names = [name for name, _ in model.run(n=2000, seed=0).tornado()]
    assert DEMOTED not in names


def test_requires_key_absent_from_BOTH_blocks_is_still_caught(golden):
    """The location-agnostic lookup must not become a hole: a `requires` key
    supplied by neither block is still a missing_required_input. Without this,
    "look in both places" could silently degrade into "look nowhere"."""
    raw = copy.deepcopy(golden)
    raw["constants"].pop(PROMOTED)

    assert "missing_required_input" in _codes(SpecModel.model_validate(raw))
