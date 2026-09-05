"""YAML/JSON spec loader and model hydration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import yaml

from valdist.adapters.registry import get_valuation
from valdist.core.factors import FactorModel
from valdist.core.marginals import Marginal
from valdist.core.model import Model
from valdist.spec.schema import SpecModel

_YAML_SUFFIXES = (".yaml", ".yml")
_JSON_SUFFIXES = (".json",)

# Every extension load_spec() understands. Exported so callers that discover
# spec files (calibrate.check_coverage) can't drift out of step with the ones
# that load them: check_coverage used to glob only "*.yaml" and silently
# skipped .yml specs, reporting a clean nan hit_rate over zero pairs.
SPEC_SUFFIXES = _YAML_SUFFIXES + _JSON_SUFFIXES


def load_spec(path: Union[str, Path]) -> SpecModel:
    """Load a YAML or JSON spec file and return a validated SpecModel."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _YAML_SUFFIXES:
        raw = yaml.safe_load(path.read_text())
    elif suffix in _JSON_SUFFIXES:
        raw = json.loads(path.read_text())
    else:
        supported = ", ".join(_YAML_SUFFIXES + _JSON_SUFFIXES)
        raise ValueError(
            f"{path}: unsupported spec extension {suffix or '(none)'!r}; "
            f"expected one of {supported}"
        )
    return SpecModel.model_validate(raw)


def hydrate(spec: SpecModel, *, validate: bool = True) -> Model:
    """Build a runnable Model from a validated spec.

    Driver order matches the insertion order of `spec.drivers`. Loadings naming
    a factor not listed in `spec.factors` are rejected by the validator.

    `validate=True` (the default) refuses to build a Model from a spec that
    `validate()` rejects, raising the whole digest as `SpecError`. Non-fatal
    notices do not block. `validate=False` is the escape hatch for a caller who
    knowingly wants a Model from a broken spec; `valdist.run()` has none.
    """
    if validate:
        from valdist.spec.validate import SpecError
        from valdist.spec.validate import validate as _validate

        errors = _validate(spec)
        if errors:
            raise SpecError(errors)

    drivers = [
        Marginal(
            name=name,
            p10=d.marginal.p10,
            p50=d.marginal.p50,
            p90=d.marginal.p90,
            family=d.marginal.family,
        )
        for name, d in spec.drivers.items()
    ]
    valuation = get_valuation(spec.valuation)
    constants = {**spec.constants, **spec.weights}
    return Model(drivers, factor_model(spec), valuation, constants=constants)


def factor_model(spec: SpecModel) -> FactorModel:
    """The dependence layer for *spec*, built the one way it is built."""
    return FactorModel(
        factors=list(spec.factors),
        loadings={name: dict(d.loadings) for name, d in spec.drivers.items()},
    )
