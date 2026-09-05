"""valdist — probabilistic intrinsic-value engine."""

from valdist.core import FactorModel, Marginal, Model, Result
from valdist.spec.loader import hydrate, load_spec
from valdist.spec.validate import SpecError, validate

__all__ = [
    "Marginal",
    "FactorModel",
    "Model",
    "Result",
    "SpecError",
    "load_spec",
    "hydrate",
    "validate",
    "run",
]

# Import adapter modules so their @register decorators execute at package load.
import valdist.adapters.equity  # noqa: F401, E402
import valdist.adapters.preferred  # noqa: F401, E402
import valdist.adapters.reit  # noqa: F401, E402


def run(spec_or_path) -> Result:
    """Load (if a path) and run a spec, returning a Result."""
    from pathlib import Path

    from valdist.spec.schema import SpecModel

    if isinstance(spec_or_path, (str, Path)):
        spec = load_spec(spec_or_path)
    elif isinstance(spec_or_path, SpecModel):
        spec = spec_or_path
    else:
        raise TypeError(f"expected str, Path, or SpecModel; got {type(spec_or_path)}")

    model = hydrate(spec)
    return model.run(n=spec.n, price=spec.price, seed=spec.seed, nu=spec.nu)
