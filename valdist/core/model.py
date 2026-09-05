from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

if TYPE_CHECKING:
    from valdist.core.results import Result

import numpy as np
from scipy import stats

from valdist.core.diagnostics import collect_diagnostics
from valdist.core.marginals import Marginal

ValuationFn = Callable[[Mapping[str, float]], float]


@runtime_checkable
class _CorrelationSource(Protocol):
    """Duck-type interface for dependence specifications."""

    def matrix(self, names: Sequence[str]) -> np.ndarray: ...

    def draw_z(
        self,
        names: Sequence[str],
        n: int,
        rng: np.random.Generator,
        sampling: Literal["mc", "lhs"] = "mc",
    ) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class Model:
    """Gaussian-copula Monte Carlo valuation engine.

    Parameters
    ----------
    drivers:
        Ordered list of Marginal objects (one per uncertain input).
    corr:
        Dependence source — a FactorModel. Must implement .matrix(names) and
        .draw_z(names, n, rng, sampling). Typed as a Protocol rather than
        FactorModel directly so a caller may supply their own, but the engine
        ships exactly one. Independence is FactorModel([], {}).
    valuation:
        Callable ``f(sample: dict) -> float`` implementing the domain
        valuation logic.  Lives in adapters/, never in core/.
    constants:
        Structural parameters (shares outstanding, debt, etc.) that are
        not sampled.  Merged into every call to ``valuation``.
    """

    def __init__(
        self,
        drivers: Sequence[Marginal],
        corr: _CorrelationSource,
        valuation: ValuationFn,
        constants: dict | None = None,
    ) -> None:
        self.drivers = list(drivers)
        self.names = [d.name for d in self.drivers]
        if len(set(self.names)) != len(self.names):
            raise ValueError("driver names must be unique")
        if not isinstance(corr, _CorrelationSource):
            missing = [
                name for name in ("matrix", "draw_z") if not callable(getattr(corr, name, None))
            ]
            raise TypeError(
                f"corr must implement _CorrelationSource (.matrix(names), "
                f".draw_z(names, n, rng, sampling)); missing: {', '.join(missing)}"
            )
        self.corr = corr
        self.valuation = valuation
        self.constants: dict = dict(constants or {})

    def _sample_inputs(
        self,
        n: int,
        rng: np.random.Generator,
        sampling: Literal["mc", "lhs"] = "mc",
        nu: float | None = None,
    ) -> np.ndarray:
        """Return an (n, d) matrix of correlated driver draws."""
        if nu is not None and nu <= 0:
            raise ValueError(f"nu must be > 0 if set, got {nu}")
        z = self.corr.draw_z(self.names, n, rng, sampling=sampling)
        if nu is None:
            u = stats.norm.cdf(z)
        else:
            w = rng.chisquare(nu, size=n) / nu
            t = z / np.sqrt(w)[:, None]
            u = stats.t.cdf(t, df=nu)
        return np.column_stack([d.ppf(u[:, i]) for i, d in enumerate(self.drivers)])

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        names = self.names
        constants = self.constants
        out = np.empty(len(X))
        for k in range(len(X)):
            row = {names[i]: X[k, i] for i in range(len(names))}
            out[k] = self.valuation({**constants, **row})
        return out

    def driver_swings(self) -> list[dict]:
        """Each driver's own effect on the valuation, isolated from the others.

        Every other driver is held at its median, this one moved from its p10 to
        its p90, and the change in value reported. Deterministic: no RNG, no run.

        Complements `Result.tornado()`, which is a rank correlation and so is
        scale-free: it says which driver value tracks, never by how much. Use
        this for magnitude - the two orderings can invert.

        The endpoints are the quantiles the model actually draws (`ppf(0.1)` /
        `ppf(0.9)`), not the typed p10/p90: three of the five families don't
        reproduce the triple they're given.
        """
        if not self.drivers:
            return []
        qs = np.array([0.1, 0.5, 0.9])
        bands = [np.asarray(d.ppf(qs), dtype=float) for d in self.drivers]
        base = np.array([[band[1] for band in bands]])
        v_base = float(self._evaluate(base)[0])

        out = []
        for i, driver in enumerate(self.drivers):
            ends = base.repeat(2, axis=0)
            ends[0, i], ends[1, i] = bands[i][0], bands[i][2]
            v_lo, v_hi = (float(v) for v in self._evaluate(ends))
            out.append(
                {
                    "name": driver.name,
                    "input_lo": float(bands[i][0]),
                    "input_hi": float(bands[i][2]),
                    "value_lo": v_lo,
                    "value_hi": v_hi,
                    "value_base": v_base,
                    "swing": abs(v_hi - v_lo),
                }
            )
        return out

    def run(
        self,
        n: int = 50_000,
        price: float | None = None,
        seed: int = 0,
        sampling: Literal["mc", "lhs"] = "mc",
        nu: float | None = None,
    ) -> Result:
        """Run n Monte Carlo draws."""
        from valdist.core.results import (
            Result,
        )  # deferred to avoid circular import at module load

        # The spec schema enforces n > 0, but Model.run() is public API and is
        # called directly (by tests, by the Python API, by Result.convergence()).
        # Without this, n=0 dies inside Result as "IndexError: index -1 is out
        # of bounds for axis 0 with size 0" — nowhere near the actual mistake.
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        rng = np.random.default_rng(seed)
        X = self._sample_inputs(n, rng, sampling=sampling, nu=nu)
        # Bind a diagnostic collector around evaluation so a valuation can report
        # per-draw conditions (e.g. that it floored an incoherent draw) without
        # core needing to know what any of them mean — see core/diagnostics.py.
        with collect_diagnostics() as diagnostics:
            value = self._evaluate(X)
        return Result(
            self,
            X,
            value,
            price=price,
            seed=seed,
            sampling=sampling,
            nu=nu,
            diagnostics=dict(diagnostics),
        )
