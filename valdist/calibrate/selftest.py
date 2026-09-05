from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, Mapping, Sequence

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from valdist.core.results import Result

_Z90 = stats.norm.ppf(0.90)  # ~1.2816


class ScenarioDriver:
    """A driver stated in VALUE terms (worst/base/best = value-destroying/-creating ends)."""

    def __init__(self, name: str, worst: float, base: float, best: float) -> None:
        self.name = name
        self.worst = worst
        self.base = base
        self.best = best
        self._zp = np.array([-_Z90, 0.0, _Z90])
        self._vp = np.array([worst, base, best])

    def value(self, z: np.ndarray) -> np.ndarray:
        """Map a shared latent z to driver values via linear interpolation."""
        return np.interp(z, self._zp, self._vp)

    def at(self, level: str) -> float:
        return {"worst": self.worst, "base": self.base, "best": self.best}[level]


class ComonotonicModel:
    """Single-factor comonotonic model: the v1 limiting-case oracle."""

    def __init__(
        self,
        drivers: Sequence[ScenarioDriver],
        valuation: Callable[[Mapping[str, float]], float],
        constants: dict | None = None,
    ) -> None:
        self.drivers = list(drivers)
        self.names = [d.name for d in self.drivers]
        self.valuation = valuation
        self.constants: dict = dict(constants or {})

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(len(X))
        for k in range(len(X)):
            row = {self.names[i]: X[k, i] for i in range(len(self.names))}
            out[k] = self.valuation({**self.constants, **row})
        return out

    def run(
        self,
        n: int = 50_000,
        price: float | None = None,
        seed: int = 0,
        sampling: Literal["mc", "lhs"] = "mc",
        nu: float | None = None,
    ) -> Result:
        """Run the comonotonic model and return a Result-compatible object."""
        from valdist.core.results import Result

        if sampling != "mc":
            raise ValueError(
                f"comonotonic oracle supports sampling='mc' only, got {sampling!r} "
                "(there is one shared latent z; nothing to stratify)."
            )
        if nu is not None:
            raise ValueError(
                f"comonotonic oracle supports nu=None only, got {nu!r} "
                "(perfect rank dependence already; there is no copula to fatten)."
            )

        rng = np.random.default_rng(seed)
        z = rng.standard_normal(n)
        X = np.column_stack([d.value(z) for d in self.drivers])
        value = self._evaluate(X)
        # No type: ignore needed. Result.model is typed as the _ResultModel
        # Protocol (names + run), which ComonotonicModel structurally satisfies.
        # It deliberately doesn't promise `.corr` (there's no dependence source
        # here), and Result.factor_attribution() handles that absence
        # explicitly rather than assuming a Model.
        return Result(self, X, value, price=price, seed=seed)

    def scenario_blend(self) -> dict[str, float]:
        """Deterministic blend with all drivers pinned to each scenario end."""
        out: dict[str, float] = {}
        for level in ("worst", "base", "best"):
            row = {d.name: d.at(level) for d in self.drivers}
            out[level] = self.valuation({**self.constants, **row})
        return out

    def oat_tornado(self) -> list[tuple[str, float, float, float]]:
        """One-at-a-time sensitivity: swing each driver worst→best, others at base."""
        base_row = {d.name: d.base for d in self.drivers}
        rows = []
        for d in self.drivers:
            lo = self.valuation({**self.constants, **base_row, d.name: d.worst})
            hi = self.valuation({**self.constants, **base_row, d.name: d.best})
            rows.append((d.name, lo, hi, hi - lo))
        rows.sort(key=lambda t: abs(t[3]), reverse=True)
        return rows
