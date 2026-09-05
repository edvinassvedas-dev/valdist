from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from valdist.core.sampler import lhs_normal


class FactorModel:
    """Factor loadings Λ and derived idiosyncratic shares ψ.

    Parameters
    ----------
    factors:
        Ordered list of named common factors (e.g. ["rate", "fundamentals"]).
    loadings:
        ``{driver_name: {factor_name: loading_value}}``.  Only non-zero
        loadings need to be listed; missing entries default to 0.
        Each driver must satisfy Σ_j λ_ij² ≤ 1 (raises ValueError otherwise).
    """

    def __init__(
        self,
        factors: list[str],
        loadings: dict[str, dict[str, float]],
    ) -> None:
        if len(factors) != len(set(factors)):
            raise ValueError(f"factor names must be unique, got {factors}")
        self.factors: list[str] = list(factors)
        self.loadings: dict[str, dict[str, float]] = {
            driver: dict(lam) for driver, lam in loadings.items()
        }
        # Validate factor references
        unknown = {
            f for driver_lam in loadings.values() for f in driver_lam if f not in self.factors
        }
        if unknown:
            raise ValueError(f"loadings reference undeclared factors: {sorted(unknown)}")
        # Validate unit-variance constraint: Σ_j λ_ij² ≤ 1 for each driver
        violations = [
            (driver, sumsq)
            for driver, lam in self.loadings.items()
            if (sumsq := sum(v**2 for v in lam.values())) > 1.0 + 1e-8
        ]
        if violations:
            parts = ", ".join(f"{n}: sum(λ²)={v:.4f}" for n, v in violations)
            raise ValueError(f"sum of squared loadings > 1 for: {parts}")

    # --- internal array construction ----------------------------------------

    def _build_arrays(self, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return (Λ, ψ) for the given ordered driver list."""
        k = len(self.factors)
        d = len(names)
        factor_idx = {f: j for j, f in enumerate(self.factors)}

        Lambda = np.zeros((d, k))
        for i, name in enumerate(names):
            for factor, lam in self.loadings.get(name, {}).items():
                Lambda[i, factor_idx[factor]] = lam

        row_sumsq = np.sum(Lambda**2, axis=1)
        violations = [(names[i], row_sumsq[i]) for i in range(d) if row_sumsq[i] > 1.0 + 1e-8]
        if violations:
            parts = ", ".join(f"{n}: sum(λ²)={v:.4f}" for n, v in violations)
            raise ValueError(f"sum of squared loadings > 1 for: {parts}")

        psi = np.sqrt(np.maximum(0.0, 1.0 - row_sumsq))
        return Lambda, psi

    def loadings_matrix(self, names: Sequence[str]) -> np.ndarray:
        """Return Λ (d, k) for the given ordered driver list: row i is driver
        i's loadings on the k common factors.
        """
        Lambda, _psi = self._build_arrays(names)
        return Lambda

    def psi(self, names: Sequence[str]) -> np.ndarray:
        """Return ψ (d,) for the given ordered driver list: driver i's
        idiosyncratic standard deviation √(1 − Σ_j λ_ij²).
        """
        _lambda, psi = self._build_arrays(names)
        return psi

    # --- public interface ----------------------------------------------------

    def matrix(self, names: Sequence[str]) -> np.ndarray:
        """Return the copula correlation matrix Σ = ΛΛ' + diag(ψ²)."""
        Lambda, psi = self._build_arrays(names)
        return Lambda @ Lambda.T + np.diag(psi**2)

    def draw_z(
        self,
        names: Sequence[str],
        n: int,
        rng: np.random.Generator,
        sampling: Literal["mc", "lhs"] = "mc",
    ) -> np.ndarray:
        """Draw n correlated standard-normal vectors via direct factor sampling."""
        Lambda, psi = self._build_arrays(names)
        k = Lambda.shape[1]
        d = len(names)
        if sampling == "mc":
            f = rng.standard_normal((n, k))  # common factor draws: (n, k)
            e = rng.standard_normal((n, d))  # idiosyncratic draws: (n, d)
        elif sampling == "lhs":
            joint = lhs_normal(n, k + d, rng)
            f, e = joint[:, :k], joint[:, k:]
        else:
            raise ValueError(f"unknown sampling {sampling!r}; expected 'mc' or 'lhs'")
        return f @ Lambda.T + e * psi  # (n, d)
