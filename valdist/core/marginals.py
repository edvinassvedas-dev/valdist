from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats

_Z90 = stats.norm.ppf(0.90)  # ~1.2816: the z-score at the 90th percentile

# The source of truth for which marginal families exist. spec/schema.py derives
# its pydantic Literal from this instead of re-listing it, so the two can't
# drift; tests/test_invariants.py checks that _fit() and ppf() both handle
# every family named here. A family added to only some of these enumerations
# used to pass the schema and then die on ppf()'s "unknown family" assertion
# mid-Monte-Carlo.
FAMILIES = frozenset({"lognormal", "lognormal3", "normal", "triangular", "pert"})

_FAMILIES = FAMILIES  # backwards-compatible alias for existing internal uses

#: Which of two tail regimes each family implements.
#:
#: "censored" - p10/p90 are hard bounds, so no draw is ever more extreme than
#: what the analyst typed. The typed p10/p90 are therefore not the 10th/90th
#: percentiles of what gets drawn - they're the min and max.
#: "extrapolated" - the family genuinely defines tails and draws beyond the
#: stated band.
#:
#: Declared here, next to the families themselves, rather than wherever it's
#: needed - a second copy (in the viewer, say) would be free to drift from
#: what it describes. `tests/test_marginal_view.py` checks this table is
#: exhaustive over FAMILIES and that each label matches what the family's own
#: ppf does at the extremes.
TAIL_REGIME: dict[str, str] = {
    "lognormal": "extrapolated",
    "lognormal3": "extrapolated",
    "normal": "extrapolated",
    "triangular": "censored",
    "pert": "censored",
}


def tail_regime(family: str) -> str:
    """Return "censored" or "extrapolated" for *family*, or raise."""
    try:
        return TAIL_REGIME[family]
    except KeyError:
        raise KeyError(f"unknown family {family!r}; expected one of {sorted(FAMILIES)}") from None


def _external_stacklevel() -> int:
    """stacklevel (as seen by the caller of this function) for the nearest
    frame outside this module.
    """
    frame = sys._getframe(1)
    level = 1
    while frame is not None and frame.f_globals.get("__name__") == __name__:
        frame = frame.f_back
        level += 1
    return level


@dataclass
class Marginal:
    """A fitted one-dimensional marginal distribution for a valuation driver."""

    name: str
    p10: float
    p50: float
    p90: float
    family: str = "lognormal"

    def __post_init__(self) -> None:
        if not (self.p10 <= self.p50 <= self.p90):
            raise ValueError(
                f"{self.name}: need p10 <= p50 <= p90, "
                f"got {self.p10}, {self.p50}, {self.p90}. "
                "Pass sorted quantiles or use Marginal.from_scenarios()."
            )
        if self.family == "lognormal" and self.p10 <= 0:
            raise ValueError(f"{self.name}: lognormal requires strictly positive values.")
        if self.family not in _FAMILIES:
            raise ValueError(f"{self.name}: unknown family {self.family!r}.")
        self._fit()

    @classmethod
    def from_scenarios(
        cls,
        name: str,
        worst: float,
        base: float,
        best: float,
        family: str = "lognormal",
    ) -> Marginal:
        """Build from unordered (worst, base, best) scenario labels."""
        lo, mid, hi = sorted((worst, base, best))
        if mid != base:
            warnings.warn(
                f"{name}: base ({base}) is not the median of "
                f"({worst}, {base}, {best}); sorted to p10/p50/p90 = ({lo}, {mid}, {hi}).",
                stacklevel=2,
            )
        return cls(name, lo, mid, hi, family=family)

    def _fit(self) -> None:
        if self.family == "normal":
            self._mu: float = self.p50
            self._sigma: float = (self.p90 - self.p10) / (2 * _Z90)
            self._warn_if_mismatched()
        elif self.family == "lognormal":
            self._mu = float(np.log(self.p50))
            self._sigma = float((np.log(self.p90) - np.log(self.p10)) / (2 * _Z90))
            self._warn_if_mismatched()
        elif self.family == "lognormal3":
            shift, err = lognormal3_shift(self.p10, self.p50, self.p90)
            if err is not None:
                raise ValueError(f"{self.name}: {err}")
            self._shift: float = shift
            self._mu = float(np.log(self.p50 - shift))
            self._sigma = float((np.log(self.p90 - shift) - np.log(self.p10 - shift)) / (2 * _Z90))
        elif self.family in {"triangular", "pert"}:
            self._lo: float = self.p10
            self._mode: float = self.p50
            self._hi: float = self.p90

    def _warn_if_mismatched(self) -> None:
        msg = quantile_fit_notice(self.family, self.p10, self.p50, self.p90)
        if msg is not None:
            warnings.warn(f"{self.name}: {msg}", stacklevel=_external_stacklevel())

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF: map uniforms in (0, 1) to driver values."""
        u = np.asarray(u, dtype=float)
        if self.family == "normal":
            # A zero-scale fit (p10 == p50 == p90, schema-legal) is a point mass
            # at p50, not a distribution. Without this, `mu + 0 * norm.ppf(0)` is
            # `0 * -inf` = NaN at the endpoints, plus a numpy RuntimeWarning.
            # triangular/pert already return the point mass for span == 0.
            if self._sigma == 0.0:
                return np.full_like(u, self.p50, dtype=float)
            return self._mu + self._sigma * stats.norm.ppf(u)
        if self.family == "lognormal":
            if self._sigma == 0.0:  # same point-mass case: exp(mu + 0*-inf) = exp(nan)
                return np.full_like(u, self.p50, dtype=float)
            return np.exp(self._mu + self._sigma * stats.norm.ppf(u))
        if self.family == "lognormal3":
            return self._shift + np.exp(self._mu + self._sigma * stats.norm.ppf(u))
        if self.family == "triangular":
            span = self._hi - self._lo
            if span == 0:
                return np.full_like(u, self._lo, dtype=float)
            c = (self._mode - self._lo) / span
            return stats.triang.ppf(u, c, loc=self._lo, scale=span)
        if self.family == "pert":
            return _pert_ppf(u, self._lo, self._mode, self._hi)
        raise AssertionError(f"unknown family {self.family!r}")


def lognormal3_shift(p10: float, p50: float, p90: float) -> tuple[float | None, str | None]:
    """Solve (p50-shift)^2 = (p10-shift)(p90-shift) for shift, so that
    Y = X - shift is exactly log-symmetric (Y50 is the geometric mean of
    Y10/Y90). This is linear in shift, not quadratic.
    """
    numerator = p10 * p90 - p50**2
    denom = p10 + p90 - 2 * p50
    if denom == 0:
        # p50 is exactly the arithmetic mean of p10/p90. The shift equation
        # degenerates to "numerator == 0", which only holds when
        # p10 == p50 == p90 (arithmetic mean == geometric mean is otherwise
        # impossible for distinct values), so this is infeasible in every
        # non-degenerate case, not a free pass.
        return None, (
            f"lognormal3 fit is infeasible – p50 ({p50:g}) is exactly the "
            "arithmetic mean of p10/p90, which admits no shift solution; "
            "try lognormal or a different family."
        )
    shift = numerator / denom
    if p10 - shift <= 0:
        return None, (
            f"lognormal3 fit is infeasible (implied shift {shift:.6g} leaves "
            f"p10 - shift = {p10 - shift:.6g} <= 0); try lognormal or a "
            "different family."
        )
    return shift, None


def _pert_ppf(u: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Inverse CDF of the PERT (modified beta) distribution on [a, c] with mode b."""
    span = c - a
    if span == 0:
        return np.full_like(u, a, dtype=float)
    alpha = 1.0 + 4.0 * (b - a) / span
    beta = 1.0 + 4.0 * (c - b) / span
    return a + span * stats.beta.ppf(u, alpha, beta)


# Flag when the realized p10/p90 would be off from the specified ones by more
# than this fraction (lognormal: multiplicative miss; normal: skew as a
# fraction of the half-range). Using a direct error-percentage rather than a
# ratio of log-half-widths, since that ratio is scale-dependent - the same
# ratio can mean a ~5% or a >150% realized miss depending on absolute size.
QUANTILE_MISMATCH_THRESHOLD = 0.20


def _lognormal_fit_notice(p10: float, p50: float, p90: float) -> str | None:
    """Message describing a lognormal quantile-fit mismatch, or None if fine."""
    if p10 <= 0 or p50 <= 0 or p90 <= 0:
        return None
    lo_half = np.log(p50) - np.log(p10)
    hi_half = np.log(p90) - np.log(p50)
    if lo_half <= 0 or hi_half <= 0:
        return None  # p10 == p50 or p50 == p90; degenerate, nothing to compare
    sigma = (lo_half + hi_half) / (2 * _Z90)
    realized_p10 = float(p50 * np.exp(-sigma * _Z90))
    realized_p90 = float(p50 * np.exp(sigma * _Z90))
    relative_error = float(np.exp(abs(lo_half - hi_half) / 2) - 1)
    if relative_error <= QUANTILE_MISMATCH_THRESHOLD:
        return None
    return (
        f"lognormal fit's p50 ({p50:g}) is far from the geometric mean of "
        f"p10/p90 ({p10:g}/{p90:g}) – the realized p10/p90 are "
        f"~{realized_p10:.4g}/~{realized_p90:.4g}, not the specified ones "
        f"({relative_error:.0%} off). Check `valdist worlds` before trusting "
        "the tails, or consider family='lognormal3' if it is feasible for "
        "this triple."
    )


def _normal_fit_notice(p10: float, p50: float, p90: float) -> str | None:
    """Message describing a normal quantile-fit mismatch, or None if fine."""
    half_range = (p90 - p10) / 2
    if half_range <= 0:
        return None
    skew = p50 - (p10 + p90) / 2
    relative_error = abs(skew) / half_range
    if relative_error <= QUANTILE_MISMATCH_THRESHOLD:
        return None
    realized_p10 = p50 - half_range
    realized_p90 = p50 + half_range
    return (
        f"normal fit's p50 ({p50:g}) is not the midpoint of p10/p90 "
        f"({p10:g}/{p90:g}) – the realized p10/p90 are "
        f"~{realized_p10:.4g}/~{realized_p90:.4g}, not the specified ones "
        f"({relative_error:.0%} of the half-range off)."
    )


def quantile_fit_notice(family: str, p10: float, p50: float, p90: float) -> str | None:
    """Non-fatal quantile-fit mismatch message for *family*, or None if fine."""
    if family == "lognormal":
        return _lognormal_fit_notice(p10, p50, p90)
    if family == "normal":
        return _normal_fit_notice(p10, p50, p90)
    return None
