from __future__ import annotations

from functools import cached_property
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

import numpy as np
from scipy import stats


@runtime_checkable
class _ResultModel(Protocol):
    """What a Result actually needs from the model that produced it."""

    names: Sequence[str]

    def run(self, *args: Any, **kwargs: Any) -> Result: ...


class Result:
    """Monte Carlo result from Model.run()."""

    def __init__(
        self,
        model: _ResultModel,
        X: np.ndarray,
        value: np.ndarray,
        price: float | None,
        seed: int,
        sampling: Literal["mc", "lhs"] = "mc",
        nu: float | None = None,
        diagnostics: dict[str, int] | None = None,
    ) -> None:
        self.model = model
        self.X = X
        self.value = value
        self.price = price
        self.seed = seed
        self.sampling = sampling
        self.nu = nu
        self.n = len(value)
        # {flag_name: n_draws} raised by the valuation. Core does not know what
        # any flag means — see core/diagnostics.py.
        self.diagnostics: dict[str, int] = dict(diagnostics or {})

    # --- headline metrics ---------------------------------------------------

    @cached_property
    def p_undervalued(self) -> float:
        """P(intrinsic value > price)."""
        self._need_price()
        return float(np.mean(self.value > self.price))

    @property
    def p_undervalued_stderr(self) -> float:
        """Monte Carlo standard error on p_undervalued (binomial)."""
        p = self.p_undervalued
        return float(np.sqrt(p * (1 - p) / self.n))

    def value_quantiles(self, qs: tuple[float, ...] = (0.10, 0.50, 0.90)) -> dict[float, float]:
        return {q: float(np.quantile(self.value, q)) for q in qs}

    def margin_of_safety(self, qs: tuple[float, ...] = (0.10, 0.50, 0.90)) -> dict[float, float]:
        """Distribution of (value / price − 1)."""
        self._need_price()
        mos = self.value / self.price - 1.0
        return {q: float(np.quantile(mos, q)) for q in qs}

    # --- honesty about uncertainty -------------------------------------------

    def diagnostic_rate(self, name: str) -> float:
        """Fraction of draws on which diagnostic *name* fired (0.0 if never)."""
        return self.diagnostics.get(name, 0) / self.n if self.n else 0.0

    def diagnostic_rate_text(self, name: str) -> str:
        """*name*'s rate as a percentage, never rounded down into "none"."""
        rate = self.diagnostic_rate(name)
        return f"{rate:.2%}" if rate >= 0.0001 else "<0.01%"

    def degenerate(self) -> bool:
        """`p_undervalued` is exactly 0 or 1 - the too-narrow-inputs case."""
        return self.price is not None and bool(self.n) and self.p_undervalued in (0.0, 1.0)

    def warnings(self) -> list[str]:
        """Plain-language warnings that this result should not be read at face value."""
        out: list[str] = []

        for name in sorted(self.diagnostics):
            count = self.diagnostics[name]
            out.append(
                f"{name}: fired on {count:,}/{self.n:,} draws "
                f"({self.diagnostic_rate_text(name)}). "
                "These draws were floored, not computed – the affected tail of "
                "the distribution is not meaningful. Narrow the bands so they "
                "stop crossing, or discount that tail."
            )

        if self.degenerate():
            p = self.p_undervalued
            verdict = "above" if p == 1.0 else "below"
            out.append(
                f"p_undervalued = {p:.0f} exactly: every draw came out {verdict} "
                "the price, so the reported ±0.0000 standard error is an "
                "artefact, not confidence. Treat this as a too-narrow-inputs "
                "warning, not a result."
            )

        return out

    def summary(self) -> dict:
        """Headline numbers as a plain dict, ready for JSON.

        The key set does not depend on whether a price was supplied: the
        price-dependent entries are present either way, set to None when there is
        no price, so a consumer never has to probe for them.
        """
        qs = (0.10, 0.25, 0.50, 0.90)
        vq = self.value_quantiles(qs)
        out: dict = {
            "n": self.n,
            "value_p10": vq[0.10],
            # P25 is the conservative value anchor, not a milder P10. Graham's
            # approach is a conservative value estimate plus a further discount,
            # so the conservative estimate is itself a low quantile. Reported at
            # P25 rather than deeper because below roughly P10 the tail is a
            # property of the marginal family (censored, extrapolated, or
            # floored) rather than of the research.
            "value_p25": vq[0.25],
            "value_p50": vq[0.50],
            "value_p90": vq[0.90],
            # Free, and the one number that says whether the distribution is
            # tight or fat: 0.85 for A58 vs. 0.57 for A15, two specs whose
            # median margins look similar but whose downsides don't.
            "p25_p50_ratio": (vq[0.25] / vq[0.50]) if vq[0.50] else None,
            "price": self.price,
            "p_undervalued": None,
            "p_undervalued_stderr": None,
            "mos_p10": None,
            "mos_p25": None,
            "mos_p50": None,
            "mos_p90": None,
        }
        if self.price is not None:
            mos = self.margin_of_safety(qs)
            out.update(
                p_undervalued=self.p_undervalued,
                p_undervalued_stderr=self.p_undervalued_stderr,
                mos_p10=mos[0.10],
                mos_p25=mos[0.25],
                mos_p50=mos[0.50],
                mos_p90=mos[0.90],
            )
        return out

    # --- sensitivity --------------------------------------------------------

    def tornado(self) -> list[tuple[str, float]]:
        """Spearman rank correlation of each driver against the valuation.

        Tells you which assumption drives the spread - but rank correlation is
        scale-free, so it does NOT say by how much. Use `Model.driver_swings()`
        for magnitude.

        A constant driver (p10 == p50 == p90, which the schema allows) has a
        zero-variance column and no defined correlation; it is reported as 0.0.
        """
        ranks_out = stats.rankdata(self.value)
        out_is_flat = bool(np.ptp(ranks_out) == 0)
        rows = []
        for i, name in enumerate(self.model.names):
            ranks_in = stats.rankdata(self.X[:, i])
            if out_is_flat or np.ptp(ranks_in) == 0:
                r = 0.0
            else:
                r = float(np.corrcoef(ranks_in, ranks_out)[0, 1])
            rows.append((name, r))
        rows.sort(key=lambda t: abs(t[1]), reverse=True)
        return rows

    # --- Phase 4 outputs ----------------------------------------------------

    def factor_attribution(self) -> dict[str, float]:
        """Normalised squared rank-correlation share per factor + idiosyncratic."""
        from valdist.core.factors import FactorModel

        # Uses getattr rather than self.model.corr: a Result can legitimately
        # carry a model with no corr at all (calibrate.selftest.ComonotonicModel
        # has one shared latent z and no dependence source). That should raise
        # the same TypeError every other unsupported source gets, not a bare
        # AttributeError from a different line.
        corr = getattr(self.model, "corr", None)
        if not isinstance(corr, FactorModel):
            raise TypeError(
                f"factor_attribution requires a FactorModel dependence source; "
                f"got {type(corr).__name__}"
            )

        fm = corr
        names = self.model.names
        n = self.n

        # Recover latent z-scores via empirical CDF → probit
        z = np.column_stack(
            [stats.norm.ppf(stats.rankdata(self.X[:, i]) / (n + 1)) for i in range(len(names))]
        )

        # Build loading matrix Λ: (d, k)
        Lambda = fm.loadings_matrix(names)

        # Factor proxies: (n, k) — one column per factor
        G = z @ Lambda  # g_j = Σ_i λ_ij * z_i

        # Squared Spearman rank correlation of each factor proxy with value
        rho_sq = np.array(
            [stats.spearmanr(G[:, j], self.value).statistic ** 2 for j in range(len(fm.factors))]
        )

        total_factor = float(rho_sq.sum())
        idio = max(0.0, 1.0 - total_factor)
        denom = total_factor + idio  # = max(total_factor, 1.0)

        result: dict[str, float] = {
            name: float(rho_sq[j]) / denom for j, name in enumerate(fm.factors)
        }
        result["idiosyncratic"] = idio / denom
        return result

    def worlds(self, qs: tuple[float, ...] = (0.1, 0.5, 0.9)) -> list[dict]:
        """Driver values at the sample closest to each output quantile."""
        scenarios = []
        for q in sorted(qs):
            target = float(np.quantile(self.value, q))
            idx = int(np.argmin(np.abs(self.value - target)))
            row: dict = {"quantile": q, "value": float(self.value[idx])}
            for i, name in enumerate(self.model.names):
                row[name] = float(self.X[idx, i])
            scenarios.append(row)
        return scenarios

    # --- discipline checks --------------------------------------------------

    def convergence(self) -> dict:
        """Re-run at 2N with the same seed and report drift in key statistics."""
        big = self.model.run(
            n=2 * self.n,
            price=self.price,
            seed=self.seed,
            sampling=self.sampling,
            nu=self.nu,
        )
        out: dict = {
            "value_p50_N": float(np.median(self.value)),
            "value_p50_2N": float(np.median(big.value)),
        }
        if self.price is not None:
            out["p_undervalued_N"] = self.p_undervalued
            out["p_undervalued_2N"] = big.p_undervalued
        return out

    def _need_price(self) -> None:
        if self.price is None:
            raise ValueError("price not set; pass price=... to Model.run().")
