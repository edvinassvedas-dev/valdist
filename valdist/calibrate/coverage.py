"""Calibration coverage backtest: do realised outcomes fall in P10-P90? Phase 5."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass
class CoverageResult:
    """Aggregate result from a coverage backtest run."""

    n_pairs: int
    n_hits: int
    hit_rate: float
    expected_rate: float
    details: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def check_coverage(
    folder: Path | str,
    band: tuple[float, float] = (0.10, 0.90),
    n: int = 50_000,
    seed: int = 0,
    sampling: Literal["mc", "lhs"] = "mc",
) -> CoverageResult:
    """Run coverage backtest over a folder of (spec, realised) pairs.

    Args:
        folder: directory containing spec + ``.real`` file pairs.
        band: ``(lo_q, hi_q)`` quantiles defining the coverage interval.
              Default ``(0.10, 0.90)`` → expected rate 0.80. Must be ascending.
        n: number of MC draws per spec run.
        seed: RNG seed used for every spec run (ensures reproducibility).
        sampling: ``"mc"`` (default) or ``"lhs"``, forwarded to every spec run.
              Threaded through so a backtest measures the sampler the analyst
              actually intends to use, not always plain MC.

    Returns:
        :class:`CoverageResult` with aggregate and per-pair statistics, plus
        ``errors`` naming every pair that could not be scored.
    """
    from valdist.spec.loader import SPEC_SUFFIXES, hydrate, load_spec

    folder = Path(folder)
    lo_q, hi_q = band
    # expected_rate is hi - lo, so a reversed band silently reported a NEGATIVE
    # "expected coverage rate" (band=(0.9, 0.1) gave -0.8) without complaint.
    if not 0.0 <= lo_q < hi_q <= 1.0:
        raise ValueError(
            f"band must be ascending quantiles within [0, 1], got (lo={lo_q!r}, hi={hi_q!r})"
        )
    expected_rate = hi_q - lo_q
    lo_key = f"p{int(round(lo_q * 100))}"
    hi_key = f"p{int(round(hi_q * 100))}"

    # Every extension load_spec understands, not just *.yaml: a folder written
    # with .yml used to match nothing and report a clean nan hit_rate on zero
    # pairs, which is the worst way to get a backtest wrong.
    spec_paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in SPEC_SUFFIXES)

    details: list[dict] = []
    errors: list[str] = []
    for spec_path in spec_paths:
        real_path = spec_path.with_suffix(".real")
        if not real_path.exists():
            continue

        # Collect, never abort. One typo'd file in a 200-pair folder shouldn't
        # destroy the whole backtest: score what's scoreable and report every
        # problem at once. Each message names its file - "could not convert
        # string to float" on its own tells you nothing about which of 200
        # pairs to go fix.
        raw_realised = real_path.read_text().strip()
        try:
            realised = float(raw_realised)
        except ValueError:
            errors.append(
                f"{real_path}: expected a single number (the realised value), got {raw_realised!r}"
            )
            continue

        try:
            spec = load_spec(spec_path)
            model = hydrate(spec)
        except Exception as exc:  # noqa: BLE001 (reported, not swallowed)
            errors.append(f"{spec_path}: could not load spec – {exc}")
            continue

        result = model.run(n=n, price=None, seed=seed, sampling=sampling, nu=spec.nu)

        p_lo = float(np.quantile(result.value, lo_q))
        p_hi = float(np.quantile(result.value, hi_q))
        hit = bool(p_lo <= realised <= p_hi)

        details.append(
            {
                "name": spec_path.stem,
                "realised": realised,
                lo_key: p_lo,
                hi_key: p_hi,
                "hit": hit,
            }
        )

    n_pairs = len(details)
    n_hits = sum(int(d["hit"]) for d in details)
    hit_rate = n_hits / n_pairs if n_pairs > 0 else math.nan

    return CoverageResult(
        n_pairs=n_pairs,
        n_hits=n_hits,
        hit_rate=hit_rate,
        expected_rate=expected_rate,
        details=details,
        errors=errors,
    )
