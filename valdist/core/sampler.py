from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import qmc


def lhs_normal(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Return an (n, d) matrix of stratified standard-normal draws."""
    sampler = qmc.LatinHypercube(d=d, seed=rng)
    u = sampler.random(n)
    return stats.norm.ppf(u)
