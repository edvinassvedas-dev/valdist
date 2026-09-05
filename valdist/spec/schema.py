"""Declarative spec schema (pydantic)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from valdist.core.marginals import FAMILIES

# Derived from core.marginals.FAMILIES, never re-listed: the two used to be
# independent enumerations nothing forced to agree, so adding a family could
# half-land - accepted by the schema, then dying on ppf()'s "unknown family"
# assertion deep inside the Monte Carlo loop. Kept as a Literal since pydantic
# needs a static type; sorted() makes the order deterministic.
FamilyName = Literal[tuple(sorted(FAMILIES))]  # type: ignore[valid-type]


class MarginalSpec(BaseModel):
    family: FamilyName
    p10: float
    p50: float
    p90: float

    @model_validator(mode="after")
    def _check_order(self) -> "MarginalSpec":
        if not (self.p10 <= self.p50 <= self.p90):
            raise ValueError(f"require p10 ≤ p50 ≤ p90, got {self.p10}/{self.p50}/{self.p90}")
        return self


class DriverSpec(BaseModel):
    marginal: MarginalSpec
    loadings: dict[str, float] = Field(default_factory=dict)


class SpecModel(BaseModel):
    """Top-level spec document."""

    schema_version: str = "1.0"
    name: str
    valuation: str
    price: float
    seed: int = 0
    n: int = Field(default=50_000, gt=0)
    # t-copula degrees of freedom. None (default/omitted) = Gaussian copula,
    # today's exact behavior. Set for genuine joint tail dependence beyond
    # what the Gaussian copula's asymptotic tail independence provides;
    # see valdist/core/model.py.
    nu: float | None = None
    factors: list[str] = Field(default_factory=list)
    drivers: dict[str, DriverSpec]
    constants: dict[str, float] = Field(default_factory=dict)
    # Blending weights (if present, must sum to 100; checked by validate()).
    weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_nu(self) -> "SpecModel":
        if self.nu is not None and self.nu <= 0:
            raise ValueError(f"nu must be > 0 if set, got {self.nu}")
        return self
