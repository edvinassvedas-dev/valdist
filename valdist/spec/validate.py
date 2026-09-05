"""Cross-field spec validator with aggregated repair digest."""

from __future__ import annotations

import math
from dataclasses import dataclass

from valdist.spec.schema import SpecModel

# Schema versions this build of valdist can actually run. Bump deliberately;
# a spec declaring anything else is rejected rather than silently reinterpreted.
SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0"})


@dataclass(frozen=True)
class ValidationError:
    """A spec violation. The spec will not run correctly until it is fixed."""

    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class SpecError(ValueError):
    """Raised when a spec that `validate()` rejects is about to be run."""

    def __init__(self, errors: list[ValidationError]) -> None:
        self.errors = list(errors)
        super().__init__(str(self))

    def __str__(self) -> str:
        lines = [f"spec has {len(self.errors)} validation error(s):"]
        lines += [f"  {e}" for e in self.errors]
        return "\n".join(lines)


@dataclass(frozen=True)
class ValidationNotice:
    """A non-fatal quality notice. The spec is valid and runnable — but the
    numbers may not be what the analyst meant (a quantile-fit mismatch, a
    suspected percent/fraction mixup).
    """

    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


def validate(spec: SpecModel) -> list[ValidationError]:
    """Return every spec violation as a list of ValidationError."""
    from valdist.adapters.registry import (
        blend_weight_inputs,
        is_registered,
        known_valuations,
        positive_year_inputs,
        required_inputs,
    )
    from valdist.core.marginals import lognormal3_shift

    errors: list[ValidationError] = []
    declared = set(spec.factors)

    for name, driver in spec.drivers.items():
        # Check Σλ² ≤ 1
        sumsq = sum(v**2 for v in driver.loadings.values())
        if sumsq > 1.0 + 1e-8:
            errors.append(
                ValidationError(
                    code="loadings_exceed_one",
                    message=f"driver '{name}': sum(λ²) = {sumsq:.4f} > 1",
                )
            )

        # Check factor references
        unknown = sorted(f for f in driver.loadings if f not in declared)
        if unknown:
            errors.append(
                ValidationError(
                    code="unknown_factor",
                    message=(f"driver '{name}': loadings reference undeclared factors {unknown}"),
                )
            )

        # Check marginal feasibility (would otherwise fail-fast at hydrate())
        m = driver.marginal
        if m.family == "lognormal" and m.p10 <= 0:
            errors.append(
                ValidationError(
                    code="lognormal_nonpositive_p10",
                    message=f"driver '{name}': lognormal requires p10 > 0, got {m.p10:g}",
                )
            )
        elif m.family == "lognormal3":
            _shift, err = lognormal3_shift(m.p10, m.p50, m.p90)
            if err is not None:
                errors.append(
                    ValidationError(code="lognormal3_infeasible", message=f"driver '{name}': {err}")
                )

    # Check blending weights sum to 100.
    #
    # This used to read only spec.weights (the top-level weights: block), but
    # no real spec populates that block - every one puts its weights in
    # constants: (w_ddm/w_affo/w_nav), which is where the adapters actually
    # read them from. So the check passed in tests, which hand-built specs
    # with a weights: block, while being unreachable on every spec a human
    # writes. A blend summing to 90 validated clean and silently renormalised
    # the analyst's mix, because both adapters divide by total_w rather than
    # by 100.
    #
    # Which inputs are weights is domain knowledge, so the adapter declares
    # them (registry blend_weights=) rather than spec/ hardcoding "w_ddm" -
    # same channel and reasoning as requires= and positive_years=. Looked up
    # across constants ∪ weights, the same all-namespace lookup
    # missing_required_input and sampled_years already use.
    declared_weights = blend_weight_inputs(spec.valuation)
    supplied = {**spec.constants, **spec.weights}
    checked: set[str] = set()

    if declared_weights and declared_weights <= supplied.keys():
        # A weight missing entirely is already reported as
        # missing_required_input; summing the partial set would only add a
        # second, more confusing error for the same root cause.
        checked = set(declared_weights)
        total = sum(supplied[k] for k in declared_weights)
        if abs(total - 100.0) > 1e-6:
            errors.append(
                ValidationError(
                    code="weights_not_100",
                    message=(
                        f"blending weights {sorted(declared_weights)} sum to {total:.4f}, "
                        "expected 100 – the adapter normalises by their sum, so a spec that "
                        "misses 100 does not fail, it silently reweights the blend into a mix "
                        "you did not ask for."
                    ),
                )
            )

    # An explicit weights: block is its own weights declaration and must sum
    # to 100 regardless of what the adapter declared, so this is not an elif.
    # Skipping it whenever the adapter declared anything left a hole: a spec
    # whose block names weights the adapter doesn't recognise was checked by
    # neither branch. The `!= checked` guard just avoids reporting the same
    # sum twice.
    if spec.weights and set(spec.weights) != checked:
        total = sum(spec.weights.values())
        if abs(total - 100.0) > 1e-6:
            errors.append(
                ValidationError(
                    code="weights_not_100",
                    message=f"weights sum to {total:.4f}, expected 100",
                )
            )

    # Check for duplicate factor names (would otherwise fail-fast inside
    # FactorModel's constructor at hydrate())
    if len(spec.factors) != len(set(spec.factors)):
        dupes = sorted({f for f in spec.factors if spec.factors.count(f) > 1})
        errors.append(
            ValidationError(
                code="duplicate_factor",
                message=f"factors declared more than once: {dupes}",
            )
        )

    # Check driver/constant/weight names don't collide — Model merges
    # constants and weights into one dict at hydrate() (spec.constants |
    # spec.weights), and then merges driver samples on top of that at
    # evaluation time; any name reused across these namespaces is silently
    # shadowed by dict-merge order.
    driver_names = set(spec.drivers)
    constant_names = set(spec.constants)
    weight_names = set(spec.weights)
    for label, a_names, b_names in (
        ("driver/constant", driver_names, constant_names),
        ("driver/weight", driver_names, weight_names),
        ("constant/weight", constant_names, weight_names),
    ):
        collisions = sorted(a_names & b_names)
        if collisions:
            errors.append(
                ValidationError(
                    code="name_collision",
                    message=f"{label} name(s) collide and would silently shadow: {collisions}",
                )
            )

    # Check the schema version is one we actually understand (it was previously
    # accepted as any string and read by nothing).
    if spec.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(
            ValidationError(
                code="unsupported_schema_version",
                message=(
                    f"schema_version {spec.schema_version!r} is not supported; "
                    f"expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
                ),
            )
        )

    # Check the valuation exists, and that every input it reads is supplied.
    # The valuation is handed drivers ∪ constants ∪ weights as one flat dict
    # (loader.hydrate + Model._evaluate), so a required name may come from any
    # of the three and it does not matter which.
    if not is_registered(spec.valuation):
        errors.append(
            ValidationError(
                code="unknown_valuation",
                message=(
                    f"valuation {spec.valuation!r} is not registered; known: {known_valuations()}"
                ),
            )
        )
    else:
        required = required_inputs(spec.valuation)
        if required is not None:
            supplied = driver_names | constant_names | weight_names
            missing = sorted(required - supplied)
            if missing:
                errors.append(
                    ValidationError(
                        code="missing_required_input",
                        message=(
                            f"valuation {spec.valuation!r} reads input(s) no driver, "
                            f"constant, or weight supplies: {missing}"
                        ),
                    )
                )

        # Projection horizons must be whole years >= 1, and must be constants.
        # Which inputs those are is domain knowledge the adapter declares
        # (registry positive_years=), so this stays domain-agnostic.
        #
        # Two distinct failures, both silent before they were checked:
        #  - a horizon of 0 doesn't merely crash: _ddm_two_stage's stage-1 loop
        #    never runs, so it builds the terminal value off the t=0 dividend -
        #    a different formula returning a plausible-looking number.
        #  - a sampled horizon (supplied as a driver) is a category error: the
        #    adapter's int() truncates a different float every draw, so the
        #    model silently becomes a step function over horizons instead of a
        #    distribution over value (measured: a mixture over {3..9} horizons,
        #    zero diagnostics, zero warnings).
        #
        # The valuation is handed drivers ∪ constants ∪ weights merged flat, so
        # a horizon may legitimately arrive from constants or weights - the
        # same all-namespace lookup missing_required_input already does.
        supplied_scalars = {**spec.constants, **spec.weights}
        for input_name in sorted(positive_year_inputs(spec.valuation) or ()):
            if input_name in spec.drivers:
                errors.append(
                    ValidationError(
                        code="sampled_years",
                        message=(
                            f"driver {input_name!r} is a projection horizon and must be a "
                            "constant, not a sampled driver – a horizon that varies per draw "
                            "makes every draw a different model, and the adapter's int() "
                            "truncation turns it into a step function over the horizon rather "
                            "than a distribution over value."
                        ),
                    )
                )
                continue  # name_collision already reports the driver+constant case
            if input_name not in supplied_scalars:
                continue  # missing_required_input already covers absence
            value = supplied_scalars[input_name]
            if not math.isfinite(value) or value < 1 or value != int(value):
                errors.append(
                    ValidationError(
                        code="nonpositive_years",
                        message=(
                            f"input {input_name!r} is a projection horizon and must be a "
                            f"whole number of years >= 1, got {value!r}"
                        ),
                    )
                )

    # A price at or below zero makes margin-of-safety (value/price - 1)
    # meaningless: negative prices invert its sign, zero makes it infinite.
    if spec.price <= 0:
        errors.append(
            ValidationError(
                code="nonpositive_price",
                message=f"price must be > 0, got {spec.price:g}",
            )
        )

    # inf/nan anywhere in the spec's numbers. Pydantic accepts both (YAML's
    # `.inf` / `.nan` parse straight through), and they poison every statistic
    # downstream without ever raising.
    for label, value in _numeric_spec_values(spec):
        if not math.isfinite(value):
            errors.append(
                ValidationError(
                    code="nonfinite_value",
                    message=f"{label} is {value:g}; specs must contain only finite numbers",
                )
            )

    return errors


def _numeric_spec_values(spec: SpecModel) -> list[tuple[str, float]]:
    """Every number a spec carries, each paired with a human-readable location."""
    values: list[tuple[str, float]] = [("price", spec.price)]
    if spec.nu is not None:
        values.append(("nu", spec.nu))
    for name, driver in spec.drivers.items():
        m = driver.marginal
        values += [
            (f"driver '{name}' p10", m.p10),
            (f"driver '{name}' p50", m.p50),
            (f"driver '{name}' p90", m.p90),
        ]
        values += [
            (f"driver '{name}' loading on '{f}'", v) for f, v in sorted(driver.loadings.items())
        ]
    values += [(f"constant '{k}'", v) for k, v in sorted(spec.constants.items())]
    values += [(f"weight '{k}'", v) for k, v in sorted(spec.weights.items())]
    return values


def check_marginal_fit(spec: SpecModel) -> list[ValidationNotice]:
    """Non-fatal quality notices per driver: quantile-fit mismatches and
    possible percent/fraction mixups (a driver's whole band sitting inside
    (-1, 1) when adapters expect percent).
    """
    from valdist.core.marginals import quantile_fit_notice

    notices: list[ValidationNotice] = []
    for name, driver in spec.drivers.items():
        m = driver.marginal
        msg = quantile_fit_notice(m.family, m.p10, m.p50, m.p90)
        if msg is not None:
            notices.append(
                ValidationNotice(code="quantile_mismatch", message=f"driver '{name}': {msg}")
            )

        # Adapters expect percent (8.0 for 8%) and divide by 100 internally.
        # A driver whose entire band sits inside (-1, 1) might be a fraction
        # entered by mistake (0.08), or might genuinely be a small percent
        # value (terminal growth = 0.5). No fixed threshold can tell those
        # apart, so this is a notice for the analyst to judge, never a hard
        # error.
        if -1.0 < m.p10 and m.p90 < 1.0:
            notices.append(
                ValidationNotice(
                    code="possible_percent_fraction_mixup",
                    message=(
                        f"driver '{name}': p10/p50/p90 = ({m.p10:g}, {m.p50:g}, {m.p90:g}) "
                        "are all inside (-1, 1) – if this driver expects PERCENT "
                        "(e.g. 8.0 for 8%), check it wasn't entered as a fraction "
                        "(e.g. 0.08). If it genuinely has sub-1% typical values, "
                        "ignore this."
                    ),
                )
            )
    return notices
