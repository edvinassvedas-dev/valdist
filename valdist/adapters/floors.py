"""Numeric floors shared by the valuation adapters, and the flags they raise."""

from __future__ import annotations

from valdist.core.diagnostics import flag

# Smallest divisor allowed in a Gordon-growth terminal value or an EPV
# perpetuity. Not tunable per-adapter on purpose: a floored draw is garbage at
# any floor, so the exact value only decides how loudly it is garbage. What
# matters is that it is counted.
MIN_DIVISOR = 1e-4


def floored(value: float, flag_name: str, floor: float = MIN_DIVISOR) -> float:
    """Floor a divisor at MIN_DIVISOR and flag the draw if the floor bit.

    The flag fires only when the floor actually changes the result, so a spec
    whose bands never cross stays completely silent. NaN is caught too: the
    test is `not (value >= floor)` rather than `value < floor`, because every
    IEEE 754 comparison against NaN is False.

    A floored draw is not a pessimistic scenario. It is an arithmetic
    placeholder that can be thousands of times too large, so it is counted and
    surfaced rather than blended in quietly.
    """
    if not (value >= floor):  # NaN-safe: False for NaN, so NaN is floored
        flag(flag_name)
        return floor
    return value


def require_positive_years(years: int, name: str, adapter: str) -> int:
    """Reject a non-positive projection horizon, loudly."""
    years = int(years)
    if years < 1:
        raise ValueError(
            f"{adapter}: {name} must be >= 1, got {years}. A projection horizon "
            "below one year has no stage-1 period, so the terminal value would "
            "be built off the t=0 cash flow – a different formula, not a shorter one."
        )
    return years


def require_nonnegative_years(years: int, name: str, adapter: str) -> int:
    """Reject a negative horizon, loudly - zero is allowed."""
    years = int(years)
    if years < 0:
        raise ValueError(
            f"{adapter}: {name} must be >= 0, got {years}. A negative horizon "
            "would discount the redemption leg forward rather than back, which "
            "inflates it – zero is legal here (already callable), negative is not."
        )
    return years


def clamped(value: float, lo: float, hi: float, flag_name: str) -> float:
    """Clamp a value into [lo, hi] and flag the draw if the clamp bit.

    `floored()`'s idiom applied to a two-sided bound, for inputs that are
    probabilities or percentages and have no meaning outside their range.
    Clamps rather than raises: one wide draw must not abort a 50,000-draw run.
    NaN-safe by the same construction as `floored()`.
    """
    if not (value >= lo):  # NaN-safe: False for NaN, so NaN clamps to lo
        flag(flag_name)
        return lo
    if not (value <= hi):
        flag(flag_name)
        return hi
    return value
