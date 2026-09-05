"""Valuation-callable registry: name → callable (+ the inputs it reads)."""

from __future__ import annotations

from typing import Callable, Iterable

_REGISTRY: dict[str, Callable] = {}
_REQUIRES: dict[str, frozenset[str]] = {}
_POSITIVE_YEARS: dict[str, frozenset[str]] = {}
_BLEND_WEIGHTS: dict[str, frozenset[str]] = {}


def register(
    name: str,
    requires: Iterable[str] | None = None,
    positive_years: Iterable[str] | None = None,
    blend_weights: Iterable[str] | None = None,
) -> Callable:
    """Decorator that registers a valuation callable under *name*."""

    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn

        # A re-registration replaces the adapter, declarations included; it
        # doesn't merge with whatever the previous one declared. A stale entry
        # left behind would let required_inputs()/positive_year_inputs() report
        # constraints from a definition that no longer exists, and validate()
        # would enforce them against the wrong adapter.
        _REQUIRES.pop(name, None)
        _POSITIVE_YEARS.pop(name, None)
        _BLEND_WEIGHTS.pop(name, None)
        if requires is not None:
            _REQUIRES[name] = frozenset(requires)
        if positive_years is not None:
            _POSITIVE_YEARS[name] = frozenset(positive_years)
        if blend_weights is not None:
            _BLEND_WEIGHTS[name] = frozenset(blend_weights)
        return fn

    return decorator


def get_valuation(name: str) -> Callable:
    """Return the callable registered under *name*, or raise KeyError."""
    if name not in _REGISTRY:
        known = sorted(_REGISTRY)
        raise KeyError(f"valuation '{name}' not registered; known: {known}")
    return _REGISTRY[name]


def is_registered(name: str) -> bool:
    """True if *name* has a registered valuation."""
    return name in _REGISTRY


def known_valuations() -> list[str]:
    """Sorted names of every registered valuation."""
    return sorted(_REGISTRY)


def required_inputs(name: str) -> frozenset[str] | None:
    """Sample-dict keys *name*'s valuation reads, or None if it didn't declare."""
    return _REQUIRES.get(name)


def positive_year_inputs(name: str) -> frozenset[str] | None:
    """Inputs of *name* that are projection horizons (must be int >= 1), or
    None if it didn't declare any."""
    return _POSITIVE_YEARS.get(name)


def blend_weight_inputs(name: str) -> frozenset[str] | None:
    """Inputs of *name* that are method-blending weights (must sum to 100), or
    None if it didn't declare any."""
    return _BLEND_WEIGHTS.get(name)
