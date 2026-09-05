"""Per-draw diagnostic flags: a valuation-agnostic channel from adapters to Result."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_collector: ContextVar[Counter[str] | None] = ContextVar("valdist_diagnostics", default=None)


def flag(name: str) -> None:
    """Record that diagnostic *name* fired on the draw being evaluated."""
    counts = _collector.get()
    if counts is not None:
        counts[name] += 1


@contextmanager
def collect_diagnostics() -> Iterator[Counter[str]]:
    """Bind a fresh collector for the duration of the block."""
    counts: Counter[str] = Counter()
    token = _collector.set(counts)
    try:
        yield counts
    finally:
        _collector.reset(token)
