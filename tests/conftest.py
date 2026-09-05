"""Shared pytest fixtures and path constants for the valdist test suite."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pytest

from valdist.core.factors import FactorModel
from valdist.core.marginals import Marginal
from valdist.core.model import Model

# Absolute paths so tests are independent of the directory pytest is invoked
# from - a relative "examples/vici.yaml" only resolves when cwd == repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: The suite runs against the synthetic fixture corpus by default, so a public
#: checkout of just this repo can run it. `setdefault` (not an unconditional
#: assignment) lets a real record still be used instead:
#:
#:     pytest -q                                   # the fixture corpus
#:     VALDIST_ANALYSES=/path/to/records pytest -q # a real record
#:
#: This has to run at conftest IMPORT time rather than in a fixture: both
#: readers resolve their root at module import, and test modules import them,
#: so a fixture would set the variable after the value it controls had been read.
os.environ.setdefault("VALDIST_ANALYSES", str(PROJECT_ROOT / "fixtures" / "analyses"))

FIXTURE_CORPUS = PROJECT_ROOT / "fixtures" / "analyses"


def skip_unless_private_record() -> None:
    """Skip a case whose subject is real data rather than the engine itself."""
    import pytest

    from gui.server import ANALYSES_DIR

    if ANALYSES_DIR == FIXTURE_CORPUS:
        pytest.skip(
            "running against the synthetic corpus; this case pins a property of "
            "real data. Run it with VALDIST_ANALYSES pointed at some."
        )


def skip_unless_fixture_corpus() -> None:
    """Skip a case that names a specific fixture by name."""
    import pytest

    from gui.server import ANALYSES_DIR

    if ANALYSES_DIR != FIXTURE_CORPUS:
        pytest.skip(
            f"running against {ANALYSES_DIR}, not the synthetic corpus; this case "
            "names a fixture by name."
        )


def resolve_code(code: str) -> str:
    """Turn a public redaction code (`A12`) into the ticker it stands for."""
    import json

    import pytest

    from gui.server import ANALYSES_DIR

    path = ANALYSES_DIR / "pseudonyms.json"
    if not path.is_file():
        pytest.skip(f"no redaction map at {path} - cannot resolve {code}")
    ticker = json.loads(path.read_text(encoding="utf-8"))["codes"].get(code)
    if ticker is None:
        pytest.skip(f"{path} does not resolve {code}")
    return ticker


VICI_SPEC_PATH = PROJECT_ROOT / "examples" / "vici.yaml"
VICI_FIXTURE_PATH = PROJECT_ROOT / "fixtures" / "2026-06-28-VICI.json"
CALIBRATION_EXAMPLE_DIR = PROJECT_ROOT / "examples" / "calibration"


@pytest.fixture
def two_driver_model() -> Callable[[float], Model]:
    """Factory for a two-driver Model sharing one factor (correlation = loading**2)."""

    def _make(loading: float = 0.8) -> Model:
        fm = FactorModel(["f"], {"a": {"f": loading}, "b": {"f": loading}})
        drivers = [
            Marginal("a", p10=-3.0, p50=0.0, p90=3.0, family="normal"),
            Marginal("b", p10=-3.0, p50=0.0, p90=3.0, family="normal"),
        ]
        return Model(drivers, fm, valuation=lambda v: v["a"] + v["b"])

    return _make
