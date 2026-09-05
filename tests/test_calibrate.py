"""Calibration coverage tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import stats as sp_stats

from valdist.adapters.registry import _REGISTRY
from valdist.calibrate.coverage import CoverageResult, check_coverage


@pytest.fixture(autouse=True)
def _register_passthrough():
    """Temporarily register a single-driver passthrough valuation for tests."""
    _REGISTRY["_test_passthrough"] = lambda s: s["x"]
    yield
    _REGISTRY.pop("_test_passthrough", None)


def _write_pair(folder: Path, name: str, realised: float) -> None:
    """Write a minimal 1-driver spec + .real file pair."""
    spec = {
        "schema_version": "1.0",
        "name": name,
        "valuation": "_test_passthrough",
        "price": 5.0,
        "seed": 0,
        "n": 50_000,
        "factors": [],
        "drivers": {
            "x": {
                "marginal": {"family": "normal", "p10": 1.0, "p50": 5.0, "p90": 9.0},
                "loadings": {},
            }
        },
        "constants": {},
    }
    (folder / f"{name}.yaml").write_text(yaml.dump(spec))
    (folder / f"{name}.real").write_text(str(realised))


# ---------------------------------------------------------------------------
# Known-coverage synthetic test
# ---------------------------------------------------------------------------


def test_calibration_coverage_known_rate(tmp_path: Path) -> None:
    """Realised values drawn from the model's own distribution give ~80% coverage."""
    Z90 = sp_stats.norm.ppf(0.90)
    sigma = (9.0 - 1.0) / (2.0 * Z90)
    rng = np.random.default_rng(42)
    M = 200
    realised_vals = rng.normal(5.0, sigma, size=M)

    for i, rv in enumerate(realised_vals):
        _write_pair(tmp_path, f"case_{i:03d}", float(rv))

    result = check_coverage(tmp_path, n=50_000, seed=0)

    assert result.n_pairs == M
    assert result.expected_rate == pytest.approx(0.80)
    assert 0.68 <= result.hit_rate <= 0.92, (
        f"hit_rate={result.hit_rate:.3f} outside [0.68, 0.92] - calibration is off"
    )


# ---------------------------------------------------------------------------
# Edge-case / structural tests
# ---------------------------------------------------------------------------


def test_coverage_result_type(tmp_path: Path) -> None:
    """check_coverage returns a CoverageResult dataclass."""
    _write_pair(tmp_path, "one", 5.0)
    result = check_coverage(tmp_path, n=5_000, seed=0)
    assert isinstance(result, CoverageResult)
    assert result.n_pairs == 1
    assert result.n_hits in (0, 1)
    assert 0.0 <= result.hit_rate <= 1.0


def test_coverage_empty_folder(tmp_path: Path) -> None:
    """Folder with no pairs returns n_pairs=0 and nan hit_rate."""
    result = check_coverage(tmp_path)
    assert result.n_pairs == 0
    assert np.isnan(result.hit_rate)


def test_coverage_realised_at_median_is_hit(tmp_path: Path) -> None:
    """Realised = P50 always falls inside [P10, P90]."""
    _write_pair(tmp_path, "mid", 5.0)  # p50=5 for the normal spec
    result = check_coverage(tmp_path, n=20_000, seed=0)
    assert result.n_hits == 1


def test_coverage_realised_far_above_is_miss(tmp_path: Path) -> None:
    """Realised value far above P90 is a miss."""
    _write_pair(tmp_path, "above", 1e9)
    result = check_coverage(tmp_path, n=20_000, seed=0)
    assert result.n_hits == 0


def test_coverage_details_keys(tmp_path: Path) -> None:
    """Each details entry has the required keys."""
    _write_pair(tmp_path, "d", 5.0)
    result = check_coverage(tmp_path, n=5_000, seed=0)
    assert len(result.details) == 1
    d = result.details[0]
    assert "name" in d
    assert "realised" in d
    assert "p10" in d
    assert "p90" in d
    assert "hit" in d


def test_coverage_band_parameter(tmp_path: Path) -> None:
    """Custom band parameter changes expected_rate."""
    _write_pair(tmp_path, "b", 5.0)
    result = check_coverage(tmp_path, band=(0.25, 0.75), n=5_000, seed=0)
    assert result.expected_rate == pytest.approx(0.50)


def test_coverage_skips_spec_without_real_file(tmp_path: Path) -> None:
    """A .yaml without a matching .real is silently skipped."""
    spec = {
        "schema_version": "1.0",
        "name": "orphan",
        "valuation": "_test_passthrough",
        "price": 5.0,
        "seed": 0,
        "n": 1000,
        "factors": [],
        "drivers": {
            "x": {
                "marginal": {"family": "normal", "p10": 1.0, "p50": 5.0, "p90": 9.0},
                "loadings": {},
            }
        },
        "constants": {},
    }
    (tmp_path / "orphan.yaml").write_text(yaml.dump(spec))
    result = check_coverage(tmp_path, n=1_000, seed=0)
    assert result.n_pairs == 0


def test_coverage_finds_yml_pairs_too(tmp_path: Path) -> None:
    """load_spec accepts .yaml AND .yml, but check_coverage globbed only *.yaml.
    A backtest folder written with .yml therefore found ZERO pairs and reported
    a clean nan hit_rate - silently, which is the worst way to get a backtest
    wrong."""
    spec = {
        "schema_version": "1.0",
        "name": "ymlcase",
        "valuation": "_test_passthrough",
        "price": 5.0,
        "seed": 0,
        "n": 2000,
        "factors": [],
        "drivers": {
            "x": {
                "marginal": {"family": "normal", "p10": 1.0, "p50": 5.0, "p90": 9.0},
                "loadings": {},
            }
        },
        "constants": {},
    }
    (tmp_path / "a.yml").write_text(yaml.dump(spec))
    (tmp_path / "a.real").write_text("5.0")

    result = check_coverage(tmp_path, n=2_000, seed=0)
    assert result.n_pairs == 1, "a .yml + .real pair was silently skipped"


def test_coverage_rejects_reversed_band(tmp_path: Path) -> None:
    """band=(0.9, 0.1) produced expected_rate = hi - lo = -0.8 - a negative
    'expected coverage rate', reported without complaint."""
    _write_pair(tmp_path, "a", realised=5.0)

    with pytest.raises(ValueError, match="band"):
        check_coverage(tmp_path, band=(0.9, 0.1), n=2_000, seed=0)
    with pytest.raises(ValueError, match="band"):
        check_coverage(tmp_path, band=(0.5, 0.5), n=2_000, seed=0)


def test_coverage_aggregates_bad_files_instead_of_aborting(tmp_path: Path) -> None:
    """The aggregate-and-report style applies to the backtest too."""
    for i in range(5):
        _write_pair(tmp_path, f"good_{i}", realised=5.0)

    _write_pair(tmp_path, "bad_real", realised=5.0)
    (tmp_path / "bad_real.real").write_text("n/a")  # not a number

    _write_pair(tmp_path, "bad_spec", realised=5.0)
    (tmp_path / "bad_spec.yaml").write_text("{{{ not valid yaml")

    result = check_coverage(tmp_path, n=1_000, seed=0)

    # The good pairs were still scored.
    assert result.n_pairs == 5, f"good pairs were lost: n_pairs={result.n_pairs}"
    assert 0.0 <= result.hit_rate <= 1.0

    # BOTH problems are reported, not just the first one encountered.
    assert len(result.errors) == 2, f"expected both bad files, got {result.errors}"
    blob = " ".join(result.errors)
    assert "bad_real.real" in blob
    assert "bad_spec.yaml" in blob


def test_coverage_errors_is_empty_for_a_clean_folder(tmp_path: Path) -> None:
    """No false positives."""
    _write_pair(tmp_path, "a", realised=5.0)
    result = check_coverage(tmp_path, n=1_000, seed=0)
    assert result.errors == []


def test_cli_calibrate_reports_bad_files_and_exits_nonzero(tmp_path: Path) -> None:
    """The CLI must surface the per-file problems and fail, rather than printing
    a coverage rate computed from a silently-truncated set of pairs."""
    from typer.testing import CliRunner

    from valdist.cli import app

    for i in range(3):
        _write_pair(tmp_path, f"good_{i}", realised=5.0)
    _write_pair(tmp_path, "bad", realised=5.0)
    (tmp_path / "bad.real").write_text("n/a")

    result = CliRunner().invoke(app, ["calibrate", str(tmp_path), "--n", "1000"])

    assert result.exit_code == 1, result.output
    assert "bad.real" in result.output
    # The good pairs are still reported - a partial backtest is still useful.
    assert "3 pair(s)" in result.output
