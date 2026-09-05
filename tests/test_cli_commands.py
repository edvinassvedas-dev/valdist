"""CLI command bodies: plot, calibrate, and the option guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import CALIBRATION_EXAMPLE_DIR, VICI_SPEC_PATH
from typer.testing import CliRunner

from valdist.cli import app

runner = CliRunner()

VICI = str(VICI_SPEC_PATH)
CALIBRATION = str(CALIBRATION_EXAMPLE_DIR)


# --------------------------------------------------------------------------- #
# run: option guards + sampling
# --------------------------------------------------------------------------- #


def test_run_rejects_unknown_sampling() -> None:
    result = runner.invoke(app, ["run", VICI, "--sampling", "bogus"])
    assert result.exit_code == 1
    assert "must be 'mc' or 'lhs'" in result.output


def test_run_accepts_lhs_sampling() -> None:
    result = runner.invoke(app, ["run", VICI, "--sampling", "lhs"])
    assert result.exit_code == 0, result.output
    assert "P(undervalued)" in result.output


# --------------------------------------------------------------------------- #
# validate: the non-fatal notice branch
# --------------------------------------------------------------------------- #


def test_validate_prints_quantile_fit_notice(tmp_path: Path) -> None:
    """A spec that is valid but whose lognormal fit misses its own p10/p90 must
    still print a notice, not a bare OK.
    """
    import textwrap

    spec = tmp_path / "notice.yaml"
    spec.write_text(
        textwrap.dedent("""
            schema_version: "1.0"
            name: "notice-spec"
            valuation: reit_v2
            price: 30.0
            n: 100
            factors: []
            drivers:
              # p10 pinned near zero -> the lognormal fit misses its own tails
              cap_rate:
                marginal: {family: lognormal, p10: 0.5, p50: 6.0, p90: 11.0}
                loadings: {}
              cost_of_equity:
                marginal: {family: normal, p10: 8.0, p50: 9.0, p90: 10.0}
                loadings: {}
              div_growth:
                marginal: {family: normal, p10: 2.0, p50: 3.0, p90: 4.0}
                loadings: {}
              div_terminal:
                marginal: {family: normal, p10: 1.0, p50: 1.5, p90: 2.0}
                loadings: {}
              affo_growth:
                marginal: {family: normal, p10: 2.0, p50: 3.0, p90: 4.0}
                loadings: {}
              affo_terminal:
                marginal: {family: normal, p10: 1.0, p50: 1.5, p90: 2.0}
                loadings: {}
            constants:
              shares: 1090.0
              dps: 1.7
              ddm_stage1_years: 10
              affo: 2680.0
              affo_years: 10
              noi: 2400.0
              nav_debt: 17000.0
              nav_other: 420.0
            weights: {w_ddm: 40.0, w_affo: 40.0, w_nav: 20.0}
            """)
    )
    result = runner.invoke(app, ["validate", str(spec)])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output
    assert "NOTICE" in result.output
    assert "quantile_mismatch" in result.output


# --------------------------------------------------------------------------- #
# plot
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["tornado", "distribution"])
def test_plot_writes_png(tmp_path: Path, kind: str) -> None:
    out = tmp_path / f"{kind}.png"
    result = runner.invoke(app, ["plot", VICI, "--kind", kind, "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists() and out.stat().st_size > 1000
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_dark_theme(tmp_path: Path) -> None:
    out = tmp_path / "dark.png"
    result = runner.invoke(
        app, ["plot", VICI, "--kind", "tornado", "--out", str(out), "--theme", "dark"]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_plot_rejects_bad_kind(tmp_path: Path) -> None:
    result = runner.invoke(app, ["plot", VICI, "--kind", "pie", "--out", str(tmp_path / "x.png")])
    assert result.exit_code == 1
    assert "must be 'tornado' or 'distribution'" in result.output


def test_plot_rejects_bad_theme(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "plot",
            VICI,
            "--kind",
            "tornado",
            "--out",
            str(tmp_path / "x.png"),
            "--theme",
            "neon",
        ],
    )
    assert result.exit_code == 1
    assert "must be 'light' or 'dark'" in result.output


# --------------------------------------------------------------------------- #
# calibrate
# --------------------------------------------------------------------------- #


def test_calibrate_reports_coverage() -> None:
    result = runner.invoke(app, ["calibrate", CALIBRATION, "--n", "2000"])
    assert result.exit_code == 0, result.output
    assert "Coverage check" in result.output
    assert "Expected rate" in result.output
    assert "Observed rate" in result.output


def test_calibrate_custom_band() -> None:
    result = runner.invoke(
        app, ["calibrate", CALIBRATION, "--lo", "0.25", "--hi", "0.75", "--n", "2000"]
    )
    assert result.exit_code == 0, result.output
    assert "P25" in result.output and "P75" in result.output
    assert "50%" in result.output  # expected rate for a P25-P75 band


def test_calibrate_empty_folder_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["calibrate", str(tmp_path)])
    assert result.exit_code == 1
    assert "No (spec, realised) pairs" in result.output


# --------------------------------------------------------------------------- #
# --sampling must reach EVERY command that draws samples (see test_invariants.py
# for the structural guard; these are the behavioural half).
# --------------------------------------------------------------------------- #


def test_calibrate_accepts_lhs_sampling() -> None:
    result = runner.invoke(app, ["calibrate", CALIBRATION, "--n", "2000", "--sampling", "lhs"])
    assert result.exit_code == 0, result.output
    assert "Coverage check" in result.output


def test_plot_accepts_lhs_sampling(tmp_path: Path) -> None:
    out = tmp_path / "lhs.png"
    result = runner.invoke(
        app, ["plot", VICI, "--kind", "tornado", "--out", str(out), "--sampling", "lhs"]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


@pytest.mark.parametrize("command", ["plot", "calibrate"])
def test_new_sampling_flags_reject_a_bogus_value(tmp_path: Path, command: str) -> None:
    args = (
        ["plot", VICI, "--kind", "tornado", "--out", str(tmp_path / "x.png")]
        if command == "plot"
        else ["calibrate", CALIBRATION, "--n", "2000"]
    )
    result = runner.invoke(app, [*args, "--sampling", "bogus"])
    assert result.exit_code == 1
    assert "must be 'mc' or 'lhs'" in result.output


def test_plot_sampling_actually_changes_the_draws(tmp_path: Path) -> None:
    """Accepting a flag and dropping it is worse than not having it. `plot` used
    to hardcode plain MC, so `valdist plot --sampling lhs` silently charted the
    same draws as the default - prove the flag reaches Model.run()."""
    from valdist.spec.loader import hydrate, load_spec

    spec = load_spec(Path(VICI))
    model = hydrate(spec)
    mc = model.run(n=2000, price=spec.price, seed=spec.seed, sampling="mc", nu=spec.nu)
    lhs = model.run(n=2000, price=spec.price, seed=spec.seed, sampling="lhs", nu=spec.nu)
    assert not (mc.value == lhs.value).all(), "mc and lhs must differ, else this proves nothing"


# --------------------------------------------------------------------------- #
# worlds (JSON surface is exercised via report elsewhere; assert the CLI shape)
# --------------------------------------------------------------------------- #


def test_worlds_prints_a_row_per_quantile() -> None:
    result = runner.invoke(app, ["worlds", VICI])
    assert result.exit_code == 0, result.output
    for label in ("P10", "P50", "P90"):
        assert label in result.output


def test_worlds_bad_spec_exits_nonzero(tmp_path: Path) -> None:
    """worlds must exit non-zero on an invalid spec, printing the digest."""
    bad = """\
schema_version: "1.0"
name: bad
valuation: test_sum_fa
price: 1.0
factors: [rate]
drivers:
  overloaded:
    marginal: {family: normal, p10: 1.0, p50: 2.0, p90: 3.0}
    loadings: {rate: 1.1}
"""
    spec_file = tmp_path / "bad.yaml"
    spec_file.write_text(bad)

    result = runner.invoke(app, ["worlds", str(spec_file)])
    assert result.exit_code != 0
    assert "loadings_exceed_one" in result.output


def test_golden_run_reports_no_warnings() -> None:
    """The reference spec's bands do not cross, so the CLI must print a clean
    report with no WARNINGS block (regression guard on the diagnostics channel).
    """
    result = runner.invoke(app, ["run", VICI])
    assert result.exit_code == 0, result.output
    assert "WARNING" not in result.output.upper()
    assert "0.9312" in result.output


def test_report_json_carries_warnings_and_diagnostics_keys() -> None:
    from valdist import report
    from valdist.spec.loader import hydrate, load_spec

    spec = load_spec(Path(VICI))
    result = hydrate(spec).run(n=2000, price=spec.price, seed=spec.seed)
    payload = json.loads(report.to_json(result))
    assert payload["warnings"] == []
    assert payload["diagnostics"] == {}
