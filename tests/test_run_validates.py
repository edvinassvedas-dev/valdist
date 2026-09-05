"""Validation given teeth on the RUN path: an invalid spec must not run."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from valdist.cli import app

runner = CliRunner()

# shares: .inf - validate() catches this as `nonfinite_value`.
_NONFINITE_SPEC = textwrap.dedent("""
    schema_version: "1.0"
    name: nonfinite
    valuation: equity_v2
    price: 25.0
    seed: 0
    n: 2000
    factors: []
    drivers:
      fcf_growth:
        marginal: {family: normal, p10: 2.0, p50: 4.0, p90: 6.0}
        loadings: {}
      terminal_growth:
        marginal: {family: normal, p10: 1.0, p50: 1.5, p90: 2.0}
        loadings: {}
      wacc:
        marginal: {family: normal, p10: 7.0, p50: 8.5, p90: 10.0}
        loadings: {}
    constants:
      shares: .inf
      debt: 1500.0
      cash: 300.0
      fcff: 500.0
      normalized_earnings: 480.0
      years: 7
      relative_value_per_share: 30.0
    weights: {w_dcf: 45.0, w_epv: 35.0, w_relative: 20.0}
    """)

# Two planted errors, so the digest must report BOTH - never fail-fast.
_TWO_ERROR_SPEC = textwrap.dedent("""
    schema_version: "9.9"
    name: two-errors
    valuation: equity_v2
    price: 25.0
    seed: 0
    n: 2000
    factors: []
    drivers:
      fcf_growth:
        marginal: {family: normal, p10: 2.0, p50: 4.0, p90: 6.0}
        loadings: {}
      terminal_growth:
        marginal: {family: normal, p10: 1.0, p50: 1.5, p90: 2.0}
        loadings: {}
      wacc:
        marginal: {family: normal, p10: 7.0, p50: 8.5, p90: 10.0}
        loadings: {}
    constants:
      shares: 200.0
      debt: 1500.0
      cash: 300.0
      fcff: 500.0
      normalized_earnings: 480.0
      years: 7
      relative_value_per_share: 30.0
    weights: {w_dcf: 60.0, w_epv: 60.0, w_relative: 20.0}
    """)


def _write(tmp_path: Path, text: str, name: str = "spec.yaml") -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


# --------------------------------------------------------------------------- #
# hydrate() is the gate
# --------------------------------------------------------------------------- #


def test_hydrate_refuses_an_invalid_spec(tmp_path: Path) -> None:
    from valdist.spec.loader import hydrate, load_spec
    from valdist.spec.validate import SpecError

    spec = load_spec(_write(tmp_path, _NONFINITE_SPEC))
    with pytest.raises(SpecError) as exc:
        hydrate(spec)
    assert "nonfinite_value" in str(exc.value)


def test_hydrate_raises_the_whole_digest_not_the_first_error(tmp_path: Path) -> None:
    """Aggregate, never fail-fast. Both planted errors must appear."""
    from valdist.spec.loader import hydrate, load_spec
    from valdist.spec.validate import SpecError

    spec = load_spec(_write(tmp_path, _TWO_ERROR_SPEC))
    with pytest.raises(SpecError) as exc:
        hydrate(spec)

    codes = {e.code for e in exc.value.errors}
    assert {"unsupported_schema_version", "weights_not_100"} <= codes, codes
    rendered = str(exc.value)
    assert "unsupported_schema_version" in rendered
    assert "weights_not_100" in rendered


def test_spec_error_renders_a_header_line_with_the_error_count(tmp_path: Path) -> None:
    """The digest's first line - "spec has N validation error(s):" - is what tells
    the analyst this is an aggregate and how much is wrong. Nothing tested
    it: every other test matches on individual codes, so a mutant that turned
    `lines += [...]` into `lines = [...]` deleted the header and stayed green.
    """
    from valdist.spec.loader import hydrate, load_spec
    from valdist.spec.validate import SpecError

    spec = load_spec(_write(tmp_path, _TWO_ERROR_SPEC))
    with pytest.raises(SpecError) as exc:
        hydrate(spec)

    lines = str(exc.value).splitlines()
    n = len(exc.value.errors)
    assert n > 1, "this spec must plant several errors for the header to mean anything"
    assert lines[0] == f"spec has {n} validation error(s):"
    assert len(lines) == n + 1, "one header line, then one line per error"


def test_spec_error_is_a_value_error(tmp_path: Path) -> None:
    """SpecError subclasses ValueError so existing `except ValueError` still catches."""
    from valdist.spec.loader import hydrate, load_spec
    from valdist.spec.validate import SpecError

    assert issubclass(SpecError, ValueError)
    spec = load_spec(_write(tmp_path, _NONFINITE_SPEC))
    with pytest.raises(ValueError):
        hydrate(spec)


def test_hydrate_escape_hatch(tmp_path: Path) -> None:
    """hydrate(validate=False) still builds a knowingly-broken Model - the
    library must not make the invalid case unreachable, only non-default."""
    from valdist.spec.loader import hydrate, load_spec

    spec = load_spec(_write(tmp_path, _NONFINITE_SPEC))
    model = hydrate(spec, validate=False)
    assert model.names == ["fcf_growth", "terminal_growth", "wacc"]


# --------------------------------------------------------------------------- #
# The public entry points inherit the gate
# --------------------------------------------------------------------------- #


def test_valdist_run_refuses_an_invalid_spec(tmp_path: Path) -> None:
    """The repro: this used to return 6.0/share with a misdirecting warning."""
    import valdist
    from valdist.spec.validate import SpecError

    with pytest.raises(SpecError, match="nonfinite_value"):
        valdist.run(_write(tmp_path, _NONFINITE_SPEC))


@pytest.mark.parametrize("command", ["run", "worlds"])
def test_cli_refuses_an_invalid_spec(tmp_path: Path, command: str) -> None:
    """The CLI must print the digest and exit 1 - not dump a traceback."""
    path = _write(tmp_path, _NONFINITE_SPEC)
    result = runner.invoke(app, [command, str(path)])

    assert result.exit_code == 1, result.output
    assert "nonfinite_value" in result.output
    assert "Traceback" not in result.output


def test_cli_plot_refuses_an_invalid_spec(tmp_path: Path) -> None:
    path = _write(tmp_path, _NONFINITE_SPEC)
    out = tmp_path / "x.png"
    result = runner.invoke(app, ["plot", str(path), "--kind", "tornado", "--out", str(out)])

    assert result.exit_code == 1, result.output
    assert "nonfinite_value" in result.output
    assert not out.exists(), "a chart was written for a spec that does not validate"


# --------------------------------------------------------------------------- #
# No false positives: notices must NOT block, and valid specs must still run
# --------------------------------------------------------------------------- #


def test_a_spec_with_only_notices_still_runs(tmp_path: Path) -> None:
    """check_marginal_fit() returns ValidationNotice, deliberately a distinct
    type from ValidationError. A quantile-fit mismatch or a suspected
    percent/fraction mixup is a hint, not a violation - it must never block."""
    import valdist
    from valdist.spec.loader import load_spec
    from valdist.spec.validate import check_marginal_fit, validate

    notice_spec = textwrap.dedent("""
        schema_version: "1.0"
        name: notices-only
        valuation: equity_v2
        price: 25.0
        seed: 0
        n: 2000
        factors: []
        drivers:
          fcf_growth:
            marginal: {family: lognormal, p10: 0.5, p50: 6.0, p90: 11.0}
            loadings: {}
          terminal_growth:
            marginal: {family: normal, p10: 1.0, p50: 1.5, p90: 2.0}
            loadings: {}
          wacc:
            marginal: {family: normal, p10: 7.0, p50: 8.5, p90: 10.0}
            loadings: {}
        constants:
          shares: 200.0
          debt: 1500.0
          cash: 300.0
          fcff: 500.0
          normalized_earnings: 480.0
          years: 7
          relative_value_per_share: 30.0
        weights: {w_dcf: 45.0, w_epv: 35.0, w_relative: 20.0}
        """)
    path = _write(tmp_path, notice_spec)
    spec = load_spec(path)

    assert validate(spec) == [], "fixture must have no hard errors"
    assert check_marginal_fit(spec), "fixture must actually produce a notice"

    result = valdist.run(path)  # must not raise
    assert result.n == 2000


def test_golden_spec_still_runs_through_every_entry_point() -> None:
    """The user's artifact must be untouched by the new gate."""
    from conftest import VICI_SPEC_PATH

    import valdist
    from valdist.spec.loader import hydrate, load_spec

    spec = load_spec(VICI_SPEC_PATH)
    hydrate(spec)  # must not raise
    r = valdist.run(VICI_SPEC_PATH)
    assert r.p_undervalued == pytest.approx(0.9312, abs=0.0001)

    cli = runner.invoke(app, ["run", str(VICI_SPEC_PATH)])
    assert cli.exit_code == 0, cli.output
