"""Enforcement for engine invariants that nothing was checking."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from _scan import python_files

# Note: valdist is deliberately not imported at module scope. The structural
# tests below read the source with `ast` and must still run - and still give a
# clear assertion message - when the package itself is broken. Importing it here
# would turn a planted `import requests` into a ModuleNotFoundError at collection
# time, i.e. a confusing red rather than a clear rule violation. The behavioural tests
# import what they need inside the function body.

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "valdist"


def _imported_modules(py_file: Path) -> set[str]:
    """Every module name imported by *py_file*, including deferred/in-function imports."""
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


# --------------------------------------------------------------------------- #
# Valuation-agnostic core
# --------------------------------------------------------------------------- #


def test_invariant_2_1_core_never_imports_adapters() -> None:
    """core/ contains no domain valuation math."""
    offenders = {}
    for py_file in sorted((PACKAGE_ROOT / "core").glob("*.py")):
        bad = {m for m in _imported_modules(py_file) if m.startswith("valdist.adapters")}
        if bad:
            offenders[py_file.name] = sorted(bad)

    assert offenders == {}, (
        f"core/ imports adapter code, violating the valuation-agnostic core rule: {offenders}. "
        "Domain valuation logic belongs in valdist/adapters/. If core needs to learn "
        "something from an adapter, route it through the diagnostics flag channel "
        "(core/diagnostics.py), which carries names rather than meaning."
    )


def test_invariant_2_1_diagnostics_channel_carries_no_domain_meaning() -> None:
    """Corollary: core counts named flags without knowing what they mean."""
    source = (PACKAGE_ROOT / "core" / "diagnostics.py").read_text()
    domain_terms = ("terminal_spread", "wacc", "cap_rate", "dcf", "ddm", "affo", "epv")
    # Only inspect executable code; the module docstring may legitimately cite
    # examples to explain the channel.
    tree = ast.parse(source)
    body = tree.body[1:] if isinstance(tree.body[0], ast.Expr) else tree.body
    code = "\n".join(ast.unparse(node) for node in body).lower()

    leaked = [t for t in domain_terms if t in code]
    assert leaked == [], (
        f"core/diagnostics.py's executable code mentions domain concepts {leaked}, "
        "violating the valuation-agnostic core rule. The channel must carry flag "
        "names, never their meaning."
    )


# --------------------------------------------------------------------------- #
# Validation aggregates
# --------------------------------------------------------------------------- #


def test_invariant_2_4_validate_aggregates_every_planted_error() -> None:
    """The validator reports all violations at once, never fail-fast."""
    from valdist.spec.schema import SpecModel
    from valdist.spec.validate import validate

    raw = {
        "schema_version": "9.9",  # unsupported_schema_version
        "name": "everything-wrong",
        "valuation": "reit_v2",  # real adapter, but inputs missing below
        "price": 10.0,
        "factors": ["f", "f"],  # duplicate_factor
        "drivers": {
            "overloaded": {  # loadings_exceed_one
                "marginal": {"family": "normal", "p10": 1.0, "p50": 2.0, "p90": 3.0},
                "loadings": {"f": 1.1},
            },
            "ghost_ref": {  # unknown_factor
                "marginal": {"family": "normal", "p10": 1.0, "p50": 2.0, "p90": 3.0},
                "loadings": {"nope": 0.5},
            },
            "bad_lognormal": {  # lognormal_nonpositive_p10
                "marginal": {
                    "family": "lognormal",
                    "p10": -1.0,
                    "p50": 2.0,
                    "p90": 5.0,
                },
                "loadings": {},
            },
            "bad_lognormal3": {  # lognormal3_infeasible
                "marginal": {
                    "family": "lognormal3",
                    "p10": 0.5,
                    "p50": 2.5,
                    "p90": 4.5,
                },
                "loadings": {},
            },
            "shares": {  # name_collision with the constant below
                "marginal": {"family": "normal", "p10": 1.0, "p50": 2.0, "p90": 3.0},
                "loadings": {},
            },
        },
        "constants": {"shares": 1000.0},
        "weights": {"w_a": 60.0, "w_b": 60.0},  # weights_not_100
    }
    errors = validate(SpecModel.model_validate(raw))
    codes = {e.code for e in errors}

    expected = {
        "unsupported_schema_version",
        "duplicate_factor",
        "loadings_exceed_one",
        "unknown_factor",
        "lognormal_nonpositive_p10",
        "lognormal3_infeasible",
        "name_collision",
        "weights_not_100",
        "missing_required_input",  # reit_v2 needs dps/affo/noi/... none supplied
    }
    missing = expected - codes
    assert missing == set(), (
        f"validate() fail-fasted or skipped checks, violating the aggregate-digest "
        f"rule. Missing from the "
        f"digest: {sorted(missing)}. Got: {sorted(codes)}"
    )


def test_invariant_2_4_every_validate_code_is_covered_by_this_suite() -> None:
    """Meta-guard: a new validate() code cannot be added without a test."""
    source = (PACKAGE_ROOT / "spec" / "validate.py").read_text()
    tree = ast.parse(source)

    reported_types = {"ValidationError", "ValidationNotice"}
    emitted: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in reported_types:
            for kw in node.keywords:
                if kw.arg == "code" and isinstance(kw.value, ast.Constant):
                    emitted.add(kw.value.value)

    assert emitted, "could not scrape any ValidationError/Notice codes - did validate.py move?"

    # Scrape the codes the tests actually assert on, as string literals in their
    # own AST - not a substring search over concatenated source. A plain
    # `code not in test_corpus` counts a code as "tested" if the string happens
    # to appear anywhere at all: in a comment, in a docstring, or as a substring
    # of a longer code (a future "nonpositive_years_v2" would mark
    # "nonpositive_years" tested without a single assertion existing).
    tested: set[str] = set()
    for test_file in sorted(Path(__file__).parent.glob("test_*.py")):
        tree = ast.parse(test_file.read_text(), filename=str(test_file))

        # Docstrings are ast.Constant strings too, and a code merely *described*
        # in prose is not a code that is *asserted on*. Exclude them.
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
            ):
                tested.add(node.value)

    untested = sorted(code for code in emitted if code is not None and code not in tested)
    assert untested == [], (
        f"validate() can emit {untested}, but no test in tests/ asserts on those codes "
        "as a string literal. Every aggregation code needs a test, or the digest "
        "guarantee "
        "is untested surface."
    )


# --------------------------------------------------------------------------- #
# Honesty about uncertainty (the one that was never implemented)
# --------------------------------------------------------------------------- #


def _certain_model():
    """Value is always far above any sane price -> p_undervalued == 1.0."""
    from valdist.core import Marginal, Model
    from valdist.core.factors import FactorModel

    fm = FactorModel(["f"], {"a": {"f": 0.5}})
    return Model(
        [Marginal("a", p10=10.0, p50=11.0, p90=12.0, family="normal")],
        fm,
        valuation=lambda v: v["a"],
    )


def test_invariant_2_6_p_undervalued_of_one_is_a_warning_not_a_result() -> None:
    """Treat p_undervalued in {0, 1} as a too-narrow-inputs warning."""
    from valdist import report

    result = _certain_model().run(n=5_000, price=1.0, seed=0)

    assert result.p_undervalued == 1.0
    assert result.p_undervalued_stderr == 0.0  # the artefact that must not stand alone

    warnings = result.warnings()
    assert warnings, "p_undervalued == 1.0 produced no warning - this is unenforced"
    # "too-narrow-inputs" (not the bare word "narrow") is the string unique to
    # this warning - the diagnostic-flag warning also contains "narrow"
    # ("Narrow the bands..."), so a bare substring match can't tell them apart.
    assert any("too-narrow-inputs" in w.lower() for w in warnings)
    assert "WARNING" in report.to_text(result).upper()


def test_invariant_2_6_p_undervalued_of_zero_warns_too() -> None:
    """Symmetric: every draw below the price is the same failure."""
    result = _certain_model().run(n=5_000, price=1_000.0, seed=0)
    assert result.p_undervalued == 0.0
    assert any("too-narrow-inputs" in w.lower() for w in result.warnings())


def test_invariant_2_6_healthy_run_warns_about_nothing() -> None:
    """No false positives: the warning must mean something when it appears."""
    from valdist.core import Marginal, Model
    from valdist.core.factors import FactorModel

    fm = FactorModel(["f"], {"a": {"f": 0.5}})
    model = Model(
        [Marginal("a", p10=1.0, p50=2.0, p90=3.0, family="normal")],
        fm,
        valuation=lambda v: v["a"],
    )
    result = model.run(n=5_000, price=2.0, seed=0)
    assert 0.0 < result.p_undervalued < 1.0
    assert result.warnings() == []


def test_invariant_2_6_mc_stderr_is_always_reported() -> None:
    """Report MC standard error on p_undervalued."""
    from valdist import report
    from valdist.core import Marginal, Model
    from valdist.core.factors import FactorModel

    fm = FactorModel(["f"], {"a": {"f": 0.5}})
    model = Model(
        [Marginal("a", p10=1.0, p50=2.0, p90=3.0, family="normal")],
        fm,
        valuation=lambda v: v["a"],
    )
    result = model.run(n=2_000, price=2.0, seed=0)

    assert "p_undervalued_stderr" in result.summary()
    assert "stderr" in report.to_text(result).lower()
    assert "p_undervalued_stderr" in json.loads(report.to_json(result))


# --------------------------------------------------------------------------- #
# Incoherent draws are floored, counted, and reported
# --------------------------------------------------------------------------- #


def test_invariant_2_8_adapters_never_use_a_bare_max_floor() -> None:
    """Adapters must floor via floors.floored(), never a bare max(x, eps)."""
    offenders = []
    for py_file in sorted((PACKAGE_ROOT / "adapters").glob("*.py")):
        if py_file.name == "floors.py":  # the one place the floor is allowed to live
            continue
        tree = ast.parse(py_file.read_text(), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "max":
                offenders.append(f"{py_file.name}:{node.lineno}: {ast.unparse(node)}")

    assert offenders == [], (
        "adapter code uses a bare max() floor, violating the floor-and-flag rule:\n  "
        + "\n  ".join(offenders)
        + "\nUse floors.floored(value, flag_name) so the floored draw is counted and "
        "reported instead of silently blended."
    )


def test_invariant_2_8_floored_draws_are_counted_and_surfaced() -> None:
    """A crossing band must be reported, not silently blended."""
    from valdist import report
    from valdist.adapters.registry import get_valuation
    from valdist.core import Marginal, Model
    from valdist.core.factors import FactorModel

    fm = FactorModel(
        ["macro"],
        {
            "fcf_growth": {"macro": 0.5},
            "terminal_growth": {"macro": 0.3},
            "wacc": {"macro": -0.4},
        },
    )
    model = Model(
        [
            Marginal("fcf_growth", p10=4.0, p50=8.0, p90=12.0, family="normal"),
            Marginal("terminal_growth", p10=1.5, p50=2.5, p90=3.5, family="normal"),
            Marginal("wacc", p10=6.0, p50=9.0, p90=12.0, family="normal"),
        ],
        fm,
        get_valuation("equity_v2"),
        constants=dict(
            shares=450.0,
            debt=4000.0,
            cash=6000.0,
            fcff=7000.0,
            years=10,
            normalized_earnings=6000.0,
            relative_value_per_share=500.0,
            w_dcf=50.0,
            w_epv=30.0,
            w_relative=20.0,
        ),
    )
    result = model.run(n=20_000, price=500.0, seed=0)

    assert result.diagnostics, "a crossing wacc/terminal_growth band raised no flag"
    assert result.diagnostic_rate("dcf_terminal_spread_floored") > 0
    assert any("floored" in w for w in result.warnings())
    assert "WARNING" in report.to_text(result).upper()


def test_invariant_2_8_golden_spec_floors_nothing() -> None:
    """No false positives: the reference spec's bands do not cross."""
    from conftest import VICI_SPEC_PATH

    from valdist.spec.loader import hydrate, load_spec

    spec = load_spec(VICI_SPEC_PATH)
    result = hydrate(spec).run(n=spec.n, price=spec.price, seed=spec.seed, nu=spec.nu)
    assert result.diagnostics == {}
    assert result.warnings() == []


# --------------------------------------------------------------------------- #
# Non-goals, made structural
# --------------------------------------------------------------------------- #


def test_nongoal_no_data_retrieval_anywhere() -> None:
    """No data retrieval or fetching. Specs are authored upstream."""
    forbidden = {
        "requests",
        "httpx",
        "urllib",
        "urllib.request",
        "aiohttp",
        "yfinance",
        "pandas_datareader",
        "alpha_vantage",
        "socket",
    }
    offenders = {}
    for py_file in python_files(PACKAGE_ROOT, "valdist/"):
        hits = {
            m for m in _imported_modules(py_file) if m in forbidden or m.split(".")[0] in forbidden
        }
        if hits:
            offenders[str(py_file.relative_to(PACKAGE_ROOT))] = sorted(hits)

    assert offenders == {}, (
        f"valdist imports network/data-retrieval modules, violating the no-fetching "
        f"rule: {offenders}. "
        "Specs are authored upstream; valdist consumes them."
    )


# --------------------------------------------------------------------------- #
# Run-time engine parameters must reach every command that draws samples
# --------------------------------------------------------------------------- #


def test_every_cli_command_that_runs_a_model_exposes_sampling() -> None:
    """`sampling` is a run-time engine choice, so every CLI command that draws
    Monte Carlo samples must expose it and forward it.
    """
    cli_src = (PACKAGE_ROOT / "cli.py").read_text()
    tree = ast.parse(cli_src)

    def _is_command(fn: ast.FunctionDef) -> bool:
        for dec in fn.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(target, ast.Attribute) and target.attr == "command":
                return True
        return False

    def _runs_a_model(fn: ast.FunctionDef) -> bool:
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "run":
                return True
            if isinstance(f, ast.Name) and f.id == "check_coverage":
                return True
        return False

    def _forwards_sampling(fn: ast.FunctionDef) -> bool:
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and any(kw.arg == "sampling" for kw in node.keywords):
                return True
        return False

    commands = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and _is_command(n)]
    assert commands, "no @app.command() functions found - did cli.py move?"

    sampling_commands = [c for c in commands if _runs_a_model(c)]
    assert sampling_commands, "no CLI command appears to run a model - scraper is broken"

    for cmd in sampling_commands:
        params = {a.arg for a in cmd.args.args} | {a.arg for a in cmd.args.kwonlyargs}
        assert "sampling" in params, (
            f"CLI command '{cmd.name}' runs a model but takes no `sampling` parameter. "
            "Every command that draws samples must expose --sampling, or it silently "
            "ignores the analyst's choice of sampler."
        )
        assert _forwards_sampling(cmd), (
            f"CLI command '{cmd.name}' accepts `sampling` but never forwards it - "
            "accepting a flag and dropping it is worse than not having it."
        )


def test_check_coverage_forwards_sampling_to_model_run() -> None:
    """The library half of the same rule: check_coverage takes `sampling`, so it
    must actually hand it to Model.run() rather than accept it and drop it."""
    import inspect

    from valdist.calibrate.coverage import check_coverage

    sig = inspect.signature(check_coverage)
    assert "sampling" in sig.parameters

    src = (PACKAGE_ROOT / "calibrate" / "coverage.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and any(kw.arg == "sampling" for kw in node.keywords)
        ):
            return
    raise AssertionError("check_coverage takes `sampling` but never passes it to model.run()")


# --------------------------------------------------------------------------- #
# Marginal families: one source of truth
# --------------------------------------------------------------------------- #


def test_marginal_families_have_a_single_source_of_truth() -> None:
    """The set of valid families is enumerated in four places - the schema's
    Literal, marginals._FAMILIES, _fit()'s branches and ppf()'s branches - and
    nothing forced them to agree. Adding a family could half-land: accepted by
    the schema, then dying on ppf()'s `raise AssertionError(unknown family)`
    deep inside the Monte Carlo loop.
    """
    from valdist.core.marginals import FAMILIES, Marginal

    src = (PACKAGE_ROOT / "core" / "marginals.py").read_text()
    tree = ast.parse(src)

    def _families_compared_in(func_name: str) -> set[str]:
        """String literals compared against self.family inside *func_name*."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                found: set[str] = set()
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Compare) and isinstance(sub.left, ast.Attribute):
                        if sub.left.attr != "family":
                            continue
                        for comp in sub.comparators:
                            if isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                                found.add(comp.value)
                            elif isinstance(comp, (ast.Set, ast.Tuple, ast.List)):
                                found |= {
                                    e.value
                                    for e in comp.elts
                                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                                }
                return found
        raise AssertionError(f"{func_name} not found in marginals.py")

    for func in ("_fit", "ppf"):
        handled = _families_compared_in(func)
        missing = FAMILIES - handled
        assert not missing, (
            f"{func}() does not handle {sorted(missing)}, but they are declared in FAMILIES. "
            "A family accepted by the schema and unhandled here dies mid-Monte-Carlo."
        )
        stray = handled - FAMILIES
        assert not stray, f"{func}() handles {sorted(stray)}, which are not in FAMILIES"

    # And the behavioural half: every declared family actually constructs and
    # produces finite values. A structural check alone would not catch a family
    # that is listed and branched on but broken.
    import warnings

    for family in sorted(FAMILIES):
        with warnings.catch_warnings():
            # No single (p10, p50, p90) triple is a clean fit for every family at
            # once; quantile-fit quality is not what this test is about.
            warnings.simplefilter("ignore", UserWarning)
            m = Marginal("x", p10=1.0, p50=2.0, p90=4.0, family=family)
        vals = m.ppf(np.array([0.1, 0.5, 0.9]))
        assert np.isfinite(vals).all(), f"{family} produced {vals}"


def test_schema_family_literal_matches_marginals_families() -> None:
    """The pydantic Literal must BE the marginals FAMILIES set, not a copy of it."""
    from typing import get_args

    from valdist.core.marginals import FAMILIES
    from valdist.spec.schema import MarginalSpec

    literal_families = set(get_args(MarginalSpec.model_fields["family"].annotation))
    assert literal_families == FAMILIES, (
        f"schema Literal {sorted(literal_families)} != marginals.FAMILIES {sorted(FAMILIES)}"
    )


def test_nongoal_plotting_is_never_a_hard_dependency() -> None:
    """No plotting as a hard dependency (optional / lazy-imported only)."""
    plot_module = PACKAGE_ROOT / "plot.py"
    tree = ast.parse(plot_module.read_text(), filename=str(plot_module))

    module_level_imports: set[str] = set()
    for node in tree.body:  # top level only, not ast.walk
        if isinstance(node, ast.Import):
            module_level_imports.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_level_imports.add(node.module)

    leaked = {m for m in module_level_imports if m.split(".")[0] == "matplotlib"}
    assert leaked == set(), (
        f"valdist/plot.py imports matplotlib at module scope ({leaked}), which breaks "
        f"the optional-plotting rule. "
        "It must be imported inside the function bodies so `import valdist` never needs it."
    )


@pytest.mark.parametrize("module", ["valdist", "valdist.core", "valdist.spec.validate"])
def test_nongoal_core_imports_without_matplotlib(module: str) -> None:
    """Behavioural half: the package imports with matplotlib absent."""
    import importlib
    import sys

    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] == "matplotlib"}
    for k in saved:
        sys.modules[k] = None  # type: ignore[assignment]
    try:
        importlib.reload(importlib.import_module(module))
    finally:
        sys.modules.update(saved)


# --------------------------------------------------------------------------- #
# P25 is the conservative value anchor, not a fourth number in a row of four
# --------------------------------------------------------------------------- #
#
# Graham's method is not "median value minus a haircut" - it is a deliberately
# conservative estimate of value, and then a discount to that. In distributional
# terms the conservative estimate is a low quantile, so P25 is reported as the
# anchor rather than as a milder version of the P10 downside check.
#
# The margin of safety's sign can flip between P10 and P25 - a spec priced
# below its P25 but above its P10 reads as a marginal buy at the anchor and a
# false negative at the tail. A thin-tailed spec makes this worse: when P10
# sits at a fraction of the median, it is describing the distribution family's
# own extrapolation, not a representative bad case, and nothing in the old
# single-quantile output said so.
#
# Why P25 and not something deeper: tails are censored, extrapolated by the
# family, or floored (never invented). Below roughly P10 a `normal` or
# `lognormal` marginal is reporting its own extrapolation rather than the
# analyst's research, so a deeper quantile would describe the distribution
# choice and not the company.
#
# The engine needed no new maths for any of this: `value_quantiles` and
# `margin_of_safety` have always taken a `qs` tuple. What changed is what the
# report names.


def _priced_result():
    import valdist

    return valdist.run(Path("examples/vici.yaml"))


def test_summary_carries_the_conservative_anchor() -> None:
    """`value_p25`, `mos_p25` and the P25/P50 ratio are part of the contract."""
    s = _priced_result().summary()
    for key in ("value_p25", "mos_p25", "p25_p50_ratio"):
        assert key in s, f"summary() is missing {key}"
    assert s["value_p10"] < s["value_p25"] < s["value_p50"], (
        f"P25 is not between P10 and P50: {s['value_p10']}, {s['value_p25']}, {s['value_p50']}"
    )
    assert s["mos_p10"] < s["mos_p25"] < s["mos_p50"], "margins are not ordered"


def test_the_p25_p50_ratio_measures_dispersion() -> None:
    """The one free number that says whether the distribution is tight or fat."""
    s = _priced_result().summary()
    assert s["p25_p50_ratio"] == pytest.approx(s["value_p25"] / s["value_p50"]), (
        "the ratio does not equal P25/P50"
    )
    assert 0.0 < s["p25_p50_ratio"] < 1.0, "a P25/P50 ratio outside (0, 1) is not a dispersion read"


def test_summary_keys_are_present_without_a_price() -> None:
    """The fixed-shape rule holds for the new keys too."""
    import valdist

    model = valdist.hydrate(valdist.load_spec(Path("examples/vici.yaml")))
    s = model.run(n=2000, price=None, seed=0).summary()
    assert s["mos_p25"] is None, "mos_p25 should be None without a price, not absent"
    assert isinstance(s["value_p25"], float), "value_p25 does not depend on price"
    assert isinstance(s["p25_p50_ratio"], float), "the ratio does not depend on price"


def test_to_text_names_p25_as_the_conservative_anchor() -> None:
    """It is reported as an anchor, with its meaning, not as a bare fourth row."""
    from valdist import report

    text = report.to_text(_priced_result())
    assert "Conservative value anchor" in text, "to_text does not name the anchor"
    assert "P25" in text and "buy below" in text.lower(), (
        "to_text does not say what the anchor means - that value is "
        "price-independent, so P25 is itself a buy-below price"
    )


def test_adding_payload_fields_did_not_bump_the_schema_version() -> None:
    """The P25 fields are additive, and the contract says additions are free."""
    from valdist.report import REPORT_SCHEMA_VERSION

    assert REPORT_SCHEMA_VERSION == "1.0", (
        "the schema version moved for a purely additive change, against the "
        "policy in its own docstring"
    )


def test_to_json_carries_the_new_fields() -> None:
    from valdist import report

    payload = json.loads(report.to_json(_priced_result()))
    for key in ("value_p25", "mos_p25", "p25_p50_ratio"):
        assert key in payload, f"to_json payload is missing {key}"
