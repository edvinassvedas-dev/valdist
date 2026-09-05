"""The GUI cannot corrupt the engine."""

from __future__ import annotations

import ast
import json
import pathlib

import pytest
from _scan import python_files

from valdist import report

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GUI_ROOT = REPO_ROOT / "gui"
PACKAGE_ROOT = REPO_ROOT / "valdist"


def _imported_modules(py_file: pathlib.Path) -> set[str]:
    """Every module named by an `import x` / `from x import ...` in one file."""
    tree = ast.parse(py_file.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


# --------------------------------------------------------------------------- #
# 3. to_json() is a versioned contract now that gui/ consumes it
# --------------------------------------------------------------------------- #


def test_to_json_carries_report_schema_version(two_driver_model) -> None:
    """The payload gui/ reads announces its own contract version."""
    result = two_driver_model(0.8).run(n=500, price=10.0, seed=0)
    payload = json.loads(report.to_json(result))
    assert "report_schema_version" in payload, (
        "to_json() must carry report_schema_version: gui/ is an external "
        "consumer, so the payload shape is a contract, not a free variable."
    )
    assert isinstance(payload["report_schema_version"], str)
    assert payload["report_schema_version"] == report.REPORT_SCHEMA_VERSION


def test_report_schema_version_is_not_the_spec_schema_version() -> None:
    """They version different things; a consumer confusing them reads the wrong
    contract. Pinned so the report key can never be renamed to `schema_version`.
    """
    assert not hasattr(report, "SCHEMA_VERSION"), (
        "report must not export a bare SCHEMA_VERSION - it would be mistaken "
        "for the spec's schema_version."
    )


# --------------------------------------------------------------------------- #
# 1. gui/ inherits the data-retrieval ban
# --------------------------------------------------------------------------- #

# Same list applies to valdist/, minus `socket`: the viewer is served
# by the standard library's http.server, which is a *serving* concern, not a
# data-retrieval one. Banning socket outright would ban the server itself while
# doing nothing to stop the actual risk, which is an outbound fetch.
_FORBIDDEN_IN_GUI = {
    "requests",
    "httpx",
    "urllib.request",
    "aiohttp",
    "yfinance",
    "pandas_datareader",
    "alpha_vantage",
}


@pytest.mark.skipif(not GUI_ROOT.exists(), reason="gui/ not present")
def test_nongoal_gui_does_no_data_retrieval() -> None:
    """The data-retrieval ban binds gui/ exactly as it binds valdist/."""
    offenders = {}
    for py_file in python_files(GUI_ROOT, "gui/"):
        hits = {
            m
            for m in _imported_modules(py_file)
            if m in _FORBIDDEN_IN_GUI or m.split(".")[0] in _FORBIDDEN_IN_GUI
        }
        if hits:
            offenders[str(py_file.relative_to(REPO_ROOT))] = sorted(hits)

    assert offenders == {}, (
        f"gui/ imports network/data-retrieval modules, violating the no-fetching "
        f"rule: {offenders}. A UI is where 'just fetch the current price' gets "
        "written; that is exactly the kind of overfitting-by-convenience this "
        "project forbids downstream."
    )


# --------------------------------------------------------------------------- #
# 2. The dependency arrow points one way
# --------------------------------------------------------------------------- #


def test_nongoal_engine_never_imports_the_gui() -> None:
    """valdist/ must stay runnable with gui/ deleted."""
    offenders = {}
    for py_file in python_files(PACKAGE_ROOT, "valdist/"):
        hits = {m for m in _imported_modules(py_file) if m == "gui" or m.startswith("gui.")}
        if hits:
            offenders[str(py_file.relative_to(REPO_ROOT))] = sorted(hits)

    assert offenders == {}, (
        f"valdist/ imports the GUI: {offenders}. The engine must "
        "not depend on its viewer; the arrow points one way only."
    )


# --------------------------------------------------------------------------- #
# 4. The viewer is read-only
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not GUI_ROOT.exists(), reason="gui/ not present")
def test_gui_never_writes() -> None:
    """A read-only viewer opens nothing for writing and removes nothing."""
    banned_calls = {"rmtree", "remove", "unlink", "rmdir", "mkdir", "makedirs", "rename"}
    offenders: dict[str, list[str]] = {}

    for py_file in python_files(GUI_ROOT, "gui/"):
        bad: list[str] = []
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)

            if name in banned_calls:
                bad.append(f"{name}()")

            # open(..., "w"/"a"/"x"/"+") and Path.open / write_text / write_bytes
            if name in {"open", "write_text", "write_bytes"}:
                if name in {"write_text", "write_bytes"}:
                    bad.append(f"{name}()")
                    continue
                mode = ""
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    mode = str(node.args[1].value)
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = str(kw.value.value)
                if any(ch in mode for ch in "wax+"):
                    bad.append(f'open(..., "{mode}")')
        if bad:
            offenders[str(py_file.relative_to(REPO_ROOT))] = sorted(bad)

    assert offenders == {}, (
        f"gui/ contains write operations, violating the read-only viewer contract: "
        f"{offenders}. The viewer renders shipped analyses; it does not edit them."
    )
