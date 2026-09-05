"""The engine's printed strings obey the dash style rule."""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINE_DIR = REPO_ROOT / "valdist"

EM_DASH = "—"

#: Marks that must not appear in a string the engine prints, and their names.
BANNED = ((EM_DASH, "em-dash"), ("--", "bare double-hyphen"))

#: A CLI flag name is syntax, not punctuation. It is exempt only at a word
#: boundary and only when a lowercase letter follows, so `--sampling` (declared
#: or quoted mid-message) passes while `a -- b` and `value--other` - the two
#: shapes prose actually takes - do not.
FLAG_TOKEN = r"(?:\A|(?<=\s))--[a-z][a-z0-9-]*"


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    """Ids of the Constant nodes that are module/class/function docstrings."""
    ids = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            ids.add(id(body[0].value))
    return ids


def dash_offenders(source: str, filename: str = "<source>") -> list[str]:
    """Non-docstring string literals in *source* carrying a banned dash mark."""
    tree = ast.parse(source)
    doc_ids = _docstring_node_ids(tree)

    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if id(node) in doc_ids:
            continue
        residue = re.sub(FLAG_TOKEN, "", node.value)
        for mark, label in BANNED:
            if mark in residue:
                offenders.append(f"{filename}:{node.lineno}: {label} in {node.value!r}")
                break
    return offenders


def _engine_files() -> list[pathlib.Path]:
    return sorted(ENGINE_DIR.rglob("*.py"))


def test_scan_is_not_vacuous() -> None:
    """The scan reaches the modules whose strings the engine actually prints."""
    files = _engine_files()
    assert files, f"no python files under {ENGINE_DIR}"
    relative = {p.relative_to(REPO_ROOT).as_posix() for p in files}
    for expected in ("valdist/cli.py", "valdist/report.py", "valdist/spec/validate.py"):
        assert expected in relative, f"{expected} missing from the dash-style scan"


def test_printed_strings_use_en_dashes() -> None:
    offenders: list[str] = []
    for path in _engine_files():
        offenders += dash_offenders(
            path.read_text(encoding="utf-8"), path.relative_to(REPO_ROOT).as_posix()
        )
    assert offenders == [], "use an en-dash (–) in strings the engine prints:\n" + "\n".join(
        offenders
    )


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("spaced double-hyphen", 'typer.echo("all inside (-1, 1) -- if this expects PERCENT")'),
        ("unspaced double-hyphen", 'raise ValueError("cap rate--terminal growth crossed")'),
        ("em-dash", f'typer.echo("floored 3 draws {EM_DASH} see DIAGNOSTICS")'),
        ("inside an f-string", 'typer.echo(f"got {n} draws -- expected more")'),
        ("in a nested call", 'log(fmt("a -- b"))'),
    ],
)
def test_flags_violations(label: str, source: str) -> None:
    assert dash_offenders(source), f"{label} not flagged"


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("bare flag declaration", 'typer.Option("mc", "--sampling", help="mc or lhs")'),
        ("flag named in a message", "typer.echo(f\"--kind must be 'tornado', got {kind!r}\")"),
        ("two flags in one message", 'typer.echo("pass --lo and --hi together")'),
        ("module docstring em-dash", f'"""valdist {EM_DASH} probabilistic value engine."""'),
        ("numpydoc underline", '"""Doc.\n\n    Parameters\n    ----------\n    x: int\n    """'),
        ("negative number", 'typer.echo(f"rank correlation {-0.775:g}")'),
        ("en-dash is the correct form", 'typer.echo("floored 3 draws – see DIAGNOSTICS")'),
    ],
)
def test_allows_legitimate_forms(label: str, source: str) -> None:
    assert dash_offenders(source) == [], f"{label} wrongly flagged"


def test_function_docstrings_are_exempt_but_their_bodies_are_not() -> None:
    """The exemption is per-literal, not per-file: one docstring does not
    launder the printed strings that follow it."""
    source = f'def f():\n    """Prose {EM_DASH} allowed here."""\n    typer.echo("a -- b")\n'
    offenders = dash_offenders(source)
    assert len(offenders) == 1
    assert "a -- b" in offenders[0]
