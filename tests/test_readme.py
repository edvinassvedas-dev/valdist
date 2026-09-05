"""The documented test counts are true."""

from __future__ import annotations

import functools
import pathlib
import re
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: (file, pattern capturing the claimed count, human description of where it lives).
#: Both files carry the same claim in different words, and both drifted together -
#: fixing one and not the other is exactly the failure this parametrisation stops.
DOC_CLAIMS = [
    (
        "README.md",
        re.compile(r"pytest\s+-q\s*#\s*([\d,]+)\s+tests"),
        "`pytest -q    # <N> tests, all green` in the Install section",
    ),
    (
        "CHANGELOG.md",
        # Either arrow. The property being checked is that the stated count is
        # true, not which glyph separates the two numbers; hardcoding U+2192
        # meant an ASCII "->" read as no claim at all, which fails as loudly as
        # a wrong number for a reason that has nothing to do with the count.
        re.compile(r"Test suite:\s*[\d,]+\s*(?:→|->)\s*([\d,]+)"),
        "`Test suite: <old> -> <N>` under [Unreleased]",
    ),
]


def _claimed_count(filename: str, pattern: re.Pattern, where: str) -> int:
    text = (REPO_ROOT / filename).read_text(encoding="utf-8")
    match = pattern.search(text)
    assert match, (
        f"{filename} must state the test count as {where}. This anchor cannot "
        "check a claim that isn't there - removing the line would silently "
        "disable it, so its absence is a failure rather than a skip."
    )
    return int(match.group(1).replace(",", ""))


@functools.lru_cache(maxsize=1)
def _actual_count() -> int:
    """Collect the full suite in a subprocess and read the total back."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    match = re.search(r"(\d+) tests? collected", proc.stdout)
    # Failing rather than skipping is deliberate. A skip here would be a vacuous
    # pass: the anchor would stop checking and report success while doing so.
    assert match, (
        "could not determine the collected test count from `pytest --collect-only`.\n"
        f"exit={proc.returncode}\nstdout tail:\n{proc.stdout[-800:]}\n"
        f"stderr tail:\n{proc.stderr[-800:]}"
    )
    return int(match.group(1))


@pytest.mark.parametrize("filename,pattern,where", DOC_CLAIMS, ids=[c[0] for c in DOC_CLAIMS])
def test_doc_states_the_real_test_count(filename: str, pattern: re.Pattern, where: str) -> None:
    """Change gates: each document's stated test count matches the suite."""
    claimed, actual = _claimed_count(filename, pattern, where), _actual_count()
    assert claimed == actual, (
        f"{filename} claims {claimed} tests; the suite collects {actual}.\n"
        f"Fix: update {where} in {filename} to {actual}.\n"
        "This is the Change-gates rule in action - a change that adds or removes "
        "tests updates both documents in the same batch, not later. Note the "
        "other document may need the same edit."
    )
