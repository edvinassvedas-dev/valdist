"""One guarded directory scan, shared by every anchor that walks a package."""

from __future__ import annotations

import pathlib


def python_files(root: pathlib.Path, label: str) -> list[pathlib.Path]:
    """Every `*.py` under *root*, refusing to hand back an empty scan."""
    files = sorted(root.rglob("*.py"))
    assert files, (
        f"{label}: no python files under {root} - this scan would pass while "
        "checking nothing. If the directory moved, point the anchor at its new "
        "home; if it was deleted deliberately, remove or skip the anchor "
        "explicitly rather than leaving it green and blind."
    )
    return files
