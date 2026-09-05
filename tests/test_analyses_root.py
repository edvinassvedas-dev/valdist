"""`VALDIST_ANALYSES` points both readers at the same directory.

`gui/` and `prices/` cannot import each other or `valdist`, so this resolution
is deliberately duplicated rather than shared - these cases keep the two
copies agreeing.
"""

from __future__ import annotations

import importlib
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONSUMERS = ["prices.fetch", "gui.server"]


def _reload(module_name: str):
    return importlib.reload(importlib.import_module(module_name))


@pytest.fixture(autouse=True)
def _restore_modules():
    """Reload both modules afterwards so a mutated env cannot leak sideways."""
    yield
    for name in CONSUMERS:
        _reload(name)


@pytest.mark.parametrize("module_name", CONSUMERS)
def test_default_is_the_repo_analyses_dir(module_name: str, monkeypatch) -> None:
    """With no env var set, both readers point at `<repo>/analyses`."""
    monkeypatch.delenv("VALDIST_ANALYSES", raising=False)
    mod = _reload(module_name)
    assert mod.ANALYSES_DIR == REPO_ROOT / "analyses", (
        f"{module_name}.ANALYSES_DIR defaulted to {mod.ANALYSES_DIR}, "
        f"expected {REPO_ROOT / 'analyses'}"
    )


@pytest.mark.parametrize("module_name", CONSUMERS)
def test_the_env_var_relocates_the_root(module_name: str, monkeypatch, tmp_path) -> None:
    """A record outside the working tree is the point: it cannot be committed."""
    monkeypatch.setenv("VALDIST_ANALYSES", str(tmp_path))
    mod = _reload(module_name)
    assert mod.ANALYSES_DIR == tmp_path, (
        f"{module_name} ignored VALDIST_ANALYSES: got {mod.ANALYSES_DIR}"
    )


def test_both_readers_agree_on_the_same_root(monkeypatch, tmp_path) -> None:
    """No shared helper can enforce this, so the test does: if one reader
    learns a new rule and the other does not, they silently disagree."""
    monkeypatch.setenv("VALDIST_ANALYSES", str(tmp_path))
    roots = {name: _reload(name).ANALYSES_DIR for name in CONSUMERS}
    assert len(set(roots.values())) == 1, f"readers disagree about the record root: {roots}"


@pytest.mark.parametrize("module_name", CONSUMERS)
def test_an_empty_env_var_falls_back_rather_than_pointing_at_cwd(
    module_name: str, monkeypatch
) -> None:
    """`Path("")` is `Path(".")`, so a naive `os.environ.get(...)` would
    silently retarget at the current working directory instead of falling
    back."""
    monkeypatch.setenv("VALDIST_ANALYSES", "")
    mod = _reload(module_name)
    assert mod.ANALYSES_DIR == REPO_ROOT / "analyses", (
        f"{module_name} treated an empty VALDIST_ANALYSES as a path: {mod.ANALYSES_DIR}"
    )
