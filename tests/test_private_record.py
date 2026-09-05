"""The private record and methodology never enter the tracked tree.

Four roots and one file are private: `analyses/` (research output), `docs/`
(the instruction sets that produce it), `notes/` (engine-scoped planning:
decisions, backlog, build history), `raw/` (source documents feeding
research), and `CLAUDE.md` (the operating principles). The engine is public;
none of those is.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    """Run git in the repo, or skip if this is not a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        pytest.skip(f"git unavailable: {exc}")
    if out.returncode != 0:
        pytest.skip(f"not a git checkout: git {' '.join(args)} -> {out.stderr.strip()}")
    return out.stdout


PRIVATE_ROOTS = ["analyses", "docs", "notes", "raw"]
PRIVATE_FILES = ["CLAUDE.md"]


@pytest.mark.parametrize("root", PRIVATE_ROOTS)
def test_no_file_under_a_private_root_is_tracked(root: str) -> None:
    """Adding the `.gitignore` line does not untrack a file already committed."""
    tracked = [line for line in _git("ls-files", f"{root}/").splitlines() if line.strip()]
    assert tracked == [], (
        f"{len(tracked)} file(s) under {root}/ are tracked, first: {tracked[:3]}. "
        f"Untrack with `git rm -r --cached {root}/`."
    )


@pytest.mark.parametrize("root", PRIVATE_ROOTS)
def test_gitignore_names_each_private_root_anchored(root: str) -> None:
    """Root-anchored: an unanchored pattern would also swallow the public
    fixture corpus at a matching subpath."""
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    lines = {line.strip() for line in text.splitlines()}
    assert f"/{root}/" in lines, f".gitignore must contain the root-anchored line `/{root}/`"


def test_the_fixture_corpus_is_not_swept_up_by_the_ignore_rule() -> None:
    """`git check-ignore` is the only authority on what a pattern actually
    matches; reading the pattern proves what was written, not what it means."""
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", "fixtures/analyses/probe.yaml"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 1, (
        "fixtures/analyses/ is matched by an ignore rule "
        f"(git check-ignore returned {out.returncode})."
    )


# --------------------------------------------------------------------------- #
# No tracked file names a ticker from the private record


#: Record tickers that also read as ordinary English (so a scan for them would
#: be all false positives). `VICI` is a deliberate exception: it is the
#: repository's public golden spec, cited by name and pinned by exact figures,
#: so its ticker is public by intent rather than by leak.
TICKER_ALLOWLIST: set[str] = {"VICI"}

_TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".html", ".json", ".toml", ".txt", ".cfg", ".ini"}


def _record_tickers() -> set[str]:
    """Tickers from the private record, never from whatever corpus the test
    suite happens to be pointed at."""
    from conftest import skip_unless_private_record

    skip_unless_private_record()

    import prices.fetch as pf

    if not pf.ANALYSES_DIR.is_dir():
        pytest.skip(f"no research record at {pf.ANALYSES_DIR}")
    names = set(pf.tickers_from_analyses())
    overrides, us_listed = pf.load_symbols(pf.ANALYSES_DIR)
    names |= set(overrides) | us_listed
    if not names:
        pytest.skip(f"record at {pf.ANALYSES_DIR} names no tickers")
    return names


def test_the_allowlist_has_not_rotted() -> None:
    stale = sorted(TICKER_ALLOWLIST - _record_tickers())
    assert stale == [], f"TICKER_ALLOWLIST names {stale}, no longer in the record. Remove them."


def _scan(patterns: dict[str, re.Pattern]) -> tuple[int, dict[str, list[str]]]:
    tracked = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files"], capture_output=True, text=True, timeout=60
    ).stdout.split()
    scanned = 0
    hits: dict[str, list[str]] = {}
    for rel in tracked:
        path = REPO_ROOT / rel
        if path.suffix not in _TEXT_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        scanned += 1
        found = sorted(t for t, pat in patterns.items() if pat.search(text))
        if found:
            hits[rel] = found
    return scanned, hits


def test_no_tracked_file_names_a_record_ticker() -> None:
    """A standalone-token match, so `A50` matches bare prose as well as a
    `2026-07-25-A50` folder name. The fix is to keep the measurement and drop
    the identity, not to delete the finding."""
    tickers = _record_tickers() - TICKER_ALLOWLIST
    patterns = {t: re.compile(rf"(?<![A-Za-z0-9]){re.escape(t)}(?![A-Za-z0-9])") for t in tickers}
    scanned, hits = _scan(patterns)
    assert scanned > 20, f"only {scanned} tracked text files scanned"
    assert hits == {}, (
        f"{len(hits)} tracked file(s) name a ticker from the private record:\n"
        + "\n".join(f"  {f}: {', '.join(t)}" for f, t in sorted(hits.items())[:15])
    )


def _wordlike_tickers() -> set[str]:
    """Record tickers whose lowercase form is ordinary English (so a case-
    insensitive scan for them is all noise). Read from the record, not
    hardcoded, since naming a ticker to exclude it would itself be a leak."""
    import json

    from gui.server import ANALYSES_DIR

    path = ANALYSES_DIR / "pseudonyms.json"
    if not path.is_file():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")).get("wordlike", []))


def test_the_wordlike_set_has_not_rotted() -> None:
    stale = sorted(_wordlike_tickers() - _record_tickers())
    assert stale == [], f"pseudonyms.json's wordlike list names {stale}, no longer in the record."


def test_no_tracked_file_names_a_record_ticker_in_lower_case() -> None:
    """The half a case-sensitive scan misses - a ticker used lowercase in code
    (a filter string, a test name) rather than as prose."""
    tickers = _record_tickers() - TICKER_ALLOWLIST - _wordlike_tickers()
    patterns = {
        t: re.compile(rf"(?<![A-Za-z0-9]){re.escape(t.lower())}(?![A-Za-z0-9])") for t in tickers
    }
    scanned, hits = _scan(patterns)
    assert scanned > 20, f"only {scanned} tracked text files scanned"
    assert hits == {}, (
        f"{len(hits)} tracked file(s) name a record ticker in lower case:\n"
        + "\n".join(f"  {f}: {', '.join(t)}" for f, t in sorted(hits.items())[:15])
    )


@pytest.mark.parametrize("name", PRIVATE_FILES)
def test_no_private_file_is_tracked(name: str) -> None:
    tracked = [line for line in _git("ls-files", name).splitlines() if line.strip()]
    assert tracked == [], f"{name} is tracked. Untrack with `git rm --cached {name}`."


@pytest.mark.parametrize("name", PRIVATE_FILES)
def test_gitignore_names_each_private_file_anchored(name: str) -> None:
    lines = {ln.strip() for ln in (REPO_ROOT / ".gitignore").read_text().splitlines()}
    assert f"/{name}" in lines, f".gitignore must contain the root-anchored line `/{name}`"
