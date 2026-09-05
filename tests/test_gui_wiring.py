"""A listener on a lasting element is bound once, on the boot path."""

from __future__ import annotations

import pathlib
import re
import typing

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
INDEX = REPO_ROOT / "gui" / "index.html"

#: Functions that run at most once per page load, so a listener bound on a
#: lasting element from inside one cannot stack. That property is checked by
#: `boot_call_offenders` rather than assumed - it is the whole of the second half.
BOOT_ONCE = ("boot", "wireMain")

#: The scan's name for code outside every function. Module scope runs once per
#: load, for the same reason and with the same licence as `BOOT_ONCE`.
MODULE = "<module>"

#: Functions the scan must find. Named so that renaming one reddens this anchor
#: instead of silently shrinking it - the vacuous-pass shape closed in
#: anchors 14, 16 and 17 (2026-07-26).
MUST_EXIST = ("boot", "wireMain", "render", "paint", "paintPortfolio")

#: A top-level function declaration: column 0, because that is this page's style
#: and nesting implies indentation.
_DECL = re.compile(r"(?:async\s+)?function\s+(\w+)\s*\(")

#: Its terminator: a closing brace in column 0. `sections()` asserts these
#: interleave with `_DECL` one for one, so a page that stops obeying the
#: convention reddens here instead of being mis-sectioned silently.
_CLOSE = re.compile(r"\}\s*$")

_LISTEN = re.compile(r"\.addEventListener\s*\(")

#: A local name bound to a shell lookup. `paint()` does exactly this
#: (`const el = $("#body")`), so without this the check would miss
#: `el.addEventListener(...)` two lines later - a rename away from the defect.
_ALIAS = re.compile(r"\b(?:const|let|var)\s+(\w+)\s*=\s*(?:\$\(|document\s*\.\s*querySelector)")

#: Receiver tails, tried in the order `_classify` tries them, and the order is
#: load-bearing twice over: `window` is an identifier too, so the global test must
#: precede the identifier one; and `document.querySelector("#main")` is a shell
#: lookup wearing a descendant's syntax, so the shell test must precede the
#: descendant one. Getting that second order wrong let a document-rooted lookup
#: pass as a freshly built child - caught here by its own control, not by review.
_GLOBAL = re.compile(r"\b(?:window|document)\s*$")
_SHELL = re.compile(r"(?:\$|document\s*\.\s*querySelector(?:All)?)\s*\([^()]*\)\s*$")
_DESCENDANT = re.compile(r"(\$\s*\([^()]*\)|[\w$]+)\s*\.\s*querySelector(?:All)?\s*\([^()]*\)\s*$")
_IDENT = re.compile(r"([A-Za-z_$][\w$]*)\s*$")

#: A line comment, blanked before anything is counted. The page discusses its own
#: boot path in prose - "Assigned in boot(), once the analysis list exists" - and
#: counting that as a call site made the boot path look like it ran twice. The
#: `(?<!:)` guard keeps `http://` in a template literal intact.
_LINE_COMMENT = re.compile(r"(?<!:)//[^\n]*")


class Section(typing.NamedTuple):
    name: str
    first: int
    last: int


class Reg(typing.NamedTuple):
    """One `addEventListener` call, and whether its target outlives a render."""

    section: str
    line: int
    receiver: str
    lasting: bool


def script_source(path: pathlib.Path = INDEX) -> str:
    """The page's JavaScript. Refuses to hand back an extraction that matched
    nothing, so no case below can assert about an empty string."""
    blocks = re.findall(r"<script[^>]*>(.*?)</script>", path.read_text(encoding="utf-8"), re.S)
    assert blocks, f"no <script> found in {path} - this scan would check nothing"
    return "\n".join(blocks)


def _code(src: str) -> str:
    """*src* with comment text blanked, positions and line numbering preserved."""
    return _LINE_COMMENT.sub(lambda m: " " * len(m.group(0)), src)


def sections(src: str) -> list[Section]:
    """Every top-level function, as (name, first line, last line)."""
    out: list[Section] = []
    name, first = None, 0
    for n, line in enumerate(_code(src).split("\n"), start=1):
        m = _DECL.match(line)
        if m:
            assert name is None, (
                f"line {n}: `function {m.group(1)}` opens while `function {name}` "
                f"(line {first}) is still open. This scan sections gui/index.html by "
                "column-0 declarations and column-0 closing braces; that convention "
                "no longer holds, so fix the page or teach this scan the new shape - "
                "do not leave it sectioning the file wrongly."
            )
            name, first = m.group(1), n
        elif name is not None and _CLOSE.match(line):
            out.append(Section(name, first, n))
            name = None
    assert name is None, (
        f"`function {name}` (line {first}) is never closed by a brace in column 0. "
        "See the sectioning note in this function's docstring."
    )
    return out


def _owners(src: str) -> tuple[dict[int, str], dict[str, set[str]]]:
    """Line number -> enclosing function, and function -> its shell aliases."""
    lines = _code(src).split("\n")
    owner = {n: MODULE for n in range(1, len(lines) + 1)}
    aliases: dict[str, set[str]] = {MODULE: set()}
    for sec in sections(src):
        body = "\n".join(lines[sec.first - 1 : sec.last])
        aliases.setdefault(sec.name, set()).update(_ALIAS.findall(body))
        for n in range(sec.first, sec.last + 1):
            owner[n] = sec.name
    for n, line in enumerate(lines, start=1):
        if owner[n] == MODULE:
            aliases[MODULE].update(_ALIAS.findall(line))
    return owner, aliases


def _classify(prefix: str, aliases: set[str]) -> tuple[str, bool]:
    """The receiver `.addEventListener` was called on, and whether it lasts."""
    prefix = prefix.rstrip()
    if m := _GLOBAL.search(prefix):
        return m.group(0).strip(), True
    if m := _SHELL.search(prefix):
        return m.group(0).strip(), True
    if m := _DESCENDANT.search(prefix):
        # A child fetched out of a container's freshly written innerHTML is new on
        # every render, which is `paintPortfolio`'s legitimate pattern. A lookup
        # rooted at `document` wears the same syntax and is not a descendant of
        # anything, which is why `_SHELL` is tried first.
        return m.group(0).strip(), False
    if m := _IDENT.search(prefix):
        return m.group(1), m.group(1) in aliases
    return prefix[-40:] or "<start of file>", True


def registrations(src: str) -> list[Reg]:
    """Every `addEventListener` in *src*, with its owner and its receiver."""
    owner, aliases = _owners(src)
    code = _code(src)
    out = []
    for m in _LISTEN.finditer(code):
        line = code.count("\n", 0, m.start()) + 1
        section = owner[line]
        receiver, lasting = _classify(code[: m.start()], aliases.get(section, set()))
        out.append(Reg(section, line, receiver, lasting))
    return out


def wiring_offenders(src: str) -> list[str]:
    """Listeners bound on a lasting element from somewhere that can run twice."""
    allowed = set(BOOT_ONCE) | {MODULE}
    return [
        f"{r.section}() line {r.line}: {r.receiver}.addEventListener - {r.receiver} "
        "survives the next render, so this binding stacks. Bind it once on the "
        "boot path and delegate, or bind on an element the caller has just built."
        for r in registrations(src)
        if r.lasting and r.section not in allowed
    ]


def boot_call_offenders(src: str) -> list[str]:
    """The boot path, entered exactly once, from the boot path or module scope."""
    owner, _ = _owners(src)
    code = _code(src)
    declared = {s.name for s in sections(src)}
    allowed = set(BOOT_ONCE) | {MODULE}
    out = []
    for callee in BOOT_ONCE:
        if callee not in declared:
            out.append(
                f"BOOT_ONCE names `{callee}` and gui/index.html no longer declares it. "
                "Renaming the boot path must redden this anchor, not shrink it."
            )
            continue
        sites = [
            owner[code.count("\n", 0, m.start()) + 1]
            for m in re.finditer(rf"(?<!function )\b{callee}\s*\(", code)
        ]
        sites = [s for s in sites if s != callee]  # its own recursion is not a call site
        if len(sites) != 1:
            out.append(
                f"`{callee}()` is called from {sorted(sites) or 'nowhere'} - expected "
                "exactly one site. Called twice, every listener it binds is bound "
                "twice; called never, nothing is bound at all."
            )
        for s in sites:
            if s not in allowed:
                out.append(
                    f"{s}() calls `{callee}()`, which binds listeners on elements that "
                    "outlive a render. If that caller can run twice, so can the binding."
                )
    return out


# --------------------------------------------------------------------------- #
# The real page
# --------------------------------------------------------------------------- #


def test_the_scan_sees_the_real_page() -> None:
    """The meta-guard. Every case below would pass against an empty string."""
    src = script_source()
    found = {s.name for s in sections(src)}
    missing = [fn for fn in MUST_EXIST if fn not in found]
    assert not missing, (
        f"{missing} are not top-level functions in gui/index.html any more. This "
        "anchor is about where listeners are bound; if the page reorganised, point "
        "it at the new shape rather than letting it check a page that has moved on."
    )

    regs = registrations(src)
    assert len(regs) == _code(src).count(".addEventListener"), (
        "the scan did not account for every addEventListener in the page's code: "
        f"{len(regs)} classified against {_code(src).count('.addEventListener')} present"
    )
    assert regs, "no listeners found at all - this anchor would police nothing"

    # The claim that is not true by construction: the scan reaches the three
    # functions that actually bind listeners, by name. Anchor 19 names cli.py and
    # report.py for the same reason - wiring that moves to a function this scan
    # cannot see would otherwise shrink the check silently.
    binders = {r.section for r in regs}
    assert binders >= {"boot", "wireMain", "paintPortfolio"}, (
        f"listeners are bound in {sorted(binders)}; expected the scan to find all "
        "of boot, wireMain and paintPortfolio. If the wiring moved, this anchor is "
        "now looking somewhere else."
    )

    lasting = [r for r in regs if r.lasting]
    fresh = [r for r in regs if not r.lasting]
    assert len(lasting) >= 2, (
        f"only {len(lasting)} lasting receivers found. Both branches of _classify "
        "must be reached by real code, or the controls below are the only evidence "
        "the classification means anything."
    )
    assert fresh, "no per-render registrations found - the false-positive branch is unreached"
    assert {r.section for r in fresh} - set(BOOT_ONCE), (
        "every fresh registration sits on the boot path, so the case that makes "
        "this check subtler than 'no listeners in render' is not actually live. "
        "paintPortfolio was that case."
    )


def test_no_lasting_listener_is_bound_outside_the_boot_path() -> None:
    """The original defect, structurally: `render()` bound on the persistent
    `#main`, so one click ran two 50,000-draw runs.
    """
    offenders = wiring_offenders(script_source())
    assert offenders == [], "\n".join(offenders)


def test_the_boot_path_is_entered_once() -> None:
    """Binding once is worth nothing if the binder runs twice."""
    offenders = boot_call_offenders(script_source())
    assert offenders == [], "\n".join(offenders)


# --------------------------------------------------------------------------- #
# Positive controls - the check fires on each shape of the defect
# --------------------------------------------------------------------------- #

_HISTORICAL = """
function render() {
  $("#main").innerHTML = "<button data-act='run'></button>";
  $("#main").addEventListener("click", e => act(e.target.dataset.act));
}
"""

_WINDOW_IN_PAINT = """
function paint() {
  window.addEventListener("resize", () => paint());
}
"""

_ALIASED = """
function paint() {
  const el = $("#body");
  el.innerHTML = "x";
  el.addEventListener("click", act);
}
"""

_DOCUMENT_QUERY = """
function render() {
  document.querySelector("#main").addEventListener("click", act);
}
"""

_UNCLASSIFIABLE = """
function render() {
  document.getElementById("main").addEventListener("click", act);
}
"""

_NESTED_IN_A_CALLBACK = """
function render() {
  ["a", "b"].forEach(k => {
    $("#main").addEventListener("click", () => act(k));
  });
}
"""


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("the historical defect, restored", _HISTORICAL),
        ("window from a repainting function", _WINDOW_IN_PAINT),
        ("a shell lookup held in a local", _ALIASED),
        ("a document-rooted query", _DOCUMENT_QUERY),
        ("a receiver the scan cannot read", _UNCLASSIFIABLE),
        ("buried in a callback", _NESTED_IN_A_CALLBACK),
    ],
)
def test_flags_bindings_that_stack(label: str, source: str) -> None:
    assert wiring_offenders(source), f"{label} not flagged"


_CALLED_FROM_RENDER = """
async function boot() {
  drawList();
}
function wireMain() {
  $("#main").addEventListener("click", act);
}
function render() {
  $("#main").innerHTML = "x";
  wireMain();
}
boot();
"""

_CALLED_TWICE = """
async function boot() {
  wireMain();
}
function wireMain() {
  $("#main").addEventListener("click", act);
}
function paint() {
  wireMain();
}
boot();
"""

_NEVER_CALLED = """
async function boot() {
  drawList();
}
function wireMain() {
  $("#main").addEventListener("click", act);
}
boot();
"""


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("the binder called from a per-render function", _CALLED_FROM_RENDER),
        ("the binder called from two places", _CALLED_TWICE),
        ("the binder never called at all", _NEVER_CALLED),
    ],
)
def test_flags_a_boot_path_entered_more_or_less_than_once(label: str, source: str) -> None:
    assert boot_call_offenders(source), f"{label} not flagged"


# --------------------------------------------------------------------------- #
# Negative controls - the legitimate shapes stay legal
# --------------------------------------------------------------------------- #

_FRESH_CHILDREN = """
function paintPortfolio() {
  $("#main").innerHTML = renderPortfolio(portfolioRows);
  $("#main").querySelectorAll(".pf-row").forEach(row =>
    row.addEventListener("click", () => select(row.dataset.open)));
}
"""

_FRESH_DESCENDANT = """
function paint() {
  $("#body").innerHTML = "<input id='price'>";
  $("#body").querySelector("#price").addEventListener("input", read);
}
"""

_ON_THE_BOOT_PATH = """
function wireMain() {
  $("#main").addEventListener("click", act);
  $("#main").addEventListener("keydown", key);
}
"""

_AT_MODULE_SCOPE = """
window.addEventListener("hashchange", open);
function render() {
  $("#main").innerHTML = "x";
}
"""


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("children of freshly written innerHTML", _FRESH_CHILDREN),
        ("one freshly written child", _FRESH_DESCENDANT),
        ("the boot path itself", _ON_THE_BOOT_PATH),
        ("module scope, which runs once", _AT_MODULE_SCOPE),
    ],
)
def test_allows_the_legitimate_shapes(label: str, source: str) -> None:
    assert wiring_offenders(source) == [], f"{label} wrongly flagged: {wiring_offenders(source)}"


def test_allows_the_real_boot_chain() -> None:
    """Module calls `boot`, `boot` calls `wireMain`, and neither is called twice."""
    source = """
async function boot() {
  wireMain();
}
function wireMain() {
  $("#main").addEventListener("click", act);
}
boot();
"""
    assert boot_call_offenders(source) == []


# --------------------------------------------------------------------------- #
# The sectioning precondition is asserted, not assumed
# --------------------------------------------------------------------------- #


def test_a_page_that_breaks_the_sectioning_convention_fails_loudly() -> None:
    """Mis-sectioning would attribute a binding to the wrong function, and the
    direction that matters is a defect attributed to `wireMain`, where it is
    allowed. So the convention is checked rather than trusted."""
    unclosed = '\nfunction render() {\n  $("#main").innerHTML = "x";\n'
    with pytest.raises(AssertionError, match="never closed"):
        sections(unclosed)

    nested = '\nfunction render() {\n  $("#main").innerHTML = "x";\nfunction paint() {\n}\n'
    with pytest.raises(AssertionError, match="still open"):
        sections(nested)
