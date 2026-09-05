"""The viewer's rendering, exercised against real payloads [gui/, no anchor]."""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
INDEX = REPO_ROOT / "gui" / "index.html"

#: Functions this file drives. Named so a rename reddens the harness rather than
#: silently shrinking what it covers.
EXERCISED = (
    "histogram",
    "renderLadder",
    "renderCompare",
    "rangeStrip",
    "renderList",
    "priceBox",
    "renderPortfolio",
    "renderFactors",
    "renderTimeline",
    "renderMarginals",
    "renderValidate",
    "renderRun",
    "renderConvergence",
    "renderWorlds",
)

#: Pure render functions this harness does not drive. Empty, and kept rather
#: than deleted: the partition test below needs somewhere honest to put a
#: function that genuinely cannot be driven, and an empty tuple states that no
#: such function exists today. It held four entries between the 2026-08-08 audit
#: that named them and the batch that closed them.
#:
#: What remains uncovered is the event-wiring half - listener registration and
#: dispatch - which needs jsdom or a browser. That is a dependency question,
#: not an omission, and this tuple is not where it belongs.
UNCOVERED = ()

#: The page's last statement. Stripped rather than stubbed around: it starts the
#: app, and the alternative is faking enough of the DOM for a boot that has
#: nothing to do with rendering.
BOOT_CALL = "\nboot();\n"

_DRIVER = """
import fs from "node:fs";
const [appPath, jobPath] = process.argv.slice(2);
// Enough of a browser for the module to evaluate. Nothing here is exercised:
// the boot call is stripped before this runs, so no DOM path is entered.
globalThis.document = { addEventListener(){}, querySelector: () => null,
                        querySelectorAll: () => [] };
globalThis.window = { addEventListener(){}, location: { hash: "" } };
globalThis.location = { protocol: "http:", hash: "" };

const src = fs.readFileSync(appPath, "utf8");
const api = new Function(src + `
  return { histogram, renderLadder, renderCompare, rangeStrip, renderList,
           priceBox, renderPortfolio, renderFactors, renderTimeline, renderMarginals,
           renderValidate, renderRun, renderConvergence, renderWorlds,
           setAnchor: v => { mosAnchor = v; },
           // The price basis is module state read by renderPortfolio, exactly as
           // the anchor is. Driving it from here keeps the render function pure
           // in the harness's sense: data in, HTML out.
           setBasis: (v, meta) => { repriced = v; portfolioMeta = meta; },
           // Probes, not render functions: they return JSON so the harness can
           // compare the client's ticker parsing against the server's.
           __tickerOf: names => JSON.stringify(names.map(n => tickerOf(n) ?? null)),
           __siblingRuns: (names, current) => {
             allNames = names; return JSON.stringify(siblingRuns(current)); },
           // renderRun reaches whatIfBanner, which reads cache.analysis for the
           // overridden-price case. Module state, driven exactly as the anchor
           // and the price basis are.
           setAnalysis: a => { cache.analysis = a; } };`)();

const job = JSON.parse(fs.readFileSync(jobPath, "utf8"));
if (job.anchor) api.setAnchor(job.anchor);
if (job.basis !== undefined) api.setBasis(job.basis, job.meta ?? null);
if (job.analysis !== undefined) api.setAnalysis(job.analysis);
process.stdout.write(api[job.fn](...job.args));
"""


def _node() -> str:
    """node, or a failure - never a skip. See this module's docstring."""
    exe = shutil.which("node")
    assert exe, (
        "node is required to exercise gui/index.html's render functions. It is "
        "not optional: skipping here would report coverage this suite does not "
        "have. Install node, or delete this file."
    )
    return exe


def _script_source() -> str:
    """The page's JavaScript, with its boot call removed."""
    html = INDEX.read_text(encoding="utf-8")
    blocks = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    assert blocks, f"no <script> found in {INDEX} - the harness would test nothing"
    src = "\n".join(blocks)
    missing = [fn for fn in EXERCISED if f"function {fn}(" not in src]
    assert not missing, (
        f"{missing} are not in gui/index.html any more. Renaming a render "
        "function must redden this harness, not shrink it silently."
    )
    assert src.endswith(BOOT_CALL), (
        "gui/index.html no longer ends with the boot call this harness strips. "
        f"Expected it to end with {BOOT_CALL!r}."
    )
    return src[: -len(BOOT_CALL)]


@pytest.fixture(scope="module")
def render(tmp_path_factory):
    """Call one render function in node and return the HTML it produces."""
    node, tmp = _node(), tmp_path_factory.mktemp("render")
    app = tmp / "app.js"
    app.write_text(_script_source(), encoding="utf-8")
    driver = tmp / "driver.mjs"
    driver.write_text(_DRIVER, encoding="utf-8")

    def call(
        fn: str, *args, anchor: str | None = None, basis=None, meta=None, analysis=None
    ) -> str:
        job = tmp / "job.json"
        payload = {"fn": fn, "args": args, "anchor": anchor}
        if basis is not None:
            payload["basis"] = basis
            payload["meta"] = meta
        if analysis is not None:
            payload["analysis"] = analysis
        job.write_text(json.dumps(payload), encoding="utf-8")
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [node, str(driver), str(app), str(job)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0, f"{fn} threw in node:\n{done.stderr}"
        return done.stdout

    return call


@pytest.fixture(scope="module")
def run_response():
    """A real `/api/run` response - real draws, real binning, no network."""
    from gui.server import ANALYSES_DIR, do_run

    name = sorted(p.name for p in ANALYSES_DIR.iterdir() if p.is_dir())[0]
    return do_run(name)


def _bar_counts(html: str) -> list[int]:
    """The count each bar carries, from its own tooltip."""
    bars = re.findall(r'title="[^"]* to [^"]*: ([\d,]+) draws"', html)
    return [int(m.replace(",", "")) for m in bars]


def test_every_render_function_is_placed_in_one_list_or_the_other() -> None:
    """A new render function is covered, or is declared uncovered. Never neither."""
    src = _script_source()
    on_page = set(re.findall(r"^function (render\w+)\(", src, re.M))
    placed = set(EXERCISED) | set(UNCOVERED)

    assert on_page, "no render functions found in the page - the harness sees nothing"
    unplaced = on_page - placed
    assert not unplaced, (
        f"{sorted(unplaced)} are render functions the harness neither drives nor "
        "declares uncovered. Add them to EXERCISED with cases, or to UNCOVERED "
        "with a reason."
    )
    stale = set(UNCOVERED) - on_page
    assert not stale, (
        f"{sorted(stale)} are listed as uncovered but no longer exist - the list "
        "is describing a page that has moved on"
    )
    assert not (set(EXERCISED) & set(UNCOVERED)), "a function cannot be both"


def test_the_harness_sees_the_real_page(render, run_response) -> None:
    """The meta-guard: an empty harness would pass every case below."""
    html = render("histogram", run_response["histogram"], run_response["payload"])
    assert "Distribution of value" in html and "<i " in html, (
        "the harness produced no recognisable histogram - it is testing nothing"
    )


def test_every_histogram_bar_reports_its_own_count(render, run_response) -> None:
    """One bar per bin, each carrying the count it was given."""
    hh = run_response["histogram"]
    html = render("histogram", hh, run_response["payload"])
    counts = _bar_counts(html)

    assert len(counts) == len(hh["counts"]), (
        f"rendered {len(counts)} bars for {len(hh['counts'])} bins"
    )
    assert counts == hh["counts"], "a bar reports a count the payload did not give it"


def test_clipped_draws_are_reported_and_not_folded_into_the_end_bars(render, run_response) -> None:
    """The honesty property the clipping fix exists for."""
    hh = run_response["histogram"]
    html = render("histogram", hh, run_response["payload"])
    counts = _bar_counts(html)

    assert sum(counts) == hh["n"] - hh["below"] - hh["above"], (
        f"bars sum to {sum(counts)} against n={hh['n']} less {hh['below']} below "
        f"and {hh['above']} above - clipped draws have been folded into a bar"
    )
    if hh["below"] or hh["above"]:
        assert "Axis clipped" in html, "dropped draws from the bars and said nothing"
        assert f"{hh['below']:,}" in html and f"{hh['above']:,}" in html, (
            "the caption does not report both tail counts"
        )


@pytest.mark.parametrize("anchor", ["p10", "p50", "p90"])
def test_the_ladder_margin_column_follows_the_anchor(render, anchor: str) -> None:
    """Header and cells move together, and read the anchor's own figures."""
    ladder = {
        "spec_price": 100.0,
        "centre": 100.0,
        "ladder": [
            {
                "price": 100.0,
                "p_undervalued": 0.5,
                "mos_p10": -0.111,
                "mos_p50": 0.222,
                "mos_p90": 0.333,
                "is_centre": True,
                "is_spec": True,
            }
        ],
    }
    html = render("renderLadder", ladder, anchor=anchor)
    expected = {"p10": "-11.1%", "p50": "+22.2%", "p90": "+33.3%"}[anchor]

    assert f"MoS ({anchor.upper()})" in html, "the header does not name the anchor"
    assert expected in html, f"the {anchor} column does not show its own margin"
    for other in {"-11.1%", "+22.2%", "+33.3%"} - {expected}:
        assert other not in html, f"showed {other} while anchored on {anchor}"


def test_compare_renders_the_whole_range(render) -> None:
    """Value and margin at all three quantiles, not the median alone."""
    side = lambda name, k: {  # noqa: E731 - a fixture shape, not a helper
        "name": name,
        "adapter": "reit_v2",
        "price": 100.0,
        "p_undervalued": 0.5,
        "value_p10": 80.0 + k,
        "value_p50": 100.0 + k,
        "value_p90": 130.0 + k,
        "mos_p10": -0.2,
        "mos_p50": 0.0 + k / 100,
        "mos_p90": 0.3,
        "warnings": 0,
        "constants": {},
        "drivers": {},
        "factors": [],
    }
    html = render("renderCompare", {"a": side("new", 1.0), "b": side("old", 0.0)})

    for label in ("Value P10", "Value P50", "Value P90", "MoS (P10)", "MoS (P50)", "MoS (P90)"):
        assert f"<td>{label}</td>" in html, f"the results table has no {label} row"


def test_the_range_strip_marks_the_conservative_anchor(render, run_response) -> None:
    """P25 reaches the run view, which is the one the reader actually opens."""
    p = run_response["payload"]
    html = render("rangeStrip", p)

    assert 'class="strip-p25"' in html, "the range strip does not mark the P25 anchor"
    assert "conservative anchor" in html, "the strip does not say what the hairline is"
    # Positioned between P10 and P50, not pinned to an edge.
    left = float(re.search(r'class="strip-p25" style="left:([\d.]+)%', html).group(1))
    assert 0.0 < left < 100.0, f"P25 marker is off the track at {left}%"


#: A stand-in for the sidebar's payload. Three tickers and two dates, so a
#: filter can be shown to discriminate on either half of a `<date>-<TICKER>`
#: name rather than only on the tail.
LISTING = [
    {"name": "2026-07-25-A12"},
    {"name": "2026-07-28-VICI"},
    {"name": "2026-07-31-A56"},
]


def _listed(html: str) -> list[str]:
    """The analysis names the sidebar actually offers, in order."""
    return re.findall(r'data-name="([^"]+)"', html)


def test_the_sidebar_lists_every_analysis_when_nothing_is_typed(render) -> None:
    """An empty filter is not a filter - it must not hide anything."""
    html = render("renderList", LISTING, "", None)
    assert _listed(html) == [a["name"] for a in LISTING]
    assert "data-portfolio" in html, "the portfolio entry is missing from the list"


def test_the_filter_matches_anywhere_in_the_name_and_ignores_case(render) -> None:
    """Substring, case-insensitive, on the whole `<date>-<TICKER>` string."""
    assert _listed(render("renderList", LISTING, "vici", None)) == ["2026-07-28-VICI"]
    assert _listed(render("renderList", LISTING, "VICI", None)) == ["2026-07-28-VICI"]
    assert _listed(render("renderList", LISTING, "07-31", None)) == ["2026-07-31-A56"]
    assert _listed(render("renderList", LISTING, "2026", None)) == [a["name"] for a in LISTING]


def test_every_typed_term_must_match(render) -> None:
    """Whitespace splits the query and all parts must hit, in any order."""
    assert _listed(render("renderList", LISTING, "vici 2026", None)) == ["2026-07-28-VICI"]
    assert _listed(render("renderList", LISTING, "2026 vici", None)) == ["2026-07-28-VICI"]
    assert _listed(render("renderList", LISTING, "vici a56", None)) == []


def test_a_filter_that_matches_nothing_says_so_and_keeps_the_way_back(render) -> None:
    """An empty pane is indistinguishable from a broken one - so it never is."""
    html = render("renderList", LISTING, "zzz", None)
    assert _listed(html) == []
    assert "No analysis matches" in html, "an empty result set said nothing"
    assert "data-portfolio" in html, "filtered away the way back to the portfolio"


def test_the_filter_reports_how_much_it_is_hiding(render) -> None:
    """`n of N`, so a reader can see the list is truncated by their own query."""
    assert "1 of 3" in render("renderList", LISTING, "vici", None)
    assert "2 of 3" in render("renderList", LISTING, "2026-07-2", None)
    assert " of 3" not in render("renderList", LISTING, "", None), (
        "counted matches with no filter applied - there is nothing being hidden"
    )


def test_the_selection_stays_marked_through_a_filter(render) -> None:
    """Filtering redraws the list, and a redraw must not lose the highlight."""
    html = render("renderList", LISTING, "2026", "2026-07-28-VICI")
    active = re.findall(r'class="item active" data-name="([^"]+)"', html)
    assert active == ["2026-07-28-VICI"], f"marked {active} active"

    # The portfolio entry uses the same sentinel channel, not a second one.
    pf = render("renderList", LISTING, "", "@all")
    assert re.search(r'class="item active"[^>]*data-portfolio', pf), (
        "the portfolio entry does not mark itself active"
    )
    assert 'class="item active" data-name' not in pf, "marked an analysis active too"


def test_the_typed_query_is_escaped_where_it_is_echoed(render) -> None:
    """The one place this viewer echoes user input back into its own markup."""
    html = render("renderList", LISTING, "<img src=x onerror=1>", None)
    assert "<img" not in html, "echoed the typed query as live markup"
    assert "&lt;img" in html, "the query is not shown back to the reader at all"


# --------------------------------------------------------------------------- #
# The sidebar shows age, and can group a company's runs together
# --------------------------------------------------------------------------- #
#
# The portfolio has flagged stale rows since it shipped; the sidebar has not, so
# age was invisible until after opening something - the wrong order for a list
# whose whole job is helping you choose what to open. Grouping is the same
# omission in the other axis: entries sorted by date alone scatter a company's
# runs across the list, and the runs that can be compared are exactly the ones
# the Timeline button is offered for.

#: Two runs of one ticker, one run of two others, and one undated folder. Ages
#: straddle the stale threshold deliberately: 7 is not stale and 8 is, which is
#: the boundary the portfolio already draws.
LISTING_AGED = [
    {"name": "2026-08-13-A12", "as_of": "2026-08-13", "age_days": 0},
    {"name": "2026-08-06-VICI", "as_of": "2026-08-06", "age_days": 7},
    {"name": "2026-07-25-A12", "as_of": "2026-07-25", "age_days": 8},
    {"name": "2026-07-20-A56", "as_of": "2026-07-20", "age_days": 25},
    {"name": "notes", "as_of": None, "age_days": None},
]


def _badges(html: str) -> dict[str, str]:
    """Each entry's rendered age badge, keyed by analysis name."""
    out = {}
    for name, rest in re.findall(r'data-name="([^"]+)"[^>]*>(.*?)</div>', html, re.S):
        m = re.search(r'<span class="age[^"]*">([^<]*)</span>', rest)
        out[name] = m.group(1) if m else ""
    return out


def test_every_entry_shows_its_age_in_the_list(render) -> None:
    """Days since the analysis date, and "today" for a same-day run."""
    badges = _badges(render("renderList", LISTING_AGED, "", None, False))
    assert badges["2026-08-13-A12"] == "today"
    assert badges["2026-08-06-VICI"] == "7d"
    assert badges["2026-07-20-A56"] == "25d"


def test_the_sidebar_marks_stale_at_the_same_boundary_as_the_portfolio(render) -> None:
    """Over seven days, matching `STALE_DAYS`. Seven itself is not stale."""
    html = render("renderList", LISTING_AGED, "", None, False)
    stale = set(re.findall(r'data-name="([^"]+)"[^>]*>\s*[^<]*<span class="age stale"', html))
    assert "2026-07-25-A12" in stale, "8 days old and not marked stale"
    assert "2026-07-20-A56" in stale, "25 days old and not marked stale"
    assert "2026-08-06-VICI" not in stale, "7 days is the threshold, not past it"
    assert "2026-08-13-A12" not in stale, "today's analysis marked stale"


def test_an_undated_folder_gets_no_badge_rather_than_a_wrong_one(render) -> None:
    """`None` age renders as nothing. Not "undefined", and not "today" - an
    entry this scheme cannot date is not fresh, it is undated."""
    html = render("renderList", LISTING_AGED, "", None, False)
    assert _badges(html)["notes"] == "", "an undated entry rendered an age badge"
    assert "undefined" not in html and "NaN" not in html


def test_grouping_keeps_every_analysis(render) -> None:
    """The property that makes the toggle safe: it re-orders, it never filters."""
    plain = set(_listed(render("renderList", LISTING_AGED, "", None, False)))
    grouped = set(_listed(render("renderList", LISTING_AGED, "", None, True)))
    assert grouped == plain, f"grouping changed the set: {plain ^ grouped}"


def test_grouping_gathers_a_ticker_s_runs_under_one_heading(render) -> None:
    """A12's two runs are adjacent and under a A12 heading, newest first."""
    html = render("renderList", LISTING_AGED, "", None, True)
    heads = re.findall(r'data-group-ticker="([^"]*)"', html)
    assert "A12" in heads, f"no A12 heading in {heads}"

    order = _listed(html)
    assert order.index("2026-08-13-A12") + 1 == order.index("2026-07-25-A12"), (
        f"A12's runs are not adjacent: {order}"
    )
    assert "2 runs" in html, "the heading does not say how many runs it gathers"


def test_single_run_tickers_stay_together_rather_than_each_getting_a_heading(render) -> None:
    """A heading per ticker is noise, not grouping, once most tickers have a
    single run. Only tickers with something to compare earn one; the rest keep
    their date order under a single heading."""
    html = render("renderList", LISTING_AGED, "", None, True)
    heads = re.findall(r'data-group-ticker="([^"]*)"', html)
    assert heads.count("VICI") == 0 and heads.count("A56") == 0, (
        f"a single-run ticker got its own heading: {heads}"
    )
    assert "" in heads, "no heading for the single runs"

    order = _listed(html)
    assert order.index("2026-08-06-VICI") < order.index("2026-07-20-A56"), (
        "single runs lost their date order"
    )


def test_grouping_and_the_filter_compose(render) -> None:
    """Both narrow the same list, so they have to hold at once."""
    html = render("renderList", LISTING_AGED, "a12", None, True)
    assert _listed(html) == ["2026-08-13-A12", "2026-07-25-A12"]
    assert "2 of 5" in html, "the count line stopped reporting what is hidden"


def test_the_group_toggle_states_which_way_it_is_set(render) -> None:
    """Offered in both states, and marked in the one it is in - a toggle that
    looks identical on and off is a toggle you press to find out."""
    off = render("renderList", LISTING_AGED, "", None, False)
    on = render("renderList", LISTING_AGED, "", None, True)
    assert 'data-group="1"' in off and 'data-group="1"' in on
    assert re.search(r'data-group="1"[^>]*class="[^"]*on', on), "the toggle is not marked when on"
    assert not re.search(r'data-group="1"[^>]*class="[^"]*on', off), "marked on while off"


def test_the_staleness_rule_is_written_once(render) -> None:
    """One definition, two callers. `adapters/floors.py`'s reasoning, in the
    viewer: three copies drift and the weakened one goes unnoticed because the
    others stay right. So the threshold comparison lives in `ageInfo` alone."""
    src = _script_source()
    comparisons = re.findall(r"[<>]=?\s*STALE_DAYS|STALE_DAYS\s*[<>]=?", src)
    assert len(comparisons) == 1, (
        f"the stale threshold is compared in {len(comparisons)} places; it belongs "
        "in ageInfo() alone, so the sidebar and the portfolio cannot disagree"
    )
    for caller in ("function renderList", "function renderPortfolio"):
        body = src[src.index(caller) : src.index(caller) + 4000]
        assert "ageInfo(" in body, f"{caller} does not use the shared ageInfo()"


# --------------------------------------------------------------------------- #
# The portfolio declares what a row has to declare, in the ranking
# --------------------------------------------------------------------------- #
#
# Neither a floored draw nor a degenerate result is a result. They were
# surfaced in the run
# view only, so the ranked table - the surface someone actually scans - showed
# "1 warning(s)" and left the kind to be discovered by opening the row.
#
# Which branches the shipped set can reach was counted before these cases were
# written (`tests/test_gui_server.py`): some rows floor draws, exercising that
# badge with real data; none reach a degenerate `p_undervalued`, so the case
# below plants it and says so rather than implying coverage the data cannot
# give.

#: A minimal ok row. The fields the flags column does not read are still present,
#: because renderPortfolio reads them for other columns.
PF_ROW = {
    "name": "2026-08-14-AAA",
    "status": "ok",
    "age_days": 0,
    "as_of": "2026-08-14",
    "p_undervalued": 0.62,
    "price": 74.90,
    "spec_price": 74.90,
    "repriced": False,
    "warnings": 0,
    "diagnostics": [],
    "degenerate": False,
    "value_p10": 60.0,
    "value_p25": 70.0,
    "value_p50": 80.0,
    "value_p90": 95.0,
    "mos_p10": -0.2,
    "mos_p25": -0.06,
    "mos_p50": 0.07,
    "mos_p90": 0.27,
}

#: The worst real case in the shipped set (`2026-07-24-A28`, 48/50,000 draws).
FLOORED = {
    "name": "dcf_terminal_spread_floored",
    "count": 48,
    "rate": 0.00096,
    "rate_text": "0.10%",
}

#: The common real case: 12 of the 13 fire on fewer than 5 draws, where a
#: percentage rounds to "0.00%" and the engine says "<0.01%" instead.
BARELY = {
    "name": "ddm_terminal_spread_floored",
    "count": 1,
    "rate": 0.00002,
    "rate_text": "<0.01%",
}

PF_META = {"current": {"available": False}}


def _flags_cell(html: str, name: str) -> str:
    """The Flags cell of one row - the last `<td>` before the row closes."""
    row = re.search(rf'data-open="{re.escape(name)}".*?</tr>', html, re.S)
    assert row, f"no row for {name}"
    return re.findall(r"<td[^>]*>(.*?)</td>", row.group(0), re.S)[-1]


def _visible(cell: str) -> str:
    """The cell's visible text, attributes stripped."""
    return re.sub(r"<[^>]*>", " ", cell)


def test_a_floored_row_says_so_in_the_ranking(render) -> None:
    """The count, the rate and the flag's own name - all three are needed."""
    rows = [{**PF_ROW, "warnings": 1, "diagnostics": [FLOORED]}]
    cell = _flags_cell(render("renderPortfolio", rows, basis=False, meta=PF_META), PF_ROW["name"])
    shown = _visible(cell)

    assert "48" in shown, "the floored draw count is not shown"
    assert "0.10%" in shown, "the rate is not shown"
    assert "dcf_terminal_spread_floored" in shown, (
        f"the flag name is not in the badge's visible text: {shown!r}"
    )


def test_a_rate_too_small_to_show_is_not_rendered_as_zero(render) -> None:
    """The case that is 12 of the 13 real ones. "0.00%" reads as "no draws were
    floored", which is the disclosure saying the opposite of what it means.
    """
    rows = [{**PF_ROW, "warnings": 1, "diagnostics": [BARELY]}]
    cell = _flags_cell(render("renderPortfolio", rows, basis=False, meta=PF_META), PF_ROW["name"])

    shown = _visible(cell)
    assert "&lt;0.01%" in shown or "<0.01%" in shown, f"rate not shown as a floor: {shown!r}"
    assert "0.00%" not in cell, "rounded a real floored draw down to nothing"


def test_both_flags_on_one_row_are_both_shown(render) -> None:
    """6 of the 13 fire two flags (A03, A26 and both A36 runs). Showing one
    would report a partial diagnosis as a complete one."""
    rows = [{**PF_ROW, "warnings": 2, "diagnostics": [FLOORED, BARELY]}]
    cell = _flags_cell(render("renderPortfolio", rows, basis=False, meta=PF_META), PF_ROW["name"])

    shown = _visible(cell)
    assert "dcf_terminal_spread_floored" in shown and "ddm_terminal_spread_floored" in shown


def test_a_degenerate_row_is_marked_as_an_artefact_not_a_result(render) -> None:
    """Planted, deliberately: no shipped analysis reaches this."""
    rows = [{**PF_ROW, "p_undervalued": 1.0, "warnings": 1, "degenerate": True}]
    html = render("renderPortfolio", rows, basis=False, meta=PF_META)
    cell = _flags_cell(html, PF_ROW["name"])

    assert "artefact" in _visible(cell).lower(), (
        f"a degenerate p_undervalued is shown as an ordinary 1.0000: {cell}"
    )
    assert "1.0000" in html, "the number is hidden rather than qualified"


def test_a_clean_row_declares_nothing_in_the_table(render) -> None:
    """No false positives. Anchor 13 requires this of the golden spec and the
    same must hold of the column that reports it: a badge on every row is a
    badge nobody reads."""
    html = render("renderPortfolio", [PF_ROW], basis=False, meta=PF_META)
    cell = _flags_cell(html, PF_ROW["name"])
    assert "floored" not in cell.lower()
    assert "artefact" not in cell.lower()
    assert cell.strip() in ("", '<td class="tag"></td>'), f"clean row declared {cell!r}"


def test_a_warning_this_column_cannot_draw_is_still_counted(render) -> None:
    """The partition, and the reason the engine's own total is still read."""
    rows = [{**PF_ROW, "warnings": 3, "diagnostics": [FLOORED]}]
    shown = _visible(
        _flags_cell(render("renderPortfolio", rows, basis=False, meta=PF_META), PF_ROW["name"])
    )
    assert "2 more" in shown, f"3 warnings, 1 drawn, and the other 2 went unmentioned: {shown!r}"


def test_a_row_from_an_older_payload_without_the_fields_still_renders(render) -> None:
    """The fields were added additively and the report schema did not bump, so a
    payload without them must degrade rather than print `undefined` - and must
    still declare that the row has something to declare, via the count."""
    older = {k: v for k, v in PF_ROW.items() if k not in ("diagnostics", "degenerate")}
    html = render("renderPortfolio", [{**older, "warnings": 1}], basis=False, meta=PF_META)

    assert "undefined" not in html and "NaN" not in html
    assert "1 more" in _visible(_flags_cell(html, PF_ROW["name"])), (
        "an older payload's warning count was dropped rather than shown"
    )


# --------------------------------------------------------------------------- #
# The tornado says what each bar is worth
# --------------------------------------------------------------------------- #
#
# The bar is a Spearman rank correlation, which is scale-free: it reports how
# monotonically value tracks a driver and never by how much. Measured on the
# golden spec, `div_growth` is the fourth bar (+0.450) and moves the value least
# of the six (0.376), while `affo_terminal` is drawn shorter (+0.317) and moves it
# three times as far (1.146). See `tests/test_swings.py` for why - leg weight and
# borrowed correlation, neither visible in a rank correlation.
#
# So the bar keeps its length and its ranking, and the swing is shown beside it.
# Nothing here renormalises the bar: rescaling it by band width is the
# constants-elasticity instrument this project rejected.


@pytest.fixture(scope="module")
def vici_run():
    """A real run of the golden spec, which is where the numbers above come
    from. Not the first analysis on disk, because these cases assert the
    inversion and it is a property of this spec."""
    from conftest import VICI_SPEC_PATH

    from gui.server import _histogram, _sensitivity
    from valdist import load_spec, report
    from valdist.spec.loader import hydrate

    spec = load_spec(VICI_SPEC_PATH)
    model = hydrate(spec)
    result = model.run(n=spec.n, price=spec.price, seed=spec.seed, nu=spec.nu)
    return {
        "payload": json.loads(report.to_json(result)),
        "text": report.to_text(result),
        "histogram": _histogram(result.value, spec.price),
        "sensitivity": _sensitivity(spec, model),
        "spec_price": spec.price,
        "price_used": spec.price,
        "overridden": False,
        "schema": {"supported": True},
    }


def _tornado_rows(html: str) -> list[str]:
    """Each tornado row's markup, in the order the panel draws them."""
    panel = re.search(r"Tornado.*?(?=<h3|$)", html, re.S)
    assert panel, "no tornado panel"
    return re.findall(
        r'<div class="dv-name">.*?(?=<div class="dv-name">|</div></div>|$)', panel.group(0), re.S
    )


def test_every_tornado_bar_says_what_it_is_worth(render, vici_run) -> None:
    """The swing, in the value's own units, beside the correlation."""
    html = render("renderRun", vici_run)
    rows = {re.search(r'dv-name">([^<]+)', r).group(1): r for r in _tornado_rows(html)}

    assert "cap_rate" in rows, f"tornado rows not found: {list(rows)}"
    assert "7.25" in rows["cap_rate"], f"cap_rate's swing of 7.252 is not shown: {rows['cap_rate']}"
    assert "0.38" in rows["div_growth"], "div_growth's swing of 0.376 is not shown"


def test_every_tornado_bar_shows_the_band_it_came_from(render, vici_run) -> None:
    """ "Does this matter?" and "was I vague about it?" are different questions,
    and the bar alone answers neither separately. The stated triple is what the
    analyst typed, which is the half a reader can act on."""
    html = render("renderRun", vici_run)
    rows = {re.search(r'dv-name">([^<]+)', r).group(1): r for r in _tornado_rows(html)}

    assert "5.5" in rows["cap_rate"] and "7.5" in rows["cap_rate"], (
        f"cap_rate's stated 5.50-7.50 band is not shown: {rows['cap_rate']}"
    )
    assert "8.25" in rows["cost_of_equity"] and "9.75" in rows["cost_of_equity"]


def test_the_bar_ranking_is_left_alone(render, vici_run) -> None:
    """The bars keep the tornado's own order and length. Re-ranking them by swing
    would silently answer a different question under the same title, and
    rescaling them by band width is the rejected constants-elasticity
    instrument."""
    html = render("renderRun", vici_run)
    order = [re.search(r'dv-name">([^<]+)', r).group(1) for r in _tornado_rows(html)]

    assert order == [x["driver"] for x in vici_run["payload"]["tornado"]], (
        f"the tornado re-ordered itself: {order}"
    )
    assert order.index("div_growth") < order.index("affo_terminal"), (
        "div_growth is no longer drawn above affo_terminal, which is the inversion "
        "these cases exist to make visible rather than to fix"
    )


def test_the_swing_is_drawn_to_scale_so_the_inversion_is_visible(render, vici_run) -> None:
    """A number in a column is read; a bar is seen. div_growth's swing bar must
    be visibly shorter than affo_terminal's even though its tornado bar is
    longer - that contrast is the whole finding."""
    html = render("renderRun", vici_run)
    rows = {re.search(r'dv-name">([^<]+)', r).group(1): r for r in _tornado_rows(html)}

    def swing_width(row: str) -> float:
        m = re.search(r'class="sw-bar"[^>]*width:([\d.]+)%', row)
        assert m, f"no swing bar drawn in row: {row}"
        return float(m.group(1))

    assert swing_width(rows["cap_rate"]) == pytest.approx(100.0, abs=0.5), (
        "the largest swing does not fill the track, so the scale is not the max"
    )
    assert swing_width(rows["div_growth"]) < swing_width(rows["affo_terminal"]), (
        "div_growth's swing bar is not shorter than affo_terminal's, so the "
        "inversion is invisible in exactly the place it should be obvious"
    )
    assert swing_width(rows["div_growth"]) < 10.0, (
        "a swing of 0.376 against a 7.252 maximum should be a stub, not a bar"
    )


def test_a_band_the_model_does_not_draw_is_flagged(render) -> None:
    """When the stated band and the swing come from different numbers, the row
    says so.
    """
    from conftest import skip_unless_fixture_corpus

    skip_unless_fixture_corpus()
    from gui.server import do_run

    out = do_run("2026-06-30-DRIFTCO")
    html = render("renderRun", out)
    rows = {re.search(r'dv-name">([^<]+)', r).group(1): r for r in _tornado_rows(html)}

    assert "sw-drift" in rows["div_growth"], f"the drifted band is not marked: {rows['div_growth']}"
    assert "sw-drift" not in rows["cost_of_equity"], "an undrifted driver was flagged"


def test_the_panel_says_the_two_columns_can_disagree(render, vici_run) -> None:
    """Two rankings side by side look like a bug unless the panel says otherwise.
    This is the same reason the timeline states its price/view split in words
    rather than leaving it to arithmetic."""
    html = render("renderRun", vici_run)
    panel = re.search(r"Tornado.*?(?=<h3|$)", html, re.S).group(0)
    assert "rank" in panel.lower() and "swing" in panel.lower()
    assert "magnitude" in panel.lower() or "how much" in panel.lower(), (
        "the caption does not say that the bar is scale-free, which is the one "
        "thing a reader needs to know to read the two columns together"
    )


def test_a_run_without_the_sensitivity_field_still_draws_its_tornado(render, vici_run) -> None:
    """Added additively and outside the report schema, so an older response must
    degrade to the tornado it always had."""
    older = {k: v for k, v in vici_run.items() if k != "sensitivity"}
    html = render("renderRun", older)

    assert _tornado_rows(html), "the tornado disappeared without its annotation"
    assert "undefined" not in html and "NaN" not in html


def test_the_range_strip_survives_a_payload_without_p25(render) -> None:
    """An older payload must render, not break."""
    old = {"value_p10": 10.0, "value_p50": 20.0, "value_p90": 30.0, "price": 15.0}
    html = render("rangeStrip", old)

    assert "Value per share" in html, "the strip did not render at all without P25"
    assert 'class="strip-p25"' not in html, "drew a P25 marker with no P25 in the payload"
    assert "NaN" not in html, "emitted NaN into the markup for a missing P25"


# --------------------------------------------------------------------------- #
#
# Which price the viewer opens with.
#
# Both surfaces used to open at the spec's own price, with the current quote one
# click away. The spec price is the number the analyst typed on the analysis
# date, so on a set spanning days to weeks the view opened at prices nobody is
# trading at - and the reader had to click per analysis to find out.
#
# The reason this is a render-level test and not a note in a docstring: the
# default is not a constant, it is a fallback chain (quote, else spec, else
# nothing), and the two ways it can be wrong are silent. Defaulting to the quote
# when there is none renders an empty price box; defaulting away from the spec
# price without leaving a way back strands the reader at a what-if with no route
# to the published number. Neither throws.

#: A page payload with a quote, as `/api/analysis` returns one.
QUOTED = {
    "name": "2026-08-08-XYZ",
    "spec_price": 71.20,
    "ticker": "XYZ",
    "as_of": "2026-08-08",
    "age_days": 0,
    "current_price": 74.90,
    "current_as_of": "2026-08-08",
    "current_live": False,
    "currency": "USD",
}
#: The same analysis with nothing in `prices.json` - the untracked-file case.
UNQUOTED = {**QUOTED, "current_price": None, "current_as_of": None, "current_live": False}


def test_the_price_box_opens_at_the_current_quote(render) -> None:
    """The change itself: an untouched field carries the quote, not the spec."""
    html = render("priceBox", QUOTED, None, False)

    assert 'value="74.9"' in html or 'value="74.90"' in html, (
        "the price box opened at something other than the current quote"
    )
    assert 'value="71.2"' not in html, "opened at the spec price with a quote available"


def test_the_price_box_falls_back_to_the_spec_price_with_no_quote(render) -> None:
    """The fallback, and the reason the default is a chain rather than a value."""
    html = render("priceBox", UNQUOTED, None, False)

    assert 'value="71.2"' in html, "no quote and no spec price either - the field is empty"
    assert "no quote" in html, "did not say why the current price is unavailable"


def test_the_spec_price_stays_one_click_away(render) -> None:
    """A default away from the published number must leave a route back to it."""
    html = render("priceBox", QUOTED, None, False)

    assert 'data-act="use-spec"' in html, "no way back to the analysis's own price"
    assert "71.20" in html, "the spec price is not shown, so its button cannot be read"


def test_a_typed_price_outranks_the_default(render) -> None:
    """The override is still the override. A default is a starting point."""
    html = render("priceBox", QUOTED, "88.5", False)

    assert 'value="88.5"' in html
    assert 'value="74.9"' not in html, "a typed price was overwritten by the quote"


def test_a_live_quote_is_used_and_labelled(render) -> None:
    """Used, because a moving price is still the current one - and marked as such."""
    html = render("priceBox", {**QUOTED, "current_live": True}, None, False)

    assert 'value="74.9"' in html, "held at the spec price because the quote was live"
    assert "live" in html.lower(), "used a moving intraday price without saying so"


def test_the_portfolio_banner_does_not_claim_spec_prices_when_repriced(render) -> None:
    """The banner's own text was unconditional, and the default makes it load-bearing."""
    rows = [
        {
            "name": "2026-08-08-XYZ",
            "status": "ok",
            "age_days": 0,
            "as_of": "2026-08-08",
            "p_undervalued": 0.62,
            "price": 74.90,
            "spec_price": 71.20,
            "repriced": True,
            "current_live": False,
            "current_as_of": "2026-08-08",
            "warnings": 0,
            "value_p10": 60.0,
            "value_p25": 70.0,
            "value_p50": 80.0,
            "value_p90": 95.0,
            "mos_p10": -0.2,
            "mos_p25": -0.06,
            "mos_p50": 0.07,
            "mos_p90": 0.27,
        }
    ]
    meta = {"current": {"available": True, "as_of": "2026-08-08"}}
    priced = render("renderPortfolio", rows, basis=True, meta=meta)
    as_analysed = render(
        "renderPortfolio",
        [{**rows[0], "repriced": False, "price": 71.20}],
        basis=False,
        meta=meta,
    )

    assert "not re-priced ones" not in priced, (
        "told the reader these are the shipped, un-re-priced numbers while every "
        "row beneath was re-priced"
    )
    assert "not re-priced ones" in as_analysed, (
        "the as-analysed basis no longer states that it is the shipped numbers - "
        "the claim has to survive on the side where it is true"
    )


# --------------------------------------------------------------------------- #
#
# Λ, rendered.
#
# The panel's job is review: loadings are specified by judgment and never
# fitted, a discipline no code can enforce - review is what enforces it.
# So the cases below are about what a reviewer can and cannot see, not about
# markup shape: a sign that reads backwards, an explicit zero that renders as a
# blank, and a driver whose commitment is invisible are each a wrong review
# rather than a wrong pixel.

#: Two factors, and drivers chosen so every readable state appears once: opposed
#: signs on one row, a driver committed to nothing, and one fully committed.
LAMBDA = {
    "name": "2026-06-28-VICI",
    "factors": ["rate", "fundamentals"],
    "unloaded_factors": [],
    "drivers": [
        {"name": "cap_rate", "loadings": [0.75, -0.30], "sum_sq": 0.6525, "psi": 0.589491},
        {"name": "nav_other", "loadings": [0.0, 0.0], "sum_sq": 0.0, "psi": 1.0},
        {"name": "cost_of_equity", "loadings": [1.0, 0.0], "sum_sq": 1.0, "psi": 0.0},
    ],
}


def test_the_sign_of_every_loading_is_shown(render) -> None:
    """A loading's sign is the reviewable half - it is the economic claim."""
    html = render("renderFactors", LAMBDA)

    assert "+0.750" in html and "-0.300" in html, "loadings are not shown with their sign"
    assert "lam-pos" in html and "lam-neg" in html, (
        "positive and negative loadings render identically, so the panel shows "
        "magnitude without direction"
    )


def test_an_explicit_zero_is_rendered_and_not_left_blank(render) -> None:
    """Zero is a statement, not an absence."""
    html = render("renderFactors", LAMBDA)

    assert html.count("0.000") >= 3, (
        "explicit zero loadings are not rendered, so a driver that loads on "
        "nothing looks like a driver whose row failed to render"
    )


def test_the_variance_budget_is_shown_per_driver(render) -> None:
    """Σλ² and ψ together, because either alone under-reports the other."""
    html = render("renderFactors", LAMBDA)

    # 0.652, not 0.653: 0.75² + 0.30² = 0.6525, and the nearest double to that
    # is a hair below it, so toFixed(3) rounds down. Pinned to what the panel
    # actually prints rather than to the arithmetic a reader would do by hand.
    assert "0.652" in html, "the commonality of a partially loaded driver is not shown"
    assert "0.589" in html, "psi is not shown"
    for label in ("&Sigma;", "&psi;"):
        assert label in html, f"the {label} column is unlabelled"


def test_a_fully_committed_driver_is_called_out(render) -> None:
    """Σλ² = 1 means ψ = 0: the draw is a deterministic function of the factors."""
    html = render("renderFactors", LAMBDA)
    assert "fully committed" in html.lower(), (
        "a driver with no idiosyncratic share left is rendered like any other"
    )


def test_a_declared_factor_nothing_loads_on_is_flagged(render) -> None:
    """Inert, valid, and the shape a mis-typed driver key leaves behind."""
    html = render("renderFactors", {**LAMBDA, "unloaded_factors": ["fx_em"]})

    assert "fx_em" in html
    assert "no driver loads on" in html.lower(), (
        "an inert factor is listed in the header like a live one"
    )


def test_the_panel_renders_a_spec_with_no_factors_at_all(render) -> None:
    """`FactorModel([], {})` is how independence is expressed, not an error."""
    flat = {
        "name": "x",
        "factors": [],
        "unloaded_factors": [],
        "drivers": [{"name": "a", "loadings": [], "sum_sq": 0.0, "psi": 1.0}],
    }
    html = render("renderFactors", flat)

    assert "independent" in html.lower(), "an independence spec renders as an empty table"
    assert "NaN" not in html and "undefined" not in html


# --------------------------------------------------------------------------- #
#
# The ticker timeline.
#
# The rendering carries two claims the payload cannot make on its own, and both
# are the kind that fail silently:
#
#   * every strip must sit on one shared axis. Per-strip axes would rescale each
#     band to its own extent, so two runs whose medians moved 8% apart would draw
#     identically - a picture that shows drift by construction and therefore
#     shows nothing. The worlds view splits axes for the opposite reason (its
#     drivers have incompatible scales); here they are the same company in the
#     same units, so sharing is required, not optional.
#   * the two p_undervalued columns must be labelled as what they are. Shown
#     without saying which is which, the reader takes the saved one - and on A12
#     that over-reports the change in view by a factor of three.

#: The real A12 pair, measured. Kept as literal numbers rather than fetched so
#: the render cases stay independent of what is on disk.
TIMELINE = {
    "ticker": "A12",
    "common_price": 74.90,
    "runs": [
        {
            "name": "2026-07-19-A12",
            "as_of": "2026-07-19",
            "status": "ok",
            "age_days": 20,
            "spec_price": 79.17,
            "warnings": 0,
            "value_p10": 54.50,
            "value_p25": 61.40,
            "value_p50": 70.30,
            "value_p90": 92.79,
            "p_undervalued": 0.2862,
            "p_undervalued_common": 0.3823,
        },
        {
            "name": "2026-07-25-A12",
            "as_of": "2026-07-25",
            "status": "ok",
            "age_days": 14,
            "spec_price": 74.90,
            "warnings": 0,
            "value_p10": 56.16,
            "value_p25": 63.21,
            "value_p50": 72.20,
            "value_p90": 94.51,
            "p_undervalued": 0.4272,
            "p_undervalued_common": 0.4272,
        },
    ],
}


def _strip_domains(html: str) -> list[tuple[float, float]]:
    """Each strip's rendered band as (left%, left%+width%)."""
    import re as _re

    return [
        (float(a), float(a) + float(b))
        for a, b in _re.findall(r'class="tl-band" style="left:([\d.]+)%;width:([\d.]+)%"', html)
    ]


def test_every_strip_sits_on_one_shared_axis(render) -> None:
    """The load-bearing property of the whole view."""
    html = render("renderTimeline", TIMELINE)
    domains = _strip_domains(html)

    assert len(domains) == 2, f"expected one band per run, got {len(domains)}"
    assert domains[0] != domains[1], (
        "both runs rendered to identical geometry from different value bands, "
        "which is what a per-strip axis does - the picture would show no drift "
        "no matter how far the view moved"
    )
    # The later run is higher on every quantile, so its band must sit to the right.
    assert domains[1][0] > domains[0][0], "the later run's band is not further right"


def test_both_p_undervalued_columns_are_labelled(render) -> None:
    """Unlabelled, the reader takes the saved figure - and it triple-counts."""
    html = render("renderTimeline", TIMELINE)

    assert "0.2862" in html and "0.4272" in html, "the saved figures are not shown"
    assert "0.3823" in html, "the common-price figure is not shown"
    assert "74.90" in html, "the common price itself is not named, so its column is unreadable"


def test_the_price_and_view_split_is_stated_not_left_to_arithmetic(render) -> None:
    """The finding, in words, because it is the reason the view exists."""
    html = render("renderTimeline", TIMELINE)

    assert "+14.1" in html and "+4.5" in html, (
        "the two swings are not both reported, so the split is left to the reader"
    )


def test_a_single_run_timeline_says_it_cannot_show_drift(render) -> None:
    """Degenerate and honest, rather than a one-row chart implying a trend."""
    one = {**TIMELINE, "runs": TIMELINE["runs"][:1], "common_price": 79.17}
    html = render("renderTimeline", one)

    assert "2026-07-19" in html, "the single run is not rendered at all"
    assert "only one" in html.lower() or "single" in html.lower(), (
        "a one-run timeline renders like a comparison, implying a drift it cannot show"
    )
    assert "+14.1" not in html


def test_a_broken_run_is_shown_in_place_rather_than_dropped(render) -> None:
    """A gap in a timeline is information; a silently shorter timeline is not."""
    broken = {
        **TIMELINE,
        "runs": [
            {
                "name": "2026-07-19-A12",
                "as_of": "2026-07-19",
                "status": "error",
                "error": "SpecError: nonpositive_price",
            },
            TIMELINE["runs"][1],
        ],
    }
    html = render("renderTimeline", broken)

    assert "2026-07-19" in html, "the broken run vanished from its own timeline"
    assert "nonpositive_price" in html, "the reason it is missing is not given"
    assert "NaN" not in html and "undefined" not in html


# --------------------------------------------------------------------------- #
#
# The marginals panel.
#
# Two claims live in the rendering rather than the payload, and both are the
# kind that look fine while being wrong:
#
#   * each driver gets its own axis. This is the deliberate opposite of the
#     timeline, and the same reasoning as the worlds view: `cap_rate` runs 5-7
#     and `nav_other` runs 1875-4730, so one shared axis would collapse every
#     narrow band to a hairline against the widest driver in the spec.
#   * the drift figure is shown for every driver, not only for those over the
#     emphasis threshold. `quantile_mismatch` has a threshold and a real 4.89%
#     tail error once sat under it and passed `validate` in silence; a panel
#     that printed a number only above its own line would make "fine" and
#     "unmeasured" look identical.

#: Two real drivers, three orders of magnitude apart, plus the bounds case.
MARGINALS = {
    "name": "2026-07-27-A50",
    "factors": ["rate"],
    "unloaded_factors": [],
    "drivers": [{"name": "cap_rate", "loadings": [0.7], "sum_sq": 0.49, "psi": 0.714143}],
    "marginals": [
        {
            "name": "cap_rate",
            "family": "lognormal",
            "p10": 5.5,
            "p50": 6.5,
            "p90": 7.5,
            "r10": 5.472,
            "r50": 6.5,
            "r90": 7.72,
            "drift": 0.11,
            "flagged": True,
            "tails": "extrapolated",
            "notice": None,
        },
        {
            "name": "nav_other",
            "family": "pert",
            "p10": -1709.0,
            "p50": -1530.0,
            "p90": 119.6,
            "r10": -1617.0,
            "r50": -1341.0,
            "r90": -866.5,
            "drift": 0.539,
            "flagged": True,
            "tails": "censored",
            "notice": None,
        },
        {
            "name": "affo_growth",
            "family": "lognormal3",
            "p10": 1.0,
            "p50": 2.0,
            "p90": 4.0,
            "r10": 1.0,
            "r50": 2.0,
            "r90": 4.0,
            "drift": 0.0,
            "flagged": False,
            "tails": "extrapolated",
            "notice": None,
        },
    ],
}


def _stated_bands(html: str) -> list[tuple[float, float]]:
    return [
        (float(a), float(b))
        for a, b in re.findall(r'class="mg-stated" style="left:([\d.]+)%;width:([\d.]+)%"', html)
    ]


def test_each_driver_gets_its_own_axis(render) -> None:
    """Bands three orders of magnitude apart must render at comparable widths."""
    html = render("renderMarginals", MARGINALS)
    bands = _stated_bands(html)

    assert len(bands) == 3, f"expected one stated band per driver, got {len(bands)}"
    widths = [w for _, w in bands]
    assert min(widths) > 30, (
        f"a band rendered at {min(widths)}% of its row - that is a shared axis "
        "collapsing the narrow drivers against the widest one"
    )


def test_the_realized_band_is_drawn_beside_the_stated_one(render) -> None:
    """Both, always. The stated triple alone is what the YAML already shows."""
    html = render("renderMarginals", MARGINALS)

    assert "mg-stated" in html and "mg-realized" in html, (
        "only one band is drawn, so the divergence the panel exists for is invisible"
    )
    assert "-866.5" in html or "-866.50" in html, "the realized p90 is not shown as a number"


def test_every_driver_shows_its_drift_including_the_unflagged(render) -> None:
    """Below the threshold, "fine" and "unmeasured" must not look the same."""
    html = render("renderMarginals", MARGINALS)

    assert "0.539" in html, "the flagged driver's drift is missing"
    assert "0.000" in html, (
        "a driver with zero drift shows nothing where its number should be, so "
        "the reader cannot tell it was measured at all"
    )


def test_the_bounds_reading_is_named_where_it_applies(render) -> None:
    """A `pert` p90 is a maximum, and a reader who does not know that misreads it."""
    html = render("renderMarginals", MARGINALS)

    assert "censored" in html.lower()
    assert "extrapolated" in html.lower()
    assert "bound" in html.lower(), (
        "the panel marks the tail regime but never says what censored means for "
        "the numbers the analyst typed"
    )


def test_a_fit_notice_is_shown_beside_its_own_band(render) -> None:
    """`validate`'s own words, next to the numbers they are about."""
    noticed = {
        **MARGINALS,
        "marginals": [
            {**MARGINALS["marginals"][0], "notice": "p50 is far from the geometric mean"}
        ],
    }
    html = render("renderMarginals", noticed)

    assert "geometric mean" in html, "the fit notice is dropped"


def test_the_panel_survives_a_spec_with_no_drivers(render) -> None:
    """Every input frozen as a constant is a legal spec, not an empty page."""
    html = render("renderMarginals", {**MARGINALS, "marginals": [], "drivers": []})

    assert "NaN" not in html and "undefined" not in html
    assert "no sampled drivers" in html.lower()


# --------------------------------------------------------------------------- #
#
# The client's own ticker parser, pinned against the server's.
#
# `siblingRuns()` decides whether the Timeline button is offered at all, and it
# parses `<date>-<TICKER>` in JavaScript while `_ticker_of` parses it in Python.
# Two implementations of one rule is the drift shape this repo keeps finding, and
# the failure is silent in the worst way: if the client's pattern narrows, the
# button quietly stops appearing and every render case above stays green, because
# they all test what renders once you get there.
#
# Found by audit on 2026-08-08. The code carried a comment saying the client is
# "deliberately NOT a second source of truth", which was the intent and not the
# fact - the honest response to that is the one this project prescribes:
# write the check, not the comment.


def _client_tickers(render, names: list[str]) -> list[str | None]:
    return json.loads(render("__tickerOf", names))


def test_the_client_and_server_agree_on_what_a_ticker_is(render) -> None:
    """Over every real analysis name and the hostile ones this must refuse."""
    from gui.server import _ticker_of, list_analyses

    names = [e["name"] for e in list_analyses()] + [
        "2026-07-25-../etc/passwd",
        "2026-07-25-A?x=1",
        "2026-07-25-A#frag",
        "notes",
        "26-07-25-A12",
        "2026-07-25-",
        "2026-07-25-" + "A" * 13,
        "2026-07-25-BRK.B",
        "2026-07-25-RDS-A",
    ]
    assert len(names) > 9, "no entries on disk - only the synthetic half would be checked"

    js = _client_tickers(render, names)
    py = [_ticker_of(n) for n in names]
    disagree = [(n, a, b) for n, a, b in zip(names, js, py, strict=True) if a != b]

    assert not disagree, (
        "the viewer and the server disagree about what a ticker is, so the "
        f"Timeline button is offered for the wrong set: {disagree}"
    )


def test_the_timeline_button_is_offered_exactly_where_it_can_answer(render) -> None:
    """`siblingRuns` must find every run of a company and nothing else."""
    names = [
        "2026-07-19-A12",
        "2026-07-25-A12",
        "2026-08-07-A32",
        "2026-07-24-A22",
        "notes",
    ]
    out = json.loads(render("__siblingRuns", names, "2026-07-19-A12"))
    assert sorted(out) == ["2026-07-19-A12", "2026-07-25-A12"], (
        f"siblingRuns picked {out} - the button would be offered on the wrong set"
    )

    alone = json.loads(render("__siblingRuns", names, "2026-08-07-A32"))
    assert alone == ["2026-08-07-A32"], (
        "a single-run ticker reports siblings, so the button appears where there "
        "is nothing to compare"
    )


# --------------------------------------------------------------------------- #
#
# The four render functions the 2026-08-08 audit found undriven.
#
# All four are pure and none touches the DOM, so they were reachable by this
# harness the whole time and simply were not added - the audit's partition test
# is what made that visible rather than inferable.
#
# Which branches the shipped set can reach, counted before these were written
# rather than after. That is the lesson from the vacuous notice sweep found the
# same day: a test over real data is decoration if no row exercises its branch.
#
#   * 13 of 46 specs produce run WARNINGS and diagnostics -> real data.
#   * 0 produce validate notices, 0 produce validate errors, and 0 have a
#     degenerate `p_undervalued` -> those branches are planted, and say so.


def _ui(html: str) -> str:
    """The rendered half of a run view, with the raw text dump cut off."""
    marker = "<h3>Full output</h3>"
    assert marker in html, "the full-output block moved - this helper is cutting at nothing"
    return html[: html.index(marker)]


@pytest.fixture(scope="module")
def warned_run():
    """A real `/api/run` response from a spec that actually floors draws."""
    from gui.server import do_run, list_analyses

    for entry in list_analyses():
        if not entry["has_yaml"]:
            continue
        out = do_run(entry["name"])
        if out["payload"].get("warnings"):
            return out
    raise AssertionError(
        "no shipped spec produces a warning any more - the WARNINGS cases below "
        "would assert about a clean run and prove nothing"
    )


@pytest.fixture(scope="module")
def clean_run():
    """A real response from a spec with nothing to declare - the other side."""
    from gui.server import do_run, list_analyses

    for entry in list_analyses():
        if not entry["has_yaml"]:
            continue
        out = do_run(entry["name"])
        if not out["payload"].get("warnings"):
            return out
    raise AssertionError("every shipped spec warns - the clean case cannot be exercised")


def test_the_warnings_block_comes_before_the_headline_number(render, warned_run) -> None:
    """A run with something to declare is not a result."""
    html = _ui(render("renderRun", warned_run))

    assert "do not quote these numbers" in html, "a floored run is not marked as one"
    assert html.index("warning(s)") < html.index('class="hero"'), (
        "the WARNINGS block renders below the headline p_undervalued, so the "
        "number is read before the reason not to trust it"
    )


def test_the_floored_draw_counts_reach_the_page(render, warned_run) -> None:
    """Floored draws are counted and surfaced, not merely counted."""
    ui = _ui(render("renderRun", warned_run))
    for flag, count in warned_run["payload"]["diagnostics"].items():
        assert flag in ui, f"the diagnostic {flag!r} reaches only the raw text dump"
        assert f"{count:,}" in ui or str(count) in ui, f"{flag}'s count is missing from the UI"


def test_a_clean_run_says_so_rather_than_saying_nothing(render, clean_run) -> None:
    """Silence and "nothing to declare" must not look the same."""
    ui = _ui(render("renderRun", clean_run))

    assert "Clean run" in ui
    assert "do not quote these numbers" not in ui


def test_the_monte_carlo_standard_error_rides_with_the_hero(render, clean_run) -> None:
    """Reported always, and at 4 dp because "0.00" reads as no uncertainty."""
    ui = _ui(render("renderRun", clean_run))
    p = clean_run["payload"]

    shown = re.search(r'class="val">([\d.]+)<', ui).group(1)
    assert shown == f"{p['p_undervalued']:.4f}", (
        f"the hero shows {shown} for a p_undervalued of {p['p_undervalued']}"
    )
    assert f"{p['p_undervalued_stderr']:.4f}" in ui, (
        "the MC standard error is not shown at 4 dp beside the hero - a stderr "
        "rounded to 0.00 reads as no uncertainty, which is an artefact worth flagging"
    )


@pytest.mark.parametrize("degenerate", [0.0, 1.0])
def test_a_degenerate_p_undervalued_is_called_an_artefact(render, clean_run, degenerate) -> None:
    """Planted: no shipped spec reaches 0 or 1, and the branch still has to work."""
    payload = {**clean_run["payload"], "p_undervalued": degenerate, "p_undervalued_stderr": 0.0}
    ui = _ui(render("renderRun", {**clean_run, "payload": payload}))

    assert "too-narrow inputs" in ui, f"p_undervalued = {degenerate} rendered as an ordinary result"


def test_an_unrecognised_report_schema_is_refused_at_the_top(render, clean_run) -> None:
    """Planted: the engine and viewer agree today, so nothing on disk reaches this."""
    html = render(
        "renderRun", {**clean_run, "schema": {"expected": "1.0", "got": "9.9", "supported": False}}
    )

    assert "Unrecognised report schema" in html
    assert html.index("Unrecognised report schema") < html.index('class="hero"')


def test_the_what_if_banner_needs_the_analysis_it_reads(render, clean_run) -> None:
    """`renderRun` reaches `whatIfBanner`, which reads module state."""
    analysis = {
        "spec_price": 50.0,
        "current_price": 61.0,
        "current_as_of": "2026-08-08",
        "current_live": False,
        "age_days": 0,
    }
    html = render(
        "renderRun",
        {**clean_run, "overridden": True, "price_used": 61.0, "spec_price": 50.0},
        analysis=analysis,
    )

    assert "only the price is current" in html, "a run at the market quote is not marked"
    assert "50.00" in html, "the spec's own price is not shown beside the one used"


def test_validate_renders_errors_with_their_codes(render) -> None:
    """Planted: no shipped spec fails validation, by design."""
    html = render(
        "renderValidate",
        {
            "ok": False,
            "errors": [
                {"code": "weights_not_100", "message": "weights sum to 90"},
                {"code": "nonpositive_price", "message": "price is -1"},
            ],
            "notices": [],
        },
    )

    assert "2 error(s)" in html
    assert "weights_not_100" in html and "nonpositive_price" in html, (
        "the digest's codes are dropped, leaving a count with no way to act on it"
    )
    assert 'class="banner err"' in html


def test_a_notice_is_not_dressed_as_a_blocking_error(render) -> None:
    """Planted, and the distinction the aggregate-digest rule turns on."""
    html = render(
        "renderValidate",
        {
            "ok": True,
            "errors": [],
            "notices": [
                {"code": "quantile_mismatch", "message": "p50 is far from the geometric mean"}
            ],
        },
    )

    assert "1 notice(s)" in html
    assert 'class="banner warn"' in html, "a non-blocking notice is styled as an error"
    assert "error(s)" not in html


def test_a_clean_spec_reports_both_kinds_of_silence(render) -> None:
    """ "Valid" and "no notices" are two facts, and every shipped spec has both."""
    from gui.server import ANALYSES_DIR, do_validate

    name = sorted(p.name for p in ANALYSES_DIR.iterdir() if p.is_dir())[0]
    html = render("renderValidate", do_validate(name))

    assert "Valid" in html and "No notices" in html


def test_convergence_shows_both_sample_sizes_and_the_drift(render) -> None:
    """The reader judges stability; the view must not assert it."""
    from gui.server import ANALYSES_DIR, do_convergence

    name = sorted(p.name for p in ANALYSES_DIR.iterdir() if p.is_dir())[0]
    d = do_convergence(name)
    html = render("renderConvergence", d)

    assert f"{d['n']:,}" in html and f"{d['n'] * 2:,}" in html, "both sample sizes are not shown"
    assert f"{d['convergence']['value_p50_N']:.2f}" in html
    assert f"{d['convergence']['value_p50_2N']:.2f}" in html
    assert "Drift" in html


def test_worlds_draws_every_driver_on_its_own_axis(render) -> None:
    """The dual-axis mistake, avoided deliberately - and the opposite of the timeline."""
    from gui.server import ANALYSES_DIR, do_worlds

    name = sorted(p.name for p in ANALYSES_DIR.iterdir() if p.is_dir())[0]
    w = do_worlds(name)
    html = render("renderWorlds", w["worlds"], w["marginals"])

    drivers = [k for k in w["worlds"][0] if k not in ("quantile", "value")]
    assert drivers, "the fixture analysis has no drivers - this proves nothing"
    for name_ in drivers:
        assert name_ in html, f"{name_} is missing from the worlds view"
    # Whitespace-normalised: the copy is a template literal that wraps mid-phrase,
    # so a raw substring check tests the source's line breaks rather than its words.
    flat = re.sub(r"\s+", " ", html)
    assert "own axis" in flat, "the view does not say the axes are not shared"
    assert "single representative draw" in flat, (
        "a world is one draw, not an average, and the view has to say so"
    )


def test_worlds_degrades_rather_than_breaking_on_an_empty_payload(render) -> None:
    html = render("renderWorlds", [], {})
    assert "No worlds" in html
    assert "NaN" not in html and "undefined" not in html
