"""JSON results + human-readable summary for a valdist Result."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from valdist.core.results import Result

#: Version of the `to_json()` payload shape, not the spec's `schema_version`
#: (that versions the input YAML; this versions the output contract).
#:
#: Once nothing outside this repo consumed `report.to_json()`, the payload
#: shape was free to change; `gui/` is now a consumer, so the shape is a
#: contract. Bump this whenever a key is removed or its meaning changes;
#: adding a key is backwards-compatible and does not require a bump.
REPORT_SCHEMA_VERSION = "1.0"


def to_json(result: Result, indent: int = 2) -> str:
    """Return result summary as a JSON string."""
    data: dict = {"report_schema_version": REPORT_SCHEMA_VERSION}
    data.update(result.summary())
    data["tornado"] = [{"driver": name, "rank_corr": round(r, 4)} for name, r in result.tornado()]
    data["diagnostics"] = dict(result.diagnostics)
    data["warnings"] = result.warnings()
    return json.dumps(data, indent=indent, allow_nan=False)


def to_text(result: Result) -> str:
    """Return a human-readable multi-line summary string."""
    s = result.summary()
    lines = [
        f"{'=' * 52}",
        f"  valdist result  –  {result.model.names}",
        f"{'=' * 52}",
        f"  n = {s['n']:,}  |  seed = {result.seed}",
        "",
        "  Value distribution:",
        f"    P10  = {s['value_p10']:>8.2f}",
        f"    P50  = {s['value_p50']:>8.2f}",
        f"    P90  = {s['value_p90']:>8.2f}",
    ]
    # The conservative anchor gets its own block, with its meaning, rather than
    # a fourth row in the distribution above. P25 answers a different question
    # from P10: not "how bad can it get" but "what is a deliberately
    # conservative estimate of value". Printed even without a price, because
    # value does not depend on price - which is exactly what makes it a
    # buy-below number rather than a statistic.
    ratio = s["p25_p50_ratio"]
    lines += [
        "",
        "  Conservative value anchor:",
        f"    P25  = {s['value_p25']:>8.2f}"
        + (f"   (P25/P50 = {ratio:.2f})" if ratio is not None else ""),
        "    Value does not depend on price, so P25 is itself a BUY BELOW",
        "    price: at or under it you are paying less than a deliberately",
        "    conservative estimate of value. A lower P25/P50 means a fatter",
        "    downside for the same median.",
    ]
    if result.price is not None:
        lines += [
            f"    Margin at the anchor = {s['mos_p25'] * 100:+.1f}%",
            "",
            f"  Price = {s['price']:>8.2f}",
            f"  P(undervalued)  = {s['p_undervalued']:.4f}"
            f"  ±{s['p_undervalued_stderr']:.4f} (MC stderr)",
            "  Margin of safety (P10/P50/P90):",
            f"    {s['mos_p10'] * 100:+.1f}% / {s['mos_p50'] * 100:+.1f}%"
            f" / {s['mos_p90'] * 100:+.1f}%",
        ]
    lines += [
        "",
        "  Tornado (Spearman rank corr):",
    ]
    for name, r in result.tornado():
        bar = "█" * int(abs(r) * 20)
        lines.append(f"    {name:<20s} {r:+.3f}  {bar}")

    # Never let a floored/degenerate run read as a clean result.
    result_warnings = result.warnings()
    if result_warnings:
        lines += ["", "  WARNINGS:"]
        for w in result_warnings:
            lines += _wrap_warning(w)

    lines.append(f"{'=' * 52}")
    return "\n".join(lines)


def _wrap_warning(text: str, width: int = 46) -> list[str]:
    """Wrap one warning to the report's box width, indented under WARNINGS."""
    import textwrap

    wrapped = textwrap.wrap(text, width=width)
    if not wrapped:
        return []
    return [f"    ! {wrapped[0]}"] + [f"      {line}" for line in wrapped[1:]]


def print_worlds(result: Result, qs: tuple[float, ...] = (0.1, 0.5, 0.9)) -> str:
    """Return a formatted worlds table as a string."""
    scenarios = result.worlds(qs=qs)
    names = result.model.names

    header = f"{'Quantile':>9}  {'Value':>8}  " + "  ".join(f"{n:>12}" for n in names)
    rows = [header, "-" * len(header)]
    for sc in scenarios:
        label = f"P{int(sc['quantile'] * 100)}"
        row = f"{label:>9}  {sc['value']:>8.2f}  " + "  ".join(f"{sc[n]:>12.4f}" for n in names)
        rows.append(row)
    return "\n".join(rows)
