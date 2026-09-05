"""Optional plotting: matplotlib, lazy-imported so it is never a hard dependency.

Install with `pip install valdist[plot]`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from valdist.core.results import Result

_THEME = {
    "light": {
        "surface": "#fcfcfb",
        "primary_ink": "#0b0b0b",
        "secondary_ink": "#52514e",
        "muted_ink": "#898781",
        "axis": "#c3c2b7",
        "positive": "#2a78d6",  # diverging blue
        "negative": "#e34948",  # diverging red
        "sequential": "#2a78d6",  # sequential blue, same hue family
    },
    "dark": {
        "surface": "#1a1a19",
        "primary_ink": "#ffffff",
        "secondary_ink": "#c3c2b7",
        "muted_ink": "#898781",
        "axis": "#383835",
        "positive": "#3987e5",
        "negative": "#e66767",
        "sequential": "#3987e5",
    },
}


def _style(theme: Literal["light", "dark"]) -> dict:
    if theme not in _THEME:
        raise ValueError(f"theme must be 'light' or 'dark', got {theme!r}")
    return _THEME[theme]


def plot_tornado(result: "Result", theme: Literal["light", "dark"] = "light") -> "Figure":
    """Horizontal diverging bar chart of Result.tornado's rank correlations."""
    import matplotlib.pyplot as plt

    c = _style(theme)
    tornado = result.tornado()  # already sorted by |corr| descending
    names = [t[0] for t in tornado]
    values = [t[1] for t in tornado]

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 1.5), facecolor=c["surface"])
    ax.set_facecolor(c["surface"])

    y = range(len(names))
    colors = [c["positive"] if v >= 0 else c["negative"] for v in values]
    bars = ax.barh(list(y), values, height=0.6, color=colors)

    for bar, v in zip(bars, values):
        offset = 0.02 if v >= 0 else -0.02
        ha = "left" if v >= 0 else "right"
        ax.text(
            v + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{v:+.2f}",
            va="center",
            ha=ha,
            color=c["primary_ink"],
            fontsize=9,
        )

    ax.set_yticks(list(y))
    ax.set_yticklabels(names, color=c["secondary_ink"])
    ax.axvline(0, color=c["axis"], linewidth=1)
    ax.set_xlim(-1.15, 1.15)
    ax.set_xlabel("Spearman rank correlation with value", color=c["secondary_ink"])
    ax.set_title("Tornado – driver sensitivity", color=c["primary_ink"], loc="left")
    ax.invert_yaxis()  # largest |corr| at top, matching result.tornado()'s order
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=c["muted_ink"])
    fig.tight_layout()
    return fig


def plot_value_distribution(
    result: "Result", theme: Literal["light", "dark"] = "light"
) -> "Figure":
    """Histogram of the value distribution with price and P10/P50/P90 overlaid."""
    import matplotlib.pyplot as plt

    c = _style(theme)
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=c["surface"])
    ax.set_facecolor(c["surface"])

    quantiles = result.value_quantiles()
    p10, p50, p90 = quantiles[0.10], quantiles[0.50], quantiles[0.90]
    lo, hi = (float(v) for v in result.value_quantiles((0.01, 0.99)).values())
    pad = (hi - lo) * 0.05
    if pad <= 0:
        # Degenerate distribution: every draw landed on the same value, so
        # lo == hi and the x-limits would collapse to zero width. matplotlib
        # warns ("identical low and high xlims") and draws an empty chart.
        # Open a symmetric window around the value so the single spike is
        # actually visible. Scale-relative, with an absolute fallback for a
        # distribution pinned at exactly 0.
        pad = abs(lo) * 0.05 or 0.5
    xlim = (lo - pad, hi + pad)

    ax.hist(
        result.value,
        bins=50,
        range=xlim,
        color=c["sequential"],
        alpha=0.85,
        edgecolor=c["surface"],
    )
    ax.set_xlim(*xlim)

    line_top = 0.90  # axes-fraction: reference lines stop here, labels sit above
    trans = ax.get_xaxis_transform()  # x in data coords, y in axes-fraction (0-1)
    ax.axvline(p10, ymax=line_top, color=c["muted_ink"], linewidth=1)
    ax.axvline(p90, ymax=line_top, color=c["muted_ink"], linewidth=1)
    ax.axvline(p50, ymax=line_top, color=c["secondary_ink"], linewidth=1.5)

    percentile_row, price_row = 0.97, 0.80  # separate rows: never visually merge
    label_kwargs = dict(transform=trans, ha="center", va="top", fontsize=8)
    ax.text(p10, percentile_row, "P10", color=c["muted_ink"], **label_kwargs)
    ax.text(p50, percentile_row, "P50", color=c["secondary_ink"], **label_kwargs)
    ax.text(p90, percentile_row, "P90", color=c["muted_ink"], **label_kwargs)

    if result.price is not None and xlim[0] <= result.price <= xlim[1]:
        ax.axvline(result.price, ymax=line_top, color=c["primary_ink"], linewidth=2)
        ax.text(
            result.price,
            price_row,
            f"Price {result.price:g}",
            transform=trans,
            ha="center",
            va="top",
            color=c["primary_ink"],
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Value", color=c["secondary_ink"])
    ax.set_ylabel("Draws", color=c["secondary_ink"])
    ax.set_title("Value distribution", color=c["primary_ink"], loc="left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(c["axis"])
    ax.tick_params(colors=c["muted_ink"])
    fig.tight_layout()
    return fig
