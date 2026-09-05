"""Optional plotting support (matplotlib, lazy-imported)."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from valdist.core.factors import FactorModel
from valdist.core.marginals import Marginal
from valdist.core.model import Model


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure each test opened."""
    import matplotlib.pyplot as plt

    plt.close("all")
    yield
    leaked = plt.get_fignums()
    plt.close("all")
    assert not leaked, f"test leaked {len(leaked)} matplotlib figure(s): {leaked}"


def _small_result():
    fm = FactorModel(["rate", "growth"], {"a": {"rate": 0.6}, "b": {"rate": -0.3, "growth": 0.4}})
    drivers = [
        Marginal("a", p10=1.0, p50=2.0, p90=3.0, family="normal"),
        Marginal("b", p10=10.0, p50=20.0, p90=30.0, family="lognormal"),
    ]
    model = Model(drivers, fm, valuation=lambda v: v["a"] + v["b"])
    return model.run(n=2000, price=18.0, seed=0)


@contextmanager
def closing(fig):
    """Yield *fig*, then close it. The autouse fixture asserts nothing leaked."""
    import matplotlib.pyplot as plt

    try:
        yield fig
    finally:
        plt.close(fig)


def test_plot_tornado_returns_figure():
    from valdist.plot import plot_tornado

    result = _small_result()
    with closing(plot_tornado(result)) as fig:
        ax = fig.axes[0]
        assert len(ax.patches) == 2  # one bar per driver


def test_plot_tornado_bar_signs_match_correlations():
    from valdist.plot import plot_tornado

    result = _small_result()
    tornado = dict(result.tornado())
    with closing(plot_tornado(result)) as fig:
        ax = fig.axes[0]
        widths = sorted(p.get_width() for p in ax.patches)
        expected = sorted(tornado.values())
        for w, e in zip(widths, expected):
            assert abs(w - e) < 1e-9


def test_plot_value_distribution_returns_figure():
    from valdist.plot import plot_value_distribution

    result = _small_result()
    with closing(plot_value_distribution(result)) as fig:
        ax = fig.axes[0]
        assert len(ax.patches) > 0  # histogram bars
        assert len(ax.lines) >= 4  # price + P10/P50/P90 reference lines


def test_plot_functions_accept_dark_theme():
    from valdist.plot import plot_tornado, plot_value_distribution

    result = _small_result()
    with closing(plot_tornado(result, theme="dark")) as fig1:
        assert fig1 is not None
    with closing(plot_value_distribution(result, theme="dark")) as fig2:
        assert fig2 is not None


def test_plot_value_distribution_handles_zero_variance():
    """A degenerate all-identical distribution gives lo == hi, so the x-limits
    collapse to a zero-width range - matplotlib then warns and draws an empty
    chart. The axis must be widened to something displayable instead."""
    import warnings

    from valdist.plot import plot_value_distribution

    model = Model(
        [Marginal("a", p10=5.0, p50=5.0, p90=5.0, family="triangular")],
        FactorModel([], {}),
        valuation=lambda v: v["a"],  # every draw evaluates to exactly 5.0
    )
    result = model.run(n=500, price=5.0, seed=0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a matplotlib warning fails the test
        fig = plot_value_distribution(result)

    with closing(fig):
        lo, hi = fig.axes[0].get_xlim()
        assert hi > lo, f"zero-width x-axis: ({lo}, {hi})"
        assert len(fig.axes[0].patches) > 0, "no histogram bars drawn"
