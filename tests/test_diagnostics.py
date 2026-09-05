"""Draw diagnostics and honesty-about-uncertainty warnings."""

from __future__ import annotations

import numpy as np
import pytest

from valdist.adapters.registry import get_valuation
from valdist.core import Marginal, Model
from valdist.core.factors import FactorModel

# --------------------------------------------------------------------------- #
# The exact repro from the review: a plausible equity_v2 spec whose wacc and
# terminal_growth bands cross. These numbers are asserted, not approximated:
# the standing rule on this project is to verify a fix against the real numbers
# that motivated it rather than trusting the mechanism on paper.
# --------------------------------------------------------------------------- #

_EQUITY_CONSTANTS = dict(
    shares=450.0,
    debt=4000.0,
    cash=6000.0,
    fcff=7000.0,
    years=10,
    normalized_earnings=6000.0,
    relative_value_per_share=500.0,
    w_dcf=50.0,
    w_epv=30.0,
    w_relative=20.0,
)


def _crossing_equity_model() -> Model:
    """wacc 6/9/12 vs terminal_growth 1.5/2.5/3.5 - the bands cross."""
    fm = FactorModel(
        ["macro"],
        {
            "fcf_growth": {"macro": 0.5},
            "terminal_growth": {"macro": 0.3},
            "wacc": {"macro": -0.4},
        },
    )
    drivers = [
        Marginal("fcf_growth", p10=4.0, p50=8.0, p90=12.0, family="normal"),
        Marginal("terminal_growth", p10=1.5, p50=2.5, p90=3.5, family="normal"),
        Marginal("wacc", p10=6.0, p50=9.0, p90=12.0, family="normal"),
    ]
    return Model(drivers, fm, get_valuation("equity_v2"), constants=_EQUITY_CONSTANTS)


# --------------------------------------------------------------------------- #
# The diagnostics channel itself (core stays valuation-agnostic)
# --------------------------------------------------------------------------- #


def test_flag_outside_a_run_is_a_no_op() -> None:
    """Calling flag() with no collector bound must not raise - an adapter is
    an ordinary callable and may legitimately be invoked outside Model.run().
    """
    from valdist.core.diagnostics import flag

    flag("some_flag")  # must not raise


def test_collector_counts_named_flags() -> None:
    from valdist.core.diagnostics import collect_diagnostics, flag

    with collect_diagnostics() as counts:
        flag("a")
        flag("a")
        flag("b")
    assert dict(counts) == {"a": 2, "b": 1}


def test_collector_is_reentrant_safe() -> None:
    """Nested/sequential runs must not leak counts into one another."""
    from valdist.core.diagnostics import collect_diagnostics, flag

    with collect_diagnostics() as first:
        flag("x")
    with collect_diagnostics() as second:
        flag("y")
    assert dict(first) == {"x": 1}
    assert dict(second) == {"y": 1}


def test_collector_is_thread_safe() -> None:
    """Concurrent runs on different threads must not bleed counts into one
    another - ContextVar is thread-local by design, but nothing previously
    exercised that concurrently rather than just via sequential nesting."""
    import threading

    from valdist.core.diagnostics import collect_diagnostics, flag

    n_threads = 8
    flags_per_thread = 200
    results: list[dict[str, int]] = [None] * n_threads  # type: ignore[list-item]
    barrier = threading.Barrier(n_threads)

    def worker(i: int) -> None:
        barrier.wait()  # maximize actual overlap between threads
        with collect_diagnostics() as counts:
            for _ in range(flags_per_thread):
                flag(f"flag_{i}")
        results[i] = dict(counts)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i, counts in enumerate(results):
        assert counts == {f"flag_{i}": flags_per_thread}, (
            f"thread {i} saw contaminated counts: {counts}"
        )


@pytest.fixture(scope="session")
def crossing_result():
    """The 50,000-draw crossing-bands run, executed ONCE for the session."""
    return _crossing_equity_model().run(n=50_000, price=500.0, seed=0)


# --------------------------------------------------------------------------- #
# Item 1: floored draws are counted and surfaced
# --------------------------------------------------------------------------- #


def test_crossing_bands_flag_the_floored_draws(crossing_result) -> None:
    """The exact review repro: 286/50,000 draws hit the terminal-spread floor.
    They must now be COUNTED, not silently blended.
    """
    result = crossing_result

    diagnostics = result.diagnostics
    assert "dcf_terminal_spread_floored" in diagnostics, (
        f"the spread floor bound on this spec but nothing was flagged: {diagnostics}"
    )
    # Pinned to the empirically-verified count from the review, +/-2 to tolerate
    # floating-point boundary drift in scipy.stats.norm.ppf across library
    # versions for the handful of draws that land right at the floor threshold
    # (np.random.default_rng's own bit stream is version-stable; this is not
    # a determinism concern) - a real regression (e.g. a changed
    # draw order, or the floor not firing at all) would miss by far more than 2.
    count = diagnostics["dcf_terminal_spread_floored"]
    assert count == pytest.approx(286, abs=2), f"floored count drifted to {count}, expected ~286"
    assert result.diagnostic_rate("dcf_terminal_spread_floored") == pytest.approx(
        286 / 50_000, abs=2 / 50_000
    )


def test_floored_draws_produce_a_warning(crossing_result) -> None:
    """Honesty: the report must SAY the tail is not meaningful."""
    result = crossing_result

    warnings_text = " ".join(result.warnings())
    assert "dcf_terminal_spread_floored" in warnings_text
    # Derived from the same live result rather than a hardcoded literal, so
    # this can't drift out of sync with the pinned-count tolerance above.
    expected_pct = f"{result.diagnostic_rate('dcf_terminal_spread_floored'):.2%}"
    assert expected_pct in warnings_text, (
        f"expected the actual computed rate {expected_pct!r} in: {warnings_text!r}"
    )


def test_report_surfaces_the_warning(crossing_result) -> None:
    """The whole point: a user running this spec must not see a clean report."""
    from valdist import report

    result = crossing_result
    text = report.to_text(result)
    assert "WARNING" in text.upper()
    assert "floor" in text.lower()

    import json

    payload = json.loads(report.to_json(result))
    assert payload["warnings"], "to_json must carry the warnings array"


def test_ddm_and_affo_floors_flag_too() -> None:
    """reit_v2's two helpers raise their own distinct flags."""
    from valdist.adapters.reit import _affo_dcf_equity, _ddm_two_stage
    from valdist.core.diagnostics import collect_diagnostics

    with collect_diagnostics() as counts:
        # rate < terminal in both cases
        _ddm_two_stage(dps=2.0, growth=0.02, years=5, terminal=0.021, rate=0.02)
        _affo_dcf_equity(affo=1000.0, shares=100.0, years=5, growth=0.02, coe=0.02, terminal=0.021)
    assert dict(counts) == {
        "ddm_terminal_spread_floored": 1,
        "affo_terminal_spread_floored": 1,
    }


# --------------------------------------------------------------------------- #
# Item 2: the unguarded divisions
# --------------------------------------------------------------------------- #


def test_epv_wacc_floor_flags_and_stays_finite() -> None:
    """_epv_equity divided by the raw sampled wacc with NO guard; a wacc draw
    <= 0 is reachable (2/50,000 on an ordinary normal 6/9/12 band) and flipped
    EPV's sign at a -18,430x multiplier.
    """
    from valdist.adapters.equity import _epv_equity
    from valdist.core.diagnostics import collect_diagnostics

    with collect_diagnostics() as counts:
        v_zero = _epv_equity(normalized_earnings=6000.0, wacc=0.0, net_debt=0.0, shares=450.0)
        v_neg = _epv_equity(normalized_earnings=6000.0, wacc=-0.004, net_debt=0.0, shares=450.0)

    assert np.isfinite(v_zero) and v_zero > 0, f"wacc=0 gave {v_zero}"
    assert np.isfinite(v_neg) and v_neg > 0, f"wacc<0 gave {v_neg} (sign flip)"
    assert counts["epv_wacc_floored"] == 2


def test_nav_cap_rate_floor_flags() -> None:
    """reit_v2's gav = noi / cap_r is the same structural exposure."""
    from valdist.core.diagnostics import collect_diagnostics

    reit_v2 = get_valuation("reit_v2")
    sample = dict(
        shares=1090.0,
        dps=1.7,
        ddm_stage1_years=10,
        affo=2680.0,
        affo_years=10,
        noi=2400.0,
        nav_debt=17000.0,
        nav_other=0.0,
        w_ddm=40.0,
        w_affo=40.0,
        w_nav=20.0,
        cost_of_equity=9.0,
        div_growth=3.0,
        div_terminal=2.0,
        affo_growth=3.0,
        affo_terminal=2.0,
        cap_rate=0.0,  # degenerate draw
    )
    with collect_diagnostics() as counts:
        value = reit_v2(sample)
    assert np.isfinite(value), f"got {value}"
    assert counts["nav_cap_rate_floored"] == 1


# --------------------------------------------------------------------------- #
# p_undervalued in {0, 1} is a warning, not a result
# --------------------------------------------------------------------------- #


def test_p_undervalued_certain_warns() -> None:
    """Honesty about uncertainty, previously not implemented anywhere: the report used to
    print `1.0000 ±0.0000`, advertising infinite confidence in exactly the
    degenerate case the principle exists to flag.
    """
    from valdist import report

    fm = FactorModel(["f"], {"a": {"f": 0.5}})
    model = Model(
        [Marginal("a", p10=10.0, p50=11.0, p90=12.0, family="normal")],
        fm,
        valuation=lambda v: v["a"],
    )
    result = model.run(n=20_000, price=1.0, seed=0)  # value >> price always

    assert result.p_undervalued == 1.0
    warnings_text = " ".join(result.warnings())
    # "too-narrow-inputs" (not the bare word "narrow") is the string unique to
    # this warning - the diagnostic-flag warning also contains "narrow"
    # ("Narrow the bands..."), so a bare substring match can't tell them apart.
    assert "too-narrow-inputs" in warnings_text.lower()
    assert "WARNING" in report.to_text(result).upper()


def test_p_undervalued_normal_does_not_warn() -> None:
    """No false positives: an ordinary spec has no warnings at all."""
    fm = FactorModel(["f"], {"a": {"f": 0.5}})
    model = Model(
        [Marginal("a", p10=1.0, p50=2.0, p90=3.0, family="normal")],
        fm,
        valuation=lambda v: v["a"],
    )
    result = model.run(n=5_000, price=2.0, seed=0)
    assert 0.0 < result.p_undervalued < 1.0
    assert result.warnings() == []


# --------------------------------------------------------------------------- #
# The two decisions `warnings()` makes, asked separately from the prose
# --------------------------------------------------------------------------- #
#
# `warnings()` returns sentences. A consumer that needs to MARK something - the
# viewer's portfolio ranking does - had either to re-derive the rule or to match
# on prose. Both are the shape this repo keeps paying for: a second copy of a
# judgment drifts, and a substring match breaks when the sentence is reworded.
# So the two decisions are now separately askable, and `warnings()` is their
# only other caller.


def test_a_rate_too_small_to_show_says_so_rather_than_rounding_to_none() -> None:
    """The "<0.01%" rule, which is the whole reason this is not `f"{r:.2%}"`."""
    fm = FactorModel(["f"], {"a": {"f": 0.5}})
    model = Model(
        [Marginal("a", p10=1.0, p50=2.0, p90=3.0, family="normal")],
        fm,
        valuation=lambda v: v["a"],
    )
    result = model.run(n=50_000, price=2.0, seed=0)

    result.diagnostics["x_floored"] = 1  # 0.002%, which rounds to 0.00%
    assert result.diagnostic_rate_text("x_floored") == "<0.01%"

    result.diagnostics["x_floored"] = 48  # 0.096%, the worst real case (A28)
    assert result.diagnostic_rate_text("x_floored") == "0.10%"

    result.diagnostics["x_floored"] = 5  # exactly the 0.01% threshold
    assert result.diagnostic_rate_text("x_floored") == "0.01%"

    assert result.diagnostic_rate_text("never_fired") == "<0.01%"


def test_the_rate_text_is_the_one_the_warning_prints(crossing_result) -> None:
    """One rule, not two. The extraction must not have changed the sentence."""
    name = next(iter(crossing_result.diagnostics))
    text = crossing_result.diagnostic_rate_text(name)
    warning = next(w for w in crossing_result.warnings() if w.startswith(name))
    assert f"({text})" in warning, f"{text!r} is not what the warning says: {warning!r}"


def test_degeneracy_is_askable_without_reading_the_prose() -> None:
    """The trigger as a fact. Both ends, and the guard that is easiest to drop
    when someone re-derives this: no price means no p_undervalued to be degenerate
    about, which is not the same as "not degenerate by luck"."""
    fm = FactorModel(["f"], {"a": {"f": 0.5}})
    mk = lambda price: Model(  # noqa: E731 - three near-identical runs read better inline
        [Marginal("a", p10=10.0, p50=11.0, p90=12.0, family="normal")],
        fm,
        valuation=lambda v: v["a"],
    ).run(n=20_000, price=price, seed=0)

    assert mk(1.0).degenerate() is True, "every draw above the price is degenerate"
    assert mk(1e9).degenerate() is True, "every draw below the price is degenerate too"
    assert mk(11.0).degenerate() is False
    assert mk(None).degenerate() is False, "no price cannot be degenerate"


def test_degeneracy_and_the_warning_cannot_disagree() -> None:
    """The composition: `warnings()` says it exactly when `degenerate()` does."""
    fm = FactorModel(["f"], {"a": {"f": 0.5}})
    for price, expected in ((1.0, True), (11.0, False)):
        result = Model(
            [Marginal("a", p10=10.0, p50=11.0, p90=12.0, family="normal")],
            fm,
            valuation=lambda v: v["a"],
        ).run(n=20_000, price=price, seed=0)
        said = any("too-narrow-inputs" in w for w in result.warnings())
        assert result.degenerate() is expected is said, (
            f"at price {price}: degenerate()={result.degenerate()}, warned={said}"
        )


# --------------------------------------------------------------------------- #
# Regression guard: the golden spec must be untouched by all of this
# --------------------------------------------------------------------------- #


def test_golden_vici_run_is_unchanged_and_warning_free() -> None:
    """examples/vici.yaml's bands do not cross, so no flag may fire and the
    documented numbers must come out identical. A diagnostics channel that
    perturbs the golden run is a failed fix.
    """
    from conftest import VICI_SPEC_PATH

    from valdist.spec.loader import hydrate, load_spec

    spec = load_spec(VICI_SPEC_PATH)
    result = hydrate(spec).run(n=spec.n, price=spec.price, seed=spec.seed, nu=spec.nu)

    assert result.diagnostics == {}, f"golden spec raised flags: {result.diagnostics}"
    assert result.warnings() == []

    summary = result.summary()
    assert summary["value_p10"] == pytest.approx(28.01, abs=0.01)
    assert summary["value_p50"] == pytest.approx(33.70, abs=0.01)
    assert summary["value_p90"] == pytest.approx(41.00, abs=0.01)
    assert result.p_undervalued == pytest.approx(0.9312, abs=0.0001)
