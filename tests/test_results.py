"""Direct coverage of Result methods not otherwise exercised:
value_quantiles() custom qs, margin_of_safety() directly, diagnostic_rate()
edge cases, the _need_price() error path, and convergence()'s actual contract
(previously only tested for param-threading and for not-crashing - nothing
asserted what it returns).
"""

from __future__ import annotations

import numpy as np
import pytest

from valdist.core.factors import FactorModel
from valdist.core.marginals import Marginal
from valdist.core.model import Model


def _priced_result(price: float | None):
    fm = FactorModel(["f"], {"a": {"f": 0.5}})
    model = Model(
        [Marginal("a", p10=8.0, p50=10.0, p90=12.0, family="normal")],
        fm,
        valuation=lambda v: v["a"],
    )
    return model.run(n=2000, price=price, seed=0)


def test_need_price_raises_on_p_undervalued_without_price() -> None:
    result = _priced_result(price=None)
    with pytest.raises(ValueError, match="price"):
        result.p_undervalued


def test_need_price_raises_on_margin_of_safety_without_price() -> None:
    result = _priced_result(price=None)
    with pytest.raises(ValueError, match="price"):
        result.margin_of_safety()


def test_value_quantiles_default_qs() -> None:
    result = _priced_result(price=10.0)
    vq = result.value_quantiles()
    assert set(vq.keys()) == {0.10, 0.50, 0.90}
    assert vq[0.10] <= vq[0.50] <= vq[0.90]


def test_value_quantiles_custom_qs() -> None:
    result = _priced_result(price=10.0)
    vq = result.value_quantiles(qs=(0.05, 0.25, 0.75, 0.95))
    assert set(vq.keys()) == {0.05, 0.25, 0.75, 0.95}
    assert vq[0.05] <= vq[0.25] <= vq[0.75] <= vq[0.95]


def test_margin_of_safety_matches_value_over_price_minus_one() -> None:
    result = _priced_result(price=10.0)
    mos = result.margin_of_safety()
    vq = result.value_quantiles()
    for q in (0.10, 0.50, 0.90):
        assert mos[q] == pytest.approx(vq[q] / 10.0 - 1.0)


def test_margin_of_safety_custom_qs() -> None:
    result = _priced_result(price=10.0)
    mos = result.margin_of_safety(qs=(0.25, 0.75))
    assert set(mos.keys()) == {0.25, 0.75}


def test_diagnostic_rate_zero_for_flag_that_never_fired() -> None:
    result = _priced_result(price=10.0)
    assert result.diagnostic_rate("some_flag_that_never_fired") == 0.0


# --------------------------------------------------------------------------- #
# convergence() - previously only tested for param-threading (test_param_
# threading.py) and for not-crashing (test_degenerate.py). Nothing asserted
# what it actually returns or that the drift it reports is sane.
# --------------------------------------------------------------------------- #


def test_convergence_reports_expected_keys_with_price() -> None:
    result = _priced_result(price=10.0)
    conv = result.convergence()

    assert set(conv) == {
        "value_p50_N",
        "value_p50_2N",
        "p_undervalued_N",
        "p_undervalued_2N",
    }
    assert all(isinstance(v, float) for v in conv.values())


def test_convergence_omits_price_keys_when_price_is_none() -> None:
    """convergence() must not raise on a price-less run - it just reports the
    value drift and skips the p_undervalued pair."""
    result = _priced_result(price=None)
    conv = result.convergence()

    assert set(conv) == {"value_p50_N", "value_p50_2N"}


def test_convergence_n_statistic_matches_the_run_it_came_from() -> None:
    """The _N half is the ALREADY-COMPUTED statistic, not a fresh re-run."""
    result = _priced_result(price=10.0)
    conv = result.convergence()

    assert conv["value_p50_N"] == pytest.approx(float(np.median(result.value)))
    assert conv["p_undervalued_N"] == pytest.approx(result.p_undervalued)


def test_convergence_2n_half_matches_an_explicit_2n_run() -> None:
    """The _2N half must be exactly Model.run(n=2N) at the same seed --
    that is the whole point of the check."""
    fm = FactorModel(["f"], {"a": {"f": 0.5}})
    model = Model(
        [Marginal("a", p10=8.0, p50=10.0, p90=12.0, family="normal")],
        fm,
        valuation=lambda v: v["a"],
    )
    result = model.run(n=2000, price=10.0, seed=0)
    conv = result.convergence()

    manual_2n = model.run(n=4000, price=10.0, seed=0)
    assert conv["value_p50_2N"] == pytest.approx(float(np.median(manual_2n.value)))
    assert conv["p_undervalued_2N"] == pytest.approx(manual_2n.p_undervalued)


SUMMARY_KEYS = {
    "n",
    "value_p10",
    # The conservative value anchor and its dispersion read, added 2026-07-31.
    # Extending this set is a deliberate contract change; the test below is what
    # forces it to be deliberate.
    "value_p25",
    "p25_p50_ratio",
    "value_p50",
    "value_p90",
    "price",
    "p_undervalued",
    "p_undervalued_stderr",
    "mos_p10",
    "mos_p25",
    "mos_p50",
    "mos_p90",
}


def test_summary_shape_is_the_same_with_and_without_a_price() -> None:
    """summary() used to return a DIFFERENT SET OF KEYS depending on whether a
    price was set: {n, value_p10/p50/p90} without one, and six more keys with
    one. Programmatic consumers had to probe (`if "p_undervalued" in s`) instead
    of relying on a stable contract. The keys are now fixed; the price-dependent
    values are None when there is no price to compare against."""
    priced = _priced_result(price=10.0).summary()
    unpriced = _priced_result(price=None).summary()

    assert set(priced) == SUMMARY_KEYS
    assert set(unpriced) == SUMMARY_KEYS, "shape still depends on whether price is set"


def test_summary_price_dependent_values_are_none_without_a_price() -> None:
    s = _priced_result(price=None).summary()

    assert s["price"] is None
    for key in (
        "p_undervalued",
        "p_undervalued_stderr",
        "mos_p10",
        "mos_p25",
        "mos_p50",
        "mos_p90",
    ):
        assert s[key] is None, f"{key} should be None when no price is set, got {s[key]!r}"

    # The value distribution is still fully reported - it does not need a price.
    assert s["n"] == 2000
    assert s["value_p10"] <= s["value_p25"] <= s["value_p50"] <= s["value_p90"]


def test_summary_price_dependent_values_are_populated_with_a_price() -> None:
    s = _priced_result(price=10.0).summary()

    assert s["price"] == 10.0
    assert 0.0 <= s["p_undervalued"] <= 1.0
    assert s["p_undervalued_stderr"] >= 0.0
    assert s["mos_p10"] <= s["mos_p50"] <= s["mos_p90"]


def test_to_json_shape_is_stable_without_a_price() -> None:
    """The JSON payload must carry the same keys either way, with nulls - a
    consumer should not have to branch on which keys exist."""
    import json

    from valdist import report

    payload = json.loads(report.to_json(_priced_result(price=None)))

    assert SUMMARY_KEYS <= set(payload)
    assert payload["price"] is None
    assert payload["p_undervalued"] is None
    # The honesty channels and the sensitivity table are still present.
    assert "tornado" in payload and "diagnostics" in payload and "warnings" in payload


def test_result_from_a_real_spec_pickles() -> None:
    """Four consecutive audits have flagged `Result.model = model` as
    "preventing pickling". It does not.
    """
    import pickle
    from pathlib import Path

    from conftest import VICI_SPEC_PATH

    from valdist.spec.loader import hydrate, load_spec

    spec = load_spec(Path(VICI_SPEC_PATH))
    result = hydrate(spec).run(n=1000, price=spec.price, seed=spec.seed, nu=spec.nu)

    restored = pickle.loads(pickle.dumps(result))

    assert restored.n == result.n
    assert restored.p_undervalued == pytest.approx(result.p_undervalued)
    np.testing.assert_array_equal(restored.value, result.value)
    # The model came along, so convergence() still works on the restored Result.
    assert restored.model.names == result.model.names


def test_valdist_run_rejects_a_bad_argument_type() -> None:
    """valdist.run() accepts a SpecModel or a path. Its TypeError branch was
    never exercised, so nothing pinned the contract it advertises."""
    import valdist

    with pytest.raises(TypeError, match="expected str, Path, or SpecModel"):
        valdist.run(123)
    with pytest.raises(TypeError, match="expected str, Path, or SpecModel"):
        valdist.run({"not": "a spec"})


def test_print_worlds_renders_a_row_per_quantile() -> None:
    """report.print_worlds() was only ever exercised through the CLI, so the
    formatter itself had no test - a broken column, a mislabelled quantile or a
    dropped driver would only surface as a CLI smoke-test string match."""
    from valdist import report

    fm = FactorModel(["f"], {"a": {"f": 0.5}, "b": {"f": -0.3}})
    model = Model(
        [
            Marginal("a", p10=1.0, p50=2.0, p90=3.0, family="normal"),
            Marginal("b", p10=10.0, p50=20.0, p90=30.0, family="normal"),
        ],
        fm,
        valuation=lambda v: v["a"] + v["b"],
    )
    result = model.run(n=2000, price=22.0, seed=0)

    text = report.print_worlds(result)
    lines = text.splitlines()

    header, rule, *rows = lines
    assert "Quantile" in header and "Value" in header
    assert "a" in header and "b" in header, "driver columns missing from header"
    assert set(rule) == {"-"}
    assert len(rows) == 3, f"expected one row per default quantile, got {rows}"
    for label, row in zip(("P10", "P50", "P90"), rows):
        assert row.strip().startswith(label), f"expected {label} row, got {row!r}"

    # Custom quantiles are honoured, and the value column is ascending.
    custom = report.print_worlds(result, qs=(0.25, 0.75)).splitlines()[2:]
    assert len(custom) == 2
    assert custom[0].strip().startswith("P25")
    assert custom[1].strip().startswith("P75")


def test_wrap_warning_handles_empty_text() -> None:
    """textwrap.wrap("") returns [], so `wrapped[0]` raised IndexError. No
    caller passes an empty warning today, but a report formatter that explodes
    on empty input is a landmine in the honesty-about-uncertainty path."""
    from valdist.report import _wrap_warning

    assert _wrap_warning("") == []
    assert _wrap_warning("   ") == []
    assert _wrap_warning("short") == ["    ! short"]


def test_factor_attribution_on_a_comonotonic_result_raises_typeerror() -> None:
    """calibrate.selftest.ComonotonicModel is passed to Result() behind a
    `# type: ignore[arg-type]` -- it is not a Model and has no `.corr`.
    """
    from valdist.calibrate.selftest import ComonotonicModel, ScenarioDriver

    oracle = ComonotonicModel(
        [ScenarioDriver("a", worst=1.0, base=2.0, best=3.0)],
        valuation=lambda v: v["a"],
    )
    result = oracle.run(n=500, price=2.0, seed=0)

    with pytest.raises(TypeError, match="FactorModel"):
        result.factor_attribution()


def test_convergence_drift_is_small_for_a_well_behaved_spec() -> None:
    """Sanity, and the reason convergence() exists: doubling n on an ordinary
    spec should barely move the estimates. A large drift here means the run is
    nowhere near converged - which is exactly what the caller wants to see."""
    fm = FactorModel(["f"], {"a": {"f": 0.5}})
    model = Model(
        [Marginal("a", p10=8.0, p50=10.0, p90=12.0, family="normal")],
        fm,
        valuation=lambda v: v["a"],
    )
    conv = model.run(n=20_000, price=10.0, seed=0).convergence()

    assert abs(conv["value_p50_2N"] - conv["value_p50_N"]) < 0.1
    assert abs(conv["p_undervalued_2N"] - conv["p_undervalued_N"]) < 0.02


# --------------------------------------------------------------------------- #
# tornado() correlation VALUES, not just rank order
#
# Everything else in the suite checks tornado()'s shape (driver names present,
# rows sorted, the degenerate constant-driver case -> 0.0). Nothing checked
# that the numbers themselves are the correlations they claim to be, and a
# mutant proved it: replacing `stats.rankdata(self.value)` with
# `stats.rankdata(None)` makes `ranks_out` a 1-element array, so `out_is_flat`
# is permanently True and EVERY driver reports 0.0. The whole suite stayed
# green. These two tests pin the values against references derived outside the
# implementation.
# --------------------------------------------------------------------------- #


def _iid_normal_model(valuation) -> Model:
    """Two independent, identically distributed standard-ish normal drivers."""
    drivers = [
        Marginal("a", p10=-1.0, p50=0.0, p90=1.0, family="normal"),
        Marginal("b", p10=-1.0, p50=0.0, p90=1.0, family="normal"),
    ]
    return Model(drivers, FactorModel([], {}), valuation=valuation)


def test_tornado_is_exactly_one_for_a_driver_the_value_is_monotone_in() -> None:
    """Spearman is rank-based, so a value that IS one of its drivers must
    correlate with it at exactly 1.0 - no tolerance, no MC error, every seed.
    The other driver is independent of it and lands near zero."""
    result = _iid_normal_model(lambda v: v["a"]).run(n=20_000, seed=0)
    corr = dict(result.tornado())

    assert corr["a"] == pytest.approx(1.0, abs=1e-12)
    assert corr["b"] == pytest.approx(0.0, abs=0.02)


def test_tornado_matches_the_closed_form_spearman_for_a_gaussian_valuation() -> None:
    """value = 2a - b, with a, b iid normal, is jointly normal with each driver,
    so Spearman has a closed form: (6/pi) * arcsin(rho/2).
    """
    result = _iid_normal_model(lambda v: 2.0 * v["a"] - v["b"]).run(n=20_000, seed=0)
    rows = result.tornado()
    corr = dict(rows)

    def spearman(rho: float) -> float:
        return float((6.0 / np.pi) * np.arcsin(rho / 2.0))

    assert corr["a"] == pytest.approx(spearman(2.0 / np.sqrt(5.0)), abs=0.02)
    assert corr["b"] == pytest.approx(spearman(-1.0 / np.sqrt(5.0)), abs=0.02)

    assert corr["a"] > 0.0 and corr["b"] < 0.0
    assert [name for name, _ in rows] == ["a", "b"], "sorted by |corr| descending"
