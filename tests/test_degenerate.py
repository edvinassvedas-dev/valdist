"""Crash paths on specs that are already schema-legal."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from valdist.core import Marginal, Model
from valdist.core.factors import FactorModel


def _model_with_constant_driver() -> Model:
    """One live driver + one constant (degenerate) driver."""
    fm = FactorModel(["f"], {"a": {"f": 0.5}, "flat": {}})
    drivers = [
        Marginal("a", p10=1.0, p50=2.0, p90=3.0, family="normal"),
        Marginal("flat", p10=5.0, p50=5.0, p90=5.0, family="triangular"),
    ]
    return Model(drivers, fm, valuation=lambda v: v["a"] + v["flat"])


# --------------------------------------------------------------------------- #
# Item 4: ComonotonicModel.run() must accept Model.run()'s signature
# --------------------------------------------------------------------------- #


def test_comonotonic_convergence_does_not_raise() -> None:
    """Result.convergence() re-runs via self.model.run(sampling=..., nu=...);
    ComonotonicModel must accept that signature like Model does.
    """
    from valdist.calibrate.selftest import ComonotonicModel, ScenarioDriver

    cm = ComonotonicModel([ScenarioDriver("a", 1.0, 2.0, 3.0)], valuation=lambda v: v["a"])
    result = cm.run(n=1000, price=2.0, seed=0)

    conv = result.convergence()  # previously: TypeError
    assert "value_p50_N" in conv
    assert "value_p50_2N" in conv
    assert np.isfinite(conv["value_p50_N"])
    assert np.isfinite(conv["value_p50_2N"])


def test_comonotonic_run_rejects_lhs_and_nu() -> None:
    """LHS and the t-copula are meaningless for a single-latent comonotonic
    oracle - it must say so rather than silently ignoring the request.
    """
    from valdist.calibrate.selftest import ComonotonicModel, ScenarioDriver

    cm = ComonotonicModel([ScenarioDriver("a", 1.0, 2.0, 3.0)], valuation=lambda v: v["a"])

    with pytest.raises(ValueError, match="comonotonic"):
        cm.run(n=100, seed=0, sampling="lhs")
    with pytest.raises(ValueError, match="comonotonic"):
        cm.run(n=100, seed=0, nu=3.0)


def test_comonotonic_run_result_records_defaults() -> None:
    """The oracle's Result carries the same sampling/nu fields Model.run() sets."""
    from valdist.calibrate.selftest import ComonotonicModel, ScenarioDriver

    cm = ComonotonicModel([ScenarioDriver("a", 1.0, 2.0, 3.0)], valuation=lambda v: v["a"])
    result = cm.run(n=500, price=2.0, seed=0)
    assert result.sampling == "mc"
    assert result.nu is None


# --------------------------------------------------------------------------- #
# Item 5: a constant driver must not produce NaN / crash the report
# --------------------------------------------------------------------------- #


def test_tornado_constant_driver_is_zero_not_nan() -> None:
    """A zero-variance driver explains none of the output spread, so its rank
    correlation is 0.0 - not NaN (and no numpy RuntimeWarning).
    """
    result = _model_with_constant_driver().run(n=1000, price=6.0, seed=0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a numpy RuntimeWarning fails the test
        tornado = dict(result.tornado())

    assert tornado["flat"] == 0.0, f"expected 0.0 for a constant driver, got {tornado['flat']}"
    assert np.isfinite(tornado["a"])


def test_report_to_text_survives_constant_driver() -> None:
    """report.to_text() previously died on int(nan) building the tornado bar."""
    from valdist import report

    result = _model_with_constant_driver().run(n=1000, price=6.0, seed=0)
    text = report.to_text(result)  # previously: ValueError
    assert "flat" in text
    assert "nan" not in text.lower()


def test_report_to_json_never_emits_bare_nan() -> None:
    """to_json() must be strict-valid JSON: bare NaN is not (RFC 8259), even
    though Python's own json.loads is lenient about it.
    """
    from valdist import report

    result = _model_with_constant_driver().run(n=1000, price=6.0, seed=0)
    payload = report.to_json(result)

    assert "NaN" not in payload

    # Reject the bare constants a lenient parser would otherwise accept.
    def _no_constants(token: str) -> float:
        raise AssertionError(f"non-standard JSON constant emitted: {token}")

    json.loads(payload, parse_constant=_no_constants)
