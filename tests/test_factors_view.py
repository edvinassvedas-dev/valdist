"""The loadings view: engine accessors + the `/api/model` payload [gui/, no anchor]."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import VICI_SPEC_PATH

from valdist.core.factors import FactorModel


@pytest.fixture
def vici_spec():
    from valdist.spec.loader import load_spec

    return load_spec(VICI_SPEC_PATH)


# --------------------------------------------------------------------------- #
# ψ, promoted to the public interface
# --------------------------------------------------------------------------- #


def test_psi_is_the_idiosyncratic_share_the_sampler_uses() -> None:
    """Not a re-derivation: the same array `draw_z` scales its noise by."""
    fm = FactorModel(["rate", "fundamentals"], {"a": {"rate": 0.8}, "b": {"fundamentals": -0.6}})
    names = ["a", "b"]

    _lambda, psi = fm._build_arrays(names)
    assert np.allclose(fm.psi(names), psi)


def test_psi_and_the_loadings_close_the_variance_budget() -> None:
    """Σλ² + ψ² = 1 per driver - what makes the panel's two columns readable."""
    fm = FactorModel(
        ["rate", "fundamentals"],
        {"a": {"rate": 0.75, "fundamentals": -0.30}, "b": {}, "c": {"rate": 1.0}},
    )
    names = ["a", "b", "c"]
    lam, psi = fm.loadings_matrix(names), fm.psi(names)

    assert np.allclose(np.sum(lam**2, axis=1) + psi**2, 1.0)
    assert psi[1] == pytest.approx(1.0), "a driver loading on nothing is wholly idiosyncratic"
    assert psi[2] == pytest.approx(0.0), "a fully committed driver has no idiosyncratic share"


def test_psi_orders_itself_by_the_names_it_is_given() -> None:
    """The panel pairs each ψ with a driver name by position, so order is load-bearing."""
    fm = FactorModel(["rate"], {"a": {"rate": 0.6}, "b": {"rate": 0.0}})

    assert np.allclose(fm.psi(["a", "b"])[::-1], fm.psi(["b", "a"]))


def test_the_loader_builds_the_factor_model_one_way(vici_spec) -> None:
    """`hydrate` and the viewer must get the same Λ, not two constructions of it."""
    from valdist.spec.loader import factor_model, hydrate

    built = factor_model(vici_spec)
    hydrated = hydrate(vici_spec)
    names = list(vici_spec.drivers)

    assert built.factors == list(vici_spec.factors)
    assert np.allclose(built.loadings_matrix(names), hydrated.corr.loadings_matrix(names))


# --------------------------------------------------------------------------- #
# The payload
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def factors_payload():
    """The loadings half of `/api/model`, for a real shipped analysis."""
    from gui.server import ANALYSES_DIR, do_model

    name = sorted(p.name for p in ANALYSES_DIR.iterdir() if p.is_dir())[0]
    payload = do_model(name)
    assert payload["drivers"], f"{name} produced no driver rows - every sweep below would pass"
    assert payload["marginals"], f"{name} produced no marginals - same"
    return payload


def test_every_driver_reports_a_loading_for_every_factor(factors_payload) -> None:
    """Aligned to the factor list, zeros filled in, not a sparse dict."""
    d = factors_payload
    assert d["factors"], "the fixture analysis declares no factors - it cannot test this"
    for row in d["drivers"]:
        assert len(row["loadings"]) == len(d["factors"]), (
            f"{row['name']} carries {len(row['loadings'])} loadings for "
            f"{len(d['factors'])} factors - the renderer cannot align them"
        )
        assert all(isinstance(v, float) for v in row["loadings"])


def test_the_payload_closes_the_variance_budget_per_driver(factors_payload) -> None:
    """Σλ² and ψ come from the engine and still add up on the wire."""
    for row in factors_payload["drivers"]:
        assert row["sum_sq"] == pytest.approx(sum(v**2 for v in row["loadings"]))
        assert row["sum_sq"] + row["psi"] ** 2 == pytest.approx(1.0)
        assert 0.0 <= row["sum_sq"] <= 1.0 + 1e-9


def test_the_driver_order_matches_the_spec(factors_payload) -> None:
    """The same order every other view uses - tornado, worlds, the YAML itself."""
    from gui.server import _spec_path
    from valdist.spec.loader import load_spec

    spec = load_spec(_spec_path(factors_payload["name"]))
    assert [r["name"] for r in factors_payload["drivers"]] == list(spec.drivers)


@pytest.fixture
def transplanted(monkeypatch, tmp_path):
    """A real analysis copied into a temp directory, so its spec can be edited."""
    import shutil

    import gui.server as gs

    src = sorted((p for p in gs.ANALYSES_DIR.iterdir() if p.is_dir()))[0]
    dst = tmp_path / src.name
    shutil.copytree(src, dst)
    monkeypatch.setattr(gs, "ANALYSES_DIR", tmp_path)

    def edit(old: str, new: str) -> str:
        spec = dst / f"{src.name}.yaml"
        text = spec.read_text(encoding="utf-8")
        assert old in text, f"{old!r} is not in {spec.name} - the case would test nothing"
        spec.write_text(text.replace(old, new, 1), encoding="utf-8")
        return src.name

    return edit


def test_a_factor_nothing_loads_on_is_named(transplanted) -> None:
    """Declared and inert is not an error, and is invisible without saying so."""
    from gui.server import do_model

    name = transplanted("factors: [", "factors: [unused_by_anything, ")

    d = do_model(name)
    assert d["unloaded_factors"] == ["unused_by_anything"]
    assert d["factors"][0] == "unused_by_anything"
    for row in d["drivers"]:
        assert row["loadings"][0] == 0.0, "an unloaded factor must still occupy its column"


def test_the_factors_view_refuses_a_spec_the_validator_rejects(transplanted) -> None:
    """The validation gate, inherited rather than reimplemented."""
    from gui.server import do_model
    from valdist.spec.validate import SpecError

    name = transplanted('schema_version: "1.0"', 'schema_version: "9.9"')

    with pytest.raises(SpecError):
        do_model(name)


def test_psi_reaches_the_payload_as_a_finite_number(factors_payload) -> None:
    """numpy floats do not survive `json.dumps`; the endpoint must hand over plain ones."""
    import json

    json.dumps(factors_payload)  # raises TypeError on a numpy scalar
    for row in factors_payload["drivers"]:
        assert math.isfinite(row["psi"]) and row["psi"] >= 0.0
