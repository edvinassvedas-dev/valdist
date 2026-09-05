"""Request-level tests for the viewer's HTTP surface."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from gui.server import ANALYSES_DIR, Handler


@pytest.fixture(scope="module")
def base_url():
    """A real server on a real port, torn down with the module."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def get(url: str, timeout: float = 180.0):
    """Status and decoded body, without raising on a 4xx/5xx."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"null"), resp.headers
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body or b"null"), exc.headers
        except ValueError:
            return exc.code, body, exc.headers


@pytest.fixture(scope="module")
def a_real_analysis() -> str:
    names = sorted(p.name for p in ANALYSES_DIR.iterdir() if p.is_dir())
    assert names, "no entries found - these tests would pass vacuously"
    return names[0]


# --------------------------------------------------------------------------- #
# The page itself
# --------------------------------------------------------------------------- #


def test_index_is_served_with_a_locked_down_csp(base_url) -> None:
    """The CSP is the structural half of "no data retrieval in the viewer"."""
    with urllib.request.urlopen(f"{base_url}/", timeout=30) as resp:
        assert resp.status == 200
        assert "text/html" in resp.headers["Content-Type"]
        assert resp.headers["Content-Security-Policy"] == "default-src 'self' 'unsafe-inline'"
        assert b"<title>" in resp.read()


def test_an_unknown_endpoint_is_404_not_500(base_url) -> None:
    status, body, _ = get(f"{base_url}/api/nope")
    assert status == 404
    assert "no such endpoint" in body["error"]


# --------------------------------------------------------------------------- #
# Path traversal, on every endpoint that takes a name
# --------------------------------------------------------------------------- #
#
# Already verified clean in the audit and pinned here so it stays that way. The
# reason it holds is worth stating: the name is membership-tested against the
# real directory listing rather than string-sanitised, so there is no escaping
# rule to get subtly wrong.

TRAVERSALS = [
    "../CLAUDE",
    "../../etc/passwd",
    "..%2f..%2fCLAUDE",
    "foo/../../CLAUDE",
    "/etc/passwd",
]


@pytest.mark.parametrize("endpoint", ["analysis", "validate", "worlds", "convergence"])
@pytest.mark.parametrize("name", TRAVERSALS)
def test_traversal_is_refused(base_url, endpoint: str, name: str) -> None:
    status, body, _ = get(f"{base_url}/api/{endpoint}?name={urllib.parse.quote(name, safe='')}")
    assert status == 404, f"{endpoint} accepted {name!r}"
    assert "traceback" not in body


@pytest.mark.parametrize("endpoint", ["analysis", "validate", "run", "worlds", "convergence"])
def test_a_missing_name_is_400_not_500(base_url, endpoint: str) -> None:
    """A malformed request is the caller's fault, and says so."""
    status, body, _ = get(f"{base_url}/api/{endpoint}")
    assert status == 400
    assert "name" in body["error"]


# --------------------------------------------------------------------------- #
# Price overrides are validated server-side
# --------------------------------------------------------------------------- #
#
# The UI checks these too, but the endpoint is reachable directly, so a
# client-side check is a convenience and not a guard. The refusals mirror the
# validator's own `nonpositive_price` / `nonfinite_value`: a price <= 0 makes
# margin-of-safety meaningless, and nan/inf poisons every statistic downstream.


@pytest.mark.parametrize("price", ["0", "-1", "nan", "inf", "-inf", "abc", "1,5"])
def test_a_bad_price_override_is_refused(base_url, a_real_analysis, price: str) -> None:
    status, body, _ = get(f"{base_url}/api/run?name={a_real_analysis}&price={price}")
    assert status == 400, f"price={price!r} was accepted"
    assert "price" in body["error"]


def test_an_empty_price_means_no_override(base_url, a_real_analysis) -> None:
    """Distinct from a bad one: the UI sends an empty field for "use the spec"."""
    status, body, _ = get(f"{base_url}/api/run?name={a_real_analysis}&price=")
    assert status == 200
    assert body["overridden"] is False
    assert body["price_used"] == body["spec_price"]


# --------------------------------------------------------------------------- #
# The endpoints that do real work, once each
# --------------------------------------------------------------------------- #


def test_run_returns_a_payload_the_page_can_render(base_url, a_real_analysis) -> None:
    """Pins the fields `renderRun` actually reads, not merely a 200."""
    status, body, _ = get(f"{base_url}/api/run?name={a_real_analysis}")
    assert status == 200
    for key in ("payload", "text", "schema", "histogram", "spec_price", "price_used"):
        assert key in body, f"/api/run response lost {key!r}"
    for key in ("p_undervalued", "p_undervalued_stderr", "tornado", "warnings", "diagnostics"):
        assert key in body["payload"], f"run payload lost {key!r}"
    assert body["schema"]["supported"] is True


def test_a_price_override_moves_only_the_price(base_url, a_real_analysis) -> None:
    """The what-if guarantee, over HTTP: value is bit-identical at any price."""
    _, base, _ = get(f"{base_url}/api/run?name={a_real_analysis}")
    _, moved, _ = get(f"{base_url}/api/run?name={a_real_analysis}&price=1000")

    assert moved["overridden"] is True
    assert moved["price_used"] == 1000
    for key in ("value_p10", "value_p50", "value_p90"):
        assert moved["payload"][key] == base["payload"][key], f"{key} moved with the price"
    assert moved["payload"]["p_undervalued"] != base["payload"]["p_undervalued"]


def test_portfolio_returns_a_row_per_analysis(base_url) -> None:
    status, body, _ = get(f"{base_url}/api/portfolio")
    assert status == 200
    names = {row["name"] for row in body["portfolio"]}
    on_disk = {p.name for p in ANALYSES_DIR.iterdir() if p.is_dir()}
    assert names == on_disk, "the portfolio must account for every analysis, broken ones included"
    assert all("status" in row for row in body["portfolio"])


def test_compare_needs_both_sides(base_url, a_real_analysis) -> None:
    status, body, _ = get(f"{base_url}/api/compare?a={a_real_analysis}")
    assert status == 400
    assert "b" in body["error"]


def test_validate_reports_the_shipped_specs_as_valid(base_url, a_real_analysis) -> None:
    status, body, _ = get(f"{base_url}/api/validate?name={a_real_analysis}")
    assert status == 200
    assert body["errors"] == []
