"""Tests for the health endpoints.

Two URLs must produce identical payloads:
  GET /health         (root alias)
  GET /api/health     (router path)

Neither requires auth, DB, or ML models.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_api_health_returns_ok(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    # Secrets audit: the response must NOT contain the words "secret" or
    # "password" as keys (no credential leakage ever).
    assert "secret" not in {k.lower() for k in body.keys()}
    assert "password" not in {k.lower() for k in body.keys()}
    # Required info fields
    for key in ("project", "environment", "demo_mode", "default_model", "version"):
        assert key in body, f"Missing '{key}' in health payload"


def test_root_health_alias_matches_api_health(client: TestClient) -> None:
    """The root /health and /api/health endpoints MUST share the single
    implementation (no duplicated code). We verify payload equivalence."""
    root = client.get("/health").json()
    api = client.get("/api/health").json()
    # Status + version always line up. The rest is static so we can
    # compare the full dicts to ensure drift can't happen.
    assert root == api


def test_health_does_not_require_auth(client: TestClient) -> None:
    """Docker health-checks must work without a JWT."""
    response = client.get("/health", headers={"Authorization": ""})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_openapi_present(client: TestClient) -> None:
    """Both health URLs should be visible in /openapi.json."""
    spec = client.get("/openapi.json").json()
    paths = spec.get("paths", {})
    assert "/health" in paths, "Root /health missing from OpenAPI spec"
    assert "/api/health" in paths, "/api/health missing from OpenAPI spec"
