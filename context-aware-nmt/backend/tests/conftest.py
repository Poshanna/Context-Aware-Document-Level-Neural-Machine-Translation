"""Shared pytest fixtures for the backend foundation phase.

Provides:
  * ``client`` — synchronous httpx TestClient pointed at the FastAPI app,
    using the ASGI transport so no server is started.
  * ``auth_headers`` — a valid JWT for a synthetic fixture user so tests
    can hit authenticated endpoints easily.
  * ``history_svc_singleton_reset`` — resets the in-process history store
    between tests to keep them deterministic.
"""

from __future__ import annotations

import datetime
import os
import secrets
import uuid

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Ensure .env is not required for tests — set required env vars before the
# app module is imported (settings is loaded eagerly via pydantic-settings).
# ---------------------------------------------------------------------------

_REQUIRED_FOR_TESTS = {
    "JWT_SECRET_KEY": "test-secret-not-for-production-please-ignore-xyz",
    "POSTGRES_PASSWORD": "test-pg-password-placeholder-only",
    "POSTGRES_SERVER": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "context_aware_nmt_test",
    "POSTGRES_USER": "postgres",
    "ENVIRONMENT": "development",
    "DEBUG": "true",
}

for k, v in _REQUIRED_FOR_TESTS.items():
    os.environ.setdefault(k, v)


from app.core.security import create_access_token  # noqa: E402
from app.main import app  # noqa: E402
from app.services.history_service import get_history_service  # noqa: E402


@pytest.fixture(autouse=True)
def history_svc_singleton_reset() -> None:
    """Clear the in-process history store before every test.

    Without this, tests that populate history would bleed into one another
    because the singleton is module-level.
    """
    svc = get_history_service()
    with svc._lock:
        svc._store.clear()
    yield


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def test_user_id() -> uuid.UUID:
    return uuid.UUID("96f109ab-8765-4321-1234-567890abcdef")


@pytest.fixture()
def auth_headers(test_user_id: uuid.UUID) -> dict[str, str]:
    """Headers containing a valid Authorization: Bearer JWT for the test user."""
    token = create_access_token(
        subject=str(test_user_id),
        expires_delta=datetime.timedelta(minutes=60),
        extra={"email": "testuser@example.invalid"},
    )
    return {"Authorization": f"Bearer {token}"}
