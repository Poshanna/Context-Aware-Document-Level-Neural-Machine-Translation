"""Tests for GET /api/history and GET /api/history/{id}.

Foundation phase:
  * Empty store -> empty page, no fabrications.
  * After submitting a text / document translation, the history page shows
    the corresponding entry, and the detail endpoint returns it.
  * Requesting an unknown id -> HTTP 404.
  * Two different users have isolated history (verified via two JWTs).
"""

from __future__ import annotations

import datetime
import uuid

from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.schemas import TranslationResponse
from app.services.translation_service import get_translation_service


class _FakeHistoryTranslationService:
    def translate_text(self, user_id: uuid.UUID, req) -> TranslationResponse:
        return TranslationResponse(
            status="ok",
            job_id=uuid.uuid4(),
            translated_text=f"fake translation for {req.text}",
            sentences=[],
            note="Test-only fake response.",
        )


def _translate_text(client: TestClient, headers: dict, text: str = "హలో") -> str:
    payload = {
        "source_lang": "te",
        "target_lang": "en",
        "pipeline": "baseline",
        "model": "nllb200",
        "text": text,
    }
    dependency = get_translation_service
    client.app.dependency_overrides[dependency] = _FakeHistoryTranslationService
    try:
        r = client.post("/api/translate/text", json=payload, headers=headers)
    finally:
        client.app.dependency_overrides.pop(dependency, None)
    assert r.status_code == 200, r.text
    return r.json()["data"]["job_id"]


def _headers_for(user_id: uuid.UUID, email: str = "x@example.invalid") -> dict[str, str]:
    tok = create_access_token(
        subject=str(user_id),
        expires_delta=datetime.timedelta(minutes=10),
        extra={"email": email},
    )
    return {"Authorization": f"Bearer {tok}"}


# ---------------------------------------------------------------------------


def test_empty_history_returns_empty_page(
    client: TestClient, auth_headers: dict
) -> None:
    response = client.get("/api/history", headers=auth_headers)
    assert response.status_code == 200
    env = response.json()
    assert env["error"] is None
    data = env["data"]
    assert data["items"] == []
    assert data["page"] == 1
    assert data["size"] == 20
    assert data["total"] == 0


def test_history_requires_auth(client: TestClient) -> None:
    response = client.get("/api/history")
    assert response.status_code == 401


def test_history_returns_populated_rows_after_translate(
    client: TestClient, auth_headers: dict
) -> None:
    job_id = _translate_text(client, auth_headers)
    response = client.get("/api/history", headers=auth_headers)
    assert response.status_code == 200
    items = response.json()["data"]["items"]
    assert len(items) == 1
    row = items[0]
    assert row["job_id"] == job_id
    assert row["pipeline"] == "baseline"
    assert row["model_name"] == "nllb200"
    assert row["source_lang"] == "te"
    assert row["target_lang"] == "en"
    assert row["input_type"] == "text"
    # Fabrication rule: we never generate fake progress.
    assert row["progress_percent"] in {0, 100}


def test_history_detail_returns_known_job(
    client: TestClient, auth_headers: dict
) -> None:
    job_id = _translate_text(client, auth_headers)
    response = client.get(f"/api/history/{job_id}", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["error"] is None
    assert body["data"]["job_id"] == job_id


def test_history_detail_unknown_returns_404(
    client: TestClient, auth_headers: dict
) -> None:
    missing = "00000000-0000-0000-0000-000000000000"
    response = client.get(f"/api/history/{missing}", headers=auth_headers)
    assert response.status_code == 404
    body = response.json()
    assert body["error"] is not None
    assert body["error"]["code"] == "not_found"


def test_history_detail_invalid_uuid_returns_404(
    client: TestClient, auth_headers: dict
) -> None:
    response = client.get("/api/history/not-a-uuid", headers=auth_headers)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_history_is_per_user_isolated(client: TestClient) -> None:
    """User B must not see user A's history entries."""
    uid_a = uuid.UUID("11111111-1111-1111-1111-111111111111")
    uid_b = uuid.UUID("22222222-2222-2222-2222-222222222222")
    ha = _headers_for(uid_a, "a@x.invalid")
    hb = _headers_for(uid_b, "b@x.invalid")

    _translate_text(client, ha, "A text one")
    _translate_text(client, ha, "A text two")
    _translate_text(client, hb, "B only")

    a_page = client.get("/api/history", headers=ha).json()["data"]
    b_page = client.get("/api/history", headers=hb).json()["data"]
    assert a_page["total"] == 2
    assert b_page["total"] == 1
    b_job_id = b_page["items"][0]["job_id"]
    # A cannot read B's detail
    forbidden = client.get(f"/api/history/{b_job_id}", headers=ha)
    assert forbidden.status_code == 404
