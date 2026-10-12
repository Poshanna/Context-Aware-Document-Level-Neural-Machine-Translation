"""Tests for POST /api/translate/text.

Baseline translation is exercised with fake translators only; no model is
downloaded or loaded during pytest.
"""

from __future__ import annotations

import uuid
from typing import ClassVar

from fastapi.testclient import TestClient
import pytest

from app.services import translation_service as translation_service_module
from app.services.translation_service import TranslationService
from app.translation import (
    ModelUnavailableError,
    TranslationContext,
    Translator,
    TranslatorInfo,
    TranslatorRegistry,
)


_ENDPOINT = "/api/translate/text"
_VALID_PAYLOAD = {
    "source_lang": "te",
    "target_lang": "en",
    "pipeline": "baseline",
    "model": "nllb200",
    "text": "మీరు ఎలా ఉన్నారు? నేను బాగున్నాను.",
}


class _FakeBaselineTranslator(Translator):
    info: ClassVar[TranslatorInfo] = TranslatorInfo(
        name="nllb200",
        max_seq_length=512,
        supported_directions={("en", "te"), ("te", "en")},
    )

    def load(self) -> None:
        self.loaded = True

    def translate_batch(
        self,
        sentences: list[str],
        *,
        src_lang: str,
        tgt_lang: str,
        context_groups: list[list[TranslationContext]] | None = None,
    ) -> list[str]:
        return [f"mocked-{tgt_lang}:{sentence}" for sentence in sentences]


class _UnavailableFakeTranslator(_FakeBaselineTranslator):
    def load(self) -> None:
        raise ModelUnavailableError("NLLB-200 pretrained checkpoint is unavailable.")


@pytest.fixture
def mocked_baseline_service(monkeypatch: pytest.MonkeyPatch) -> TranslationService:
    translator_registry = TranslatorRegistry()
    translator_registry.populate()
    translator_registry.register("nllb200", _FakeBaselineTranslator)
    service = TranslationService(translator_registry=translator_registry)
    monkeypatch.setattr(translation_service_module, "_default_service", service)
    return service


@pytest.fixture
def unavailable_baseline_service(monkeypatch: pytest.MonkeyPatch) -> TranslationService:
    translator_registry = TranslatorRegistry()
    translator_registry.populate()
    translator_registry.register("nllb200", _UnavailableFakeTranslator)
    service = TranslationService(translator_registry=translator_registry)
    monkeypatch.setattr(translation_service_module, "_default_service", service)
    return service


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_text_translate_returns_mocked_baseline_translation(
    client: TestClient,
    auth_headers: dict,
    mocked_baseline_service: TranslationService,
) -> None:
    del mocked_baseline_service
    response = client.post(_ENDPOINT, json=_VALID_PAYLOAD, headers=auth_headers)
    assert response.status_code == 200, response.text
    env = response.json()
    assert env["error"] is None
    data = env["data"]
    assert data["status"] == "ok"
    assert data["translated_text"].startswith("mocked-en:")
    assert data["tokens_used"] is None
    # A job_id must exist (we will reuse it for history round-trip tests).
    assert uuid.UUID(data["job_id"]).version == 4
    # The fake translator returns exactly one result per segmented sentence.
    assert len(data["sentences"]) >= 1
    for s in data["sentences"]:
        assert s["sentence_idx"] >= 0
        assert isinstance(s["source"], str) and s["source"] != ""
        assert isinstance(s["target"], str) and s["target"].startswith("mocked-en:")
        assert isinstance(s["context"], list)
    # The response labels this as the sentence-level pretrained-checkpoint baseline.
    assert isinstance(data["note"], str) and len(data["note"]) > 10
    assert "baseline" in data["note"]


def test_model_unavailable_returns_controlled_503(
    client: TestClient,
    auth_headers: dict,
    unavailable_baseline_service: TranslationService,
) -> None:
    del unavailable_baseline_service
    response = client.post(_ENDPOINT, json=_VALID_PAYLOAD, headers=auth_headers)

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "model_unavailable"
    assert "checkpoint is unavailable" in error["message"]


def test_symmetry_en_to_te_and_context_aware_pipeline(
    client: TestClient, auth_headers: dict
) -> None:
    payload = {
        **_VALID_PAYLOAD,
        "source_lang": "en",
        "target_lang": "te",
        "pipeline": "context_aware",
        "model": "indictrans2",
        "text": "Hello. How are you today?",
    }
    response = client.post(_ENDPOINT, json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "not_implemented"
    assert data["translated_text"] is None


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_text_translate_requires_auth(client: TestClient) -> None:
    response = client.post(_ENDPOINT, json=_VALID_PAYLOAD)
    assert response.status_code == 401
    body = response.json()
    assert "error" in body  # Envelope shape


# ---------------------------------------------------------------------------
# Validation errors (422 from Pydantic / 400 from service)
# ---------------------------------------------------------------------------


def test_rejects_empty_text(client: TestClient, auth_headers: dict) -> None:
    payload = {**_VALID_PAYLOAD, "text": ""}
    response = client.post(_ENDPOINT, json=payload, headers=auth_headers)
    # Pydantic schema rejects empty strings first (min_length=1)
    assert response.status_code in (400, 422)
    assert response.json().get("error") is not None


def test_rejects_blank_whitespace_text(client: TestClient, auth_headers: dict) -> None:
    payload = {**_VALID_PAYLOAD, "text": "     \n\t  "}
    # Pydantic allows whitespace (it passes min_length=1). Service rejects.
    response = client.post(_ENDPOINT, json=payload, headers=auth_headers)
    assert response.status_code in (400, 422)
    assert response.json().get("error") is not None


def test_rejects_same_source_and_target_language(
    client: TestClient, auth_headers: dict
) -> None:
    payload = {**_VALID_PAYLOAD, "source_lang": "en", "target_lang": "en"}
    response = client.post(_ENDPOINT, json=payload, headers=auth_headers)
    assert response.status_code == 400, response.text
    code = response.json()["error"]["code"]
    assert code == "same_language"


def test_rejects_unsupported_language(client: TestClient, auth_headers: dict) -> None:
    payload = {**_VALID_PAYLOAD, "source_lang": "fr", "target_lang": "en"}
    # Pydantic regex pattern on the field rejects this with 422 envelope.
    response = client.post(_ENDPOINT, json=payload, headers=auth_headers)
    assert response.status_code == 422
    assert response.json().get("error") is not None


def test_rejects_unsupported_pair_for_model(
    client: TestClient, auth_headers: dict
) -> None:
    # IndicTrans2 does NOT support Hindi in this phase.
    payload = {
        **_VALID_PAYLOAD,
        "source_lang": "en",
        "target_lang": "hi",
        "model": "indictrans2",
    }
    response = client.post(_ENDPOINT, json=payload, headers=auth_headers)
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "unsupported_language_pair"


def test_rejects_invalid_pipeline(client: TestClient, auth_headers: dict) -> None:
    payload = {**_VALID_PAYLOAD, "pipeline": "magic_mode"}
    response = client.post(_ENDPOINT, json=payload, headers=auth_headers)
    assert response.status_code == 422  # Pydantic pattern
    assert response.json().get("error") is not None


def test_rejects_invalid_model(client: TestClient, auth_headers: dict) -> None:
    payload = {**_VALID_PAYLOAD, "model": "gpt7"}
    response = client.post(_ENDPOINT, json=payload, headers=auth_headers)
    assert response.status_code == 422
    assert response.json().get("error") is not None


def test_history_is_populated_after_successful_translate_call(
    client: TestClient,
    auth_headers: dict,
    mocked_baseline_service: TranslationService,
) -> None:
    """Successful text translate must record a row in history service."""
    del mocked_baseline_service
    tx = client.post(_ENDPOINT, json=_VALID_PAYLOAD, headers=auth_headers)
    assert tx.status_code == 200
    job_id = tx.json()["data"]["job_id"]

    hist = client.get("/api/history", headers=auth_headers)
    assert hist.status_code == 200
    items = hist.json()["data"]["items"]
    assert any(h["job_id"] == job_id for h in items)
