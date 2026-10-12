"""Tests for causal document context and the context-aware API foundation."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.context import ContextRetrieverRegistry
from app.context.formatter import format_context_input
from app.context.lastn_retriever import LastNRetriever
from app.core.config import Settings
from app.preprocessing.sentence_splitter import SentenceSplitter
from app.schemas import TranslationResponse
from app.services.context_conditioned_translation_service import (
    ContextConditionedTranslationService,
)
from app.services.context_translation_service import (
    ContextAwareTranslationService,
    get_context_translation_service,
)
from app.services.translation_service import TranslationServiceError


def test_sentence_splitter_preserves_order_and_handles_english_telugu_punctuation() -> None:
    splitter = SentenceSplitter("en")
    text = 'First sentence. "Second question?" Telugu వాక్యం। Last line\nNext line.'

    assert splitter.split(text) == [
        "First sentence.",
        '"Second question?"',
        "Telugu వాక్యం।",
        "Last line",
        "Next line.",
    ]


def test_sentence_splitter_empty_and_single_sentence() -> None:
    splitter = SentenceSplitter("te")

    assert splitter.split(" \n ") == []
    assert splitter.split("శుభోదయం.") == ["శుభోదయం."]


def test_lastn_window_has_no_future_sentence_leakage() -> None:
    retriever = LastNRetriever()
    retriever.build_index(["A", "B", "C", "D"])

    hits = retriever.get_context(2, k=2, window=2)

    assert [hit.idx for hit in hits] == [0, 1]
    assert all(hit.idx < 2 for hit in hits)
    assert retriever.get_context(2, k=0, window=2) == []


def test_context_formatter_separates_context_and_current_sentence() -> None:
    assert format_context_input(["Earlier one.", "Earlier two."], "Current.") == (
        "CONTEXT:\nEarlier one.\nEarlier two.\n\nCURRENT SENTENCE:\nCurrent."
    )
    assert format_context_input([], "Only sentence.") == (
        "CONTEXT:\n\nCURRENT SENTENCE:\nOnly sentence."
    )


class FakeTranslationService:
    def __init__(self) -> None:
        self.translated_inputs: list[str] = []

    def translate_text(self, *, user_id: uuid.UUID, req) -> TranslationResponse:
        self.translated_inputs.append(req.text)
        return TranslationResponse(
            status="ok",
            job_id=uuid.uuid4(),
            translated_text=f"fake:{req.text}",
            sentences=[],
            note="Mocked baseline response.",
        )


class FakeTranslatorBackend:
    def __init__(self) -> None:
        self.model_inputs: list[str] = []

    def translate_batch(self, sentences: list[str], *, src_lang: str, tgt_lang: str, context_groups=None) -> list[str]:
        self.model_inputs.extend(sentences)
        return [f"fake:{sentence}" for sentence in sentences]


def _service() -> tuple[ContextAwareTranslationService, FakeTranslationService, FakeTranslatorBackend]:
    fake_translator = FakeTranslationService()
    fake_backend = FakeTranslatorBackend()
    context_registry = ContextRetrieverRegistry()
    context_registry.register("lastn", LastNRetriever)
    settings = Settings(
        _env_file=None,
        JWT_SECRET_KEY="test-secret",
        POSTGRES_PASSWORD="test-password",
        CONTEXT_K=2,
    )
    context_conditioned_service = ContextConditionedTranslationService(
        translation_service=fake_translator,
        translator_backend=fake_backend,
        settings=settings,
    )
    service = ContextAwareTranslationService(
        translation_service=fake_translator,
        context_registry=context_registry,
        settings=settings,
        context_conditioned_service=context_conditioned_service,
    )
    return service, fake_translator, fake_backend


def test_context_service_uses_prior_context_but_translates_current_sentences_only() -> None:
    service, fake_translator, _ = _service()

    result = service.translate_document(
        user_id=uuid.uuid4(),
        source_lang="en",
        target_lang="te",
        text="First. Second. Third.",
        context_window=1,
    )

    assert result.source_sentences == ["First.", "Second.", "Third."]
    assert fake_translator.translated_inputs == result.source_sentences
    assert [sentence.context for sentence in result.sentences] == [
        [],
        ["First."],
        ["Second."],
    ]
    assert [sentence.context_indices for sentence in result.sentences] == [
        [],
        [0],
        [1],
    ]
    assert all(sentence.formatted_input.startswith("CONTEXT:\n") for sentence in result.sentences)
    assert result.metadata["translation_uses_context"] is False
    assert result.metadata["context_is_causal"] is True


def test_context_service_empty_and_single_sentence_documents() -> None:
    service, fake_translator, _ = _service()

    empty = service.translate_document(
        user_id=uuid.uuid4(),
        source_lang="en",
        target_lang="te",
        text=" \n ",
    )
    single = service.translate_document(
        user_id=uuid.uuid4(),
        source_lang="te",
        target_lang="en",
        text="శుభోదయం.",
    )

    assert empty.source_sentences == []
    assert empty.translations == []
    assert single.sentences[0].context == []
    assert fake_translator.translated_inputs == ["శుభోదయం."]


def test_context_service_prefix_conditioned_uses_context_in_backend_input() -> None:
    service, _, fake_backend = _service()

    result = service.translate_document(
        user_id=uuid.uuid4(),
        source_lang="en",
        target_lang="te",
        text="Ravi bought a book. He read it that night.",
        context_window=1,
        context_mode="prefix_conditioned",
    )

    assert result.metadata["context_mode"] == "prefix_conditioned"
    assert result.metadata["context_used_for_translation"] is True
    assert result.sentences[1].context == ["Ravi bought a book."]
    assert result.sentences[1].formatted_input == (
        "CONTEXT:\nRavi bought a book.\n\nCURRENT SENTENCE:\nHe read it that night."
    )
    assert fake_backend.model_inputs[0] == (
        "CONTEXT:\n\nCURRENT SENTENCE:\nRavi bought a book."
    )
    assert fake_backend.model_inputs[1] == (
        "CONTEXT:\nRavi bought a book.\n\nCURRENT SENTENCE:\nHe read it that night."
    )


@pytest.mark.parametrize(
    ("context_window", "expected_context"),
    [
        (0, [[], []]),
        (1, [[], ["First sentence."]]),
        (2, [[], ["First sentence."]]),
    ],
)
def test_context_service_window_modes_are_stable(context_window: int, expected_context: list[list[str]]) -> None:
    service, _, fake_backend = _service()

    result = service.translate_document(
        user_id=uuid.uuid4(),
        source_lang="en",
        target_lang="te",
        text="First sentence. Second sentence.",
        context_window=context_window,
        context_mode="prefix_conditioned",
    )

    assert [sentence.context for sentence in result.sentences] == expected_context
    assert all("CONTEXT:" in sentence.formatted_input for sentence in result.sentences)
    assert len(fake_backend.model_inputs) == len(result.sentences)
    assert fake_backend.model_inputs[1] == (
        "CONTEXT:\nFirst sentence.\n\nCURRENT SENTENCE:\nSecond sentence."
    ) if context_window > 0 else fake_backend.model_inputs[1] == "CONTEXT:\n\nCURRENT SENTENCE:\nSecond sentence."


def test_context_service_rejects_same_or_unsupported_languages() -> None:
    service, _, _ = _service()

    for source_lang, target_lang in (("en", "en"), ("hi", "en")):
        with pytest.raises(TranslationServiceError) as exc_info:
            service.translate_document(
                user_id=uuid.uuid4(),
                source_lang=source_lang,
                target_lang=target_lang,
                text="Hello.",
            )
        assert exc_info.value.code in {"same_language", "invalid_language"}


def test_context_endpoint_validation_and_mocked_response(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    fake_service, fake_translator, _ = _service()
    client.app.dependency_overrides[get_context_translation_service] = lambda: fake_service
    try:
        response = client.post(
            "/api/translate/document-context",
            headers=auth_headers,
            json={
                "source_lang": "en",
                "target_lang": "te",
                "text": "Ravi bought a book. He read it.",
                "context_window": 1,
            },
        )
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["model_backend"] == "nllb200"
        assert data["sentences"][1]["context_indices"] == [0]
        assert data["metadata"]["translation_uses_context"] is False
        assert fake_translator.translated_inputs == data["source_sentences"]

        invalid = client.post(
            "/api/translate/document-context",
            headers=auth_headers,
            json={
                "source_lang": "en",
                "target_lang": "te",
                "text": "Hello.",
                "context_window": 201,
            },
        )
        assert invalid.status_code == 422
    finally:
        client.app.dependency_overrides.pop(get_context_translation_service, None)


def test_context_endpoint_supports_prefix_conditioned_mode(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    fake_service, _, _ = _service()
    client.app.dependency_overrides[get_context_translation_service] = lambda: fake_service
    try:
        response = client.post(
            "/api/translate/document-context",
            headers=auth_headers,
            json={
                "source_lang": "en",
                "target_lang": "te",
                "text": "First sentence. Second sentence.",
                "context_window": 1,
                "context_mode": "prefix_conditioned",
            },
        )
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["metadata"]["context_mode"] == "prefix_conditioned"
        assert data["metadata"]["context_used_for_translation"] is True
        assert data["sentences"][1]["formatted_input"] == (
            "CONTEXT:\nFirst sentence.\n\nCURRENT SENTENCE:\nSecond sentence."
        )
    finally:
        client.app.dependency_overrides.pop(get_context_translation_service, None)


def test_context_endpoint_requires_authentication(client: TestClient) -> None:
    response = client.post(
        "/api/translate/document-context",
        json={
            "source_lang": "en",
            "target_lang": "te",
            "text": "Hello.",
        },
    )

    assert response.status_code == 401
