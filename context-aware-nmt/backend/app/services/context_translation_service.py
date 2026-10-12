"""Causal document context pipeline around the sentence-level baseline.

The context is retrieved and formatted for observability. The implemented
experimental mode can optionally prefix that context into the model input while
preserving the sentence-level NLLB backend and its pretrained checkpoint.
"""

from __future__ import annotations

import uuid
from typing import Any

from app.context import ContextRetrieverRegistry, registry as default_context_registry
from app.context.formatter import format_context_input
from app.core.config import Settings, get_settings
from app.preprocessing.sentence_splitter import SentenceSplitter
from app.schemas import (
    DocumentContextSentenceOut,
    DocumentContextTranslationResponse,
    TextTranslateIn,
)
from app.services.context_conditioned_translation_service import (
    ContextConditionedTranslationService,
)
from app.services.translation_service import TranslationService, TranslationServiceError


class ContextAwareTranslationService:
    """Construct causal recent-context windows and translate each current sentence."""

    def __init__(
        self,
        translation_service: TranslationService | None = None,
        *,
        context_registry: ContextRetrieverRegistry | None = None,
        settings: Settings | None = None,
        context_conditioned_service: ContextConditionedTranslationService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.translation_service = translation_service or TranslationService(
            settings=self.settings
        )
        self.context_registry = context_registry or default_context_registry
        self.context_registry.populate()
        self.sentence_splitter = SentenceSplitter()
        self.context_conditioned_service = (
            context_conditioned_service
            or ContextConditionedTranslationService(
                translation_service=self.translation_service,
                settings=self.settings,
            )
        )

    def translate_document(
        self,
        *,
        user_id: uuid.UUID,
        source_lang: str,
        target_lang: str,
        text: str,
        model: str = "nllb200",
        context_window: int | None = None,
        context_mode: str = "metadata_only",
        generation_config: dict[str, Any] | None = None,
    ) -> DocumentContextTranslationResponse:
        sentences = self.sentence_splitter.split(text)
        return self.translate_sentences(
            user_id=user_id,
            source_lang=source_lang,
            target_lang=target_lang,
            sentences=sentences,
            model=model,
            context_window=context_window,
            context_mode=context_mode,
            generation_config=generation_config,
        )

    def translate_sentences(
        self,
        *,
        user_id: uuid.UUID,
        source_lang: str,
        target_lang: str,
        sentences: list[str],
        model: str = "nllb200",
        context_window: int | None = None,
        context_mode: str = "metadata_only",
        generation_config: dict[str, Any] | None = None,
    ) -> DocumentContextTranslationResponse:
        if source_lang not in {"en", "te"} or target_lang not in {"en", "te"}:
            raise TranslationServiceError(
                "Document-context translation supports only 'en' and 'te'.",
                code="invalid_language",
            )
        if source_lang == target_lang:
            raise TranslationServiceError(
                "Source and target language must be different.",
                code="same_language",
            )
        if model != "nllb200":
            raise TranslationServiceError(
                "The document-context foundation currently uses the NLLB-200 baseline.",
                code="invalid_model",
            )
        if context_mode not in {"metadata_only", "prefix_conditioned"}:
            raise TranslationServiceError(
                "context_mode must be either 'metadata_only' or 'prefix_conditioned'.",
                code="invalid_context_mode",
            )
        if any(not isinstance(sentence, str) for sentence in sentences):
            raise TranslationServiceError(
                "Every document sentence must be a string.",
                code="invalid_document",
            )

        effective_window = (
            self.settings.CONTEXT_K if context_window is None else context_window
        )
        if not 0 <= effective_window <= 200:
            raise TranslationServiceError(
                "context_window must be between 0 and 200.",
                code="invalid_context_window",
            )

        retriever = self.context_registry.new("lastn")
        retriever.build_index(sentences)
        sentence_outputs: list[DocumentContextSentenceOut] = []
        translations: list[str] = []
        context_used_for_translation = context_mode == "prefix_conditioned"

        for sentence_idx, source_sentence in enumerate(sentences):
            hits = retriever.get_context(
                sentence_idx,
                k=effective_window,
                window=effective_window,
                min_score=0.0,
            )
            context_sentences = [hit.text for hit in hits]
            formatted_input = format_context_input(
                context_sentences,
                source_sentence,
            )

            if context_mode == "metadata_only":
                baseline_response = self.translation_service.translate_text(
                    user_id=user_id,
                    req=TextTranslateIn(
                        source_lang=source_lang,
                        target_lang=target_lang,
                        pipeline="baseline",
                        model=model,
                        text=source_sentence,
                    ),
                )
                if baseline_response.status != "ok" or baseline_response.translated_text is None:
                    raise TranslationServiceError(
                        baseline_response.note or "Baseline translator returned no translation.",
                        code="translation_failed",
                        status_code=502,
                    )
                translation = baseline_response.translated_text
            else:
                conditioned_result = self.context_conditioned_service.translate_sentence(
                    source_lang=source_lang,
                    target_lang=target_lang,
                    current_sentence=source_sentence,
                    preceding_context=context_sentences,
                    context_window=effective_window,
                    model=model,
                    generation_config=(
                        generation_config
                        if generation_config is not None
                        else {"context_window": effective_window}
                    ),
                )
                translation = conditioned_result["translation"]
                formatted_input = conditioned_result["formatted_input"]

            translations.append(translation)
            sentence_outputs.append(
                DocumentContextSentenceOut(
                    sentence_idx=sentence_idx,
                    source=source_sentence,
                    translation=translation,
                    context=context_sentences,
                    context_indices=[hit.idx for hit in hits],
                    formatted_input=formatted_input,
                )
            )

        return DocumentContextTranslationResponse(
            source_lang=source_lang,
            target_lang=target_lang,
            model_backend=model,
            source_sentences=list(sentences),
            translations=translations,
            sentences=sentence_outputs,
            metadata={
                "context_strategy": "lastn_nonsemantic",
                "context_mode": context_mode,
                "context_window": effective_window,
                "context_is_causal": True,
                "translation_uses_context": context_used_for_translation,
                "context_used_for_translation": context_used_for_translation,
                "model_backend": model,
                "translation_model_type": "sentence_level_pretrained_checkpoint",
                "generation_config": generation_config
                if generation_config is not None
                else {"context_window": effective_window},
            },
        )


_default_context_translation_service: ContextAwareTranslationService | None = None


def get_context_translation_service() -> ContextAwareTranslationService:
    global _default_context_translation_service
    if _default_context_translation_service is None:
        _default_context_translation_service = ContextAwareTranslationService()
    return _default_context_translation_service
