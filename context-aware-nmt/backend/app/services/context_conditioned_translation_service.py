"""Experimental context-conditioned translation around the existing NLLB backend.

This is intentionally not a new translation model. It is a sentence-level
pretrained checkpoint reused with a context-prefixed input string so the model
receives prior sentences in the same request as the current sentence. That is an
input-conditioning experiment, not native document-level NMT.
"""

from __future__ import annotations

from typing import Any

from app.core.config import Settings, get_settings
from app.services.translation_service import TranslationService, TranslationServiceError
from app.translation import registry as default_translator_registry


class ContextConditionedTranslationService:
    """Translate a current sentence using preceding context as a prefixed input."""

    def __init__(
        self,
        translation_service: TranslationService | None = None,
        *,
        settings: Settings | None = None,
        translator_registry=None,
        translator_backend: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.translation_service = translation_service or TranslationService(
            settings=self.settings
        )
        self.translator_registry = translator_registry or default_translator_registry
        self.translator_registry.populate()
        self.translator_backend = translator_backend

    def build_model_input(
        self,
        *,
        current_sentence: str,
        preceding_context: list[str],
    ) -> str:
        """Return the exact deterministic context-conditioned text sent to NLLB."""
        context_block = "\n".join(preceding_context)
        if context_block:
            return f"CONTEXT:\n{context_block}\n\nCURRENT SENTENCE:\n{current_sentence}"
        return f"CONTEXT:\n\nCURRENT SENTENCE:\n{current_sentence}"

    def translate_sentence(
        self,
        *,
        source_lang: str,
        target_lang: str,
        current_sentence: str,
        preceding_context: list[str],
        context_window: int | None = None,
        model: str = "nllb200",
        generation_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Translate a sentence with a context-prefixed model input."""
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
                "The context-conditioned experiment currently uses the NLLB-200 backend.",
                code="invalid_model",
            )
        if not isinstance(current_sentence, str) or not current_sentence.strip():
            raise TranslationServiceError(
                "The current sentence must be a non-empty string.",
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

        context_for_model = list(preceding_context[:effective_window]) if effective_window else []
        model_input = self.build_model_input(
            current_sentence=current_sentence,
            preceding_context=context_for_model,
        )

        if self.translator_backend is None and model not in self.translator_registry.available():
            raise TranslationServiceError(
                f"The '{model}' translator is unavailable in this runtime.",
                code="model_unavailable",
                status_code=503,
            )

        try:
            translator = self.translator_backend
            if translator is None:
                translator = self.translator_registry.get(model, auto_load=True)
            translated = translator.translate_batch(
                [model_input],
                src_lang=source_lang,
                tgt_lang=target_lang,
            )
        except TranslationServiceError:
            raise
        except Exception as exc:  # pragma: no cover - backend-specific failure path
            raise TranslationServiceError(
                f"The context-conditioned translation backend failed for {source_lang}->{target_lang}.",
                code="translation_failed",
                status_code=502,
            ) from exc

        if not translated or len(translated) != 1:
            raise TranslationServiceError(
                "The context-conditioned backend returned an unexpected output shape.",
                code="translation_failed",
                status_code=502,
            )

        return {
            "translation": translated[0],
            "formatted_input": model_input,
            "context": context_for_model,
            "context_window": effective_window,
            "model_backend": model,
            "generation_config": generation_config,
            "translation_model_type": "sentence_level_pretrained_checkpoint",
        }
