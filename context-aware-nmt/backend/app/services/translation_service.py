"""Service layer for baseline translation orchestration.

It validates requests, segments input, calls the registered sentence-level
baseline translator, and keeps the context-aware path explicitly unimplemented.
This module is FastAPI-free so it can be tested and reused from a CLI.
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings, get_settings
from app.models import InputType, JobStatus
from app.preprocessing import DocumentChunk
from app.schemas import (
    ContextSnippetOut,
    PerSentenceOut,
    TextTranslateIn,
    TranslationResponse,
)
from app.translation import (
    ModelUnavailableError,
    TranslationBackendError,
    registry as default_translator_registry,
)


@dataclass
class TranslationServiceError(Exception):
    """Raised by the service for domain-level validation failures."""

    message: str
    code: str = "validation_error"
    status_code: int = 400


# Canonical allow-lists — duplicated defensively in routers so callers get
# early feedback, but the service is the single source of truth.
ALLOWED_LANGS = {"te", "en", "hi"}
ALLOWED_PIPELINES = {"baseline", "context_aware"}
ALLOWED_MODELS = {"indictrans2", "nllb200"}

# Request validation allow-list. Actual availability is determined from the
# populated translator registry below.
_SUPPORTED_PAIRS_BY_MODEL: dict[str, set[tuple[str, str]]] = {
    "indictrans2": {("te", "en"), ("en", "te")},
    "nllb200": {("te", "en"), ("en", "te")},
}

_LANG_LABELS: dict[str, tuple[str, str | None]] = {
    "en": ("English", None),
    "te": ("Telugu", "తెలుగు"),
    "hi": ("Hindi", "हिन्दी"),
}


def get_language_metadata() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (languages, pairs) descriptions for the languages endpoint.

    Honesty rule: pairs have ``implemented: False`` if no model supports
    them yet.
    """

    languages = [
        {
            "code": code,
            "label": label,
            "native_label": native,
            "note": None,
        }
        for code, (label, native) in _LANG_LABELS.items()
    ]

    default_translator_registry.populate()
    all_pairs: list[tuple[str, str]] = [
        ("en", "te"), ("te", "en"),
    ]

    pairs: list[dict[str, Any]] = []
    for src, tgt in all_pairs:
        models_supported = [
            name
            for name in default_translator_registry.available()
            if default_translator_registry.supports(name, src, tgt)
        ]
        implemented = bool(models_supported)
        note: str | None = None
        if not implemented:
            note = "Planned — no model backend currently supports this pair."
        elif src == "hi" or tgt == "hi":
            note = (
                "Hindi is supported by the registered baseline backend but is not "
                "part of the English-Telugu project baseline scope."
            )
        pairs.append({
            "source_lang": src,
            "target_lang": tgt,
            "source_label": _LANG_LABELS[src][0],
            "target_label": _LANG_LABELS[tgt][0],
            "implemented": implemented,
            "models_supported": models_supported,
            "note": note,
        })

    return languages, pairs


def _validate_translation_options(
    *,
    source_lang: str,
    target_lang: str,
    pipeline: str,
    model: str,
) -> None:
    """Raise :class:`TranslationServiceError` if any option is invalid."""

    if source_lang == target_lang:
        raise TranslationServiceError(
            "Source and target language must be different.",
            code="same_language",
        )

    bad_langs = {source_lang, target_lang} - ALLOWED_LANGS
    if bad_langs:
        raise TranslationServiceError(
            f"Unsupported language code(s): {sorted(bad_langs)}. Allowed: {sorted(ALLOWED_LANGS)}.",
            code="invalid_language",
        )

    if pipeline not in ALLOWED_PIPELINES:
        raise TranslationServiceError(
            f"Unknown pipeline '{pipeline}'. Allowed: {sorted(ALLOWED_PIPELINES)}.",
            code="invalid_translation_mode",
        )

    if model not in ALLOWED_MODELS:
        raise TranslationServiceError(
            f"Unknown model '{model}'. Allowed: {sorted(ALLOWED_MODELS)}.",
            code="invalid_model",
        )

    implemented = _SUPPORTED_PAIRS_BY_MODEL.get(model, set())
    if (source_lang, target_lang) not in implemented:
        raise TranslationServiceError(
            f"Model '{model}' does not support the pair {source_lang}→{target_lang}.",
            code="unsupported_language_pair",
        )

    # Context retrieval is only meaningful with >=1 prior sentences. We
    # don't reject it here; callers just get 0 context hits on 1-sentence
    # inputs. But reject invalid strategy names early.
    # (Strategy-level validation lives in the schemas / ContextRetriever registry.)


class TranslationService:
    """Orchestrate baseline translation and preserve the unimplemented context path."""

    def __init__(self, settings: Settings | None = None, translator_registry=None) -> None:
        self.settings = settings or get_settings()
        self.translator_registry = translator_registry or default_translator_registry
        self.translator_registry.populate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate_text(
        self,
        user_id: uuid.UUID,
        req: TextTranslateIn,
    ) -> TranslationResponse:
        """Translate baseline text or acknowledge the unimplemented context path."""

        _validate_translation_options(
            source_lang=req.source_lang,
            target_lang=req.target_lang,
            pipeline=req.pipeline,
            model=req.model,
        )

        text = req.text
        if not text or not text.strip():
            raise TranslationServiceError(
                "Input text must not be empty.",
                code="empty_text",
            )

        sentences_out = self._split_sentences(text)
        translated_sentences: list[str] | None = None
        if req.pipeline == "baseline":
            if req.model not in self.translator_registry.available():
                raise TranslationServiceError(
                    f"The '{req.model}' translator is unavailable in this runtime. "
                    "The configured baseline backend is NLLB-200.",
                    code="model_unavailable",
                    status_code=503,
                )
            if not self.translator_registry.supports(
                req.model, req.source_lang, req.target_lang
            ):
                raise TranslationServiceError(
                    f"Model '{req.model}' does not support "
                    f"{req.source_lang}→{req.target_lang} in the baseline registry.",
                    code="unsupported_language_pair",
                )
            try:
                translator = self.translator_registry.get(req.model, auto_load=True)
                translated_sentences = translator.translate_batch(
                    sentences_out,
                    src_lang=req.source_lang,
                    tgt_lang=req.target_lang,
                )
            except ModelUnavailableError as exc:
                raise TranslationServiceError(
                    str(exc),
                    code="model_unavailable",
                    status_code=503,
                ) from exc
            except TranslationBackendError as exc:
                raise TranslationServiceError(
                    str(exc),
                    code="translation_failed",
                    status_code=502,
                ) from exc
            except KeyError as exc:
                raise TranslationServiceError(
                    f"No translator is registered for model '{req.model}'.",
                    code="model_unavailable",
                    status_code=503,
                ) from exc

            if len(translated_sentences) != len(sentences_out):
                raise TranslationServiceError(
                    "The translator returned an unexpected number of sentences.",
                    code="translation_failed",
                    status_code=502,
                )

        per_sentence: list[PerSentenceOut] = [
            PerSentenceOut(
                sentence_idx=i,
                source=s,
                target=(
                    translated_sentences[i]
                    if translated_sentences is not None
                    else None
                ),
                context=[],
            )
            for i, s in enumerate(sentences_out)
        ]

        if translated_sentences is not None:
            return TranslationResponse(
                status="ok",
                job_id=uuid.uuid4(),
                translated_text=" ".join(translated_sentences),
                tokens_used=None,
                sentences=per_sentence,
                note=(
                    "Sentence-level baseline using the NLLB-200 pretrained checkpoint; "
                    "no document context is used."
                ),
            )

        return TranslationResponse(
            status="not_implemented",
            job_id=uuid.uuid4(),
            translated_text=None,
            tokens_used=None,
            sentences=per_sentence,
            note=(
                "The context-aware pipeline is not implemented. This response "
                "contains sentence segmentation metadata only; no context or "
                "translation was fabricated."
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers — later phases wire these to registries
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Placeholder sentence splitter.

        Naively splits on ``.?!`` plus Indic danda ``।``.  The full
        SentenceSplitter class from preprocessing is wired in a later
        phase; this keeps the foundation API structurally sound.
        """

        if not text.strip():
            return []

        import re

        # Split on terminal punctuation, keep the delimiter attached.
        parts = re.split(r"(?<=[.!?।\n])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]


# Sentinel singleton — convenient when callers don't want to pass settings.
_default_service: TranslationService | None = None


def get_translation_service() -> TranslationService:
    global _default_service
    if _default_service is None:
        _default_service = TranslationService()
    return _default_service
