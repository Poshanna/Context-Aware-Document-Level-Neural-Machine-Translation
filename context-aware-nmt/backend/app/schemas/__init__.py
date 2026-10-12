"""Pydantic request/response schemas.

Only validating, no logic.  Response models are deliberately explicit
(no ``model_`` named fields leaking ORM internals).
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ======================================================================
# Generic envelope — keeps frontend error handling simple
# ======================================================================

T = TypeVar('T')


class Envelope(BaseModel, Generic[T]):
    data: T | None = None
    error: dict[str, Any] | None = None


# ======================================================================
# Auth
# ======================================================================

class UserRegisterIn(BaseModel):
    email: EmailStr = Field(..., max_length=320)
    password: str = Field(..., min_length=8, max_length=256)
    full_name: str | None = Field(default=None, max_length=255)


class UserLoginIn(BaseModel):
    email: EmailStr = Field(..., max_length=320)
    password: str = Field(..., max_length=256)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    email: EmailStr
    full_name: str | None
    created_at: datetime.datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class AuthMeOut(UserOut):
    pass


# ======================================================================
# Shared enumerations / options
# ======================================================================

class TranslateOptions(BaseModel):
    strategy: str = Field("sbert", pattern="^(sbert|lastn)$", description="Context retrieval strategy.")
    k: int = Field(3, ge=0, le=20, description="How many prior sentences to retrieve.")
    window: int = Field(20, ge=1, le=200, description="Size of candidate window (no look-ahead).")
    min_score: float = Field(0.2, ge=0.0, le=1.0, description="Minimum similarity to keep a context.")


class ContextSnippetOut(BaseModel):
    source_sent_idx: int
    score: float
    snippet: str


class PerSentenceOut(BaseModel):
    sentence_idx: int
    source: str
    target: str | None = None
    context: list[ContextSnippetOut] = Field(default_factory=list)


# ======================================================================
# Text translation
# ======================================================================

class TextTranslateIn(BaseModel):
    source_lang: str = Field(..., pattern="^(te|en|hi)$", max_length=5)
    target_lang: str = Field(..., pattern="^(te|en|hi)$", max_length=5)
    pipeline: str = Field(..., pattern="^(baseline|context_aware)$")
    model: str = Field(..., pattern="^(indictrans2|nllb200)$")
    text: str = Field(..., min_length=1, max_length=50_000)
    context_opts: TranslateOptions | None = None


class DocumentContextTranslateIn(BaseModel):
    source_lang: str = Field(..., pattern="^(en|te)$", max_length=5)
    target_lang: str = Field(..., pattern="^(en|te)$", max_length=5)
    text: str = Field(..., max_length=50_000)
    context_window: int | None = Field(default=None, ge=0, le=200)
    context_mode: str = Field(
        default="metadata_only",
        pattern="^(metadata_only|prefix_conditioned)$",
    )
    model: str = Field(default="nllb200", pattern="^(nllb200)$")


class DocumentContextSentenceOut(BaseModel):
    sentence_idx: int
    source: str
    translation: str
    context: list[str] = Field(default_factory=list)
    context_indices: list[int] = Field(default_factory=list)
    formatted_input: str


class DocumentContextTranslationResponse(BaseModel):
    status: str = Field(default="ok", pattern="^(ok)$")
    source_lang: str
    target_lang: str
    model_backend: str
    source_sentences: list[str] = Field(default_factory=list)
    translations: list[str] = Field(default_factory=list)
    sentences: list[DocumentContextSentenceOut] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextTranslateOut(BaseModel):
    job_id: uuid.UUID | None = None
    translated_text: str
    tokens_used: int | None = None
    sentences: list[PerSentenceOut] = Field(default_factory=list)


# ======================================================================
# Document translation / jobs
# ======================================================================

class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    pipeline: str
    model_name: str
    source_lang: str
    target_lang: str
    input_type: str
    status: str
    progress_percent: int
    error_message: str | None
    total_sentences: int | None
    source_text_preview: str | None
    created_at: datetime.datetime
    completed_at: datetime.datetime | None


class JobDetailOut(JobOut):
    sentences: list[PerSentenceOut] = Field(default_factory=list)


class ReplayJobIn(BaseModel):
    pipeline: str | None = Field(default=None, pattern="^(baseline|context_aware)$")
    model: str | None = Field(default=None, pattern="^(indictrans2|nllb200)$")
    target_lang: str | None = Field(default=None, pattern="^(te|en|hi)$")


# ======================================================================
# History
# ======================================================================

class HistoryPageOut(BaseModel):
    items: list[JobOut]
    page: int
    size: int
    total: int


# ======================================================================
# Evaluation
# ======================================================================

class MetricValueOut(BaseModel):
    aggregate: float
    per_sentence: list[float] | None = None


class PipelineMetricsOut(BaseModel):
    pipeline: str
    metrics: dict[str, MetricValueOut]  # metric_name -> values


class EvaluationRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    model_name: str
    pipelines_evaluated: list[str]
    dataset_name: str | None
    num_sentences: int | None
    status: str
    error_message: str | None
    created_at: datetime.datetime
    completed_at: datetime.datetime | None


class EvaluationSummaryOut(EvaluationRunOut):
    per_pipeline: dict[str, dict[str, float]]  # pipeline -> metric -> aggregate
    compare: list[dict[str, Any]] = Field(default_factory=list)


class CompareSentenceRow(BaseModel):
    idx: int
    source: str
    reference: str | None = None
    hyp_baseline: str | None = None
    hyp_context: str | None = None
    deltas: dict[str, float] | None = None


# ======================================================================
# Foundation Phase: additional named schemas for clarity
# ======================================================================


TranslationRequest = TextTranslateIn


class TranslationResponse(BaseModel):
    status: str = Field(
        ...,
        description="Processing status. 'not_implemented' means the ML backend is not wired yet.",
        pattern="^(ok|not_implemented|error|queued|running|completed|failed)$",
    )
    job_id: uuid.UUID | None = None
    translated_text: str | None = None
    tokens_used: int | None = None
    sentences: list[PerSentenceOut] = Field(default_factory=list)
    note: str | None = Field(
        default=None,
        description="Human-readable clarification, e.g. explaining 'not_implemented'.",
    )


class DocumentTranslationRequest(BaseModel):
    source_lang: str = Field(..., pattern="^(te|en|hi)$", max_length=5)
    target_lang: str = Field(..., pattern="^(te|en|hi)$", max_length=5)
    pipeline: str = Field(..., pattern="^(baseline|context_aware)$")
    model: str = Field(..., pattern="^(indictrans2|nllb200)$")
    filename: str = Field(..., max_length=512)
    content_type: str | None = None
    size_bytes: int = Field(..., ge=0)


class TranslationJobResponse(BaseModel):
    status: str = Field(
        ...,
        pattern="^(queued|validated|not_implemented|failed)$",
    )
    job_id: uuid.UUID
    source_lang: str
    target_lang: str
    pipeline: str
    model_name: str
    input_type: str = "file"
    file_info: dict[str, Any] = Field(default_factory=dict)
    progress_percent: int = 0
    created_at: datetime.datetime
    note: str | None = None


class HistoryItemResponse(BaseModel):
    job_id: uuid.UUID
    pipeline: str
    model_name: str
    source_lang: str
    target_lang: str
    input_type: str
    status: str
    progress_percent: int
    source_text_preview: str | None
    created_at: datetime.datetime
    completed_at: datetime.datetime | None


class HistoryResponse(BaseModel):
    items: list[HistoryItemResponse]
    page: int
    size: int
    total: int


class LanguagePair(BaseModel):
    source_lang: str
    target_lang: str
    source_label: str
    target_label: str
    implemented: bool
    models_supported: list[str] = Field(default_factory=list)
    note: str | None = None


class LanguageOut(BaseModel):
    code: str
    label: str
    native_label: str | None = None
    note: str | None = None


class LanguageResponse(BaseModel):
    languages: list[LanguageOut]
    pairs: list[LanguagePair]
