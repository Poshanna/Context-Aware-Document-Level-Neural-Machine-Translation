"""SQLAlchemy ORM models for the application.

These models define the *shape* of each table.  No queries live here —
queries go into ``services`` or a later ``repositories`` layer once the
query volume justifies it.
"""

from __future__ import annotations

import datetime
import enum
import uuid

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, relationship  # noqa: F401


class Base(DeclarativeBase):
    """Shared declarative base.  Add shared mixins here later if needed."""


# ======================================================================
# Shared enum types
# ======================================================================

class Pipeline(str, enum.Enum):
    BASELINE = "baseline"
    CONTEXT_AWARE = "context_aware"


class ModelName(str, enum.Enum):
    INDICTRANS2 = "indictrans2"
    NLLB200 = "nllb200"


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class InputType(str, enum.Enum):
    TEXT = "text"
    FILE = "file"


class FileType(str, enum.Enum):
    SOURCE = "source"
    TRANSLATED = "translated"


class MetricName(str, enum.Enum):
    BLEU = "bleu"
    CHRF = "chrf"
    BERT_P = "bertscore_p"
    BERT_R = "bertscore_r"
    BERT_F1 = "bertscore_f1"
    COMET = "comet"


# ======================================================================
# Users
# ======================================================================

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = Column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = Column(String(320), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = Column(String(255), nullable=False)
    full_name: Mapped[str | None] = Column(String(255), nullable=True)
    created_at: Mapped[datetime.datetime] = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# ======================================================================
# Translation jobs & per-sentence details
# ======================================================================

class TranslationJob(Base):
    __tablename__ = "translation_jobs"

    id: Mapped[uuid.UUID] = Column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = Column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    pipeline: Mapped[Pipeline] = Column(Enum(Pipeline), nullable=False)
    model_name: Mapped[ModelName] = Column(Enum(ModelName), nullable=False)
    source_lang: Mapped[str] = Column(String(10), nullable=False)
    target_lang: Mapped[str] = Column(String(10), nullable=False)
    input_type: Mapped[InputType] = Column(Enum(InputType), nullable=False)
    status: Mapped[JobStatus] = Column(Enum(JobStatus), nullable=False, default=JobStatus.QUEUED)
    progress_percent: Mapped[int] = Column(SmallInteger, nullable=False, default=0)
    error_message: Mapped[str | None] = Column(Text, nullable=True)
    source_text_preview: Mapped[str | None] = Column(String(500), nullable=True)
    total_sentences: Mapped[int | None] = Column(Integer, nullable=True)
    created_at: Mapped[datetime.datetime] = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime.datetime | None] = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_jobs_user_created", "user_id", "created_at"),
    )


class TranslationSentence(Base):
    __tablename__ = "translation_sentences"

    id: Mapped[int] = Column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = Column(
        Uuid, ForeignKey("translation_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sentence_idx: Mapped[int] = Column(Integer, nullable=False)
    source_text: Mapped[str] = Column(Text, nullable=False)
    target_text: Mapped[str | None] = Column(Text, nullable=True)
    processing_meta: Mapped[dict | None] = Column(JSON, nullable=True)  # page, paragraph_id, …

    __table_args__ = (
        UniqueConstraint("job_id", "sentence_idx", name="uq_job_sentence"),
    )


class ContextUsage(Base):
    """Records which source-sentence contexts were used for each translated sentence."""

    __tablename__ = "context_usages"

    id: Mapped[int] = Column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = Column(
        Uuid, ForeignKey("translation_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_sentence_idx: Mapped[int] = Column(Integer, nullable=False)
    source_sentence_idx: Mapped[int] = Column(Integer, nullable=False)
    score: Mapped[float] = Column(Float, nullable=False)
    strategy_used: Mapped[str] = Column(String(20), nullable=False)  # sbert / lastn

    __table_args__ = (
        Index("ix_context_per_target", "job_id", "target_sentence_idx"),
    )


# ======================================================================
# Uploaded / produced files
# ======================================================================

class DocumentFile(Base):
    __tablename__ = "document_files"

    id: Mapped[uuid.UUID] = Column(Uuid, primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID | None] = Column(
        Uuid, ForeignKey("translation_jobs.id", ondelete="SET NULL"), nullable=True
    )
    user_id: Mapped[uuid.UUID] = Column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    file_type: Mapped[FileType] = Column(Enum(FileType), nullable=False)
    original_name: Mapped[str] = Column(Text, nullable=False)
    stored_path: Mapped[str] = Column(Text, nullable=False)  # on-disk inside volume
    mime_type: Mapped[str] = Column(String(100), nullable=False)
    size_bytes: Mapped[int] = Column(BigInteger, nullable=False)
    sha256: Mapped[str] = Column(String(64), nullable=False)
    created_at: Mapped[datetime.datetime] = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ======================================================================
# Evaluation runs
# ======================================================================

class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[uuid.UUID] = Column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = Column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    model_name: Mapped[str] = Column(String(50), nullable=False)
    pipelines_evaluated: Mapped[list] = Column(JSON, nullable=False)  # ["baseline","context_aware"]
    dataset_name: Mapped[str | None] = Column(String(200), nullable=True)
    num_sentences: Mapped[int | None] = Column(Integer, nullable=True)
    status: Mapped[JobStatus] = Column(Enum(JobStatus), nullable=False, default=JobStatus.QUEUED)
    config_snapshot: Mapped[dict | None] = Column(JSON, nullable=True)  # seed, k, strategy, …
    error_message: Mapped[str | None] = Column(Text, nullable=True)
    started_at: Mapped[datetime.datetime | None] = Column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime.datetime | None] = Column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime.datetime] = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[int] = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = Column(
        Uuid, ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    pipeline: Mapped[str] = Column(String(20), nullable=False)
    metric_name: Mapped[MetricName] = Column(Enum(MetricName), nullable=False)
    aggregate_value: Mapped[float] = Column(Float, nullable=False)
    per_sentence_values: Mapped[list | None] = Column(JSON, nullable=True)

    __table_args__ = (
        UniqueConstraint("run_id", "pipeline", "metric_name", name="uq_run_pipeline_metric"),
    )
