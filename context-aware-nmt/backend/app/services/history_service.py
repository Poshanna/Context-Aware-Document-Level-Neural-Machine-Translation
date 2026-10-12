"""Service layer for translation history.

In the foundation phase we do **not** yet have a working DB session, so the
service operates against an in-memory, per-process store.  The service
surface (methods + signatures) matches what Phase 2 (DB) will implement,
so that routers and tests are stable across the boundary.

The service deliberately never fabricates entries. When nothing has been
stored, list_history returns an empty page, and get_history_item raises
``HistoryNotFound``.
"""

from __future__ import annotations

import datetime
import threading
import uuid
from dataclasses import dataclass, field

from app.schemas import HistoryItemResponse, HistoryResponse


class HistoryNotFound(Exception):
    """Raised when a requested history id does not exist."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"History item '{job_id}' not found.")
        self.job_id = job_id


@dataclass
class _HistoryRecord:
    job_id: uuid.UUID
    user_id: uuid.UUID
    pipeline: str
    model_name: str
    source_lang: str
    target_lang: str
    input_type: str
    status: str
    progress_percent: int = 0
    source_text_preview: str | None = None
    sentences: list[dict] = field(default_factory=list)
    created_at: datetime.datetime | None = None
    completed_at: datetime.datetime | None = None

    def to_response(self) -> HistoryItemResponse:
        now = datetime.datetime.now(datetime.timezone.utc)
        return HistoryItemResponse(
            job_id=self.job_id,
            pipeline=self.pipeline,
            model_name=self.model_name,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            input_type=self.input_type,
            status=self.status,
            progress_percent=self.progress_percent,
            source_text_preview=self.source_text_preview,
            created_at=self.created_at or now,
            completed_at=self.completed_at,
        )


class HistoryService:
    """Stores and retrieves user translation history."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._store: dict[uuid.UUID, _HistoryRecord] = {}

    # ------------------------------------------------------------------
    # Write helpers (called by translation/document services in later phases)
    # ------------------------------------------------------------------

    def add_text_job(
        self,
        *,
        user_id: uuid.UUID,
        job_id: uuid.UUID,
        pipeline: str,
        model_name: str,
        source_lang: str,
        target_lang: str,
        status: str,
        source_text: str,
    ) -> HistoryItemResponse:
        preview = (source_text or "")[:500]
        with self._lock:
            rec = _HistoryRecord(
                job_id=job_id,
                user_id=user_id,
                pipeline=pipeline,
                model_name=model_name,
                source_lang=source_lang,
                target_lang=target_lang,
                input_type="text",
                status=status,
                progress_percent=100 if status in {"completed", "not_implemented", "error"} else 0,
                source_text_preview=preview or None,
                created_at=datetime.datetime.now(datetime.timezone.utc),
                completed_at=datetime.datetime.now(datetime.timezone.utc)
                if status in {"completed", "not_implemented", "error"}
                else None,
            )
            self._store[job_id] = rec
            return rec.to_response()

    def add_document_job(
        self,
        *,
        user_id: uuid.UUID,
        job_id: uuid.UUID,
        pipeline: str,
        model_name: str,
        source_lang: str,
        target_lang: str,
        status: str,
        source_filename: str,
    ) -> HistoryItemResponse:
        preview = (source_filename or "")[:500]
        with self._lock:
            rec = _HistoryRecord(
                job_id=job_id,
                user_id=user_id,
                pipeline=pipeline,
                model_name=model_name,
                source_lang=source_lang,
                target_lang=target_lang,
                input_type="file",
                status=status,
                progress_percent=100 if status in {"validated", "not_implemented", "error"} else 0,
                source_text_preview=preview or None,
                created_at=datetime.datetime.now(datetime.timezone.utc),
                completed_at=datetime.datetime.now(datetime.timezone.utc)
                if status in {"validated", "not_implemented", "error"}
                else None,
            )
            self._store[job_id] = rec
            return rec.to_response()

    # ------------------------------------------------------------------
    # Read API used by routers
    # ------------------------------------------------------------------

    def list_history(
        self,
        *,
        user_id: uuid.UUID,
        page: int = 1,
        size: int = 20,
        query: str | None = None,
    ) -> HistoryResponse:
        page = max(1, int(page))
        size = max(1, min(200, int(size)))

        with self._lock:
            items = [r for r in self._store.values() if r.user_id == user_id]
            if query:
                q = query.strip().lower()
                items = [
                    r for r in items
                    if (r.source_text_preview and q in r.source_text_preview.lower())
                    or q in r.pipeline.lower()
                    or q in r.model_name.lower()
                    or q in r.source_lang.lower()
                    or q in r.target_lang.lower()
                ]

            items.sort(
                key=lambda r: r.created_at or datetime.datetime.min.replace(
                    tzinfo=datetime.timezone.utc
                ),
                reverse=True,
            )

            total = len(items)
            start = (page - 1) * size
            page_items = [r.to_response() for r in items[start:start + size]]

        return HistoryResponse(
            items=page_items,
            page=page,
            size=size,
            total=total,
        )

    def get_history_item(
        self,
        *,
        user_id: uuid.UUID,
        job_id: str,
    ) -> HistoryItemResponse:
        try:
            job_uuid = uuid.UUID(str(job_id))
        except (ValueError, AttributeError) as exc:
            raise HistoryNotFound(job_id) from exc

        with self._lock:
            rec = self._store.get(job_uuid)
            if rec is None or rec.user_id != user_id:
                raise HistoryNotFound(job_id)
            return rec.to_response()


_default_history_service: HistoryService | None = None


def get_history_service() -> HistoryService:
    global _default_history_service
    if _default_history_service is None:
        _default_history_service = HistoryService()
    return _default_history_service
