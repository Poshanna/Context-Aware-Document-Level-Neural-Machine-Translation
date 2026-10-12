"""Service layer for document translation orchestration.

Foundation phase:
  * Validates the uploaded file (extension, max size).
  * Computes lightweight metadata (filename, MIME, size, SHA-256).
  * Attempts to instantiate the correct DocumentProcessor from the
    registry and reports whether chunks could be extracted yet.
  * Returns a job-like response with ``status = "validated"`` (or
    ``"not_implemented"`` when the processor itself raises).

Services deliberately do NOT import FastAPI. File bytes are received as
a plain ``bytes`` buffer so the service works equally well from CLI,
worker, or tests.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import uuid
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings, get_settings
from app.preprocessing import DocumentProcessorRegistry
from app.schemas import (
    DocumentTranslationRequest,
    TranslationJobResponse,
)
from app.services.translation_service import (
    TranslationServiceError,
    _validate_translation_options,  # noqa: F401  (re-exported for routers)
)


class DocumentServiceError(Exception):
    """Raised by :class:`DocumentService` for invalid inputs / IO failures."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "document_error",
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


ALLOWED_EXTENSIONS = {"txt", "pdf", "docx"}
MIME_BY_EXT: dict[str, str] = {
    "txt": "text/plain",
    "pdf": "application/pdf",
    "docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
}


def _guess_extension(filename: str) -> str | None:
    if not filename or "." not in filename:
        return None
    return filename.rsplit(".", 1)[-1].lower()


class DocumentService:
    """Validates and (in later phases) extracts + reassembles documents."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_document(
        self,
        *,
        user_id: uuid.UUID,
        source_lang: str,
        target_lang: str,
        pipeline: str,
        model: str,
        filename: str,
        content_type: str | None,
        file_bytes: bytes,
    ) -> TranslationJobResponse:
        """Validate and acknowledge a document upload.

        Foundation phase: returns a job with ``status = "validated"`` or
        ``status = "not_implemented"`` depending on whether the
        DocumentProcessor is available. **No translation is performed.**
        """

        # --- Reuse the translation-level option validator ---
        _validate_translation_options(
            source_lang=source_lang,
            target_lang=target_lang,
            pipeline=pipeline,
            model=model,
        )

        # --- File validation ---
        ext = _guess_extension(filename)
        if not ext:
            raise DocumentServiceError(
                "Could not determine file extension from filename.",
                code="missing_extension",
            )
        if ext not in ALLOWED_EXTENSIONS:
            raise DocumentServiceError(
                f"Unsupported file type '.{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}.",
                code="unsupported_file_type",
            )

        max_bytes = max(1, int(self.settings.MAX_UPLOAD_SIZE_MB)) * 1024 * 1024
        size_bytes = len(file_bytes)
        if size_bytes == 0:
            raise DocumentServiceError(
                "Uploaded file is empty.",
                code="missing_file",
            )
        if size_bytes > max_bytes:
            raise DocumentServiceError(
                f"File too large: {size_bytes} bytes exceeds limit of {max_bytes} bytes "
                f"({self.settings.MAX_UPLOAD_SIZE_MB} MB).",
                code="oversized_file",
                status_code=413,
            )

        sha256 = hashlib.sha256(file_bytes).hexdigest()

        # --- Attempt processor lookup ---
        # The DocumentProcessor registry.populate() may be empty in this
        # phase, so catch KeyError and degrade to status=not_implemented.
        processor_status = "not_available"
        num_chunks: int | None = None
        try:
            registry = DocumentProcessorRegistry()
            registry.populate()
            processor = registry.for_extension(ext)
            # Attempt extract_chunks; if the processor raises NI, catch it.
            from pathlib import Path

            import tempfile

            with tempfile.NamedTemporaryFile(
                suffix=f".{ext}", delete=False
            ) as tmp:
                tmp.write(file_bytes)
                tmp_path = Path(tmp.name)
            try:
                chunks = processor.extract_chunks(tmp_path)
                num_chunks = len(chunks)
                processor_status = "ok"
            except NotImplementedError:
                processor_status = "not_implemented"
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
        except KeyError:
            processor_status = "not_registered"

        if processor_status == "ok":
            status = "validated"
        else:
            status = "not_implemented"

        note = (
            "Document validated; full chunk extraction + translation "
            "pipeline implemented in later phases."
        )
        if processor_status == "not_implemented":
            note = (
                "File is valid, but the DocumentProcessor for this format is not "
                "implemented yet (foundation phase)."
            )
        elif processor_status == "not_registered":
            note = (
                "File is valid, but no DocumentProcessor is registered for this "
                "extension yet."
            )

        return TranslationJobResponse(
            status=status,
            job_id=uuid.uuid4(),
            source_lang=source_lang,
            target_lang=target_lang,
            pipeline=pipeline,
            model_name=model,
            input_type="file",
            file_info={
                "filename": filename,
                "extension": ext,
                "content_type": content_type or MIME_BY_EXT.get(ext),
                "size_bytes": size_bytes,
                "sha256": sha256,
                "processor_status": processor_status,
                "estimated_chunks": num_chunks,
            },
            progress_percent=0,
            created_at=datetime.datetime.now(datetime.timezone.utc),
            note=note,
        )


_default_document_service: DocumentService | None = None


def get_document_service() -> DocumentService:
    global _default_document_service
    if _default_document_service is None:
        _default_document_service = DocumentService()
    return _default_document_service
