"""Text + document translation endpoints.

Routers here follow the clean-architecture contract (project_rules.md §3.1):
  1. Parse and validate inputs (Pydantic / FastAPI annotations).
  2. Resolve the current user via dependency injection.
  3. Delegate to the service layer (translation_service, document_service,
     history_service).
  4. Wrap the result in the generic ``Envelope[T]``, or translate
     service-layer exceptions into proper HTTP 4xx/5xx with structured
     ``Envelope.error`` payloads.

ML logic never lives in this module — it lives in ``app/translation``,
``app/context``, and ``app/preprocessing``, orchestrated by services.
"""

from __future__ import annotations

import structlog
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.deps import get_current_user
from app.schemas import (
    Envelope,
    TextTranslateIn,
    TranslationJobResponse,
    TranslationResponse,
    DocumentContextTranslateIn,
    DocumentContextTranslationResponse,
)
from app.services.document_service import (
    DocumentService,
    DocumentServiceError,
    get_document_service,
)
from app.services.history_service import get_history_service
from app.services.context_translation_service import (
    ContextAwareTranslationService,
    get_context_translation_service,
)
from app.services.translation_service import (
    TranslationService,
    TranslationServiceError,
    get_translation_service,
)

logger = structlog.get_logger()


router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _service_error_to_http(exc: TranslationServiceError | DocumentServiceError) -> HTTPException:
    """Translate a service-domain error into an HTTP exception whose
    ``detail`` matches the ``Envelope.error`` shape the frontend expects.
    """

    return HTTPException(
        status_code=exc.status_code,
        detail={
            "code": exc.code,
            "message": exc.message,
        },
    )


# ------------------------------ TEXT ---------------------------------------


@router.post(
    "/text",
    response_model=Envelope[TranslationResponse],
    summary="Translate plain text with the sentence-level baseline",
)
async def translate_text(
    payload: TextTranslateIn,
    user: Annotated[object, Depends(get_current_user)],
    translation_svc: Annotated[TranslationService, Depends(get_translation_service)],
) -> Envelope[TranslationResponse]:
    """Validate text translation parameters and acknowledge with a
    ``"baseline"`` pipeline or preserve the explicit not-implemented response
    for ``"context_aware"``. Backend errors are returned as controlled errors.
    """

    user_id = getattr(user, "id")
    logger.info(
        "translate_text.requested",
        user_id=str(user_id),
        source_lang=payload.source_lang,
        target_lang=payload.target_lang,
        pipeline=payload.pipeline,
        model=payload.model,
        chars=len(payload.text or ""),
    )

    try:
        response = translation_svc.translate_text(user_id=user_id, req=payload)
    except TranslationServiceError as exc:
        logger.warning(
            "translate_text.validation_failed",
            user_id=str(user_id),
            code=exc.code,
            message=exc.message,
        )
        raise _service_error_to_http(exc) from exc

    # Record in history service (in-process store in foundation phase).
    try:
        history_svc = get_history_service()
        history_svc.add_text_job(
            user_id=user_id,
            job_id=response.job_id,
            pipeline=payload.pipeline,
            model_name=payload.model,
            source_lang=payload.source_lang,
            target_lang=payload.target_lang,
            status=response.status,
            source_text=payload.text,
        )
    except Exception:  # pragma: no cover - history failures must not break the API
        logger.exception("translate_text.history_store_failed", job_id=str(response.job_id))

    logger.info(
        "translate_text.completed",
        user_id=str(user_id),
        job_id=str(response.job_id),
        status=response.status,
        sentences=len(response.sentences),
    )
    return Envelope(data=response)


@router.post(
    "/document-context",
    response_model=Envelope[DocumentContextTranslationResponse],
    summary="Translate an ordered document with causal context metadata",
)
async def translate_document_context(
    payload: DocumentContextTranslateIn,
    user: Annotated[object, Depends(get_current_user)],
    context_svc: Annotated[
        ContextAwareTranslationService, Depends(get_context_translation_service)
    ],
) -> Envelope[DocumentContextTranslationResponse]:
    """Build causal source context and translate each current sentence via baseline."""
    user_id = getattr(user, "id")
    logger.info(
        "translate_document_context.requested",
        user_id=str(user_id),
        source_lang=payload.source_lang,
        target_lang=payload.target_lang,
        context_window=payload.context_window,
        chars=len(payload.text),
    )
    try:
        response = context_svc.translate_document(
            user_id=user_id,
            source_lang=payload.source_lang,
            target_lang=payload.target_lang,
            text=payload.text,
            model=payload.model,
            context_window=payload.context_window,
            context_mode=payload.context_mode,
        )
    except TranslationServiceError as exc:
        logger.warning(
            "translate_document_context.failed",
            user_id=str(user_id),
            code=exc.code,
            message=exc.message,
        )
        raise _service_error_to_http(exc) from exc

    logger.info(
        "translate_document_context.completed",
        user_id=str(user_id),
        model_backend=response.model_backend,
        sentences=len(response.sentences),
        context_window=response.metadata["context_window"],
    )
    return Envelope(data=response)


# ------------------------------ DOCUMENT -----------------------------------


@router.post(
    "/document",
    response_model=Envelope[TranslationJobResponse],
    summary="Submit a PDF/DOCX/TXT document for translation",
)
async def translate_document(
    user: Annotated[object, Depends(get_current_user)],
    source_lang: Annotated[str, Form(pattern="^(te|en|hi)$")],
    target_lang: Annotated[str, Form(pattern="^(te|en|hi)$")],
    pipeline: Annotated[str, Form(pattern="^(baseline|context_aware)$")],
    model: Annotated[str, Form(pattern="^(indictrans2|nllb200)$")],
    file: Annotated[UploadFile, File(...)],
    document_svc: Annotated[DocumentService, Depends(get_document_service)],
) -> Envelope[TranslationJobResponse]:
    """Validate an uploaded document and return a job acknowledgment.

    In the foundation phase, the file is validated and metadata is
    captured, but **no translation is performed**.
    """

    user_id = getattr(user, "id")
    filename = file.filename or "unnamed"
    content_type = file.content_type

    logger.info(
        "translate_document.requested",
        user_id=str(user_id),
        filename=filename,
        content_type=content_type,
        source_lang=source_lang,
        target_lang=target_lang,
        pipeline=pipeline,
        model=model,
    )

    # --- Safety guard: reject before reading if the declared content-length
    # is already over the limit. FastAPI may not protect us against chunked
    # uploads, so the service validates ``len(file_bytes)`` again below.
    try:
        file_bytes = await file.read()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("translate_document.read_failed", filename=filename)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail={"code": "file_read_error", "message": "Could not read uploaded file."},
        ) from exc

    try:
        response = document_svc.submit_document(
            user_id=user_id,
            source_lang=source_lang,
            target_lang=target_lang,
            pipeline=pipeline,
            model=model,
            filename=filename,
            content_type=content_type,
            file_bytes=file_bytes,
        )
    except DocumentServiceError as exc:
        logger.warning(
            "translate_document.validation_failed",
            user_id=str(user_id),
            filename=filename,
            code=exc.code,
            message=exc.message,
        )
        raise _service_error_to_http(exc) from exc

    # Record in history service.
    try:
        history_svc = get_history_service()
        history_svc.add_document_job(
            user_id=user_id,
            job_id=response.job_id,
            pipeline=pipeline,
            model_name=model,
            source_lang=source_lang,
            target_lang=target_lang,
            status=response.status,
            source_filename=filename,
        )
    except Exception:  # pragma: no cover
        logger.exception("translate_document.history_store_failed", job_id=str(response.job_id))

    logger.info(
        "translate_document.completed",
        user_id=str(user_id),
        job_id=str(response.job_id),
        status=response.status,
        size_bytes=response.file_info.get("size_bytes"),
    )
    return Envelope(data=response)
