"""History list, detail, and replay endpoints.

Foundation phase:
  * History is backed by an in-memory ``HistoryService`` store (no DB).
  * Endpoints return empty pages / 404s when no history exists — we
    never fabricate entries.
  * Each successful translate text / document call already stores an
    entry in this service, so round-trips are testable.
"""

from __future__ import annotations

import structlog
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_current_user
from app.schemas import Envelope, HistoryResponse, HistoryItemResponse
from app.services.history_service import (
    HistoryNotFound,
    HistoryService,
    get_history_service,
)

logger = structlog.get_logger()


router = APIRouter()


@router.get(
    "",
    response_model=Envelope[HistoryResponse],
    summary="List translation history for the authenticated user",
)
async def list_history(
    user: Annotated[object, Depends(get_current_user)],
    history_svc: Annotated[HistoryService, Depends(get_history_service)],
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
    q: str | None = Query(default=None, description="Optional keyword filter on preview/pipeline/model/lang."),
) -> Envelope[HistoryResponse]:
    user_id = getattr(user, "id")
    logger.debug(
        "history.list_requested",
        user_id=str(user_id),
        page=page,
        size=size,
        has_query=bool(q),
    )
    page_out = history_svc.list_history(
        user_id=user_id,
        page=page,
        size=size,
        query=q,
    )
    return Envelope(data=page_out)


@router.get(
    "/{job_id}",
    response_model=Envelope[HistoryItemResponse],
    summary="Get a single history entry by its job id",
)
async def history_detail(
    job_id: str,
    user: Annotated[object, Depends(get_current_user)],
    history_svc: Annotated[HistoryService, Depends(get_history_service)],
) -> Envelope[HistoryItemResponse]:
    user_id = getattr(user, "id")
    logger.debug(
        "history.detail_requested",
        user_id=str(user_id),
        job_id=job_id,
    )
    try:
        item = history_svc.get_history_item(user_id=user_id, job_id=job_id)
    except HistoryNotFound as exc:
        logger.info(
            "history.not_found",
            user_id=str(user_id),
            job_id=job_id,
        )
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": str(exc)},
        ) from exc
    return Envelope(data=item)
