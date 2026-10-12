"""Job status / streaming / download endpoints.

Skeleton: returns a placeholder not-implemented payload so the frontend
has URLs to target.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_user
from app.schemas import Envelope


router = APIRouter()


@router.get("/{job_id}", response_model=Envelope[dict])
async def get_job_status(
    job_id: str,
    _user: Annotated[object, Depends(get_current_user)],
) -> Envelope[dict]:
    if len(job_id) != 36:  # UUID shape
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid job id.")
    return Envelope(
        data={
            "id": job_id,
            "status": "queued",
            "progress_percent": 0,
            "message": "Skeleton only — job runner implemented in a later phase.",
        }
    )


@router.get("/{job_id}/download")
async def download_job_output(
    job_id: str,
    _user: Annotated[object, Depends(get_current_user)],
    format: str = "txt",  # noqa: A002  (matches OpenAPI param name)
) -> Envelope[dict]:
    if format not in {"txt", "docx"}:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unsupported download format.")
    return Envelope(
        data={
            "job_id": job_id,
            "format": format,
            "message": "Skeleton only — file reassembly implemented in a later phase.",
        }
    )
