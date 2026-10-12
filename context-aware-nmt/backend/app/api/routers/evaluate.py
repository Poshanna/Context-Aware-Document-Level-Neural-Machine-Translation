"""Evaluation run / dashboard endpoints.

Skeleton: validates inputs and returns placeholders.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.schemas import Envelope


router = APIRouter()


@router.post("/run")
async def start_evaluation_run(
    _user: Annotated[object, Depends(get_current_user)],
    pipelines: list[str] | None = None,
    metrics: list[str] | None = None,
    allow_path: bool = False,
) -> Envelope[dict]:
    settings = get_settings()
    if allow_path and not settings.EVAL_ALLOW_LOCAL_PATHS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "allow_path disabled by env. Upload source and reference files.",
        )
    pipelines = pipelines or ["baseline", "context_aware"]
    metrics = metrics or ["bleu", "chrf", "bertscore"]
    bad_pipes = [p for p in pipelines if p not in {"baseline", "context_aware"}]
    bad_metrics = [
        m for m in metrics
        if m not in {"bleu", "chrf", "bertscore", "comet"}
    ]
    if bad_pipes or bad_metrics:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid pipes/metrics: {bad_pipes}{bad_metrics}")
    if "comet" in metrics and not settings.ENABLE_COMET:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "COMET disabled in env. Set ENABLE_COMET=1 and ensure GPU is available.",
        )
    return Envelope(
        data={
            "run_id": "00000000-0000-0000-0000-000000000000",
            "pipelines": pipelines,
            "metrics": metrics,
            "message": "Skeleton only — metrics computation in later phase.",
        }
    )


@router.get("/runs")
async def list_evaluations(
    _user: Annotated[object, Depends(get_current_user)],
) -> Envelope[dict]:
    return Envelope(data={"items": [], "total": 0})


@router.get("/runs/{run_id}")
async def get_evaluation_summary(
    run_id: str,
    _user: Annotated[object, Depends(get_current_user)],
) -> Envelope[dict]:
    return Envelope(
        data={
            "id": run_id,
            "model_name": "nllb200",
            "pipelines_evaluated": ["baseline", "context_aware"],
            "per_pipeline": {},
            "compare": [],
            "message": "Skeleton only.",
        }
    )


@router.get("/runs/{run_id}/compare")
async def get_evaluation_compare(
    run_id: str,
    _user: Annotated[object, Depends(get_current_user)],
) -> Envelope[dict]:
    return Envelope(
        data={
            "run_id": run_id,
            "rows": [],
            "message": "Skeleton only.",
        }
    )
