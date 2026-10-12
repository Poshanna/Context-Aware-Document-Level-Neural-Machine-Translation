"""Health-check endpoints.

Two URLs are provided, both served by the **same** function to avoid
duplicate implementations:
  * ``GET /health``      — Docker / load-balancer liveness probe (no auth).
  * ``GET /api/health``  — Browser / OpenAPI convenience (no auth).

The health endpoint is intentionally cheap and must **not** require the
DB, ML models, or GPU to be available. It reports static information
about the running configuration (project, env, demo mode flag, default
model name) without leaking secrets.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings


router = APIRouter(tags=["health"])


# NOTE: We intentionally do NOT cache this result — settings can change
# between tests. FastAPI will invoke the handler once per request.
def _build_health_payload() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "project": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        "demo_mode": bool(settings.DEMO_MODE),
        "default_model": settings.MODEL_DEFAULT,
        "version": "0.1.0",
    }


@router.get("/health", summary="Liveness + readiness probe")
async def api_health() -> dict:
    """Return minimal liveness payload (auth-free)."""
    return _build_health_payload()


async def root_health_handler(request: Request) -> JSONResponse:
    """Mounted directly in main.py at ``/health``. Reuses the same payload."""
    return JSONResponse(content=_build_health_payload())
