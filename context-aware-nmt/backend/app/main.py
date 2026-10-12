"""Application entrypoint — FastAPI app factory.

This module composes the app: loads settings, wires up routers, and
registers lifespan events for one-time model loading.

Architecture rules (project_rules.md §3) enforced here:
  * Routers have NO business logic — they call services.
  * ML / context / preprocessing / evaluation are pure libraries behind
    registries; they are NOT imported directly from routers (services do).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routers import (
    auth,
    evaluate,
    health as health_router,
    history,
    jobs,
    languages as languages_router,
    translate,
)
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.schemas import Envelope


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage long-lived resources.

    * Configures logging once on startup.
    * NOTE: We intentionally do NOT download or load HuggingFace models
      here automatically. Each registry provides `load(name)` which is
      called on first translation request — or can be called via a CLI
      admin command later. This keeps startup fast and avoids accidental
      huge downloads.
    """
    settings = get_settings()
    configure_logging(settings=settings)
    yield


def _build_error_response(code: str, message: str, *, http_status: int) -> JSONResponse:
    """Produce a JSON response shaped like ``Envelope.error`` so every
    failure surface (HTTPException, validation errors, 500s) shares the
    same envelope format the frontend knows how to render.
    """

    envelope = Envelope[Any](
        data=None,
        error={
            "code": code,
            "message": message,
            "http_status": http_status,
        },
    )
    return JSONResponse(
        status_code=http_status,
        content=envelope.model_dump(mode="json"),
    )


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=(
            "Context-Aware Document-Level Neural Machine Translation for "
            "Indian Languages. Research code — all models are pretrained; "
            "no training is performed by this server."
        ),
        version="0.1.0",
        lifespan=lifespan,
        debug=settings.DEBUG,
        contact={"name": "BE AI&ML Project Team"},
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Mount /health at root (Docker healthcheck convenience) WITHOUT
    # duplicating the implementation. Same payload as /api/health.
    app.add_api_route(
        "/health",
        health_router.root_health_handler,
        methods=["GET"],
        include_in_schema=True,
        tags=["health"],
        summary="Root-level health alias (same payload as /api/health)",
    )

    # --- Routers -------------------------------------------------------
    # Routers MUST NOT contain business logic; delegate to services.
    app.include_router(health_router.router, prefix="/api", tags=["health"])
    app.include_router(
        languages_router.router, prefix="/api/languages", tags=["languages"]
    )
    app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
    app.include_router(
        translate.router, prefix="/api/translate", tags=["translate"]
    )
    app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
    app.include_router(history.router, prefix="/api/history", tags=["history"])
    app.include_router(
        evaluate.router, prefix="/api/evaluate", tags=["evaluate"]
    )

    # --- Exception handlers: standardize errors into Envelope.error. ---

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        del request
        # Routers use a convention: detail may be a dict {"code":..., "message":...}
        # from _service_error_to_http, or a plain string.
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail and "message" in detail:
            code = str(detail["code"])
            message = str(detail["message"])
        else:
            code = {
                401: "unauthorized",
                403: "forbidden",
                404: "not_found",
                405: "method_not_allowed",
                413: "request_entity_too_large",
                422: "validation_error",
            }.get(exc.status_code, f"http_{exc.status_code}")
            message = str(detail) if detail is not None else "HTTP error."
        return _build_error_response(
            code=code,
            message=message,
            http_status=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        del request
        errors = exc.errors()
        # Build a concise, user-readable message that lists the first
        # failing path. The raw errors list is preserved so the UI can
        # highlight specific fields.
        first = errors[0] if errors else None
        loc = ".".join(str(p) for p in first["loc"]) if first else "body"
        msg = f"Validation error in {loc}: {first['msg'] if first else 'invalid'}"
        return _build_error_response(
            code="validation_error",
            message=msg,
            http_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    @app.exception_handler(404)
    async def _not_found_handler(request: Request, exc: Any) -> JSONResponse:
        del request, exc
        return _build_error_response(
            code="not_found",
            message="The requested resource does not exist.",
            http_status=status.HTTP_404_NOT_FOUND,
        )

    @app.exception_handler(405)
    async def _method_not_allowed_handler(request: Request, exc: Any) -> JSONResponse:
        del request, exc
        return _build_error_response(
            code="method_not_allowed",
            message="HTTP method not allowed on this endpoint.",
            http_status=status.HTTP_405_METHOD_NOT_ALLOWED,
        )

    @app.exception_handler(500)
    async def _server_error_handler(request: Request, exc: Any) -> JSONResponse:
        del request, exc
        # Do NOT leak the exception message. Generic 500 only.
        return _build_error_response(
            code="internal_server_error",
            message="An internal server error occurred.",
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return app


app = create_app()
