"""Structured logging configuration.

Keeps log output JSON-friendly in production; human-readable in dev.
Care is taken to never render ``SecretStr`` values (config module
uses them for JWT + DB password, so we simply never dump settings
objects directly into logs).
"""

from __future__ import annotations

import logging

import structlog

from app.core.config import Settings


def configure_logging(*, settings: Settings) -> None:
    """One-time logging setup. Called from the app lifespan."""
    dev_mode = settings.ENVIRONMENT == "development"

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
    ]
    if dev_mode:
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.EventRenamer("msg"))
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Align standard `logging` with structlog level
    level = logging.DEBUG if settings.DEBUG else logging.INFO
    logging.basicConfig(format="%(message)s", level=level)
    # Silence noisy third-party loggers
    logging.getLogger("transformers").setLevel(max(level, logging.INFO))
    logging.getLogger("huggingface_hub").setLevel(max(level, logging.WARNING))
