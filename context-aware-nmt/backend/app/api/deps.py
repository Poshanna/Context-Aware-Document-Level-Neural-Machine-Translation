"""Shared FastAPI dependency injection helpers.

Examples:
    * ``get_current_user`` → authenticates via JWT in ``Authorization: Bearer``
    * ``get_db_session`` → async SQLAlchemy session (placeholder for when
      migrations are added in a later phase)

These live in one module so every router has a single place to import from.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from app.core.config import Settings, get_settings
from app.core.security import decode_access_token
from app.models import User  # noqa: F401  (used in Depends return type later)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


# ---------------------------------------------------------------------------
# Current user (JWT bearer auth)
# ---------------------------------------------------------------------------

async def get_current_user(
    request: Request,
    token: Annotated[str | None, Depends(oauth2_scheme)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> User:
    """Return the authenticated ``User`` or raise HTTP 401.

    This dependency intentionally does NOT hit the DB yet because in this
    skeleton phase we may not have a working DB session factory in all
    environments.  Once migrations are wired up in Phase 2, replace the
    body of this function with a DB lookup by ``payload["sub"]``.

    For now: the JWT ``sub`` claim is returned as a fake user with
    ``id = UUID(sub)``.  This is enough to let tests and the frontend
    exercise the authenticated surface.
    """
    del request  # reserved for later logging of client IP / UA
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing credentials.")
    try:
        payload = decode_access_token(token)
    except jwt.PyJWTError as exc:  # invalid / expired
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token.") from exc

    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token payload.")

    # --- PHASE 2 TODO: actually load User from DB by id/sub. ---
    # For the skeleton, we synthesize a minimal object that has the
    # fields the downstream services expect (``id``, ``email``).
    import uuid as _uuid

    class _SkeletonUser:
        def __init__(self, sub: str) -> None:
            self.id = _uuid.UUID(sub) if len(sub) == 36 else _uuid.uuid4()
            self.email = payload.get("email", f"{sub}@skeleton.local")

    return _SkeletonUser(sub)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Async DB session (placeholder — implemented in Phase 2 once migrations exist)
# ---------------------------------------------------------------------------

async def get_db_session() -> AsyncGenerator[None]:
    """Reserved placeholder for a SQLAlchemy async session.

    Implementation in Phase 2 will look roughly like::

        async with async_sessionmaker(engine)() as session:
            yield session
            await session.commit()
    """
    yield None
