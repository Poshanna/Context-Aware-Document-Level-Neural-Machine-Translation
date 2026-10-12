"""Cryptographic helpers: password hashing + JWT creation/decoding.

All secret strings are handled via ``pydantic.SecretStr`` to avoid
accidental emission to logs.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from passlib.context import CryptContext

from app.core.config import get_settings


def _build_pwd_context() -> CryptContext:
    """Return a ``CryptContext`` for the configured scheme.

    Argon2id is preferred; bcrypt is retained as a safe fallback for
    environments where argon2 fails to build.
    """
    scheme = get_settings().PASSWORD_HASH_SCHEME
    schemes = ["argon2", "bcrypt"] if scheme == "argon2" else ["bcrypt", "argon2"]
    return CryptContext(schemes=schemes, deprecated="auto")


pwd_context: CryptContext = _build_pwd_context()


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def hash_password(plain_password: str) -> str:
    """Return a one-way password hash suitable for storage."""
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Constant-time comparison between a plain password and its hash."""
    return pwd_context.verify(plain_password, hashed_password)


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------

def create_access_token(
    subject: str | int | Any,
    expires_delta: timedelta | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Create a signed JWT access token.

    ``subject`` should typically be the user ID (UUID string) so that
    the token is globally unique across password resets.
    """
    settings = get_settings()
    now = datetime.now(UTC)
    expire = now + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": now,
        "exp": expire,
    }
    if extra:
        payload.update(extra)

    secret = settings.JWT_SECRET_KEY.get_secret_value()
    return jwt.encode(payload, secret, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """Validate a JWT and return its payload.

    Raises ``jwt.PyJWTError`` subclasses for any failure.  Callers are
    responsible for translating this to HTTP 401.
    """
    settings = get_settings()
    secret = settings.JWT_SECRET_KEY.get_secret_value()
    return jwt.decode(token, secret, algorithms=[settings.JWT_ALGORITHM])
