"""Authentication endpoints.

Skeleton implementation: hashes passwords, issues JWTs, validates tokens.

DB persistence is Phase 2; for the skeleton we simulate persistence with
an in-process dict so that the frontend can exercise the full login
flow end-to-end.

**Note (academic integrity):** passwords are hashed, never logged, never
returned in responses.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_user
from app.core.security import create_access_token, hash_password, verify_password
from app.schemas import (
    AuthMeOut,
    Envelope,
    TokenOut,
    UserLoginIn,
    UserOut,
    UserRegisterIn,
)


router = APIRouter()

# In-memory user store for the skeleton.  Replaced by a PostgreSQL
# repository in Phase 2.  Keyed by lowercased email.
_skeleton_users: dict[str, dict] = {}


@router.post("/register", response_model=Envelope[TokenOut])
async def register(payload: UserRegisterIn) -> Envelope[TokenOut]:
    key = payload.email.lower()
    if key in _skeleton_users:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Email already registered.")

    user_id = uuid.uuid4()
    record = {
        "id": user_id,
        "email": payload.email,
        "full_name": payload.full_name,
        "password_hash": hash_password(payload.password),
    }
    _skeleton_users[key] = record

    user = UserOut(
        id=record["id"],
        email=record["email"],
        full_name=record["full_name"],
        created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )
    token = create_access_token(subject=str(user.id), extra={"email": user.email})
    return Envelope(data=TokenOut(access_token=token, user=user))


@router.post("/login", response_model=Envelope[TokenOut])
async def login(payload: UserLoginIn) -> Envelope[TokenOut]:
    record = _skeleton_users.get(payload.email.lower())
    if not record or not verify_password(payload.password, record["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials.")

    user = UserOut(
        id=record["id"],
        email=record["email"],
        full_name=record["full_name"],
        created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )
    token = create_access_token(subject=str(user.id), extra={"email": user.email})
    return Envelope(data=TokenOut(access_token=token, user=user))


@router.get("/me", response_model=Envelope[AuthMeOut])
async def me(
    current_user: Annotated[object, Depends(get_current_user)],
) -> Envelope[AuthMeOut]:
    # ``current_user`` is the skeleton object from deps.get_current_user.
    import datetime as _dt

    return Envelope(
        data=AuthMeOut(
            id=getattr(current_user, "id"),
            email=getattr(current_user, "email"),
            full_name=None,
            created_at=_dt.datetime.now(_dt.timezone.utc),
        )
    )
