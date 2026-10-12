"""Verify password hashing round-trips and JWT encoding/decoding.

These tests intentionally do NOT hit the web API or DB — they prove the
pure-function core works in the skeleton.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_password_roundtrip() -> None:
    pw = "correct horse battery staple 🐎"
    h = hash_password(pw)
    assert h != pw
    assert verify_password(pw, h) is True
    assert verify_password("wrong", h) is False


def test_jwt_roundtrip() -> None:
    sub = "96f109ab-8765-4321-1234-567890abcdef"
    tok = create_access_token(subject=sub, expires_delta=timedelta(minutes=5))
    payload = decode_access_token(tok)
    assert payload["sub"] == sub
    assert "iat" in payload and "exp" in payload
