"""Languages endpoint — currently supported languages + pairs.

Honesty rule (project_rules.md §1.2 + §1.3):
  * A language pair is only flagged ``implemented: True`` if the
    corresponding Translator subclass actually lists it in
    ``info.supported_directions``.
  * Hindi hooks are explicitly marked as (registered in registry, not yet
    surfaced in UI / eval) — we never pretend it is fully implemented.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas import Envelope, LanguageOut, LanguagePair, LanguageResponse
from app.services.translation_service import get_language_metadata


router = APIRouter(tags=["languages"])


@router.get("", summary="List supported languages + direction pairs")
async def list_languages() -> Envelope[LanguageResponse]:
    """Return metadata about currently supported languages and pairs.

    This call is cheap and auth-free so the frontend can populate
    dropdowns before a user signs in.
    """

    languages_raw, pairs_raw = get_language_metadata()
    languages = [LanguageOut(**item) for item in languages_raw]
    pairs = [LanguagePair(**item) for item in pairs_raw]

    return Envelope(
        data=LanguageResponse(languages=languages, pairs=pairs)
    )
