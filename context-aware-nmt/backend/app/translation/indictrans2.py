"""Placeholder for IndicTrans2 backend.

Phase 3 will implement:
  * Correct SPM preprocessing/postprocessing per AI4Bharat docs.
  * Directional checkpoints te↔en.
  * Budget-aware context truncation to the model's max length.
"""
from __future__ import annotations

from app.translation import Translator, TranslatorInfo, TranslationContext


class IndicTrans2Translator(Translator):
    info = TranslatorInfo(
        name="indictrans2",
        max_seq_length=256,
        supported_directions={
            ("te", "en"), ("en", "te"),
            # ("hi", "en"), ("en", "hi"),  # future (different checkpoint names)
        },
    )

    def load(self) -> None:  # pragma: no cover - skeleton only
        raise NotImplementedError("Phase 3 implementation.")

    def translate_batch(  # pragma: no cover - skeleton only
        self,
        sentences: list[str],
        *,
        src_lang: str,
        tgt_lang: str,
        context_groups: list[list[TranslationContext]] | None = None,
    ) -> list[str]:
        raise NotImplementedError("Phase 3 implementation.")
