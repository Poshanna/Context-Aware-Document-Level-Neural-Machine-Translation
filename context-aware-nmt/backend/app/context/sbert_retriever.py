"""Placeholder SBERT-based semantic retriever.

Phase 3 will implement using the multilingual MiniLM model referenced in
settings.  Streaming-friendly: embed each sentence once on its turn so
the full doc is never re-embedded.
"""
from __future__ import annotations

from app.context import ContextHit, ContextRetriever


class SentenceBERTRetriever(ContextRetriever):
    name = "sbert"

    def build_index(self, sentences: list[str]) -> None:  # pragma: no cover
        raise NotImplementedError("Phase 3 implementation.")

    def get_context(  # pragma: no cover
        self,
        current_idx: int,
        *,
        k: int,
        window: int,
        min_score: float = 0.0,
    ) -> list[ContextHit]:
        raise NotImplementedError("Phase 3 implementation.")
