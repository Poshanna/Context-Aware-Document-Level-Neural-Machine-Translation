"""Last-N non-semantic retriever for immediately preceding sentences.

Returns up to ``min(k, window)`` preceding sentences. This is positional
context, not semantic retrieval.
"""
from __future__ import annotations

from app.context import ContextHit, ContextRetriever


class LastNRetriever(ContextRetriever):
    name = "lastn"

    def __init__(self) -> None:
        self._sentences: list[str] = []

    def build_index(self, sentences: list[str]) -> None:
        self._sentences = list(sentences)

    def get_context(
        self,
        current_idx: int,
        *,
        k: int,
        window: int,
        min_score: float = 0.0,
    ) -> list[ContextHit]:
        if k <= 0 or window <= 0 or current_idx <= 0:
            return []
        start = max(0, current_idx - min(k, window))
        end = min(current_idx, len(self._sentences))
        hits: list[ContextHit] = []
        for i in range(start, end):
            hits.append(ContextHit(idx=i, score=1.0, text=self._sentences[i]))
        # Scores are equal because this strategy ranks by recency, not semantics.
        return [h for h in hits if h.score >= min_score]
