"""Context retrieval ABC + registry.

A :class:`ContextRetriever` selects the `k` most relevant prior
sentences (within a sliding `window`) to feed the translator.

CRITICAL (academic honesty + realism):
  Retrieval is **strictly causal** — for a sentence at index `i` only
  indices `< i` are candidates. No look-ahead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextHit:
    """One retrieved context sentence."""

    idx: int          # Source sentence index (< current)
    score: float      # Similarity score. Semantics depends on strategy.
    text: str         # The original source text of the retrieved sentence.


class ContextRetriever(ABC):
    """Common interface for every retrieval strategy."""

    name: str

    @abstractmethod
    def build_index(self, sentences: list[str]) -> None:
        """Prepare any persistent state for a new document.

        Semantic strategies (SBERT) will embed each sentence here and
        cache vectors. Heuristic strategies (last-N) may no-op.
        """
        raise NotImplementedError

    @abstractmethod
    def get_context(
        self,
        current_idx: int,
        *,
        k: int,
        window: int,
        min_score: float = 0.0,
    ) -> list[ContextHit]:
        """Return up to ``k`` prior sentences for sentence ``current_idx``.

        Only sentences in ``[max(0, current_idx - window) .. current_idx - 1]``
        are eligible.  Hits are sorted by ``score`` descending.
        """
        raise NotImplementedError


class ContextRetrieverRegistry:
    """Keyed store for :class:`ContextRetriever` implementations."""

    def __init__(self) -> None:
        self._items: dict[str, type[ContextRetriever]] = {}
        self._populated = False

    def register(self, name: str, cls: type[ContextRetriever]) -> None:
        self._items[name] = cls

    def available(self) -> list[str]:
        return sorted(self._items.keys())

    def new(self, name: str) -> ContextRetriever:
        if name not in self._items:
            raise KeyError(
                f"Unknown retriever '{name}'. Available: {self.available()}"
            )
        return self._items[name]()

    def populate(self) -> None:
        """Register the initial deterministic, non-semantic strategy."""
        if self._populated:
            return
        from app.context.lastn_retriever import LastNRetriever

        self.register("lastn", LastNRetriever)
        self._populated = True


registry = ContextRetrieverRegistry()
