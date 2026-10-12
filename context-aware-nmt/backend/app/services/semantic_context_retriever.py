"""
Semantic context retrieval for document-level translation.

The retriever:
- considers only preceding sentences
- never uses future sentences
- never uses the current sentence as context
- ranks candidates using sentence embeddings
- preserves document order in the final selected context
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class ContextCandidate:
    """A preceding sentence that can potentially be used as context."""

    sentence_id: str
    text: str
    similarity: float


class SemanticContextRetriever:
    """
    Retrieve semantically relevant preceding sentences.

    Context selection is source-side only. Target/reference sentences
    are never used during retrieval.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 2,
        min_similarity: float = 0.0,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        self.model_name = model_name
        self.top_k = top_k
        self.min_similarity = min_similarity

        self._model = None

    def _load_model(self):
        """Lazy-load the embedding model."""

        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

        return self._model

    @staticmethod
    def _cosine_similarity(
        query: np.ndarray,
        candidates: np.ndarray,
    ) -> np.ndarray:
        """Calculate cosine similarity between one query and candidates."""

        query_norm = np.linalg.norm(query)

        if query_norm == 0:
            return np.zeros(
                candidates.shape[0],
                dtype=np.float32,
            )

        candidate_norms = np.linalg.norm(
            candidates,
            axis=1,
            keepdims=True,
        )

        candidate_norms = np.maximum(
            candidate_norms,
            1e-12,
        )

        normalized_query = query / query_norm
        normalized_candidates = candidates / candidate_norms

        return normalized_candidates @ normalized_query

    def retrieve(
        self,
        current_sentence: str,
        preceding_sentences: Sequence[dict],
    ) -> List[ContextCandidate]:
        """
        Retrieve relevant preceding context.

        Expected sentence format:

        {
            "sentence_id": "12.1",
            "text": "Previous sentence."
        }

        The caller must provide only preceding sentences
        belonging to the same document.
        """

        if not current_sentence or not current_sentence.strip():
            raise ValueError(
                "current_sentence must not be empty"
            )

        if not preceding_sentences:
            return []

        candidates = []

        for item in preceding_sentences:
            sentence_id = str(
                item.get("sentence_id", "")
            ).strip()

            text = str(
                item.get("text", "")
            ).strip()

            if not sentence_id or not text:
                continue

            # Never use the current sentence as context.
            if text == current_sentence.strip():
                continue

            candidates.append(
                {
                    "sentence_id": sentence_id,
                    "text": text,
                }
            )

        if not candidates:
            return []

        model = self._load_model()

        texts = [
            item["text"]
            for item in candidates
        ]

        embeddings = model.encode(
            [current_sentence] + texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

        query_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]

        similarities = self._cosine_similarity(
            query_embedding,
            candidate_embeddings,
        )

        ranked = []

        for item, similarity in zip(
            candidates,
            similarities,
        ):
            score = float(similarity)

            if score < self.min_similarity:
                continue

            ranked.append(
                ContextCandidate(
                    sentence_id=item["sentence_id"],
                    text=item["text"],
                    similarity=score,
                )
            )

        # Rank by semantic relevance.
        ranked.sort(
            key=lambda item: (
                -item.similarity,
                item.sentence_id,
            )
        )

        selected = ranked[: self.top_k]

        # Restore original document order.
        original_position = {
            item["sentence_id"]: index
            for index, item in enumerate(candidates)
        }

        selected.sort(
            key=lambda item: original_position[
                item.sentence_id
            ]
        )

        return selected

    def retrieve_texts(
        self,
        current_sentence: str,
        preceding_sentences: Sequence[dict],
    ) -> List[str]:
        """Return only the selected context text."""

        return [
            candidate.text
            for candidate in self.retrieve(
                current_sentence=current_sentence,
                preceding_sentences=preceding_sentences,
            )
        ]