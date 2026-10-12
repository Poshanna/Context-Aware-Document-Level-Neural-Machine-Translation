"""Tests for the context retrieval module (last-N strategy only — SBERT is
Phase 3). The last-N implementation already works in the skeleton because it
has no heavy dependencies, so we exercise it to prove the ABC contract.
"""
from __future__ import annotations

import pytest

from app.context.lastn_retriever import LastNRetriever


def test_lastn_retrieves_only_prior_sentences() -> None:
    r = LastNRetriever()
    sents = [f"S{i}" for i in range(6)]
    r.build_index(sents)
    hits = r.get_context(3, k=2, window=20)
    assert all(h.idx < 3 for h in hits)
    assert [h.idx for h in hits] == [1, 2]
    assert [h.text for h in hits] == ["S1", "S2"]


def test_lastn_at_zero_returns_empty() -> None:
    r = LastNRetriever()
    r.build_index(["only"])
    assert r.get_context(0, k=3, window=20) == []


def test_lastn_respects_window() -> None:
    r = LastNRetriever()
    r.build_index([f"S{i}" for i in range(10)])
    hits = r.get_context(9, k=3, window=2)
    # window=2 restricts candidates to indices 7,8; k=3 can only take 2
    assert [h.idx for h in hits] == [7, 8]
