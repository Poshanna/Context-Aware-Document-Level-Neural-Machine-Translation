"""Placeholder BERTScore using the published multilingual-cased checkpoint.

Phase 4 will implement:
  * ``bert-base-multilingual-cased`` as default (works for te/en/hi).
  * By default report F1 (``bertscore_f1``); precision/recall are stored
    in ``MetricResult.details`` if needed.
  * Layer used: default layer 9 per the BERTScore paper recommendation
    for multilingual models.
"""
from __future__ import annotations

from app.evaluation import Metric, MetricResult


class BERTScoreMetric(Metric):
    name = "bertscore"
    higher_is_better = True

    def compute(  # pragma: no cover - skeleton only
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[str],
        *,
        src_lang: str,
        tgt_lang: str,
    ) -> MetricResult:
        raise NotImplementedError("Phase 4 implementation.")
