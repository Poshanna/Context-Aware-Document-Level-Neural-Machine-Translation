"""Placeholder chrF++ (character n-gram F-score) via sacrebleu.

chrF is preferred for Indian languages because it is robust to the lack
of standard whitespace tokenisation.  Default params: ``char_order=6,
word_order=2``, matching the WMT shared-task setup.
"""
from __future__ import annotations

from app.evaluation import Metric, MetricResult


class CHRFMetric(Metric):
    name = "chrf"
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
        del sources, src_lang, tgt_lang
        if not hypotheses or len(hypotheses) != len(references):
            raise ValueError("chrF++ requires equally sized, non-empty hypotheses and references.")

        from sacrebleu.metrics import CHRF

        result = CHRF(char_order=6, word_order=2, beta=2).corpus_score(
            hypotheses, [references]
        )
        return MetricResult(
            aggregate=float(result.score),
            details={
                "aggregation": "corpus",
                "char_order": 6,
                "word_order": 2,
                "beta": 2,
                "lowercase": False,
                "whitespace": False,
                "eps_smoothing": False,
            },
        )
