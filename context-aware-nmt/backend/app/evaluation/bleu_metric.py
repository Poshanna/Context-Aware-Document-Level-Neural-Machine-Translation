"""Placeholder BLEU metric using sacrebleu.

Phase 4 will implement:
  * Tokenizer selection: English → sacrebleu ``13a``; Telugu → ``none``
    (te whitespace tokenization is poor, sacrebleu defaults are fine for
    char-level BLEU variants); Hindi → ``intl``.
  * Sentence-level smoothed variant via sacrebleu sentence_bleu with
    smooth_method="add-k".
"""
from __future__ import annotations

from app.evaluation import Metric, MetricResult


class BLEUMetric(Metric):
    name = "bleu"
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
        del sources, src_lang
        if not hypotheses or len(hypotheses) != len(references):
          raise ValueError("BLEU requires equally sized, non-empty hypotheses and references.")

        from sacrebleu.metrics import BLEU

        tokenizer = "13a" if tgt_lang == "en" else "none"
        result = BLEU(tokenize=tokenizer).corpus_score(hypotheses, [references])
        return MetricResult(
          aggregate=float(result.score),
          details={"aggregation": "corpus", "tokenizer": tokenizer},
        )
