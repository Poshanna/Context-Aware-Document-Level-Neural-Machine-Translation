"""Evaluation interface for future context-conditioned experiments.

This module defines the data structure for real evaluation later, but intentionally
keeps the actual metric calculation disabled until the project is ready to run a
measured comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextEvaluationExample:
    """One source-reference-hypothesis comparison row for context-aware work."""

    source: str
    reference: str
    baseline_hypothesis: str
    context_conditioned_hypothesis: str
    source_lang: str = "en"
    target_lang: str = "te"


class ContextEvaluationInterface:
    """Small interface for later metric execution and result packaging."""

    def __init__(self, *, metrics: list[str] | None = None) -> None:
        self.metrics = metrics or ["bleu", "chrf++", "bertscore", "comet"]

    def validate_example(self, example: ContextEvaluationExample) -> None:
        missing = [
            name
            for name, value in {
                "source": example.source,
                "reference": example.reference,
                "baseline_hypothesis": example.baseline_hypothesis,
                "context_conditioned_hypothesis": example.context_conditioned_hypothesis,
            }.items()
            if value is None or value == ""
        ]
        if missing:
            raise ValueError(f"Missing required evaluation fields: {missing}")

    def run(
        self,
        examples: list[ContextEvaluationExample],
    ) -> dict[str, dict[str, Any]]:
        """Placeholder that keeps experimental evaluation reproducible but not scored."""
        if not examples:
            return {metric: {"aggregate": None, "per_sentence": None} for metric in self.metrics}

        for example in examples:
            self.validate_example(example)

        raise NotImplementedError(
            "Metric execution is intentionally not run yet. This interface is prepared "
            "for later BLEU, chrF++, BERTScore, and COMET evaluation once a measured "
            "comparison workflow is approved."
        )


def prepare_context_evaluation_example(
    *,
    source: str,
    reference: str,
    baseline_hypothesis: str,
    context_conditioned_hypothesis: str,
    source_lang: str = "en",
    target_lang: str = "te",
) -> ContextEvaluationExample:
    return ContextEvaluationExample(
        source=source,
        reference=reference,
        baseline_hypothesis=baseline_hypothesis,
        context_conditioned_hypothesis=context_conditioned_hypothesis,
        source_lang=source_lang,
        target_lang=target_lang,
    )
