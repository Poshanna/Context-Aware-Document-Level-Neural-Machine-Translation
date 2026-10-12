"""Evaluation metric ABC + registry.

A :class:`Metric` takes parallel lists of ``(sources, hypotheses, references)``
and returns:

  * an **aggregate** corpus-level score (float),
  * optionally a **per_sentence** list of sentence-level scores.

Honesty rules (see ``project_rules.md`` §1):
  * Numbers MUST come from the corresponding library.  Never fudge.
  * If a metric can't be computed, raise a descriptive exception so the
    caller can surface it as an API error rather than silently return 0.
  * Metric is **deterministic by default**: run once, hash outputs,
    compare on rerun — unit test ``TR-8.2`` enforces this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class MetricResult:
    aggregate: float
    per_sentence: list[float] | None = None
    details: dict | None = None  # Metric-specific extra info (e.g., BERTScore P/R/F1 trio)


class Metric(ABC):
    """Common interface for every evaluation metric."""

    name: str
    higher_is_better: bool = True

    @abstractmethod
    def compute(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[str],
        *,
        src_lang: str,
        tgt_lang: str,
    ) -> MetricResult:
        """Evaluate ``hypotheses`` against ``references``.

        ``sources`` is provided for source-side metrics like COMET-QE;
        reference-only metrics can ignore it.
        """
        raise NotImplementedError


class MetricRegistry:
    def __init__(self) -> None:
        self._items: dict[str, type[Metric]] = {}

    def register(self, name: str, cls: type[Metric]) -> None:
        self._items[name] = cls

    def available(self) -> list[str]:
        return sorted(self._items.keys())

    def new(self, name: str) -> Metric:
        if name not in self._items:
            raise KeyError(
                f"Unknown metric '{name}'. Available: {self.available()}"
            )
        return self._items[name]()

    def populate(self) -> None:  # pragma: no cover - wire-up
        from app.evaluation.bleu_metric import BLEUMetric
        from app.evaluation.chrf_metric import CHRFMetric

        self.register("bleu", BLEUMetric)
        self.register("chrf++", CHRFMetric)


registry = MetricRegistry()
