"""Placeholder COMET metric (optional, GPU-heavy).

Rules (see ``project_rules.md``):
  * Loaded ONLY if ``ENABLE_COMET=1`` and a CUDA device exists.
  * If skipped, every API / CLI caller MUST explicitly report that
    COMET was not computed (never hide, never fabricate).
"""
from __future__ import annotations

from app.evaluation import Metric, MetricResult


class COMETMetric(Metric):
    name = "comet"
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
        raise NotImplementedError("Phase 4 implementation. Only available if ENABLE_COMET=1.")
