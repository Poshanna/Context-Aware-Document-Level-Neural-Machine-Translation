"""Run a deterministic, paired DocHPLT held-out evaluation subset."""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
import uuid
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.evaluation import registry as metric_registry  # noqa: E402
from app.evaluation.dochplt_adapter import (  # noqa: E402
    ALIGNER_SCORE_THRESHOLD,
    DATASET_CONFIG,
    DATASET_NAME,
    DATASET_REVISION,
    DATASET_SPLIT,
    DocHPLTContextExample,
    DocHPLTTrustedAdapter,
)
from app.evaluation.document_evaluation import ProvenanceStatus  # noqa: E402
from app.schemas import TextTranslateIn  # noqa: E402
from app.services.context_conditioned_translation_service import (  # noqa: E402
    ContextConditionedTranslationService,
)
from app.services.translation_service import TranslationService  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "data/raw/dochplt/en-te/train-00000-of-00030.parquet"
SEED = 42
CONTEXT_WINDOW = 2
MAX_SAMPLES = 500
RUN_LABEL = "deterministic held-out evaluation subset from the DocHPLT training corpus"


def _validated_rows(shard_path: Path, adapter: DocHPLTTrustedAdapter) -> list[dict[str, Any]]:
    if not shard_path.is_file():
        raise FileNotFoundError(f"DocHPLT shard not found: {shard_path}")

    rows: list[dict[str, Any]] = []
    for row in pq.read_table(str(shard_path)).to_pylist():
        if not isinstance(row, dict):
            continue
        try:
            adapter._validate_record_schema(row)
            adapter._validate_row_identity(row)
            adapter._validate_alignment_references(row)
        except (TypeError, ValueError):
            continue
        rows.append(dict(row))
    if not rows:
        raise ValueError(f"No rows pass the existing DocHPLT validation contract: {shard_path}")
    return rows


def _candidate_examples(
    *, shard_path: Path, adapter: DocHPLTTrustedAdapter
) -> tuple[DocHPLTContextExample, ...]:
    source_rows = _validated_rows(shard_path, adapter)
    alignment_count = sum(len(row.get("alignment", ())) for row in source_rows)
    with tempfile.TemporaryDirectory(prefix="dochplt-bounded-") as temporary_directory:
        filtered_shard = Path(temporary_directory) / "en-te" / shard_path.name
        filtered_shard.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(source_rows), filtered_shard)
        subset = adapter.build_evaluation_subset(
            shard_path=filtered_shard,
            max_samples=max(1, alignment_count),
            seed=SEED,
            context_window=CONTEXT_WINDOW,
            aligner_score_threshold=ALIGNER_SCORE_THRESHOLD,
        )

    source_location = str(shard_path.resolve())
    examples = tuple(
        replace(
            example,
            provenance=adapter.verified_provenance(
                status=ProvenanceStatus.VERIFIED_DATASET,
                dataset_name=example.provenance.dataset_name or DATASET_NAME,
                dataset_revision=example.provenance.dataset_revision or DATASET_REVISION,
                split=example.provenance.split or DATASET_SPLIT,
                record_id=example.provenance.record_id or "",
                source_location=source_location,
                reference_location=source_location,
                alignment_location=source_location,
                unit_id=example.provenance.unit_id,
                unit_type=example.provenance.unit_type,
            ),
        )
        for example in subset.selected_examples
    )
    return examples


def _unique_sample(
    candidates: tuple[DocHPLTContextExample, ...], requested_count: int
) -> tuple[list[DocHPLTContextExample], int, int]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[DocHPLTContextExample] = []
    duplicate_candidates = 0
    for example in candidates:
        key = (example.src_doc_id, example.source_sentence_id, example.target_sentence_id)
        if key in seen:
            duplicate_candidates += 1
            continue
        seen.add(key)
        unique.append(example)
    if len(unique) < requested_count:
        raise ValueError(
            f"Only {len(unique)} unique eligible examples are available; {requested_count} requested."
        )
    selected = unique[:requested_count]
    selected_keys = [
        (item.src_doc_id, item.source_sentence_id, item.target_sentence_id)
        for item in selected
    ]
    duplicate_count = len(selected_keys) - len(set(selected_keys))
    return selected, duplicate_count, duplicate_candidates


def _translate_baseline(
    service: TranslationService, source_sentence: str
) -> tuple[str | None, str | None]:
    try:
        response = service.translate_text(
            user_id=None,
            req=TextTranslateIn(
                source_lang="en",
                target_lang="te",
                pipeline="baseline",
                model="nllb200",
                text=source_sentence,
            ),
        )
        if response.status != "ok" or not isinstance(response.translated_text, str):
            return None, response.note or "Baseline translation returned no output."
        return response.translated_text, None
    except Exception as exc:  # noqa: BLE001 - preserve actual inference failures
        return None, f"{type(exc).__name__}: {exc}"


def _translate_context(
    service: ContextConditionedTranslationService,
    example: DocHPLTContextExample,
) -> tuple[str | None, str | None, str | None]:
    try:
        response = service.translate_sentence(
            source_lang="en",
            target_lang="te",
            current_sentence=example.source_sentence,
            preceding_context=list(example.context_sentences),
            context_window=CONTEXT_WINDOW,
            model="nllb200",
        )
        hypothesis = response.get("translation")
        if not isinstance(hypothesis, str):
            return None, response.get("formatted_input"), "Context translation returned no text."
        return hypothesis, response.get("formatted_input"), None
    except Exception as exc:  # noqa: BLE001 - preserve actual inference failures
        return None, None, f"{type(exc).__name__}: {exc}"


def _metric_score(name: str, sources: list[str], hypotheses: list[str], references: list[str]) -> float:
    metric_registry.populate()
    metric = metric_registry.new(name)
    result = metric.compute(
        sources,
        hypotheses,
        references,
        src_lang="en",
        tgt_lang="te",
    )
    return float(result.aggregate)


def _metric_result(
    name: str, sources: list[str], hypotheses: list[str], references: list[str]
) -> dict[str, Any]:
    if not hypotheses:
        return {"status": "unavailable", "score": None, "num_examples": 0}
    try:
        return {
            "status": "computed",
            "score": _metric_score(name, sources, hypotheses, references),
            "num_examples": len(hypotheses),
        }
    except Exception as exc:  # noqa: BLE001 - record metric failures in the run artifact
        return {
            "status": "failed",
            "score": None,
            "num_examples": len(hypotheses),
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _per_example_metrics(
    source: str, hypothesis: str, reference: str
) -> dict[str, float | None]:
    scores: dict[str, float | None] = {"bleu": None, "chrf++": None}
    for name in scores:
        try:
            scores[name] = _metric_score(name, [source], [hypothesis], [reference])
        except Exception:
            scores[name] = None
    return scores


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _paired_metric_summary(differences: list[float]) -> dict[str, Any]:
    ties = sum(value == 0 for value in differences)
    context_better = sum(value > 0 for value in differences)
    baseline_better = sum(value < 0 for value in differences)
    return {
        "mean_difference": statistics.fmean(differences) if differences else None,
        "context_better_count": context_better,
        "baseline_better_count": baseline_better,
        "tied_count": ties,
        "paired_example_count": len(differences),
    }


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def main() -> int:
    shard_path = DATASET_PATH
    adapter = DocHPLTTrustedAdapter()
    candidates = _candidate_examples(shard_path=shard_path, adapter=adapter)
    selected, duplicate_count, duplicate_candidates_skipped = _unique_sample(
        candidates, MAX_SAMPLES
    )

    settings = get_settings()
    translation_service = TranslationService(settings=settings)
    context_service = ContextConditionedTranslationService(
        translation_service=translation_service,
        settings=settings,
    )

    examples: list[dict[str, Any]] = []
    for index, example in enumerate(selected, start=1):
        baseline_hypothesis, baseline_error = _translate_baseline(
            translation_service, example.source_sentence
        )
        context_hypothesis, formatted_input, context_error = _translate_context(
            context_service, example
        )
        pair_failed = baseline_error is not None or context_error is not None
        baseline_example_metrics: dict[str, float | None] = {"bleu": None, "chrf++": None}
        context_example_metrics: dict[str, float | None] = {"bleu": None, "chrf++": None}
        differences: dict[str, float | None] = {"bleu": None, "chrf++": None}

        if not pair_failed and baseline_hypothesis is not None and context_hypothesis is not None:
            baseline_example_metrics = _per_example_metrics(
                example.source_sentence, baseline_hypothesis, example.target_reference
            )
            context_example_metrics = _per_example_metrics(
                example.source_sentence, context_hypothesis, example.target_reference
            )
            for metric_name in differences:
                baseline_score = baseline_example_metrics[metric_name]
                context_score = context_example_metrics[metric_name]
                if baseline_score is not None and context_score is not None:
                    differences[metric_name] = context_score - baseline_score

        examples.append(
            {
                "example_index": index,
                "source_document_id": example.src_doc_id,
                "target_document_id": example.tgt_doc_id,
                "context_sentence_ids": list(example.context_sentence_ids),
                "context_sentences": list(example.context_sentences),
                "current_source_sentence_id": example.source_sentence_id,
                "current_source_sentence": example.source_sentence,
                "target_sentence_id": example.target_sentence_id,
                "reference": example.target_reference,
                "alignment_score": float(example.alignment_score),
                "baseline_hypothesis": baseline_hypothesis,
                "context_hypothesis": context_hypothesis,
                "context_model_input": formatted_input,
                "context_was_supplied": bool(formatted_input and example.context_sentences),
                "baseline_status": "failed" if baseline_error else "computed",
                "context_status": "failed" if context_error else "computed",
                "paired_status": "failed" if pair_failed else "complete",
                "baseline_failure": baseline_error,
                "context_failure": context_error,
                "per_example_metrics": {
                    "baseline": baseline_example_metrics,
                    "context": context_example_metrics,
                },
                "paired_metric_differences": differences,
                "dataset_provenance": {
                    "status": example.provenance.status.value,
                    "dataset_name": example.provenance.dataset_name,
                    "dataset_revision": example.provenance.dataset_revision,
                    "split": example.provenance.split,
                    "record_id": example.provenance.record_id,
                    "source_location": example.provenance.source_location,
                    "reference_location": example.provenance.reference_location,
                    "alignment_location": example.provenance.alignment_location,
                    "adapter_name": example.provenance.adapter_name,
                    "unit_id": example.provenance.unit_id,
                    "unit_type": example.provenance.unit_type.value if example.provenance.unit_type else None,
                },
            }
        )
        if index % 25 == 0 or index == len(selected):
            print(f"Processed {index}/{len(selected)} paired examples")

    paired_examples = [row for row in examples if row["paired_status"] == "complete"]
    sources = [row["current_source_sentence"] for row in paired_examples]
    references = [row["reference"] for row in paired_examples]
    baseline_hypotheses = [row["baseline_hypothesis"] for row in paired_examples]
    context_hypotheses = [row["context_hypothesis"] for row in paired_examples]

    baseline_metrics = {
        "bleu": _metric_result("bleu", sources, baseline_hypotheses, references),
        "chrf++": _metric_result("chrf++", sources, baseline_hypotheses, references),
    }
    context_metrics = {
        "bleu": _metric_result("bleu", sources, context_hypotheses, references),
        "chrf++": _metric_result("chrf++", sources, context_hypotheses, references),
    }
    bleu_differences = [
        row["paired_metric_differences"]["bleu"]
        for row in paired_examples
        if row["paired_metric_differences"]["bleu"] is not None
    ]
    chrf_differences = [
        row["paired_metric_differences"]["chrf++"]
        for row in paired_examples
        if row["paired_metric_differences"]["chrf++"] is not None
    ]

    source_doc_counts = Counter(row["source_document_id"] for row in examples)
    target_doc_counts = Counter(row["target_document_id"] for row in examples)
    alignment_scores = [row["alignment_score"] for row in examples]
    inference_failure_count = sum(row["paired_status"] == "failed" for row in examples)
    baseline_failure_count = sum(row["baseline_status"] == "failed" for row in examples)
    context_failure_count = sum(row["context_status"] == "failed" for row in examples)

    run_id = _run_id()
    run_dir = PROJECT_ROOT / "experiments" / "evaluation" / "dochplt" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "label": RUN_LABEL,
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "revision": DATASET_REVISION,
        "source_shard": "data/raw/dochplt/en-te/train-00000-of-00030.parquet",
        "seed": SEED,
        "context_window": CONTEXT_WINDOW,
        "aligner_score_threshold": ALIGNER_SCORE_THRESHOLD,
        "requested_sample_count": MAX_SAMPLES,
        "actual_sample_count": len(examples),
        "unique_source_documents": len(source_doc_counts),
        "unique_target_documents": len(target_doc_counts),
        "examples_per_source_document": dict(sorted(source_doc_counts.items())),
        "examples_per_target_document": dict(sorted(target_doc_counts.items())),
        "source_examples_per_document_minimum": min(source_doc_counts.values()),
        "source_examples_per_document_maximum": max(source_doc_counts.values()),
        "target_examples_per_document_minimum": min(target_doc_counts.values()),
        "target_examples_per_document_maximum": max(target_doc_counts.values()),
        "duplicate_count": duplicate_count,
        "duplicate_candidate_count_skipped": duplicate_candidates_skipped,
        "inference_failure_count": inference_failure_count,
        "baseline_failure_count": baseline_failure_count,
        "context_failure_count": context_failure_count,
        "paired_metric_example_count": len(paired_examples),
        "baseline_metrics": baseline_metrics,
        "context_metrics": context_metrics,
        "paired_comparison": {
            "bleu": _paired_metric_summary(bleu_differences),
            "chrf++": _paired_metric_summary(chrf_differences),
            "statistical_significance_tested": False,
            "per_example_metric_method": "Existing metric implementation applied to each example as a singleton corpus; corpus BLEU remains unsmoothed.",
        },
        "alignment_score_statistics": _distribution(alignment_scores),
        "provenance": {
            "status": "verified_dataset",
            "dataset_name": DATASET_NAME,
            "dataset_revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "source_location": str(shard_path.resolve()),
            "reference_location": str(shard_path.resolve()),
            "alignment_location": str(shard_path.resolve()),
            "adapter_name": adapter.adapter_name,
            "unit_type": "document",
            "filtered_invalid_or_mixed_alignment_rows": True,
            "note": "All selected examples retain the adapter-issued provenance for the original pinned shard. No cross-row or inferred alignment/context is used.",
        },
        "document_level_evaluation": False,
        "context_conditioning": "prefix",
        "evaluation_scope": "Pipeline evaluation only; not an official test set. Both aggregate conditions use the same complete paired examples and references.",
        "model_name": "nllb200",
        "run_id": run_id,
    }

    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Artifact directory: {run_dir}")
    print(f"Summary: {run_dir / 'summary.json'}")
    print(f"Examples: {run_dir / 'examples.json'}")
    print(f"Actual examples: {len(examples)}; unique source docs: {len(source_doc_counts)}; unique target docs: {len(target_doc_counts)}")
    print(f"Paired inference failures: {inference_failure_count}")
    print(f"Baseline metrics: {json.dumps(baseline_metrics, ensure_ascii=False)}")
    print(f"Context metrics: {json.dumps(context_metrics, ensure_ascii=False)}")
    print(f"Paired comparison: {json.dumps(summary['paired_comparison'], ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
