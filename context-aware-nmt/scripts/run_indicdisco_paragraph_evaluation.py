#!/usr/bin/env python3
"""Bounded paragraph-level IndicDISCO-MT baseline evaluation.

This script uses only explicitly supplied, pinned core paragraph rows. It does
not infer document identity/order or context, and it scores assembled paragraph
outputs only. Running this CLI performs real NLLB inference and metrics; tests
use injected synthetic records/services and never call the CLI against real data.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import sys
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import Settings, get_settings  # noqa: E402
from app.evaluation import registry as metric_registry  # noqa: E402
from app.evaluation.experiment_runner import (  # noqa: E402
    _checkpoint_generation_defaults,
    _corpus_metric,
    _device_name,
    _software_versions,
)
from app.evaluation.indicdisco_adapter import (  # noqa: E402
    INDICDISCO_DATASET,
    INDICDISCO_PARAGRAPH_COMPONENT,
    INDICDISCO_REVISION,
    INDICDISCO_SOURCE_LANGUAGE,
    INDICDISCO_TARGET_LANGUAGE,
    IndicDiscoMetric,
    IndicDiscoMTTrustedAdapter,
    IndicDiscoParagraphRecord,
    metric_eligibility,
)
from app.schemas import TextTranslateIn  # noqa: E402
from app.services.translation_service import TranslationService  # noqa: E402
from scripts.validate_indicdisco_structure import (  # noqa: E402
    DEFAULT_MAX_BYTES,
    MAX_RANGE_BYTES,
    ParsedCsvPrefix,
    SourceInput,
    classify_source,
    official_source_url,
    read_bounded_csv_prefix,
    verify_local_source,
)

DEFAULT_MAX_CANDIDATE_ROWS = 100
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "evaluation" / "indicdisco_paragraph"
EXPECTED_MAX_NEW_TOKENS = 256
EXPECTED_MAX_LENGTH = 200
_LANGUAGE_CODES = {"te": INDICDISCO_SOURCE_LANGUAGE, "en": INDICDISCO_TARGET_LANGUAGE}
MetricComputer = Callable[[str, list[str], list[str], list[str], str, str], dict[str, Any]]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def load_paragraph_records(
    *,
    source: str,
    revision: str,
    component: str,
    split: str,
    source_lang: str,
    target_lang: str,
    max_candidate_rows: int,
    max_bytes: int,
) -> tuple[list[IndicDiscoParagraphRecord], list[dict[str, str]], int, int, SourceInput, ParsedCsvPrefix]:
    if (source_lang, target_lang) != ("te", "en"):
        raise ValueError("IndicDISCO-MT core release direction is Telugu->English (`te`->`en`).")
    if revision != INDICDISCO_REVISION:
        raise ValueError("revision must match the pinned IndicDISCO-MT commit.")
    if component != INDICDISCO_PARAGRAPH_COMPONENT:
        raise ValueError("only the Indicdisco_MT paragraph component is supported.")
    if not 1 <= max_candidate_rows <= DEFAULT_MAX_CANDIDATE_ROWS:
        raise ValueError(f"max_candidate_rows must be between 1 and {DEFAULT_MAX_CANDIDATE_ROWS}.")
    if not 1 <= max_bytes <= MAX_RANGE_BYTES:
        raise ValueError(f"max_bytes must be between 1 and {MAX_RANGE_BYTES}.")

    source_input = classify_source(
        source,
        revision=revision,
        component=component,
        split=split,
    )
    source_input = verify_local_source(source_input, split=split)
    parsed, bytes_read = read_bounded_csv_prefix(
        source_input,
        max_rows=max_candidate_rows,
        max_bytes=max_bytes,
    )
    records: list[IndicDiscoParagraphRecord] = []
    exclusions: list[dict[str, str]] = []
    adapter = IndicDiscoMTTrustedAdapter()

    for row_index, values in enumerate(parsed.rows):
        row_locator = f"{source_input.location}#row={row_index}"
        if not source_input.pinned_official:
            exclusions.append({"row_locator": row_locator, "reason": "source_not_verified_against_pinned_commit"})
            continue
        if parsed.header != ("source", "target"):
            exclusions.append({"row_locator": row_locator, "reason": "unexpected_core_paragraph_schema"})
            continue
        if len(values) != 2:
            exclusions.append({"row_locator": row_locator, "reason": "row_column_count_does_not_match_header"})
            continue
        if not values[0].strip():
            exclusions.append({"row_locator": row_locator, "reason": "empty_source_paragraph"})
            continue
        if not values[1].strip():
            exclusions.append({"row_locator": row_locator, "reason": "empty_reference_paragraph"})
            continue

        try:
            record = adapter.adapt_row(
                row=dict(zip(parsed.header, values)),
                revision=revision,
                split=split,
                component=component,
                row_locator=row_locator,
                source_location=f"{source_input.location}#row={row_index}:source",
                reference_location=f"{source_input.location}#row={row_index}:target",
                alignment_location=f"{source_input.location}#row={row_index}:source,target",
                source_language=INDICDISCO_SOURCE_LANGUAGE,
                target_language=INDICDISCO_TARGET_LANGUAGE,
            )
        except (TypeError, ValueError) as exc:
            exclusions.append({"row_locator": row_locator, "reason": f"adapter_rejected: {exc}"})
            continue

        paragraph_bleu = metric_eligibility(record, IndicDiscoMetric.PARAGRAPH_BLEU)
        paragraph_chrf = metric_eligibility(record, IndicDiscoMetric.PARAGRAPH_CHRF)
        if not paragraph_bleu.eligible or not paragraph_chrf.eligible:
            exclusions.append({"row_locator": row_locator, "reason": "paragraph_metric_ineligible"})
            continue
        records.append(record)

    return records, exclusions, len(parsed.rows), bytes_read, source_input, parsed


def select_bounded_sample(
    records: Sequence[IndicDiscoParagraphRecord],
    *,
    max_samples: int,
    seed: int,
) -> list[IndicDiscoParagraphRecord]:
    if isinstance(max_samples, bool) or max_samples < 1:
        raise ValueError("max_samples must be at least 1.")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be non-negative.")
    sample_size = min(max_samples, len(records))
    if sample_size == len(records):
        return list(records)
    return random.Random(seed).sample(list(records), sample_size)


def translate_paragraphs(
    records: Sequence[IndicDiscoParagraphRecord],
    *,
    translation_service: Any,
    source_lang: str,
    target_lang: str,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for record in records:
        prediction: dict[str, Any] = {
            "dataset": record.dataset,
            "revision": record.revision,
            "component": record.component,
            "split": record.split,
            "row_locator": record.row_locator,
            "source_language": record.source_language,
            "target_language": record.target_language,
            "source": record.source_text,
            "reference": record.reference_text,
            "hypothesis": None,
            "inference_status": "pending",
            "document_id": None,
            "context_used": False,
            "metric_eligibility": {
                "paragraph_bleu": False,
                "paragraph_chrf++": False,
            },
        }
        try:
            response = translation_service.translate_text(
                user_id=None,
                req=TextTranslateIn(
                    source_lang=source_lang,
                    target_lang=target_lang,
                    pipeline="baseline",
                    model="nllb200",
                    text=record.source_text,
                ),
            )
            if response.status != "ok" or not isinstance(response.translated_text, str) or not response.translated_text.strip():
                raise RuntimeError(response.note or "Baseline translation returned no paragraph hypothesis.")
            prediction["hypothesis"] = response.translated_text
            prediction["inference_status"] = "succeeded"
            prediction["metric_eligibility"] = {
                "paragraph_bleu": metric_eligibility(record, IndicDiscoMetric.PARAGRAPH_BLEU).eligible,
                "paragraph_chrf++": metric_eligibility(record, IndicDiscoMetric.PARAGRAPH_CHRF).eligible,
            }
        except Exception as exc:  # noqa: BLE001 - preserve per-row backend failures without fabricating text
            prediction["inference_status"] = "failed"
            prediction["inference_error"] = f"{type(exc).__name__}: {exc}"
        predictions.append(prediction)
    return predictions


def compute_paragraph_metrics(
    predictions: Sequence[dict[str, Any]],
    *,
    source_lang: str,
    target_lang: str,
    metric_computer: MetricComputer = _corpus_metric,
) -> dict[str, dict[str, Any]]:
    names = ("bleu", "chrf++")
    if not predictions:
        return {
            f"corpus_{name}": {"status": "unavailable", "score": None, "reason": "no eligible paragraph pairs"}
            for name in names
        }
    if any(row.get("inference_status") != "succeeded" for row in predictions):
        return {
            f"corpus_{name}": {
                "status": "failed",
                "score": None,
                "reason": "one or more paragraph translations failed; complete-sample corpus metric suppressed",
            }
            for name in names
        }
    if any(
        not row.get("metric_eligibility", {}).get("paragraph_bleu")
        or not row.get("metric_eligibility", {}).get("paragraph_chrf++")
        for row in predictions
    ):
        return {
            f"corpus_{name}": {"status": "unavailable", "score": None, "reason": "paragraph metric eligibility failed"}
            for name in names
        }

    if metric_computer is _corpus_metric:
        metric_registry.populate()
    sources = [str(row["source"]) for row in predictions]
    hypotheses = [str(row["hypothesis"]) for row in predictions]
    references = [str(row["reference"]) for row in predictions]
    return {
        f"corpus_{name}": metric_computer(name, sources, hypotheses, references, source_lang, target_lang)
        for name in names
    }


def metric_configuration(target_lang: str) -> dict[str, Any]:
    return {
        "aggregation": "corpus",
        "normalization": "none; preserve source, reference, and hypothesis strings",
        "bleu": {
            "implementation": "app.evaluation.bleu_metric.BLEUMetric via SacreBLEU",
            "tokenizer": "13a" if target_lang == "en" else "none",
            "smoothing": "none",
        },
        "chrf++": {
            "implementation": "app.evaluation.chrf_metric.CHRFMetric via SacreBLEU",
            "char_order": 6,
            "word_order": 2,
            "beta": 2,
            "lowercase": False,
            "whitespace": False,
            "eps_smoothing": False,
        },
        "sacrebleu_version": _package_version("sacrebleu"),
    }


def build_artifact(
    *,
    predictions: Sequence[dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
    dataset: str,
    revision: str,
    component: str,
    split: str,
    source_lang: str,
    target_lang: str,
    requested_max_samples: int,
    seed: int,
    model_name: str,
    model_configuration: dict[str, Any],
    device: str,
    bytes_read: int,
    candidate_row_count: int,
    exclusions: Sequence[dict[str, str]],
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "dataset": dataset,
        "revision": revision,
        "component": component,
        "split": split,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "requested_max_samples": requested_max_samples,
        "actual_sample_count": len(predictions),
        "successful_hypothesis_count": sum(row.get("hypothesis") is not None for row in predictions),
        "seed": seed,
        "sampling_strategy": "random_without_replacement_from_bounded_prefix",
        "candidate_row_count": candidate_row_count,
        "bytes_read": bytes_read,
        "sample_row_locators": [row["row_locator"] for row in predictions],
        "exclusions": list(exclusions),
        "per_sample": list(predictions),
        "metrics": metrics,
        "metric_configuration": metric_configuration(target_lang),
        "model_name": model_name,
        "model_configuration": model_configuration,
        "device": device,
        "rights_status": "unresolved",
        "evaluation_granularity": "paragraph",
        "document_level_evaluation": False,
        "context_used": False,
        "context_available": False,
        "sentence_alignment_available": False,
        "document_identity_available": False,
        "no_fabricated_document_ids": True,
        "inference_performed": any(
            row.get("inference_status") in {"succeeded", "failed"} for row in predictions
        ),
        "no_inference": not any(
            row.get("inference_status") in {"succeeded", "failed"} for row in predictions
        ),
        "metrics_computed": any(result.get("status") == "computed" for result in metrics.values()),
    }


def evaluate_records(
    records: Sequence[IndicDiscoParagraphRecord],
    *,
    max_samples: int,
    seed: int,
    source_lang: str,
    target_lang: str,
    translation_service: Any,
    metric_computer: MetricComputer = _corpus_metric,
) -> tuple[list[IndicDiscoParagraphRecord], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    sampled = select_bounded_sample(records, max_samples=max_samples, seed=seed)
    predictions = translate_paragraphs(
        sampled,
        translation_service=translation_service,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    metrics = compute_paragraph_metrics(
        predictions,
        source_lang=source_lang,
        target_lang=target_lang,
        metric_computer=metric_computer,
    )
    return sampled, predictions, metrics


def _load_records(args: argparse.Namespace):
    if (args.source_lang, args.target_lang) != ("te", "en"):
        raise ValueError("IndicDISCO-MT core release direction is Telugu->English (`te`->`en`).")
    source = classify_source(
        args.source,
        revision=args.revision,
        component=args.component,
        split=args.split,
    )
    source = verify_local_source(source, split=args.split)
    parsed, bytes_read = read_bounded_csv_prefix(
        source,
        max_rows=args.max_candidate_rows,
        max_bytes=args.max_bytes,
    )
    adapter = IndicDiscoMTTrustedAdapter()
    records: list[IndicDiscoParagraphRecord] = []
    exclusions: list[dict[str, str]] = []
    for row_index, values in enumerate(parsed.rows):
        locator = f"{source.location}#row={row_index}"
        if not source.pinned_official:
            exclusions.append({"row_locator": locator, "reason": "source_not_verified_against_pinned_commit"})
            continue
        if parsed.header != ("source", "target"):
            exclusions.append({"row_locator": locator, "reason": "unexpected_core_paragraph_schema"})
            continue
        if len(values) != 2:
            exclusions.append({"row_locator": locator, "reason": "row_column_count_does_not_match_header"})
            continue
        if not values[0].strip():
            exclusions.append({"row_locator": locator, "reason": "empty_source_paragraph"})
            continue
        if not values[1].strip():
            exclusions.append({"row_locator": locator, "reason": "empty_reference_paragraph"})
            continue
        try:
            record = adapter.adapt_row(
                row=dict(zip(parsed.header, values)),
                revision=args.revision,
                split=args.split,
                component=args.component,
                row_locator=locator,
                source_location=f"{source.location}#row={row_index}:source",
                reference_location=f"{source.location}#row={row_index}:target",
                alignment_location=f"{source.location}#row={row_index}:source,target",
                source_language=INDICDISCO_SOURCE_LANGUAGE,
                target_language=INDICDISCO_TARGET_LANGUAGE,
            )
        except (TypeError, ValueError) as exc:
            exclusions.append({"row_locator": locator, "reason": f"adapter_rejected: {exc}"})
            continue
        if not metric_eligibility(record, IndicDiscoMetric.PARAGRAPH_BLEU).eligible or not metric_eligibility(record, IndicDiscoMetric.PARAGRAPH_CHRF).eligible:
            exclusions.append({"row_locator": locator, "reason": "paragraph_metric_ineligible"})
            continue
        records.append(record)
    return records, exclusions, len(parsed.rows), bytes_read


def _runtime_configuration(settings: Settings) -> tuple[str, dict[str, Any], str]:
    if settings.TRANSLATION_MAX_OUTPUT_LENGTH != EXPECTED_MAX_NEW_TOKENS:
        raise ValueError(
            "NLLB TRANSLATION_MAX_OUTPUT_LENGTH must remain 256 for this evaluation; "
            "restore the existing setting rather than changing generation behavior."
        )
    checkpoint = settings.MODEL_NLLB200_LOCAL_PATH or settings.MODEL_NLLB200
    generation_config = _checkpoint_generation_defaults(settings)
    model_configuration = {
        "model": "nllb200",
        "checkpoint": checkpoint,
        "input_max_length": settings.TRANSLATION_MAX_INPUT_LENGTH,
        "max_new_tokens": settings.TRANSLATION_MAX_OUTPUT_LENGTH,
        "max_length": EXPECTED_MAX_LENGTH,
        "max_length_semantics": "existing checkpoint/default metadata; NLLB service explicitly passes max_new_tokens",
        "do_sample": False,
        "num_beams": 1,
        "checkpoint_generation_config": generation_config,
        "input_handling": "complete paragraph passed to baseline service; service retains its existing sentence segmentation and joins sentence translations back into one paragraph hypothesis; no preceding context",
    }
    return checkpoint, model_configuration, _device_name(settings)


def run_from_args(args: argparse.Namespace) -> Path:
    records, exclusions, candidate_count, bytes_read = _load_records(args)
    if len(records) > args.max_candidate_rows:
        raise ValueError("bounded loader returned more rows than the requested candidate limit.")
    sampled = select_bounded_sample(records, max_samples=args.max_samples, seed=args.seed)

    settings = getattr(args, "settings", None) or get_settings()
    model_name, model_configuration, device = _runtime_configuration(settings)
    if sampled:
        translation_service = getattr(args, "translation_service", None) or TranslationService(settings=settings)
        sampled, predictions, metrics = evaluate_records(
            sampled,
            max_samples=len(sampled),
            seed=args.seed,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            translation_service=translation_service,
            metric_computer=getattr(args, "metric_computer", None) or _corpus_metric,
        )
    else:
        predictions = []
        metrics = compute_paragraph_metrics(
            [],
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            metric_computer=getattr(args, "metric_computer", None) or _corpus_metric,
        )

    artifact = build_artifact(
        predictions=predictions,
        metrics=metrics,
        dataset=INDICDISCO_DATASET,
        revision=args.revision,
        component=args.component,
        split=args.split,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        requested_max_samples=args.max_samples,
        seed=args.seed,
        model_name=model_name,
        model_configuration=model_configuration,
        device=device,
        bytes_read=bytes_read,
        candidate_row_count=candidate_count,
        exclusions=exclusions,
    )
    run_dir = Path(args.output_dir).expanduser().resolve() / artifact["run_id"]
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = run_dir / "paragraph_evaluation.json"
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return artifact_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded IndicDISCO-MT paragraph evaluation; running this command performs NLLB inference and corpus metrics."
    )
    parser.add_argument("--source", required=True, help="Exact pinned official HF CSV URL or explicit local CSV path.")
    parser.add_argument("--revision", default=INDICDISCO_REVISION)
    parser.add_argument("--component", default=INDICDISCO_PARAGRAPH_COMPONENT)
    parser.add_argument("--split", required=True, choices=("train", "dev", "test"))
    parser.add_argument("--source-lang", required=True, choices=("te",))
    parser.add_argument("--target-lang", required=True, choices=("en",))
    parser.add_argument("--max-samples", type=positive_int, default=3)
    parser.add_argument("--seed", type=nonnegative_int, default=42)
    parser.add_argument("--max-candidate-rows", type=positive_int, default=DEFAULT_MAX_CANDIDATE_ROWS)
    parser.add_argument("--max-bytes", type=positive_int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        artifact_path = run_from_args(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": "complete", "artifact": str(artifact_path)}, indent=2))
    return 0


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
