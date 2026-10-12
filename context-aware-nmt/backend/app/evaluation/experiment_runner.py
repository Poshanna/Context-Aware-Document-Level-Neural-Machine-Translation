"""Reproducible evaluation over an existing processed JSONL test split."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.evaluation import registry as metric_registry
from app.services.context_conditioned_translation_service import (
    ContextConditionedTranslationService,
)
from app.services.translation_service import TranslationService

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUPPORTED_PAIRS = {("en", "te"), ("te", "en")}


def validate_evaluation_args(
    *, source_lang: str, target_lang: str, max_samples: int, context_window: int, seed: int
) -> None:
    if (source_lang, target_lang) not in SUPPORTED_PAIRS:
        raise ValueError("Evaluation supports only en->te and te->en.")
    if max_samples < 1:
        raise ValueError("max_samples must be at least 1.")
    if not 0 <= context_window <= 200:
        raise ValueError("context_window must be between 0 and 200.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")


def seed_evaluation(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _orient_record(
    record: dict[str, Any], source_lang: str, target_lang: str, line_index: int
) -> tuple[str, str]:
    source_text = record.get("source_text")
    target_text = record.get("target_text")
    record_source_lang = record.get("source_language", "en")
    record_target_lang = record.get("target_language", "te")
    if not isinstance(source_text, str) or not isinstance(target_text, str):
        raise TypeError(f"Missing source_text/target_text at physical line {line_index + 1}.")
    if (record_source_lang, record_target_lang) == (source_lang, target_lang):
        return source_text, target_text
    if (record_source_lang, record_target_lang) == (target_lang, source_lang):
        return target_text, source_text
    raise ValueError(
        f"Record at physical line {line_index + 1} has unsupported language pair "
        f"{record_source_lang!r}->{record_target_lang!r}."
    )


def load_reproducible_sample(
    path: Path | str,
    *,
    max_samples: int,
    seed: int,
    source_lang: str,
    target_lang: str,
) -> tuple[list[dict[str, Any]], int]:
    """Reservoir-sample JSONL rows once, without modifying the source split."""
    validate_evaluation_args(
        source_lang=source_lang,
        target_lang=target_lang,
        max_samples=max_samples,
        context_window=0,
        seed=seed,
    )
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    record_count = 0
    with Path(path).open("r", encoding="utf-8") as source_file:
        for line_index, line in enumerate(source_file):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at physical line {line_index + 1}.") from exc
            if not isinstance(record, dict):
                raise TypeError(f"Expected a JSON object at physical line {line_index + 1}.")
            _orient_record(record, source_lang, target_lang, line_index)
            entry = {"line_index": line_index, "record": record}
            if record_count < max_samples:
                reservoir.append(entry)
            else:
                replacement = rng.randrange(record_count + 1)
                if replacement < max_samples:
                    reservoir[replacement] = entry
            record_count += 1
    reservoir.sort(key=lambda item: item["line_index"])
    return reservoir, record_count


def legitimate_context(
    record: dict[str, Any], *, source_lang: str, target_lang: str, context_window: int
) -> tuple[list[str], list[int]]:
    """Return only explicit same-document context strictly before this sentence."""
    document_id = record.get("document_id")
    sentence_index = record.get("sentence_index")
    raw_context = record.get("context")
    if (
        not isinstance(document_id, str)
        or not document_id.strip()
        or not isinstance(sentence_index, int)
        or isinstance(sentence_index, bool)
        or not isinstance(raw_context, list)
        or context_window == 0
    ):
        return [], []

    eligible: list[tuple[int, str]] = []
    seen_indices: set[int] = set()
    for context_item in raw_context:
        if not isinstance(context_item, dict):
            continue
        context_index = context_item.get("sentence_index")
        if (
            context_item.get("document_id") != document_id
            or not isinstance(context_index, int)
            or isinstance(context_index, bool)
            or context_index < 0
            or context_index >= sentence_index
            or context_index in seen_indices
        ):
            continue
        context_text, _ = _orient_record(
            context_item, source_lang, target_lang, context_index
        )
        if context_text.strip():
            eligible.append((context_index, context_text))
            seen_indices.add(context_index)
    eligible.sort(key=lambda item: item[0])
    eligible = eligible[-context_window:]
    return [text for _, text in eligible], [index for index, _ in eligible]


def _software_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in ("sacrebleu", "transformers", "torch", "numpy", "bert-score", "unbabel-comet"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _device_name(settings: Settings) -> str:
    if settings.FORCE_DEVICE in {"cpu", "cuda"}:
        return settings.FORCE_DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "unknown"


def _checkpoint_generation_defaults(settings: Settings) -> dict[str, Any]:
    checkpoint = settings.MODEL_NLLB200_LOCAL_PATH or settings.MODEL_NLLB200
    try:
        if settings.MODEL_NLLB200_LOCAL_PATH:
            config_path = Path(settings.MODEL_NLLB200_LOCAL_PATH) / "generation_config.json"
            if config_path.is_file():
                return json.loads(config_path.read_text(encoding="utf-8"))
        from huggingface_hub import try_to_load_from_cache

        cached_path = try_to_load_from_cache(checkpoint, "generation_config.json")
        if isinstance(cached_path, str) and Path(cached_path).is_file():
            return json.loads(Path(cached_path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - metadata failure must not block evaluation
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    return {"status": "unavailable", "reason": "generation_config.json not present in local cache"}


def _corpus_metric(
    name: str, sources: list[str], hypotheses: list[str], references: list[str],
    source_lang: str, target_lang: str,
) -> dict[str, Any]:
    if not hypotheses:
        return {"status": "unavailable", "score": None, "reason": "no eligible predictions"}
    if importlib.util.find_spec("sacrebleu") is None:
        return {"status": "unavailable", "score": None, "reason": "sacrebleu is not installed"}
    try:
        metric = metric_registry.new(name)
        result = metric.compute(
            sources, hypotheses, references, src_lang=source_lang, tgt_lang=target_lang
        )
        return {
            "status": "computed",
            "score": result.aggregate,
            "details": result.details or {},
            "num_examples": len(hypotheses),
        }
    except Exception as exc:  # noqa: BLE001 - report metric-library failures in the artifact
        return {"status": "failed", "score": None, "reason": f"{type(exc).__name__}: {exc}"}


def _bertscore(
    hypotheses: list[str], references: list[str], settings: Settings, device: str, enabled: bool
) -> dict[str, Any]:
    if not enabled:
        return {"status": "unavailable", "score": None, "reason": "not requested"}
    if not hypotheses:
        return {"status": "unavailable", "score": None, "reason": "no eligible predictions"}
    if importlib.util.find_spec("bert_score") is None:
        return {"status": "unavailable", "score": None, "reason": "bert-score is not installed"}
    try:
        from bert_score import score

        precision, recall, f1 = score(
            hypotheses, references, model_type=settings.BERT_SCORE_MODEL,
            device=device, verbose=False,
        )
        return {
            "status": "computed",
            "score": float(f1.mean().item()),
            "num_examples": len(hypotheses),
            "details": {
                "model": settings.BERT_SCORE_MODEL,
                "precision_mean": float(precision.mean().item()),
                "recall_mean": float(recall.mean().item()),
                "f1_mean": float(f1.mean().item()),
            },
        }
    except Exception as exc:  # noqa: BLE001 - report optional metric failures in the artifact
        return {"status": "failed", "score": None, "reason": f"{type(exc).__name__}: {exc}"}


def _comet(
    sources: list[str], hypotheses: list[str], references: list[str],
    settings: Settings, device: str, enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"status": "unavailable", "score": None, "reason": "not requested"}
    if not settings.ENABLE_COMET:
        return {"status": "unavailable", "score": None, "reason": "ENABLE_COMET is disabled"}
    if not hypotheses:
        return {"status": "unavailable", "score": None, "reason": "no eligible predictions"}
    if importlib.util.find_spec("comet") is None:
        return {"status": "unavailable", "score": None, "reason": "unbabel-comet is not installed"}
    try:
        from comet import download_model, load_from_checkpoint

        checkpoint = download_model(settings.COMET_MODEL)
        model = load_from_checkpoint(checkpoint)
        prediction = model.predict(
            [
                {"src": source, "mt": hypothesis, "ref": reference}
                for source, hypothesis, reference in zip(sources, hypotheses, references)
            ],
            batch_size=8,
            gpus=1 if device == "cuda" else 0,
        )
        return {
            "status": "computed",
            "score": float(prediction.system_score),
            "num_examples": len(hypotheses),
            "details": {"model": settings.COMET_MODEL},
        }
    except Exception as exc:  # noqa: BLE001 - report optional metric failures in the artifact
        return {"status": "failed", "score": None, "reason": f"{type(exc).__name__}: {exc}"}


def run_evaluation(
    *, source_lang: str = "en", target_lang: str = "te", max_samples: int = 50,
    seed: int = 42, context_window: int = 1,
    output_dir: Path | str | None = None, dataset_path: Path | str | None = None,
    settings: Settings | None = None, translation_service: Any | None = None,
    context_service: Any | None = None, include_bertscore: bool = False,
    include_comet: bool = False,
) -> Path:
    validate_evaluation_args(
        source_lang=source_lang, target_lang=target_lang, max_samples=max_samples,
        context_window=context_window, seed=seed,
    )
    seed_evaluation(seed)
    settings = settings or get_settings()
    dataset_path = Path(dataset_path or PROJECT_ROOT / "data/processed/test.en-te.jsonl")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Processed test split not found: {dataset_path}")
    sample, total_records = load_reproducible_sample(
        dataset_path, max_samples=max_samples, seed=seed,
        source_lang=source_lang, target_lang=target_lang,
    )
    if not sample:
        raise ValueError(f"Processed test split contains no records: {dataset_path}")

    translation_service = translation_service or TranslationService(settings=settings)
    context_service = context_service or ContextConditionedTranslationService(
        translation_service=translation_service, settings=settings,
    )
    from app.schemas import TextTranslateIn

    predictions: list[dict[str, Any]] = []
    for selected in sample:
        record = selected["record"]
        source, reference = _orient_record(record, source_lang, target_lang, selected["line_index"])
        context_sentences, context_indices = legitimate_context(
            record, source_lang=source_lang, target_lang=target_lang,
            context_window=context_window,
        )
        row: dict[str, Any] = {
            "sample_index": selected["line_index"],
            "record_id": record.get("id"),
            "source": source,
            "reference": reference,
            "baseline_translation": None,
            "context_conditioned_translation": None,
            "context_available": bool(context_sentences),
            "context_sentences": context_sentences,
            "context_indices": context_indices,
            "baseline_status": "pending",
            "status": "context_conditioned" if context_sentences else "context_conditioned_not_available",
        }
        try:
            response = translation_service.translate_text(
                user_id=None,
                req=TextTranslateIn(
                    source_lang=source_lang, target_lang=target_lang,
                    pipeline="baseline", model="nllb200", text=source,
                ),
            )
            if response.status != "ok" or response.translated_text is None:
                raise RuntimeError(response.note or "Baseline translation returned no output.")
            row["baseline_translation"] = response.translated_text
            row["baseline_status"] = "computed"
        except Exception as exc:  # noqa: BLE001 - persist backend failures per example
            row["baseline_status"] = "failed"
            row["baseline_error"] = f"{type(exc).__name__}: {exc}"

        if context_sentences:
            try:
                response = context_service.translate_sentence(
                    source_lang=source_lang, target_lang=target_lang,
                    current_sentence=source, preceding_context=context_sentences,
                    context_window=context_window, model="nllb200",
                )
                row["context_conditioned_translation"] = response["translation"]
                row["context_conditioned_model_input"] = response["formatted_input"]
                row["status"] = "context_conditioned"
            except Exception as exc:  # noqa: BLE001 - persist backend failures per example
                row["status"] = "failed"
                row["context_conditioned_error"] = f"{type(exc).__name__}: {exc}"
        predictions.append(row)

    baseline_rows = [row for row in predictions if row["baseline_status"] == "computed"]
    context_rows = [
        row for row in predictions
        if row["context_available"] and row["context_conditioned_translation"] is not None
    ]
    device = _device_name(settings)
    metric_registry.populate()

    def metric_set(rows: list[dict[str, Any]], translation_field: str, *, baseline: bool) -> dict[str, Any]:
        if baseline and len(rows) != len(predictions):
            return {
                name: {"status": "failed", "score": None, "reason": "one or more baseline translations failed"}
                for name in ("bleu", "chrf++", "bertscore", "comet")
            }
        sources = [row["source"] for row in rows]
        hypotheses = [row[translation_field] for row in rows]
        references = [row["reference"] for row in rows]
        results = {
            name: _corpus_metric(name, sources, hypotheses, references, source_lang, target_lang)
            for name in ("bleu", "chrf++")
        }
        results["bertscore"] = _bertscore(
            hypotheses, references, settings, device, include_bertscore
        )
        results["comet"] = _comet(
            sources, hypotheses, references, settings, device, include_comet
        )
        return results

    metrics = {
        "baseline_sentence_level": metric_set(
            baseline_rows, "baseline_translation", baseline=True
        ),
        "context_conditioned": metric_set(
            context_rows, "context_conditioned_translation", baseline=False
        ),
    }
    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = Path(output_dir or PROJECT_ROOT / "experiments/evaluation") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = settings.MODEL_NLLB200_LOCAL_PATH or settings.MODEL_NLLB200
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": timestamp,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "model_checkpoint": checkpoint,
        "context_strategy": "explicit_document_context_only",
        "context_window": context_window,
        "device": device,
        "generation_settings": {
            "max_input_length": settings.TRANSLATION_MAX_INPUT_LENGTH,
            "max_new_tokens": settings.TRANSLATION_MAX_OUTPUT_LENGTH,
            "do_sample": False,
            "num_beams": 1,
            "seed": seed,
            "checkpoint_generation_config": _checkpoint_generation_defaults(settings),
            "note": "The backend passes max_new_tokens; this takes precedence over checkpoint max_length when both are set.",
        },
        "software_versions": _software_versions(),
        "dataset": {
            "name": "AI4Bharat Samanantar", "path": str(dataset_path.resolve()),
            "split": "test", "record_count": total_records,
            "sentence_level_only": True, "adjacent_records_assumed_related": False,
        },
        "sampling": {
            "strategy": "reservoir_sampling_without_replacement", "seed": seed,
            "requested_max_samples": max_samples, "selected_sample_count": len(sample),
            "selected_indices": [item["line_index"] for item in sample],
            "selected_record_ids": [item["record"].get("id") for item in sample],
        },
        "evaluation_modes": [
            "baseline_sentence_level", "context_conditioned", "context_conditioned_not_available",
        ],
        "context_limitation": (
            "Samanantar is sentence-level. Context is used only for records with explicit document IDs "
            "and prior same-document sentence records; adjacency is never treated as context."
        ),
        "metric_methodology": {
            "normalization": "No additional normalization; source/reference strings are preserved.",
            "bleu": "SacreBLEU corpus BLEU; tokenizer 13a for English target, none for Telugu target; unsmoothed.",
            "chrf++": "SacreBLEU corpus chrF++; char_order=6, word_order=2, beta=2; other options use library defaults.",
            "bertscore": {"requested": include_bertscore, "model": settings.BERT_SCORE_MODEL},
            "comet": {"requested": include_comet, "enabled_setting": bool(settings.ENABLE_COMET), "model": settings.COMET_MODEL},
        },
    }

    def write_json(name: str, payload: Any) -> None:
        (run_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    write_json("summary.json", {"experiment": metadata, "metrics": metrics})
    write_json("config.json", metadata)
    write_json("metrics.json", metrics)
    write_json("system_info.json", {"device": device, "software_versions": metadata["software_versions"]})
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as output:
        for row in predictions:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (run_dir / "README.md").write_text(
        f"# Evaluation Run\n\nRun `{run_id}` evaluates {len(sample)} sampled Samanantar test sentences "
        f"for `{source_lang}->{target_lang}`. See `summary.json` for metric status and scores, and "
        "`predictions.jsonl` for per-example results. Samanantar is sentence-level; adjacent rows "
        "are not assumed to be documents. Context-conditioned outputs require explicit, prior, "
        "same-document context; no context relationship is inferred from row order.\n",
        encoding="utf-8",
    )
    return run_dir