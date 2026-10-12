"""Run a bounded baseline evaluation through the trusted Samanantar adapter."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.evaluation.document_evaluation import (  # noqa: E402
    AlignmentEvidenceKind,
    AlignmentStatus,
    ContextStatus,
    EvaluationRecord,
    EvaluationTrack,
    ProvenanceStatus,
    ReferenceAvailability,
    TextUnit,
)
from app.evaluation.experiment_runner import run_evaluation  # noqa: E402
from app.evaluation.samanantar_adapter import (  # noqa: E402
    DATASET_NAME,
    SamanantarPreparedAdapter,
    SamanantarSample,
    SamanantarSplit,
)


@dataclass(frozen=True)
class BridgeRow:
    runner_record: dict[str, object]
    original_line_index: int
    record_id: int
    source_location: str
    reference_location: str
    alignment_location: str


def bridge_samanantar_records(sample: SamanantarSample) -> tuple[BridgeRow, ...]:
    """Convert trusted sentence pairs to the existing runner's JSONL contract."""
    if sample.split is not SamanantarSplit.TEST:
        raise ValueError("the Phase 12B smoke bridge accepts only the test split.")
    if not sample.records:
        raise ValueError("the trusted adapter returned no accepted records.")
    if sample.dataset_name != DATASET_NAME:
        raise ValueError("sample dataset is not AI4Bharat Samanantar.")
    if (
        sample.revision_scope != "upstream_revision"
        or sample.upstream_revision is None
        or sample.dataset_revision != sample.upstream_revision
    ):
        raise ValueError("sample does not carry a distinct verified upstream revision.")
    if not sample.preprocessing_run_id or sample.preprocessing_seed < 0:
        raise ValueError("sample preprocessing provenance is incomplete.")

    rows: list[BridgeRow] = []
    seen_ids: set[int] = set()
    seen_lines: set[int] = set()
    for record in sample.records:
        if not isinstance(record, EvaluationRecord):
            raise TypeError("sample records must be Phase 11 EvaluationRecord values.")
        provenance = record.provenance
        if provenance.status is not ProvenanceStatus.VERIFIED_DATASET:
            raise ValueError("Samanantar bridge rejects non-verified provenance.")
        if (
            provenance.dataset_name != DATASET_NAME
            or provenance.dataset_revision != sample.upstream_revision
            or provenance.split != sample.split.value
            or provenance.adapter_name != "samanantar_prepared_jsonl_v1"
        ):
            raise ValueError("record provenance does not match the Samanantar sample.")
        if record.track is not EvaluationTrack.SENTENCE_LEVEL:
            raise ValueError("Samanantar bridge accepts sentence_level records only.")
        if (
            record.current_text_unit is not TextUnit.SENTENCE
            or record.reference_unit is not TextUnit.SENTENCE
            or record.reference_availability is not ReferenceAvailability.AVAILABLE
            or record.alignment_status is not AlignmentStatus.SENTENCE_ALIGNED
            or record.alignment_evidence is None
            or record.alignment_evidence.kind
            is not AlignmentEvidenceKind.DATASET_SENTENCE_ANNOTATION
        ):
            raise ValueError("record lacks an aligned sentence-level reference.")
        if (
            record.source_language != "en"
            or record.target_language != "te"
            or not record.current_text.strip()
            or record.reference_translation is None
            or not record.reference_translation.strip()
        ):
            raise ValueError("record is not a complete English-Telugu sentence pair.")
        if (
            record.context_status is not ContextStatus.UNAVAILABLE
            or record.preceding_context
            or record.context_supplied_to_model
            or record.context_conditioned_translation is not None
            or record.document_or_passage_id is not None
            or provenance.unit_id is not None
        ):
            raise ValueError("Samanantar bridge rejects fabricated document/context metadata.")
        if provenance.reference_location is None or provenance.alignment_location is None:
            raise ValueError("record provenance is missing reference/alignment locations.")

        try:
            record_id = int(provenance.record_id or "")
        except ValueError as exc:
            raise ValueError("Samanantar record ID must preserve the prepared integer ID.") from exc
        if record_id < 0 or str(record_id) != provenance.record_id:
            raise ValueError("Samanantar record ID is malformed.")
        if record_id in seen_ids:
            raise ValueError("duplicate Samanantar record IDs cannot enter the bridge.")
        seen_ids.add(record_id)

        source_match = re.fullmatch(r"(.+)#L([1-9][0-9]*):source_text", provenance.source_location or "")
        reference_match = re.fullmatch(
            r"(.+)#L([1-9][0-9]*):target_text", provenance.reference_location
        )
        if (
            source_match is None
            or reference_match is None
            or source_match.group(1) != sample.source_file
            or reference_match.group(1) != sample.source_file
            or source_match.group(2) != reference_match.group(2)
        ):
            raise ValueError("source/reference locations do not identify one prepared JSONL row.")
        line_number = int(source_match.group(2))
        if line_number in seen_lines:
            raise ValueError("duplicate prepared source line locations cannot enter the bridge.")
        seen_lines.add(line_number)

        rows.append(
            BridgeRow(
                runner_record={
                    "id": record_id,
                    "source_language": "en",
                    "target_language": "te",
                    "source_text": record.current_text,
                    "target_text": record.reference_translation,
                    "data_source": "samanantar",
                },
                original_line_index=line_number - 1,
                record_id=record_id,
                source_location=provenance.source_location,
                reference_location=provenance.reference_location,
                alignment_location=provenance.alignment_location,
            )
        )
    return tuple(rows)


def run_bounded_samanantar_baseline(
    *,
    split: str,
    max_samples: int,
    seed: int,
    source_lang: str = "en",
    target_lang: str = "te",
    phase: str = "12B",
    adapter: SamanantarPreparedAdapter | None = None,
    evaluation_runner: Callable[..., Path] | None = None,
    output_root: Path | str | None = None,
) -> Path:
    phase_name = str(phase).strip().upper().replace("PHASE ", "")
    if phase_name == "12B":
        if split != "test":
            raise ValueError("Phase 12B requires the test split.")
        if seed != 42:
            raise ValueError("Phase 12B keeps the fixed seed=42 smoke contract.")
        if max_samples != 5:
            raise ValueError("Phase 12B keeps the fixed 5-example smoke contract.")
        return run_samanantar_baseline(
            split=split,
            max_samples=max_samples,
            seed=seed,
            source_lang=source_lang,
            target_lang=target_lang,
            adapter=adapter,
            evaluation_runner=evaluation_runner,
            output_root=output_root,
        )
    if phase_name == "12C":
        if split != "test":
            raise ValueError("Phase 12C supports only the sentence-level test split.")
        if source_lang != "en" or target_lang != "te":
            raise ValueError("Phase 12C supports only en->te.")
        if seed < 0:
            raise ValueError("seed must be non-negative.")
        if max_samples < 1:
            raise ValueError("max_samples must be at least 1.")
        return _run_samanantar_baseline_impl(
            split=split,
            max_samples=max_samples,
            seed=seed,
            source_lang=source_lang,
            target_lang=target_lang,
            adapter=adapter,
            evaluation_runner=evaluation_runner,
            output_root=output_root,
        )
    raise ValueError("phase must be '12B' or '12C'.")


def run_samanantar_baseline(
    *,
    split: str,
    max_samples: int,
    seed: int,
    source_lang: str = "en",
    target_lang: str = "te",
    adapter: SamanantarPreparedAdapter | None = None,
    evaluation_runner: Callable[..., Path] | None = None,
    output_root: Path | str | None = None,
) -> Path:
    return _run_samanantar_baseline_impl(
        split=split,
        max_samples=max_samples,
        seed=seed,
        source_lang=source_lang,
        target_lang=target_lang,
        adapter=adapter,
        evaluation_runner=evaluation_runner,
        output_root=output_root,
    )


def _run_samanantar_baseline_impl(
    *,
    split: str,
    max_samples: int,
    seed: int,
    source_lang: str,
    target_lang: str,
    adapter: SamanantarPreparedAdapter | None = None,
    evaluation_runner: Callable[..., Path] | None = None,
    output_root: Path | str | None = None,
) -> Path:
    if source_lang != "en" or target_lang != "te":
        raise ValueError("the Phase 12B Samanantar baseline smoke supports only en->te.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if max_samples < 1:
        raise ValueError("max_samples must be at least 1.")

    trusted_adapter = adapter or SamanantarPreparedAdapter()
    sample = trusted_adapter.load_sample(split=split, max_samples=max_samples)
    if sample.rows_inspected > max_samples:
        raise ValueError("adapter inspected more than max_samples rows.")
    if sample.accepted_count != max_samples:
        raise ValueError(
            f"expected {max_samples} accepted smoke records, got {sample.accepted_count}; "
            "refusing to run metrics on a partial sample."
        )
    bridge_rows = bridge_samanantar_records(sample)
    if len(bridge_rows) != max_samples:
        raise ValueError("bridge record count does not match the requested smoke sample.")

    evaluator = evaluation_runner or run_evaluation
    destination = Path(
        output_root or PROJECT_ROOT / "experiments" / "evaluation" / "samanantar_baseline_smoke"
    )
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="samanantar-baseline-bridge-") as temp_dir:
        bridge_path = Path(temp_dir) / "test.en-te.jsonl"
        bridge_path.write_text(
            "".join(
                json.dumps(row.runner_record, ensure_ascii=False, sort_keys=True) + "\n"
                for row in bridge_rows
            ),
            encoding="utf-8",
        )
        run_dir = evaluator(
            source_lang=source_lang,
            target_lang=target_lang,
            max_samples=max_samples,
            seed=seed,
            context_window=0,
            output_dir=destination,
            dataset_path=bridge_path,
            include_bertscore=False,
            include_comet=False,
        )
        _attach_samanantar_provenance(
            run_dir=Path(run_dir),
            sample=sample,
            bridge_rows=bridge_rows,
            max_samples=max_samples,
            seed=seed,
        )
        _require_completed_baseline(Path(run_dir), expected_count=max_samples)
    return Path(run_dir)


def _attach_samanantar_provenance(
    *,
    run_dir: Path,
    sample: SamanantarSample,
    bridge_rows: tuple[BridgeRow, ...],
    max_samples: int,
    seed: int,
) -> None:
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    experiment = summary.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("existing evaluator summary is missing experiment metadata.")

    dataset_metadata = experiment["dataset"]
    dataset_metadata.update(
        {
            "name": sample.dataset_name,
            "path": sample.source_file,
            "split": sample.split.value,
            "record_count": sample.expected_split_count,
            "sentence_level_only": True,
            "adjacent_records_assumed_related": False,
            "upstream_revision": sample.upstream_revision,
            "preprocessing_run_id": sample.preprocessing_run_id,
            "preprocessing_seed": sample.preprocessing_seed,
        }
    )
    experiment["phase"] = "12B" if max_samples == 5 and seed == 42 else "12C"
    experiment["evaluation_modes"] = ["baseline_sentence_level"]
    experiment["context_window"] = 0
    experiment["context_limitation"] = (
        "Samanantar is sentence-level. Context is unavailable; no document IDs or "
        "relationships are inferred from adjacent rows."
    )
    experiment["sampling"].update(
        {
            "strategy": "trusted_adapter_bounded_prefix",
            "seed": seed,
            "requested_max_samples": max_samples,
            "selected_sample_count": len(bridge_rows),
            "selected_indices": [row.original_line_index for row in bridge_rows],
            "selected_record_ids": [row.record_id for row in bridge_rows],
        }
    )
    config.update(experiment)
    summary["experiment"] = experiment
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    prediction_path = run_dir / "predictions.jsonl"
    predictions = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(predictions) != len(bridge_rows):
        raise ValueError("existing evaluator prediction count does not match bridged records.")
    for prediction, bridge_row in zip(predictions, bridge_rows):
        if prediction.get("record_id") != bridge_row.record_id:
            raise ValueError("existing evaluator changed a bridged Samanantar record ID.")
        prediction["sample_index"] = bridge_row.original_line_index
    prediction_path.write_text(
        "".join(
            json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n"
            for prediction in predictions
        ),
        encoding="utf-8",
    )

    sidecar = {
        "schema_version": 1,
        "run_id": experiment["run_id"],
        "dataset": sample.dataset_name,
        "upstream_revision": sample.upstream_revision,
        "upstream_source_url": sample.upstream_source_url,
        "preprocessing_run_id": sample.preprocessing_run_id,
        "preprocessing_seed": sample.preprocessing_seed,
        "prepared_files": dict(sample.prepared_files),
        "split": sample.split.value,
        "source_file": sample.source_file,
        "expected_split_count": sample.expected_split_count,
        "rows_inspected": sample.rows_inspected,
        "sample_count": len(bridge_rows),
        "sample_ids": [row.record_id for row in bridge_rows],
        "phase": "12B" if max_samples == 5 and seed == 42 else "12C",
        "source_language": "en",
        "target_language": "te",
        "evaluation_track": "sentence_level",
        "reference_granularity": "sentence",
        "context_availability": "unavailable",
        "context_supplied_to_model": False,
        "model_checkpoint": experiment["model_checkpoint"],
        "generation_settings": experiment["generation_settings"],
        "metric_methodology": experiment["metric_methodology"],
        "bridge_input": "ephemeral five-row JSONL using the existing evaluation runner schema",
        "records": [
            {
                "record_id": row.record_id,
                "original_line_index": row.original_line_index,
                "source_location": row.source_location,
                "reference_location": row.reference_location,
                "alignment_location": row.alignment_location,
                "track": "sentence_level",
                "unit": "sentence",
                "document_id": None,
                "context_available": False,
            }
            for row in bridge_rows
        ],
        "exclusions": [
            {"line_number": item.line_number, "record_id": item.record_id, "reason": item.reason}
            for item in sample.exclusions
        ],
    }
    (run_dir / "samanantar_provenance.json").write_text(
        json.dumps(sidecar, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _require_completed_baseline(run_dir: Path, *, expected_count: int) -> None:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    predictions = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(predictions) != expected_count or any(
        row.get("baseline_status") != "computed" for row in predictions
    ):
        raise RuntimeError(
            f"baseline smoke did not translate all {expected_count} records; artifact preserved at {run_dir}"
        )
    metrics = summary["metrics"]["baseline_sentence_level"]
    for metric_name in ("bleu", "chrf++"):
        metric = metrics.get(metric_name, {})
        if metric.get("status") != "computed" or metric.get("num_examples") != expected_count:
            raise RuntimeError(
                f"{metric_name} was not computed for all {expected_count} records; "
                f"artifact preserved at {run_dir}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded Samanantar English-to-Telugu baseline evaluation. "
            "Phase 12B keeps the fixed 5-example smoke contract; Phase 12C supports larger "
            "bounded sentence-level runs such as 25 or 50 examples, but they must not be "
            "presented as final model-quality results."
        )
    )
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-lang", choices=["en"], default="en")
    parser.add_argument("--target-lang", choices=["te"], default="te")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_samples < 1:
        parser.error("--max-samples must be at least 1.")
    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    phase = "12B" if args.max_samples == 5 and args.seed == 42 and args.split == "test" else "12C"
    try:
        run_dir = run_bounded_samanantar_baseline(
            split=args.split,
            max_samples=args.max_samples,
            seed=args.seed,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            phase=phase,
        )
    except (FileNotFoundError, ValueError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    metrics = summary["metrics"]["baseline_sentence_level"]
    print(
        json.dumps(
            {
                "status": "complete",
                "artifact_dir": str(run_dir.resolve()),
                "sample_count": summary["experiment"]["sampling"]["selected_sample_count"],
                "sample_ids": summary["experiment"]["sampling"]["selected_record_ids"],
                "bleu": metrics["bleu"],
                "chrf++": metrics["chrf++"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())