"""Read-only adapter for the project's prepared Samanantar sentence JSONL."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from app.evaluation.document_evaluation import (
    AlignmentEvidence,
    AlignmentEvidenceKind,
    AlignmentStatus,
    ContextStatus,
    DatasetProvenance,
    EvaluationRecord,
    EvaluationTrack,
    ProvenanceStatus,
    ReferenceAvailability,
    TextUnit,
    TrustedDatasetAdapter,
)

DATASET_NAME = "AI4Bharat Samanantar"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREPROCESSING_REPORT = PROJECT_ROOT / "experiments" / "dataset_preprocessing_report.json"
UPSTREAM_SOURCE_URL = "https://indicnlp.ai4bharat.org/samanantar/"
EXPECTED_PREPARED_FILES = {
    "train": "train.en-te.jsonl",
    "val": "val.en-te.jsonl",
    "test": "test.en-te.jsonl",
}
EXPECTED_FIELDS = {
    "id",
    "source_text",
    "target_text",
    "source_language",
    "target_language",
    "data_source",
}
CONTEXT_FIELDS = {
    "document_id",
    "document_or_passage_id",
    "sentence_id",
    "sentence_index",
    "current_unit_position",
    "context",
    "context_sentences",
}


class SamanantarSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True)
class AdapterExclusion:
    line_number: int
    record_id: str | None
    reason: str


@dataclass(frozen=True)
class SamanantarSample:
    dataset_name: str
    dataset_revision: str
    revision_scope: str
    upstream_revision: str | None
    upstream_source_url: str
    preprocessing_run_id: str
    preprocessing_seed: int
    prepared_files: tuple[tuple[str, str], ...]
    split: SamanantarSplit
    expected_split_count: int | None
    source_file: str
    rows_inspected: int
    records: tuple[EvaluationRecord, ...]
    exclusions: tuple[AdapterExclusion, ...]

    @property
    def accepted_count(self) -> int:
        return len(self.records)

    @property
    def excluded_count(self) -> int:
        return len(self.exclusions)


class SamanantarPreparedAdapter(TrustedDatasetAdapter):
    """Validate a bounded prefix of the already-prepared Samanantar split.

    The default data/report paths are fixed project paths. This class is
    read-only and does not invoke preprocessing, inference, or metrics. Tests
    may subclass it with temporary fixture paths.
    """

    @property
    def adapter_name(self) -> str:
        return "samanantar_prepared_jsonl_v1"

    @property
    def processed_dir(self) -> Path:
        return PROCESSED_DIR

    @property
    def preprocessing_report(self) -> Path:
        return PREPROCESSING_REPORT

    def load_sample(self, *, split: str | SamanantarSplit, max_samples: int) -> SamanantarSample:
        if not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples < 1:
            raise ValueError("max_samples must be a positive integer.")
        try:
            selected_split = SamanantarSplit(split)
        except ValueError as exc:
            raise ValueError("split must be train, val, or test.") from exc

        metadata = self._read_preprocessing_metadata()
        report_provenance = _parse_provenance(metadata)
        expected_count = _split_count(metadata, selected_split)
        prepared_files = dict(report_provenance.prepared_files)
        source_path = self.processed_dir / prepared_files[selected_split.value]
        if not source_path.is_file():
            raise FileNotFoundError(f"Prepared Samanantar split not found: {source_path}")

        records: list[EvaluationRecord] = []
        exclusions: list[AdapterExclusion] = []
        seen_ids: set[int] = set()
        rows_inspected = 0
        source_path_text = _source_location(source_path)

        with source_path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if rows_inspected >= max_samples:
                    break
                rows_inspected += 1
                if not line.strip():
                    exclusions.append(AdapterExclusion(line_number, None, "blank_line"))
                    continue
                try:
                    raw_record = json.loads(line)
                except json.JSONDecodeError:
                    exclusions.append(AdapterExclusion(line_number, None, "invalid_json"))
                    continue

                record_id, reason = validate_prepared_record(raw_record, seen_ids)
                if reason is not None:
                    exclusions.append(AdapterExclusion(line_number, record_id, reason))
                    continue
                assert isinstance(raw_record, dict)
                assert record_id is not None

                source_location = f"{source_path_text}#L{line_number}:source_text"
                reference_location = f"{source_path_text}#L{line_number}:target_text"
                alignment_location = (
                    f"{source_path_text}#L{line_number}:id,source_language,"
                    "target_language,source_text,target_text"
                )
                try:
                    provenance = self._provenance_for_record(
                        split=selected_split,
                        revision=report_provenance.upstream_revision,
                        record_id=record_id,
                        source_location=source_location,
                        reference_location=reference_location,
                        alignment_location=alignment_location,
                    )
                    evidence_kind = (
                        AlignmentEvidenceKind.SYNTHETIC_FIXTURE
                        if provenance.status is ProvenanceStatus.SYNTHETIC
                        else AlignmentEvidenceKind.DATASET_SENTENCE_ANNOTATION
                    )
                    records.append(
                        EvaluationRecord(
                            track=EvaluationTrack.SENTENCE_LEVEL,
                            source_language="en",
                            target_language="te",
                            current_text=raw_record["source_text"],
                            current_text_unit=TextUnit.SENTENCE,
                            alignment_status=AlignmentStatus.SENTENCE_ALIGNED,
                            reference_availability=ReferenceAvailability.AVAILABLE,
                            provenance=provenance,
                            reference_translation=raw_record["target_text"],
                            reference_unit=TextUnit.SENTENCE,
                            alignment_evidence=AlignmentEvidence(evidence_kind),
                            context_status=ContextStatus.UNAVAILABLE,
                        )
                    )
                except (TypeError, ValueError):
                    exclusions.append(
                        AdapterExclusion(line_number, str(record_id), "provenance_unavailable")
                    )

        return SamanantarSample(
            dataset_name=DATASET_NAME,
            dataset_revision=report_provenance.upstream_revision,
            revision_scope="upstream_revision",
            upstream_revision=report_provenance.upstream_revision,
            upstream_source_url=report_provenance.upstream_source_url,
            preprocessing_run_id=report_provenance.preprocessing_run_id,
            preprocessing_seed=report_provenance.preprocessing_seed,
            prepared_files=report_provenance.prepared_files,
            split=selected_split,
            expected_split_count=expected_count,
            source_file=_relative_or_absolute(source_path),
            rows_inspected=rows_inspected,
            records=tuple(records),
            exclusions=tuple(exclusions),
        )

    def _provenance_for_record(
        self,
        *,
        split: SamanantarSplit,
        revision: str,
        record_id: int,
        source_location: str,
        reference_location: str,
        alignment_location: str,
    ) -> DatasetProvenance:
        return self.verified_provenance(
            status=ProvenanceStatus.VERIFIED_DATASET,
            dataset_name=DATASET_NAME,
            dataset_revision=revision,
            split=split.value,
            record_id=str(record_id),
            source_location=source_location,
            reference_location=reference_location,
            alignment_location=alignment_location,
        )

    def _read_preprocessing_metadata(self) -> dict[str, Any]:
        if not self.preprocessing_report.is_file():
            raise FileNotFoundError(
                f"Samanantar preprocessing report not found: {self.preprocessing_report}"
            )
        try:
            with self.preprocessing_report.open("r", encoding="utf-8") as report_file:
                metadata = json.load(report_file)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError("Samanantar preprocessing report is unreadable.") from exc
        if not isinstance(metadata, dict):
            raise ValueError("Samanantar preprocessing report must be a JSON object.")
        if metadata.get("source_dataset") != DATASET_NAME:
            raise ValueError("preprocessing report does not identify AI4Bharat Samanantar.")
        if metadata.get("language_pair") != "en-te":
            raise ValueError("preprocessing report is not for the en-te pair.")
        if metadata.get("src_lang") != "en" or metadata.get("tgt_lang") != "te":
            raise ValueError("preprocessing report language fields are inconsistent.")
        _require_nonempty_string(metadata.get("generation_ts_utc"), "generation_ts_utc")
        seed = metadata.get("random_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("preprocessing report random_seed must be a non-negative integer.")
        _parse_provenance(metadata)
        return metadata


def validate_prepared_record(
    raw_record: object,
    seen_ids: set[int],
) -> tuple[int | None, str | None]:
    if not isinstance(raw_record, dict):
        return None, "record_not_object"
    if CONTEXT_FIELDS.intersection(raw_record):
        return _valid_record_id(raw_record.get("id")), "unexpected_document_context_fields"

    record_id = _valid_record_id(raw_record.get("id"))
    if record_id is None:
        return None, "missing_or_invalid_record_id"
    if record_id in seen_ids:
        return record_id, "duplicate_record_id"
    seen_ids.add(record_id)

    if not isinstance(raw_record.get("source_text"), str) or not raw_record["source_text"].strip():
        return record_id, "missing_or_invalid_source_text"
    if not isinstance(raw_record.get("target_text"), str) or not raw_record["target_text"].strip():
        return record_id, "missing_or_invalid_reference_text"
    if raw_record.get("source_language") != "en" or raw_record.get("target_language") != "te":
        return record_id, "invalid_language_pair"
    if not isinstance(raw_record.get("data_source"), str) or not raw_record["data_source"].strip():
        return record_id, "invalid_data_source"
    if set(raw_record) != EXPECTED_FIELDS:
        return record_id, "missing_or_unexpected_fields"
    return record_id, None


@dataclass(frozen=True)
class _ProvenanceMetadata:
    upstream_revision: str
    upstream_source_url: str
    preprocessing_run_id: str
    preprocessing_seed: int
    prepared_files: tuple[tuple[str, str], ...]


def _parse_provenance(metadata: dict[str, Any]) -> _ProvenanceMetadata:
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("preprocessing report provenance is missing or malformed.")
    upstream = provenance.get("upstream")
    preprocessing = provenance.get("preprocessing")
    if not isinstance(upstream, dict) or not isinstance(preprocessing, dict):
        raise ValueError("upstream/preprocessing provenance blocks are required.")

    if upstream.get("dataset_name") != DATASET_NAME:
        raise ValueError("upstream provenance dataset_name is invalid.")
    source_url = upstream.get("source_url")
    if source_url != UPSTREAM_SOURCE_URL:
        raise ValueError("upstream provenance source_url is missing or invalid.")
    revision = upstream.get("revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("upstream provenance revision is missing or malformed.")
    if (
        upstream.get("language_pair") != "en-te"
        or upstream.get("source_language") != "en"
        or upstream.get("target_language") != "te"
    ):
        raise ValueError("upstream provenance language pair is invalid.")

    run_id = preprocessing.get("run_id")
    seed = preprocessing.get("seed")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("preprocessing provenance run_id is missing or malformed.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("preprocessing provenance seed is missing or malformed.")
    if seed != metadata.get("random_seed"):
        raise ValueError("preprocessing provenance seed disagrees with report random_seed.")
    expected_run_id = f"samanantar-en-te-{metadata['generation_ts_utc']}-seed-{seed}"
    if run_id != expected_run_id:
        raise ValueError("preprocessing provenance run_id is inconsistent with report timestamp/seed.")

    prepared = preprocessing.get("prepared_files")
    if not isinstance(prepared, dict) or prepared != EXPECTED_PREPARED_FILES:
        raise ValueError("preprocessing provenance prepared_files is missing or malformed.")
    return _ProvenanceMetadata(
        upstream_revision=revision,
        upstream_source_url=source_url,
        preprocessing_run_id=run_id,
        preprocessing_seed=seed,
        prepared_files=tuple((name, prepared[name]) for name in ("train", "val", "test")),
    )


def _split_count(metadata: dict[str, Any], split: SamanantarSplit) -> int | None:
    field_name = {
        SamanantarSplit.TRAIN: "train_count",
        SamanantarSplit.VAL: "validation_count",
        SamanantarSplit.TEST: "test_count",
    }[split]
    value = metadata.get(field_name)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _source_location(path: Path) -> str:
    return _relative_or_absolute(path)


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _valid_record_id(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _require_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"preprocessing report {field_name} must be a non-empty string.")
