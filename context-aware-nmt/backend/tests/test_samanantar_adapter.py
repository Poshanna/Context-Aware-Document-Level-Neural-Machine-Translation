from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.evaluation.document_evaluation import (
    AlignmentEvidenceKind,
    ContextStatus,
    DatasetProvenance,
    EvaluationTrack,
    MetricName,
    ProvenanceStatus,
    ReferenceAvailability,
    TextUnit,
    metric_eligibility,
)
from app.evaluation.samanantar_adapter import SamanantarPreparedAdapter

PINNED_SAMANANTAR_REVISION = "ead34c5b22a0354d350e7dfdd6aece7df5a48f37"


def _fixture_row(record_id: int = 41) -> dict[str, object]:
    return {
        "id": record_id,
        "source_text": "Synthetic English sentence.",
        "target_text": "కృత్రిమ తెలుగు వాక్యం.",
        "source_language": "en",
        "target_language": "te",
        "data_source": "samanantar",
    }


class _SyntheticFileAdapter(SamanantarPreparedAdapter):
    def __init__(self, processed_dir: Path, report_path: Path) -> None:
        self._fixture_processed_dir = processed_dir
        self._fixture_report_path = report_path

    @property
    def processed_dir(self) -> Path:
        return self._fixture_processed_dir

    @property
    def preprocessing_report(self) -> Path:
        return self._fixture_report_path


def _write_fixture(
    tmp_path: Path,
    rows: list[object],
    *,
    split: str = "test",
    upstream_revision: object = PINNED_SAMANANTAR_REVISION,
) -> SamanantarPreparedAdapter:
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    with (processed_dir / f"{split}.en-te.jsonl").open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    report_path = tmp_path / "experiments" / "dataset_preprocessing_report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "source_dataset": "AI4Bharat Samanantar",
                "language_pair": "en-te",
                "src_lang": "en",
                "tgt_lang": "te",
                "generation_ts_utc": "2026-10-01T01:02:03Z",
                "random_seed": 17,
                "train_count": 10,
                "validation_count": 4,
                "test_count": len(rows),
                "provenance": {
                    "upstream": {
                        "dataset_name": "AI4Bharat Samanantar",
                        "source_url": "https://indicnlp.ai4bharat.org/samanantar/",
                        "revision": upstream_revision,
                        "language_pair": "en-te",
                        "source_language": "en",
                        "target_language": "te",
                    },
                    "preprocessing": {
                        "run_id": "samanantar-en-te-2026-10-01T01:02:03Z-seed-17",
                        "seed": 17,
                        "prepared_files": {
                            "train": "train.en-te.jsonl",
                            "val": "val.en-te.jsonl",
                            "test": "test.en-te.jsonl",
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return _SyntheticFileAdapter(processed_dir, report_path)


def test_valid_sentence_pair_gets_adapter_issued_provenance(tmp_path: Path) -> None:
    sample = _write_fixture(tmp_path, [_fixture_row()]).load_sample(
        split="test", max_samples=1
    )

    assert sample.accepted_count == 1
    record = sample.records[0]
    assert record.track is EvaluationTrack.SENTENCE_LEVEL
    assert record.current_text_unit is TextUnit.SENTENCE
    assert record.provenance.status is ProvenanceStatus.VERIFIED_DATASET
    assert record.provenance.dataset_name == "AI4Bharat Samanantar"
    assert record.provenance.record_id == "41"


@pytest.mark.parametrize(
    ("changes", "expected_reason"),
    [
        ({"source_text": ""}, "missing_or_invalid_source_text"),
        ({"target_text": None}, "missing_or_invalid_reference_text"),
        ({"id": None}, "missing_or_invalid_record_id"),
        ({"source_language": "te"}, "invalid_language_pair"),
        ({"target_language": "en"}, "invalid_language_pair"),
    ],
)
def test_invalid_required_fields_are_excluded_deterministically(
    changes: dict[str, object], expected_reason: str, tmp_path: Path
) -> None:
    row = _fixture_row()
    row.update(changes)
    sample = _write_fixture(tmp_path, [row]).load_sample(split="test", max_samples=1)

    assert sample.accepted_count == 0
    assert sample.exclusions[0].reason == expected_reason


def test_malformed_json_is_excluded_without_repair(tmp_path: Path) -> None:
    adapter = _write_fixture(tmp_path, [])
    with (tmp_path / "data" / "processed" / "test.en-te.jsonl").open(
        "w", encoding="utf-8"
    ) as output:
        output.write('{"id":\n')

    sample = adapter.load_sample(split="test", max_samples=1)

    assert sample.records == ()
    assert sample.exclusions[0].reason == "invalid_json"


def test_provenance_contains_upstream_and_local_run_locations(
    tmp_path: Path,
) -> None:
    sample = _write_fixture(tmp_path, [_fixture_row(99)]).load_sample(
        split="test", max_samples=1
    )
    provenance = sample.records[0].provenance

    assert sample.dataset_revision == PINNED_SAMANANTAR_REVISION
    assert sample.revision_scope == "upstream_revision"
    assert sample.upstream_revision == PINNED_SAMANANTAR_REVISION
    assert sample.preprocessing_run_id == "samanantar-en-te-2026-10-01T01:02:03Z-seed-17"
    assert sample.preprocessing_seed == 17
    assert dict(sample.prepared_files) == {
        "train": "train.en-te.jsonl",
        "val": "val.en-te.jsonl",
        "test": "test.en-te.jsonl",
    }
    assert provenance.dataset_revision == sample.dataset_revision
    assert provenance.split == "test"
    assert provenance.record_id == "99"
    assert provenance.source_location.endswith("#L1:source_text")
    assert provenance.reference_location.endswith("#L1:target_text")
    assert "source_language,target_language,source_text,target_text" in provenance.alignment_location


def test_missing_upstream_revision_fails_closed(tmp_path: Path) -> None:
    adapter = _write_fixture(tmp_path, [_fixture_row()], upstream_revision=None)

    with pytest.raises(ValueError, match="revision is missing or malformed"):
        adapter.load_sample(split="test", max_samples=1)


@pytest.mark.parametrize("revision", ["ead34c5", "g" * 40, ""])
def test_malformed_upstream_revision_fails_closed(tmp_path: Path, revision: str) -> None:
    adapter = _write_fixture(tmp_path, [_fixture_row()], upstream_revision=revision)

    with pytest.raises(ValueError, match="revision is missing or malformed"):
        adapter.load_sample(split="test", max_samples=1)


def test_upstream_revision_is_distinct_from_local_preprocessing_run(tmp_path: Path) -> None:
    sample = _write_fixture(tmp_path, [_fixture_row()]).load_sample(
        split="test", max_samples=1
    )

    assert sample.dataset_revision == sample.upstream_revision
    assert sample.dataset_revision != sample.preprocessing_run_id
    assert sample.revision_scope == "upstream_revision"


def test_samanantar_record_has_no_document_id_or_context(tmp_path: Path) -> None:
    sample = _write_fixture(tmp_path, [_fixture_row()]).load_sample(
        split="test", max_samples=1
    )
    record = sample.records[0]

    assert record.provenance.unit_id is None
    assert record.provenance.unit_type is None
    assert record.context_status is ContextStatus.UNAVAILABLE
    assert record.preceding_context == ()
    assert record.context_supplied_to_model is False
    assert record.context_conditioned_translation is None


def test_context_fields_are_rejected_not_consumed(tmp_path: Path) -> None:
    row = _fixture_row()
    row["document_id"] = "caller-added-document"
    sample = _write_fixture(tmp_path, [row]).load_sample(split="test", max_samples=1)

    assert sample.accepted_count == 0
    assert sample.exclusions[0].reason == "unexpected_document_context_fields"


def test_unknown_fields_are_rejected_without_silent_repair(tmp_path: Path) -> None:
    row = _fixture_row()
    row["source_sentence"] = "replacement"
    sample = _write_fixture(tmp_path, [row]).load_sample(split="test", max_samples=1)

    assert sample.accepted_count == 0
    assert sample.exclusions[0].reason == "missing_or_unexpected_fields"


def test_missing_data_source_is_rejected(tmp_path: Path) -> None:
    row = _fixture_row()
    row["data_source"] = ""
    sample = _write_fixture(tmp_path, [row]).load_sample(split="test", max_samples=1)

    assert sample.exclusions[0].reason == "invalid_data_source"


def test_duplicate_record_ids_are_excluded(tmp_path: Path) -> None:
    sample = _write_fixture(tmp_path, [_fixture_row(7), _fixture_row(7)]).load_sample(
        split="test", max_samples=2
    )

    assert sample.accepted_count == 1
    assert sample.exclusions[0].reason == "duplicate_record_id"


def test_prefix_sample_preserves_order_and_respects_max_samples(tmp_path: Path) -> None:
    sample = _write_fixture(tmp_path, [_fixture_row(index) for index in range(10)]).load_sample(
        split="test", max_samples=3
    )

    assert sample.rows_inspected == 3
    assert [record.provenance.record_id for record in sample.records] == ["0", "1", "2"]
    assert sample.expected_split_count == 10


def test_reader_stops_before_lines_after_the_requested_sample(tmp_path: Path) -> None:
    adapter = _write_fixture(tmp_path, [_fixture_row(1), _fixture_row(2), _fixture_row(3)])
    with (tmp_path / "data" / "processed" / "test.en-te.jsonl").open(
        "a", encoding="utf-8"
    ) as output:
        output.write("{malformed after requested sample}\n")

    sample = adapter.load_sample(split="test", max_samples=3)

    assert sample.rows_inspected == 3
    assert sample.accepted_count == 3
    assert sample.exclusions == ()


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_supported_split_names_and_expected_counts_are_mapped(
    split: str, tmp_path: Path
) -> None:
    sample = _write_fixture(tmp_path, [_fixture_row()], split=split).load_sample(
        split=split, max_samples=1
    )

    assert sample.split.value == split
    assert sample.expected_split_count == {"train": 10, "val": 4, "test": 1}[split]


def test_invalid_split_and_sample_size_fail_before_loading(tmp_path: Path) -> None:
    adapter = _write_fixture(tmp_path, [_fixture_row()])
    with pytest.raises(ValueError, match="split must be train, val, or test"):
        adapter.load_sample(split="dev", max_samples=1)
    with pytest.raises(ValueError, match="max_samples must be a positive integer"):
        adapter.load_sample(split="test", max_samples=0)


def test_synthetic_provenance_remains_untrusted_for_metrics(tmp_path: Path) -> None:
    class SyntheticFixtureAdapter(SamanantarPreparedAdapter):
        @property
        def processed_dir(self) -> Path:
            return tmp_path / "data" / "processed"

        @property
        def preprocessing_report(self) -> Path:
            return tmp_path / "experiments" / "dataset_preprocessing_report.json"

        def _provenance_for_record(self, **_kwargs: object) -> DatasetProvenance:
            return DatasetProvenance.synthetic("synthetic fixture provenance")

    fixture = _write_fixture(tmp_path, [_fixture_row()])
    adapter = SyntheticFixtureAdapter()
    sample = adapter.load_sample(split="test", max_samples=1)
    record = sample.records[0]

    assert record.provenance.status is ProvenanceStatus.SYNTHETIC
    assert metric_eligibility(record, MetricName.SENTENCE_BLEU) == (
        False,
        "provenance is synthetic",
    )
