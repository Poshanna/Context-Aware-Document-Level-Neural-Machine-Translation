from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.evaluation.dochplt_adapter import (
    ALIGNER_SCORE_THRESHOLD,
    DEFAULT_CONTEXT_WINDOW,
    DocHPLTTrustedAdapter,
)


@pytest.fixture
def valid_doc_row() -> dict[str, object]:
    return {
        "src_doc_id": "doc-001",
        "tgt_doc_id": "doc-001-t",
        "lang_pair": "en-te",
        "src_doc": {
            "ids": ["1.1", "2.1", "3.1", "4.1", "5.1"],
            "sentences": ["A", "B", "C", "D", "E"],
        },
        "tgt_doc": {
            "ids": ["11.1", "12.1", "13.1", "14.1", "15.1"],
            "sentences": ["AA", "BB", "CC", "DD", "EE"],
        },
        "alignment": [
            {"src": ["1.1"], "tgt": ["11.1"], "aligner-score": 0.91, "bicleaner-score": 0.5, "bifixer-score": 0.4},
            {"src": ["2.1"], "tgt": ["12.1"], "aligner-score": 0.88, "bicleaner-score": 0.5, "bifixer-score": 0.4},
            {"src": ["3.1"], "tgt": ["13.1"], "aligner-score": 0.80, "bicleaner-score": 0.5, "bifixer-score": 0.4},
            {"src": ["4.1"], "tgt": ["14.1"], "aligner-score": 0.75, "bicleaner-score": 0.5, "bifixer-score": 0.4},
            {"src": ["5.1"], "tgt": ["15.1"], "aligner-score": 0.83, "bicleaner-score": 0.5, "bifixer-score": 0.4},
        ],
    }


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def test_valid_schema_accepted(tmp_path: Path, valid_doc_row: dict[str, object]) -> None:
    shard = tmp_path / "en-te" / "train-00000-of-00030.parquet"
    shard.parent.mkdir(parents=True)
    _write_parquet(shard, [valid_doc_row])

    subset = DocHPLTTrustedAdapter().build_evaluation_subset(
        shard_path=shard,
        max_samples=5,
        seed=7,
        context_window=2,
        aligner_score_threshold=ALIGNER_SCORE_THRESHOLD,
    )

    assert subset.records
    example = sorted(subset.records, key=lambda item: item.source_sentence_id)[-1]
    assert example.src_doc_id == "doc-001"
    assert example.tgt_doc_id == "doc-001-t"
    assert example.lang_pair == "en-te"
    assert example.context_sentence_ids == ("3.1", "4.1")
    assert example.context_sentences == ("C", "D")
    assert example.source_sentence_id == "5.1"
    assert example.target_sentence_id == "15.1"
    assert example.alignment_score == pytest.approx(0.83)
    assert example.provenance.dataset_name == "DocHPLT"
    assert example.provenance.split == "train"
    assert example.provenance.source_location.endswith(str(shard))


def test_wrong_language_pair_rejected(tmp_path: Path, valid_doc_row: dict[str, object]) -> None:
    shard = tmp_path / "bad-parquet.parquet"
    row = json.loads(json.dumps(valid_doc_row))
    row["lang_pair"] = "hi-te"
    _write_parquet(shard, [row])

    with pytest.raises(ValueError, match="lang_pair"):
        DocHPLTTrustedAdapter().load_shard(shard)


def test_missing_required_column_rejected(tmp_path: Path, valid_doc_row: dict[str, object]) -> None:
    shard = tmp_path / "missing-column.parquet"
    row = json.loads(json.dumps(valid_doc_row))
    row.pop("alignment")
    _write_parquet(shard, [row])

    with pytest.raises(ValueError, match="alignment"):
        DocHPLTTrustedAdapter().load_shard(shard)


def test_duplicate_source_sentence_ids_rejected(tmp_path: Path, valid_doc_row: dict[str, object]) -> None:
    shard = tmp_path / "dup-src.parquet"
    row = json.loads(json.dumps(valid_doc_row))
    row["src_doc"]["ids"] = ["1.1", "2.1", "2.1"]
    row["src_doc"]["sentences"] = ["A", "B", "B2"]
    _write_parquet(shard, [row])

    with pytest.raises(ValueError, match="source sentence IDs"):
        DocHPLTTrustedAdapter().load_shard(shard)


def test_duplicate_target_sentence_ids_rejected(tmp_path: Path, valid_doc_row: dict[str, object]) -> None:
    shard = tmp_path / "dup-tgt.parquet"
    row = json.loads(json.dumps(valid_doc_row))
    row["tgt_doc"]["ids"] = ["11.1", "12.1", "12.1"]
    row["tgt_doc"]["sentences"] = ["AA", "BB", "BB2"]
    _write_parquet(shard, [row])

    with pytest.raises(ValueError, match="target sentence IDs"):
        DocHPLTTrustedAdapter().load_shard(shard)


def test_invalid_source_alignment_id_rejected(tmp_path: Path, valid_doc_row: dict[str, object]) -> None:
    shard = tmp_path / "bad-src-align.parquet"
    row = json.loads(json.dumps(valid_doc_row))
    row["alignment"][0]["src"] = ["missing.1"]
    _write_parquet(shard, [row])

    with pytest.raises(ValueError, match="source alignment"):
        DocHPLTTrustedAdapter().load_shard(shard)


def test_invalid_target_alignment_id_rejected(tmp_path: Path, valid_doc_row: dict[str, object]) -> None:
    shard = tmp_path / "bad-tgt-align.parquet"
    row = json.loads(json.dumps(valid_doc_row))
    row["alignment"][0]["tgt"] = ["missing.1"]
    _write_parquet(shard, [row])

    with pytest.raises(ValueError, match="target alignment"):
        DocHPLTTrustedAdapter().load_shard(shard)


def test_1_to_many_alignment_rejected(tmp_path: Path, valid_doc_row: dict[str, object]) -> None:
    shard = tmp_path / "1-to-many.parquet"
    row = json.loads(json.dumps(valid_doc_row))
    row["alignment"][0] = {"src": ["1.1", "2.1"], "tgt": ["11.1"], "aligner-score": 0.80, "bicleaner-score": 0.5, "bifixer-score": 0.4}
    _write_parquet(shard, [row])

    with pytest.raises(ValueError, match="1-to-1"):
        DocHPLTTrustedAdapter().build_evaluation_subset(
            shard_path=shard,
            max_samples=1,
            context_window=2,
        )


def test_many_to_1_alignment_rejected(tmp_path: Path, valid_doc_row: dict[str, object]) -> None:
    shard = tmp_path / "many-to-1.parquet"
    row = json.loads(json.dumps(valid_doc_row))
    row["alignment"][0] = {"src": ["1.1"], "tgt": ["11.1", "12.1"], "aligner-score": 0.80, "bicleaner-score": 0.5, "bifixer-score": 0.4}
    _write_parquet(shard, [row])

    with pytest.raises(ValueError, match="1-to-1"):
        DocHPLTTrustedAdapter().build_evaluation_subset(
            shard_path=shard,
            max_samples=1,
            context_window=2,
        )


def test_many_to_many_alignment_rejected(tmp_path: Path, valid_doc_row: dict[str, object]) -> None:
    shard = tmp_path / "many-to-many.parquet"
    row = json.loads(json.dumps(valid_doc_row))
    row["alignment"][0] = {"src": ["1.1", "2.1"], "tgt": ["11.1", "12.1"], "aligner-score": 0.80, "bicleaner-score": 0.5, "bifixer-score": 0.4}
    _write_parquet(shard, [row])

    with pytest.raises(ValueError, match="1-to-1"):
        DocHPLTTrustedAdapter().build_evaluation_subset(
            shard_path=shard,
            max_samples=1,
            context_window=2,
        )


def test_alignment_below_0_70_rejected(tmp_path: Path, valid_doc_row: dict[str, object]) -> None:
    shard = tmp_path / "below-threshold.parquet"
    row = json.loads(json.dumps(valid_doc_row))
    row["alignment"] = [
        {"src": ["1.1"], "tgt": ["11.1"], "aligner-score": 0.69, "bicleaner-score": 0.5, "bifixer-score": 0.4},
        {"src": ["2.1"], "tgt": ["12.1"], "aligner-score": 0.69, "bicleaner-score": 0.5, "bifixer-score": 0.4},
        {"src": ["3.1"], "tgt": ["13.1"], "aligner-score": 0.69, "bicleaner-score": 0.5, "bifixer-score": 0.4},
    ]
    _write_parquet(shard, [row])

    subset = DocHPLTTrustedAdapter().build_evaluation_subset(
        shard_path=shard,
        max_samples=5,
        seed=1,
        aligner_score_threshold=0.70,
    )

    assert subset.selected_examples == ()
    assert subset.stats["eligible_1to1_alignments"] == 0


def test_alignment_exacly_0_70_accepted(tmp_path: Path, valid_doc_row: dict[str, object]) -> None:
    shard = tmp_path / "at-threshold.parquet"
    row = json.loads(json.dumps(valid_doc_row))
    row["alignment"] = [{"src": ["2.1"], "tgt": ["12.1"], "aligner-score": 0.70, "bicleaner-score": 0.5, "bifixer-score": 0.4}]
    _write_parquet(shard, [row])

    subset = DocHPLTTrustedAdapter().build_evaluation_subset(
        shard_path=shard,
        max_samples=5,
        seed=1,
        aligner_score_threshold=0.70,
    )

    assert subset.selected_examples
    assert subset.selected_examples[0].alignment_score == pytest.approx(0.70)


def test_missing_preceding_context_rejected(tmp_path: Path, valid_doc_row: dict[str, object]) -> None:
    shard = tmp_path / "no-context.parquet"
    row = json.loads(json.dumps(valid_doc_row))
    row["src_doc"]["ids"] = ["1.1", "2.1"]
    row["src_doc"]["sentences"] = ["A", "B"]
    row["tgt_doc"]["ids"] = ["11.1", "12.1"]
    row["tgt_doc"]["sentences"] = ["AA", "BB"]
    row["src_doc"]["ids"] = ["1.1"]
    row["src_doc"]["sentences"] = ["A"]
    row["tgt_doc"]["ids"] = ["11.1"]
    row["tgt_doc"]["sentences"] = ["AA"]
    row["alignment"] = [{"src": ["1.1"], "tgt": ["11.1"], "aligner-score": 0.80, "bicleaner-score": 0.5, "bifixer-score": 0.4}]
    _write_parquet(shard, [row])

    subset = DocHPLTTrustedAdapter().build_evaluation_subset(
        shard_path=shard,
        max_samples=5,
        seed=1,
        context_window=2,
    )

    assert subset.selected_examples == ()
    assert subset.stats["rejected_no_context"] == 1


def test_context_is_only_preceding_sentences(tmp_path: Path, valid_doc_row: dict[str, object]) -> None:
    shard = tmp_path / "context-order.parquet"
    row = json.loads(json.dumps(valid_doc_row))
    row["alignment"] = [
        {"src": ["1.1"], "tgt": ["11.1"], "aligner-score": 0.80, "bicleaner-score": 0.5, "bifixer-score": 0.4},
        {"src": ["3.1"], "tgt": ["13.1"], "aligner-score": 0.80, "bicleaner-score": 0.5, "bifixer-score": 0.4},
    ]
    _write_parquet(shard, [row])

    subset = DocHPLTTrustedAdapter().build_evaluation_subset(
        shard_path=shard,
        max_samples=5,
        seed=1,
        context_window=2,
    )

    example = subset.selected_examples[0]
    assert example.context_sentence_ids == ("1.1", "2.1")
    assert example.context_sentences == ("A", "B")


def test_future_sentences_are_never_included(tmp_path: Path, valid_doc_row: dict[str, object]) -> None:
    shard = tmp_path / "future-context.parquet"
    row = json.loads(json.dumps(valid_doc_row))
    _write_parquet(shard, [row])

    subset = DocHPLTTrustedAdapter().build_evaluation_subset(
        shard_path=shard,
        max_samples=5,
        seed=1,
        context_window=2,
    )

    selected = subset.selected_examples[0]
    assert "5.1" not in selected.context_sentence_ids


def test_current_sentence_is_never_duplicated_in_context(tmp_path: Path, valid_doc_row: dict[str, object]) -> None:
    shard = tmp_path / "current-in-context.parquet"
    row = json.loads(json.dumps(valid_doc_row))
    _write_parquet(shard, [row])

    subset = DocHPLTTrustedAdapter().build_evaluation_subset(
        shard_path=shard,
        max_samples=5,
        seed=1,
        context_window=2,
    )

    selected = subset.selected_examples[0]
    assert selected.source_sentence_id not in selected.context_sentence_ids
    assert selected.source_sentence not in selected.context_sentences


def test_context_never_crosses_document_boundaries(tmp_path: Path, valid_doc_row: dict[str, object]) -> None:
    shard = tmp_path / "boundaries.parquet"
    row_one = json.loads(json.dumps(valid_doc_row))
    row_two = json.loads(json.dumps(valid_doc_row))
    row_two["src_doc_id"] = "doc-002"
    row_two["tgt_doc_id"] = "doc-002-t"
    row_two["src_doc"]["ids"] = ["1.1", "2.1", "3.1"]
    row_two["src_doc"]["sentences"] = ["X", "Y", "Z"]
    row_two["tgt_doc"]["ids"] = ["21.1", "22.1", "23.1"]
    row_two["tgt_doc"]["sentences"] = ["XX", "YY", "ZZ"]
    row_two["alignment"] = [{"src": ["3.1"], "tgt": ["23.1"], "aligner-score": 0.83, "bicleaner-score": 0.5, "bifixer-score": 0.4}]
    _write_parquet(shard, [row_one, row_two])

    subset = DocHPLTTrustedAdapter().build_evaluation_subset(
        shard_path=shard,
        max_samples=10,
        seed=0,
        context_window=2,
    )

    assert subset.records
    assert all(example.document_id in {"doc-001", "doc-002"} for example in subset.records)
    assert all(example.context_sentence_ids == [] or example.src_doc_id == example.document_id for example in subset.records)


def test_explicit_target_alignment_is_respected_even_when_sentence_numbers_differ(tmp_path: Path) -> None:
    shard = tmp_path / "explicit-target.parquet"
    row = {
        "src_doc_id": "doc-001",
        "tgt_doc_id": "doc-001-t",
        "lang_pair": "en-te",
        "src_doc": {"ids": ["1.1", "2.1", "3.1"], "sentences": ["one", "two", "three"]},
        "tgt_doc": {"ids": ["10.1", "11.1", "12.1"], "sentences": ["ten", "eleven", "twelve"]},
        "alignment": [{"src": ["2.1"], "tgt": ["11.1"], "aligner-score": 0.80, "bicleaner-score": 0.5, "bifixer-score": 0.4}],
    }
    _write_parquet(shard, [row])

    subset = DocHPLTTrustedAdapter().build_evaluation_subset(
        shard_path=shard,
        max_samples=5,
        seed=0,
        context_window=2,
        aligner_score_threshold=0.70,
    )

    example = subset.selected_examples[0]
    assert example.source_sentence_id == "2.1"
    assert example.target_sentence_id == "11.1"
    assert example.source_sentence == "two"
    assert example.target_reference == "eleven"
    assert example.context_sentence_ids == ("1.1",)


def test_deterministic_sampling_with_same_seed_produces_same_examples(tmp_path: Path, valid_doc_row: dict[str, object]) -> None:
    shard = tmp_path / "deterministic.parquet"
    rows = [valid_doc_row, json.loads(json.dumps(valid_doc_row))]
    rows[1]["src_doc_id"] = "doc-002"
    rows[1]["tgt_doc_id"] = "doc-002-t"
    rows[1]["src_doc"]["ids"] = ["1.1", "2.1", "3.1", "4.1", "5.1"]
    rows[1]["tgt_doc"]["ids"] = ["21.1", "22.1", "23.1", "24.1", "25.1"]
    rows[1]["alignment"] = [{"src": ["5.1"], "tgt": ["25.1"], "aligner-score": 0.83, "bicleaner-score": 0.5, "bifixer-score": 0.4}]
    _write_parquet(shard, rows)

    first = DocHPLTTrustedAdapter().build_evaluation_subset(shard_path=shard, max_samples=2, seed=11, context_window=2)
    second = DocHPLTTrustedAdapter().build_evaluation_subset(shard_path=shard, max_samples=2, seed=11, context_window=2)

    assert [example.source_sentence_id for example in first.selected_examples] == [example.source_sentence_id for example in second.selected_examples]
    assert [example.target_sentence_id for example in first.selected_examples] == [example.target_sentence_id for example in second.selected_examples]


def test_different_seeds_can_produce_different_valid_samples(tmp_path: Path, valid_doc_row: dict[str, object]) -> None:
    shard = tmp_path / "different-seed.parquet"
    rows = [valid_doc_row, json.loads(json.dumps(valid_doc_row))]
    rows[1]["src_doc_id"] = "doc-002"
    rows[1]["tgt_doc_id"] = "doc-002-t"
    rows[1]["src_doc"]["ids"] = ["1.1", "2.1", "3.1", "4.1", "5.1"]
    rows[1]["tgt_doc"]["ids"] = ["21.1", "22.1", "23.1", "24.1", "25.1"]
    rows[1]["alignment"] = [{"src": ["5.1"], "tgt": ["25.1"], "aligner-score": 0.83, "bicleaner-score": 0.5, "bifixer-score": 0.4}]
    _write_parquet(shard, rows)

    first = DocHPLTTrustedAdapter().build_evaluation_subset(shard_path=shard, max_samples=2, seed=3, context_window=2)
    second = DocHPLTTrustedAdapter().build_evaluation_subset(shard_path=shard, max_samples=2, seed=4, context_window=2)

    assert first.selected_examples != second.selected_examples


def test_provenance_is_present(tmp_path: Path, valid_doc_row: dict[str, object]) -> None:
    shard = tmp_path / "provenance.parquet"
    _write_parquet(shard, [valid_doc_row])

    subset = DocHPLTTrustedAdapter().build_evaluation_subset(
        shard_path=shard,
        max_samples=1,
        seed=0,
        context_window=2,
    )

    provenance = subset.selected_examples[0].provenance
    assert provenance.dataset_name == "DocHPLT"
    assert provenance.dataset_revision == "76a591ec3c014d0ac4193a3cbdfc12689616a8d3"
    assert provenance.split == "train"
    assert provenance.source_location.endswith(str(shard))
    assert provenance.alignment_location.endswith(str(shard))


def test_no_artificial_alignment_is_generated(tmp_path: Path) -> None:
    shard = tmp_path / "no-artificial.parquet"
    row = {
        "src_doc_id": "doc-001",
        "tgt_doc_id": "doc-001-t",
        "lang_pair": "en-te",
        "src_doc": {"ids": ["1.1", "2.1", "3.1"], "sentences": ["A", "B", "C"]},
        "tgt_doc": {"ids": ["10.1", "11.1", "12.1"], "sentences": ["AA", "BB", "CC"]},
        "alignment": [{"src": ["3.1"], "tgt": ["12.1"], "aligner-score": 0.83, "bicleaner-score": 0.5, "bifixer-score": 0.4}],
    }
    _write_parquet(shard, [row])

    subset = DocHPLTTrustedAdapter().build_evaluation_subset(
        shard_path=shard,
        max_samples=1,
        seed=0,
        context_window=2,
    )

    selected = subset.selected_examples[0]
    assert selected.alignment_score == pytest.approx(0.83)
    assert selected.source_sentence_id == "3.1"
    assert selected.target_sentence_id == "12.1"
    assert selected.target_reference == "CC"
