"""Trusted DocHPLT adapter for clean 1-to-1 same-document context evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from app.evaluation.document_evaluation import (
    DatasetProvenance,
    ProvenanceStatus,
    TextUnit,
    TrustedDatasetAdapter,
)

DATASET_NAME = "DocHPLT"
DATASET_CONFIG = "en-te"
DATASET_SPLIT = "train"
DATASET_REVISION = "76a591ec3c014d0ac4193a3cbdfc12689616a8d3"
SOURCE_SHARD = "en-te/train-00000-of-00030.parquet"
ALIGNER_SCORE_THRESHOLD = 0.70
DEFAULT_CONTEXT_WINDOW = 2
ADAPTER_NAME = "dochplt_parquet_v1"

REQUIRED_COLUMNS = {
    "src_doc_id",
    "tgt_doc_id",
    "lang_pair",
    "src_doc",
    "tgt_doc",
    "alignment",
}


@dataclass(frozen=True)
class DocHPLTContextExample:
    document_id: str
    src_doc_id: str
    tgt_doc_id: str
    context_sentence_ids: tuple[str, ...]
    context_sentences: tuple[str, ...]
    source_sentence_id: str
    source_sentence: str
    target_sentence_id: str
    target_reference: str
    alignment_score: float
    lang_pair: str
    provenance: DatasetProvenance


@dataclass(frozen=True)
class DocHPLTSubset:
    dataset_name: str
    dataset_config: str
    split: str
    revision: str
    source_shard: str
    aligner_score_threshold: float
    context_window: int
    seed: int
    stats: dict[str, int]
    records: tuple[DocHPLTContextExample, ...]
    selected_examples: tuple[DocHPLTContextExample, ...]


class DocHPLTTrustedAdapter(TrustedDatasetAdapter):
    """Validate DocHPLT parquet rows and build a clean evaluation subset."""

    @property
    def adapter_name(self) -> str:
        return ADAPTER_NAME

    def load_shard(self, shard_path: str | Path) -> tuple[dict[str, Any], ...]:
        path = Path(shard_path)
        if not path.is_file():
            raise FileNotFoundError(f"DocHPLT shard not found: {path}")

        table = pq.read_table(str(path))
        rows = table.to_pylist()
        validated: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("DocHPLT record must be a mapping.")
            self._validate_record_schema(row)
            self._validate_row_identity(row)
            self._validate_alignment_references(row)
            validated.append(dict(row))
        return tuple(validated)

    def build_evaluation_subset(
        self,
        *,
        shard_path: str | Path,
        max_samples: int = 64,
        seed: int = 0,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        aligner_score_threshold: float = ALIGNER_SCORE_THRESHOLD,
    ) -> DocHPLTSubset:
        if not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples < 1:
            raise ValueError("max_samples must be a positive integer.")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer.")
        if not isinstance(context_window, int) or isinstance(context_window, bool) or context_window < 0:
            raise ValueError("context_window must be a non-negative integer.")
        if not isinstance(aligner_score_threshold, (int, float)):
            raise ValueError("aligner_score_threshold must be numeric.")

        threshold = float(aligner_score_threshold)
        rows = self.load_shard(shard_path)
        stats = {
            "documents_read": 0,
            "eligible_1to1_alignments": 0,
            "eligible_with_context": 0,
            "selected_examples": 0,
            "rejected_non_1to1": 0,
            "rejected_low_alignment_score": 0,
            "rejected_no_context": 0,
            "invalid_alignment_references": 0,
            "invalid_records": 0,
        }

        eligible_examples: list[DocHPLTContextExample] = []
        for row in rows:
            stats["documents_read"] += 1
            try:
                self._validate_record_schema(row)
                self._validate_row_identity(row)
            except ValueError:
                stats["invalid_records"] += 1
                continue

            src_doc = row["src_doc"]
            tgt_doc = row["tgt_doc"]
            src_ids = [str(item) for item in src_doc["ids"]]
            tgt_ids = [str(item) for item in tgt_doc["ids"]]
            src_sentences = [str(item) for item in src_doc["sentences"]]
            tgt_sentences = [str(item) for item in tgt_doc["sentences"]]
            alignments = row.get("alignment", [])

            if not isinstance(alignments, Sequence) or isinstance(alignments, (str, bytes)):
                stats["rejected_non_1to1"] += 1
                continue

            if any(
                not isinstance(alignment, Mapping)
                or len(alignment.get("src", [])) != 1
                or len(alignment.get("tgt", [])) != 1
                for alignment in alignments
            ):
                stats["rejected_non_1to1"] += 1
                continue

            per_row_valid = 0
            for alignment in alignments:
                score = alignment.get("aligner-score")
                if score is None or not isinstance(score, (int, float)):
                    stats["invalid_alignment_references"] += 1
                    continue
                score_value = float(score)
                if score_value < threshold:
                    stats["rejected_low_alignment_score"] += 1
                    continue

                src_id = str(alignment["src"][0])
                tgt_id = str(alignment["tgt"][0])
                if src_id not in src_ids or tgt_id not in tgt_ids:
                    stats["invalid_alignment_references"] += 1
                    continue

                per_row_valid += 1
                stats["eligible_1to1_alignments"] += 1

                src_index = src_ids.index(src_id)
                if src_index == 0:
                    stats["rejected_no_context"] += 1
                    continue

                preceding_ids = src_ids[max(0, src_index - context_window):src_index]
                if not preceding_ids:
                    stats["rejected_no_context"] += 1
                    continue

                tgt_index = tgt_ids.index(tgt_id)
                context_sentences = tuple(
                    src_sentences[src_ids.index(prev_id)] for prev_id in preceding_ids
                )
                context_sentence_ids = tuple(preceding_ids)
                source_sentence = src_sentences[src_index]
                target_reference = tgt_sentences[tgt_index]
                provenance = self.verified_provenance(
                    status=ProvenanceStatus.VERIFIED_DATASET,
                    dataset_name=DATASET_NAME,
                    dataset_revision=DATASET_REVISION,
                    split=DATASET_SPLIT,
                    record_id=f"{row['src_doc_id']}:{src_id}->{tgt_id}",
                    source_location=str(Path(shard_path)),
                    reference_location=str(Path(shard_path)),
                    alignment_location=str(Path(shard_path)),
                    unit_id=row["src_doc_id"],
                    unit_type=TextUnit.DOCUMENT,
                )
                example = DocHPLTContextExample(
                    document_id=row["src_doc_id"],
                    src_doc_id=row["src_doc_id"],
                    tgt_doc_id=row["tgt_doc_id"],
                    context_sentence_ids=context_sentence_ids,
                    context_sentences=context_sentences,
                    source_sentence_id=src_id,
                    source_sentence=source_sentence,
                    target_sentence_id=tgt_id,
                    target_reference=target_reference,
                    alignment_score=score_value,
                    lang_pair=row["lang_pair"],
                    provenance=provenance,
                )
                eligible_examples.append(example)

            if per_row_valid == 0:
                stats["eligible_with_context"] += 0
            else:
                stats["eligible_with_context"] += len(
                    [
                        item
                        for item in eligible_examples
                        if item.src_doc_id == row["src_doc_id"]
                        and item.document_id == row["src_doc_id"]
                    ]
                )

        if eligible_examples:
            ranked = sorted(
                eligible_examples,
                key=lambda example: (
                    hashlib.sha256(
                        (
                            f"{seed}:{example.document_id}:{example.source_sentence_id}:"
                            f"{example.target_sentence_id}"
                        ).encode("utf-8")
                    ).digest(),
                    _sentence_sort_key(example.source_sentence_id),
                ),
            )
            selected = tuple(ranked[: min(max_samples, len(ranked))])
        else:
            selected = ()
        stats["selected_examples"] = len(selected)
        stats["eligible_with_context"] = len(eligible_examples)

        return DocHPLTSubset(
            dataset_name=DATASET_NAME,
            dataset_config=DATASET_CONFIG,
            split=DATASET_SPLIT,
            revision=DATASET_REVISION,
            source_shard=SOURCE_SHARD,
            aligner_score_threshold=threshold,
            context_window=context_window,
            seed=seed,
            stats=stats,
            records=tuple(eligible_examples),
            selected_examples=selected,
        )

    def _validate_record_schema(self, row: Mapping[str, Any]) -> None:
        missing = sorted(REQUIRED_COLUMNS - set(row))
        if missing:
            raise ValueError(f"missing required DocHPLT column(s): {', '.join(missing)}")
        if row["lang_pair"] != DATASET_CONFIG:
            raise ValueError("DocHPLT lang_pair must be en-te.")
        src_doc = row["src_doc"]
        tgt_doc = row["tgt_doc"]
        if not isinstance(src_doc, Mapping) or not isinstance(tgt_doc, Mapping):
            raise ValueError("src_doc and tgt_doc must each be structured document objects.")
        if "ids" not in src_doc or "sentences" not in src_doc:
            raise ValueError("src_doc must contain ids and sentences.")
        if "ids" not in tgt_doc or "sentences" not in tgt_doc:
            raise ValueError("tgt_doc must contain ids and sentences.")
        if not isinstance(src_doc["ids"], Sequence) or not isinstance(src_doc["sentences"], Sequence):
            raise ValueError("src_doc ids and sentences must be sequences.")
        if not isinstance(tgt_doc["ids"], Sequence) or not isinstance(tgt_doc["sentences"], Sequence):
            raise ValueError("tgt_doc ids and sentences must be sequences.")

    def _validate_row_identity(self, row: Mapping[str, Any]) -> None:
        src_doc = row["src_doc"]
        tgt_doc = row["tgt_doc"]
        src_ids = [str(item) for item in src_doc["ids"]]
        tgt_ids = [str(item) for item in tgt_doc["ids"]]
        if len(src_ids) != len(set(src_ids)):
            raise ValueError("DocHPLT source sentence IDs must be unique.")
        if len(tgt_ids) != len(set(tgt_ids)):
            raise ValueError("DocHPLT target sentence IDs must be unique.")
        if src_ids != sorted(src_ids, key=_sentence_sort_key):
            raise ValueError("DocHPLT source sentence IDs must preserve document order.")

    def _validate_alignment_references(self, row: Mapping[str, Any]) -> None:
        src_doc = row["src_doc"]
        tgt_doc = row["tgt_doc"]
        src_ids = [str(item) for item in src_doc["ids"]]
        tgt_ids = [str(item) for item in tgt_doc["ids"]]
        alignments = row.get("alignment", [])
        if not isinstance(alignments, Sequence) or isinstance(alignments, (str, bytes)):
            raise ValueError("DocHPLT alignment must be a list of alignment objects.")
        for alignment in alignments:
            if not isinstance(alignment, Mapping):
                raise ValueError("DocHPLT alignment entries must be mappings.")
            src = alignment.get("src")
            tgt = alignment.get("tgt")
            score = alignment.get("aligner-score")
            if not isinstance(src, Sequence) or isinstance(src, (str, bytes)):
                raise ValueError("DocHPLT source alignment must be a list of sentence IDs.")
            if not isinstance(tgt, Sequence) or isinstance(tgt, (str, bytes)):
                raise ValueError("DocHPLT target alignment must be a list of sentence IDs.")
            if len(src) != 1 or len(tgt) != 1:
                raise ValueError("DocHPLT only 1-to-1 alignments are accepted.")
            src_id = str(src[0])
            tgt_id = str(tgt[0])
            if src_id not in src_ids:
                raise ValueError("DocHPLT source alignment references invalid source sentence ID.")
            if tgt_id not in tgt_ids:
                raise ValueError("DocHPLT target alignment references invalid target sentence ID.")
            if score is None or not isinstance(score, (int, float)):
                raise ValueError("DocHPLT alignment requires an aligner-score value.")
            if float(score) < ALIGNER_SCORE_THRESHOLD:
                continue


def _sentence_sort_key(value: str) -> tuple[int, int, str]:
    if not value:
        return (0, 0, value)
    head, sep, tail = str(value).partition(".")
    try:
        return (int(head), int(tail) if tail else 0, value)
    except ValueError:
        return (0, 0, value)
