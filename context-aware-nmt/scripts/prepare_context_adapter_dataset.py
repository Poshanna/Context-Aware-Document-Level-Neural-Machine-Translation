from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dochplt"
    / "en-te"
    / "train-00000-of-00030.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "context_adapter"
)

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VAL_FILE = OUTPUT_DIR / "val.jsonl"
STATS_FILE = OUTPUT_DIR / "dataset_stats.json"

SEED = 42

MAX_TRAIN_EXAMPLES = 1000
MAX_VAL_EXAMPLES = 100
MAX_CONTEXT_SENTENCES = 2


def parse_document(value: Any) -> list[dict[str, str]]:
    """
    Convert the actual DocHPLT document structure:

        {
            "ids": [...],
            "sentences": [...]
        }

    into:

        [
            {"id": "...", "text": "..."},
            ...
        ]
    """

    if not isinstance(value, dict):
        return []

    ids = value.get("ids")
    sentences = value.get("sentences")

    if not isinstance(ids, list):
        return []

    if not isinstance(sentences, list):
        return []

    if len(ids) != len(sentences):
        return []

    result = []

    for sentence_id, sentence_text in zip(ids, sentences):

        if sentence_id is None:
            continue

        if not isinstance(sentence_text, str):
            continue

        sentence_text = sentence_text.strip()

        if not sentence_text:
            continue

        result.append(
            {
                "id": str(sentence_id),
                "text": sentence_text,
            }
        )

    return result


def parse_alignment(value: Any) -> list[dict[str, Any]]:
    """
    DocHPLT alignment is already returned by PyArrow as a list of
    dictionaries.

    Example:

        [
            {
                "src": ["2.1"],
                "tgt": ["11.1", "12.1"],
                "aligner-score": 0.23,
                ...
            }
        ]
    """

    if not isinstance(value, list):
        return []

    result = []

    for item in value:

        if not isinstance(item, dict):
            continue

        src = item.get("src")
        tgt = item.get("tgt")

        if not isinstance(src, list):
            continue

        if not isinstance(tgt, list):
            continue

        result.append(item)

    return result


def main() -> None:

    print("=" * 80)
    print("DocHPLT Context Adapter Dataset Preparation")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"DocHPLT shard not found:\n{INPUT_FILE}"
        )

    print()
    print(f"Input file: {INPUT_FILE}")

    table = pq.read_table(INPUT_FILE)

    print(f"Rows loaded: {table.num_rows}")

    print()
    print("Available columns:")

    for column in table.column_names:
        print(f"  - {column}")

    required_columns = {
        "src_doc_id",
        "tgt_doc_id",
        "lang_pair",
        "src_doc",
        "tgt_doc",
        "alignment",
    }

    missing = required_columns.difference(
        table.column_names
    )

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    src_doc_ids = table["src_doc_id"].to_pylist()
    tgt_doc_ids = table["tgt_doc_id"].to_pylist()
    lang_pairs = table["lang_pair"].to_pylist()
    src_docs = table["src_doc"].to_pylist()
    tgt_docs = table["tgt_doc"].to_pylist()
    alignments = table["alignment"].to_pylist()

    print()
    print("DocHPLT document-level schema confirmed.")

    # ------------------------------------------------------------------
    # First document inspection
    # ------------------------------------------------------------------

    first_source = parse_document(src_docs[0])
    first_target = parse_document(tgt_docs[0])
    first_alignment = parse_alignment(alignments[0])

    print()
    print("=" * 80)
    print("First-document inspection")
    print("=" * 80)

    print()
    print(f"Source document ID: {src_doc_ids[0]}")
    print(f"Target document ID: {tgt_doc_ids[0]}")
    print(f"Language pair: {lang_pairs[0]}")

    print()
    print(
        f"Source sentences: {len(first_source)}"
    )

    print(
        f"Target sentences: {len(first_target)}"
    )

    print(
        f"Alignment groups: {len(first_alignment)}"
    )

    if first_source:
        print()
        print("First source sentence:")

        print(
            json.dumps(
                first_source[0],
                ensure_ascii=False,
                indent=2,
            )
        )

    if first_target:
        print()
        print("First target sentence:")

        print(
            json.dumps(
                first_target[0],
                ensure_ascii=False,
                indent=2,
            )
        )

    if first_alignment:
        print()
        print("First alignment group:")

        print(
            json.dumps(
                first_alignment[0],
                ensure_ascii=False,
                indent=2,
            )
        )

    # ------------------------------------------------------------------
    # Build examples
    # ------------------------------------------------------------------

    candidates = []

    documents_with_source = 0
    documents_with_target = 0
    documents_with_alignment = 0

    total_alignment_groups = 0

    one_to_one_groups = 0
    one_to_many_groups = 0
    many_to_one_groups = 0
    many_to_many_groups = 0

    for row_index in range(table.num_rows):

        source_sentences = parse_document(
            src_docs[row_index]
        )

        target_sentences = parse_document(
            tgt_docs[row_index]
        )

        if not source_sentences:
            continue

        documents_with_source += 1

        if not target_sentences:
            continue

        documents_with_target += 1

        source_by_id = {
            item["id"]: item
            for item in source_sentences
        }

        target_by_id = {
            item["id"]: item
            for item in target_sentences
        }

        source_order = [
            item["id"]
            for item in source_sentences
        ]

        source_position = {
            sentence_id: position
            for position, sentence_id in enumerate(
                source_order
            )
        }

        alignment_groups = parse_alignment(
            alignments[row_index]
        )

        if not alignment_groups:
            continue

        documents_with_alignment += 1
        total_alignment_groups += len(
            alignment_groups
        )

        for group in alignment_groups:

            src_ids = group["src"]
            tgt_ids = group["tgt"]

            # ----------------------------------------------------------
            # Keep ONLY one source ↔ one target.
            # ----------------------------------------------------------

            if len(src_ids) == 1 and len(tgt_ids) == 1:

                one_to_one_groups += 1

                source_id = str(src_ids[0])
                target_id = str(tgt_ids[0])

                if source_id not in source_by_id:
                    continue

                if target_id not in target_by_id:
                    continue

                current_position = source_position.get(
                    source_id
                )

                if current_position is None:
                    continue

                # Need preceding context.
                if current_position == 0:
                    continue

                context_start = max(
                    0,
                    current_position
                    - MAX_CONTEXT_SENTENCES,
                )

                context = []

                for context_position in range(
                    context_start,
                    current_position,
                ):

                    context_id = source_order[
                        context_position
                    ]

                    context_text = source_by_id[
                        context_id
                    ]["text"]

                    if context_text.strip():

                        context.append(
                            {
                                "id": context_id,
                                "text": context_text,
                            }
                        )

                if not context:
                    continue

                current_source = source_by_id[
                    source_id
                ]["text"]

                target_text = target_by_id[
                    target_id
                ]["text"]

                candidates.append(
                    {
                        "source_document_id": str(
                            src_doc_ids[row_index]
                        ),
                        "target_document_id": str(
                            tgt_doc_ids[row_index]
                        ),
                        "source_sentence_id": source_id,
                        "target_sentence_id": target_id,
                        "source_sentence": current_source,
                        "target_sentence": target_text,
                        "context": context,
                        "alignment_type": "1-to-1",
                        "alignment_score": group.get(
                            "aligner-score"
                        ),
                        "bicleaner_score": group.get(
                            "bicleaner-score"
                        ),
                        "bifixer_score": group.get(
                            "bifixer-score"
                        ),
                    }
                )

            elif len(src_ids) == 1 and len(tgt_ids) > 1:

                one_to_many_groups += 1

            elif len(src_ids) > 1 and len(tgt_ids) == 1:

                many_to_one_groups += 1

            elif len(src_ids) > 1 and len(tgt_ids) > 1:

                many_to_many_groups += 1

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("Dataset statistics")
    print("=" * 80)

    print()
    print(
        f"Documents loaded:             {table.num_rows}"
    )

    print(
        f"With source sentences:        {documents_with_source}"
    )

    print(
        f"With target sentences:        {documents_with_target}"
    )

    print(
        f"With alignment groups:        {documents_with_alignment}"
    )

    print()
    print(
        f"Total alignment groups:       {total_alignment_groups}"
    )

    print(
        f"1-to-1 alignment groups:      {one_to_one_groups}"
    )

    print(
        f"1-to-many alignment groups:   {one_to_many_groups}"
    )

    print(
        f"many-to-1 alignment groups:   {many_to_one_groups}"
    )

    print(
        f"many-to-many groups:          {many_to_many_groups}"
    )

    print()
    print(
        f"Context-aware candidates:     {len(candidates)}"
    )

    if not candidates:
        raise RuntimeError(
            "No context-aware candidates were created."
        )

    # ------------------------------------------------------------------
    # Deterministic sampling
    # ------------------------------------------------------------------

    rng = random.Random(SEED)

    rng.shuffle(candidates)

    train_examples = candidates[
        :MAX_TRAIN_EXAMPLES
    ]

    remaining = candidates[
        MAX_TRAIN_EXAMPLES:
    ]

    val_examples = remaining[
        :MAX_VAL_EXAMPLES
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Write train
    # ------------------------------------------------------------------

    with TRAIN_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for example in train_examples:

            handle.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ------------------------------------------------------------------
    # Write validation
    # ------------------------------------------------------------------

    with VAL_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for example in val_examples:

            handle.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    stats = {
        "dataset": "DocHPLT",
        "config": "en-te",
        "source_shard": str(INPUT_FILE),
        "seed": SEED,
        "max_context_sentences": MAX_CONTEXT_SENTENCES,
        "documents_loaded": table.num_rows,
        "documents_with_source": documents_with_source,
        "documents_with_target": documents_with_target,
        "documents_with_alignment": documents_with_alignment,
        "alignment_groups": total_alignment_groups,
        "one_to_one_groups": one_to_one_groups,
        "one_to_many_groups": one_to_many_groups,
        "many_to_one_groups": many_to_one_groups,
        "many_to_many_groups": many_to_many_groups,
        "context_aware_candidates": len(candidates),
        "train_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "context_source": (
            "preceding source sentences from same source document"
        ),
        "target_side_used_for_context": False,
        "alignment_inference": False,
        "official_test_set": False,
        "purpose": "context adapter training preparation",
    }

    STATS_FILE.write_text(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("Dataset preparation completed")
    print("=" * 80)

    print()
    print(f"Train file: {TRAIN_FILE}")
    print(f"Validation file: {VAL_FILE}")
    print(f"Stats file: {STATS_FILE}")

    print()
    print(
        f"Training examples:   {len(train_examples)}"
    )

    print(
        f"Validation examples: {len(val_examples)}"
    )

    # Show one complete example.
    if train_examples:

        print()
        print("=" * 80)
        print("First prepared training example")
        print("=" * 80)

        print(
            json.dumps(
                train_examples[0],
                ensure_ascii=False,
                indent=2,
            )
        )

    print()
    print("Context rules:")
    print(
        "  - preceding sentences from the SAME source document"
    )
    print(
        "  - maximum context sentences:",
        MAX_CONTEXT_SENTENCES,
    )
    print(
        "  - only 1-to-1 source-target alignments"
    )
    print(
        "  - target text NOT used for context selection"
    )

    print()
    print("This is NOT an official DocHPLT test set.")


if __name__ == "__main__":
    main()