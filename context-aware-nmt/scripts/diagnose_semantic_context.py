"""
Diagnose semantic context retrieval on the manually reviewed
DocHPLT evaluation subset.

This script does NOT run NLLB translation.

It checks which preceding source sentences are selected
as semantic context for each evaluation example.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from app.services.semantic_context_retriever import (  # noqa: E402
    SemanticContextRetriever,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MANIFEST_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "evaluation"
    / "dochplt"
    / "final_eval_manifest.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "evaluation"
    / "dochplt"
    / "semantic_context_diagnostic.json"
)

TOP_K = 2


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:

    print("=" * 80)
    print("DocHPLT Semantic Context Retrieval Diagnostic")
    print("=" * 80)

    print()
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Top-K: {TOP_K}")
    print()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)

    examples = manifest.get("examples", [])

    if not examples:
        raise ValueError(
            "No examples found in the manifest."
        )

    print(
        f"Loaded {len(examples)} evaluation examples."
    )

    print()
    print("Loading sentence-transformer model...")

    retriever = SemanticContextRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        top_k=TOP_K,
        min_similarity=0.0,
    )

    # Force model loading before processing examples.
    retriever._load_model()

    print("Embedding model loaded.")
    print()

    results = []

    for example in examples:

        evaluation_index = example.get(
            "evaluation_index"
        )

        document_id = example.get(
            "source_document_id",
            "",
        )

        sentence_id = example.get(
            "source_sentence_id",
            "",
        )

        current_sentence = example.get(
            "source_sentence",
            "",
        )

        # IMPORTANT:
        # The manifest stores context under:
        #
        # "context": [
        #     {"id": "...", "text": "..."},
        #     ...
        # ]
        #
        # These are already validated preceding
        # same-document source sentences.

        raw_context = example.get(
            "context",
            [],
        )

        preceding_sentences = []

        for item in raw_context:

            if not isinstance(item, dict):
                continue

            context_id = str(
                item.get("id", "")
            ).strip()

            context_text = str(
                item.get("text", "")
            ).strip()

            if not context_id or not context_text:
                continue

            preceding_sentences.append(
                {
                    "sentence_id": context_id,
                    "text": context_text,
                }
            )

        selected = retriever.retrieve(
            current_sentence=current_sentence,
            preceding_sentences=preceding_sentences,
        )

        print("-" * 80)
        print(
            f"Example {evaluation_index}"
        )
        print("-" * 80)

        print(
            f"Document ID: {document_id}"
        )

        print(
            f"Current sentence ID: {sentence_id}"
        )

        print()
        print("CURRENT SENTENCE:")
        print(current_sentence)

        print()
        print("AVAILABLE PRECEDING CONTEXT:")

        if not preceding_sentences:
            print("  [none]")
        else:
            for candidate in preceding_sentences:
                print(
                    f"  [{candidate['sentence_id']}] "
                    f"{candidate['text']}"
                )

        print()
        print("SEMANTICALLY SELECTED CONTEXT:")

        if not selected:
            print("  [none]")

        for rank, candidate in enumerate(
            selected,
            start=1,
        ):
            print(
                f"  {rank}. "
                f"[{candidate.sentence_id}] "
                f"similarity="
                f"{candidate.similarity:.6f}"
            )

            print(
                f"     {candidate.text}"
            )

        results.append(
            {
                "evaluation_index": evaluation_index,
                "manual_review_index": example.get(
                    "manual_review_index"
                ),
                "source_document_id": document_id,
                "source_sentence_id": sentence_id,
                "source_sentence": current_sentence,
                "available_context": preceding_sentences,
                "selected_context": [
                    {
                        "sentence_id": candidate.sentence_id,
                        "text": candidate.text,
                        "similarity": candidate.similarity,
                    }
                    for candidate in selected
                ],
            }
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "dataset": "DocHPLT",
                "config": "en-te",
                "split": "train",
                "evaluation_type": (
                    "manually reviewed deterministic "
                    "subset from a DocHPLT training shard"
                ),
                "official_test_set": False,
                "document_level_nmt": False,
                "embedding_model": (
                    "sentence-transformers/"
                    "all-MiniLM-L6-v2"
                ),
                "top_k": TOP_K,
                "target_side_used_for_ranking": False,
                "sample_count": len(results),
                "results": results,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 80)
    print("Diagnostic completed.")
    print(f"Saved: {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()