from __future__ import annotations

import json
from pathlib import Path


RUN_DIR = Path(
    "experiments/evaluation/dochplt/20261006T051941Z-8768e7db"
)

EXAMPLES_PATH = RUN_DIR / "examples.json"
OUTPUT_PATH = RUN_DIR / "diagnostic.json"


def main() -> None:
    if not EXAMPLES_PATH.exists():
        raise FileNotFoundError(
            f"Examples file not found: {EXAMPLES_PATH}"
        )

    with EXAMPLES_PATH.open("r", encoding="utf-8") as f:
        examples = json.load(f)

    print("=" * 90)
    print("DocHPLT 50-Example Diagnostic")
    print("=" * 90)
    print(f"Examples: {len(examples)}")
    print()

    diagnostics = []

    for example in examples:
        source = example["current_source_sentence"]
        reference = example["target_reference"]
        context = example.get("context_sentences", [])
        context_ids = example.get("context_sentence_ids", [])

        score = example.get("alignment_score")

        source_doc = example.get("source_document_id")
        target_doc = example.get("target_document_id")

        source_id = example.get("current_source_sentence_id")
        target_id = example.get("target_sentence_id")

        reasons = []

        # Basic validity
        if not source.strip():
            reasons.append("empty_source")

        if not reference.strip():
            reasons.append("empty_reference")

        # Alignment threshold
        if score is None:
            reasons.append("missing_alignment_score")
        elif score < 0.70:
            reasons.append("below_0.70_threshold")

        # Document consistency
        if not source_doc:
            reasons.append("missing_source_document_id")

        if not target_doc:
            reasons.append("missing_target_document_id")

        # Sentence IDs
        if not source_id:
            reasons.append("missing_source_sentence_id")

        if not target_id:
            reasons.append("missing_target_sentence_id")

        # Context
        if not context:
            reasons.append("no_context")

        if len(context) != len(context_ids):
            reasons.append("context_id_count_mismatch")

        # Exact equality
        if source.strip() == reference.strip():
            reasons.append("source_equals_reference")

        # Very short material
        if len(source.strip()) < 5:
            reasons.append("very_short_source")

        if len(reference.strip()) < 5:
            reasons.append("very_short_reference")

        # Suspicious lexical overlap / likely metadata-like content
        metadata_terms = [
            "comments",
            "subscribe",
            "http://",
            "https://",
            "facebook",
            "email",
            "copyright",
        ]

        source_lower = source.lower()
        reference_lower = reference.lower()

        if any(term in source_lower for term in metadata_terms):
            reasons.append("metadata_like_source")

        if any(term in reference_lower for term in metadata_terms):
            reasons.append("metadata_like_reference")

        # Context relevance heuristic:
        # If context and current sentence share no alphanumeric
        # tokens, flag it for manual inspection.
        source_tokens = {
            token.strip(".,!?():;\"'").lower()
            for token in source.split()
            if token.strip(".,!?():;\"'")
        }

        context_tokens = {
            token.strip(".,!?():;\"'").lower()
            for sentence in context
            for token in sentence.split()
            if token.strip(".,!?():;\"'")
        }

        shared_tokens = source_tokens.intersection(context_tokens)

        if context and not shared_tokens:
            reasons.append("low_lexical_overlap_with_context")

        diagnostics.append(
            {
                "example_index": example["example_index"],
                "source_document_id": source_doc,
                "target_document_id": target_doc,
                "source_sentence_id": source_id,
                "target_sentence_id": target_id,
                "alignment_score": score,
                "context_sentence_ids": context_ids,
                "source": source,
                "reference": reference,
                "context": context,
                "baseline_hypothesis": example.get(
                    "baseline_hypothesis"
                ),
                "context_hypothesis": example.get(
                    "context_hypothesis"
                ),
                "suspicious": bool(reasons),
                "suspicious_reasons": reasons,
            }
        )

    suspicious = [
        item for item in diagnostics
        if item["suspicious"]
    ]

    clean = [
        item for item in diagnostics
        if not item["suspicious"]
    ]

    report = {
        "run_id": "20261006T051941Z-8768e7db",
        "total_examples": len(diagnostics),
        "suspicious_examples": len(suspicious),
        "clean_examples": len(clean),
        "diagnostics": diagnostics,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Total examples      : {len(diagnostics)}")
    print(f"Potentially suspect : {len(suspicious)}")
    print(f"Currently clean     : {len(clean)}")
    print()

    print("=" * 90)
    print("SUSPICIOUS EXAMPLES")
    print("=" * 90)

    for item in suspicious:
        print(
            f"Example {item['example_index']:>2} | "
            f"score={item['alignment_score']:.3f} | "
            f"{', '.join(item['suspicious_reasons'])}"
        )
        print(f"  Source    : {item['source']}")
        print(f"  Reference : {item['reference']}")
        print()

    print("=" * 90)
    print(f"Saved diagnostic: {OUTPUT_PATH}")
    print("=" * 90)


if __name__ == "__main__":
    main()