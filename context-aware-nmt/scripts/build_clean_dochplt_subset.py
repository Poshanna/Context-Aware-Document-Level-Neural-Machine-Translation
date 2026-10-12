import json
import re
from pathlib import Path


RUN_DIR = Path(
    "experiments/evaluation/dochplt/"
    "20261006T051941Z-8768e7db"
)

EXAMPLES_PATH = RUN_DIR / "examples.json"
OUTPUT_PATH = RUN_DIR / "clean_subset.json"

MIN_ALIGNMENT_SCORE = 0.70
MIN_SOURCE_CHARS = 10
MIN_REFERENCE_CHARS = 2


METADATA_PATTERNS = [
    r"\bfacebook comments\b",
    r"\bpost comments\b",
    r"\bsubscribe to\b",
    r"\bjavascript enabled\b",
    r"\bour email is\b",
    r"\bcomments \(atom\)",
    r"\bdisclaimer:",
    r"\bhttp://",
    r"\bhttps://",
]


def is_metadata_like(text: str) -> bool:
    text_lower = text.lower().strip()

    return any(
        re.search(pattern, text_lower)
        for pattern in METADATA_PATTERNS
    )


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def main():
    if not EXAMPLES_PATH.exists():
        raise FileNotFoundError(
            f"Missing examples file: {EXAMPLES_PATH}"
        )

    with EXAMPLES_PATH.open("r", encoding="utf-8") as f:
        examples = json.load(f)

    clean = []
    rejected = []

    seen_pairs = set()

    for example in examples:
        reasons = []

        source = example.get("current_source_sentence", "").strip()
        reference = example.get("target_reference", "").strip()

        source_doc = example.get("source_document_id")
        target_doc = example.get("target_document_id")

        source_id = example.get("current_source_sentence_id")
        target_id = example.get("target_sentence_id")

        context_ids = example.get("context_sentence_ids", [])
        context_sentences = example.get("context_sentences", [])

        score = example.get("alignment_score")

        # ---------------------------------------------------------
        # Basic validity
        # ---------------------------------------------------------

        if not source:
            reasons.append("empty_source")

        if not reference:
            reasons.append("empty_reference")

        if not source_doc or not target_doc:
            reasons.append("missing_document_id")

        if not source_id or not target_id:
            reasons.append("missing_sentence_id")

        if score is None:
            reasons.append("missing_alignment_score")
        elif score < MIN_ALIGNMENT_SCORE:
            reasons.append("alignment_below_threshold")

        # ---------------------------------------------------------
        # Translation-pair quality
        # ---------------------------------------------------------

        if normalize(source) == normalize(reference):
            reasons.append("source_equals_reference")

        if len(source) < MIN_SOURCE_CHARS:
            reasons.append("source_too_short")

        if len(reference) < MIN_REFERENCE_CHARS:
            reasons.append("reference_too_short")

        if is_metadata_like(source):
            reasons.append("metadata_like_source")

        if is_metadata_like(reference):
            reasons.append("metadata_like_reference")

        # ---------------------------------------------------------
        # Context requirements
        # ---------------------------------------------------------

        if not context_ids:
            reasons.append("missing_context")

        if len(context_ids) != len(context_sentences):
            reasons.append("context_id_sentence_mismatch")

        # Context must be actual text.
        if context_sentences:
            if any(not str(x).strip() for x in context_sentences):
                reasons.append("empty_context_sentence")

        # ---------------------------------------------------------
        # Duplicate pair detection
        # ---------------------------------------------------------

        pair_key = (
            source_doc,
            target_doc,
            normalize(source),
            normalize(reference),
        )

        if pair_key in seen_pairs:
            reasons.append("duplicate_translation_pair")

        seen_pairs.add(pair_key)

        # ---------------------------------------------------------
        # Accept/reject
        # ---------------------------------------------------------

        if reasons:
            rejected.append(
                {
                    "example_index": example.get("example_index"),
                    "reasons": reasons,
                    "source": source,
                    "reference": reference,
                    "alignment_score": score,
                }
            )
        else:
            clean.append(example)

    output = {
        "dataset": "DocHPLT",
        "source_run": str(RUN_DIR),
        "selection_policy": {
            "minimum_alignment_score": MIN_ALIGNMENT_SCORE,
            "minimum_source_chars": MIN_SOURCE_CHARS,
            "minimum_reference_chars": MIN_REFERENCE_CHARS,
            "exclude_source_equals_reference": True,
            "exclude_metadata_like_text": True,
            "require_document_ids": True,
            "require_sentence_ids": True,
            "require_preceding_context": True,
            "exclude_duplicate_translation_pairs": True,
        },
        "total_input_examples": len(examples),
        "clean_examples": len(clean),
        "rejected_examples": len(rejected),
        "clean_example_indices": [
            x["example_index"] for x in clean
        ],
        "rejected": rejected,
        "examples": clean,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 90)
    print("DocHPLT Clean Evaluation Subset")
    print("=" * 90)
    print(f"Input examples     : {len(examples)}")
    print(f"Clean examples     : {len(clean)}")
    print(f"Rejected examples  : {len(rejected)}")
    print()
    print("Clean example IDs:")
    print(
        ", ".join(
            str(x["example_index"])
            for x in clean
        )
    )
    print()
    print(f"Saved: {OUTPUT_PATH}")
    print("=" * 90)


if __name__ == "__main__":
    main()