import json
from pathlib import Path


INPUT = Path(
    "experiments/evaluation/dochplt/"
    "manual_review_sample.json"
)

OUTPUT = Path(
    "experiments/evaluation/dochplt/"
    "final_eval_manifest.json"
)

# Manually reviewed and accepted examples.
ACCEPTED_INDICES = {
    1,
    10,
    12,
    14,
    15,
    18,
    19,
    20,
    23,
    24,
    46,
    50,
}


def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing manual review file: {INPUT}"
        )

    with INPUT.open("r", encoding="utf-8") as f:
        data = json.load(f)

    selected = []

    for item in data["review_items"]:

        review_index = item["review_index"]

        if review_index in ACCEPTED_INDICES:
            example = item["example"]

            selected.append(
                {
                    "evaluation_index": len(selected) + 1,
                    "manual_review_index": review_index,
                    "source_document_id": example[
                        "source_document_id"
                    ],
                    "target_document_id": example[
                        "target_document_id"
                    ],
                    "source_sentence_id": example[
                        "source_sentence_id"
                    ],
                    "target_sentence_id": example[
                        "target_sentence_id"
                    ],
                    "source_sentence": example[
                        "source_sentence"
                    ],
                    "target_reference": example[
                        "target_reference"
                    ],
                    "context": example["context"],
                    "alignment_score": example[
                        "alignment_score"
                    ],
                    "context_relevance_score": example[
                        "context_relevance_score"
                    ],
                }
            )

    if len(selected) != len(ACCEPTED_INDICES):
        raise RuntimeError(
            "Accepted review indices do not match "
            "the available manual-review sample."
        )

    output = {
        "dataset": "DocHPLT",
        "config": "en-te",
        "split": "train",
        "evaluation_type": (
            "manually reviewed deterministic subset "
            "from a DocHPLT training shard"
        ),
        "official_test_set": False,
        "document_level_nmt": False,
        "model": "facebook/nllb-200-distilled-600M",
        "direction": "en-te",
        "alignment_threshold": 0.70,
        "context_window": 2,
        "manual_review_seed": 42,
        "source_manual_review_sample": str(INPUT),
        "selection_policy": {
            "manual_review_required": True,
            "accepted_translation_pair": True,
            "accepted_context_relevance": True,
            "target_reference_not_used_for_automated_ranking": True,
        },
        "accepted_manual_review_indices": sorted(
            ACCEPTED_INDICES
        ),
        "sample_count": len(selected),
        "examples": selected,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 90)
    print("Final DocHPLT Evaluation Manifest")
    print("=" * 90)

    print(f"Manual review items : {len(data['review_items'])}")
    print(f"Accepted examples   : {len(selected)}")
    print()
    print("Accepted review IDs:")
    print(", ".join(str(x) for x in sorted(ACCEPTED_INDICES)))
    print()

    print("Evaluation examples:")

    for example in selected:
        print(
            f"{example['evaluation_index']:2}. "
            f"review={example['manual_review_index']:2} | "
            f"alignment={example['alignment_score']:.3f} | "
            f"{example['source_sentence']}"
        )

    print()
    print(f"Saved: {OUTPUT}")
    print("=" * 90)


if __name__ == "__main__":
    main()