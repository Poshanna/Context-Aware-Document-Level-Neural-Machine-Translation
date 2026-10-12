import json
import random
from pathlib import Path


INPUT = Path(
    "experiments/evaluation/dochplt/"
    "refined_context_candidates.json"
)

OUTPUT = Path(
    "experiments/evaluation/dochplt/"
    "manual_review_sample.json"
)

SAMPLE_SIZE = 50
SEED = 42


def main():

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    with INPUT.open("r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = data["candidates"]

    # Deterministic sample from the refined candidate pool.
    rng = random.Random(SEED)

    if len(candidates) <= SAMPLE_SIZE:
        selected = list(candidates)
    else:
        selected = rng.sample(candidates, SAMPLE_SIZE)

    selected.sort(
        key=lambda x: (
            x["source_document_id"],
            x["source_sentence_id"],
            x["target_sentence_id"],
        )
    )

    review_items = []

    for index, candidate in enumerate(selected, 1):

        review_items.append(
            {
                "review_index": index,
                "example": candidate,
                "manual_decision": None,
                "review_reason": None,
            }
        )

    output = {
        "dataset": "DocHPLT",
        "config": "en-te",
        "source_pool_size": len(candidates),
        "sample_size": len(review_items),
        "seed": SEED,
        "selection_method": (
            "Deterministic random sample from refined candidate pool"
        ),
        "target_reference_not_used_for_sampling": True,
        "review_items": review_items,
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
    print("DocHPLT Manual Review Sample")
    print("=" * 90)
    print(f"Refined candidate pool : {len(candidates)}")
    print(f"Review sample size      : {len(review_items)}")
    print(f"Random seed             : {SEED}")
    print()
    print(f"Saved: {OUTPUT}")
    print("=" * 90)


if __name__ == "__main__":
    main()