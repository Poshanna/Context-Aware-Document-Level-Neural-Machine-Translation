from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "evaluation"
    / "dochplt"
    / "20261007T163620Z-final-1000"
)

INPUT_PATH = (
    EVALUATION_DIR
    / "error_analysis.json"
)

OUTPUT_PATH = (
    EVALUATION_DIR
    / "qualitative_examples.json"
)


# ---------------------------------------------------------------------
# Terms that should not appear in report-ready examples
# ---------------------------------------------------------------------

EXCLUDED_TERMS = [
    "porn",
    "sex",
    "xxx",
    "nude",
    "adult video",
]


def contains_excluded_content(text: str) -> bool:

    lowered = text.lower()

    return any(
        term in lowered
        for term in EXCLUDED_TERMS
    )


def contains_url(text: str) -> bool:

    return bool(
        re.search(
            r"https?://|www\.",
            text,
            re.IGNORECASE,
        )
    )


def looks_like_fragment(text: str) -> bool:

    stripped = text.strip()

    if len(stripped) < 8:
        return True

    return False


def is_clean_example(item: dict) -> bool:

    source = item[
        "source_sentence"
    ]

    reference = item[
        "reference"
    ]

    baseline = item[
        "baseline_prediction"
    ]

    adapter = item[
        "adapter_prediction"
    ]

    all_text = " ".join(
        [
            source,
            reference,
            baseline,
            adapter,
        ]
    )

    if contains_excluded_content(
        all_text
    ):
        return False

    if contains_url(source):
        return False

    if looks_like_fragment(source):
        return False

    if not reference.strip():
        return False

    if not baseline.strip():
        return False

    if not adapter.strip():
        return False

    return True


def format_item(
    item: dict,
    rank: int,
    category: str,
) -> dict:

    return {
        "rank": rank,
        "category": category,
        "evaluation_index":
            item["evaluation_index"],
        "source":
            item["source_sentence"],
        "reference":
            item["reference"],
        "baseline":
            item["baseline_prediction"],
        "context_adapter":
            item["adapter_prediction"],
        "bleu_change":
            item["bleu_delta"],
        "chrf_change":
            item["chrf_delta"],
        "selected_context":
            item.get(
                "selected_context",
                [],
            ),
    }


def main():

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Missing:\n{INPUT_PATH}"
        )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        analysis = json.load(f)

    examples = analysis[
        "all_examples"
    ]

    clean = [
        item
        for item in examples
        if is_clean_example(item)
    ]

    # -------------------------------------------------------------
    # Strong improvements
    # -------------------------------------------------------------

    improvements = sorted(
        clean,
        key=lambda x: (
            x["chrf_delta"]
            + x["bleu_delta"]
        ),
        reverse=True,
    )

    selected_improvements = (
        improvements[:5]
    )

    # -------------------------------------------------------------
    # Strong regressions
    # -------------------------------------------------------------

    regressions = sorted(
        clean,
        key=lambda x: (
            x["chrf_delta"]
            + x["bleu_delta"]
        ),
    )

    selected_regressions = (
        regressions[:5]
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    output = {
        "dataset": "DocHPLT",
        "evaluation_examples":
            len(examples),
        "clean_examples_available":
            len(clean),
        "note":
            "Automatically selected report-ready "
            "qualitative examples. Adult-content "
            "examples and URL-containing examples "
            "were excluded.",
        "strong_improvements": [
            format_item(
                item,
                index + 1,
                "improvement",
            )
            for index, item
            in enumerate(
                selected_improvements
            )
        ],
        "strong_regressions": [
            format_item(
                item,
                index + 1,
                "regression",
            )
            for index, item
            in enumerate(
                selected_regressions
            )
        ],
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -------------------------------------------------------------
    # Print
    # -------------------------------------------------------------

    print("=" * 80)
    print(
        "REPORT-READY QUALITATIVE EXAMPLES"
    )
    print("=" * 80)

    print()
    print(
        f"Total examples: "
        f"{len(examples)}"
    )

    print(
        f"Clean examples: "
        f"{len(clean)}"
    )

    print()
    print("=" * 80)
    print("TOP IMPROVEMENTS")
    print("=" * 80)

    for item in selected_improvements:

        print()
        print(
            f"Example #{item['evaluation_index']}"
        )

        print(
            f"chrF++ Δ: "
            f"{item['chrf_delta']:+.4f}"
        )

        print(
            f"Source: "
            f"{item['source_sentence']}"
        )

        print(
            f"Reference: "
            f"{item['reference']}"
        )

        print(
            f"Baseline: "
            f"{item['baseline_prediction']}"
        )

        print(
            f"Adapter: "
            f"{item['adapter_prediction']}"
        )

    print()
    print("=" * 80)
    print("TOP REGRESSIONS")
    print("=" * 80)

    for item in selected_regressions:

        print()
        print(
            f"Example #{item['evaluation_index']}"
        )

        print(
            f"chrF++ Δ: "
            f"{item['chrf_delta']:+.4f}"
        )

        print(
            f"Source: "
            f"{item['source_sentence']}"
        )

        print(
            f"Reference: "
            f"{item['reference']}"
        )

        print(
            f"Baseline: "
            f"{item['baseline_prediction']}"
        )

        print(
            f"Adapter: "
            f"{item['adapter_prediction']}"
        )

    print()
    print("=" * 80)
    print(
        "Saved to:"
    )
    print(
        OUTPUT_PATH
    )
    print("=" * 80)


if __name__ == "__main__":
    main()