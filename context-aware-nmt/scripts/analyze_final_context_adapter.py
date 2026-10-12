from __future__ import annotations

import json
import sys
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF


# ---------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# Latest final evaluation
# ---------------------------------------------------------------------

EVALUATION_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "evaluation"
    / "dochplt"
    / "20261007T163620Z-final-1000"
)

EXAMPLES_PATH = (
    EVALUATION_DIR
    / "examples.json"
)

SUMMARY_PATH = (
    EVALUATION_DIR
    / "summary.json"
)

OUTPUT_PATH = (
    EVALUATION_DIR
    / "error_analysis.json"
)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

BLEU_METRIC = BLEU(
    effective_order=True
)

CHRF_METRIC = CHRF()


def sentence_bleu(
    prediction: str,
    reference: str,
) -> float:

    return float(
        BLEU_METRIC.sentence_score(
            prediction,
            [reference],
        ).score
    )


def sentence_chrf(
    prediction: str,
    reference: str,
) -> float:

    return float(
        CHRF_METRIC.sentence_score(
            prediction,
            [reference],
        ).score
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    if not EXAMPLES_PATH.exists():

        raise FileNotFoundError(
            f"Examples file not found:\n"
            f"{EXAMPLES_PATH}"
        )

    with EXAMPLES_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        examples = json.load(f)

    print("=" * 80)
    print(
        "FINAL CONTEXT ADAPTER ERROR ANALYSIS"
    )
    print("=" * 80)

    print(
        f"Examples loaded: {len(examples)}"
    )

    results = []

    adapter_bleu_wins = 0
    baseline_bleu_wins = 0
    bleu_ties = 0

    adapter_chrf_wins = 0
    baseline_chrf_wins = 0
    chrf_ties = 0

    both_adapter_wins = 0
    both_baseline_wins = 0

    for item in examples:

        index = item[
            "evaluation_index"
        ]

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

        baseline_bleu = sentence_bleu(
            baseline,
            reference,
        )

        adapter_bleu = sentence_bleu(
            adapter,
            reference,
        )

        baseline_chrf = sentence_chrf(
            baseline,
            reference,
        )

        adapter_chrf = sentence_chrf(
            adapter,
            reference,
        )

        bleu_delta = (
            adapter_bleu
            - baseline_bleu
        )

        chrf_delta = (
            adapter_chrf
            - baseline_chrf
        )

        if adapter_bleu > baseline_bleu:
            adapter_bleu_wins += 1

        elif baseline_bleu > adapter_bleu:
            baseline_bleu_wins += 1

        else:
            bleu_ties += 1

        if adapter_chrf > baseline_chrf:
            adapter_chrf_wins += 1

        elif baseline_chrf > adapter_chrf:
            baseline_chrf_wins += 1

        else:
            chrf_ties += 1

        if (
            adapter_bleu > baseline_bleu
            and adapter_chrf > baseline_chrf
        ):
            both_adapter_wins += 1

        if (
            baseline_bleu > adapter_bleu
            and baseline_chrf > adapter_chrf
        ):
            both_baseline_wins += 1

        results.append(
            {
                "evaluation_index": index,
                "source_sentence": source,
                "reference": reference,
                "baseline_prediction": baseline,
                "adapter_prediction": adapter,
                "baseline_bleu": baseline_bleu,
                "adapter_bleu": adapter_bleu,
                "bleu_delta": bleu_delta,
                "baseline_chrf": baseline_chrf,
                "adapter_chrf": adapter_chrf,
                "chrf_delta": chrf_delta,
                "selected_context":
                    item.get(
                        "selected_context",
                        [],
                    ),
            }
        )

    # -----------------------------------------------------------------
    # Sort improvements/regressions
    # -----------------------------------------------------------------

    top_bleu_improvements = sorted(
        results,
        key=lambda x: x["bleu_delta"],
        reverse=True,
    )[:10]

    top_bleu_regressions = sorted(
        results,
        key=lambda x: x["bleu_delta"],
    )[:10]

    top_chrf_improvements = sorted(
        results,
        key=lambda x: x["chrf_delta"],
        reverse=True,
    )[:10]

    top_chrf_regressions = sorted(
        results,
        key=lambda x: x["chrf_delta"],
    )[:10]

    # -----------------------------------------------------------------
    # Average sentence-level changes
    # -----------------------------------------------------------------

    average_bleu_delta = (
        sum(
            x["bleu_delta"]
            for x in results
        )
        / len(results)
    )

    average_chrf_delta = (
        sum(
            x["chrf_delta"]
            for x in results
        )
        / len(results)
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    analysis = {

        "dataset":
            "DocHPLT",

        "evaluation_directory":
            str(EVALUATION_DIR),

        "examples":
            len(results),

        "metric_note":
            "Sentence-level BLEU and chrF++ "
            "are diagnostic comparisons. "
            "The main corpus-level results "
            "remain those reported in summary.json.",

        "wins": {

            "bleu": {
                "adapter":
                    adapter_bleu_wins,
                "baseline":
                    baseline_bleu_wins,
                "ties":
                    bleu_ties,
            },

            "chrf": {
                "adapter":
                    adapter_chrf_wins,
                "baseline":
                    baseline_chrf_wins,
                "ties":
                    chrf_ties,
            },

            "both_metrics": {
                "adapter":
                    both_adapter_wins,
                "baseline":
                    both_baseline_wins,
            },
        },

        "average_sentence_level_change": {

            "bleu":
                average_bleu_delta,

            "chrf":
                average_chrf_delta,
        },

        "top_bleu_improvements":
            top_bleu_improvements,

        "top_bleu_regressions":
            top_bleu_regressions,

        "top_chrf_improvements":
            top_chrf_improvements,

        "top_chrf_regressions":
            top_chrf_regressions,

        "all_examples":
            results,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            analysis,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------------------
    # Terminal report
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("WIN / LOSS ANALYSIS")
    print("=" * 80)

    print()
    print("Sentence BLEU:")
    print(
        f"  Adapter wins : "
        f"{adapter_bleu_wins}"
    )
    print(
        f"  Baseline wins: "
        f"{baseline_bleu_wins}"
    )
    print(
        f"  Ties         : "
        f"{bleu_ties}"
    )

    print()
    print("Sentence chrF++:")
    print(
        f"  Adapter wins : "
        f"{adapter_chrf_wins}"
    )
    print(
        f"  Baseline wins: "
        f"{baseline_chrf_wins}"
    )
    print(
        f"  Ties         : "
        f"{chrf_ties}"
    )

    print()
    print("Both metrics:")
    print(
        f"  Adapter wins both   : "
        f"{both_adapter_wins}"
    )
    print(
        f"  Baseline wins both  : "
        f"{both_baseline_wins}"
    )

    print()
    print(
        "Average sentence-level change:"
    )

    print(
        f"  BLEU   : "
        f"{average_bleu_delta:+.4f}"
    )

    print(
        f"  chrF++ : "
        f"{average_chrf_delta:+.4f}"
    )

    # -----------------------------------------------------------------
    # Top examples
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("TOP 10 chrF++ IMPROVEMENTS")
    print("=" * 80)

    for item in top_chrf_improvements:

        print()
        print(
            f"Example #{item['evaluation_index']}"
        )

        print(
            f"chrF++ change: "
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
    print("TOP 10 chrF++ REGRESSIONS")
    print("=" * 80)

    for item in top_chrf_regressions:

        print()
        print(
            f"Example #{item['evaluation_index']}"
        )

        print(
            f"chrF++ change: "
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
        "ERROR ANALYSIS SAVED"
    )
    print("=" * 80)

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()