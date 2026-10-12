from __future__ import annotations

import json
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF


RUN_DIR = Path(
    "experiments/evaluation/dochplt/20261007T152623Z-95a19f84"
)

EXAMPLES_PATH = RUN_DIR / "examples.json"
OUTPUT_PATH = RUN_DIR / "win_analysis.json"

BLEU_METRIC = BLEU(effective_order=True)
CHRF_METRIC = CHRF()


def sentence_scores(prediction: str, reference: str) -> dict:
    bleu = BLEU_METRIC.sentence_score(
        prediction,
        [reference],
    ).score

    chrf = CHRF_METRIC.sentence_score(
        prediction,
        [reference],
    ).score

    return {
        "bleu": bleu,
        "chrf": chrf,
    }


def classify(baseline_score: float, adapter_score: float) -> str:
    tolerance = 1e-9

    if adapter_score > baseline_score + tolerance:
        return "adapter_win"

    if adapter_score < baseline_score - tolerance:
        return "baseline_win"

    return "tie"


def main() -> None:
    if not EXAMPLES_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation examples not found: {EXAMPLES_PATH}"
        )

    with EXAMPLES_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        examples = json.load(f)

    if isinstance(examples, dict):
        if "examples" in examples:
            examples = examples["examples"]
        else:
            raise ValueError(
                "examples.json does not contain an 'examples' list."
            )

    results = []

    bleu_wins = 0
    bleu_losses = 0
    bleu_ties = 0

    chrf_wins = 0
    chrf_losses = 0
    chrf_ties = 0

    both_wins = 0
    both_losses = 0

    total_bleu_change = 0.0
    total_chrf_change = 0.0

    for index, example in enumerate(examples, start=1):
        reference = (
            example.get("reference")
            or example.get("target_reference")
            or example.get("target_sentence")
            or ""
        )

        baseline_prediction = (
            example.get("baseline_prediction")
            or example.get("baseline_translation")
            or ""
        )

        adapter_prediction = (
            example.get("adapter_prediction")
            or example.get("context_adapter_prediction")
            or example.get("context_prediction")
            or ""
        )

        baseline_scores = sentence_scores(
            baseline_prediction,
            reference,
        )

        adapter_scores = sentence_scores(
            adapter_prediction,
            reference,
        )

        bleu_change = (
            adapter_scores["bleu"]
            - baseline_scores["bleu"]
        )

        chrf_change = (
            adapter_scores["chrf"]
            - baseline_scores["chrf"]
        )

        bleu_result = classify(
            baseline_scores["bleu"],
            adapter_scores["bleu"],
        )

        chrf_result = classify(
            baseline_scores["chrf"],
            adapter_scores["chrf"],
        )

        if bleu_result == "adapter_win":
            bleu_wins += 1
        elif bleu_result == "baseline_win":
            bleu_losses += 1
        else:
            bleu_ties += 1

        if chrf_result == "adapter_win":
            chrf_wins += 1
        elif chrf_result == "baseline_win":
            chrf_losses += 1
        else:
            chrf_ties += 1

        if (
            bleu_result == "adapter_win"
            and chrf_result == "adapter_win"
        ):
            both_wins += 1

        if (
            bleu_result == "baseline_win"
            and chrf_result == "baseline_win"
        ):
            both_losses += 1

        total_bleu_change += bleu_change
        total_chrf_change += chrf_change

        results.append(
            {
                "evaluation_index": example.get(
                    "evaluation_index",
                    index,
                ),
                "source_sentence": example.get(
                    "source_sentence",
                    example.get("source", ""),
                ),
                "reference": reference,
                "baseline_prediction": baseline_prediction,
                "adapter_prediction": adapter_prediction,
                "baseline_bleu": baseline_scores["bleu"],
                "adapter_bleu": adapter_scores["bleu"],
                "bleu_change": bleu_change,
                "bleu_result": bleu_result,
                "baseline_chrf": baseline_scores["chrf"],
                "adapter_chrf": adapter_scores["chrf"],
                "chrf_change": chrf_change,
                "chrf_result": chrf_result,
            }
        )

    count = len(results)

    summary = {
        "evaluation_run": str(RUN_DIR),
        "examples_analyzed": count,
        "metric_type": (
            "sentence-level diagnostic metrics; "
            "not the primary corpus-level evaluation"
        ),
        "bleu": {
            "adapter_wins": bleu_wins,
            "baseline_wins": bleu_losses,
            "ties": bleu_ties,
            "adapter_win_rate_percent": (
                100.0 * bleu_wins / count if count else 0.0
            ),
        },
        "chrf": {
            "adapter_wins": chrf_wins,
            "baseline_wins": chrf_losses,
            "ties": chrf_ties,
            "adapter_win_rate_percent": (
                100.0 * chrf_wins / count if count else 0.0
            ),
        },
        "both_metrics": {
            "adapter_wins_both": both_wins,
            "baseline_wins_both": both_losses,
        },
        "average_sentence_level_change": {
            "bleu": (
                total_bleu_change / count
                if count
                else 0.0
            ),
            "chrf": (
                total_chrf_change / count
                if count
                else 0.0
            ),
        },
        "results": results,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 70)
    print("CONTEXT ADAPTER — PER-EXAMPLE WIN ANALYSIS")
    print("=" * 70)

    print(f"Examples analyzed : {count}")

    print("\nBLEU")
    print(f"  Adapter wins    : {bleu_wins}")
    print(f"  Baseline wins   : {bleu_losses}")
    print(f"  Ties            : {bleu_ties}")
    print(
        f"  Adapter win rate: "
        f"{100.0 * bleu_wins / count:.2f}%"
        if count
        else "  Adapter win rate: 0.00%"
    )

    print("\nchrF++")
    print(f"  Adapter wins    : {chrf_wins}")
    print(f"  Baseline wins   : {chrf_losses}")
    print(f"  Ties            : {chrf_ties}")
    print(
        f"  Adapter win rate: "
        f"{100.0 * chrf_wins / count:.2f}%"
        if count
        else "  Adapter win rate: 0.00%"
    )

    print("\nBoth BLEU + chrF++")
    print(f"  Adapter wins both : {both_wins}")
    print(f"  Baseline wins both: {both_losses}")

    print("\nAverage sentence-level change")
    print(
        f"  BLEU   : "
        f"{total_bleu_change / count:.4f}"
        if count
        else "  BLEU   : 0.0000"
    )
    print(
        f"  chrF++ : "
        f"{total_chrf_change / count:.4f}"
        if count
        else "  chrF++ : 0.0000"
    )

    print("\nTop 10 adapter improvements by chrF++")
    print("-" * 70)

    top_improvements = sorted(
        results,
        key=lambda x: x["chrf_change"],
        reverse=True,
    )[:10]

    for item in top_improvements:
        print(
            f"#{item['evaluation_index']:>3} "
            f"chrF change={item['chrf_change']:+.2f} "
            f"BLEU change={item['bleu_change']:+.2f}"
        )

    print("\nTop 10 adapter regressions by chrF++")
    print("-" * 70)

    top_regressions = sorted(
        results,
        key=lambda x: x["chrf_change"],
    )[:10]

    for item in top_regressions:
        print(
            f"#{item['evaluation_index']:>3} "
            f"chrF change={item['chrf_change']:+.2f} "
            f"BLEU change={item['bleu_change']:+.2f}"
        )

    print("\nSaved:")
    print(OUTPUT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()