from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "evaluation"
    / "dochplt"
    / "20261007T163620Z-final-100"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Final validation results
# ---------------------------------------------------------------------

systems = [
    "NLLB-200",
    "Context Adapter",
]

bleu_scores = [
    7.34970429,
    13.98655412,
]

chrf_scores = [
    25.10662704,
    31.93517151,
]


# ---------------------------------------------------------------------
# Graph 1 — BLEU
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 5))

bars = plt.bar(
    systems,
    bleu_scores,
)

plt.title(
    "NLLB-200 vs Context Adapter — BLEU"
)

plt.ylabel(
    "BLEU"
)

plt.xlabel(
    "Translation System"
)

plt.ylim(
    0,
    max(bleu_scores) * 1.25,
)

for bar, value in zip(
    bars,
    bleu_scores,
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "final_bleu_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------
# Graph 2 — chrF++
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 5))

bars = plt.bar(
    systems,
    chrf_scores,
)

plt.title(
    "NLLB-200 vs Context Adapter — chrF++"
)

plt.ylabel(
    "chrF++"
)

plt.xlabel(
    "Translation System"
)

plt.ylim(
    0,
    max(chrf_scores) * 1.25,
)

for bar, value in zip(
    bars,
    chrf_scores,
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "final_chrf_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------
# Context-window diagnostic
# ---------------------------------------------------------------------

context_windows = [
    "NLLB\nBaseline",
    "Adapter\nk=0",
    "Adapter\nk=1",
    "Adapter\nk=2",
]

window_bleu = [
    7.34970429,
    14.83014381,
    14.55826863,
    13.98655412,
]

window_chrf = [
    25.10662704,
    28.19750966,
    31.73087789,
    31.93517151,
]


# ---------------------------------------------------------------------
# Graph 3 — Context window
# ---------------------------------------------------------------------

x = range(
    len(context_windows)
)

width = 0.35

plt.figure(figsize=(10, 6))

bleu_positions = [
    value - width / 2
    for value in x
]

chrf_positions = [
    value + width / 2
    for value in x
]

bleu_bars = plt.bar(
    bleu_positions,
    window_bleu,
    width=width,
    label="BLEU",
)

chrf_bars = plt.bar(
    chrf_positions,
    window_chrf,
    width=width,
    label="chrF++",
)

plt.title(
    "Context-Window Diagnostic"
)

plt.ylabel(
    "Score"
)

plt.xlabel(
    "Configuration"
)

plt.xticks(
    list(x),
    context_windows,
)

plt.legend()

plt.ylim(
    0,
    max(window_chrf) * 1.25,
)


for bar, value in zip(
    bleu_bars,
    window_bleu,
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )


for bar, value in zip(
    chrf_bars,
    window_chrf,
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "context_window_diagnostic.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------

print("=" * 75)
print(
    "FINAL RESULT GRAPHS CREATED"
)
print("=" * 75)

print()
print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Created files:"
)

print(
    "1. final_bleu_comparison.png"
)

print(
    "2. final_chrf_comparison.png"
)

print(
    "3. context_window_diagnostic.png"
)

print("=" * 75)