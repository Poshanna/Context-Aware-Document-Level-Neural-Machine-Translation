"""
Evaluate semantic context selection using the existing project
ContextConditionedTranslationService.

Important:
- Uses the existing NLLB-200 backend.
- Semantic retrieval is source-side only.
- Target/reference translations are never used for context ranking.
- Evaluation uses the same 12 manually reviewed DocHPLT examples.
- This is prefix-conditioning around a sentence-level NLLB model,
  NOT a native document-level NMT architecture.
"""

from __future__ import annotations

import json
import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ============================================================================
# PROJECT IMPORTS
# ============================================================================

from app.services.context_conditioned_translation_service import (
    ContextConditionedTranslationService,
)
from app.services.semantic_context_retriever import (
    SemanticContextRetriever,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

MANIFEST_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "evaluation"
    / "dochplt"
    / "final_eval_manifest.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "evaluation"
    / "dochplt"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "te"

TOP_K = 2
MIN_SIMILARITY = 0.0

MAX_EXAMPLES = 12


# ============================================================================
# HELPERS
# ============================================================================

def clean_text(value) -> str:
    if value is None:
        return ""

    return str(value).strip()


def safe_float(value):
    try:
        value = float(value)

        if math.isnan(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )


def compute_bleu(references, hypotheses) -> float:
    import sacrebleu

    result = sacrebleu.corpus_bleu(
        hypotheses,
        [references],
    )

    return float(result.score)


def compute_chrf(references, hypotheses) -> float:
    import sacrebleu

    result = sacrebleu.corpus_chrf(
        hypotheses,
        [references],
        word_order=2,
    )

    return float(result.score)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("DocHPLT Semantic Context Evaluation")
    print("=" * 80)

    print()
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Top-K: {TOP_K}")
    print(f"Minimum similarity: {MIN_SIMILARITY}")
    print(f"Direction: {SOURCE_LANGUAGE} -> {TARGET_LANGUAGE}")

    # ------------------------------------------------------------------------
    # Load evaluation manifest
    # ------------------------------------------------------------------------

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    examples = manifest.get("examples", [])

    if not examples:
        raise ValueError(
            "No examples found in final_eval_manifest.json."
        )

    examples = examples[:MAX_EXAMPLES]

    print()
    print(f"Loaded {len(examples)} evaluation examples.")

    # ------------------------------------------------------------------------
    # Initialize existing project translation service
    # ------------------------------------------------------------------------

    print()
    print("Initializing existing NLLB context-conditioned service...")

    translation_service = ContextConditionedTranslationService()

    print("Translation service initialized.")

    # ------------------------------------------------------------------------
    # Initialize semantic retriever
    # ------------------------------------------------------------------------

    print()
    print("Loading semantic context retriever...")

    retriever = SemanticContextRetriever(
        model_name=EMBEDDING_MODEL,
        top_k=TOP_K,
        min_similarity=MIN_SIMILARITY,
    )

    # Explicitly load the embedding model before evaluation.
    retriever._load_model()

    print("Semantic retriever loaded.")

    # ------------------------------------------------------------------------
    # Evaluation containers
    # ------------------------------------------------------------------------

    references = []
    hypotheses = []

    results = []
    failures = []

    # ------------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("Running semantic-context evaluation")
    print("=" * 80)

    for position, example in enumerate(
        examples,
        start=1,
    ):

        evaluation_index = example.get(
            "evaluation_index",
            position,
        )

        current_sentence = clean_text(
            example.get("source_sentence")
        )

        target_reference = clean_text(
            example.get("target_reference")
        )

        raw_context = example.get(
            "context",
            [],
        )

        # --------------------------------------------------------------------
        # Validate source/reference
        # --------------------------------------------------------------------

        if not current_sentence:
            failures.append(
                {
                    "evaluation_index": evaluation_index,
                    "reason": "empty_source_sentence",
                }
            )
            continue

        if not target_reference:
            failures.append(
                {
                    "evaluation_index": evaluation_index,
                    "reason": "empty_target_reference",
                }
            )
            continue

        # --------------------------------------------------------------------
        # Convert manifest context to retriever format
        # --------------------------------------------------------------------

        preceding_sentences = []

        for item in raw_context:

            if not isinstance(item, dict):
                continue

            sentence_id = item.get("id")
            text_value = clean_text(
                item.get("text")
            )

            if not text_value:
                continue

            preceding_sentences.append(
                {
                    "sentence_id": sentence_id,
                    "text": text_value,
                }
            )

        # --------------------------------------------------------------------
        # Semantic retrieval
        # --------------------------------------------------------------------

        selected_candidates = retriever.retrieve(
            current_sentence=current_sentence,
            preceding_sentences=preceding_sentences,
        )

        selected_context = [
            candidate.text
            for candidate in selected_candidates
        ]

        selected_metadata = [
            {
                "sentence_id": candidate.sentence_id,
                "text": candidate.text,
                "similarity": safe_float(
                    candidate.similarity
                ),
            }
            for candidate in selected_candidates
        ]

        # --------------------------------------------------------------------
        # Display example
        # --------------------------------------------------------------------

        print()
        print("-" * 80)
        print(
            f"Example {position}/{len(examples)}"
        )
        print("-" * 80)

        print(
            f"Evaluation index: {evaluation_index}"
        )

        print()
        print("CURRENT SENTENCE:")
        print(current_sentence)

        print()
        print("SEMANTICALLY SELECTED CONTEXT:")

        if selected_metadata:

            for rank, item in enumerate(
                selected_metadata,
                start=1,
            ):

                similarity = item["similarity"]

                if similarity is None:
                    similarity_text = "N/A"
                else:
                    similarity_text = (
                        f"{similarity:.6f}"
                    )

                print(
                    f"  {rank}. "
                    f"[{item['sentence_id']}] "
                    f"similarity={similarity_text}"
                )

                print(
                    f"     {item['text']}"
                )

        else:
            print("  [none]")

        # --------------------------------------------------------------------
        # Translation using EXISTING project service
        # --------------------------------------------------------------------

        try:

            translation_result = (
                translation_service.translate_sentence(
                    source_lang=SOURCE_LANGUAGE,
                    target_lang=TARGET_LANGUAGE,
                    current_sentence=current_sentence,
                    preceding_context=selected_context,
                    context_window=len(selected_context),
                    model="nllb200",
                )
            )

            hypothesis = clean_text(
                translation_result.get(
                    "translation"
                )
            )

            if not hypothesis:
                raise RuntimeError(
                    "Translation returned an empty string."
                )

        except Exception as exc:

            failures.append(
                {
                    "evaluation_index": evaluation_index,
                    "reason": "translation_failed",
                    "error": str(exc),
                }
            )

            print()
            print("TRANSLATION FAILED:")
            print(str(exc))

            continue

        # --------------------------------------------------------------------
        # Display translation
        # --------------------------------------------------------------------

        print()
        print("REFERENCE:")
        print(target_reference)

        print()
        print("SEMANTIC-CONTEXT OUTPUT:")
        print(hypothesis)

        # --------------------------------------------------------------------
        # Store evaluation data
        # --------------------------------------------------------------------

        references.append(
            target_reference
        )

        hypotheses.append(
            hypothesis
        )

        results.append(
            {
                "evaluation_index": evaluation_index,
                "source_sentence": current_sentence,
                "target_reference": target_reference,
                "available_context": preceding_sentences,
                "selected_context": selected_metadata,
                "conditioned_input": translation_result.get(
                    "formatted_input"
                ),
                "hypothesis": hypothesis,
                "model_backend": translation_result.get(
                    "model_backend"
                ),
                "translation_model_type": translation_result.get(
                    "translation_model_type"
                ),
            }
        )

    # =========================================================================
    # METRICS
    # =========================================================================

    if not references:
        raise RuntimeError(
            "No successful translations were produced."
        )

    bleu = compute_bleu(
        references,
        hypotheses,
    )

    chrf = compute_chrf(
        references,
        hypotheses,
    )

    print()
    print("=" * 80)
    print("Evaluation Metrics")
    print("=" * 80)

    print()
    print(
        f"Examples evaluated: {len(references)}"
    )

    print(
        f"Failures: {len(failures)}"
    )

    print(
        f"BLEU: {bleu:.8f}"
    )

    print(
        f"chrF++: {chrf:.8f}"
    )

    # =========================================================================
    # EXISTING EXPERIMENT RESULTS
    # =========================================================================

    baseline_bleu = 41.27782044840121
    baseline_chrf = 53.482361839304126

    naive_prefix_bleu = 10.21926721578879
    naive_prefix_chrf = 37.024279422389014

    semantic_vs_baseline_bleu = (
        bleu - baseline_bleu
    )

    semantic_vs_baseline_chrf = (
        chrf - baseline_chrf
    )

    semantic_vs_naive_bleu = (
        bleu - naive_prefix_bleu
    )

    semantic_vs_naive_chrf = (
        chrf - naive_prefix_chrf
    )

    # =========================================================================
    # COMPARISON
    # =========================================================================

    print()
    print("=" * 80)
    print("Three-Way Comparison")
    print("=" * 80)

    print()
    print(
        f"{'Configuration':<40}"
        f"{'BLEU':>15}"
        f"{'chrF++':>15}"
    )

    print("-" * 70)

    print(
        f"{'Sentence-level NLLB baseline':<40}"
        f"{baseline_bleu:>15.4f}"
        f"{baseline_chrf:>15.4f}"
    )

    print(
        f"{'Naive raw-prefix context':<40}"
        f"{naive_prefix_bleu:>15.4f}"
        f"{naive_prefix_chrf:>15.4f}"
    )

    print(
        f"{'Semantic-selected context':<40}"
        f"{bleu:>15.4f}"
        f"{chrf:>15.4f}"
    )

    print()
    print(
        "Semantic context vs baseline:"
    )

    print(
        f"  BLEU change: "
        f"{semantic_vs_baseline_bleu:+.4f}"
    )

    print(
        f"  chrF++ change: "
        f"{semantic_vs_baseline_chrf:+.4f}"
    )

    print()
    print(
        "Semantic context vs naive prefix:"
    )

    print(
        f"  BLEU change: "
        f"{semantic_vs_naive_bleu:+.4f}"
    )

    print(
        f"  chrF++ change: "
        f"{semantic_vs_naive_chrf:+.4f}"
    )

    # =========================================================================
    # CONCLUSION
    # =========================================================================

    if (
        bleu > baseline_bleu
        and chrf > baseline_chrf
    ):

        conclusion = (
            "The semantic-context configuration improved both BLEU "
            "and chrF++ over the sentence-level NLLB-200 baseline "
            "on this manually reviewed DocHPLT evaluation subset."
        )

    elif (
        bleu > baseline_bleu
        or chrf > baseline_chrf
    ):

        conclusion = (
            "The semantic-context configuration improved one metric "
            "but not both over the sentence-level NLLB-200 baseline "
            "on this manually reviewed DocHPLT evaluation subset."
        )

    else:

        conclusion = (
            "The semantic-context configuration did not improve "
            "the evaluation metrics over the sentence-level NLLB-200 "
            "baseline on this manually reviewed DocHPLT evaluation subset."
        )

    print()
    print(
        "CONCLUSION:"
    )

    print(
        conclusion
    )

    print()
    print(
        "Important: this is a manually reviewed deterministic "
        "subset from a DocHPLT training shard, not an official "
        "DocHPLT test set."
    )

    print(
        "Important: semantic context is supplied through "
        "prefix-conditioning to a sentence-level NLLB-200 model. "
        "This is not a native document-level NMT architecture."
    )

    # =========================================================================
    # SAVE ARTIFACTS
    # =========================================================================

    run_id = (
        f"{utc_timestamp()}-"
        f"{uuid.uuid4().hex[:8]}"
    )

    run_dir = (
        OUTPUT_ROOT
        / run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
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
        "model_backend": "nllb200",
        "direction": "en-te",
        "embedding_model": EMBEDDING_MODEL,
        "top_k": TOP_K,
        "min_similarity": MIN_SIMILARITY,
        "source_side_context_only": True,
        "target_reference_used_for_ranking": False,
        "examples": len(references),
        "failures": len(failures),
        "bleu": bleu,
        "chrf_plus_plus": chrf,
        "baseline_bleu": baseline_bleu,
        "baseline_chrf_plus_plus": baseline_chrf,
        "naive_prefix_bleu": naive_prefix_bleu,
        "naive_prefix_chrf_plus_plus": naive_prefix_chrf,
        "semantic_vs_baseline_bleu": (
            semantic_vs_baseline_bleu
        ),
        "semantic_vs_baseline_chrf_plus_plus": (
            semantic_vs_baseline_chrf
        ),
        "semantic_vs_naive_prefix_bleu": (
            semantic_vs_naive_bleu
        ),
        "semantic_vs_naive_prefix_chrf_plus_plus": (
            semantic_vs_naive_chrf
        ),
        "conclusion": conclusion,
        "run_id": run_id,
    }

    with (
        run_dir / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with (
        run_dir / "examples.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with (
        run_dir / "failures.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            failures,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 80)
    print("Artifacts saved")
    print("=" * 80)

    print()
    print("Run directory:")
    print(run_dir)

    print()
    print("Summary:")
    print(
        run_dir / "summary.json"
    )

    print()
    print("Examples:")
    print(
        run_dir / "examples.json"
    )

    print()
    print("Failures:")
    print(
        run_dir / "failures.json"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()