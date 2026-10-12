from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


import pyarrow.parquet as pq


# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dochplt"
    / "en-te"
    / "train-00000-of-00030.parquet"
)

DATASET_NAME = "DocHPLT"
DATASET_CONFIG = "en-te"
DATASET_SPLIT = "train"

DATASET_REVISION = (
    "76a591ec3c014d0ac4193a3cbdfc12689616a8d3"
)

SOURCE_SHARD = "train-00000-of-00030.parquet"

DEFAULT_MAX_SAMPLES = 5
DEFAULT_SEED = 42
DEFAULT_CONTEXT_WINDOW = 2
ALIGNER_SCORE_THRESHOLD = 0.70

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "evaluation"
    / "dochplt"
)


# ============================================================================
# DATA LOADING
# ============================================================================


def _load_valid_rows_for_smoke(
    shard_path: Path,
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    """
    Load valid DocHPLT rows from the local Parquet shard.

    Only one-to-one alignments at or above the supplied alignment-score
    threshold are retained.
    """

    if not shard_path.exists():
        raise FileNotFoundError(
            f"DocHPLT shard not found: {shard_path}"
        )

    table = pq.read_table(shard_path)

    required_columns = {
        "src_doc_id",
        "tgt_doc_id",
        "lang_pair",
        "src_doc",
        "tgt_doc",
        "alignment",
    }

    missing = required_columns.difference(table.column_names)

    if missing:
        raise ValueError(
            "DocHPLT shard is missing required columns: "
            f"{sorted(missing)}"
        )

    rows = table.to_pylist()

    valid_rows: list[dict[str, Any]] = []

    for row in rows:

        if row.get("lang_pair") != DATASET_CONFIG:
            continue

        src_doc = row.get("src_doc") or {}
        tgt_doc = row.get("tgt_doc") or {}
        alignments = row.get("alignment") or []

        src_ids = src_doc.get("ids") or []
        src_sentences = src_doc.get("sentences") or []

        tgt_ids = tgt_doc.get("ids") or []
        tgt_sentences = tgt_doc.get("sentences") or []

        if not src_ids or not src_sentences:
            continue

        if not tgt_ids or not tgt_sentences:
            continue

        if len(src_ids) != len(src_sentences):
            continue

        if len(tgt_ids) != len(tgt_sentences):
            continue

        src_map = dict(zip(src_ids, src_sentences))
        tgt_map = dict(zip(tgt_ids, tgt_sentences))

        if len(src_map) != len(src_ids):
            continue

        if len(tgt_map) != len(tgt_ids):
            continue

        valid_alignments: list[dict[str, Any]] = []

        for alignment in alignments:

            if not isinstance(alignment, dict):
                continue

            src = alignment.get("src") or []
            tgt = alignment.get("tgt") or []
            score = alignment.get("aligner-score")

            # Strictly retain only one-to-one alignments.
            if len(src) != 1 or len(tgt) != 1:
                continue

            if score is None:
                continue

            try:
                score_value = float(score)
            except (TypeError, ValueError):
                continue

            if score_value < threshold:
                continue

            src_id = src[0]
            tgt_id = tgt[0]

            if src_id not in src_map:
                continue

            if tgt_id not in tgt_map:
                continue

            valid_alignments.append(
                {
                    "src": [src_id],
                    "tgt": [tgt_id],
                    "aligner-score": score_value,
                }
            )

        if not valid_alignments:
            continue

        valid_rows.append(
            {
                "src_doc_id": row.get("src_doc_id"),
                "tgt_doc_id": row.get("tgt_doc_id"),
                "lang_pair": row.get("lang_pair"),
                "src_doc": {
                    "ids": src_ids,
                    "sentences": src_sentences,
                },
                "tgt_doc": {
                    "ids": tgt_ids,
                    "sentences": tgt_sentences,
                },
                "alignment": valid_alignments,
            }
        )

    return valid_rows


# ============================================================================
# MANIFEST LOADING
# ============================================================================


def _load_manifest_examples(
    manifest_path: Path,
) -> list[SimpleNamespace]:
    """
    Load explicitly manually-reviewed examples from the final manifest.
    """

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Evaluation manifest not found: {manifest_path}"
        )

    payload = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Manifest must contain a top-level JSON object."
        )

    examples = payload.get("examples")

    if not isinstance(examples, list):
        raise ValueError(
            "Manifest must contain an 'examples' list."
        )

    if not examples:
        raise ValueError(
            "Manifest contains no evaluation examples."
        )

    selected: list[SimpleNamespace] = []

    for index, item in enumerate(examples, start=1):

        if not isinstance(item, dict):
            raise ValueError(
                f"Manifest example {index} must be a JSON object."
            )

        source_document_id = item.get("source_document_id")
        target_document_id = item.get("target_document_id")
        source_sentence_id = item.get("source_sentence_id")
        target_sentence_id = item.get("target_sentence_id")
        source_sentence = item.get("source_sentence")
        target_reference = item.get("target_reference")
        alignment_score = item.get("alignment_score")

        required_values = {
            "source_document_id": source_document_id,
            "target_document_id": target_document_id,
            "source_sentence_id": source_sentence_id,
            "target_sentence_id": target_sentence_id,
            "source_sentence": source_sentence,
            "target_reference": target_reference,
            "alignment_score": alignment_score,
        }

        missing = [
            name
            for name, value in required_values.items()
            if value is None
        ]

        if missing:
            raise ValueError(
                f"Manifest example {index} is missing: {missing}"
            )

        # ------------------------------------------------------------------
        # Context
        # ------------------------------------------------------------------

        context = item.get("context", [])

        if context is None:
            context = []

        if not isinstance(context, list):
            raise ValueError(
                f"Manifest example {index}: "
                "'context' must be a list."
            )

        context_sentence_ids: list[str] = []
        context_sentences: list[str] = []

        for context_index, context_item in enumerate(
            context,
            start=1,
        ):

            if not isinstance(context_item, dict):
                raise ValueError(
                    f"Manifest example {index}: "
                    f"context item {context_index} "
                    "must be an object."
                )

            context_id = context_item.get("id")
            context_text = context_item.get("text")

            if context_id is None:
                raise ValueError(
                    f"Manifest example {index}: "
                    f"context item {context_index} "
                    "is missing 'id'."
                )

            if context_text is None:
                raise ValueError(
                    f"Manifest example {index}: "
                    f"context item {context_index} "
                    "is missing 'text'."
                )

            context_sentence_ids.append(str(context_id))
            context_sentences.append(str(context_text))

        # ------------------------------------------------------------------
        # Alignment score
        # ------------------------------------------------------------------

        try:
            alignment_score = float(alignment_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Manifest example {index}: "
                "invalid alignment_score."
            ) from exc

        # ------------------------------------------------------------------
        # Provenance
        # ------------------------------------------------------------------

        provenance = item.get("provenance")

        if isinstance(provenance, dict):
            unit_type_value = provenance.get(
                "unit_type",
                "sentence",
            )

            if isinstance(unit_type_value, dict):
                unit_type_value = unit_type_value.get(
                    "value",
                    "sentence",
                )
        else:
            unit_type_value = "sentence"

        selected.append(
            SimpleNamespace(
                source_document_id=source_document_id,
                target_document_id=target_document_id,
                context_sentence_ids=context_sentence_ids,
                context_sentences=context_sentences,
                source_sentence_id=source_sentence_id,
                source_sentence=source_sentence,
                target_sentence_id=target_sentence_id,
                target_reference=target_reference,
                alignment_score=alignment_score,
                provenance=SimpleNamespace(
                    unit_type=SimpleNamespace(
                        value=unit_type_value
                    )
                ),
            )
        )

    return selected


# ============================================================================
# ARTIFACT DIRECTORY
# ============================================================================


def _build_run_dir(
    output_root: Path,
) -> tuple[Path, str]:
    """Create a unique evaluation artifact directory."""

    run_id = (
        datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        + "-"
        + uuid.uuid4().hex[:8]
    )

    run_dir = output_root / run_id

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    return run_dir, run_id


# ============================================================================
# METRICS
# ============================================================================


def _safe_metric_score(
    metric_name: str,
    predictions: List[str],
    references: List[str],
) -> Optional[float]:
    """
    Compute BLEU or chrF++ directly using SacreBLEU.

    BLEU and chrF++ are corpus-level metrics.

    Returns:
        Float metric score when computation succeeds.
        None only when the metric cannot be computed.
    """

    try:
        import sacrebleu

        if not predictions:
            raise ValueError(
                "No predictions were provided."
            )

        if not references:
            raise ValueError(
                "No references were provided."
            )

        if len(predictions) != len(references):
            raise ValueError(
                "Prediction/reference length mismatch: "
                f"{len(predictions)} predictions vs "
                f"{len(references)} references."
            )

        normalized_name = metric_name.strip().lower()

        if normalized_name == "bleu":

            result = sacrebleu.corpus_bleu(
                predictions,
                [references],
            )

            return float(result.score)

        if normalized_name in {"chrf++", "chrf"}:

            result = sacrebleu.corpus_chrf(
                predictions,
                [references],
                word_order=2,
            )

            return float(result.score)

        raise ValueError(
            f"Unsupported metric: {metric_name}"
        )

    except Exception as exc:

        print(
            f"[metric-error] {metric_name}: "
            f"{type(exc).__name__}: {exc}"
        )

        return None


# ============================================================================
# TRANSLATION HELPERS
# ============================================================================


def _translate_batch(
    translator: Any,
    texts: list[str],
) -> list[str]:
    """
    Translate a list of English sentences/inputs using NLLB-200.

    DocHPLT evaluation is English -> Telugu.
    """

    if not texts:
        return []

    translations = translator.translate_batch(
        texts,
        src_lang="en",
        tgt_lang="te",
    )

    if not isinstance(translations, list):
        raise RuntimeError(
            "NLLB translator returned a non-list result."
        )

    if len(translations) != len(texts):
        raise RuntimeError(
            "NLLB translator returned an unexpected number "
            "of translations."
        )

    return [
        str(translation).strip()
        for translation in translations
    ]


def _translate_baseline(
    translator: Any,
    source_sentences: list[str],
) -> list[str]:
    """
    Baseline:
    translate only the current sentence.
    """

    return _translate_batch(
        translator,
        source_sentences,
    )


def _translate_context(
    translator: Any,
    examples: list[SimpleNamespace],
) -> list[str]:
    """
    Experimental prefix-conditioned translation.

    Format:

        CONTEXT:
        previous sentence 1
        previous sentence 2

        CURRENT SENTENCE:
        current sentence

    This is an experimental prefix-conditioning approach around
    sentence-level NLLB-200.

    It is NOT native document-level NMT.
    """

    formatted_inputs: list[str] = []

    for example in examples:

        context_sentences = list(
            example.context_sentences
        )

        current_sentence = example.source_sentence

        if context_sentences:

            formatted_input = (
                "CONTEXT:\n"
                + "\n".join(context_sentences)
                + "\n\nCURRENT SENTENCE:\n"
                + current_sentence
            )

        else:

            formatted_input = (
                "CURRENT SENTENCE:\n"
                + current_sentence
            )

        formatted_inputs.append(formatted_input)

    return _translate_batch(
        translator,
        formatted_inputs,
    )


# ============================================================================
# EVALUATION
# ============================================================================


def _evaluate_examples(
    *,
    selected_examples: list[SimpleNamespace],
    translator: Any,
) -> tuple[
    list[dict[str, Any]],
    dict[str, float | None],
    dict[str, float | None],
]:

    source_sentences = [
        example.source_sentence
        for example in selected_examples
    ]

    references = [
        example.target_reference
        for example in selected_examples
    ]

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    baseline_hypotheses = _translate_baseline(
        translator,
        source_sentences,
    )

    # ------------------------------------------------------------------
    # Context-conditioned
    # ------------------------------------------------------------------

    context_hypotheses = _translate_context(
        translator,
        selected_examples,
    )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    baseline_metrics = {
        "bleu": _safe_metric_score(
            "bleu",
            baseline_hypotheses,
            references,
        ),
        "chrf++": _safe_metric_score(
            "chrf++",
            baseline_hypotheses,
            references,
        ),
    }

    context_metrics = {
        "bleu": _safe_metric_score(
            "bleu",
            context_hypotheses,
            references,
        ),
        "chrf++": _safe_metric_score(
            "chrf++",
            context_hypotheses,
            references,
        ),
    }

    # ------------------------------------------------------------------
    # Per-example output
    # ------------------------------------------------------------------

    example_rows: list[dict[str, Any]] = []

    for index, (
        example,
        baseline,
        context_hypothesis,
    ) in enumerate(
        zip(
            selected_examples,
            baseline_hypotheses,
            context_hypotheses,
        ),
        start=1,
    ):

        if example.context_sentences:

            formatted_context_input = (
                "CONTEXT:\n"
                + "\n".join(
                    example.context_sentences
                )
                + "\n\nCURRENT SENTENCE:\n"
                + example.source_sentence
            )

        else:

            formatted_context_input = (
                "CURRENT SENTENCE:\n"
                + example.source_sentence
            )

        example_rows.append(
            {
                "example_index": index,

                "source_document_id": (
                    example.source_document_id
                ),

                "target_document_id": (
                    example.target_document_id
                ),

                "source_sentence_id": (
                    example.source_sentence_id
                ),

                "target_sentence_id": (
                    example.target_sentence_id
                ),

                "context_sentence_ids": (
                    example.context_sentence_ids
                ),

                "context_sentences": (
                    example.context_sentences
                ),

                "source_sentence": (
                    example.source_sentence
                ),

                "reference": (
                    example.target_reference
                ),

                "alignment_score": (
                    example.alignment_score
                ),

                "baseline_hypothesis": (
                    baseline
                ),

                "context_hypothesis": (
                    context_hypothesis
                ),

                "context_was_supplied": bool(
                    example.context_sentences
                ),

                "formatted_context_input": (
                    formatted_context_input
                ),
            }
        )

    return (
        example_rows,
        baseline_metrics,
        context_metrics,
    )


# ============================================================================
# MAIN
# ============================================================================


def main() -> int:

    parser = argparse.ArgumentParser(
        description="Run a deterministic DocHPLT evaluation."
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=(
            "Number of examples to select when "
            "no manifest is supplied."
        ),
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=(
            "Use an explicit DocHPLT evaluation "
            "manifest instead of random sampling."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--context-window",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=ALIGNER_SCORE_THRESHOLD,
    )

    parser.add_argument(
        "--shard-path",
        type=str,
        default=str(DATASET_PATH),
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate arguments
    # ------------------------------------------------------------------

    if args.max_samples <= 0:
        raise ValueError(
            "--max-samples must be greater than zero."
        )

    if args.context_window < 0:
        raise ValueError(
            "--context-window cannot be negative."
        )

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(
            "--threshold must be between 0 and 1."
        )

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------

    shard_path = Path(args.shard_path)

    if not shard_path.is_absolute():
        shard_path = PROJECT_ROOT / shard_path

    output_root = Path(args.output_root)

    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Select evaluation examples
    # ------------------------------------------------------------------

    manifest_used: str | None = None

    if args.manifest:

        manifest_path = Path(args.manifest)

        if not manifest_path.is_absolute():
            manifest_path = (
                PROJECT_ROOT / manifest_path
            )

        selected_examples = _load_manifest_examples(
            manifest_path
        )

        manifest_used = str(manifest_path)

        evaluation_selection_mode = (
            "explicit_manual_review_manifest"
        )

        print(
            f"Loaded {len(selected_examples)} "
            "examples from manifest:"
        )

        print(
            f"  {manifest_path}"
        )

    else:

        from backend.app.evaluation.dochplt_adapter import (
            DocHPLTEvaluationAdapter,
        )

        print("Loading DocHPLT shard:")
        print(f"  {shard_path}")

        valid_rows = _load_valid_rows_for_smoke(
            shard_path,
            threshold=args.threshold,
        )

        if not valid_rows:
            raise RuntimeError(
                "No valid DocHPLT rows were found."
            )

        adapter = DocHPLTEvaluationAdapter(
            dataset_name=DATASET_NAME,
            config=DATASET_CONFIG,
            split=DATASET_SPLIT,
            revision=DATASET_REVISION,
            aligner_score_threshold=args.threshold,
        )

        subset = adapter.build_evaluation_subset(
            rows=valid_rows,
            max_samples=args.max_samples,
            seed=args.seed,
            context_window=args.context_window,
        )

        selected_examples = list(
            subset.selected_examples
        )

        evaluation_selection_mode = (
            "deterministic_adapter_sampling"
        )

        print(
            f"Selected {len(selected_examples)} "
            "evaluation examples."
        )

    if not selected_examples:
        raise RuntimeError(
            "No evaluation examples were selected."
        )

    # ------------------------------------------------------------------
    # Create run directory
    # ------------------------------------------------------------------

    run_dir, run_id = _build_run_dir(
        output_root
    )

    print()
    print("=" * 72)
    print("DocHPLT Evaluation")
    print("=" * 72)

    print(
        f"Dataset        : {DATASET_NAME}"
    )

    print(
        f"Config         : {DATASET_CONFIG}"
    )

    print(
        f"Split          : {DATASET_SPLIT}"
    )

    print(
        f"Revision       : {DATASET_REVISION}"
    )

    print(
        f"Shard          : {shard_path}"
    )

    print(
        f"Selection mode : "
        f"{evaluation_selection_mode}"
    )

    print(
        f"Sample count   : "
        f"{len(selected_examples)}"
    )

    print(
        f"Context window : "
        f"{args.context_window}"
    )

    print(
        f"Threshold      : "
        f"{args.threshold}"
    )

    print("=" * 72)

    # ------------------------------------------------------------------
    # Load registered NLLB-200 translator
    # ------------------------------------------------------------------

    from backend.app.translation import (
        registry as default_translator_registry,
    )

    default_translator_registry.populate()

    print()
    print(
        "Loading registered NLLB-200 translator..."
    )

    translator = default_translator_registry.get(
        "nllb200",
        auto_load=True,
    )

    if translator is None:
        raise RuntimeError(
            "Registered NLLB-200 translator could not be loaded."
        )

    print(
        "NLLB-200 translator loaded successfully."
    )

    print()
    print(
        "Running baseline and "
        "context-conditioned inference..."
    )

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------

    inference_failures: list[
        dict[str, Any]
    ] = []

    try:

        (
            example_rows,
            baseline_metrics,
            context_metrics,
        ) = _evaluate_examples(
            selected_examples=selected_examples,
            translator=translator,
        )

    except Exception as exc:

        inference_failures.append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        )

        example_rows = []

        baseline_metrics = {
            "bleu": None,
            "chrf++": None,
        }

        context_metrics = {
            "bleu": None,
            "chrf++": None,
        }

        print(
            "Inference/evaluation failed:"
        )

        print(
            f"  {exc}"
        )

    # ------------------------------------------------------------------
    # Build artifact
    # ------------------------------------------------------------------

    artifact = {
        "label": (
            "DocHPLT deterministic held-out "
            "evaluation subset smoke run"
        ),

        "dataset": {
            "name": DATASET_NAME,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "revision": DATASET_REVISION,
            "source_shard": SOURCE_SHARD,
            "source_path": str(shard_path),
            "aligner_score_threshold": args.threshold,
        },

        "model_name": (
            "facebook/"
            "nllb-200-distilled-600M"
        ),

        # This is a sentence-level NLLB experiment with
        # context supplied as a textual prefix.
        "document_level_evaluation": False,

        "context_conditioning": "prefix",

        "seed": args.seed,

        "context_window": args.context_window,

        "threshold": args.threshold,

        # When a manifest is used, the manifest determines
        # the actual number of evaluation examples.
        "max_samples": len(selected_examples),

        "sample_count": len(selected_examples),

        "selection_mode": evaluation_selection_mode,

        "manifest_path": manifest_used,

        "results": {
            "baseline": baseline_metrics,
            "context_conditioned": context_metrics,
        },

        "inference_failures": inference_failures,

        "run_id": run_id,

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------

    summary_path = (
        run_dir / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Save per-example results
    # ------------------------------------------------------------------

    examples_path = (
        run_dir / "examples.json"
    )

    examples_path.write_text(
        json.dumps(
            example_rows,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------

    print()

    print(
        f"Artifact saved to: "
        f"{summary_path}"
    )

    print()

    print("Results:")

    print(
        f"  Baseline BLEU   : "
        f"{baseline_metrics.get('bleu')}"
    )

    print(
        f"  Baseline chrF++ : "
        f"{baseline_metrics.get('chrf++')}"
    )

    print(
        f"  Context BLEU    : "
        f"{context_metrics.get('bleu')}"
    )

    print(
        f"  Context chrF++  : "
        f"{context_metrics.get('chrf++')}"
    )

    print(
        f"  Examples        : "
        f"{len(selected_examples)}"
    )

    print(
        f"  Failures        : "
        f"{len(inference_failures)}"
    )

    print()
    print(
        "Evaluation completed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())