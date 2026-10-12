from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import torch
from sentence_transformers import SentenceTransformer
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput


# ---------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.app.services.context_adapter_model import ContextAdapter
from backend.app.services.semantic_context_retriever import (
    SemanticContextRetriever,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_NAME = "facebook/nllb-200-distilled-600M"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "context_adapter"
    / "context_adapter.pt"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "context_adapter"
    / "val.jsonl"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "evaluation"
    / "dochplt"
)

DEVICE = torch.device("cpu")

SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "tel_Telu"

MAX_SOURCE_LENGTH = 128
MAX_NEW_TOKENS = 128

# Final evaluation size
EVALUATION_LIMIT = 1000

# Maximum semantic context sentences
CONTEXT_TOP_K = 2

SEED = 42


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def load_validation_examples(
    limit: int,
) -> list[dict]:

    examples = []

    with VALIDATION_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            examples.append(
                json.loads(line)
            )

            if len(examples) >= limit:
                break

    return examples


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_models():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Loading NLLB model...")

    nllb = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    nllb.to(DEVICE)
    nllb.eval()

    print("Loading MiniLM context encoder...")

    context_encoder = SentenceTransformer(
        EMBEDDING_MODEL,
        device=str(DEVICE),
    )

    return (
        tokenizer,
        nllb,
        context_encoder,
    )


def load_adapter() -> ContextAdapter:

    print("Loading Context Adapter checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    context_dim = int(
        checkpoint["context_dim"]
    )

    hidden_dim = int(
        checkpoint["hidden_dim"]
    )

    adapter = ContextAdapter(
        context_dim=context_dim,
        hidden_dim=hidden_dim,
    )

    adapter.load_state_dict(
        checkpoint["model_state_dict"]
    )

    adapter.to(DEVICE)
    adapter.eval()

    return adapter


# ---------------------------------------------------------------------
# Source encoding
# ---------------------------------------------------------------------

def encode_source(
    tokenizer,
    source_sentence: str,
):

    tokenizer.src_lang = SOURCE_LANGUAGE

    encoded = tokenizer(
        [source_sentence],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SOURCE_LENGTH,
    )

    return {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }


# ---------------------------------------------------------------------
# Semantic context retrieval
# ---------------------------------------------------------------------

def retrieve_context(
    retriever: SemanticContextRetriever,
    current_sentence: str,
    context_items: list[dict],
) -> list[dict]:

    if not context_items:
        return []

    candidates = []

    for item in context_items:

        text = item.get("text")

        if not text:
            continue

        candidates.append(
            {
                "sentence_id": item["id"],
                "text": text,
            }
        )

    if not candidates:
        return []

    selected = retriever.retrieve(
        current_sentence=current_sentence,
        preceding_sentences=candidates,
    )

    selected = selected[
        :CONTEXT_TOP_K
    ]

    return [
        {
            "id": item.sentence_id,
            "text": item.text,
            "similarity": float(
                item.similarity
            ),
        }
        for item in selected
    ]


# ---------------------------------------------------------------------
# Context embedding
# ---------------------------------------------------------------------

def get_context_embedding(
    context_encoder: SentenceTransformer,
    selected_context: list[dict],
) -> torch.Tensor:

    # No context case
    if not selected_context:

        return torch.zeros(
            (1, 384),
            dtype=torch.float32,
            device=DEVICE,
        )

    context_text = " ".join(
        item["text"]
        for item in selected_context
        if item.get("text")
    )

    embedding = context_encoder.encode(
        [context_text],
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    # SentenceTransformer may return
    # inference tensors. Clone/detach so they
    # can safely be used by PyTorch modules.
    embedding = (
        embedding
        .detach()
        .clone()
        .to(
            device=DEVICE,
            dtype=torch.float32,
        )
    )

    return embedding


# ---------------------------------------------------------------------
# NLLB baseline translation
# ---------------------------------------------------------------------

@torch.no_grad()
def baseline_translate(
    tokenizer,
    nllb,
    source_sentence: str,
) -> str:

    inputs = encode_source(
        tokenizer,
        source_sentence,
    )

    forced_bos_token_id = (
        tokenizer.convert_tokens_to_ids(
            TARGET_LANGUAGE
        )
    )

    # Prevent the generation warning caused
    # by max_length in the model generation config.
    nllb.generation_config.max_length = None

    generated = nllb.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=1,
        do_sample=False,
    )

    output = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0]

    return output.strip()


# ---------------------------------------------------------------------
# Context Adapter translation
# ---------------------------------------------------------------------

@torch.no_grad()
def adapter_translate(
    tokenizer,
    nllb,
    adapter,
    source_sentence: str,
    context_embedding: torch.Tensor,
) -> str:

    inputs = encode_source(
        tokenizer,
        source_sentence,
    )

    # Original NLLB encoder
    encoder_outputs = nllb.model.encoder(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        return_dict=True,
    )

    # Context fusion
    fused_hidden_states = adapter(
        encoder_outputs.last_hidden_state,
        context_embedding,
    )

    fused_outputs = BaseModelOutput(
        last_hidden_state=fused_hidden_states
    )

    forced_bos_token_id = (
        tokenizer.convert_tokens_to_ids(
            TARGET_LANGUAGE
        )
    )

    # Prevent generation warning.
    nllb.generation_config.max_length = None

    generated = nllb.generate(
        encoder_outputs=fused_outputs,
        attention_mask=inputs["attention_mask"],
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=1,
        do_sample=False,
    )

    output = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0]

    return output.strip()


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(
    predictions: list[str],
    references: list[str],
) -> dict:

    bleu = BLEU()
    chrf = CHRF()

    bleu_score = bleu.corpus_score(
        predictions,
        [references],
    ).score

    chrf_score = chrf.corpus_score(
        predictions,
        [references],
    ).score

    return {
        "bleu": float(
            bleu_score
        ),
        "chrf": float(
            chrf_score
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    set_seed(SEED)

    print()
    print("=" * 75)
    print(
        "FINAL CONTEXT ADAPTER EVALUATION"
    )
    print("=" * 75)

    print(
        f"Model           : {MODEL_NAME}"
    )

    print(
        f"Context encoder : {EMBEDDING_MODEL}"
    )

    print(
        f"Checkpoint      : {CHECKPOINT_PATH}"
    )

    print(
        f"Validation data : {VALIDATION_PATH}"
    )

    print(
        f"Evaluation size : {EVALUATION_LIMIT}"
    )

    print(
        f"Context top-k   : {CONTEXT_TOP_K}"
    )

    print(
        f"Device          : {DEVICE}"
    )

    print("=" * 75)

    # -----------------------------------------------------------------
    # Validate paths
    # -----------------------------------------------------------------

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "Context Adapter checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    if not VALIDATION_PATH.exists():

        raise FileNotFoundError(
            "Validation dataset not found:\n"
            f"{VALIDATION_PATH}"
        )

    # -----------------------------------------------------------------
    # Load validation data
    # -----------------------------------------------------------------

    examples = load_validation_examples(
        EVALUATION_LIMIT
    )

    if not examples:

        raise RuntimeError(
            "No validation examples were loaded."
        )

    print()
    print(
        f"Loaded validation examples: "
        f"{len(examples)}"
    )

    # -----------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------

    (
        tokenizer,
        nllb,
        context_encoder,
    ) = load_models()

    adapter = load_adapter()

    print()
    print(
        "Trainable Context Adapter parameters: "
        f"{sum(p.numel() for p in adapter.parameters())}"
    )

    # -----------------------------------------------------------------
    # Semantic retriever
    # -----------------------------------------------------------------

    retriever = SemanticContextRetriever(
        model_name=EMBEDDING_MODEL,
        top_k=CONTEXT_TOP_K,
        min_similarity=0.0,
    )

    print(
        "Semantic retriever loaded."
    )

    # -----------------------------------------------------------------
    # BASELINE
    # -----------------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "RUNNING NLLB BASELINE"
    )
    print("=" * 75)

    baseline_predictions = []

    references = []

    baseline_failures = []

    for index, example in enumerate(
        examples,
        start=1,
    ):

        source = example[
            "source_sentence"
        ]

        reference = example[
            "target_sentence"
        ]

        references.append(
            reference
        )

        try:

            prediction = baseline_translate(
                tokenizer,
                nllb,
                source,
            )

            baseline_predictions.append(
                prediction
            )

        except Exception as exc:

            print(
                f"\nBaseline failure "
                f"at example {index}: "
                f"{exc}"
            )

            baseline_predictions.append(
                ""
            )

            baseline_failures.append(
                {
                    "evaluation_index": index,
                    "error": str(exc),
                }
            )

        if index % 100 == 0:

            print(
                f"Baseline progress: "
                f"{index}/{len(examples)}"
            )

    baseline_metrics = calculate_metrics(
        baseline_predictions,
        references,
    )

    print()
    print(
        "NLLB baseline metrics:"
    )

    print(
        f"  BLEU   : "
        f"{baseline_metrics['bleu']:.8f}"
    )

    print(
        f"  chrF++ : "
        f"{baseline_metrics['chrf']:.8f}"
    )

    # -----------------------------------------------------------------
    # CONTEXT ADAPTER
    # -----------------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "RUNNING CONTEXT ADAPTER"
    )
    print("=" * 75)

    adapter_predictions = []

    adapter_failures = []

    detailed_examples = []

    for index, example in enumerate(
        examples,
        start=1,
    ):

        source = example[
            "source_sentence"
        ]

        reference = example[
            "target_sentence"
        ]

        available_context = example.get(
            "context",
            [],
        )

        try:

            # ---------------------------------------------------------
            # Semantic context retrieval
            # ---------------------------------------------------------

            selected_context = (
                retrieve_context(
                    retriever,
                    source,
                    available_context,
                )
            )

            # ---------------------------------------------------------
            # Context embedding
            # ---------------------------------------------------------

            context_embedding = (
                get_context_embedding(
                    context_encoder,
                    selected_context,
                )
            )

            # ---------------------------------------------------------
            # Context Adapter translation
            # ---------------------------------------------------------

            prediction = adapter_translate(
                tokenizer,
                nllb,
                adapter,
                source,
                context_embedding,
            )

            adapter_predictions.append(
                prediction
            )

            detailed_examples.append(
                {
                    "evaluation_index": index,
                    "source_sentence": source,
                    "reference": reference,
                    "baseline_prediction":
                        baseline_predictions[
                            index - 1
                        ],
                    "adapter_prediction":
                        prediction,
                    "selected_context":
                        selected_context,
                }
            )

        except Exception as exc:

            print(
                f"\nAdapter failure "
                f"at example {index}: "
                f"{exc}"
            )

            adapter_predictions.append(
                ""
            )

            adapter_failures.append(
                {
                    "evaluation_index": index,
                    "error": str(exc),
                }
            )

            detailed_examples.append(
                {
                    "evaluation_index": index,
                    "source_sentence": source,
                    "reference": reference,
                    "baseline_prediction":
                        baseline_predictions[
                            index - 1
                        ],
                    "adapter_prediction": "",
                    "selected_context": [],
                    "error": str(exc),
                }
            )

        if index % 100 == 0:

            print(
                f"Adapter progress: "
                f"{index}/{len(examples)}"
            )

    adapter_metrics = calculate_metrics(
        adapter_predictions,
        references,
    )

    # -----------------------------------------------------------------
    # Metric changes
    # -----------------------------------------------------------------

    bleu_change = (
        adapter_metrics["bleu"]
        - baseline_metrics["bleu"]
    )

    chrf_change = (
        adapter_metrics["chrf"]
        - baseline_metrics["chrf"]
    )

    # Percentage changes
    if baseline_metrics["bleu"] != 0:

        bleu_relative_change = (
            bleu_change
            / baseline_metrics["bleu"]
        ) * 100

    else:

        bleu_relative_change = None

    if baseline_metrics["chrf"] != 0:

        chrf_relative_change = (
            chrf_change
            / baseline_metrics["chrf"]
        ) * 100

    else:

        chrf_relative_change = None

    # -----------------------------------------------------------------
    # Create output directory
    # -----------------------------------------------------------------

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    run_dir = (
        OUTPUT_ROOT
        / f"{timestamp}-final-1000"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    summary = {

        "dataset": "DocHPLT",

        "config": "en-te",

        "split": "validation",

        "evaluation_type":
            "deterministic 1000-example "
            "subset of prepared validation data",

        "official_test_set": False,

        "document_level_nmt": False,

        "model":
            MODEL_NAME,

        "context_encoder":
            EMBEDDING_MODEL,

        "checkpoint":
            str(CHECKPOINT_PATH),

        "validation_file":
            str(VALIDATION_PATH),

        "examples":
            len(examples),

        "context_top_k":
            CONTEXT_TOP_K,

        "source_side_context_only":
            True,

        "target_side_used_for_ranking":
            False,

        "retrieval_method":
            "semantic similarity using "
            "all-MiniLM-L6-v2",

        "decoding": {

            "num_beams":
                1,

            "do_sample":
                False,

            "max_source_length":
                MAX_SOURCE_LENGTH,

            "max_new_tokens":
                MAX_NEW_TOKENS,
        },

        "baseline": {

            "model":
                MODEL_NAME,

            "metrics":
                baseline_metrics,

            "failures":
                len(
                    baseline_failures
                ),
        },

        "context_adapter": {

            "metrics":
                adapter_metrics,

            "failures":
                len(
                    adapter_failures
                ),

            "context_top_k":
                CONTEXT_TOP_K,
        },

        "absolute_change": {

            "bleu":
                bleu_change,

            "chrf":
                chrf_change,
        },

        "relative_change_percent": {

            "bleu":
                bleu_relative_change,

            "chrf":
                chrf_relative_change,
        },

        "notes": [

            "The baseline and Context Adapter "
            "are evaluated on exactly the same "
            "validation examples.",

            "This is not an official DocHPLT "
            "test set.",

            "The evaluation uses a deterministic "
            "subset of the prepared validation data.",

            "The Context Adapter is an experimental "
            "context-fusion mechanism built on "
            "sentence-level NLLB-200.",

            "Source-side context is retrieved "
            "without using target/reference text "
            "for ranking.",

            "The context encoder and NLLB backbone "
            "remain frozen during evaluation.",

            "Only the trained Context Adapter "
            "checkpoint is loaded as the learned "
            "context-fusion component.",
        ],
    }

    # -----------------------------------------------------------------
    # Save summary
    # -----------------------------------------------------------------

    summary_path = (
        run_dir
        / "summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------------------
    # Save per-example results
    # -----------------------------------------------------------------

    examples_path = (
        run_dir
        / "examples.json"
    )

    with examples_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            detailed_examples,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------------------
    # Save failures
    # -----------------------------------------------------------------

    failures_path = (
        run_dir
        / "failures.json"
    )

    with failures_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "baseline":
                    baseline_failures,

                "adapter":
                    adapter_failures,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------------------
    # Final terminal report
    # -----------------------------------------------------------------

    print()
    print()
    print("=" * 75)
    print(
        "FINAL 1000-EXAMPLE RESULT"
    )
    print("=" * 75)

    print(
        f"Examples: {len(examples)}"
    )

    print()
    print(
        "NLLB baseline"
    )

    print(
        f"  BLEU   : "
        f"{baseline_metrics['bleu']:.8f}"
    )

    print(
        f"  chrF++ : "
        f"{baseline_metrics['chrf']:.8f}"
    )

    print()
    print(
        "Context Adapter"
    )

    print(
        f"  BLEU   : "
        f"{adapter_metrics['bleu']:.8f}"
    )

    print(
        f"  chrF++ : "
        f"{adapter_metrics['chrf']:.8f}"
    )

    print()
    print(
        "Absolute improvement"
    )

    print(
        f"  Δ BLEU   : "
        f"{bleu_change:+.8f}"
    )

    print(
        f"  Δ chrF++ : "
        f"{chrf_change:+.8f}"
    )

    print()
    print(
        "Relative improvement"
    )

    if bleu_relative_change is not None:

        print(
            f"  BLEU   : "
            f"{bleu_relative_change:+.2f}%"
        )

    else:

        print(
            "  BLEU   : N/A"
        )

    if chrf_relative_change is not None:

        print(
            f"  chrF++ : "
            f"{chrf_relative_change:+.2f}%"
        )

    else:

        print(
            "  chrF++ : N/A"
        )

    print()
    print(
        "Failures"
    )

    print(
        f"  Baseline : "
        f"{len(baseline_failures)}"
    )

    print(
        f"  Adapter  : "
        f"{len(adapter_failures)}"
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        run_dir
    )

    print()
    print("=" * 75)


if __name__ == "__main__":
    main()