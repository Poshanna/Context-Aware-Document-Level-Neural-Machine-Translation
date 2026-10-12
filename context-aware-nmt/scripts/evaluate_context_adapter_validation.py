from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
from sacrebleu.metrics import BLEU, CHRF
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.context_adapter_model import ContextAdapter
from backend.app.services.semantic_context_retriever import (
    SemanticContextRetriever,
)


# ============================================================
# CONFIGURATION
# ============================================================

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "tel_Telu"

CONTEXT_DIM = 384
HIDDEN_DIM = 1024

TOP_K = 2

MAX_SOURCE_LENGTH = 128
MAX_NEW_TOKENS = 128

EVALUATION_LIMIT = 100
SEED = 42

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "context_adapter"
    / "val.jsonl"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "context_adapter"
    / "context_adapter.pt"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "evaluation"
    / "dochplt"
)


# ============================================================
# HELPERS
# ============================================================

def clean_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )


def load_jsonl(path: Path, limit: int):
    examples = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            examples.append(json.loads(line))

            if len(examples) >= limit:
                break

    return examples


def translate_baseline(
    model,
    tokenizer,
    source_text,
    device,
):
    tokenizer.src_lang = SOURCE_LANGUAGE

    inputs = tokenizer(
        [source_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SOURCE_LENGTH,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    target_language_token = tokenizer.convert_tokens_to_ids(
        TARGET_LANGUAGE
    )

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            forced_bos_token_id=target_language_token,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=1,
            do_sample=False,
        )

    return clean_text(
        tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("100-EXAMPLE CONTEXT ADAPTER VALIDATION")
    print("=" * 70)

    device = torch.device("cpu")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {device}")
    print(f"NLLB model: {NLLB_MODEL}")
    print(f"Context encoder: {EMBEDDING_MODEL}")
    print(f"Validation file: {VALIDATION_PATH}")
    print(f"Adapter checkpoint: {CHECKPOINT_PATH}")
    print(f"Evaluation limit: {EVALUATION_LIMIT}")
    print()

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {VALIDATION_PATH}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # ========================================================
    # LOAD VALIDATION DATA
    # ========================================================

    print("Loading validation dataset...")

    examples = load_jsonl(
        VALIDATION_PATH,
        EVALUATION_LIMIT,
    )

    print(
        f"Validation examples loaded: {len(examples)}"
    )

    if len(examples) < EVALUATION_LIMIT:
        print(
            "WARNING: Validation file contains fewer "
            f"than {EVALUATION_LIMIT} examples."
        )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    print()
    print("Loading Context Adapter checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    checkpoint_context_dim = checkpoint.get(
        "context_dim",
        CONTEXT_DIM,
    )

    checkpoint_hidden_dim = checkpoint.get(
        "hidden_dim",
        HIDDEN_DIM,
    )

    checkpoint_embedding_model = checkpoint.get(
        "embedding_model",
        EMBEDDING_MODEL,
    )

    checkpoint_nllb_model = checkpoint.get(
        "nllb_model",
        NLLB_MODEL,
    )

    print(
        f"Context dimension: {checkpoint_context_dim}"
    )

    print(
        f"Hidden dimension: {checkpoint_hidden_dim}"
    )

    print(
        f"Embedding model: {checkpoint_embedding_model}"
    )

    print(
        f"NLLB model: {checkpoint_nllb_model}"
    )

    # ========================================================
    # LOAD TOKENIZER
    # ========================================================

    print()
    print("Loading NLLB tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_nllb_model
    )

    tokenizer.src_lang = SOURCE_LANGUAGE

    # ========================================================
    # LOAD NLLB
    # ========================================================

    print("Loading NLLB model...")

    nllb = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint_nllb_model
    )

    nllb.to(device)
    nllb.eval()

    for parameter in nllb.parameters():
        parameter.requires_grad = False

    print(
        f"NLLB class: {nllb.__class__.__name__}"
    )

    # ========================================================
    # LOAD MINILM
    # ========================================================

    print()
    print("Loading context encoder...")

    context_encoder = SentenceTransformer(
        checkpoint_embedding_model,
        device="cpu",
    )

    context_encoder.eval()

    # ========================================================
    # LOAD ADAPTER
    # ========================================================

    print("Loading Context Adapter...")

    adapter = ContextAdapter(
        context_dim=checkpoint_context_dim,
        hidden_dim=checkpoint_hidden_dim,
    )

    adapter.load_state_dict(
        checkpoint["model_state_dict"]
    )

    adapter.to(device)
    adapter.eval()

    for parameter in adapter.parameters():
        parameter.requires_grad = False

    print(
        "Trainable adapter parameters: 0"
    )

    # ========================================================
    # RETRIEVER
    # ========================================================

    print()
    print("Creating semantic context retriever...")

    retriever = SemanticContextRetriever(
        model_name=checkpoint_embedding_model,
        top_k=TOP_K,
        min_similarity=0.0,
    )

    # ========================================================
    # METRICS
    # ========================================================

    bleu = BLEU()
    chrf = CHRF()

    baseline_predictions = []
    adapter_predictions = []
    references = []

    results = []
    failures = []

    # ========================================================
    # EVALUATION
    # ========================================================

    print()
    print("=" * 70)
    print("STARTING 100-EXAMPLE EVALUATION")
    print("=" * 70)
    print()

    for index, example in enumerate(
        examples,
        start=1,
    ):

        try:

            source = clean_text(
                example["source_sentence"]
            )

            reference = clean_text(
                example["target_sentence"]
            )

            # ------------------------------------------------
            # BASELINE
            # ------------------------------------------------

            baseline_prediction = translate_baseline(
                nllb,
                tokenizer,
                source,
                device,
            )

            # ------------------------------------------------
            # AVAILABLE CONTEXT
            # ------------------------------------------------

            available_context = []

            for item in example.get(
                "context",
                [],
            ):

                text = clean_text(
                    item.get("text", "")
                )

                if text:
                    available_context.append(
                        {
                            "sentence_id": str(
                                item.get("id", "")
                            ),
                            "text": text,
                        }
                    )

            # ------------------------------------------------
            # SEMANTIC RETRIEVAL
            # ------------------------------------------------

            selected_context = retriever.retrieve(
                current_sentence=source,
                preceding_sentences=available_context,
            )

            selected_context_texts = [
                item.text
                for item in selected_context
            ]

            # ------------------------------------------------
            # ENCODE CURRENT SENTENCE
            # ------------------------------------------------

            tokenizer.src_lang = SOURCE_LANGUAGE

            source_inputs = tokenizer(
                [source],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SOURCE_LENGTH,
            )

            input_ids = source_inputs[
                "input_ids"
            ].to(device)

            attention_mask = source_inputs[
                "attention_mask"
            ].to(device)

            # ------------------------------------------------
            # NLLB ENCODER
            # ------------------------------------------------

            with torch.no_grad():

                encoder_outputs = nllb.get_encoder()(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

                encoder_hidden = (
                    encoder_outputs.last_hidden_state
                )

            # ------------------------------------------------
            # CONTEXT EMBEDDING
            # ------------------------------------------------

            if selected_context_texts:

                context_text = " ".join(
                    selected_context_texts
                )

                context_embedding = (
                    context_encoder.encode(
                        [context_text],
                        convert_to_tensor=True,
                        show_progress_bar=False,
                    )
                )

                context_embedding = (
                    context_embedding
                    .detach()
                    .clone()
                    .float()
                    .to(device)
                )

            else:

                context_embedding = torch.zeros(
                    (
                        1,
                        checkpoint_context_dim,
                    ),
                    dtype=torch.float32,
                    device=device,
                )

            # ------------------------------------------------
            # ADAPTER FUSION
            # ------------------------------------------------

            with torch.no_grad():

                fused_hidden = adapter(
                    encoder_hidden,
                    context_embedding,
                )

            adapted_encoder = BaseModelOutput(
                last_hidden_state=fused_hidden
            )

            # ------------------------------------------------
            # ADAPTER GENERATION
            # ------------------------------------------------

            target_language_token = (
                tokenizer.convert_tokens_to_ids(
                    TARGET_LANGUAGE
                )
            )

            with torch.no_grad():

                generated_ids = nllb.generate(
                    encoder_outputs=adapted_encoder,
                    attention_mask=attention_mask,
                    forced_bos_token_id=(
                        target_language_token
                    ),
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=1,
                    do_sample=False,
                )

            adapter_prediction = clean_text(
                tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0]
            )

            baseline_predictions.append(
                baseline_prediction
            )

            adapter_predictions.append(
                adapter_prediction
            )

            references.append(reference)

            results.append(
                {
                    "index": index,
                    "source_sentence": source,
                    "reference": reference,
                    "baseline_prediction": (
                        baseline_prediction
                    ),
                    "adapter_prediction": (
                        adapter_prediction
                    ),
                    "selected_context": [
                        {
                            "id": item.sentence_id,
                            "text": item.text,
                            "similarity": float(
                                item.similarity
                            ),
                        }
                        for item in selected_context
                    ],
                }
            )

            if index <= 10 or index % 10 == 0:
                print(
                    f"Example {index}/{len(examples)}"
                )
                print(
                    f"  Source: {source[:100]}"
                )
                print(
                    f"  Baseline: "
                    f"{baseline_prediction[:100]}"
                )
                print(
                    f"  Adapter: "
                    f"{adapter_prediction[:100]}"
                )
                print()

        except Exception as exc:

            failures.append(
                {
                    "index": index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

            print(
                f"WARNING: Example {index} failed: {exc}"
            )

    # ========================================================
    # AGGREGATE METRICS
    # ========================================================

    if not results:
        raise RuntimeError(
            "No successful evaluation examples."
        )

    baseline_bleu = bleu.corpus_score(
        baseline_predictions,
        [references],
    ).score

    baseline_chrf = chrf.corpus_score(
        baseline_predictions,
        [references],
    ).score

    adapter_bleu = bleu.corpus_score(
        adapter_predictions,
        [references],
    ).score

    adapter_chrf = chrf.corpus_score(
        adapter_predictions,
        [references],
    ).score

    bleu_delta = (
        adapter_bleu - baseline_bleu
    )

    chrf_delta = (
        adapter_chrf - baseline_chrf
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    run_id = (
        utc_timestamp()
        + "-"
        + uuid.uuid4().hex[:8]
    )

    output_dir = OUTPUT_ROOT / run_id

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "dataset": "DocHPLT",
        "config": "en-te",
        "source_split": "validation",
        "evaluation_type": (
            "deterministic subset of prepared "
            "DocHPLT context-aware validation data"
        ),
        "official_test_set": False,
        "document_level_nmt": False,
        "evaluation_limit": EVALUATION_LIMIT,
        "successful_examples": len(results),
        "failed_examples": len(failures),
        "seed": SEED,
        "nllb_model": checkpoint_nllb_model,
        "context_encoder": checkpoint_embedding_model,
        "context_adapter_checkpoint": str(
            CHECKPOINT_PATH
        ),
        "context_dim": checkpoint_context_dim,
        "hidden_dim": checkpoint_hidden_dim,
        "top_k": TOP_K,
        "baseline_bleu": baseline_bleu,
        "baseline_chrf": baseline_chrf,
        "context_adapter_bleu": adapter_bleu,
        "context_adapter_chrf": adapter_chrf,
        "bleu_absolute_change": bleu_delta,
        "chrf_absolute_change": chrf_delta,
        "adapter_improved_bleu": (
            adapter_bleu > baseline_bleu
        ),
        "adapter_improved_chrf": (
            adapter_chrf > baseline_chrf
        ),
        "run_id": run_id,
    }

    with (
        output_dir / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with (
        output_dir / "examples.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with (
        output_dir / "failures.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            failures,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("100-EXAMPLE EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Successful examples: {len(results)}"
    )

    print(
        f"Failed examples: {len(failures)}"
    )

    print()
    print("NLLB BASELINE")
    print(
        f"  BLEU:   {baseline_bleu:.8f}"
    )
    print(
        f"  chrF++: {baseline_chrf:.8f}"
    )

    print()
    print("CONTEXT ADAPTER")
    print(
        f"  BLEU:   {adapter_bleu:.8f}"
    )
    print(
        f"  chrF++: {adapter_chrf:.8f}"
    )

    print()
    print("ADAPTER CHANGE")
    print(
        f"  BLEU:   {bleu_delta:+.8f}"
    )
    print(
        f"  chrF++: {chrf_delta:+.8f}"
    )

    print()
    print(
        f"Results directory: {output_dir}"
    )

    print()
    print(
        "IMPORTANT: This is NOT an official "
        "DocHPLT test-set result."
    )

    print(
        "It is a deterministic evaluation subset "
        "from the prepared validation data."
    )


if __name__ == "__main__":
    main()