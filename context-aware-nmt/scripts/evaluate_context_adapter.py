from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from transformers.modeling_outputs import BaseModelOutput
from sentence_transformers import SentenceTransformer
from sacrebleu.metrics import BLEU, CHRF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.context_adapter_model import ContextAdapter
from backend.app.services.semantic_context_retriever import (
    SemanticContextRetriever,
)

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "tel_Telu"

CONTEXT_DIM = 384
HIDDEN_DIM = 1024
TOP_K = 2

MAX_SOURCE_LENGTH = 128
MAX_NEW_TOKENS = 128

MANIFEST_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "evaluation"
    / "dochplt"
    / "final_eval_manifest.json"
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


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def clean_text(text):
    return " ".join(str(text).split()).strip()


def main():
    print("=" * 70)
    print("CONTEXT ADAPTER EVALUATION")
    print("=" * 70)

    device = torch.device("cpu")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {device}")
    print(f"NLLB model: {NLLB_MODEL}")
    print(f"Context encoder: {EMBEDDING_MODEL}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    print("Loading evaluation manifest...")

    manifest = load_json(MANIFEST_PATH)
    examples = manifest["examples"]

    print(f"Evaluation examples loaded: {len(examples)}")
    print(
        f"Official test set: "
        f"{manifest.get('official_test_set')}"
    )
    print(
        f"Evaluation type: "
        f"{manifest.get('evaluation_type')}"
    )
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
        f"Checkpoint context dimension: "
        f"{checkpoint_context_dim}"
    )

    print(
        f"Checkpoint hidden dimension: "
        f"{checkpoint_hidden_dim}"
    )

    print(
        f"Checkpoint embedding model: "
        f"{checkpoint_embedding_model}"
    )

    print(
        f"Checkpoint NLLB model: "
        f"{checkpoint_nllb_model}"
    )

    if checkpoint_context_dim != CONTEXT_DIM:
        raise ValueError(
            "Checkpoint context dimension mismatch."
        )

    if checkpoint_hidden_dim != HIDDEN_DIM:
        raise ValueError(
            "Checkpoint hidden dimension mismatch."
        )

    print()
    print("Loading NLLB tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_nllb_model
    )

    tokenizer.src_lang = SOURCE_LANGUAGE

    print("Loading NLLB model...")

    nllb = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint_nllb_model
    )

    nllb.to(device)
    nllb.eval()

    for parameter in nllb.parameters():
        parameter.requires_grad = False

    print(
        f"NLLB model class: "
        f"{nllb.__class__.__name__}"
    )

    print(
        f"Decoder start token ID: "
        f"{nllb.config.decoder_start_token_id}"
    )

    print(
        f"EOS token ID: "
        f"{nllb.config.eos_token_id}"
    )

    print(
        f"PAD token ID: "
        f"{nllb.config.pad_token_id}"
    )

    print()
    print("Loading context encoder...")

    context_encoder = SentenceTransformer(
        checkpoint_embedding_model,
        device="cpu",
    )

    context_encoder.eval()

    print("Creating Context Adapter...")

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

    adapter_parameters = sum(
        parameter.numel()
        for parameter in adapter.parameters()
    )

    print(
        f"Adapter parameters: {adapter_parameters}"
    )

    print()
    print("Creating semantic context retriever...")

    retriever = SemanticContextRetriever(
        model_name=checkpoint_embedding_model,
        top_k=TOP_K,
        min_similarity=0.0,
    )

    bleu_metric = BLEU()
    chrf_metric = CHRF()

    results = []
    failures = []

    print()
    print("=" * 70)
    print("STARTING EVALUATION")
    print("=" * 70)
    print()

    for index, example in enumerate(
        examples,
        start=1,
    ):
        try:
            current_sentence = clean_text(
                example["source_sentence"]
            )

            if "target_reference" in example:
                reference = clean_text(
                    example["target_reference"]
                )
            else:
                reference = clean_text(
                    example["target_sentence"]
                )

            available_context = []

            for item in example.get(
                "context",
                [],
            ):
                context_id = str(
                    item.get("id", "")
                ).strip()

                context_text = clean_text(
                    item.get("text", "")
                )

                if context_text:
                    available_context.append(
                        {
                            "sentence_id": context_id,
                            "text": context_text,
                        }
                    )

            selected_context = retriever.retrieve(
                current_sentence=current_sentence,
                preceding_sentences=available_context,
            )

            selected_context_texts = [
                candidate.text
                for candidate in selected_context
            ]

            selected_context_ids = [
                candidate.sentence_id
                for candidate in selected_context
            ]

            selected_context_scores = [
                float(candidate.similarity)
                for candidate in selected_context
            ]

            tokenizer.src_lang = SOURCE_LANGUAGE

            source_inputs = tokenizer(
                [current_sentence],
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

            with torch.no_grad():
                encoder_outputs = nllb.get_encoder()(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

                encoder_hidden_states = (
                    encoder_outputs.last_hidden_state
                )

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
                    .to(device)
                    .float()
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

            with torch.no_grad():
                fused_hidden_states = adapter(
                    encoder_hidden_states,
                    context_embedding,
                )

            adapted_encoder_outputs = BaseModelOutput(
                last_hidden_state=fused_hidden_states
            )

            target_language_token = (
                tokenizer.convert_tokens_to_ids(
                    TARGET_LANGUAGE
                )
            )

            with torch.no_grad():
                generated_ids = nllb.generate(
                    encoder_outputs=(
                        adapted_encoder_outputs
                    ),
                    attention_mask=attention_mask,
                    forced_bos_token_id=(
                        target_language_token
                    ),
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=1,
                    do_sample=False,
                )

            prediction = clean_text(
                tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0]
            )

            example_bleu = (
                bleu_metric.sentence_score(
                    prediction,
                    [reference],
                ).score
            )

            example_chrf = (
                chrf_metric.sentence_score(
                    prediction,
                    [reference],
                ).score
            )

            results.append(
                {
                    "evaluation_index": index,
                    "manifest_evaluation_index": example.get(
                        "evaluation_index",
                        index,
                    ),
                    "manual_review_index": example.get(
                        "manual_review_index"
                    ),
                    "source_document_id": example.get(
                        "source_document_id"
                    ),
                    "target_document_id": example.get(
                        "target_document_id"
                    ),
                    "source_sentence_id": example.get(
                        "source_sentence_id"
                    ),
                    "target_sentence_id": example.get(
                        "target_sentence_id"
                    ),
                    "source_sentence": current_sentence,
                    "target_reference": reference,
                    "available_context": available_context,
                    "selected_context": [
                        {
                            "id": candidate.sentence_id,
                            "text": candidate.text,
                            "similarity": float(
                                candidate.similarity
                            ),
                        }
                        for candidate in selected_context
                    ],
                    "selected_context_ids": (
                        selected_context_ids
                    ),
                    "selected_context_scores": (
                        selected_context_scores
                    ),
                    "prediction": prediction,
                    "bleu": float(example_bleu),
                    "chrf": float(example_chrf),
                    "context_used": bool(
                        selected_context_texts
                    ),
                    "target_side_used_for_ranking": False,
                    "failed": False,
                }
            )

            print(
                f"Example {index}/{len(examples)}"
            )
            print(
                f"  Current: {current_sentence}"
            )
            print(
                f"  Selected context: "
                f"{len(selected_context_texts)}"
            )
            print(
                f"  Prediction: {prediction}"
            )
            print(
                f"  Reference: {reference}"
            )
            print(
                f"  BLEU: {example_bleu:.4f}"
            )
            print(
                f"  chrF++: {example_chrf:.4f}"
            )
            print()

        except Exception as exc:
            failures.append(
                {
                    "evaluation_index": index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

            print(
                f"WARNING: Example {index} failed: {exc}"
            )
            print()

    if not results:
        raise RuntimeError(
            "Evaluation failed: zero successful examples."
        )

    predictions = [
        item["prediction"]
        for item in results
    ]

    references = [
        item["target_reference"]
        for item in results
    ]

    corpus_bleu = bleu_metric.corpus_score(
        predictions,
        [references],
    ).score

    corpus_chrf = chrf_metric.corpus_score(
        predictions,
        [references],
    ).score

    baseline_bleu = 41.27782044840121
    baseline_chrf = 53.482361839304126

    bleu_delta = (
        corpus_bleu - baseline_bleu
    )

    chrf_delta = (
        corpus_chrf - baseline_chrf
    )

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
        "dataset": manifest.get(
            "dataset",
            "DocHPLT",
        ),
        "config": manifest.get(
            "config",
            "en-te",
        ),
        "split": manifest.get(
            "split",
            "train",
        ),
        "evaluation_type": manifest.get(
            "evaluation_type"
        ),
        "official_test_set": False,
        "document_level_nmt": False,
        "evaluation_subset_size": len(results),
        "failures": len(failures),
        "model": checkpoint_nllb_model,
        "context_encoder": checkpoint_embedding_model,
        "context_adapter_checkpoint": str(
            CHECKPOINT_PATH
        ),
        "context_dim": checkpoint_context_dim,
        "hidden_dim": checkpoint_hidden_dim,
        "top_k": TOP_K,
        "target_side_used_for_ranking": False,
        "baseline_bleu": baseline_bleu,
        "baseline_chrf": baseline_chrf,
        "context_adapter_bleu": corpus_bleu,
        "context_adapter_chrf": corpus_chrf,
        "bleu_absolute_change": bleu_delta,
        "chrf_absolute_change": chrf_delta,
        "adapter_improved_bleu": (
            corpus_bleu > baseline_bleu
        ),
        "adapter_improved_chrf": (
            corpus_chrf > baseline_chrf
        ),
        "successful_examples": len(results),
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

    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Successful examples: {len(results)}"
    )

    print(
        f"Failed examples: {len(failures)}"
    )

    print()
    print("Context Adapter:")
    print(
        f"  BLEU:   {corpus_bleu:.8f}"
    )
    print(
        f"  chrF++: {corpus_chrf:.8f}"
    )

    print()
    print("Sentence-level NLLB baseline:")
    print(
        f"  BLEU:   {baseline_bleu:.8f}"
    )
    print(
        f"  chrF++: {baseline_chrf:.8f}"
    )

    print()
    print("Absolute change:")
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
        "DocHPLT test-set evaluation."
    )

    print(
        "It is a manually reviewed deterministic "
        "subset from a DocHPLT training shard."
    )


if __name__ == "__main__":
    main()