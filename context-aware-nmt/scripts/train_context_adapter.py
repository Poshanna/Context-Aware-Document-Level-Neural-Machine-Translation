from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
from transformers import AutoTokenizer, M2M100ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

# ---------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.context_adapter_model import ContextAdapter


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_NAME = "facebook/nllb-200-distilled-600M"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "context_adapter"
    / "train.jsonl"
)

VAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "context_adapter"
    / "val.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "context_adapter"

CHECKPOINT_PATH = OUTPUT_DIR / "context_adapter.pt"
REPORT_PATH = OUTPUT_DIR / "training_report.json"

SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "tel_Telu"

CONTEXT_DIM = 384
NLLB_HIDDEN_DIM = 1024

MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 128

TRAIN_LIMIT = 1000
VALIDATION_LIMIT = 100

EPOCHS = 1
BATCH_SIZE = 1
LEARNING_RATE = 1e-4

SEED = 42

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    records: list[dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

            if limit is not None and len(records) >= limit:
                break

    return records


def get_context_text(example: dict) -> str:
    context = example.get("context", [])

    texts: list[str] = []

    for item in context:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
        else:
            text = str(item).strip()

        if text:
            texts.append(text)

    return " ".join(texts)


# ---------------------------------------------------------------------
# Context embedding
# ---------------------------------------------------------------------

def encode_context(
    embedding_model: SentenceTransformer,
    context_text: str,
) -> torch.Tensor:
    if not context_text.strip():
        return torch.zeros(
            CONTEXT_DIM,
            dtype=torch.float32,
            device=DEVICE,
        )

    embedding = embedding_model.encode(
        context_text,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # SentenceTransformer may return an inference-mode tensor.
    # Clone/detach converts it into a normal tensor that can safely
    # participate in the adapter's autograd graph.
    embedding = embedding.clone().detach()

    return embedding.to(
        device=DEVICE,
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------

def compute_loss(
    example: dict,
    tokenizer,
    nllb: M2M100ForConditionalGeneration,
    embedding_model: SentenceTransformer,
    adapter: ContextAdapter,
) -> torch.Tensor:

    source_text = str(example["source_sentence"]).strip()
    target_text = str(example["target_sentence"]).strip()

    context_text = get_context_text(example)

    if not source_text:
        raise ValueError("Empty source sentence")

    if not target_text:
        raise ValueError("Empty target reference")

    # -------------------------------------------------------------
    # Source encoding
    # -------------------------------------------------------------

    tokenizer.src_lang = SOURCE_LANGUAGE

    source_inputs = tokenizer(
        source_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SOURCE_LENGTH,
    )

    input_ids = source_inputs["input_ids"].to(DEVICE)
    attention_mask = source_inputs["attention_mask"].to(DEVICE)

    # -------------------------------------------------------------
    # Target labels
    # -------------------------------------------------------------

    tokenizer.tgt_lang = TARGET_LANGUAGE

    target_inputs = tokenizer(
        text_target=target_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TARGET_LENGTH,
    )

    labels = target_inputs["input_ids"].to(DEVICE)

    labels[labels == tokenizer.pad_token_id] = -100

    # -------------------------------------------------------------
    # Context embedding
    # -------------------------------------------------------------

    context_embedding = encode_context(
        embedding_model,
        context_text,
    ).unsqueeze(0)

    # -------------------------------------------------------------
    # NLLB encoder
    # -------------------------------------------------------------

    with torch.no_grad():
        encoder_outputs = nllb.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

    encoder_hidden_states = encoder_outputs.last_hidden_state

    # -------------------------------------------------------------
    # Context adapter
    # -------------------------------------------------------------

    fused_hidden_states = adapter(
        encoder_hidden_states,
        context_embedding,
    )

    fused_encoder_outputs = BaseModelOutput(
        last_hidden_state=fused_hidden_states,
    )

    # -------------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT manually create decoder_input_ids.
    #
    # Transformers 5.18.0 M2M100 forward() automatically performs:
    #
    # shift_tokens_right(
    #     labels,
    #     pad_token_id,
    #     decoder_start_token_id
    # )
    #
    # This keeps training consistent with the installed NLLB/M2M100
    # implementation.
    # -------------------------------------------------------------

    outputs = nllb(
        encoder_outputs=fused_encoder_outputs,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True,
    )

    loss = outputs.loss

    if loss is None:
        raise RuntimeError("NLLB returned no loss")

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {loss.item()}")

    return loss


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("CONTEXT ADAPTER TRAINING")
    print("=" * 70)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {DEVICE}")
    print(f"NLLB model: {MODEL_NAME}")
    print(f"Context encoder: {EMBEDDING_MODEL}")
    print(f"Source language: {SOURCE_LANGUAGE}")
    print(f"Target language: {TARGET_LANGUAGE}")
    print(f"Train limit: {TRAIN_LIMIT}")
    print(f"Validation limit: {VALIDATION_LIMIT}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()

    # -------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------

    print("Loading training dataset...")

    train_examples = load_jsonl(
        TRAIN_PATH,
        limit=TRAIN_LIMIT,
    )

    validation_examples = load_jsonl(
        VAL_PATH,
        limit=VALIDATION_LIMIT,
    )

    print(f"Training examples loaded: {len(train_examples)}")
    print(f"Validation examples loaded: {len(validation_examples)}")

    if not train_examples:
        raise RuntimeError("No training examples found")

    # -------------------------------------------------------------
    # Load tokenizer + NLLB
    # -------------------------------------------------------------

    print()
    print("Loading NLLB tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )

    print("Loading NLLB model...")

    nllb = M2M100ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
    )

    nllb.to(DEVICE)
    nllb.eval()

    # Freeze NLLB
    for parameter in nllb.parameters():
        parameter.requires_grad = False

    print(f"NLLB model class: {nllb.__class__.__name__}")
    print(f"Decoder start token ID: {nllb.config.decoder_start_token_id}")
    print(f"EOS token ID: {nllb.config.eos_token_id}")
    print(f"PAD token ID: {nllb.config.pad_token_id}")

    # -------------------------------------------------------------
    # Load MiniLM
    # -------------------------------------------------------------

    print()
    print("Loading context encoder...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=str(DEVICE),
    )

    embedding_model.eval()

    for parameter in embedding_model.parameters():
        parameter.requires_grad = False

    # -------------------------------------------------------------
    # Create adapter
    # -------------------------------------------------------------

    print()
    print("Creating context adapter...")

    adapter = ContextAdapter(
        context_dim=CONTEXT_DIM,
        hidden_dim=NLLB_HIDDEN_DIM,
    )

    adapter.to(DEVICE)
    adapter.train()

    trainable_parameters = sum(
        parameter.numel()
        for parameter in adapter.parameters()
        if parameter.requires_grad
    )

    print(f"Trainable adapter parameters: {trainable_parameters}")

    # -------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------

    optimizer = AdamW(
        adapter.parameters(),
        lr=LEARNING_RATE,
    )

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------

    training_losses: list[float] = []
    successful_steps = 0

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(EPOCHS):

        print()
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        epoch_losses: list[float] = []

        for index, example in enumerate(
            train_examples,
            start=1,
        ):

            optimizer.zero_grad(
                set_to_none=True,
            )

            try:

                loss = compute_loss(
                    example=example,
                    tokenizer=tokenizer,
                    nllb=nllb,
                    embedding_model=embedding_model,
                    adapter=adapter,
                )

                loss.backward()

                # Safety check
                has_invalid_gradient = False

                for parameter in adapter.parameters():
                    if parameter.grad is not None:
                        if not torch.isfinite(
                            parameter.grad
                        ).all():
                            has_invalid_gradient = True
                            break

                if has_invalid_gradient:
                    raise RuntimeError(
                        "Non-finite adapter gradient"
                    )

                optimizer.step()

                loss_value = float(loss.item())

                epoch_losses.append(loss_value)
                training_losses.append(loss_value)

                successful_steps += 1

                if (
                    index == 1
                    or index % 100 == 0
                    or index == len(train_examples)
                ):
                    print(
                        f"Step {index}/{len(train_examples)} "
                        f"| Loss: {loss_value:.6f}"
                    )

            except Exception as exc:

                print(
                    f"WARNING: step {index} failed: {exc}"
                )

                optimizer.zero_grad(
                    set_to_none=True,
                )

        if epoch_losses:

            epoch_average = (
                sum(epoch_losses)
                / len(epoch_losses)
            )

            print(
                f"Epoch {epoch + 1} average loss: "
                f"{epoch_average:.6f}"
            )

    if successful_steps == 0:
        raise RuntimeError(
            "Training failed: zero successful steps"
        )

    # -------------------------------------------------------------
    # Save checkpoint
    # -------------------------------------------------------------

    checkpoint = {
        "model_state_dict": adapter.state_dict(),
        "context_dim": CONTEXT_DIM,
        "hidden_dim": NLLB_HIDDEN_DIM,
        "embedding_model": EMBEDDING_MODEL,
        "nllb_model": MODEL_NAME,
        "source_language": SOURCE_LANGUAGE,
        "target_language": TARGET_LANGUAGE,
        "seed": SEED,
    }

    torch.save(
        checkpoint,
        CHECKPOINT_PATH,
    )

    average_training_loss = (
        sum(training_losses)
        / len(training_losses)
    )

    report = {
        "model": MODEL_NAME,
        "context_encoder": EMBEDDING_MODEL,
        "device": str(DEVICE),
        "train_examples_available": len(train_examples),
        "validation_examples_available": len(validation_examples),
        "train_limit": TRAIN_LIMIT,
        "validation_limit": VALIDATION_LIMIT,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "trainable_parameters": trainable_parameters,
        "training_losses": [
            average_training_loss
        ],
        "successful_steps": successful_steps,
        "official_test_set": False,
        "purpose": (
            "context adapter training experiment "
            "using native M2M100 label shifting"
        ),
        "nllb_frozen": True,
        "context_encoder_frozen": True,
        "adapter_trainable": True,
        "manual_decoder_input_ids": False,
        "native_label_shift": True,
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(f"Successful steps: {successful_steps}")
    print(
        f"Average training loss: "
        f"{average_training_loss:.6f}"
    )

    print()
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
