from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
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

SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "tel_Telu"

CONTEXT_DIM = 384
NLLB_HIDDEN_DIM = 1024

MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 128

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------
# Load first training example
# ---------------------------------------------------------------------

def load_first_example() -> dict:

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    with TRAIN_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                return json.loads(line)

    raise RuntimeError(
        "Training dataset is empty"
    )


# ---------------------------------------------------------------------
# Extract context text
# ---------------------------------------------------------------------

def get_context_text(example: dict) -> str:

    context = example.get(
        "context",
        [],
    )

    texts = []

    for item in context:

        if isinstance(item, dict):

            text = str(
                item.get(
                    "text",
                    "",
                )
            ).strip()

        else:

            text = str(item).strip()

        if text:
            texts.append(text)

    return " ".join(texts)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("CONTEXT ADAPTER TRAINING SMOKE TEST")
    print("=" * 70)

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"NLLB model: {MODEL_NAME}"
    )

    print(
        f"Context encoder: {EMBEDDING_MODEL}"
    )

    print(
        f"Source language: {SOURCE_LANGUAGE}"
    )

    print(
        f"Target language: {TARGET_LANGUAGE}"
    )

    print()

    # -------------------------------------------------------------
    # Load example
    # -------------------------------------------------------------

    example = load_first_example()

    source_text = str(
        example["source_sentence"]
    ).strip()

    # IMPORTANT:
    # The prepared dataset uses "target_sentence",
    # not "target_reference".
    target_text = str(
        example["target_sentence"]
    ).strip()

    context_text = get_context_text(
        example
    )

    print("Source:")
    print(source_text)

    print()
    print("Target:")
    print(target_text)

    print()
    print("Context:")
    print(context_text)

    # -------------------------------------------------------------
    # Load tokenizer
    # -------------------------------------------------------------

    print()
    print("Loading NLLB tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # -------------------------------------------------------------
    # Load NLLB
    # -------------------------------------------------------------

    print("Loading NLLB...")

    nllb = (
        M2M100ForConditionalGeneration
        .from_pretrained(
            MODEL_NAME
        )
    )

    nllb.to(DEVICE)
    nllb.eval()

    # Freeze NLLB

    for parameter in nllb.parameters():
        parameter.requires_grad = False

    print(
        "NLLB loaded."
    )

    print(
        "Model class:",
        nllb.__class__.__name__,
    )

    print(
        "Decoder start token ID:",
        nllb.config.decoder_start_token_id,
    )

    print(
        "EOS token ID:",
        nllb.config.eos_token_id,
    )

    print(
        "PAD token ID:",
        nllb.config.pad_token_id,
    )

    # -------------------------------------------------------------
    # Load MiniLM
    # -------------------------------------------------------------

    print()
    print("Loading MiniLM...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=str(DEVICE),
    )

    embedding_model.eval()

    for parameter in embedding_model.parameters():
        parameter.requires_grad = False

    print(
        "MiniLM loaded."
    )

    # -------------------------------------------------------------
    # Create adapter
    # -------------------------------------------------------------

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

    print(
        "Trainable adapter parameters:",
        trainable_parameters,
    )

    # -------------------------------------------------------------
    # Encode source
    # -------------------------------------------------------------

    tokenizer.src_lang = SOURCE_LANGUAGE

    source_inputs = tokenizer(
        source_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SOURCE_LENGTH,
    )

    input_ids = source_inputs[
        "input_ids"
    ].to(DEVICE)

    attention_mask = source_inputs[
        "attention_mask"
    ].to(DEVICE)

    print(
        "Source input shape:",
        tuple(input_ids.shape),
    )

    # -------------------------------------------------------------
    # Encode context
    # -------------------------------------------------------------

    if context_text:

        context_embedding = (
            embedding_model.encode(
                context_text,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

        # SentenceTransformer may return an inference-mode tensor.
        # Clone/detach makes it safe for the adapter autograd graph.

        context_embedding = (
            context_embedding
            .clone()
            .detach()
        )

        context_embedding = (
            context_embedding.to(
                device=DEVICE,
                dtype=torch.float32,
            )
        )

    else:

        context_embedding = torch.zeros(
            CONTEXT_DIM,
            dtype=torch.float32,
            device=DEVICE,
        )

    context_embedding = (
        context_embedding
        .unsqueeze(0)
    )

    print(
        "Context embedding shape:",
        tuple(context_embedding.shape),
    )

    # -------------------------------------------------------------
    # NLLB encoder
    # -------------------------------------------------------------

    with torch.no_grad():

        encoder_outputs = (
            nllb.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        )

    encoder_hidden = (
        encoder_outputs.last_hidden_state
    )

    print(
        "Encoder hidden shape:",
        tuple(encoder_hidden.shape),
    )

    # -------------------------------------------------------------
    # Context adapter
    # -------------------------------------------------------------

    fused_hidden = adapter(
        encoder_hidden,
        context_embedding,
    )

    print(
        "Fused hidden shape:",
        tuple(fused_hidden.shape),
    )

    # -------------------------------------------------------------
    # Encode target labels
    # -------------------------------------------------------------

    tokenizer.tgt_lang = TARGET_LANGUAGE

    target_inputs = tokenizer(
        text_target=target_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TARGET_LENGTH,
    )

    labels = target_inputs[
        "input_ids"
    ].to(DEVICE)

    labels[
        labels == tokenizer.pad_token_id
    ] = -100

    print(
        "Target label shape:",
        tuple(labels.shape),
    )

    # -------------------------------------------------------------
    # Native M2M100 label shifting
    # -------------------------------------------------------------
    #
    # IMPORTANT:
    #
    # We intentionally DO NOT manually construct
    # decoder_input_ids.
    #
    # Transformers 5.18.0 performs:
    #
    # shift_tokens_right(
    #     labels,
    #     self.config.pad_token_id,
    #     self.config.decoder_start_token_id
    # )
    #
    # internally when labels are supplied.
    # -------------------------------------------------------------

    print()
    print(
        "Using native M2M100 label shifting..."
    )

    fused_encoder_outputs = BaseModelOutput(
        last_hidden_state=fused_hidden
    )

    outputs = nllb(
        encoder_outputs=fused_encoder_outputs,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True,
    )

    loss = outputs.loss

    # -------------------------------------------------------------
    # Validate loss
    # -------------------------------------------------------------

    if loss is None:
        raise RuntimeError(
            "NLLB returned no loss."
        )

    print(
        "Loss:",
        loss.item(),
    )

    loss_finite = bool(
        torch.isfinite(loss)
    )

    print(
        "Loss finite:",
        loss_finite,
    )

    if not loss_finite:
        raise RuntimeError(
            "Loss is not finite."
        )

    # -------------------------------------------------------------
    # Backward pass
    # -------------------------------------------------------------

    print()
    print(
        "Running backward pass..."
    )

    loss.backward()

    # -------------------------------------------------------------
    # Check adapter gradients
    # -------------------------------------------------------------

    adapter_gradient_tensors = 0

    for parameter in adapter.parameters():

        if parameter.grad is None:
            continue

        if torch.isfinite(
            parameter.grad
        ).all():

            adapter_gradient_tensors += 1

    print(
        "Adapter tensors receiving gradients:",
        adapter_gradient_tensors,
    )

    if adapter_gradient_tensors == 0:
        raise RuntimeError(
            "No adapter tensors received valid gradients."
        )

    # -------------------------------------------------------------
    # Final result
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()