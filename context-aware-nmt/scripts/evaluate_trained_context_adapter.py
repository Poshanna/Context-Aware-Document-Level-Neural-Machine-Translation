from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from sacrebleu.metrics import BLEU, CHRF
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MANIFEST_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "evaluation"
    / "dochplt"
    / "final_eval_manifest.json"
)

ADAPTER_PATH = (
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
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "facebook/nllb-200-distilled-600M"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "tel_Telu"

CONTEXT_DIM = 384
NLLB_HIDDEN_DIM = 1024

MAX_SOURCE_LENGTH = 128
MAX_NEW_TOKENS = 128

DEVICE = torch.device("cpu")


# ============================================================
# ESTABLISHED BASELINE RESULTS
# ============================================================

BASELINE_BLEU = 41.27782044840121
BASELINE_CHRF = 53.482361839304126

PREFIX_BLEU = 10.21926721578879
PREFIX_CHRF = 37.024279422389014


# ============================================================
# CONTEXT ADAPTER
# ============================================================

class ContextAdapter(torch.nn.Module):
    """
    Learned context adapter used during training.

    Input:
        NLLB encoder hidden states:
            [batch, source_length, 1024]

        MiniLM context embedding:
            [batch, 384]

    Output:
        Fused encoder hidden states:
            [batch, source_length, 1024]
    """

    def __init__(
        self,
        context_dim: int = CONTEXT_DIM,
        hidden_dim: int = NLLB_HIDDEN_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.context_projection = torch.nn.Sequential(
            torch.nn.Linear(
                context_dim,
                hidden_dim,
            ),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )

        self.gate = torch.nn.Sequential(
            torch.nn.Linear(
                hidden_dim * 2,
                hidden_dim,
            ),
            torch.nn.Sigmoid(),
        )

        # Same initialization used during training.
        gate_linear = self.gate[0]

        torch.nn.init.xavier_uniform_(
            gate_linear.weight
        )

        torch.nn.init.constant_(
            gate_linear.bias,
            -4.0,
        )

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        context_embedding: torch.Tensor,
    ) -> torch.Tensor:

        projected_context = (
            self.context_projection(
                context_embedding
            )
        )

        context_sequence = (
            projected_context
            .unsqueeze(1)
            .expand(
                -1,
                encoder_hidden_states.size(1),
                -1,
            )
        )

        fusion_input = torch.cat(
            [
                encoder_hidden_states,
                context_sequence,
            ],
            dim=-1,
        )

        gate = self.gate(
            fusion_input
        )

        fused = (
            encoder_hidden_states
            + gate * context_sequence
        )

        return fused


# ============================================================
# MANIFEST
# ============================================================

def load_manifest() -> dict:

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "Evaluation manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    payload = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Evaluation manifest must be a JSON object."
        )

    examples = payload.get(
        "examples",
        [],
    )

    if not isinstance(examples, list):
        raise ValueError(
            "Manifest 'examples' must be a list."
        )

    if len(examples) != 12:
        raise ValueError(
            "Expected exactly 12 manually reviewed "
            "DocHPLT examples, but found "
            f"{len(examples)}."
        )

    return payload


# ============================================================
# CONTEXT
# ============================================================

def build_context_text(
    example: dict,
) -> str:

    context = example.get(
        "context",
        [],
    )

    texts = []

    for item in context:

        if not isinstance(item, dict):
            continue

        text = str(
            item.get(
                "text",
                "",
            )
        ).strip()

        if text:
            texts.append(text)

    return " ".join(texts)


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    print("=" * 70)
    print("LOADING NLLB")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    nllb = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    ).to(DEVICE)

    nllb.eval()

    for parameter in nllb.parameters():
        parameter.requires_grad = False

    print("NLLB loaded")
    print(
        f"Model class: "
        f"{nllb.__class__.__name__}"
    )

    print()
    print("=" * 70)
    print("LOADING MINILM CONTEXT ENCODER")
    print("=" * 70)

    context_encoder = SentenceTransformer(
        EMBEDDING_MODEL,
        device=str(DEVICE),
    )

    context_encoder.eval()

    for parameter in context_encoder.parameters():
        parameter.requires_grad = False

    print("MiniLM loaded")
    print(
        f"Embedding dimension: "
        f"{CONTEXT_DIM}"
    )

    print()
    print("=" * 70)
    print("LOADING TRAINED CONTEXT ADAPTER")
    print("=" * 70)

    if not ADAPTER_PATH.is_file():
        raise FileNotFoundError(
            "Context adapter checkpoint not found:\n"
            f"{ADAPTER_PATH}"
        )

    adapter = ContextAdapter(
        context_dim=CONTEXT_DIM,
        hidden_dim=NLLB_HIDDEN_DIM,
    ).to(DEVICE)

    # --------------------------------------------------------
    # IMPORTANT:
    # The training script saves a structured checkpoint:
    #
    # {
    #     "model_state_dict": ...,
    #     "context_dim": ...,
    #     "hidden_dim": ...,
    #     ...
    # }
    #
    # Therefore we must extract model_state_dict.
    # --------------------------------------------------------

    checkpoint = torch.load(
        ADAPTER_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise ValueError(
            "Context adapter checkpoint must "
            "be a dictionary."
        )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Context adapter checkpoint does not "
            "contain 'model_state_dict'."
        )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    if not isinstance(
        state_dict,
        dict,
    ):
        raise ValueError(
            "'model_state_dict' must be a dictionary."
        )

    adapter.load_state_dict(
        state_dict,
        strict=True,
    )

    adapter.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in adapter.parameters()
    )

    print("Context Adapter loaded")
    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    print()
    print(
        "Checkpoint metadata:"
    )

    metadata_keys = [
        "context_dim",
        "hidden_dim",
        "embedding_model",
        "nllb_model",
        "source_language",
        "target_language",
        "seed",
    ]

    for key in metadata_keys:

        if key in checkpoint:
            print(
                f"  {key}: "
                f"{checkpoint[key]}"
            )

    return (
        tokenizer,
        nllb,
        context_encoder,
        adapter,
    )


# ============================================================
# CONTEXT EMBEDDING
# ============================================================

@torch.no_grad()
def encode_context(
    context_encoder: SentenceTransformer,
    context_text: str,
) -> torch.Tensor:

    if not context_text.strip():

        return torch.zeros(
            CONTEXT_DIM,
            dtype=torch.float32,
            device=DEVICE,
        )

    embedding = context_encoder.encode(
        context_text,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # SentenceTransformer may return an inference tensor.
    # Clone it before using it elsewhere.
    embedding = embedding.clone().detach()

    embedding = embedding.to(
        device=DEVICE,
        dtype=torch.float32,
    )

    return embedding


# ============================================================
# ADAPTER TRANSLATION
# ============================================================

@torch.no_grad()
def translate_with_adapter(
    tokenizer,
    nllb,
    context_encoder,
    adapter,
    source_text: str,
    context_text: str,
):

    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    tokenizer.src_lang = SOURCE_LANGUAGE

    source_tokens = tokenizer(
        source_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SOURCE_LENGTH,
    )

    input_ids = (
        source_tokens["input_ids"]
        .to(DEVICE)
    )

    attention_mask = (
        source_tokens["attention_mask"]
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # CONTEXT EMBEDDING
    # --------------------------------------------------------

    context_embedding = encode_context(
        context_encoder,
        context_text,
    )

    context_embedding = (
        context_embedding
        .unsqueeze(0)
    )

    # --------------------------------------------------------
    # NLLB ENCODER
    # --------------------------------------------------------

    encoder_outputs = (
        nllb.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
    )

    # --------------------------------------------------------
    # LEARNED CONTEXT FUSION
    # --------------------------------------------------------

    fused_hidden_states = adapter(
        encoder_outputs.last_hidden_state,
        context_embedding,
    )

    adapted_encoder_outputs = (
        BaseModelOutput(
            last_hidden_state=
                fused_hidden_states
        )
    )

    # --------------------------------------------------------
    # TARGET LANGUAGE
    # --------------------------------------------------------

    target_language_id = (
        tokenizer.convert_tokens_to_ids(
            TARGET_LANGUAGE
        )
    )

    if target_language_id is None:
        raise ValueError(
            "Could not resolve target language "
            f"token: {TARGET_LANGUAGE}"
        )

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    generated_ids = nllb.generate(
        encoder_outputs=
            adapted_encoder_outputs,

        attention_mask=attention_mask,

        forced_bos_token_id=
            target_language_id,

        max_new_tokens=
            MAX_NEW_TOKENS,

        do_sample=False,

        num_beams=1,
    )

    translation = (
        tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        .strip()
    )

    return translation


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    hypotheses,
    references,
):

    bleu_metric = BLEU()

    chrf_metric = CHRF(
        word_order=2
    )

    bleu_score = (
        bleu_metric
        .corpus_score(
            hypotheses,
            [references],
        )
        .score
    )

    chrf_score = (
        chrf_metric
        .corpus_score(
            hypotheses,
            [references],
        )
        .score
    )

    return (
        float(bleu_score),
        float(chrf_score),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "TRAINED CONTEXT ADAPTER EVALUATION"
    )
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Manifest: {MANIFEST_PATH}"
    )

    print(
        f"Adapter: {ADAPTER_PATH}"
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = load_manifest()

    examples = manifest[
        "examples"
    ]

    print()
    print(
        f"Evaluation examples: "
        f"{len(examples)}"
    )

    print(
        "Official test set: False"
    )

    print(
        "Evaluation type: manually reviewed "
        "deterministic DocHPLT subset"
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    (
        tokenizer,
        nllb,
        context_encoder,
        adapter,
    ) = load_models()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    hypotheses = []
    references = []

    example_results = []
    failures = []

    print()
    print("=" * 70)
    print("RUNNING TRANSLATIONS")
    print("=" * 70)

    for index, example in enumerate(
        examples,
        start=1,
    ):

        evaluation_index = example.get(
            "evaluation_index",
            index,
        )

        source_text = example[
            "source_sentence"
        ]

        reference = example[
            "target_reference"
        ]

        context_items = example.get(
            "context",
            [],
        )

        context_text = (
            build_context_text(
                example
            )
        )

        print()
        print(
            f"[{index}/{len(examples)}]"
        )

        print(
            f"Evaluation index: "
            f"{evaluation_index}"
        )

        print(
            f"Source: "
            f"{source_text}"
        )

        print(
            f"Context sentences: "
            f"{len(context_items)}"
        )

        try:

            hypothesis = (
                translate_with_adapter(
                    tokenizer=tokenizer,
                    nllb=nllb,
                    context_encoder=
                        context_encoder,
                    adapter=adapter,
                    source_text=
                        source_text,
                    context_text=
                        context_text,
                )
            )

            print(
                f"Adapter output: "
                f"{hypothesis}"
            )

            hypotheses.append(
                hypothesis
            )

            references.append(
                reference
            )

            example_results.append(
                {
                    "evaluation_index":
                        evaluation_index,

                    "manual_review_index":
                        example.get(
                            "manual_review_index"
                        ),

                    "source_document_id":
                        example.get(
                            "source_document_id"
                        ),

                    "target_document_id":
                        example.get(
                            "target_document_id"
                        ),

                    "source_sentence_id":
                        example.get(
                            "source_sentence_id"
                        ),

                    "target_sentence_id":
                        example.get(
                            "target_sentence_id"
                        ),

                    "source_sentence":
                        source_text,

                    "target_reference":
                        reference,

                    "context":
                        context_items,

                    "context_text":
                        context_text,

                    "hypothesis":
                        hypothesis,

                    "status":
                        "success",
                }
            )

        except Exception as exc:

            print(
                "ERROR: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            failures.append(
                {
                    "evaluation_index":
                        evaluation_index,

                    "source_sentence":
                        source_text,

                    "error_type":
                        type(exc).__name__,

                    "error":
                        str(exc),
                }
            )

            example_results.append(
                {
                    "evaluation_index":
                        evaluation_index,

                    "source_sentence":
                        source_text,

                    "target_reference":
                        reference,

                    "status":
                        "failed",
                }
            )

    # --------------------------------------------------------
    # Require successful evaluation
    # --------------------------------------------------------

    if not hypotheses:

        raise RuntimeError(
            "No successful translations "
            "were produced."
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    bleu_score, chrf_score = (
        calculate_metrics(
            hypotheses,
            references,
        )
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Successful examples: "
        f"{len(hypotheses)}"
    )

    print(
        f"Failed examples: "
        f"{len(failures)}"
    )

    print(
        f"BLEU:   "
        f"{bleu_score:.8f}"
    )

    print(
        f"chrF++: "
        f"{chrf_score:.8f}"
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    adapter_minus_baseline_bleu = (
        bleu_score
        - BASELINE_BLEU
    )

    adapter_minus_baseline_chrf = (
        chrf_score
        - BASELINE_CHRF
    )

    adapter_minus_prefix_bleu = (
        bleu_score
        - PREFIX_BLEU
    )

    adapter_minus_prefix_chrf = (
        chrf_score
        - PREFIX_CHRF
    )

    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(
        "Sentence-level NLLB baseline:"
    )

    print(
        f"  BLEU   = "
        f"{BASELINE_BLEU:.8f}"
    )

    print(
        f"  chrF++ = "
        f"{BASELINE_CHRF:.8f}"
    )

    print()

    print(
        "Naive / semantic prefix:"
    )

    print(
        f"  BLEU   = "
        f"{PREFIX_BLEU:.8f}"
    )

    print(
        f"  chrF++ = "
        f"{PREFIX_CHRF:.8f}"
    )

    print()

    print(
        "Trained Context Adapter:"
    )

    print(
        f"  BLEU   = "
        f"{bleu_score:.8f}"
    )

    print(
        f"  chrF++ = "
        f"{chrf_score:.8f}"
    )

    print()

    print(
        "Adapter - baseline:"
    )

    print(
        f"  BLEU   = "
        f"{adapter_minus_baseline_bleu:+.8f}"
    )

    print(
        f"  chrF++ = "
        f"{adapter_minus_baseline_chrf:+.8f}"
    )

    print()

    print(
        "Adapter - prefix:"
    )

    print(
        f"  BLEU   = "
        f"{adapter_minus_prefix_bleu:+.8f}"
    )

    print(
        f"  chrF++ = "
        f"{adapter_minus_prefix_chrf:+.8f}"
    )

    # --------------------------------------------------------
    # Save evaluation artifacts
    # --------------------------------------------------------

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    run_dir = (
        OUTPUT_ROOT
        / f"{timestamp}-context-adapter"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    summary = {
        "experiment":
            "trained_context_adapter",

        "dataset":
            "DocHPLT",

        "config":
            "en-te",

        "split":
            "train",

        "evaluation_type":
            (
                "manually reviewed deterministic "
                "subset from a DocHPLT training shard"
            ),

        "official_test_set":
            False,

        "document_level_nmt":
            False,

        "model":
            MODEL_NAME,

        "context_encoder":
            EMBEDDING_MODEL,

        "adapter_checkpoint":
            str(ADAPTER_PATH),

        "direction":
            "en-te",

        "context_window":
            2,

        "sample_count":
            len(hypotheses),

        "failure_count":
            len(failures),

        "target_side_used_for_ranking":
            False,

        "bleu":
            bleu_score,

        "chrf++":
            chrf_score,

        "baseline_bleu":
            BASELINE_BLEU,

        "baseline_chrf++":
            BASELINE_CHRF,

        "prefix_bleu":
            PREFIX_BLEU,

        "prefix_chrf++":
            PREFIX_CHRF,

        "adapter_minus_baseline_bleu":
            adapter_minus_baseline_bleu,

        "adapter_minus_baseline_chrf++":
            adapter_minus_baseline_chrf,

        "adapter_minus_prefix_bleu":
            adapter_minus_prefix_bleu,

        "adapter_minus_prefix_chrf++":
            adapter_minus_prefix_chrf,
    }

    summary_path = (
        run_dir / "summary.json"
    )

    examples_path = (
        run_dir / "examples.json"
    )

    failures_path = (
        run_dir / "failures.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    examples_path.write_text(
        json.dumps(
            example_results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    failures_path.write_text(
        json.dumps(
            failures,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final paths
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ARTIFACTS")
    print("=" * 70)

    print(
        f"Run directory: "
        f"{run_dir}"
    )

    print(
        f"Summary: "
        f"{summary_path}"
    )

    print(
        f"Examples: "
        f"{examples_path}"
    )

    print(
        f"Failures: "
        f"{failures_path}"
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()