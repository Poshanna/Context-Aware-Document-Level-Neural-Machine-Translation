from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from app.translation.nllb200 import NLLB200Translator
from app.core.config import get_settings


def main() -> None:
    print("=" * 80)
    print("Context Adapter Dimension Inspection")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load existing NLLB backend
    # ------------------------------------------------------------------

    print()
    print("Loading existing NLLB-200 backend...")

    settings = get_settings()

    translator = NLLB200Translator(
        settings=settings
    )

    translator.load()

    print("NLLB-200 loaded.")

    # ------------------------------------------------------------------
    # Inspect model
    # ------------------------------------------------------------------

    model = translator._model

    if model is None:
        raise RuntimeError(
            "NLLB model was not loaded."
        )

    config = model.config

    print()
    print("=" * 80)
    print("NLLB Model Configuration")
    print("=" * 80)

    print()
    print(
        f"Model class: "
        f"{model.__class__.__name__}"
    )

    print(
        f"Hidden size: "
        f"{getattr(config, 'd_model', 'N/A')}"
    )

    print(
        f"Encoder layers: "
        f"{getattr(config, 'encoder_layers', 'N/A')}"
    )

    print(
        f"Decoder layers: "
        f"{getattr(config, 'decoder_layers', 'N/A')}"
    )

    print(
        f"Attention heads: "
        f"{getattr(config, 'encoder_attention_heads', 'N/A')}"
    )

    print(
        f"Vocabulary size: "
        f"{getattr(config, 'vocab_size', 'N/A')}"
    )

    # ------------------------------------------------------------------
    # Load MiniLM
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("Loading Semantic Embedding Model")
    print("=" * 80)

    from sentence_transformers import SentenceTransformer

    embedding_model_name = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embedder = SentenceTransformer(
        embedding_model_name
    )

    dimension = embedder.get_sentence_embedding_dimension()

    print()
    print(
        f"Embedding model: "
        f"{embedding_model_name}"
    )

    print(
        f"Embedding dimension: "
        f"{dimension}"
    )

    # ------------------------------------------------------------------
    # Compatibility summary
    # ------------------------------------------------------------------

    nllb_hidden_size = getattr(
        config,
        "d_model",
        None,
    )

    print()
    print("=" * 80)
    print("Adapter Dimension Summary")
    print("=" * 80)

    print()

    print(
        f"NLLB encoder representation: "
        f"{nllb_hidden_size}"
    )

    print(
        f"Semantic context representation: "
        f"{dimension}"
    )

    if (
        isinstance(nllb_hidden_size, int)
        and isinstance(dimension, int)
    ):
        print()
        print(
            "Required adapter:"
        )

        print(
            f"Linear({dimension} -> {nllb_hidden_size})"
        )

    print()
    print("=" * 80)
    print("Inspection completed")
    print("=" * 80)


if __name__ == "__main__":
    main()