#!/usr/bin/env python3
"""Compare sentence-level baseline translation against a context-prefixed input."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings
from app.schemas import TextTranslateIn
from app.services.context_translation_service import ContextAwareTranslationService
from app.services.translation_service import TranslationService


def _sentence_splitter(text: str) -> list[str]:
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    sentences: list[str] = []
    for line in lines:
        if not line:
            continue
        for chunk in [piece.strip() for piece in line.split(".") if piece.strip()]:
            if chunk:
                sentences.append(chunk + ("." if not chunk.endswith(".") and not chunk.endswith("?") and not chunk.endswith("!") else ""))
    return sentences if sentences else [text.strip()] if text.strip() else []


def _safe_translate(service: TranslationService, *, source_lang: str, target_lang: str, sentence: str) -> str:
    response = service.translate_text(
        user_id=None,  # type: ignore[arg-type]
        req=TextTranslateIn(
            source_lang=source_lang,
            target_lang=target_lang,
            pipeline="baseline",
            model="nllb200",
            text=sentence,
        ),
    )
    if response.status != "ok" or response.translated_text is None:
        raise RuntimeError(response.note or "Baseline translation failed.")
    return response.translated_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline vs context-conditioned translation for a short sample.")
    parser.add_argument("--source-lang", required=True, choices=["en", "te"])
    parser.add_argument("--target-lang", required=True, choices=["en", "te"])
    parser.add_argument("--text", required=True)
    parser.add_argument("--context-window", type=int, default=1)
    args = parser.parse_args()

    settings = get_settings()
    translation_service = TranslationService(settings=settings)
    context_service = ContextAwareTranslationService(settings=settings)

    sentences = _sentence_splitter(args.text)

    baseline_outputs: list[str] = []
    prefix_outputs: list[str] = []
    contexts: list[list[str]] = []
    model_inputs: list[str] = []
    metadata: dict[str, object] = {
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "context_window": args.context_window,
        "context_mode": "prefix_conditioned",
    }

    print("SOURCE DOCUMENT")
    print("================")
    print(args.text)
    print()

    print("BASELINE")
    print("========")
    for sentence in sentences:
        try:
            translation = _safe_translate(translation_service, source_lang=args.source_lang, target_lang=args.target_lang, sentence=sentence)
            baseline_outputs.append(translation)
            print(f"- {sentence} -> {translation}")
        except Exception as exc:  # pragma: no cover - runtime-dependent path
            baseline_outputs.append(f"[not available: {exc}]")
            print(f"- {sentence} -> [not available: {exc}]")
    print()

    print("PREFIX-CONDITIONED")
    print("==================")
    try:
        document_response = context_service.translate_document(
            user_id=None,  # type: ignore[arg-type]
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            text=args.text,
            context_window=args.context_window,
            context_mode="prefix_conditioned",
        )
        prefix_outputs = document_response.translations
        contexts = [sentence.context for sentence in document_response.sentences]
        model_inputs = [sentence.formatted_input for sentence in document_response.sentences]
        metadata.update({
            "context_used_for_translation": document_response.metadata.get("context_used_for_translation", True),
            "model_backend": document_response.model_backend,
            "translation_model_type": document_response.metadata.get("translation_model_type"),
        })
        for sentence, translation in zip(document_response.sentences, document_response.translations):
            print(f"- {sentence.source} -> {translation}")
    except Exception as exc:  # pragma: no cover - runtime-dependent path
        print(f"[prefix-conditioned translation unavailable: {exc}]")
        metadata.update({
            "context_used_for_translation": False,
            "model_backend": "nllb200",
            "translation_model_type": "sentence_level_pretrained_checkpoint",
            "error": str(exc),
        })
    print()

    print("CONTEXT USED")
    print("============")
    for index, sentence in enumerate(sentences):
        context = contexts[index] if index < len(contexts) else []
        print(f"{index + 1}. {sentence} -> {context if context else '[none]'}")
    print()

    print("MODEL INPUT")
    print("===========")
    if model_inputs:
        for index, model_input in enumerate(model_inputs, start=1):
            print(f"{index}.\n{model_input}\n")
    else:
        print("[no model input recorded]")
    print()

    print("METADATA")
    print("========")
    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
