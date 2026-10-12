"""Run one explicit real-checkpoint baseline translation smoke test."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings  # noqa: E402
from app.schemas import TextTranslateIn  # noqa: E402
from app.services.translation_service import (  # noqa: E402
    TranslationService,
    TranslationServiceError,
)


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Explicitly run one sentence-level NLLB-200 baseline translation. "
            "This command may load/download the configured pretrained checkpoint."
        )
    )
    parser.add_argument(
        "--source-lang",
        choices=["en", "te"],
        default=settings.TRANSLATION_SOURCE_LANGUAGE,
    )
    parser.add_argument(
        "--target-lang",
        choices=["en", "te"],
        default=settings.TRANSLATION_TARGET_LANGUAGE,
    )
    parser.add_argument("--model", choices=["nllb200", "indictrans2"], default=settings.MODEL_DEFAULT)
    parser.add_argument("--text", default="Hello, how are you?")
    return parser


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.source_lang == args.target_lang:
        parser.error("--source-lang and --target-lang must differ")

    request = TextTranslateIn(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        pipeline="baseline",
        model=args.model,
        text=args.text,
    )
    service = TranslationService(settings=settings)
    try:
        response = service.translate_text(user_id=uuid.uuid4(), req=request)
    except TranslationServiceError as exc:
        print(f"ERROR [{exc.code}]: {exc.message}", file=sys.stderr)
        return 1

    if response.status != "ok" or response.translated_text is None:
        print("ERROR: baseline translation returned no translation.", file=sys.stderr)
        return 1
    print(response.translated_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
