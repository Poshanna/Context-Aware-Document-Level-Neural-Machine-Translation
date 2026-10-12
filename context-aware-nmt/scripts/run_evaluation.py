#!/usr/bin/env python3
"""Run a reproducible sentence-level or legitimate-context evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.evaluation.experiment_runner import run_evaluation


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate NLLB baseline and only legitimate context-conditioned examples."
    )
    parser.add_argument("--source-lang", required=True, choices=["en", "te"])
    parser.add_argument("--target-lang", required=True, choices=["en", "te"])
    parser.add_argument("--max-samples", type=_positive_int, default=50)
    parser.add_argument("--seed", type=_nonnegative_int, default=42)
    parser.add_argument("--context-window", type=_nonnegative_int, default=1)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments/evaluation")
    parser.add_argument(
        "--include-bertscore", action="store_true",
        help="Explicitly compute optional BERTScore; its checkpoint may need to be downloaded.",
    )
    parser.add_argument(
        "--include-comet", action="store_true",
        help="Explicitly request COMET; requires ENABLE_COMET=1 and its optional package/model.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.source_lang == args.target_lang:
        parser.error("--source-lang and --target-lang must differ")
    try:
        run_dir = run_evaluation(
            source_lang=args.source_lang, target_lang=args.target_lang,
            max_samples=args.max_samples, seed=args.seed,
            context_window=args.context_window, output_dir=args.output_dir,
            include_bertscore=args.include_bertscore, include_comet=args.include_comet,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": "complete", "artifact_dir": str(run_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())