#!/usr/bin/env python3
"""Validate local OPUS TED2020 English-Telugu XML inputs; never downloads data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.evaluation.ted2020_loader import load_ted2020, write_report


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a locally supplied TED2020 en-te cesAlign XML and language "
            "XML directories/ZIP archives. No downloads or text exports are performed."
        )
    )
    parser.add_argument("--alignment", type=Path, required=True, help="Local en-te.xml or en-te.xml.gz")
    parser.add_argument("--english", type=Path, required=True, help="Local English XML directory or ZIP")
    parser.add_argument("--telugu", type=Path, required=True, help="Local Telugu XML directory or ZIP")
    parser.add_argument("--context-window", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--report-output", type=Path,
        help="Optional destination for JSON statistics only; no corpus text is written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = load_ted2020(
            args.alignment,
            args.english,
            args.telugu,
            context_window=args.context_window,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.error(str(exc))
    write_report(result.report, args.report_output)
    return 0 if result.report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
