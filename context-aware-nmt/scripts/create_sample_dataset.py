"""Extract a small committed-size sample dataset from a processed JSONL split.

Outputs two parallel ``.txt`` files::

    data/samples/samanantar_sample.en.txt
    data/samples/samanantar_sample.te.txt

Usage examples::

    python scripts/create_sample_dataset.py --size 1000
    python scripts/create_sample_dataset.py --size 10000 --i-accept-the-large-size
    python scripts/create_sample_dataset.py --source data/processed/train.en-te.jsonl \\
        --output-dir data/samples --size 50

The default ``--source`` split is ``train.en-te.jsonl`` because we want the
sample to reflect the training distribution that the model actually sees.

Per project_rules.md §1.4 — committed samples under ``data/samples/`` MUST be
tiny (<50 sentences).  Larger samples require the ``--i-accept-the-large-size``
acknowledgement flag and MUST NOT be committed to Git.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.preprocessing.dataset_splitter import read_split_jsonl  # noqa: E402


DEFAULT_SOURCE = PROJECT_ROOT / "data" / "processed" / "train.en-te.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "samples"
HARD_MAX_WITHOUT_FLAG = 10000
RECOMMENDED_COMMIT_MAX = 50
SYNTHETIC_PAIRS = [
    ("Synthetic test sentence one.", "కృత్రిమ పరీక్ష వాక్యం ఒకటి."),
    ("Synthetic test sentence two.", "కృత్రిమ పరీక్ష వాక్యం రెండు."),
    ("Synthetic test sentence three.", "కృత్రిమ పరీక్ష వాక్యం మూడు."),
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Extract a sample from a processed split, or create explicitly labeled "
            "synthetic test data. No document relationships are generated."
        )
    )
    p.add_argument(
        "--size",
        type=int,
        required=True,
        help=f"How many sentence pairs to extract (recommended ≤ {RECOMMENDED_COMMIT_MAX} for Git commit).",
    )
    p.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Source JSONL split (default: {DEFAULT_SOURCE}).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write sample txt files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Write fixed synthetic_test records instead of reading Samanantar output.",
    )
    p.add_argument(
        "--i-accept-the-large-size",
        action="store_true",
        help=(
            f"Acknowledge that --size > {HARD_MAX_WITHOUT_FLAG} is intentional. "
            "Large samples must NOT be committed to Git."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.size <= 0:
        print(f"ERROR: --size must be positive (got {args.size}).", file=sys.stderr)
        return 2

    if args.size > HARD_MAX_WITHOUT_FLAG and not args.i_accept_the_large_size:
        print(
            f"ERROR: --size={args.size} exceeds the safety ceiling of "
            f"{HARD_MAX_WITHOUT_FLAG}. Re-run with --i-accept-the-large-size to "
            f"confirm you will not commit the output files to Git. For committed "
            f"samples we recommend --size ≤ {RECOMMENDED_COMMIT_MAX} per "
            "project_rules.md §1.4.",
            file=sys.stderr,
        )
        return 2

    if args.synthetic:
        if args.size > len(SYNTHETIC_PAIRS):
            print(
                f"ERROR: synthetic test data contains only {len(SYNTHETIC_PAIRS)} "
                "controlled pairs; choose a smaller --size.",
                file=sys.stderr,
            )
            return 2
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "synthetic_test.jsonl"
        with output_path.open("w", encoding="utf-8") as output_file:
            for index, (source_text, target_text) in enumerate(
                SYNTHETIC_PAIRS[: args.size]
            ):
                record = {
                    "id": index,
                    "source_text": source_text,
                    "target_text": target_text,
                    "source_language": "en",
                    "target_language": "te",
                    "data_source": "synthetic_test",
                }
                output_file.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
        print(
            f"Wrote {args.size} explicitly synthetic test pairs to {output_path}. "
            "They contain no document IDs or document relationships."
        )
        return 0

    if not args.source.exists():
        print(
            f"ERROR: source JSONL not found at {args.source}. "
            "Run `python scripts/prepare_samanantar.py` first to produce the split files.",
            file=sys.stderr,
        )
        return 1

    records = read_split_jsonl(args.source)
    take = records[: args.size]
    if len(take) < args.size:
        print(
            f"Warning: requested --size={args.size} but source has only "
            f"{len(records)} rows; taking all {len(records)}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if records:
        source_lang = records[0].source_language
        target_lang = records[0].target_language
    else:
        source_lang, target_lang = "en", "te"
    source_path = args.output_dir / f"samanantar_sample.{source_lang}.txt"
    target_path = args.output_dir / f"samanantar_sample.{target_lang}.txt"

    src_chars = 0
    tgt_chars = 0
    with source_path.open("w", encoding="utf-8") as enf, target_path.open(
        "w", encoding="utf-8"
    ) as tef:
        for r in take:
            # Samanantar loader records source/target as the pair specified during load.
            # For the standard en-te direction source_lang="en" target_lang="te", so
            # write accordingly. If user swapped directions, the files will still be
            # parallel lines even if the lang suffixes are backwards in that non-standard case.
            enf.write(r.source_text + "\n")
            tef.write(r.target_text + "\n")
            src_chars += len(r.source_text)
            tgt_chars += len(r.target_text)

    print(
        f"Wrote {len(take)} parallel pairs:\n"
        f"  - {source_path} ({src_chars} source chars)\n"
        f"  - {target_path} ({tgt_chars} target chars)\n"
    )
    if args.size > RECOMMENDED_COMMIT_MAX:
        print(
            f"Note: {len(take)} rows > recommended commit max "
            f"({RECOMMENDED_COMMIT_MAX}). Do NOT `git add` these sample files."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
