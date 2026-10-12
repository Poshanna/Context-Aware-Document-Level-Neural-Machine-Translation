"""Main Samanantar preprocessing entrypoint.

Runs the full pipeline::

    Raw dataset (user-downloaded under SAMANANTAR_DATA_PATH)
    ↓ load (SamanantarLoader, paired text / JSONL / TSV / Parquet auto-detect)
      ↓ en↔te pair filter (loader already restricts by src_lang/tgt_lang)
      ↓ clean  (normalize whitespace → drop empties → dedupe → length filter)
      ↓ split  (deterministic 80/10/10 seeded shuffle)
      ↓ write JSONL splits to data/processed/
      ↓ (optional) generate sample to data/samples/
      ↓ write experiments/dataset_preprocessing_report.json

**Never** auto-downloads the Samanantar tarball — if the expected raw files are
not present this script exits with code 1 and a clear remediation message
pointing at :doc:`docs/dataset.md`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings  # noqa: E402
from app.preprocessing.dataset_loader import (  # noqa: E402
    CleaningReport,
    SamanantarLoader,
    clean_records,
)
from app.preprocessing.dataset_splitter import (  # noqa: E402
    split_dataset,
    write_splits_jsonl,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORT_PATH = EXPERIMENTS_DIR / "dataset_preprocessing_report.json"
DEFAULT_SAMPLE_SIZE = 0
SAMANANTAR_SOURCE_URL = "https://indicnlp.ai4bharat.org/samanantar/"
SAMANANTAR_UPSTREAM_REVISION = "ead34c5b22a0354d350e7dfdd6aece7df5a48f37"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the full Samanantar preprocessing pipeline and produce split "
            "JSONL files + a preprocessing statistics report."
        )
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=(
            "If > 0 also extract N rows from train split and write "
            "data/samples/samanantar_sample.{en,te}.txt via "
            "create_sample_dataset.py."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Split random seed (default: read DATASET_SEED from Settings / .env or fallback 42).",
    )
    p.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=(
            "Override SAMANANTAR_DATA_PATH for this run. If omitted we use the "
            "Settings value from .env. If that is also None, we error out with "
            "instructions to download."
        ),
    )
    p.add_argument(
        "--src-lang",
        default="en",
        choices=["en", "te"],
    )
    p.add_argument(
        "--tgt-lang",
        default="te",
        choices=["en", "te"],
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=2,
        help="Minimum source/target characters (default 2).",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=1000,
        help="Maximum source/target characters per sentence (default 1000).",
    )
    p.add_argument(
        "--train",
        type=float,
        default=0.8,
        help="Train split proportion (default 0.8).",
    )
    p.add_argument(
        "--val",
        type=float,
        default=0.1,
        help="Validation split proportion (default 0.1).",
    )
    p.add_argument(
        "--test",
        type=float,
        default=0.1,
        help="Test split proportion (default 0.1).",
    )
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help=f"Where to write split JSONL files (default: {PROCESSED_DIR}).",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=REPORT_PATH,
        help=f"Where to write the statistics JSON (default: {REPORT_PATH}).",
    )
    return p


def _resolve_data_path(cli_arg: Path | None) -> Path:
    if cli_arg is not None:
        return cli_arg if cli_arg.is_absolute() else PROJECT_ROOT / cli_arg
    settings = get_settings()
    cfg = settings.SAMANANTAR_DATA_PATH
    if cfg:
        path = Path(cfg)
        return path if path.is_absolute() else PROJECT_ROOT / path
    raise FileNotFoundError(
        "No Samanantar data path configured.\n"
        "  - Either set SAMANANTAR_DATA_PATH in your .env file, or\n"
        "  - pass --data-path <extracted-tarball-dir> on the command line.\n"
        "Download the AI4Bharat Samanantar en-te tarball manually and extract it, then "
        "see docs/dataset.md for layout expectations. Scripts NEVER auto-download this dataset."
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.src_lang == args.tgt_lang:
        print("ERROR: --src-lang and --tgt-lang must differ.", file=sys.stderr)
        return 2
    if args.sample_size < 0:
        print("ERROR: --sample-size cannot be negative.", file=sys.stderr)
        return 2

    # 1. Resolve data path (fail fast if missing — before any real work)
    try:
        data_path = _resolve_data_path(args.data_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # 2. Seed: CLI → Settings.DATASET_SEED → 42 fallback
    seed: int
    if args.seed is not None:
        seed = args.seed
    else:
        try:
            seed = int(get_settings().DATASET_SEED)
        except (AttributeError, TypeError, ValueError):
            seed = 42

    # 3. Load records (also verifies file layout)
    try:
        loader = SamanantarLoader(
            data_path=data_path,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    raw_records = loader.load()
    loading_report = loader.loading_report
    raw_count = loading_report.input_count
    print(
        f"[prepare_samanantar] Parsed {len(raw_records)} records from "
        f"{raw_count} input rows ({args.src_lang}->{args.tgt_lang}); "
        f"malformed={loading_report.malformed_count}, "
        f"other_language_pairs={loading_report.language_filtered_count}"
    )

    # 4. Clean records
    cleaned, report = clean_records(
        raw_records,
        min_chars=args.min_chars,
        max_chars_source=args.max_chars,
        max_chars_target=args.max_chars,
    )
    print(
        "[prepare_samanantar] Cleaning complete: "
        f"input={report.input_count} "
        f"nulls_removed={report.null_removed_count} "
        f"duplicates_removed={report.duplicate_removed_count} "
        f"length_filtered={report.length_filtered_count} "
        f"malformed_flagged={report.malformed_flagged_count} "
        f"output={report.output_count}"
    )

    # 5. Split
    splits = split_dataset(cleaned, train=args.train, val=args.val, test=args.test, seed=seed)
    print(
        "[prepare_samanantar] Split complete (seed=%d): train=%d val=%d test=%d"
        % (seed, len(splits.train), len(splits.val), len(splits.test))
    )

    # 6. Write JSONL splits
    counts = write_splits_jsonl(
        splits, args.processed_dir, language_pair=(args.src_lang, args.tgt_lang)
    )
    print(f"[prepare_samanantar] Wrote splits to {args.processed_dir}")

    # 7. Optional: sample
    if args.sample_size > 0:
        from scripts.create_sample_dataset import main as sample_main  # type: ignore  # scripts has no __init__
        sample_argv: list[str] = [
            "--size",
            str(args.sample_size),
            "--source",
            str(
                args.processed_dir
                / f"train.{args.src_lang}-{args.tgt_lang}.jsonl"
            ),
            "--output-dir",
            str(PROJECT_ROOT / "data" / "samples"),
        ]
        rc = sample_main(sample_argv)
        if rc != 0:
            return rc

    # 8. Write report JSON
    args.report.parent.mkdir(parents=True, exist_ok=True)
    generation_ts_utc = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    language_pair = f"{args.src_lang}-{args.tgt_lang}"
    prepared_files = {
        split_name: f"{split_name}.{language_pair}.jsonl"
        for split_name in ("train", "val", "test")
    }
    preprocessing_run_id = (
        f"samanantar-{language_pair}-{generation_ts_utc}-seed-{seed}"
    )
    report_obj = {
        "raw_count": int(raw_count),
        "input_count": int(raw_count),
        "valid_count": int(len(raw_records)),
        "malformed_record_count": int(loading_report.malformed_count),
        "language_filtered_count": int(loading_report.language_filtered_count),
        "null_removed_count": int(report.null_removed_count),
        "duplicate_removed_count": int(report.duplicate_removed_count),
        "length_filtered_count": int(report.length_filtered_count),
        "malformed_flagged_count": int(report.malformed_flagged_count),
        "final_count": int(report.output_count),
        "removed_count": int(raw_count - report.output_count),
        "train_count": int(counts.get("train", len(splits.train))),
        "validation_count": int(counts.get("val", len(splits.val))),
        "test_count": int(counts.get("test", len(splits.test))),
        "random_seed": int(seed),
        "generation_ts_utc": generation_ts_utc,
        "source_dataset": "AI4Bharat Samanantar",
        "source_data_path": str(data_path),
        "provenance": {
            "upstream": {
                "dataset_name": "AI4Bharat Samanantar",
                "source_url": SAMANANTAR_SOURCE_URL,
                "revision": SAMANANTAR_UPSTREAM_REVISION,
                "language_pair": language_pair,
                "source_language": args.src_lang,
                "target_language": args.tgt_lang,
            },
            "preprocessing": {
                "run_id": preprocessing_run_id,
                "seed": int(seed),
                "prepared_files": prepared_files,
            },
        },
        "cleaning": {
            "min_chars": int(args.min_chars),
            "max_chars": int(args.max_chars),
            "malformed_length_ratio_threshold": 0.95,
        },
        "split": {
            "train_ratio": float(args.train),
            "val_ratio": float(args.val),
            "test_ratio": float(args.test),
        },
        "language_pair": f"{args.src_lang}-{args.tgt_lang}",
        "src_lang": args.src_lang,
        "tgt_lang": args.tgt_lang,
    }
    with args.report.open("w", encoding="utf-8") as fh:
        json.dump(report_obj, fh, sort_keys=True, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"[prepare_samanantar] Wrote statistics report to {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
