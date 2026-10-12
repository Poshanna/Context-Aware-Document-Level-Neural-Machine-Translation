"""Validate the processed Samanantar dataset splits.

Checks (see spec FR-7.2):
1. Required fields present on every record.
2. No empty source_text / target_text.
3. No duplicate source-target pairs across splits (leakage check).
4. source_language ∈ {en, te, hi}, target_language ∈ {en, te, hi}.
5. Per-split source/target length statistics (min / max / mean chars).
6. Cross-split pair overlap: should be 0.

Writes a structured JSON report to ``experiments/dataset_validation_report.json``
and prints a human-readable summary to stdout.

Exit codes:
    0 — all checks OK
    1 — severe issues (missing required fields, cross-split leakage)
    2 — warnings only (non-severe issues like unusual length stats)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.preprocessing.dataset_loader import ParallelRecord  # noqa: E402


DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_REPORT = PROJECT_ROOT / "experiments" / "dataset_validation_report.json"

_VALID_LANGS = {"en", "te"}
_REQUIRED_FIELDS = {
    "id",
    "source_text",
    "target_text",
    "source_language",
    "target_language",
    "data_source",
}


@dataclass
class Issue:
    severity: str  # "error" | "warning"
    code: str
    message: str
    split: str | None = None
    record_id: int | None = None


@dataclass
class ValidationResult:
    ok: bool = True
    issues: list[Issue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    has_severe: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _len_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0, "count": 0}
    return {
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": round(float(statistics.fmean(values)), 3),
        "median": int(statistics.median_low(values)),
        "count": int(len(values)),
    }


def _pairs(records: list[ParallelRecord]) -> set[tuple[str, str]]:
    return {(r.source_text, r.target_text) for r in records}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_cross_split_overlap(
    splits: dict[str, list[ParallelRecord]], result: ValidationResult
) -> None:
    names = list(splits.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            common = _pairs(splits[a_name]) & _pairs(splits[b_name])
            if common:
                result.ok = False
                result.has_severe = True
                result.issues.append(
                    Issue(
                        severity="error",
                        code="cross_split_leak",
                        message=(
                            f"{len(common)} pair(s) found in BOTH {a_name} and {b_name}. "
                            f"Example: {next(iter(common))!r}"
                        ),
                        split=f"{a_name}<->{b_name}",
                    )
                )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_validation(
    processed_dir: Path | str,
    src_lang: str = "en",
    tgt_lang: str = "te",
    min_chars: int = 2,
    max_chars: int = 1000,
) -> ValidationResult:
    processed_dir = Path(processed_dir)
    if (src_lang, tgt_lang) not in {("en", "te"), ("te", "en")}:
        raise ValueError("Validation supports only the en-te or te-en language pair.")
    split_files = {
        "train": processed_dir / f"train.{src_lang}-{tgt_lang}.jsonl",
        "val": processed_dir / f"val.{src_lang}-{tgt_lang}.jsonl",
        "test": processed_dir / f"test.{src_lang}-{tgt_lang}.jsonl",
    }

    result = ValidationResult()
    summary: dict[str, Any] = {}
    splits_records: dict[str, list[ParallelRecord]] = {}

    for split_name, path in split_files.items():
        if not path.exists():
            result.ok = False
            result.has_severe = True
            result.issues.append(
                Issue(
                    severity="error",
                    code="missing_split_file",
                    message=f"Split file not found: {path}",
                    split=split_name,
                )
            )
            continue
        source_lengths: list[int] = []
        target_lengths: list[int] = []
        seen_pairs: set[tuple[str, str]] = set()
        pair_records: list[ParallelRecord] = []
        parsed_count = 0

        def add_issue(
            code: str, message: str, record_id: int | None, severity: str = "error"
        ) -> None:
            result.issues.append(
                Issue(severity, code, message, split_name, record_id)
            )
            if severity == "error":
                result.ok = False
                result.has_severe = True

        try:
            with path.open("r", encoding="utf-8", errors="strict") as fh:
                for line_number, line in enumerate(fh, start=1):
                    if not line.strip():
                        continue
                    try:
                        raw_record = json.loads(line)
                    except json.JSONDecodeError:
                        add_issue(
                            "malformed_record",
                            f"Invalid JSON on line {line_number}.",
                            None,
                        )
                        continue
                    if not isinstance(raw_record, dict):
                        add_issue(
                            "malformed_record",
                            f"JSON on line {line_number} is not an object.",
                            None,
                        )
                        continue

                    parsed_count += 1
                    missing = _REQUIRED_FIELDS - set(raw_record)
                    raw_id = raw_record.get("id")
                    record_id = (
                        raw_id
                        if isinstance(raw_id, int) and not isinstance(raw_id, bool)
                        else None
                    )
                    if missing:
                        add_issue(
                            "missing_required_fields",
                            f"Record on line {line_number} missing fields: {sorted(missing)}",
                            record_id,
                        )
                    if record_id is None or record_id < 0:
                        add_issue("invalid_id", f"Invalid id on line {line_number}.", None)

                    source_text = raw_record.get("source_text")
                    target_text = raw_record.get("target_text")
                    text_values = (
                        ("source_text", source_text),
                        ("target_text", target_text),
                    )
                    for field_name, text in text_values:
                        if not isinstance(text, str):
                            add_issue(
                                "malformed_record",
                                f"{field_name} must be a string (line {line_number}).",
                                record_id,
                            )
                            continue
                        if not text.strip():
                            add_issue(
                                f"empty_{field_name}",
                                f"{field_name} is empty (id={record_id}).",
                                record_id,
                            )
                            continue
                        if len(text) < min_chars or len(text) > max_chars:
                            add_issue(
                                "length_constraint",
                                f"{field_name} length {len(text)} is outside "
                                f"[{min_chars}, {max_chars}] (id={record_id}).",
                                record_id,
                            )
                        if "\ufffd" in text or any(
                            0xD800 <= ord(character) <= 0xDFFF for character in text
                        ):
                            add_issue(
                                "invalid_unicode",
                                f"{field_name} contains invalid Unicode (id={record_id}).",
                                record_id,
                            )

                    source_language = raw_record.get("source_language")
                    target_language = raw_record.get("target_language")
                    if not isinstance(source_language, str) or source_language not in _VALID_LANGS:
                        add_issue(
                            "invalid_language_label",
                            f"source_language={source_language!r} is unsupported (id={record_id}).",
                            record_id,
                        )
                    if not isinstance(target_language, str) or target_language not in _VALID_LANGS:
                        add_issue(
                            "invalid_language_label",
                            f"target_language={target_language!r} is unsupported (id={record_id}).",
                            record_id,
                        )
                    if (source_language, target_language) != (src_lang, tgt_lang):
                        add_issue(
                            "language_pair_mismatch",
                            f"Expected {src_lang}->{tgt_lang}, got "
                            f"{source_language!r}->{target_language!r} (id={record_id}).",
                            record_id,
                        )
                    data_source = raw_record.get("data_source")
                    if not isinstance(data_source, str) or not data_source.strip():
                        add_issue(
                            "invalid_data_source",
                            f"data_source must be a non-empty string (id={record_id}).",
                            record_id,
                        )

                    if not isinstance(source_text, str) or not isinstance(target_text, str):
                        continue
                    source_lengths.append(len(source_text))
                    target_lengths.append(len(target_text))
                    if source_text.strip() and target_text.strip():
                        pair = (source_text, target_text)
                        if pair in seen_pairs:
                            add_issue(
                                "duplicate_pair",
                                f"Duplicate parallel pair in {split_name} (id={record_id}).",
                                record_id,
                            )
                        seen_pairs.add(pair)
                        record = ParallelRecord(
                            id=record_id if record_id is not None else line_number,
                            source_text=source_text,
                            target_text=target_text,
                            source_language=str(source_language),
                            target_language=str(target_language),
                        )
                        pair_records.append(record)
                        if tgt_lang == "te" and not any(
                            0x0C00 <= ord(character) <= 0x0C7F
                            for character in target_text
                        ):
                            add_issue(
                                "warning",
                                f"target_text has no Telugu-script character (id={record_id}).",
                                record_id,
                                severity="warning",
                            )
        except UnicodeDecodeError:
            add_issue(
                "malformed_record",
                f"{path.name} is not valid UTF-8.",
                None,
            )

        splits_records[split_name] = pair_records
        summary[split_name] = {
            "count": parsed_count,
            "source_length": _len_stats(source_lengths),
            "target_length": _len_stats(target_lengths),
        }

    if splits_records:
        _check_cross_split_overlap(splits_records, result)

    result.summary = summary
    return result


def _issue_to_dict(i: Issue) -> dict:
    return {
        "severity": i.severity,
        "code": i.code,
        "message": i.message,
        "split": i.split,
        "record_id": i.record_id,
    }


def _write_report(
    result: ValidationResult, report_path: Path | str
) -> None:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "ok": bool(result.ok and not result.has_severe),
        "has_severe_issues": bool(result.has_severe),
        "total_issues": len(result.issues),
        "error_count": sum(1 for i in result.issues if i.severity == "error"),
        "warning_count": sum(1 for i in result.issues if i.severity == "warning"),
        "issues": [_issue_to_dict(i) for i in result.issues],
        "summary": result.summary,
        "generated_ts_utc": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, sort_keys=True, ensure_ascii=False, indent=2)
        fh.write("\n")


def _print_human(result: ValidationResult) -> None:
    print("Dataset Validation Report")
    print("=" * 60)
    status = "OK" if (result.ok and not result.has_severe) else "FAIL"
    print(f"Overall: {status}")
    print(
        f"issues={len(result.issues)}  "
        f"errors={sum(1 for i in result.issues if i.severity == 'error')}  "
        f"warnings={sum(1 for i in result.issues if i.severity == 'warning')}"
    )
    print()
    for split, stats in result.summary.items():
        print(
            f"  {split}: count={stats['count']}  "
            f"src[{stats['source_length']['min']}–{stats['source_length']['max']}]chars  "
            f"tgt[{stats['target_length']['min']}–{stats['target_length']['max']}]chars"
        )
    if result.issues:
        print()
        print("Issues (top 10):")
        for i in result.issues[:10]:
            split = f"[{i.split}] " if i.split else ""
            print(f"  {i.severity.upper():7s} {i.code:30s} {split}{i.message}")
        if len(result.issues) > 10:
            print(f"  ... (+{len(result.issues) - 10} more issues in report JSON)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=f"Directory containing train/val/test JSONL splits (default: {DEFAULT_PROCESSED_DIR}).",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Write validation JSON here (default: {DEFAULT_REPORT}).",
    )
    p.add_argument("--src-lang", choices=["en", "te"], default="en")
    p.add_argument("--tgt-lang", choices=["en", "te"], default="te")
    p.add_argument("--min-chars", type=int, default=2)
    p.add_argument("--max-chars", type=int, default=1000)
    args = p.parse_args(argv)

    if args.src_lang == args.tgt_lang:
        p.error("--src-lang and --tgt-lang must differ")
    result = run_validation(
        args.processed_dir,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    _write_report(result, args.report)
    _print_human(result)

    if result.has_severe or not result.ok:
        return 1
    if any(i.severity == "warning" for i in result.issues):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
