"""Inspect a bounded metadata-only sample through the trusted Samanantar adapter."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.evaluation.samanantar_adapter import (  # noqa: E402
    SamanantarPreparedAdapter,
    SamanantarSplit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read a bounded prefix of an existing prepared Samanantar split and "
            "print metadata only. No datasets are downloaded."
        )
    )
    parser.add_argument("--split", choices=[item.value for item in SamanantarSplit], default="test")
    parser.add_argument("--max-samples", type=int, default=5)
    return parser


def _record_summary(record: object) -> dict[str, object]:
    provenance = record.provenance
    return {
        "record_id": provenance.record_id,
        "record_location": provenance.source_location,
        "source_present": bool(record.current_text.strip()),
        "reference_present": bool(
            record.reference_translation and record.reference_translation.strip()
        ),
        "track": record.track.value,
        "unit": record.current_text_unit.value,
        "reference_availability": record.reference_availability.value,
        "alignment_status": record.alignment_status.value,
        "context_availability": record.context_status.value,
        "context_supplied_to_model": record.context_supplied_to_model,
        "document_or_passage_id": record.document_or_passage_id,
        "provenance_status": provenance.status.value,
    }


def build_inspection_report(*, split: str, max_samples: int) -> dict[str, object]:
    sample = SamanantarPreparedAdapter().load_sample(split=split, max_samples=max_samples)
    exclusion_counts = Counter(item.reason for item in sample.exclusions)
    summary_limit = min(max_samples, 10)
    summaries = [_record_summary(record) for record in sample.records[:summary_limit]]
    return {
        "dataset": sample.dataset_name,
        "dataset_revision": sample.dataset_revision,
        "revision_scope": sample.revision_scope,
        "upstream_revision": sample.upstream_revision,
        "upstream_source_url": sample.upstream_source_url,
        "preprocessing_run_id": sample.preprocessing_run_id,
        "preprocessing_seed": sample.preprocessing_seed,
        "prepared_files": dict(sample.prepared_files),
        "split": sample.split.value,
        "source_file": sample.source_file,
        "max_samples": max_samples,
        "expected_split_count": sample.expected_split_count,
        "rows_inspected": sample.rows_inspected,
        "accepted_count": sample.accepted_count,
        "excluded_count": sample.excluded_count,
        "exclusion_reasons": dict(sorted(exclusion_counts.items())),
        "summary_records": summaries,
        "summary_records_omitted": max(0, sample.accepted_count - len(summaries)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_samples < 1:
        parser.error("--max-samples must be at least 1.")
    report = build_inspection_report(split=args.split, max_samples=args.max_samples)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())