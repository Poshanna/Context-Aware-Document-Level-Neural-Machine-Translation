#!/usr/bin/env python3
"""Bounded structural validation for the pinned IndicDISCO-MT paragraph split.

The source must be supplied explicitly. This script never downloads an entire
CSV, never writes source text, and does not run a model or a metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.evaluation.indicdisco_adapter import (  # noqa: E402
    INDICDISCO_DATASET,
    INDICDISCO_PARAGRAPH_COMPONENT,
    INDICDISCO_REPOSITORY,
    INDICDISCO_REVISION,
    INDICDISCO_SOURCE_LANGUAGE,
    INDICDISCO_TARGET_LANGUAGE,
    IndicDiscoMetric,
    IndicDiscoMTTrustedAdapter,
    metric_eligibility,
)

MAX_RANGE_BYTES = 65536
RANGE_CHUNK_BYTES = 1024
DEFAULT_MAX_ROWS = 3
DEFAULT_MAX_BYTES = 16384
DEFAULT_REPORT_DIR = PROJECT_ROOT / "experiments" / "evaluation" / "indicdisco_structure"
EXPECTED_FIELDS = ("source", "target")


@dataclass(frozen=True)
class SourceInput:
    location: str
    kind: str
    pinned_official: bool
    local_path: Path | None = None
    official_file_oid: str | None = None
    official_file_size: int | None = None
    verification_method: str | None = None


@dataclass(frozen=True)
class ParsedCsvPrefix:
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    trailing_partial_record: bool
    parse_warning: str | None


def official_source_url(split: str) -> str:
    return (
        "https://huggingface.co/datasets/"
        f"{INDICDISCO_REPOSITORY}/resolve/{INDICDISCO_REVISION}/"
        f"{INDICDISCO_PARAGRAPH_COMPONENT}/Tel/"
        f"tel_indicdisco-mt_{split}.csv"
    )


def classify_source(
    source: str,
    *,
    revision: str,
    component: str,
    split: str,
) -> SourceInput:
    _validate_run_metadata(revision=revision, component=component, split=split)
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in {"http", "https"}:
        expected_url = official_source_url(split)
        if source != expected_url:
            raise ValueError("remote source must be the exact pinned official IndicDISCO-MT URL.")
        if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
            raise ValueError("remote source must use the official HTTPS Hugging Face host.")
        return SourceInput(
            source,
            "pinned_official_url",
            True,
            verification_method="revision_pinned_official_url",
        )

    local_path = Path(source).expanduser().resolve()
    expected_name = f"tel_indicdisco-mt_{split}.csv"
    if local_path.name != expected_name or not local_path.is_file():
        raise ValueError(f"local source must be an existing {expected_name} file.")
    return SourceInput(str(local_path), "explicit_local_unverified", False, local_path)


def fetch_official_file_metadata(split: str) -> tuple[str, int]:
    """Fetch the pinned Hub tree's Git blob OID and size for one CSV file."""

    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test.")
    expected_path = (
        f"{INDICDISCO_PARAGRAPH_COMPONENT}/Tel/"
        f"tel_indicdisco-mt_{split}.csv"
    )
    url = (
        "https://huggingface.co/api/datasets/"
        f"{INDICDISCO_REPOSITORY}/tree/{INDICDISCO_REVISION}/"
        f"{INDICDISCO_PARAGRAPH_COMPONENT}/Tel?recursive=false&expand=true"
    )
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=45) as response:
        metadata = json.load(response)
    if not isinstance(metadata, list):
        raise ValueError("Pinned Hugging Face tree metadata has an unexpected shape.")
    matches = [entry for entry in metadata if isinstance(entry, dict) and entry.get("path") == expected_path]
    if len(matches) != 1:
        raise ValueError("Pinned Hugging Face tree metadata did not identify exactly one source file.")
    oid = matches[0].get("oid")
    size = matches[0].get("size")
    if not isinstance(oid, str) or re.fullmatch(r"[0-9a-f]{40}", oid) is None:
        raise ValueError("Pinned Hugging Face tree metadata has no valid Git blob OID.")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("Pinned Hugging Face tree metadata has no valid file size.")
    return oid, size


def git_blob_oid(path: Path) -> str:
    """Return the Git blob object ID for a local file; this is not a new upstream checksum."""

    size = path.stat().st_size
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as input_file:
        while chunk := input_file.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def verify_local_source(
    source: SourceInput,
    *,
    split: str,
    metadata_fetcher: Callable[[str], tuple[str, int]] = fetch_official_file_metadata,
) -> SourceInput:
    """Verify explicit local bytes against the official pinned tree OID and size."""

    if source.local_path is None:
        return source
    official_oid, official_size = metadata_fetcher(split)
    local_size = source.local_path.stat().st_size
    if local_size != official_size:
        raise ValueError(
            f"local source size {local_size} does not match pinned official file size {official_size}."
        )
    local_oid = git_blob_oid(source.local_path)
    if local_oid != official_oid:
        raise ValueError("local source Git blob OID does not match the pinned official file OID.")
    return replace(
        source,
        kind="local_verified_against_pinned_hub_git_blob_oid",
        pinned_official=True,
        official_file_oid=official_oid,
        official_file_size=official_size,
        verification_method="pinned_huggingface_git_blob_oid_and_size",
    )


def parse_csv_prefix(
    payload: bytes,
    *,
    max_rows: int,
    at_eof: bool,
) -> ParsedCsvPrefix:
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer.")

    trailing_partial = False
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        if exc.end != len(payload):
            raise ValueError("CSV prefix contains invalid UTF-8 before its end.") from exc
        payload = payload[:exc.start]
        text = payload.decode("utf-8-sig")
        trailing_partial = True

    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = tuple(next(reader))
    except StopIteration:
        return ParsedCsvPrefix((), (), trailing_partial, None)
    except csv.Error as exc:
        return ParsedCsvPrefix((), (), True, f"csv_header_parse_error: {exc}")

    rows: list[tuple[str, ...]] = []
    warning: str | None = None
    while len(rows) < max_rows:
        try:
            rows.append(tuple(next(reader)))
        except StopIteration:
            break
        except csv.Error as exc:
            trailing_partial = True
            warning = f"incomplete_trailing_csv_record: {exc}"
            break

    if (
        not at_eof
        and warning is None
        and payload
        and payload[-1] not in b"\r\n"
        and len(rows) < max_rows
    ):
        if rows:
            rows.pop()
        trailing_partial = True
        warning = warning or "incomplete_trailing_csv_record_at_byte_limit"

    return ParsedCsvPrefix(header, tuple(rows), trailing_partial, warning)


def read_bounded_csv_prefix(
    source: SourceInput,
    *,
    max_rows: int,
    max_bytes: int,
) -> tuple[ParsedCsvPrefix, int]:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 1 <= max_bytes <= MAX_RANGE_BYTES:
        raise ValueError(f"max_bytes must be between 1 and {MAX_RANGE_BYTES}.")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= 100:
        raise ValueError("max_rows must be between 1 and 100.")

    payload = bytearray()
    total_size: int | None = None
    at_eof = False
    if source.local_path is not None:
        total_size = source.local_path.stat().st_size
        with source.local_path.open("rb") as input_file:
            while len(payload) < max_bytes:
                chunk = input_file.read(min(RANGE_CHUNK_BYTES, max_bytes - len(payload)))
                if not chunk:
                    at_eof = True
                    break
                payload.extend(chunk)
                at_eof = len(payload) >= total_size
                parsed = parse_csv_prefix(bytes(payload), max_rows=max_rows, at_eof=at_eof)
                if len(parsed.rows) >= max_rows or at_eof:
                    return parsed, len(payload)
    else:
        offset = 0
        while offset < max_bytes:
            requested = min(RANGE_CHUNK_BYTES, max_bytes - offset)
            request = urllib.request.Request(
                source.location,
                headers={"Range": f"bytes={offset}-{offset + requested - 1}"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206:
                    raise ValueError("official source did not honor the bounded byte-range request.")
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if match is None or int(match.group(1)) != offset:
                    raise ValueError("official source returned an invalid Content-Range.")
                total_size = int(match.group(3))
                chunk = response.read()
            if len(chunk) > requested:
                raise ValueError("official source returned more bytes than requested.")
            if not chunk:
                at_eof = True
                break
            payload.extend(chunk)
            offset += len(chunk)
            at_eof = offset >= total_size
            parsed = parse_csv_prefix(bytes(payload), max_rows=max_rows, at_eof=at_eof)
            if len(parsed.rows) >= max_rows or at_eof:
                return parsed, len(payload)

    return parse_csv_prefix(bytes(payload), max_rows=max_rows, at_eof=at_eof), len(payload)


def inspect_rows(
    parsed: ParsedCsvPrefix,
    *,
    source: SourceInput,
    revision: str,
    split: str,
    component: str,
    source_language: str,
    target_language: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    headers_match = parsed.header == EXPECTED_FIELDS
    adapter = IndicDiscoMTTrustedAdapter()

    for row_index, values in enumerate(parsed.rows):
        row_locator = f"{source.location}#row={row_index}"
        row_summary: dict[str, Any] = {
            "row_locator": row_locator,
            "source_language": source_language,
            "target_language": target_language,
            "source_paragraph_field_present": len(values) >= 1,
            "source_paragraph_nonempty": len(values) >= 1 and bool(values[0].strip()),
            "source_paragraph_present": len(values) >= 1 and bool(values[0].strip()),
            "target_paragraph_field_present": len(values) >= 2,
            "target_paragraph_nonempty": len(values) >= 2 and bool(values[1].strip()),
            "target_paragraph_present": len(values) >= 2 and bool(values[1].strip()),
            "same_document_causal_context_available": False,
            "previous_paragraph_context_available": False,
            "provenance_status": "verified_dataset" if source.pinned_official else "unverified",
        }
        reason: str | None = None
        if not source.pinned_official:
            reason = "local_source_not_verified_against_pinned_commit_in_bounded_mode"
        elif not headers_match:
            reason = "unexpected_core_component_schema"
        elif len(values) != len(parsed.header):
            reason = "row_column_count_does_not_match_header"
        elif not values[0].strip():
            reason = "empty_source_paragraph"
        elif not values[1].strip():
            reason = "empty_target_paragraph"

        if reason is None:
            source_location = f"{source.location}#row={row_index}:source"
            reference_location = f"{source.location}#row={row_index}:target"
            alignment_location = f"{source.location}#row={row_index}:source,target"
            try:
                record = adapter.adapt_row(
                    row=dict(zip(parsed.header, values)),
                    revision=revision,
                    split=split,
                    component=component,
                    row_locator=row_locator,
                    source_location=source_location,
                    reference_location=reference_location,
                    alignment_location=alignment_location,
                    source_language=source_language,
                    target_language=target_language,
                )
            except (TypeError, ValueError) as exc:
                reason = str(exc)
            else:
                metric_checks = {}
                for metric in IndicDiscoMetric:
                    decision = metric_eligibility(record, metric)
                    metric_checks[metric.value] = {
                        "eligible": decision.eligible,
                        "reason": decision.reason,
                    }
                row_summary.update(
                    {
                        "source_paragraph_chars": len(record.source_text),
                        "target_paragraph_chars": len(record.reference_text),
                        "unit_type": record.unit_type.value,
                        "alignment": record.alignment.value,
                        "document_id": record.document_id,
                        "ordering": record.ordering,
                        "cross_row_context": record.cross_row_context.value,
                        "paragraph_pair_represented_directly_by_row": True,
                        "provenance": {
                            "dataset": record.provenance.dataset_name,
                            "revision": record.provenance.dataset_revision,
                            "split": record.provenance.split,
                            "component": component,
                            "record_locator": record.provenance.record_id,
                            "source_location": record.provenance.source_location,
                            "reference_location": record.provenance.reference_location,
                            "alignment_location": record.provenance.alignment_location,
                            "status": record.provenance_status.value,
                            "unit_id": record.provenance.unit_id,
                        },
                        "metric_eligibility": metric_checks,
                    }
                )
                accepted.append(row_summary)
        if reason is not None:
            row_summary["paragraph_pair_represented_directly_by_row"] = (
                headers_match
                and len(values) == len(parsed.header)
                and bool(values[0].strip())
                and bool(values[1].strip())
            )
            row_summary["exclusion_reason"] = reason
            excluded.append(row_summary)

    return accepted, excluded


def structural_capabilities(header: tuple[str, ...], accepted_count: int) -> dict[str, Any]:
    normalized = {re.sub(r"[^a-z0-9]", "", field.lower()) for field in header}
    sentence_ids = {"sentenceid", "sentid", "sentenceidentifier"}
    document_ids = {"documentid", "docid", "articleid"}
    parent_document_ids = {"parentdocumentid", "parentdocid", "parentarticleid"}
    ordering_fields = {
        "sentenceindex",
        "sentenceorder",
        "paragraphindex",
        "paragraphorder",
        "unitorder",
        "sequence",
        "position",
        "order",
    }
    sentence_alignment_fields = {
        "sentencealignment",
        "sentencealignments",
        "source sentence id".replace(" ", ""),
        "target sentence id".replace(" ", ""),
        "sentencepairid",
    }
    return {
        "explicit_sentence_ids_exist": bool(normalized & sentence_ids),
        "explicit_document_ids_exist": bool(normalized & document_ids),
        "parent_document_ids_exist": bool(normalized & parent_document_ids),
        "explicit_ordering_fields_exist": bool(normalized & ordering_fields),
        "sentence_alignment_metadata_exists": bool(normalized & sentence_alignment_fields),
        "paragraph_alignment_represented_directly_by_row": (
            set(header) == set(EXPECTED_FIELDS) and accepted_count > 0
        ),
        "document_identity_available": False,
        "ordering_available": False,
        "sentence_alignment_available": False,
        "context_available": False,
        "same_document_causal_context_available": False,
        "previous_paragraph_context_available": False,
        "cross_row_context_available": False,
        "within_paragraph_sentence_context_available": False,
    }


def build_report(
    *,
    source: SourceInput,
    revision: str,
    component: str,
    split: str,
    source_language: str,
    target_language: str,
    parsed: ParsedCsvPrefix,
    bytes_read: int,
) -> dict[str, Any]:
    accepted, excluded = inspect_rows(
        parsed,
        source=source,
        revision=revision,
        split=split,
        component=component,
        source_language=source_language,
        target_language=target_language,
    )
    capability = structural_capabilities(parsed.header, len(accepted))

    def aggregate_metric(metric: IndicDiscoMetric) -> dict[str, Any]:
        decisions = [row["metric_eligibility"][metric.value] for row in accepted]
        eligible = bool(decisions) and all(item["eligible"] for item in decisions)
        reasons = sorted({item["reason"] for item in decisions if item["reason"]})
        return {
            "eligible": eligible,
            "computed": False,
            "eligible_rows": sum(item["eligible"] for item in decisions),
            "reason": "; ".join(reasons) if reasons else "no accepted paragraph pairs",
        }

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    reasons: dict[str, int] = {}
    for item in excluded:
        reason = item["exclusion_reason"]
        reasons[reason] = reasons.get(reason, 0) + 1

    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": INDICDISCO_DATASET,
        "repository": INDICDISCO_REPOSITORY,
        "pinned_revision": revision,
        "component": component,
        "split": split,
        "source": source.location,
        "source_kind": source.kind,
        "source_provenance_verified": source.pinned_official,
        "source_verification": {
            "method": source.verification_method,
            "official_git_blob_oid": source.official_file_oid,
            "official_file_size_bytes": source.official_file_size,
        },
        "source_language": source_language,
        "target_language": target_language,
        "language_labels_embedded_in_csv": False,
        "max_rows_requested": None,
        "bytes_read": bytes_read,
        "observed_schema": list(parsed.header),
        "inspected_row_count": len(parsed.rows),
        "accepted_row_count": len(accepted),
        "excluded_row_count": len(excluded),
        "exclusion_reason_counts": reasons,
        "accepted_rows": accepted,
        "excluded_rows": excluded,
        "parse_status": {
            "trailing_partial_record": parsed.trailing_partial_record,
            "warning": parsed.parse_warning,
        },
        "structural_capabilities": capability,
        "metric_eligibility": {
            "paragraph_bleu": aggregate_metric(IndicDiscoMetric.PARAGRAPH_BLEU),
            "paragraph_chrf++": aggregate_metric(IndicDiscoMetric.PARAGRAPH_CHRF),
            "sentence_bleu": aggregate_metric(IndicDiscoMetric.SENTENCE_BLEU),
            "sentence_chrf++": aggregate_metric(IndicDiscoMetric.SENTENCE_CHRF),
            "document_context_bleu": aggregate_metric(IndicDiscoMetric.DOCUMENT_CONTEXT_BLEU),
            "document_context_chrf++": aggregate_metric(IndicDiscoMetric.DOCUMENT_CONTEXT_CHRF),
        },
        "provenance": {
            "status": "verified_dataset" if source.pinned_official else "unverified",
            "row_locator_semantics": "record provenance only; never a document ID",
            "unit_id": None,
        },
        "rights_status": "unresolved",
        "no_inference": True,
        "no_metrics_computed": True,
    }


def run_validation(args: argparse.Namespace) -> Path:
    _validate_run_metadata(
        revision=args.revision,
        component=args.component,
        split=args.split,
    )
    if (args.source_language, args.target_language) != (
        INDICDISCO_SOURCE_LANGUAGE,
        INDICDISCO_TARGET_LANGUAGE,
    ):
        raise ValueError("source/target language must be Telugu -> English as released.")
    if isinstance(args.max_rows, bool) or not 1 <= args.max_rows <= 100:
        raise ValueError("max_rows must be between 1 and 100.")
    if isinstance(args.max_bytes, bool) or not 1 <= args.max_bytes <= MAX_RANGE_BYTES:
        raise ValueError(f"max_bytes must be between 1 and {MAX_RANGE_BYTES}.")

    source = classify_source(
        args.source,
        revision=args.revision,
        component=args.component,
        split=args.split,
    )
    source = verify_local_source(source, split=args.split)
    parsed, bytes_read = read_bounded_csv_prefix(
        source,
        max_rows=args.max_rows,
        max_bytes=args.max_bytes,
    )
    report = build_report(
        source=source,
        revision=args.revision,
        component=args.component,
        split=args.split,
        source_language=args.source_language,
        target_language=args.target_language,
        parsed=parsed,
        bytes_read=bytes_read,
    )
    report["max_rows_requested"] = args.max_rows
    report_dir = Path(args.output_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{report['run_id']}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report_path


def _validate_run_metadata(*, revision: str, component: str, split: str) -> None:
    if revision != INDICDISCO_REVISION:
        raise ValueError("revision must be the pinned IndicDISCO-MT commit.")
    if component != INDICDISCO_PARAGRAPH_COMPONENT:
        raise ValueError("component must be Indicdisco_MT.")
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bounded structural validation only for a pinned IndicDISCO-MT CSV split."
    )
    parser.add_argument("--source", required=True, help="Exact pinned HF URL or explicit local test CSV path.")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--component", required=True)
    parser.add_argument("--split", required=True, choices=("train", "dev", "test"))
    parser.add_argument("--source-language", required=True)
    parser.add_argument("--target-language", required=True)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report_path = run_validation(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
