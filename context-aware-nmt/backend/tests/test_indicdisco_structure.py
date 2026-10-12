from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_indicdisco_structure import (
    DEFAULT_MAX_BYTES,
    INDICDISCO_PARAGRAPH_COMPONENT,
    INDICDISCO_REVISION,
    INDICDISCO_SOURCE_LANGUAGE,
    INDICDISCO_TARGET_LANGUAGE,
    build_report,
    classify_source,
    git_blob_oid,
    inspect_rows,
    official_source_url,
    parse_csv_prefix,
    read_bounded_csv_prefix,
    verify_local_source,
)


def _pinned_source(split: str = "test"):
    return classify_source(
        official_source_url(split),
        revision=INDICDISCO_REVISION,
        component=INDICDISCO_PARAGRAPH_COMPONENT,
        split=split,
    )


def _parsed_fixture():
    return parse_csv_prefix(
        (
            b'source,target\n"Telugu paragraph one.","English paragraph one."\n'
            b'"Telugu paragraph two.","English paragraph two."\n'
        ),
        max_rows=3,
        at_eof=True,
    )


def test_official_source_is_revision_and_split_pinned() -> None:
    source = _pinned_source("test")

    assert source.pinned_official is True
    assert source.kind == "pinned_official_url"
    assert INDICDISCO_REVISION in source.location
    assert source.location.endswith("Indicdisco_MT/Tel/tel_indicdisco-mt_test.csv")

    with pytest.raises(ValueError, match="exact pinned official"):
        classify_source(
            "https://example.invalid/tel_indicdisco-mt_test.csv",
            revision=INDICDISCO_REVISION,
            component=INDICDISCO_PARAGRAPH_COMPONENT,
            split="test",
        )


def test_parser_returns_only_complete_rows_within_the_limit() -> None:
    parsed = parse_csv_prefix(
        b"source,target\nsource one,target one\nsource two,partial",
        max_rows=3,
        at_eof=False,
    )

    assert parsed.header == ("source", "target")
    assert parsed.rows == (("source one", "target one"),)
    assert parsed.trailing_partial_record is True


def test_parser_keeps_complete_rows_before_truncated_multiline_record() -> None:
    parsed = parse_csv_prefix(
        b'source,target\n"source line one\nsource line two","complete reference"\n'
        b'"truncated source,unterminated',
        max_rows=3,
        at_eof=False,
    )

    assert parsed.rows == (("source line one\nsource line two", "complete reference"),)
    assert parsed.trailing_partial_record is True
    assert parsed.parse_warning is not None


def test_structural_report_uses_adapter_gates_and_keeps_context_unavailable() -> None:
    source = _pinned_source()
    parsed = _parsed_fixture()
    report = build_report(
        source=source,
        revision=INDICDISCO_REVISION,
        component=INDICDISCO_PARAGRAPH_COMPONENT,
        split="test",
        source_language=INDICDISCO_SOURCE_LANGUAGE,
        target_language=INDICDISCO_TARGET_LANGUAGE,
        parsed=parsed,
        bytes_read=86,
    )

    flags = report["structural_capabilities"]
    assert report["inspected_row_count"] == 2
    assert report["accepted_row_count"] == 2
    assert report["excluded_row_count"] == 0
    assert flags["document_identity_available"] is False
    assert flags["ordering_available"] is False
    assert flags["sentence_alignment_available"] is False
    assert flags["context_available"] is False
    assert flags["same_document_causal_context_available"] is False
    assert flags["previous_paragraph_context_available"] is False
    assert flags["paragraph_alignment_represented_directly_by_row"] is True
    assert report["metric_eligibility"]["paragraph_bleu"]["eligible"] is True
    assert report["metric_eligibility"]["paragraph_chrf++"]["eligible"] is True
    assert report["metric_eligibility"]["sentence_bleu"]["eligible"] is False
    assert report["metric_eligibility"]["sentence_chrf++"]["eligible"] is False
    assert report["metric_eligibility"]["document_context_bleu"]["eligible"] is False
    assert report["rights_status"] == "unresolved"
    assert report["no_inference"] is True
    assert report["no_metrics_computed"] is True
    assert report["accepted_rows"][0]["document_id"] is None
    assert report["accepted_rows"][0]["ordering"] is None
    assert report["accepted_rows"][0]["source_paragraph_field_present"] is True
    assert report["accepted_rows"][0]["source_paragraph_nonempty"] is True
    assert report["accepted_rows"][0]["target_paragraph_field_present"] is True
    assert report["accepted_rows"][0]["target_paragraph_nonempty"] is True
    assert report["accepted_rows"][0]["row_locator"].endswith("#row=0")


def test_empty_paragraph_is_excluded_with_exact_reason() -> None:
    source = _pinned_source()
    parsed = parse_csv_prefix(
        b"source,target\n\"\",\"English paragraph\"\n",
        max_rows=1,
        at_eof=True,
    )

    accepted, excluded = inspect_rows(
        parsed,
        source=source,
        revision=INDICDISCO_REVISION,
        split="test",
        component=INDICDISCO_PARAGRAPH_COMPONENT,
        source_language=INDICDISCO_SOURCE_LANGUAGE,
        target_language=INDICDISCO_TARGET_LANGUAGE,
    )

    assert accepted == []
    assert excluded[0]["row_locator"].endswith("#row=0")
    assert excluded[0]["exclusion_reason"] == "empty_source_paragraph"


def test_local_source_is_bounded_and_fails_closed_as_unverified(tmp_path: Path) -> None:
    local_file = tmp_path / "tel_indicdisco-mt_test.csv"
    local_file.write_bytes(
        b"source,target\nTelugu paragraph,English paragraph\n" + b"x" * 10000
    )
    source = classify_source(
        str(local_file),
        revision=INDICDISCO_REVISION,
        component=INDICDISCO_PARAGRAPH_COMPONENT,
        split="test",
    )

    parsed, bytes_read = read_bounded_csv_prefix(source, max_rows=1, max_bytes=DEFAULT_MAX_BYTES)
    accepted, excluded = inspect_rows(
        parsed,
        source=source,
        revision=INDICDISCO_REVISION,
        split="test",
        component=INDICDISCO_PARAGRAPH_COMPONENT,
        source_language=INDICDISCO_SOURCE_LANGUAGE,
        target_language=INDICDISCO_TARGET_LANGUAGE,
    )

    assert bytes_read <= DEFAULT_MAX_BYTES
    assert source.pinned_official is False
    assert accepted == []
    assert excluded[0]["exclusion_reason"] == "local_source_not_verified_against_pinned_commit_in_bounded_mode"


def test_local_source_is_verified_only_when_matching_official_blob_metadata(tmp_path: Path) -> None:
    content = b"source,target\nTelugu paragraph,English paragraph\n"
    local_file = tmp_path / "tel_indicdisco-mt_test.csv"
    local_file.write_bytes(content)
    source = classify_source(
        str(local_file),
        revision=INDICDISCO_REVISION,
        component=INDICDISCO_PARAGRAPH_COMPONENT,
        split="test",
    )
    metadata_calls: list[str] = []

    def official_metadata(split: str) -> tuple[str, int]:
        metadata_calls.append(split)
        return git_blob_oid(local_file), len(content)

    verified = verify_local_source(source, split="test", metadata_fetcher=official_metadata)

    assert metadata_calls == ["test"]
    assert verified.pinned_official is True
    assert verified.kind == "local_verified_against_pinned_hub_git_blob_oid"
    assert verified.official_file_oid == git_blob_oid(local_file)
    assert verified.official_file_size == len(content)
    assert verified.verification_method == "pinned_huggingface_git_blob_oid_and_size"


def test_local_source_mismatch_against_official_oid_fails_closed(tmp_path: Path) -> None:
    local_file = tmp_path / "tel_indicdisco-mt_test.csv"
    local_file.write_bytes(b"source,target\nmodified,content\n")
    source = classify_source(
        str(local_file),
        revision=INDICDISCO_REVISION,
        component=INDICDISCO_PARAGRAPH_COMPONENT,
        split="test",
    )

    with pytest.raises(ValueError, match="Git blob OID does not match"):
        verify_local_source(
            source,
            split="test",
            metadata_fetcher=lambda split: ("0" * 40, local_file.stat().st_size),
        )


def test_local_source_size_mismatch_fails_before_hash_comparison(tmp_path: Path) -> None:
    local_file = tmp_path / "tel_indicdisco-mt_test.csv"
    local_file.write_bytes(b"source,target\nsmall,local\n")
    source = classify_source(
        str(local_file),
        revision=INDICDISCO_REVISION,
        component=INDICDISCO_PARAGRAPH_COMPONENT,
        split="test",
    )

    with pytest.raises(ValueError, match="does not match pinned official file size"):
        verify_local_source(
            source,
            split="test",
            metadata_fetcher=lambda split: (git_blob_oid(local_file), local_file.stat().st_size + 1),
        )


def test_wrong_revision_fails_closed() -> None:
    with pytest.raises(ValueError, match="pinned IndicDISCO-MT commit"):
        classify_source(
            official_source_url("test"),
            revision="not-the-pinned-revision",
            component=INDICDISCO_PARAGRAPH_COMPONENT,
            split="test",
        )