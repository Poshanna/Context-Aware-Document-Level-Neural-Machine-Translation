from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.evaluation.ted2020_loader import extract_document_id, load_ted2020

DOC_ID = "314"
EN_REF = f"en/ted2020-{DOC_ID}.xml.gz"
TE_REF = f"te/ted2020-{DOC_ID}.xml.gz"


def _language_xml(language: str, sentences: list[tuple[str, str]]) -> bytes:
    body = "".join(
        f'<s id="{sentence_id}"><w>{text}</w></s>'
        for sentence_id, text in sentences
    )
    return f'<text id="1"><p>{body}</p></text>'.encode()


def _write_archive(
    path: Path, language: str, sentences: list[tuple[str, str]]
) -> Path:
    member = f"TED2020/xml/{language}/ted2020-{DOC_ID}.xml"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, _language_xml(language, sentences))
    return path


def _write_alignment(path: Path, links: list[str], *, group_id: str = DOC_ID) -> Path:
    xml = (
        '<cesAlign version="1.0"><linkGrp '
        f'fromDoc="en/ted2020-{group_id}.xml.gz" '
        f'toDoc="te/ted2020-{group_id}.xml.gz" targType="s">'
        f'{"".join(links)}</linkGrp></cesAlign>'
    )
    path.write_text(xml, encoding="utf-8")
    return path


@pytest.fixture
def release_inputs(tmp_path: Path) -> tuple[Path, Path]:
    english = _write_archive(
        tmp_path / "english.zip",
        "en",
        [("1", "English one"), ("2", "English two"), ("3", "English three")],
    )
    telugu = _write_archive(
        tmp_path / "telugu.zip",
        "te",
        [
            ("1", "తెలుగు ఒకటి"),
            ("2", "తెలుగు రెండు"),
            ("4", "తెలుగు నాలుగు"),
            ("5", "తెలుగు ఐదు"),
        ],
    )
    return english, telugu


def test_valid_one_to_one_and_path_suffix_resolution(
    tmp_path: Path, release_inputs: tuple[Path, Path]
) -> None:
    english, telugu = release_inputs
    alignment = _write_alignment(tmp_path / "en-te.xml", ['<link id="L1" xtargets="2;1"/>'])

    result = load_ted2020(alignment, english, telugu, context_window=1)

    assert result.report["documents"] == 1
    assert result.report["links"] == 1
    assert result.report["alignment_counts"]["one_to_one"] == 1
    assert result.report["included_one_to_one_records"] == 1
    assert result.report["valid"] is True
    assert result.records == [{
        "document_id": DOC_ID,
        "sentence_id": "2",
        "source_sentence": "English two",
        "reference_translation": "తెలుగు ఒకటి",
        "previous_context_sentence_ids": ["1"],
        "alignment": {
            "source_span": ["2"],
            "reference_span": ["1"],
            "method": "xml-alignment",
        },
        "phenomena": [],
        "provenance_id": f"TED2020-v1:{DOC_ID}:2:1",
    }]


@pytest.mark.parametrize(
    ("xtargets", "kind"),
    [
        ("1;1 2", "one_to_many"),
        ("1 2;1", "many_to_one"),
        ("1 2;1 2", "many_to_many"),
    ],
)
def test_non_one_to_one_alignments_are_reported_but_not_emitted(
    tmp_path: Path,
    release_inputs: tuple[Path, Path],
    xtargets: str,
    kind: str,
) -> None:
    english, telugu = release_inputs
    alignment = _write_alignment(tmp_path / "en-te.xml", [f'<link xtargets="{xtargets}"/>'])

    result = load_ted2020(alignment, english, telugu)

    assert result.report["alignment_counts"][kind] == 1
    assert result.report["excluded_non_one_to_one"] == 1
    assert result.records == []


def test_missing_source_and_target_sentence_references_are_counted(
    tmp_path: Path, release_inputs: tuple[Path, Path]
) -> None:
    english, telugu = release_inputs
    alignment = _write_alignment(
        tmp_path / "en-te.xml",
        ['<link xtargets="absent;also-absent"/>'],
    )

    result = load_ted2020(alignment, english, telugu)

    assert result.report["missing_source_references"] == 1
    assert result.report["missing_target_references"] == 1
    assert result.report["valid"] is False
    assert result.records == []


def test_missing_document_is_reported_without_records(
    tmp_path: Path, release_inputs: tuple[Path, Path]
) -> None:
    english, telugu = release_inputs
    alignment = _write_alignment(
        tmp_path / "en-te.xml", ['<link xtargets="1;1"/>'], group_id="999"
    )

    result = load_ted2020(alignment, english, telugu)

    assert result.report["unresolved_document_references"] == 2
    assert result.report["unresolved_document_reference_details"] == [
        "en/ted2020-999.xml.gz", "te/ted2020-999.xml.gz"
    ]
    assert result.records == []


def test_duplicate_sentence_id_is_reported_and_alignment_excluded(
    tmp_path: Path,
) -> None:
    english = _write_archive(
        tmp_path / "english.zip", "en", [("1", "first"), ("1", "duplicate")]
    )
    telugu = _write_archive(tmp_path / "telugu.zip", "te", [("1", "సూచన")])
    alignment = _write_alignment(tmp_path / "en-te.xml", ['<link xtargets="1;1"/>'])

    result = load_ted2020(alignment, english, telugu)

    assert result.report["duplicate_sentence_ids"] == 1
    assert result.report["malformed_links"] == 1
    assert result.records == []


def test_records_follow_document_sentence_order_not_link_order(
    tmp_path: Path, release_inputs: tuple[Path, Path]
) -> None:
    english, telugu = release_inputs
    alignment = _write_alignment(
        tmp_path / "en-te.xml",
        ['<link xtargets="3;4"/>', '<link xtargets="1;1"/>', '<link xtargets="2;2"/>'],
    )

    result = load_ted2020(alignment, english, telugu)

    assert [record["sentence_id"] for record in result.records] == ["1", "2", "3"]


def test_context_is_same_document_prior_only_and_windowed(
    tmp_path: Path, release_inputs: tuple[Path, Path]
) -> None:
    english, telugu = release_inputs
    alignment = _write_alignment(
        tmp_path / "en-te.xml",
        ['<link xtargets="3;4"/>', '<link xtargets="2;2"/>'],
    )

    result = load_ted2020(alignment, english, telugu, context_window=1)

    assert [record["sentence_id"] for record in result.records] == ["2", "3"]
    assert result.records[0]["previous_context_sentence_ids"] == ["1"]
    assert result.records[1]["previous_context_sentence_ids"] == ["2"]
    assert all(record["document_id"] == DOC_ID for record in result.records)
    assert all(
        record["sentence_id"] not in record["previous_context_sentence_ids"]
        for record in result.records
    )


def test_context_window_zero_emits_no_context(
    tmp_path: Path, release_inputs: tuple[Path, Path]
) -> None:
    english, telugu = release_inputs
    alignment = _write_alignment(tmp_path / "en-te.xml", ['<link xtargets="3;4"/>'])

    result = load_ted2020(alignment, english, telugu, context_window=0)

    assert result.records[0]["previous_context_sentence_ids"] == []


def test_malformed_link_and_sentence_count_mismatch_are_reported(
    tmp_path: Path, release_inputs: tuple[Path, Path]
) -> None:
    english, telugu = release_inputs
    alignment = _write_alignment(
        tmp_path / "en-te.xml",
        ['<link xtargets="1 2"/>', '<link xtargets="1;1"/>'],
    )

    result = load_ted2020(alignment, english, telugu)

    assert result.report["malformed_links"] == 1
    assert result.report["sentence_count_mismatches"] == 1
    assert result.report["links"] == 2


def test_document_id_convention_and_invalid_path() -> None:
    assert extract_document_id("en/ted2020-1183.xml.gz") == "1183"
    assert extract_document_id("te/ted2020-1183.xml") == "1183"
    with pytest.raises(ValueError):
        extract_document_id("en/unrelated.xml")


def test_invalid_inputs_and_context_window_fail_fast(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="context_window"):
        load_ted2020(tmp_path / "missing.xml", tmp_path, tmp_path, context_window=-1)
    with pytest.raises(FileNotFoundError):
        load_ted2020(tmp_path / "missing.xml", tmp_path, tmp_path)
