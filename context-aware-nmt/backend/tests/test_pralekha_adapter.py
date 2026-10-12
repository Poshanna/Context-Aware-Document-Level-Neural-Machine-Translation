from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.evaluation.document_evaluation import (
    AlignmentStatus,
    DatasetProvenance,
    EvaluationTrack,
    ProvenanceStatus,
    ReferenceAvailability,
    TextUnit,
    TrustedDatasetAdapter,
)
from app.evaluation.pralekha_adapter import (
    PraLekhaDocumentPair,
    PraLekhaDocumentUnit,
    PraLekhaMetric,
    build_causal_context,
    metric_eligibility,
    require_pralekha_provenance,
)


class _SyntheticAdapter(TrustedDatasetAdapter):
    @property
    def adapter_name(self) -> str:
        return "pralekha_test_adapter_v1"


def _provenance(*, unit_id: str, document_id: str = "doc-1", split: str = "test") -> DatasetProvenance:
    return _SyntheticAdapter().verified_provenance(
        status=ProvenanceStatus.VERIFIED_DATASET,
        dataset_name="AI4Bharat PRALEKHA",
        dataset_revision="4c16a270141891dd5c3e98f6f4f207f11bd11236",
        split=split,
        record_id="pair-42",
        source_location=f"fixture://{document_id}.en.txt#L1:source_text",
        reference_location=f"fixture://{document_id}.te.txt#L1:target_text",
        alignment_location=f"fixture://{document_id}.json#L1:n_id,doc_id,lang,source_text,target_text",
        unit_id=unit_id,
        unit_type=TextUnit.DOCUMENT,
    )


def _unit(*, unit_id: str, order: int, text: str, document_id: str = "doc-1") -> PraLekhaDocumentUnit:
    return PraLekhaDocumentUnit(
        document_id=document_id,
        unit_id=unit_id,
        unit_order=order,
        source_text=text,
        reference_text=f"ref:{text}",
        alignment_status=AlignmentStatus.DOCUMENT_PAIRED,
        provenance=_provenance(unit_id=unit_id, document_id=document_id),
        source_language="en",
        target_language="te",
    )


def test_provenance_validation_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="required PRALEKHA provenance"):
        require_pralekha_provenance(
            DatasetProvenance.unverified(),
            document_id="doc-1",
            pair_id="pair-1",
            split="test",
            language="en",
        )


def test_document_identity_and_pair_identity_are_preserved() -> None:
    pair = PraLekhaDocumentPair(
        dataset_name="AI4Bharat PRALEKHA",
        dataset_revision="4c16a270141891dd5c3e98f6f4f207f11bd11236",
        split="test",
        pair_id="pair-42",
        source_document_id="src-doc-17",
        target_document_id="tgt-doc-17",
        source_language="en",
        target_language="te",
        source_text="Hello world.",
        target_text="హలో ప్రపంచం.",
        provenance=_provenance(unit_id="doc-17", document_id="src-doc-17"),
        unit_order=0,
    )

    assert pair.source_document_id == "src-doc-17"
    assert pair.target_document_id == "tgt-doc-17"
    assert pair.pair_id == "pair-42"
    assert pair.source_text == "Hello world."
    assert pair.target_text == "హలో ప్రపంచం."


def test_unit_ordering_is_preserved_and_duplicates_are_rejected() -> None:
    first = _unit(unit_id="u-1", order=0, text="first")
    second = _unit(unit_id="u-2", order=1, text="second")
    duplicates = (first, _unit(unit_id="u-1", order=1, text="duplicate"))

    assert [item.unit_order for item in (first, second)] == [0, 1]
    with pytest.raises(ValueError, match="duplicate unit"):
        build_causal_context(duplicates, current_unit_id="u-2", last_n=2)


def test_causal_context_uses_preceding_units_only() -> None:
    units = (
        _unit(unit_id="u-1", order=0, text="first"),
        _unit(unit_id="u-2", order=1, text="second"),
        _unit(unit_id="u-3", order=2, text="third"),
    )

    context = build_causal_context(units, current_unit_id="u-3", last_n=2)

    assert [item.unit_id for item in context] == ["u-1", "u-2"]
    assert [item.unit_order for item in context] == [0, 1]


def test_future_units_and_cross_document_context_are_rejected() -> None:
    doc_a = (
        _unit(unit_id="u-1", order=0, text="a-1", document_id="doc-a"),
        _unit(unit_id="u-2", order=1, text="a-2", document_id="doc-a"),
    )
    doc_b = (_unit(unit_id="u-3", order=0, text="b-1", document_id="doc-b"),)

    with pytest.raises(ValueError, match="future or current"):
        build_causal_context(doc_a, current_unit_id="u-1", last_n=2)

    with pytest.raises(ValueError, match="same verified document"):
        build_causal_context(doc_a + doc_b, current_unit_id="u-2", last_n=1)


def test_last_n_context_handles_window_and_order() -> None:
    units = tuple(_unit(unit_id=f"u-{index}", order=index, text=f"text-{index}") for index in range(5))

    assert [item.unit_id for item in build_causal_context(units, current_unit_id="u-4", last_n=2)] == ["u-2", "u-3"]
    assert [item.unit_id for item in build_causal_context(units, current_unit_id="u-4", last_n=1)] == ["u-3"]


def test_missing_alignment_rejection_uses_document_scope() -> None:
    record = _unit(unit_id="u-1", order=0, text="one")
    with pytest.raises(ValueError, match="document alignment"):
        metric_eligibility(
            record,
            PraLekhaMetric.DOCUMENT_BLEU,
            source_reference_available=False,
            sentence_alignment_present=False,
        )


def test_metric_eligibility_mark_sentence_blu_is_unavailable_without_sentence_alignment() -> None:
    record = _unit(unit_id="u-1", order=0, text="one")

    assert metric_eligibility(record, PraLekhaMetric.SENTENCE_BLEU, source_reference_available=False, sentence_alignment_present=False) == (
        False,
        "sentence-level reference unavailable without sentence alignment",
    )


def test_exclusion_reasons_are_preserved_deterministically() -> None:
    record = _unit(unit_id="u-1", order=0, text="one")
    record = record.with_exclusion_reason("missing document pair id")

    assert record.exclusion_reason == "missing document pair id"
    assert metric_eligibility(record, PraLekhaMetric.DOCUMENT_BLEU, source_reference_available=True, sentence_alignment_present=False)[0] is False


def test_serialization_is_reproducible_for_reporting() -> None:
    pair = PraLekhaDocumentPair(
        dataset_name="AI4Bharat PRALEKHA",
        dataset_revision="4c16a270141891dd5c3e98f6f4f207f11bd11236",
        split="dev",
        pair_id="pair-42",
        source_document_id="src-doc-99",
        target_document_id="tgt-doc-99",
        source_language="en",
        target_language="te",
        source_text="English document.",
        target_text="తెలుగు పత్రం.",
        provenance=_provenance(unit_id="unit-99", document_id="src-doc-99", split="dev"),
        unit_order=0,
    )

    payload = json.loads(json.dumps(pair.to_dict(), ensure_ascii=False))
    assert payload["pair_id"] == "pair-42"
    assert payload["source_document_id"] == "src-doc-99"
    assert payload["target_document_id"] == "tgt-doc-99"
    assert payload["split"] == "dev"
