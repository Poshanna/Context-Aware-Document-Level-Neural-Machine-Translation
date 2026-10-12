from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = BACKEND_ROOT.parent / "scripts"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from app.evaluation.document_evaluation import (
    AlignmentEvidence,
    AlignmentEvidenceKind,
    AlignmentStatus,
    ComparisonCondition,
    ContextStrategy,
    ContextStatus,
    DatasetProvenance,
    EvaluationRecord,
    EvaluationTrack,
    MetricName,
    PriorSourceText,
    ProvenanceStatus,
    ReferenceAvailability,
    TextUnit,
    TrustedDatasetAdapter,
    build_comparison_report,
    metric_eligibility,
)
from run_document_evaluation import main as run_document_evaluation_demo


def _sentence_record(**overrides: object) -> EvaluationRecord:
    values: dict[str, object] = {
        "track": EvaluationTrack.SENTENCE_LEVEL,
        "source_language": "en",
        "target_language": "te",
        "current_text": "Synthetic current sentence.",
        "current_text_unit": TextUnit.SENTENCE,
        "alignment_status": AlignmentStatus.SENTENCE_ALIGNED,
        "reference_availability": ReferenceAvailability.AVAILABLE,
        "provenance": DatasetProvenance.synthetic("sentence-test-fixture"),
        "reference_translation": "కృత్రిమ ప్రస్తుత వాక్యం.",
        "reference_unit": TextUnit.SENTENCE,
        "alignment_evidence": AlignmentEvidence(AlignmentEvidenceKind.SYNTHETIC_FIXTURE),
        "baseline_translation": "baseline output",
    }
    values.update(overrides)
    return EvaluationRecord(**values)  # type: ignore[arg-type]


def _prior(
    unit_id: str,
    position: int,
    text: str,
    *,
    unit_type: TextUnit = TextUnit.DOCUMENT,
    label: str = "document-test-fixture",
) -> PriorSourceText:
    return PriorSourceText(
        unit_id,
        position,
        text,
        DatasetProvenance.synthetic(label, unit_id=unit_id, unit_type=unit_type),
    )


def _document_context_record(**overrides: object) -> EvaluationRecord:
    unit_id = str(overrides.get("document_or_passage_id", "synthetic-document-a"))
    provenance = DatasetProvenance.synthetic(
        "document-test-fixture", unit_id=unit_id, unit_type=TextUnit.DOCUMENT
    )
    values: dict[str, object] = {
        "track": EvaluationTrack.DOCUMENT_CONTEXT,
        "source_language": "en",
        "target_language": "te",
        "current_text": "Current source sentence.",
        "current_text_unit": TextUnit.SENTENCE,
        "alignment_status": AlignmentStatus.DOCUMENT_PAIRED,
        "reference_availability": ReferenceAvailability.UNAVAILABLE,
        "provenance": provenance,
        "alignment_evidence": AlignmentEvidence(AlignmentEvidenceKind.SYNTHETIC_FIXTURE),
        "document_or_passage_id": unit_id,
        "current_unit_position": 2,
        "preceding_context": (_prior(unit_id, 1, "Prior source sentence."),),
        "context_status": ContextStatus.AVAILABLE_NOT_USED,
        "context_strategy": ContextStrategy.LAST_N,
        "context_window": 1,
        "baseline_translation": "baseline output",
    }
    values.update(overrides)
    return EvaluationRecord(**values)  # type: ignore[arg-type]


def test_valid_synthetic_sentence_level_record_is_not_research_metric_eligible() -> None:
    record = _sentence_record()

    assert record.track is EvaluationTrack.SENTENCE_LEVEL
    assert record.current_text_unit is TextUnit.SENTENCE
    assert metric_eligibility(record, MetricName.SENTENCE_BLEU) == (
        False,
        "provenance is synthetic",
    )
    assert metric_eligibility(record, MetricName.SENTENCE_CHRF) == (
        False,
        "provenance is synthetic",
    )


def test_document_context_record_can_have_no_sentence_reference() -> None:
    record = _document_context_record()

    assert record.reference_availability is ReferenceAvailability.UNAVAILABLE
    assert record.context_status is ContextStatus.AVAILABLE_NOT_USED
    assert record.context_supplied_to_model is False
    assert metric_eligibility(record, MetricName.SENTENCE_BLEU) == (
        False,
        "reference unavailable",
    )
    report = build_comparison_report([record]).to_dict()
    track = next(item for item in report["tracks"] if item["track"] == "document_context")
    assert track["records"][0]["reference_availability"] == "unavailable"
    assert track["qualitative_observations"][0]["record_index"] == 0


def test_valid_context_conditioned_record_marks_context_as_supplied() -> None:
    record = _document_context_record(
        context_status=ContextStatus.SUPPLIED,
        context_supplied_to_model=True,
        context_conditioned_translation="context-conditioned output",
    )

    assert record.context_status is ContextStatus.SUPPLIED
    assert record.context_supplied_to_model is True


def test_valid_passage_context_record_preserves_passage_scope() -> None:
    passage_id = "synthetic-passage-a"
    provenance = DatasetProvenance.synthetic(
        "passage-test-fixture", unit_id=passage_id, unit_type=TextUnit.PASSAGE
    )
    record = EvaluationRecord(
        track=EvaluationTrack.PASSAGE_CONTEXT,
        source_language="en",
        target_language="te",
        current_text="Synthetic complete passage.",
        current_text_unit=TextUnit.PASSAGE,
        alignment_status=AlignmentStatus.PASSAGE_PAIRED,
        reference_availability=ReferenceAvailability.AVAILABLE,
        provenance=provenance,
        reference_translation="కృత్రిమ పూర్తి భాగం.",
        reference_unit=TextUnit.PASSAGE,
        alignment_evidence=AlignmentEvidence(AlignmentEvidenceKind.SYNTHETIC_FIXTURE),
        document_or_passage_id=passage_id,
        baseline_translation="synthetic passage output",
    )

    assert record.current_text_unit is TextUnit.PASSAGE
    assert record.reference_unit is TextUnit.PASSAGE
    assert metric_eligibility(record, MetricName.PASSAGE_BLEU) == (
        False,
        "provenance is synthetic",
    )


def test_track_unit_mismatch_is_rejected() -> None:
    passage_id = "synthetic-passage-a"
    with pytest.raises(ValueError, match="track identity must match"):
        EvaluationRecord(
            track=EvaluationTrack.DOCUMENT_CONTEXT,
            source_language="en",
            target_language="te",
            current_text="Current source sentence.",
            current_text_unit=TextUnit.SENTENCE,
            alignment_status=AlignmentStatus.UNVERIFIED,
            reference_availability=ReferenceAvailability.UNAVAILABLE,
            provenance=DatasetProvenance.synthetic(
                "passage-test-fixture", unit_id=passage_id, unit_type=TextUnit.PASSAGE
            ),
            document_or_passage_id=passage_id,
        )


def test_future_context_is_rejected() -> None:
    unit_id = "synthetic-document-a"
    with pytest.raises(ValueError, match="future or current"):
        _document_context_record(
            preceding_context=(_prior(unit_id, 2, "Same-order sentence."),)
        )


def test_current_unit_position_must_be_nonnegative_integer() -> None:
    with pytest.raises(ValueError, match="current_unit_position must be a non-negative integer"):
        _document_context_record(current_unit_position=True)


def test_document_boundary_violation_is_rejected() -> None:
    with pytest.raises(ValueError, match="crosses a document/passage boundary"):
        _document_context_record(
            preceding_context=(_prior("synthetic-document-b", 1, "Other document."),)
        )


def test_missing_or_incompatible_alignment_evidence_is_rejected() -> None:
    with pytest.raises(TypeError):
        EvaluationRecord(
            track=EvaluationTrack.SENTENCE_LEVEL,
            source_language="en",
            target_language="te",
            current_text="source",
            current_text_unit=TextUnit.SENTENCE,
            reference_availability=ReferenceAvailability.AVAILABLE,
            provenance=DatasetProvenance.synthetic("alignment-test-fixture"),
            reference_translation="reference",
            reference_unit=TextUnit.SENTENCE,
        )  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="does not support alignment_status"):
        _sentence_record(
            alignment_evidence=AlignmentEvidence(AlignmentEvidenceKind.DATASET_PASSAGE_PAIR_ID)
        )


def test_unavailable_reference_suppresses_metrics_and_reports_reason() -> None:
    record = _document_context_record()
    eligible, reason = metric_eligibility(
        record,
        MetricName.SENTENCE_CHRF,
        condition=ComparisonCondition.BASELINE,
    )

    assert eligible is False
    assert reason == "reference unavailable"
    report = build_comparison_report([record]).to_dict()
    track = next(item for item in report["tracks"] if item["track"] == "document_context")
    eligibility = next(
        item
        for item in track["records"][0]["metric_eligibility"]
        if item["metric"] == "sentence_chrf++"
        and item["condition"] == "baseline"
    )
    assert eligibility["eligible"] is False
    assert eligibility["reason"] == "reference unavailable"


def test_baseline_and_context_outputs_share_the_same_current_sentence() -> None:
    record = _document_context_record(
        context_status=ContextStatus.SUPPLIED,
        context_supplied_to_model=True,
        context_conditioned_translation="context-conditioned output",
    )
    report = build_comparison_report([record]).to_dict()
    track = next(item for item in report["tracks"] if item["track"] == "document_context")
    item = track["records"][0]

    assert item["current_text"] == record.current_text
    assert item["baseline_translation"] == "baseline output"
    assert item["context_conditioned_translation"] == "context-conditioned output"


def test_report_order_preserves_source_sentence_order_deterministically() -> None:
    later = _document_context_record(
        current_unit_position=3,
        preceding_context=(
            _prior("synthetic-document-a", 1, "First prior."),
            _prior("synthetic-document-a", 2, "Second prior."),
        ),
        context_window=2,
    )
    earlier = _document_context_record(
        current_unit_position=1,
        preceding_context=(_prior("synthetic-document-a", 0, "Prior."),),
    )

    first_report = build_comparison_report([later, earlier]).to_dict()
    second_report = build_comparison_report([later, earlier]).to_dict()
    track = next(item for item in first_report["tracks"] if item["track"] == "document_context")

    assert first_report == second_report
    assert [item["current_unit_position"] for item in track["records"]] == [1, 3]


def test_exclusion_reason_is_reported_and_suppresses_metrics() -> None:
    record = _sentence_record(exclusion_reason="reference failed manual verification")
    report = build_comparison_report([record]).to_dict()
    track = next(item for item in report["tracks"] if item["track"] == "sentence_level")

    assert track["excluded_records"] == [
        {"record_index": 0, "reason": "reference failed manual verification"}
    ]
    eligibility = next(
        item
        for item in track["records"][0]["metric_eligibility"]
        if item["metric"] == "sentence_bleu"
    )
    assert eligibility["eligible"] is False
    assert eligibility["reason"] == "excluded: reference failed manual verification"


def test_document_reference_is_not_silently_used_as_sentence_reference() -> None:
    with pytest.raises(ValueError, match="available references must match"):
        _document_context_record(
            reference_availability=ReferenceAvailability.AVAILABLE,
            reference_translation="paired synthetic document reference",
            reference_unit=TextUnit.DOCUMENT,
        )


def test_unavailable_context_cannot_contain_context_items() -> None:
    with pytest.raises(ValueError, match="unavailable context"):
        _document_context_record(
            context_status=ContextStatus.UNAVAILABLE,
            preceding_context=(_prior("synthetic-document-a", 1, "Prior."),),
        )


def test_context_items_must_be_in_unique_source_order_and_within_window() -> None:
    unit_id = "synthetic-document-a"
    with pytest.raises(ValueError, match="unique and in source order"):
        _document_context_record(
            preceding_context=(
                _prior(unit_id, 2, "Second prior."),
                _prior(unit_id, 1, "First prior."),
            ),
            current_unit_position=4,
            context_window=2,
        )

    with pytest.raises(ValueError, match="complete eligible preceding window"):
        _document_context_record(
            preceding_context=(
                _prior(unit_id, 0, "First prior."),
                _prior(unit_id, 1, "Second prior."),
            ),
            context_window=1,
        )


def test_invalid_alignment_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="alignment_status must be an AlignmentStatus"):
        _sentence_record(alignment_status="fabricated_sentence_alignment")


def test_cli_requires_the_synthetic_demo_flag() -> None:
    with pytest.raises(SystemExit) as error:
        run_document_evaluation_demo([])

    assert error.value.code == 2


def test_cli_demo_reports_synthetic_tracks_only(capsys: pytest.CaptureFixture[str]) -> None:
    import json

    assert run_document_evaluation_demo(["--demo"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert {track["track"] for track in output["tracks"]} == {
        "sentence_level",
        "document_context",
        "passage_context",
    }
    items = [record for track in output["tracks"] for record in track["records"]]
    assert all(item["provenance"]["status"] == "synthetic" for item in items)
    assert all(
        not any(metric["eligible"] for metric in item["metric_eligibility"])
        for item in items
    )


def test_verified_provenance_cannot_be_forged_by_direct_construction() -> None:
    with pytest.raises(TypeError, match="TrustedDatasetAdapter"):
        DatasetProvenance(status=ProvenanceStatus.VERIFIED_DATASET)  # type: ignore[call-arg]

    with pytest.raises(TypeError, match="TrustedDatasetAdapter"):
        DatasetProvenance._from_adapter(
            object(),  # type: ignore[arg-type]
            status=ProvenanceStatus.VERIFIED_DATASET,
            dataset_name="",
            dataset_revision="",
            split="",
            record_id="",
            source_location="",
            reference_location=None,
            alignment_location=None,
            unit_id=None,
            unit_type=None,
        )


def test_arbitrary_document_id_does_not_establish_context_provenance() -> None:
    with pytest.raises(ValueError, match="track identity must match"):
        EvaluationRecord(
            track=EvaluationTrack.DOCUMENT_CONTEXT,
            source_language="en",
            target_language="te",
            current_text="Current sentence.",
            current_text_unit=TextUnit.SENTENCE,
            alignment_status=AlignmentStatus.UNVERIFIED,
            reference_availability=ReferenceAvailability.UNAVAILABLE,
            provenance=DatasetProvenance.unverified(),
            document_or_passage_id="caller-invented-unit",
        )


def test_duplicate_current_positions_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate current positions"):
        build_comparison_report([_document_context_record(), _document_context_record()])


def test_duplicate_context_positions_are_rejected() -> None:
    unit_id = "synthetic-document-a"
    with pytest.raises(ValueError, match="context source positions must be unique"):
        _document_context_record(
            current_unit_position=2,
            preceding_context=(
                _prior(unit_id, 1, "First duplicate position."),
                _prior(unit_id, 1, "Second duplicate position."),
            ),
            context_window=2,
        )


def test_last_n_rejects_stale_but_prior_context() -> None:
    unit_id = "synthetic-document-a"
    with pytest.raises(ValueError, match="complete eligible preceding window"):
        _document_context_record(
            current_unit_position=5,
            preceding_context=(_prior(unit_id, 2, "Prior, but not last-N."),),
            context_window=1,
        )


def test_synthetic_provenance_cannot_authorize_metrics_with_plausible_unit_id() -> None:
    record = _document_context_record(
        document_or_passage_id="synthetic-unit-with-plausible-label",
    )

    assert record.provenance.status is ProvenanceStatus.SYNTHETIC
    assert record.provenance.dataset_name is None
    assert record.provenance.record_id is None
    assert metric_eligibility(record, MetricName.SENTENCE_BLEU) == (
        False,
        "reference unavailable",
    )


def test_audit_report_contains_provenance_reference_context_and_outputs() -> None:
    record = _document_context_record(
        context_status=ContextStatus.SUPPLIED,
        context_supplied_to_model=True,
        context_conditioned_translation="synthetic context output",
    )
    report = build_comparison_report([record]).to_dict()
    track = next(item for item in report["tracks"] if item["track"] == "document_context")
    item = track["records"][0]

    assert item["source_language"] == "en"
    assert item["target_language"] == "te"
    assert item["track"] == "document_context"
    assert item["provenance"]["status"] == "synthetic"
    assert {
        "dataset_name",
        "dataset_revision",
        "split",
        "record_id",
        "source_location",
        "reference_location",
        "alignment_location",
        "adapter_name",
        "unit_id",
        "unit_type",
    } <= set(item["provenance"])
    assert item["provenance"]["unit_id"] == "synthetic-document-a"
    assert item["provenance"]["dataset_name"] is None
    assert item["document_or_passage_id"] == "synthetic-document-a"
    assert item["current_unit_position"] == 2
    assert item["current_text_unit"] == "sentence"
    assert item["reference_availability"] == "unavailable"
    assert item["reference_unit"] is None
    assert item["alignment_status"] == "document_paired"
    assert item["alignment_evidence"]["kind"] == "synthetic_fixture"
    assert item["context_strategy"] == "last_n"
    assert item["context_window"] == 1
    assert item["context_status"] == "context_supplied_to_model"
    assert item["context_supplied_to_model"] is True
    assert item["baseline_translation"] == "baseline output"
    assert item["context_conditioned_translation"] == "synthetic context output"
    assert item["metric_eligibility"]
    assert item["exclusion_reason"] is None