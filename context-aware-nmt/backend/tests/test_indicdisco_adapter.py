from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.evaluation.document_evaluation import ProvenanceStatus
from app.evaluation.indicdisco_adapter import (
    INDICDISCO_PARAGRAPH_COMPONENT,
    INDICDISCO_REVISION,
    INDICDISCO_SOURCE_LANGUAGE,
    INDICDISCO_TARGET_LANGUAGE,
    IndicDiscoAvailability,
    IndicDiscoMetric,
    IndicDiscoRightsStatus,
    IndicDiscoUnitType,
    IndicDiscoMTTrustedAdapter,
    metric_eligibility,
)


def _record(*, row_locator: str = "fixture://test.csv#row=0"):
    return IndicDiscoMTTrustedAdapter().adapt_row(
        row={"source": "తెలుగు మూల పేరా.", "target": "English reference paragraph."},
        revision=INDICDISCO_REVISION,
        split="test",
        component=INDICDISCO_PARAGRAPH_COMPONENT,
        row_locator=row_locator,
        source_location=f"fixture://test.csv#{row_locator}:source",
        reference_location=f"fixture://test.csv#{row_locator}:target",
        alignment_location=f"fixture://test.csv#{row_locator}:source,target",
        source_language=INDICDISCO_SOURCE_LANGUAGE,
        target_language=INDICDISCO_TARGET_LANGUAGE,
    )


def test_adapter_preserves_pinned_revision_component_and_row_locator() -> None:
    record = _record(row_locator="fixture://test.csv#row=7")

    assert record.revision == "13f4a8caa71b00859ee4235713aa253c3b2fe193"
    assert record.component == "Indicdisco_MT"
    assert record.row_locator == "fixture://test.csv#row=7"
    assert record.provenance.record_id == record.row_locator
    assert record.provenance_status is ProvenanceStatus.VERIFIED_DATASET


def test_record_preserves_paragraph_unit_text_and_languages() -> None:
    record = _record()

    assert record.unit_type is IndicDiscoUnitType.PARAGRAPH
    assert record.alignment.value == "paragraph_pair"
    assert record.source_text == "తెలుగు మూల పేరా."
    assert record.reference_text == "English reference paragraph."
    assert record.source_language == "Telugu"
    assert record.target_language == "English"


def test_document_identity_order_and_all_context_states_remain_unavailable() -> None:
    record = _record()

    assert record.document_id is None
    assert record.ordering is None
    assert record.cross_row_context is IndicDiscoAvailability.UNAVAILABLE
    assert record.previous_paragraph_context is IndicDiscoAvailability.UNAVAILABLE
    assert record.same_document_context is IndicDiscoAvailability.UNAVAILABLE
    assert record.within_paragraph_sentence_context is IndicDiscoAvailability.UNAVAILABLE


def test_cross_row_context_is_rejected() -> None:
    adapter = IndicDiscoMTTrustedAdapter()
    current = _record(row_locator="fixture://test.csv#row=0")
    preceding = _record(row_locator="fixture://test.csv#row=1")

    with pytest.raises(ValueError, match="cross-row context is unavailable"):
        adapter.reject_cross_row_context(current, (preceding,))


@pytest.mark.parametrize(
    "metric",
    [IndicDiscoMetric.PARAGRAPH_BLEU, IndicDiscoMetric.PARAGRAPH_CHRF],
)
def test_paragraph_metrics_are_structurally_eligible(metric: IndicDiscoMetric) -> None:
    decision = metric_eligibility(_record(), metric)

    assert decision.eligible is True
    assert "structural eligibility only" in decision.reason
    assert "rights remain unresolved" in decision.reason


@pytest.mark.parametrize(
    "metric",
    [IndicDiscoMetric.SENTENCE_BLEU, IndicDiscoMetric.SENTENCE_CHRF],
)
def test_sentence_metrics_are_unavailable(metric: IndicDiscoMetric) -> None:
    decision = metric_eligibility(_record(), metric)

    assert decision.eligible is False
    assert "sentence references" in decision.reason


@pytest.mark.parametrize(
    "metric",
    [IndicDiscoMetric.DOCUMENT_CONTEXT_BLEU, IndicDiscoMetric.DOCUMENT_CONTEXT_CHRF],
)
def test_document_context_metrics_are_unavailable(metric: IndicDiscoMetric) -> None:
    decision = metric_eligibility(_record(), metric)

    assert decision.eligible is False
    assert "document identity" in decision.reason


def test_missing_revision_or_provenance_location_fails_closed() -> None:
    adapter = IndicDiscoMTTrustedAdapter()
    arguments = {
        "row": {"source": "source paragraph", "target": "reference paragraph"},
        "revision": INDICDISCO_REVISION,
        "split": "test",
        "component": INDICDISCO_PARAGRAPH_COMPONENT,
        "row_locator": "fixture://test.csv#row=0",
        "source_location": "fixture://test.csv#row=0:source",
        "reference_location": "fixture://test.csv#row=0:target",
        "alignment_location": "fixture://test.csv#row=0:source,target",
        "source_language": INDICDISCO_SOURCE_LANGUAGE,
        "target_language": INDICDISCO_TARGET_LANGUAGE,
    }

    with pytest.raises(ValueError, match="revision"):
        adapter.adapt_row(**{**arguments, "revision": ""})
    with pytest.raises(ValueError, match="source_location"):
        adapter.adapt_row(**{**arguments, "source_location": ""})


def test_wrong_component_or_row_shape_is_rejected() -> None:
    adapter = IndicDiscoMTTrustedAdapter()
    base = {
        "row": {"source": "source paragraph", "target": "reference paragraph"},
        "revision": INDICDISCO_REVISION,
        "split": "test",
        "component": INDICDISCO_PARAGRAPH_COMPONENT,
        "row_locator": "fixture://test.csv#row=0",
        "source_location": "fixture://test.csv#row=0:source",
        "reference_location": "fixture://test.csv#row=0:target",
        "alignment_location": "fixture://test.csv#row=0:source,target",
        "source_language": INDICDISCO_SOURCE_LANGUAGE,
        "target_language": INDICDISCO_TARGET_LANGUAGE,
    }

    with pytest.raises(ValueError, match="only the Indicdisco_MT paragraph"):
        adapter.adapt_row(**{**base, "component": "ProAlign"})
    with pytest.raises(ValueError, match="exactly source and target"):
        adapter.adapt_row(**{**base, "row": {"source": "source", "target": "target", "sno": "1"}})


def test_rights_status_remains_unresolved() -> None:
    assert _record().rights_status is IndicDiscoRightsStatus.UNRESOLVED