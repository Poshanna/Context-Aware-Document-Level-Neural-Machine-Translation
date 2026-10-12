"""Trusted paragraph-level records for the pinned IndicDISCO-MT release.

This module adapts caller-loaded rows only. It does not download files, split
paragraphs, infer document IDs, build context, translate, or score outputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from app.evaluation.document_evaluation import (
    DatasetProvenance,
    ProvenanceStatus,
    TrustedDatasetAdapter,
)

INDICDISCO_DATASET = "IndicDISCO-MT"
INDICDISCO_REPOSITORY = "HimangY/IndicDISCO-MT"
INDICDISCO_REVISION = "13f4a8caa71b00859ee4235713aa253c3b2fe193"
INDICDISCO_PARAGRAPH_COMPONENT = "Indicdisco_MT"
INDICDISCO_SOURCE_LANGUAGE = "Telugu"
INDICDISCO_TARGET_LANGUAGE = "English"
INDICDISCO_ADAPTER_NAME = "indicdisco_mt_paragraph_csv_v1"


class IndicDiscoUnitType(str, Enum):
    PARAGRAPH = "paragraph"


class IndicDiscoAlignment(str, Enum):
    PARAGRAPH_PAIR = "paragraph_pair"


class IndicDiscoAvailability(str, Enum):
    UNAVAILABLE = "unavailable"


class IndicDiscoRightsStatus(str, Enum):
    UNRESOLVED = "unresolved"


class IndicDiscoMetric(str, Enum):
    PARAGRAPH_BLEU = "paragraph_bleu"
    PARAGRAPH_CHRF = "paragraph_chrf++"
    SENTENCE_BLEU = "sentence_bleu"
    SENTENCE_CHRF = "sentence_chrf++"
    DOCUMENT_CONTEXT_BLEU = "document_context_bleu"
    DOCUMENT_CONTEXT_CHRF = "document_context_chrf++"


@dataclass(frozen=True)
class IndicDiscoMetricEligibility:
    metric: IndicDiscoMetric
    eligible: bool
    reason: str


@dataclass(frozen=True)
class IndicDiscoParagraphRecord:
    """One complete same-row paragraph pair; row locator is not a document ID."""

    dataset: str
    revision: str
    split: str
    component: str
    row_locator: str
    source_language: str
    target_language: str
    source_text: str
    reference_text: str
    provenance: DatasetProvenance
    unit_type: IndicDiscoUnitType = IndicDiscoUnitType.PARAGRAPH
    alignment: IndicDiscoAlignment = IndicDiscoAlignment.PARAGRAPH_PAIR
    document_id: None = None
    ordering: None = None
    cross_row_context: IndicDiscoAvailability = IndicDiscoAvailability.UNAVAILABLE
    previous_paragraph_context: IndicDiscoAvailability = IndicDiscoAvailability.UNAVAILABLE
    same_document_context: IndicDiscoAvailability = IndicDiscoAvailability.UNAVAILABLE
    within_paragraph_sentence_context: IndicDiscoAvailability = IndicDiscoAvailability.UNAVAILABLE
    rights_status: IndicDiscoRightsStatus = IndicDiscoRightsStatus.UNRESOLVED

    def __post_init__(self) -> None:
        for name, value in (
            ("dataset", self.dataset),
            ("revision", self.revision),
            ("split", self.split),
            ("component", self.component),
            ("row_locator", self.row_locator),
            ("source_language", self.source_language),
            ("target_language", self.target_language),
            ("source_text", self.source_text),
            ("reference_text", self.reference_text),
        ):
            _require_text(value, name)

        if self.dataset != INDICDISCO_DATASET:
            raise ValueError("dataset must be IndicDISCO-MT.")
        if self.revision != INDICDISCO_REVISION:
            raise ValueError("revision must match the pinned IndicDISCO-MT commit.")
        if self.split not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev, or test.")
        if self.component != INDICDISCO_PARAGRAPH_COMPONENT:
            raise ValueError("component must be Indicdisco_MT for paragraph records.")
        if (self.source_language, self.target_language) != (
            INDICDISCO_SOURCE_LANGUAGE,
            INDICDISCO_TARGET_LANGUAGE,
        ):
            raise ValueError("language direction must be Telugu to English as released.")
        if self.unit_type is not IndicDiscoUnitType.PARAGRAPH:
            raise ValueError("unit_type must be paragraph.")
        if self.alignment is not IndicDiscoAlignment.PARAGRAPH_PAIR:
            raise ValueError("alignment must be paragraph_pair.")
        if self.document_id is not None or self.ordering is not None:
            raise ValueError("document_id and ordering are unavailable in this release.")
        for name in (
            "cross_row_context",
            "previous_paragraph_context",
            "same_document_context",
            "within_paragraph_sentence_context",
        ):
            if getattr(self, name) is not IndicDiscoAvailability.UNAVAILABLE:
                raise ValueError(f"{name} must remain unavailable.")
        if self.rights_status is not IndicDiscoRightsStatus.UNRESOLVED:
            raise ValueError("dataset rights status must remain unresolved.")

        if not isinstance(self.provenance, DatasetProvenance):
            raise ValueError("record requires structured DatasetProvenance.")
        if not self.provenance.is_verified:
            raise ValueError("accepted paragraph records require verified dataset provenance.")
        if (
            self.provenance.dataset_name != self.dataset
            or self.provenance.dataset_revision != self.revision
            or self.provenance.split != self.split
            or self.provenance.record_id != self.row_locator
        ):
            raise ValueError("provenance must match dataset, revision, split, and row locator.")
        if self.provenance.unit_id is not None or self.provenance.unit_type is not None:
            raise ValueError("provenance cannot claim an unavailable document or paragraph ID.")
        if (
            self.provenance.source_location is None
            or self.provenance.reference_location is None
            or self.provenance.alignment_location is None
        ):
            raise ValueError("source, reference, and paragraph-pair locations are required.")
        if self.provenance.adapter_name != INDICDISCO_ADAPTER_NAME:
            raise ValueError("provenance must be issued by the IndicDISCO-MT paragraph adapter.")

    @property
    def provenance_status(self) -> ProvenanceStatus:
        return self.provenance.status


class IndicDiscoMTTrustedAdapter(TrustedDatasetAdapter):
    """Validate an already-loaded core IndicDISCO-MT paragraph row."""

    @property
    def adapter_name(self) -> str:
        return INDICDISCO_ADAPTER_NAME

    def adapt_row(
        self,
        *,
        row: Mapping[str, object],
        revision: str,
        split: str,
        component: str,
        row_locator: str,
        source_location: str,
        reference_location: str,
        alignment_location: str,
        source_language: str,
        target_language: str,
    ) -> IndicDiscoParagraphRecord:
        if revision != INDICDISCO_REVISION:
            raise ValueError("revision is missing or does not match the pinned IndicDISCO-MT commit.")
        if split not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev, or test.")
        if component != INDICDISCO_PARAGRAPH_COMPONENT:
            raise ValueError("only the Indicdisco_MT paragraph component is supported.")
        for name, value in (
            ("row_locator", row_locator),
            ("source_location", source_location),
            ("reference_location", reference_location),
            ("alignment_location", alignment_location),
            ("source_language", source_language),
            ("target_language", target_language),
        ):
            _require_text(value, name)
        if not isinstance(row, Mapping) or set(row) != {"source", "target"}:
            raise ValueError("core paragraph rows must contain exactly source and target fields.")
        source_text = row.get("source")
        reference_text = row.get("target")
        _require_text(source_text, "source")
        _require_text(reference_text, "target")
        if (source_language, target_language) != (
            INDICDISCO_SOURCE_LANGUAGE,
            INDICDISCO_TARGET_LANGUAGE,
        ):
            raise ValueError("language direction must be Telugu to English as released.")

        provenance = self.verified_provenance(
            status=ProvenanceStatus.VERIFIED_DATASET,
            dataset_name=INDICDISCO_DATASET,
            dataset_revision=revision,
            split=split,
            record_id=row_locator,
            source_location=source_location,
            reference_location=reference_location,
            alignment_location=alignment_location,
        )
        return IndicDiscoParagraphRecord(
            dataset=INDICDISCO_DATASET,
            revision=revision,
            split=split,
            component=component,
            row_locator=row_locator,
            source_language=source_language,
            target_language=target_language,
            source_text=source_text,
            reference_text=reference_text,
            provenance=provenance,
        )

    def reject_cross_row_context(
        self,
        record: IndicDiscoParagraphRecord,
        preceding_rows: Sequence[IndicDiscoParagraphRecord],
    ) -> None:
        if not isinstance(record, IndicDiscoParagraphRecord):
            raise TypeError("record must be an IndicDiscoParagraphRecord.")
        if preceding_rows:
            raise ValueError(
                "cross-row context is unavailable: the release does not establish "
                "parent-document identity or paragraph order."
            )


def metric_eligibility(
    record: IndicDiscoParagraphRecord,
    metric: IndicDiscoMetric,
) -> IndicDiscoMetricEligibility:
    """Return structural eligibility only; never compute a metric."""

    if not isinstance(record, IndicDiscoParagraphRecord):
        raise TypeError("record must be an IndicDiscoParagraphRecord.")
    if not isinstance(metric, IndicDiscoMetric):
        raise ValueError("metric must be an IndicDiscoMetric.")
    if metric in {IndicDiscoMetric.PARAGRAPH_BLEU, IndicDiscoMetric.PARAGRAPH_CHRF}:
        return IndicDiscoMetricEligibility(
            metric=metric,
            eligible=True,
            reason=(
                "same-row paragraph reference is available; structural eligibility only; "
                "rights remain unresolved and no score was computed"
            ),
        )
    if metric in {IndicDiscoMetric.SENTENCE_BLEU, IndicDiscoMetric.SENTENCE_CHRF}:
        return IndicDiscoMetricEligibility(
            metric=metric,
            eligible=False,
            reason="sentence references and sentence alignment are unavailable in the core paragraph rows",
        )
    return IndicDiscoMetricEligibility(
        metric=metric,
        eligible=False,
        reason="document identity, cross-row ordering, and document context are unavailable",
    )


def _require_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
