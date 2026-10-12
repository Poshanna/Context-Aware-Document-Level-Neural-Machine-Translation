"""Typed records and reporting safeguards for document-context experiments.

This module does not load datasets, run translation models, or compute metrics.
It only validates evidence and reports whether a record is eligible to be
scored at the granularity its reference actually supports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable


class EvaluationTrack(str, Enum):
    SENTENCE_LEVEL = "sentence_level"
    DOCUMENT_CONTEXT = "document_context"
    PASSAGE_CONTEXT = "passage_context"


class TextUnit(str, Enum):
    SENTENCE = "sentence"
    DOCUMENT = "document"
    PASSAGE = "passage"


class ProvenanceStatus(str, Enum):
    SYNTHETIC = "synthetic"
    VERIFIED_DATASET = "verified_dataset"
    VERIFIED_ANNOTATION = "verified_annotation"
    UNVERIFIED = "unverified"


class AlignmentStatus(str, Enum):
    SENTENCE_ALIGNED = "sentence_aligned"
    DOCUMENT_PAIRED = "document_paired"
    PASSAGE_PAIRED = "passage_paired"
    UNVERIFIED = "unverified"


class AlignmentEvidenceKind(str, Enum):
    DATASET_SENTENCE_ANNOTATION = "dataset_sentence_annotation"
    EXTERNAL_SENTENCE_ANNOTATION = "external_sentence_annotation"
    DATASET_DOCUMENT_PAIR_ID = "dataset_document_pair_id"
    DATASET_PASSAGE_PAIR_ID = "dataset_passage_pair_id"
    SYNTHETIC_FIXTURE = "synthetic_fixture"


class ReferenceAvailability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ContextStatus(str, Enum):
    UNAVAILABLE = "context_unavailable"
    AVAILABLE_NOT_USED = "context_available_not_used"
    SUPPLIED = "context_supplied_to_model"


class ContextStrategy(str, Enum):
    LAST_N = "last_n"


class MetricName(str, Enum):
    SENTENCE_BLEU = "sentence_bleu"
    SENTENCE_CHRF = "sentence_chrf++"
    DOCUMENT_BLEU = "document_bleu"
    DOCUMENT_CHRF = "document_chrf++"
    PASSAGE_BLEU = "passage_bleu"
    PASSAGE_CHRF = "passage_chrf++"
    BERTSCORE = "bertscore"
    COMET = "comet"


class ComparisonCondition(str, Enum):
    BASELINE = "baseline"
    CONTEXT_CONDITIONED = "context_conditioned"


_EVIDENCE_REQUIREMENTS = {
    AlignmentEvidenceKind.DATASET_SENTENCE_ANNOTATION: (
        AlignmentStatus.SENTENCE_ALIGNED,
        ProvenanceStatus.VERIFIED_DATASET,
    ),
    AlignmentEvidenceKind.EXTERNAL_SENTENCE_ANNOTATION: (
        AlignmentStatus.SENTENCE_ALIGNED,
        ProvenanceStatus.VERIFIED_ANNOTATION,
    ),
    AlignmentEvidenceKind.DATASET_DOCUMENT_PAIR_ID: (
        AlignmentStatus.DOCUMENT_PAIRED,
        ProvenanceStatus.VERIFIED_DATASET,
    ),
    AlignmentEvidenceKind.DATASET_PASSAGE_PAIR_ID: (
        AlignmentStatus.PASSAGE_PAIRED,
        ProvenanceStatus.VERIFIED_DATASET,
    ),
    AlignmentEvidenceKind.SYNTHETIC_FIXTURE: (None, ProvenanceStatus.SYNTHETIC),
}

_UNIT_ALIGNMENT = {
    TextUnit.SENTENCE: AlignmentStatus.SENTENCE_ALIGNED,
    TextUnit.DOCUMENT: AlignmentStatus.DOCUMENT_PAIRED,
    TextUnit.PASSAGE: AlignmentStatus.PASSAGE_PAIRED,
}


_PROVENANCE_SEAL = object()


@dataclass(frozen=True, init=False)
class DatasetProvenance:
    status: ProvenanceStatus
    dataset_name: str | None
    dataset_revision: str | None
    split: str | None
    record_id: str | None
    unit_id: str | None
    unit_type: TextUnit | None
    source_location: str | None
    reference_location: str | None
    alignment_location: str | None
    adapter_name: str | None
    synthetic_label: str | None

    def __init__(self, **_values: object) -> None:
        raise TypeError(
            "Use DatasetProvenance.synthetic(), DatasetProvenance.unverified(), "
            "or TrustedDatasetAdapter.verified_provenance()."
        )

    @classmethod
    def synthetic(
        cls,
        label: str,
        *,
        unit_id: str | None = None,
        unit_type: TextUnit | None = None,
    ) -> DatasetProvenance:
        _require_text(label, "synthetic_label")
        if (unit_id is None) != (unit_type is None):
            raise ValueError("synthetic unit_id and unit_type must be provided together.")
        if unit_id is not None:
            _require_text(unit_id, "synthetic unit_id")
            if unit_type not in {TextUnit.DOCUMENT, TextUnit.PASSAGE}:
                raise ValueError("synthetic context unit_type must be document or passage.")
        return cls._create(
            _token=_PROVENANCE_SEAL,
            status=ProvenanceStatus.SYNTHETIC,
            dataset_name=None,
            dataset_revision=None,
            split=None,
            record_id=None,
            unit_id=unit_id,
            unit_type=unit_type,
            source_location=None,
            reference_location=None,
            alignment_location=None,
            adapter_name=None,
            synthetic_label=label,
        )

    @classmethod
    def unverified(cls) -> DatasetProvenance:
        return cls._create(
            _token=_PROVENANCE_SEAL,
            status=ProvenanceStatus.UNVERIFIED,
            dataset_name=None,
            dataset_revision=None,
            split=None,
            record_id=None,
            unit_id=None,
            unit_type=None,
            source_location=None,
            reference_location=None,
            alignment_location=None,
            adapter_name=None,
            synthetic_label=None,
        )

    @classmethod
    def _from_adapter(
        cls,
        adapter: TrustedDatasetAdapter,
        *,
        status: ProvenanceStatus,
        dataset_name: str,
        dataset_revision: str,
        split: str,
        record_id: str,
        source_location: str,
        reference_location: str | None,
        alignment_location: str | None,
        unit_id: str | None,
        unit_type: TextUnit | None,
    ) -> DatasetProvenance:
        if not isinstance(adapter, TrustedDatasetAdapter):
            raise TypeError("Verified provenance must be issued by a TrustedDatasetAdapter.")
        if status not in {
            ProvenanceStatus.VERIFIED_DATASET,
            ProvenanceStatus.VERIFIED_ANNOTATION,
        }:
            raise ValueError("Trusted adapters may issue only verified provenance.")
        adapter_name = adapter.adapter_name
        for field_name, value in (
            ("adapter_name", adapter_name),
            ("dataset_name", dataset_name),
            ("dataset_revision", dataset_revision),
            ("split", split),
            ("record_id", record_id),
            ("source_location", source_location),
        ):
            _require_text(value, field_name)
        if (unit_id is None) != (unit_type is None):
            raise ValueError("unit_id and unit_type must be provided together.")
        if unit_id is not None:
            _require_text(unit_id, "unit_id")
            if unit_type not in {TextUnit.DOCUMENT, TextUnit.PASSAGE}:
                raise ValueError("unit_type must be document or passage.")
        return cls._create(
            _token=_PROVENANCE_SEAL,
            status=status,
            dataset_name=dataset_name,
            dataset_revision=dataset_revision,
            split=split,
            record_id=record_id,
            unit_id=unit_id,
            unit_type=unit_type,
            source_location=source_location,
            reference_location=reference_location,
            alignment_location=alignment_location,
            adapter_name=adapter_name,
            synthetic_label=None,
        )

    @classmethod
    def _create(cls, *, _token: object, **values: object) -> DatasetProvenance:
        if _token is not _PROVENANCE_SEAL:
            raise TypeError("Provenance construction is restricted to trusted factories.")
        instance = object.__new__(cls)
        for name in cls.__dataclass_fields__:
            object.__setattr__(instance, name, values[name])
        instance._validate()
        return instance

    def _validate(self) -> None:
        if not isinstance(self.status, ProvenanceStatus):
            raise ValueError("provenance status must be a ProvenanceStatus.")
        if self.status in {
            ProvenanceStatus.VERIFIED_DATASET,
            ProvenanceStatus.VERIFIED_ANNOTATION,
        }:
            for name in (
                "dataset_name",
                "dataset_revision",
                "split",
                "record_id",
                "source_location",
                "adapter_name",
            ):
                _require_text(getattr(self, name), name)
            if self.synthetic_label is not None:
                raise ValueError("verified provenance cannot have a synthetic label.")
            for name in ("reference_location", "alignment_location"):
                value = getattr(self, name)
                if value is not None:
                    _require_text(value, name)
        elif self.status is ProvenanceStatus.SYNTHETIC:
            if any(
                getattr(self, name) is not None
                for name in (
                    "dataset_name",
                    "dataset_revision",
                    "split",
                    "record_id",
                    "source_location",
                    "reference_location",
                    "alignment_location",
                    "adapter_name",
                )
            ):
                raise ValueError("synthetic provenance cannot claim dataset locations or IDs.")
            _require_text(self.synthetic_label, "synthetic_label")
        else:
            if any(
                getattr(self, name) is not None
                for name in (
                    "dataset_name",
                    "dataset_revision",
                    "split",
                    "record_id",
                    "unit_id",
                    "unit_type",
                    "source_location",
                    "reference_location",
                    "alignment_location",
                    "adapter_name",
                    "synthetic_label",
                )
            ):
                raise ValueError("unverified provenance cannot claim dataset metadata.")

    @property
    def is_verified(self) -> bool:
        return self.status in {
            ProvenanceStatus.VERIFIED_DATASET,
            ProvenanceStatus.VERIFIED_ANNOTATION,
        }

    def same_context_source(self, other: DatasetProvenance) -> bool:
        if self.status is ProvenanceStatus.SYNTHETIC or other.status is ProvenanceStatus.SYNTHETIC:
            return (
                self.status is other.status is ProvenanceStatus.SYNTHETIC
                and self.synthetic_label == other.synthetic_label
            )
        return (
            self.is_verified
            and other.is_verified
            and self.dataset_name == other.dataset_name
            and self.dataset_revision == other.dataset_revision
            and self.split == other.split
            and self.adapter_name == other.adapter_name
        )


class TrustedDatasetAdapter(ABC):
    """Trusted loader boundary responsible for grounding provenance in loaded rows."""

    @property
    @abstractmethod
    def adapter_name(self) -> str:
        """Stable implementation name for audit records."""

    def verified_provenance(
        self,
        *,
        status: ProvenanceStatus,
        dataset_name: str,
        dataset_revision: str,
        split: str,
        record_id: str,
        source_location: str,
        reference_location: str | None = None,
        alignment_location: str | None = None,
        unit_id: str | None = None,
        unit_type: TextUnit | None = None,
    ) -> DatasetProvenance:
        return DatasetProvenance._from_adapter(
            self,
            status=status,
            dataset_name=dataset_name,
            dataset_revision=dataset_revision,
            split=split,
            record_id=record_id,
            source_location=source_location,
            reference_location=reference_location,
            alignment_location=alignment_location,
            unit_id=unit_id,
            unit_type=unit_type,
        )


@dataclass(frozen=True)
class AlignmentEvidence:
    kind: AlignmentEvidenceKind

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AlignmentEvidenceKind):
            raise ValueError("Alignment evidence kind must be an AlignmentEvidenceKind.")


@dataclass(frozen=True)
class PriorSourceText:
    """A prior source unit with provenance and a local order, not a sentence ID."""

    document_or_passage_id: str
    source_order: int
    text: str
    provenance: DatasetProvenance

    def __post_init__(self) -> None:
        _require_text(self.document_or_passage_id, "context document_or_passage_id")
        _require_order(self.source_order, "context source_order")
        _require_text(self.text, "context text")
        if not isinstance(self.provenance, DatasetProvenance):
            raise ValueError("context requires structured DatasetProvenance.")
        if self.provenance.unit_id != self.document_or_passage_id:
            raise ValueError("context unit ID must match its provenance unit ID.")
        if self.provenance.status is ProvenanceStatus.UNVERIFIED:
            raise ValueError("unverified provenance cannot establish context membership.")


@dataclass(frozen=True)
class EvaluationRecord:
    track: EvaluationTrack
    source_language: str
    target_language: str
    current_text: str
    current_text_unit: TextUnit
    alignment_status: AlignmentStatus
    reference_availability: ReferenceAvailability
    provenance: DatasetProvenance
    reference_translation: str | None = None
    reference_unit: TextUnit | None = None
    alignment_evidence: AlignmentEvidence | None = None
    document_or_passage_id: str | None = None
    current_unit_position: int | None = None
    preceding_context: tuple[PriorSourceText, ...] = ()
    context_status: ContextStatus = ContextStatus.UNAVAILABLE
    context_strategy: ContextStrategy | None = None
    context_window: int = 0
    context_supplied_to_model: bool = False
    baseline_translation: str | None = None
    context_conditioned_translation: str | None = None
    exclusion_reason: str | None = None
    qualitative_observation: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.track, EvaluationTrack):
            raise ValueError("track must be an EvaluationTrack.")
        if not isinstance(self.current_text_unit, TextUnit):
            raise ValueError("current_text_unit must be a TextUnit.")
        if not isinstance(self.alignment_status, AlignmentStatus):
            raise ValueError("alignment_status must be an AlignmentStatus.")
        if not isinstance(self.reference_availability, ReferenceAvailability):
            raise ValueError("reference_availability must be a ReferenceAvailability.")
        if not isinstance(self.context_status, ContextStatus):
            raise ValueError("context_status must be a ContextStatus.")
        if self.context_strategy is not None and not isinstance(self.context_strategy, ContextStrategy):
            raise ValueError("context_strategy must be a ContextStrategy.")
        _require_text(self.source_language, "source_language")
        _require_text(self.target_language, "target_language")
        _require_text(self.current_text, "current_text")
        if not isinstance(self.provenance, DatasetProvenance):
            raise ValueError("record requires structured DatasetProvenance.")
        if isinstance(self.context_window, bool) or not isinstance(self.context_window, int):
            raise ValueError("context_window must be an integer.")
        if self.context_window < 0 or self.context_window > 200:
            raise ValueError("context_window must be between 0 and 200.")
        if self.current_unit_position is not None:
            _require_order(self.current_unit_position, "current_unit_position")
        if not isinstance(self.context_supplied_to_model, bool):
            raise ValueError("context_supplied_to_model must be a boolean.")
        if not isinstance(self.preceding_context, tuple):
            raise ValueError("preceding_context must be a tuple to preserve immutable order.")
        if any(not isinstance(item, PriorSourceText) for item in self.preceding_context):
            raise ValueError("preceding_context entries must be PriorSourceText records.")

        self._validate_track_contract()
        self._validate_alignment()
        self._validate_reference()
        self._validate_context()

        for name, value in (
            ("baseline_translation", self.baseline_translation),
            ("context_conditioned_translation", self.context_conditioned_translation),
        ):
            if value is not None:
                _require_text(value, name)
        if self.context_conditioned_translation is not None and not self.context_supplied_to_model:
            raise ValueError("A context-conditioned translation requires context supplied to the model.")
        if self.exclusion_reason is not None:
            _require_text(self.exclusion_reason, "exclusion_reason")
        if self.qualitative_observation is not None:
            _require_text(self.qualitative_observation, "qualitative_observation")

    def _validate_track_contract(self) -> None:
        if self.track is EvaluationTrack.SENTENCE_LEVEL:
            if self.current_text_unit is not TextUnit.SENTENCE:
                raise ValueError("sentence_level requires a sentence current_text_unit.")
            if self.document_or_passage_id is not None or self.provenance.unit_id is not None:
                raise ValueError("sentence_level cannot infer document or passage identity.")
            if self.context_status is not ContextStatus.UNAVAILABLE or self.preceding_context:
                raise ValueError("sentence_level does not accept document/passage context.")
            if self.context_strategy is not None or self.context_window != 0:
                raise ValueError("sentence_level cannot declare a document context strategy/window.")
            if self.context_supplied_to_model or self.context_conditioned_translation is not None:
                raise ValueError("sentence_level is the no-context baseline track.")
            if (
                self.reference_availability is ReferenceAvailability.AVAILABLE
                and self.reference_unit is not TextUnit.SENTENCE
            ):
                raise ValueError("sentence_level references must be sentence-level.")
            if self.provenance.unit_type is not None:
                raise ValueError("sentence_level provenance cannot claim document/passage identity.")
            return

        expected_unit_type = (
            TextUnit.DOCUMENT
            if self.track is EvaluationTrack.DOCUMENT_CONTEXT
            else TextUnit.PASSAGE
        )
        allowed_current_units = (
            {TextUnit.SENTENCE, TextUnit.DOCUMENT}
            if self.track is EvaluationTrack.DOCUMENT_CONTEXT
            else {TextUnit.SENTENCE, TextUnit.PASSAGE}
        )
        if self.current_text_unit not in allowed_current_units:
            raise ValueError(f"{self.track.value} current_text_unit is incompatible with its track.")
        _require_text(self.document_or_passage_id, "document_or_passage_id")
        if (
            self.provenance.unit_id != self.document_or_passage_id
            or self.provenance.unit_type is not expected_unit_type
        ):
            raise ValueError("track identity must match the structured provenance unit ID and type.")
        if self.provenance.status is ProvenanceStatus.UNVERIFIED:
            raise ValueError("unverified identity cannot establish document/passage track membership.")
        allowed_reference_units = (
            {TextUnit.SENTENCE, TextUnit.DOCUMENT}
            if self.track is EvaluationTrack.DOCUMENT_CONTEXT
            else {TextUnit.SENTENCE, TextUnit.PASSAGE}
        )
        if (
            self.reference_availability is ReferenceAvailability.AVAILABLE
            and self.reference_unit not in allowed_reference_units
        ):
            raise ValueError(f"{self.track.value} reference_unit is incompatible with its track.")
        if (
            self.reference_availability is ReferenceAvailability.AVAILABLE
            and self.reference_unit is not self.current_text_unit
        ):
            raise ValueError("available references must match the current text granularity.")
        if self.current_text_unit is TextUnit.SENTENCE and self.current_unit_position is None:
            raise ValueError("sentence records in context tracks require a current_unit_position.")
        if self.current_text_unit is not TextUnit.SENTENCE and self.context_status is not ContextStatus.UNAVAILABLE:
            raise ValueError("context prefixes require a sentence-level current input.")

    def _validate_alignment(self) -> None:
        if self.alignment_status is AlignmentStatus.UNVERIFIED:
            if self.alignment_evidence is not None:
                raise ValueError("unverified alignment cannot carry alignment evidence.")
            return
        if not isinstance(self.alignment_evidence, AlignmentEvidence):
            raise ValueError("paired/aligned records require typed alignment evidence.")
        expected_status, expected_provenance = _EVIDENCE_REQUIREMENTS[self.alignment_evidence.kind]
        if expected_status is not None and self.alignment_status is not expected_status:
            raise ValueError("alignment evidence kind does not support alignment_status.")
        if self.provenance.status is not expected_provenance:
            raise ValueError("alignment evidence kind conflicts with provenance status.")
        if self.provenance.is_verified and self.provenance.alignment_location is None:
            raise ValueError("verified alignment requires an alignment_location.")

    def _validate_reference(self) -> None:
        if self.reference_availability is ReferenceAvailability.UNAVAILABLE:
            if self.reference_translation is not None or self.reference_unit is not None:
                raise ValueError("unavailable references must not contain reference text or unit.")
            return
        if not isinstance(self.reference_unit, TextUnit):
            raise ValueError("available references require a reference_unit.")
        _require_text(self.reference_translation, "reference_translation")
        if self.alignment_status is not _UNIT_ALIGNMENT[self.reference_unit]:
            raise ValueError("reference_unit and alignment_status do not describe the same alignment scope.")
        if self.alignment_evidence is None:
            raise ValueError("available references require alignment evidence.")
        if self.provenance.is_verified and self.provenance.reference_location is None:
            raise ValueError("verified references require a reference_location.")

    def _validate_context(self) -> None:
        if self.context_status is ContextStatus.UNAVAILABLE:
            if self.preceding_context or self.context_supplied_to_model:
                raise ValueError("unavailable context cannot contain or supply context items.")
            if self.context_strategy is None and self.context_window != 0:
                raise ValueError("context_window requires a declared context_strategy.")
            if self.context_strategy is ContextStrategy.LAST_N and self.context_window < 1:
                raise ValueError("last_n requires a positive context_window.")
            return

        if not self.preceding_context:
            raise ValueError("available context status requires preceding source text.")
        if self.track is EvaluationTrack.SENTENCE_LEVEL:
            raise ValueError("sentence_level cannot have available context.")
        if not self.preceding_context:
            raise ValueError("available context status requires preceding source text.")
        if self.current_text_unit is not TextUnit.SENTENCE:
            raise ValueError("context prefixes require current_text_unit=sentence.")
        if self.current_unit_position is None:
            raise ValueError("context requires a current_unit_position.")
        if self.context_strategy is not ContextStrategy.LAST_N:
            raise ValueError("available context requires the explicit last_n strategy.")
        if self.context_window < 1:
            raise ValueError("last_n requires a positive context_window.")
        if self.provenance.status is ProvenanceStatus.UNVERIFIED:
            raise ValueError("unverified IDs cannot establish context membership.")

        expected_unit_type = (
            TextUnit.DOCUMENT
            if self.track is EvaluationTrack.DOCUMENT_CONTEXT
            else TextUnit.PASSAGE
        )
        actual_positions = tuple(item.source_order for item in self.preceding_context)
        if len(set(actual_positions)) != len(actual_positions):
            raise ValueError("context source positions must be unique.")
        if actual_positions != tuple(sorted(actual_positions)):
            raise ValueError("context positions must be unique and in source order.")

        seen_record_ids: set[str] = set()
        for item in self.preceding_context:
            context_provenance = item.provenance
            if item.document_or_passage_id != self.document_or_passage_id:
                raise ValueError("context crosses a document/passage boundary.")
            if item.source_order >= self.current_unit_position:
                raise ValueError("future or current source context is not allowed.")
            if context_provenance.unit_type is not expected_unit_type:
                raise ValueError("context provenance has a different unit type.")
            if not self.provenance.same_context_source(context_provenance):
                raise ValueError("context provenance does not match the current source dataset/unit.")
            if context_provenance.unit_id != self.document_or_passage_id:
                raise ValueError("context provenance unit ID does not match the current unit.")
            if context_provenance.is_verified:
                if context_provenance.source_location is None or context_provenance.record_id is None:
                    raise ValueError("verified context requires a record ID and source_location.")
                if context_provenance.record_id in seen_record_ids:
                    raise ValueError("context record IDs must be unique.")
                seen_record_ids.add(context_provenance.record_id)

        expected_positions = tuple(
            range(max(0, self.current_unit_position - self.context_window), self.current_unit_position)
        )
        if actual_positions != expected_positions:
            raise ValueError("last_n context must be the complete eligible preceding window in source order.")

        if self.context_status is ContextStatus.AVAILABLE_NOT_USED:
            if self.context_supplied_to_model:
                raise ValueError("available_not_used context must not be marked supplied.")
        elif self.context_status is ContextStatus.SUPPLIED:
            if not self.context_supplied_to_model:
                raise ValueError("supplied context status requires context_supplied_to_model=True.")


@dataclass(frozen=True)
class MetricEligibility:
    condition: ComparisonCondition
    metric: MetricName
    eligible: bool
    reason: str | None


@dataclass(frozen=True)
class AuditedEvaluationItem:
    record_index: int
    source_language: str
    target_language: str
    track: EvaluationTrack
    provenance: DatasetProvenance
    document_or_passage_id: str | None
    current_unit_position: int | None
    current_text_unit: TextUnit
    current_text: str
    reference_translation: str | None
    reference_availability: ReferenceAvailability
    reference_unit: TextUnit | None
    alignment_status: AlignmentStatus
    alignment_evidence: AlignmentEvidence | None
    preceding_context: tuple[PriorSourceText, ...]
    context_strategy: ContextStrategy | None
    context_window: int
    context_status: ContextStatus
    context_supplied_to_model: bool
    baseline_translation: str | None
    context_conditioned_translation: str | None
    metric_eligibility: tuple[MetricEligibility, ...]
    exclusion_reason: str | None
    qualitative_observation: str | None


@dataclass(frozen=True)
class QualitativeObservation:
    record_index: int
    observation: str


@dataclass(frozen=True)
class ExcludedRecord:
    record_index: int
    reason: str


@dataclass(frozen=True)
class TrackReport:
    track: EvaluationTrack
    records: tuple[AuditedEvaluationItem, ...]
    qualitative_observations: tuple[QualitativeObservation, ...]
    excluded_records: tuple[ExcludedRecord, ...]


@dataclass(frozen=True)
class EvaluationReport:
    tracks: tuple[TrackReport, ...]

    def to_dict(self) -> dict[str, object]:
        payload = _json_value(asdict(self))
        if not isinstance(payload, dict):
            raise TypeError("Evaluation report serialization did not produce a JSON object.")
        return payload


def metric_eligibility(
    record: EvaluationRecord,
    metric: MetricName,
    *,
    condition: ComparisonCondition = ComparisonCondition.BASELINE,
) -> tuple[bool, str | None]:
    """Return a score gate only; this function never calculates a metric."""
    _require_enum(metric, MetricName, "metric")
    _require_enum(condition, ComparisonCondition, "condition")
    if record.exclusion_reason is not None:
        return False, f"excluded: {record.exclusion_reason}"
    if record.reference_availability is ReferenceAvailability.UNAVAILABLE:
        return False, "reference unavailable"
    if not record.provenance.is_verified:
        return False, f"provenance is {record.provenance.status.value}"

    prediction = (
        record.baseline_translation
        if condition is ComparisonCondition.BASELINE
        else record.context_conditioned_translation
    )
    if prediction is None:
        return False, f"{condition.value} translation unavailable"

    if metric in {MetricName.SENTENCE_BLEU, MetricName.SENTENCE_CHRF}:
        expected_unit = TextUnit.SENTENCE
    elif metric in {MetricName.DOCUMENT_BLEU, MetricName.DOCUMENT_CHRF}:
        expected_unit = TextUnit.DOCUMENT
    elif metric in {MetricName.PASSAGE_BLEU, MetricName.PASSAGE_CHRF}:
        expected_unit = TextUnit.PASSAGE
    else:
        expected_unit = record.current_text_unit

    if record.current_text_unit is not expected_unit or record.reference_unit is not expected_unit:
        return False, "prediction and reference granularity do not match the requested metric"
    if record.alignment_status is not _UNIT_ALIGNMENT[expected_unit]:
        return False, "alignment evidence does not support the requested metric granularity"
    if record.provenance.reference_location is None or record.provenance.alignment_location is None:
        return False, "verified reference/alignment locations are incomplete"
    return True, None


def build_comparison_report(records: Iterable[EvaluationRecord]) -> EvaluationReport:
    """Build deterministic, independently auditable reports without scoring."""
    indexed_records = list(enumerate(records))
    if any(not isinstance(record, EvaluationRecord) for _, record in indexed_records):
        raise TypeError("records must contain only EvaluationRecord values.")

    seen_positions: set[tuple[object, ...]] = set()
    seen_record_ids: set[tuple[object, ...]] = set()
    for _, record in indexed_records:
        provenance = record.provenance
        if provenance.is_verified:
            source_key = (
                provenance.dataset_name,
                provenance.dataset_revision,
                provenance.split,
                provenance.record_id,
            )
            if source_key in seen_record_ids:
                raise ValueError("duplicate verified source record IDs are not allowed.")
            seen_record_ids.add(source_key)
        if record.document_or_passage_id is not None and record.current_unit_position is not None:
            position_key = (
                record.track,
                provenance.dataset_name,
                provenance.dataset_revision,
                provenance.split,
                provenance.synthetic_label,
                record.document_or_passage_id,
                record.current_unit_position,
            )
            if position_key in seen_positions:
                raise ValueError("duplicate current positions within a document/passage are not allowed.")
            seen_positions.add(position_key)

    def order_key(item: tuple[int, EvaluationRecord]) -> tuple[object, ...]:
        input_index, record = item
        provenance = record.provenance
        return (
            record.track.value,
            provenance.dataset_name or "",
            provenance.dataset_revision or "",
            provenance.split or "",
            record.document_or_passage_id or "",
            record.current_unit_position if record.current_unit_position is not None else -1,
            provenance.record_id or "",
            input_index,
        )

    ordered = sorted(indexed_records, key=order_key)
    track_reports: list[TrackReport] = []
    for track in EvaluationTrack:
        selected = [(index, record) for index, record in ordered if record.track is track]
        items: list[AuditedEvaluationItem] = []
        qualitative: list[QualitativeObservation] = []
        excluded: list[ExcludedRecord] = []

        for index, record in selected:
            eligibility = tuple(
                MetricEligibility(
                    condition,
                    metric,
                    *metric_eligibility(record, metric, condition=condition),
                )
                for condition in ComparisonCondition
                for metric in MetricName
            )
            items.append(
                AuditedEvaluationItem(
                    record_index=index,
                    source_language=record.source_language,
                    target_language=record.target_language,
                    track=record.track,
                    provenance=record.provenance,
                    document_or_passage_id=record.document_or_passage_id,
                    current_unit_position=record.current_unit_position,
                    current_text_unit=record.current_text_unit,
                    current_text=record.current_text,
                    reference_translation=record.reference_translation,
                    reference_availability=record.reference_availability,
                    reference_unit=record.reference_unit,
                    alignment_status=record.alignment_status,
                    alignment_evidence=record.alignment_evidence,
                    preceding_context=record.preceding_context,
                    context_strategy=record.context_strategy,
                    context_window=record.context_window,
                    context_status=record.context_status,
                    context_supplied_to_model=record.context_supplied_to_model,
                    baseline_translation=record.baseline_translation,
                    context_conditioned_translation=record.context_conditioned_translation,
                    metric_eligibility=eligibility,
                    exclusion_reason=record.exclusion_reason,
                    qualitative_observation=record.qualitative_observation,
                )
            )

            if record.exclusion_reason is not None:
                excluded.append(ExcludedRecord(index, record.exclusion_reason))
            elif not any(item.eligible for item in eligibility):
                if record.reference_availability is ReferenceAvailability.UNAVAILABLE:
                    default_observation = "Reference unavailable; report qualitative observations only."
                elif not record.provenance.is_verified:
                    default_observation = "Provenance is not verified; research metrics are suppressed."
                else:
                    default_observation = "No reference metric is eligible at this granularity."
                qualitative.append(
                    QualitativeObservation(
                        index,
                        record.qualitative_observation or default_observation,
                    )
                )

        track_reports.append(
            TrackReport(
                track=track,
                records=tuple(items),
                qualitative_observations=tuple(qualitative),
                excluded_records=tuple(excluded),
            )
        )

    return EvaluationReport(tuple(track_reports))


def _require_enum(value: object, enum_type: type[Enum], field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise ValueError(f"{field_name} must be a {enum_type.__name__}.")


def _require_text(value: str | None, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _require_order(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value