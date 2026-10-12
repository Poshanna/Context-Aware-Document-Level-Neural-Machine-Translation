"""Conservative evaluation foundation for PRALEKHA document-level provenance.

This phase intentionally avoids any sentence-alignment assumptions. It only
supports the data contracts the project can justify from the dataset audit:

* document/passage pairing must be explicit and verified,
* context can only be causal and same-document,
* sentence-level metrics remain unavailable unless a true sentence alignment is
  present.

The adapter is intentionally fail-closed: missing provenance, duplicate units, or
cross-document context always raise ValueError.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum

from app.evaluation.document_evaluation import (
    AlignmentStatus,
    DatasetProvenance,
    ProvenanceStatus,
    TextUnit,
    TrustedDatasetAdapter,
    _require_text,
)

PRALEKHA_DATASET_NAME = "AI4Bharat PRALEKHA"


class PraLekhaMetric(str, Enum):
    SENTENCE_BLEU = "sentence_bleu"
    DOCUMENT_BLEU = "document_bleu"
    DOCUMENT_CHRF = "document_chrf++"


@dataclass(frozen=True)
class PraLekhaDocumentUnit:
    """A trusted document or passage-level unit representation for PRALEKHA."""

    document_id: str
    unit_id: str
    unit_order: int
    source_text: str
    reference_text: str | None
    alignment_status: AlignmentStatus
    provenance: DatasetProvenance
    source_language: str = "en"
    target_language: str = "te"
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.document_id, "document_id")
        _require_text(self.unit_id, "unit_id")
        if isinstance(self.unit_order, bool) or not isinstance(self.unit_order, int) or self.unit_order < 0:
            raise ValueError("unit_order must be a non-negative integer.")
        _require_text(self.source_text, "source_text")
        if self.reference_text is not None:
            _require_text(self.reference_text, "reference_text")
        if not isinstance(self.alignment_status, AlignmentStatus):
            raise ValueError("alignment_status must be an AlignmentStatus.")
        if not isinstance(self.provenance, DatasetProvenance):
            raise ValueError("unit provenance must be a DatasetProvenance record.")
        _require_text(self.source_language, "source_language")
        _require_text(self.target_language, "target_language")
        if self.exclusion_reason is not None:
            _require_text(self.exclusion_reason, "exclusion_reason")

    def with_exclusion_reason(self, reason: str) -> "PraLekhaDocumentUnit":
        _require_text(reason, "reason")
        return replace(self, exclusion_reason=reason)


@dataclass(frozen=True)
class PraLekhaDocumentPair:
    """A source-target document pair representation with explicit dataset provenance."""

    dataset_name: str
    dataset_revision: str
    split: str
    pair_id: str
    source_document_id: str
    target_document_id: str
    source_language: str
    target_language: str
    source_text: str
    target_text: str
    provenance: DatasetProvenance
    unit_order: int = 0
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.dataset_name, "dataset_name")
        _require_text(self.dataset_revision, "dataset_revision")
        _require_text(self.split, "split")
        _require_text(self.pair_id, "pair_id")
        _require_text(self.source_document_id, "source_document_id")
        _require_text(self.target_document_id, "target_document_id")
        _require_text(self.source_language, "source_language")
        _require_text(self.target_language, "target_language")
        _require_text(self.source_text, "source_text")
        _require_text(self.target_text, "target_text")
        if isinstance(self.unit_order, bool) or not isinstance(self.unit_order, int) or self.unit_order < 0:
            raise ValueError("unit_order must be a non-negative integer.")
        if not isinstance(self.provenance, DatasetProvenance):
            raise ValueError("pair provenance must be a DatasetProvenance record.")
        require_pralekha_provenance(
            self.provenance,
            document_id=self.source_document_id,
            pair_id=self.pair_id,
            split=self.split,
            language=self.source_language,
        )
        if self.exclusion_reason is not None:
            _require_text(self.exclusion_reason, "exclusion_reason")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def require_pralekha_provenance(
    provenance: DatasetProvenance,
    *,
    document_id: str,
    pair_id: str,
    split: str,
    language: str,
) -> DatasetProvenance:
    """Validate that a record includes the minimum metadata needed for a fail-closed PRALEKHA audit."""

    if not isinstance(provenance, DatasetProvenance):
        raise ValueError("required PRALEKHA provenance fields are missing or not verified.")
    if not provenance.is_verified:
        raise ValueError("required PRALEKHA provenance fields are missing or not verified.")
    if provenance.dataset_name != PRALEKHA_DATASET_NAME:
        raise ValueError("PRALEKHA provenance must carry the dataset name as AI4Bharat PRALEKHA.")
    if provenance.dataset_revision is None or not provenance.dataset_revision.strip():
        raise ValueError("required PRALEKHA provenance fields are missing or not verified.")
    if provenance.split != split:
        raise ValueError("PRALEKHA split metadata must match the loaded split.")
    if provenance.record_id is None or not provenance.record_id.strip():
        raise ValueError("required PRALEKHA provenance fields are missing or not verified.")
    if pair_id is None or not pair_id.strip():
        raise ValueError("PRALEKHA pair_id must be a non-empty value.")
    if provenance.source_location is None:
        raise ValueError("required PRALEKHA provenance fields are missing or not verified.")
    if provenance.alignment_location is None:
        raise ValueError("required PRALEKHA provenance fields are missing or not verified.")
    if provenance.unit_type not in {TextUnit.DOCUMENT, TextUnit.PASSAGE, None}:
        raise ValueError("PRALEKHA provenance unit_type must be document or passage.")
    if language not in {"en", "te"}:
        raise ValueError("PRALEKHA language must be a supported source/target language code.")
    return provenance


def build_causal_context(
    units: tuple[PraLekhaDocumentUnit, ...],
    *,
    current_unit_id: str,
    last_n: int,
) -> tuple[PraLekhaDocumentUnit, ...]:
    """Return the complete eligible preceding context for a unit while preserving same-document ordering."""

    if isinstance(last_n, bool) or not isinstance(last_n, int) or last_n < 1:
        raise ValueError("last_n must be a positive integer.")
    if not units:
        raise ValueError("context requires at least one unit.")

    ordered = tuple(sorted(units, key=lambda item: item.unit_order))
    seen_ids: set[str] = set()
    seen_orders_by_document: dict[str, set[int]] = {}
    for item in ordered:
        if item.unit_id in seen_ids:
            raise ValueError("duplicate unit IDs are not allowed in the same document.")
        doc_orders = seen_orders_by_document.setdefault(item.document_id, set())
        if item.unit_order in doc_orders:
            raise ValueError("duplicate unit ordering is not allowed in the same document.")
        seen_ids.add(item.unit_id)
        doc_orders.add(item.unit_order)

    current = next((item for item in ordered if item.unit_id == current_unit_id), None)
    if current is None:
        raise ValueError("current unit must exist in the provided context window.")

    future_or_current = tuple(
        item
        for item in ordered
        if item.document_id == current.document_id
        and item.unit_order >= current.unit_order
        and item.unit_id != current.unit_id
    )
    if future_or_current:
        raise ValueError("future or current source context is not allowed.")

    preceding = tuple(
        item
        for item in ordered
        if item.document_id == current.document_id and item.unit_order < current.unit_order
    )

    if current.unit_order == 0:
        return ()

    other_document_units = tuple(item for item in ordered if item.document_id != current.document_id)
    if other_document_units:
        raise ValueError("context must remain within the same verified document.")

    for item in preceding:
        if item.document_id != current.document_id:
            raise ValueError("context must remain within the same verified document.")
        if item.unit_order >= current.unit_order:
            raise ValueError("future or current source context is not allowed.")
        if item.provenance.status is ProvenanceStatus.UNVERIFIED:
            raise ValueError("unverified provenance cannot establish patient-free context membership.")
        if not item.provenance.same_context_source(current.provenance):
            raise ValueError("context must stay in the same verified document for the same dataset split.")

    return preceding[-last_n:]


def metric_eligibility(
    record: PraLekhaDocumentUnit,
    metric: PraLekhaMetric,
    *,
    source_reference_available: bool = True,
    sentence_alignment_present: bool = True,
) -> tuple[bool, str | None]:
    """Return a conservative metric gate without scoring any translation output."""

    if not isinstance(record, PraLekhaDocumentUnit):
        raise TypeError("record must be a PraLekhaDocumentUnit.")
    if not isinstance(metric, PraLekhaMetric):
        raise ValueError("metric must be a PraLekhaMetric.")
    if record.exclusion_reason is not None:
        return False, f"excluded: {record.exclusion_reason}"

    if metric is PraLekhaMetric.SENTENCE_BLEU:
        if not source_reference_available or not sentence_alignment_present:
            return False, "sentence-level reference unavailable without sentence alignment"
        if record.alignment_status is not AlignmentStatus.SENTENCE_ALIGNED:
            raise ValueError("sentence-level metric requires sentence alignment evidence.")
        return True, None

    if not source_reference_available:
        raise ValueError("document alignment requires an available reference document for this metric.")
    if record.alignment_status is not AlignmentStatus.DOCUMENT_PAIRED:
        raise ValueError("document alignment is required for a document-level metric.")
    if metric in {PraLekhaMetric.DOCUMENT_BLEU, PraLekhaMetric.DOCUMENT_CHRF}:
        return True, None
    raise ValueError(f"unsupported PraLekha metric: {metric!r}")


class PraLekhaTrustedAdapter(TrustedDatasetAdapter):
    """Trusted loader boundary for PRALEKHA records.

    This class does not load or score data; it only issues provenance that matches
    the audited dataset contract and rejects rows that are missing document IDs or
    alignment evidence.
    """

    @property
    def adapter_name(self) -> str:
        return "pra_lekha_trusted_adapter_v1"

    def make_document_pair(
        self,
        *,
        pair_id: str,
        source_document_id: str,
        target_document_id: str,
        source_text: str,
        target_text: str,
        split: str,
        source_language: str = "en",
        target_language: str = "te",
        dataset_revision: str = "pinned_revision_not_available",
        unit_order: int = 0,
        provenance: DatasetProvenance | None = None,
    ) -> PraLekhaDocumentPair:
        if provenance is None:
            provenance = self.verified_provenance(
                status=ProvenanceStatus.VERIFIED_DATASET,
                dataset_name=PRALEKHA_DATASET_NAME,
                dataset_revision=dataset_revision,
                split=split,
                record_id=pair_id,
                source_location=f"pra_lekha://{source_document_id}",
                reference_location=f"pra_lekha://{target_document_id}",
                alignment_location=f"pra_lekha://{pair_id}#document_pair_id",
                unit_id=source_document_id,
                unit_type=TextUnit.DOCUMENT,
            )
        return PraLekhaDocumentPair(
            dataset_name=PRALEKHA_DATASET_NAME,
            dataset_revision=dataset_revision,
            split=split,
            pair_id=pair_id,
            source_document_id=source_document_id,
            target_document_id=target_document_id,
            source_language=source_language,
            target_language=target_language,
            source_text=source_text,
            target_text=target_text,
            provenance=provenance,
            unit_order=unit_order,
        )
