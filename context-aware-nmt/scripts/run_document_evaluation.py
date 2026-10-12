"""Print a synthetic Phase 11 comparison report without loading data or models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.evaluation.document_evaluation import (
    AlignmentEvidence,
    AlignmentEvidenceKind,
    AlignmentStatus,
    ContextStrategy,
    ContextStatus,
    DatasetProvenance,
    EvaluationRecord,
    EvaluationTrack,
    PriorSourceText,
    ReferenceAvailability,
    TextUnit,
    build_comparison_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print a built-in synthetic evaluation report. This CLI does not load "
            "datasets, run NLLB, or compute metrics."
        )
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run only the built-in synthetic demonstration.",
    )
    return parser


def build_demo_report() -> dict[str, object]:
    sentence = EvaluationRecord(
        track=EvaluationTrack.SENTENCE_LEVEL,
        source_language="en",
        target_language="te",
        current_text="Synthetic source sentence.",
        current_text_unit=TextUnit.SENTENCE,
        alignment_status=AlignmentStatus.SENTENCE_ALIGNED,
        reference_availability=ReferenceAvailability.AVAILABLE,
        provenance=DatasetProvenance.synthetic("built-in sentence demo"),
        reference_translation="కృత్రిమ మూల వాక్యం.",
        reference_unit=TextUnit.SENTENCE,
        alignment_evidence=AlignmentEvidence(AlignmentEvidenceKind.SYNTHETIC_FIXTURE),
        baseline_translation="synthetic baseline output",
    )
    unit_id = "synthetic-document-1"
    document_provenance = DatasetProvenance.synthetic(
        "built-in document demo", unit_id=unit_id, unit_type=TextUnit.DOCUMENT
    )
    context_record = EvaluationRecord(
        track=EvaluationTrack.DOCUMENT_CONTEXT,
        source_language="en",
        target_language="te",
        current_text="Synthetic current sentence in a document.",
        current_text_unit=TextUnit.SENTENCE,
        alignment_status=AlignmentStatus.DOCUMENT_PAIRED,
        reference_availability=ReferenceAvailability.UNAVAILABLE,
        provenance=document_provenance,
        alignment_evidence=AlignmentEvidence(AlignmentEvidenceKind.SYNTHETIC_FIXTURE),
        document_or_passage_id=unit_id,
        current_unit_position=1,
        preceding_context=(
            PriorSourceText(
                unit_id,
                0,
                "Synthetic prior sentence.",
                DatasetProvenance.synthetic(
                    "built-in document demo", unit_id=unit_id, unit_type=TextUnit.DOCUMENT
                ),
            ),
        ),
        context_status=ContextStatus.SUPPLIED,
        context_strategy=ContextStrategy.LAST_N,
        context_window=1,
        context_supplied_to_model=True,
        baseline_translation="synthetic sentence-only output",
        context_conditioned_translation="synthetic prefixed-context output",
        qualitative_observation="Synthetic example; no quality conclusion is implied.",
    )
    passage_id = "synthetic-passage-1"
    passage_provenance = DatasetProvenance.synthetic(
        "built-in passage demo", unit_id=passage_id, unit_type=TextUnit.PASSAGE
    )
    passage_record = EvaluationRecord(
        track=EvaluationTrack.PASSAGE_CONTEXT,
        source_language="en",
        target_language="te",
        current_text="Synthetic complete passage.",
        current_text_unit=TextUnit.PASSAGE,
        alignment_status=AlignmentStatus.PASSAGE_PAIRED,
        reference_availability=ReferenceAvailability.AVAILABLE,
        provenance=passage_provenance,
        reference_translation="కృత్రిమ పూర్తి భాగం.",
        reference_unit=TextUnit.PASSAGE,
        alignment_evidence=AlignmentEvidence(AlignmentEvidenceKind.SYNTHETIC_FIXTURE),
        document_or_passage_id=passage_id,
        baseline_translation="synthetic passage output",
    )
    return build_comparison_report([sentence, context_record, passage_record]).to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.demo:
        parser.error("This CLI is synthetic-only; pass --demo. Dataset inputs are not supported.")
    print(json.dumps(build_demo_report(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())