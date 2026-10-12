from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from app.evaluation.document_evaluation import (
    AlignmentEvidence,
    AlignmentEvidenceKind,
    AlignmentStatus,
    ContextStatus,
    DatasetProvenance,
    EvaluationRecord,
    EvaluationTrack,
    ProvenanceStatus,
    ReferenceAvailability,
    TextUnit,
    TrustedDatasetAdapter,
)
from app.evaluation.experiment_runner import load_reproducible_sample
from app.evaluation.samanantar_adapter import SamanantarSample, SamanantarSplit
from run_samanantar_baseline import (
    bridge_samanantar_records,
    run_bounded_samanantar_baseline,
)

UPSTREAM_REVISION = "ead34c5b22a0354d350e7dfdd6aece7df5a48f37"


class _SyntheticProvenanceAdapter(TrustedDatasetAdapter):
    @property
    def adapter_name(self) -> str:
        return "samanantar_prepared_jsonl_v1"


def _sample(
    _fixture_root: Path | None = None,
    *,
    ids: tuple[int, ...] = (31,),
    source_texts: tuple[str, ...] | None = None,
    reference_texts: tuple[str, ...] | None = None,
) -> SamanantarSample:
    source_texts = source_texts or tuple(f"source {record_id}" for record_id in ids)
    reference_texts = reference_texts or tuple(f"reference {record_id}" for record_id in ids)
    source_file = "synthetic-fixture://test.en-te.jsonl"
    adapter = _SyntheticProvenanceAdapter()
    records = []
    for index, (record_id, source, reference) in enumerate(
        zip(ids, source_texts, reference_texts), start=1
    ):
        provenance = adapter.verified_provenance(
            status=ProvenanceStatus.VERIFIED_DATASET,
            dataset_name="AI4Bharat Samanantar",
            dataset_revision=UPSTREAM_REVISION,
            split="test",
            record_id=str(record_id),
            source_location=f"{source_file}#L{index}:source_text",
            reference_location=f"{source_file}#L{index}:target_text",
            alignment_location=(
                f"{source_file}#L{index}:id,source_language,target_language,source_text,target_text"
            ),
        )
        records.append(
            EvaluationRecord(
                track=EvaluationTrack.SENTENCE_LEVEL,
                source_language="en",
                target_language="te",
                current_text=source,
                current_text_unit=TextUnit.SENTENCE,
                alignment_status=AlignmentStatus.SENTENCE_ALIGNED,
                reference_availability=ReferenceAvailability.AVAILABLE,
                provenance=provenance,
                reference_translation=reference,
                reference_unit=TextUnit.SENTENCE,
                alignment_evidence=AlignmentEvidence(
                    AlignmentEvidenceKind.DATASET_SENTENCE_ANNOTATION
                ),
                context_status=ContextStatus.UNAVAILABLE,
            )
        )
    return SamanantarSample(
        dataset_name="AI4Bharat Samanantar",
        dataset_revision=UPSTREAM_REVISION,
        revision_scope="upstream_revision",
        upstream_revision=UPSTREAM_REVISION,
        upstream_source_url="https://indicnlp.ai4bharat.org/samanantar/",
        preprocessing_run_id="synthetic-preprocessing-run",
        preprocessing_seed=42,
        prepared_files=(
            ("train", "train.en-te.jsonl"),
            ("val", "val.en-te.jsonl"),
            ("test", "test.en-te.jsonl"),
        ),
        split=SamanantarSplit.TEST,
        expected_split_count=None,
        source_file=source_file,
        rows_inspected=len(records),
        records=tuple(records),
        exclusions=(),
    )


def test_trusted_record_converts_to_existing_runner_shape() -> None:
    payload = bridge_samanantar_records(_sample(Path("fixture")))[0].runner_record

    assert payload == {
        "id": 31,
        "source_language": "en",
        "target_language": "te",
        "source_text": "source 31",
        "target_text": "reference 31",
        "data_source": "samanantar",
    }


def test_source_and_reference_are_preserved_exactly() -> None:
    sample = _sample(
        Path("fixture"),
        source_texts=("  exact source text  ",),
        reference_texts=("  exact reference text  ",),
    )
    row = bridge_samanantar_records(sample)[0].runner_record

    assert row["source_text"] == "  exact source text  "
    assert row["target_text"] == "  exact reference text  "


def test_record_id_and_all_provenance_locations_are_preserved() -> None:
    record = _sample().records[0]
    row = bridge_samanantar_records(_sample())[0]

    assert row.record_id == 31
    assert row.source_location.endswith("#L1:source_text")
    assert row.reference_location.endswith("#L1:target_text")
    assert "#L1:id,source_language,target_language" in row.alignment_location
    assert record.provenance.dataset_revision == UPSTREAM_REVISION
    assert record.provenance.split == "test"
    assert record.provenance.adapter_name == "samanantar_prepared_jsonl_v1"


def test_context_remains_unavailable_and_no_document_identity_is_added() -> None:
    record = _sample(Path("fixture")).records[0]
    row = bridge_samanantar_records(_sample(Path("fixture")))[0].runner_record

    assert record.context_status is ContextStatus.UNAVAILABLE
    assert record.document_or_passage_id is None
    assert "document_id" not in row
    assert "context" not in row


def test_untrusted_provenance_is_rejected() -> None:
    sample = _sample(Path("fixture"))
    record = sample.records[0]
    untrusted = replace(
        record,
        alignment_status=AlignmentStatus.UNVERIFIED,
        alignment_evidence=None,
        reference_availability=ReferenceAvailability.UNAVAILABLE,
        reference_translation=None,
        reference_unit=None,
        provenance=DatasetProvenance.unverified(),
    )
    sample = replace(sample, records=(untrusted,))

    with pytest.raises(ValueError, match="non-verified provenance"):
        bridge_samanantar_records(sample)


def test_non_sentence_record_is_rejected() -> None:
    record = _sample().records[0]
    with pytest.raises(ValueError, match="sentence_level requires"):
        replace(record, current_text_unit=TextUnit.DOCUMENT)


def test_bridge_preserves_sample_record_order() -> None:
    rows = bridge_samanantar_records(_sample(ids=(31, 7, 55)))

    assert [row.record_id for row in rows] == [31, 7, 55]
    assert [row.original_line_index for row in rows] == [0, 1, 2]


def test_bridge_output_is_accepted_by_existing_evaluator_loader(tmp_path: Path) -> None:
    rows = bridge_samanantar_records(_sample(ids=(73,)))
    bridge_path = tmp_path / "bridge.jsonl"
    bridge_path.write_text(
        "".join(json.dumps(row.runner_record) + "\n" for row in rows),
        encoding="utf-8",
    )

    loaded, total = load_reproducible_sample(
        bridge_path,
        max_samples=1,
        seed=42,
        source_lang="en",
        target_lang="te",
    )

    assert total == 1
    assert loaded[0]["record"]["id"] == 73
    assert loaded[0]["record"]["source_text"] == "source 73"


def test_phase_12c_supports_larger_bounded_sentence_samples(tmp_path: Path) -> None:
    sample = _sample(ids=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150))

    class FixedAdapter:
        def load_sample(self, *, split: str, max_samples: int) -> SamanantarSample:
            assert split == "test"
            assert max_samples == 15
            return sample

    def fake_runner(
        *,
        source_lang: str,
        target_lang: str,
        max_samples: int,
        seed: int,
        context_window: int,
        output_dir: Path | str | None,
        dataset_path: Path | str | None,
        include_bertscore: bool,
        include_comet: bool,
    ) -> Path:
        run_dir = Path(output_dir) / "phase12c"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "experiment": {
                "run_id": "phase12c-test",
                "dataset": {"name": "AI4Bharat Samanantar", "path": str(dataset_path), "split": "test"},
                "sampling": {
                    "requested_max_samples": max_samples,
                    "selected_sample_count": max_samples,
                    "selected_indices": list(range(max_samples)),
                    "selected_record_ids": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
                },
                "model_checkpoint": "nllb200",
                "generation_settings": {"max_new_tokens": 256},
                "metric_methodology": {"bleu": "corpus bleu", "chrf++": "corpus chrf++"},
            },
            "metrics": {
                "baseline_sentence_level": {
                    "bleu": {"status": "computed", "num_examples": max_samples},
                    "chrf++": {"status": "computed", "num_examples": max_samples},
                }
            },
        }
        (run_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "summary.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "predictions.jsonl").write_text(
            "".join(
                json.dumps({"record_id": row.record_id, "baseline_status": "computed"}) + "\n"
                for row in bridge_samanantar_records(sample)
            ),
            encoding="utf-8",
        )
        return run_dir

    run_dir = run_bounded_samanantar_baseline(
        split="test",
        max_samples=15,
        seed=42,
        source_lang="en",
        target_lang="te",
        phase="12C",
        adapter=FixedAdapter(),
        evaluation_runner=fake_runner,
        output_root=tmp_path / "bounded",
    )

    assert run_dir.name == "phase12c"
    assert run_dir.parent == tmp_path / "bounded"
    assert run_dir.exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment"]["sampling"]["requested_max_samples"] == 15
    assert summary["experiment"]["context_window"] == 0
    assert summary["experiment"]["evaluation_modes"] == ["baseline_sentence_level"]


def test_phase_12b_stays_locked_to_the_smoke_contract() -> None:
    with pytest.raises(ValueError, match="Phase 12B.*5"):
        run_bounded_samanantar_baseline(
            split="test",
            max_samples=6,
            seed=42,
            source_lang="en",
            target_lang="te",
            phase="12B",
        )