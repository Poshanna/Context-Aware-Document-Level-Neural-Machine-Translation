from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from app.core.config import get_settings
from app.evaluation.bleu_metric import BLEUMetric
from app.evaluation.chrf_metric import CHRFMetric
from app.evaluation.experiment_runner import (
    legitimate_context,
    load_reproducible_sample,
    run_evaluation,
    validate_evaluation_args,
)
from run_evaluation import build_parser


def _write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_sampling_is_repeatable_without_replacement_and_preserves_text(tmp_path: Path) -> None:
    path = _write_rows(
        tmp_path / "test.jsonl",
        [
            {"id": i, "source_language": "en", "target_language": "te",
             "source_text": f" source {i} ", "target_text": f" reference {i} "}
            for i in range(12)
        ],
    )
    first, total = load_reproducible_sample(
        path, max_samples=5, seed=42, source_lang="en", target_lang="te"
    )
    repeated, _ = load_reproducible_sample(
        path, max_samples=5, seed=42, source_lang="en", target_lang="te"
    )
    other_seed, _ = load_reproducible_sample(
        path, max_samples=5, seed=43, source_lang="en", target_lang="te"
    )
    assert total == 12
    assert [item["line_index"] for item in first] == [item["line_index"] for item in repeated]
    assert [item["line_index"] for item in first] != [item["line_index"] for item in other_seed]
    assert len({item["line_index"] for item in first}) == 5
    record = first[0]["record"]
    assert record["source_text"] == f" source {record['id']} "
    assert record["target_text"] == f" reference {record['id']} "


def test_reverse_direction_preserves_original_pair_values(tmp_path: Path) -> None:
    path = _write_rows(tmp_path / "reverse.jsonl", [{
        "id": 7, "source_language": "en", "target_language": "te",
        "source_text": " exact English ", "target_text": " ఖచ్చితమైన తెలుగు ",
    }])
    sample, _ = load_reproducible_sample(
        path, max_samples=1, seed=0, source_lang="te", target_lang="en"
    )
    from app.evaluation.experiment_runner import _orient_record

    assert _orient_record(sample[0]["record"], "te", "en", 0) == (
        " ఖచ్చితమైన తెలుగు ", " exact English "
    )


def test_context_requires_same_document_and_strictly_prior_indices() -> None:
    record = {
        "document_id": "doc-a", "sentence_index": 4,
        "source_text": "current", "target_text": "ప్రస్తుత",
        "source_language": "en", "target_language": "te",
        "context": [
            {"document_id": "doc-a", "sentence_index": 1, "source_text": "prior",
             "target_text": "ముందు", "source_language": "en", "target_language": "te"},
            {"document_id": "doc-a", "sentence_index": 6, "source_text": "future",
             "target_text": "తరువాత", "source_language": "en", "target_language": "te"},
            {"document_id": "doc-b", "sentence_index": 2, "source_text": "other",
             "target_text": "వేరే", "source_language": "en", "target_language": "te"},
        ],
    }
    assert legitimate_context(
        record, source_lang="en", target_lang="te", context_window=2
    ) == (["prior"], [1])
    assert legitimate_context(
        {"source_text": "isolated", "context": [record]},
        source_lang="en", target_lang="te", context_window=2,
    ) == ([], [])


def test_corpus_metrics_match_exact_reference_with_fourgrams() -> None:
    hypothesis = "ఈ రోజు చాలా మంచి తెలుగు వాక్యం"
    sources = ["source"]
    assert BLEUMetric().compute(
        sources, [hypothesis], [hypothesis], src_lang="en", tgt_lang="te"
    ).aggregate == pytest.approx(100.0)
    assert CHRFMetric().compute(
        sources, [hypothesis], [hypothesis], src_lang="en", tgt_lang="te"
    ).aggregate == pytest.approx(100.0)


class _FakeTranslationService:
    def translate_text(self, *, user_id, req):
        return SimpleNamespace(status="ok", translated_text=f"BASE::{req.text}", note=None)


class _FakeContextService:
    def __init__(self) -> None:
        self.calls = []

    def translate_sentence(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "translation": f"PREFIX::{kwargs['current_sentence']}",
            "formatted_input": "mocked model input",
        }


def test_artifact_marks_sentence_corpus_context_unavailable(tmp_path: Path) -> None:
    dataset = _write_rows(tmp_path / "test.en-te.jsonl", [{
        "id": 13, "source_language": "en", "target_language": "te",
        "source_text": " exact source ", "target_text": " ఖచ్చితమైన సూచన ",
        "data_source": "samanantar",
    }])
    context_service = _FakeContextService()
    run_dir = run_evaluation(
        source_lang="en", target_lang="te", max_samples=1, seed=42,
        context_window=1, output_dir=tmp_path / "runs", dataset_path=dataset,
        settings=get_settings(), translation_service=_FakeTranslationService(),
        context_service=context_service,
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    row = json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8"))
    assert summary["experiment"]["sampling"]["selected_record_ids"] == [13]
    assert row["source"] == " exact source "
    assert row["reference"] == " ఖచ్చితమైన సూచన "
    assert row["baseline_translation"] == "BASE:: exact source "
    assert row["context_conditioned_translation"] is None
    assert row["context_available"] is False
    assert row["status"] == "context_conditioned_not_available"
    assert context_service.calls == []
    assert summary["metrics"]["context_conditioned"]["bleu"]["status"] == "unavailable"
    assert summary["experiment"]["generation_settings"]["max_new_tokens"] == 256
    assert summary["experiment"]["generation_settings"]["do_sample"] is False
    assert {"config.json", "metrics.json", "system_info.json", "README.md"}.issubset(
        {path.name for path in run_dir.iterdir()}
    )


def test_explicit_prior_context_runs_prefix_translation(tmp_path: Path) -> None:
    dataset = _write_rows(tmp_path / "document.jsonl", [{
        "id": 9, "source_language": "en", "target_language": "te",
        "source_text": "Current sentence", "target_text": "ప్రస్తుత వాక్యం",
        "document_id": "real-doc", "sentence_index": 3,
        "context": [{
            "document_id": "real-doc", "sentence_index": 2,
            "source_text": "Prior sentence", "target_text": "మునుపటి వాక్యం",
            "source_language": "en", "target_language": "te",
        }, {
            "document_id": "real-doc", "sentence_index": 4,
            "source_text": "Future sentence", "target_text": "భవిష్యత్ వాక్యం",
            "source_language": "en", "target_language": "te",
        }],
    }])
    context_service = _FakeContextService()
    run_dir = run_evaluation(
        source_lang="en", target_lang="te", max_samples=1, seed=42,
        context_window=2, output_dir=tmp_path / "runs", dataset_path=dataset,
        settings=get_settings(), translation_service=_FakeTranslationService(),
        context_service=context_service,
    )
    row = json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8"))
    assert row["context_available"] is True
    assert row["context_sentences"] == ["Prior sentence"]
    assert row["context_indices"] == [2]
    assert row["context_conditioned_translation"] == "PREFIX::Current sentence"
    assert row["status"] == "context_conditioned"
    assert len(context_service.calls) == 1


class _FailingTranslationService:
    def translate_text(self, *, user_id, req):
        raise RuntimeError("mock translation failure")


def test_translation_failure_marks_metrics_failed(tmp_path: Path) -> None:
    dataset = _write_rows(tmp_path / "test.en-te.jsonl", [{
        "id": 1, "source_language": "en", "target_language": "te",
        "source_text": "source sentence", "target_text": "తెలుగు వాక్యం",
    }])
    run_dir = run_evaluation(
        source_lang="en", target_lang="te", max_samples=1, seed=42,
        context_window=0, output_dir=tmp_path / "runs", dataset_path=dataset,
        settings=get_settings(), translation_service=_FailingTranslationService(),
        context_service=_FakeContextService(),
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["baseline_sentence_level"]["bleu"]["status"] == "failed"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_lang": "en", "target_lang": "en", "max_samples": 5, "context_window": 1, "seed": 1},
        {"source_lang": "en", "target_lang": "te", "max_samples": 0, "context_window": 1, "seed": 1},
        {"source_lang": "en", "target_lang": "te", "max_samples": 5, "context_window": 201, "seed": 1},
        {"source_lang": "en", "target_lang": "te", "max_samples": 5, "context_window": 1, "seed": -1},
    ],
)
def test_argument_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        validate_evaluation_args(**kwargs)


def test_cli_rejects_zero_samples() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "--source-lang", "en", "--target-lang", "te", "--max-samples", "0"
        ])