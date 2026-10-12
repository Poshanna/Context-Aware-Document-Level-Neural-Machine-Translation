from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.evaluation.indicdisco_adapter import (
    INDICDISCO_PARAGRAPH_COMPONENT,
    INDICDISCO_REVISION,
    INDICDISCO_SOURCE_LANGUAGE,
    INDICDISCO_TARGET_LANGUAGE,
    IndicDiscoMTTrustedAdapter,
)
from scripts.run_indicdisco_paragraph_evaluation import (
    build_artifact,
    evaluate_records,
    metric_configuration,
    select_bounded_sample,
    translate_paragraphs,
)


def _record(index: int):
    locator = f"fixture://test.csv#row={index}"
    return IndicDiscoMTTrustedAdapter().adapt_row(
        row={"source": f"Telugu paragraph {index}.", "target": f"English reference {index}."},
        revision=INDICDISCO_REVISION,
        split="test",
        component=INDICDISCO_PARAGRAPH_COMPONENT,
        row_locator=locator,
        source_location=f"{locator}:source",
        reference_location=f"{locator}:target",
        alignment_location=f"{locator}:source,target",
        source_language=INDICDISCO_SOURCE_LANGUAGE,
        target_language=INDICDISCO_TARGET_LANGUAGE,
    )


class _FakeTranslationService:
    def __init__(self, *, fail_on: set[str] | None = None) -> None:
        self.fail_on = fail_on or set()
        self.inputs: list[tuple[str, str, str, str]] = []

    def translate_text(self, *, user_id: object, req: Any) -> SimpleNamespace:
        del user_id
        self.inputs.append((req.source_lang, req.target_lang, req.pipeline, req.text))
        if req.text in self.fail_on:
            raise RuntimeError("synthetic NLLB failure")
        return SimpleNamespace(status="ok", translated_text=f"Hypothesis for {req.text}", note=None)


def _no_metric_compute(calls: list[tuple[Any, ...]]):
    def compute(name: str, sources: list[str], hypotheses: list[str], references: list[str], src_lang: str, tgt_lang: str):
        calls.append((name, sources, hypotheses, references, src_lang, tgt_lang))
        return {"status": "unavailable", "score": None, "reason": "metrics disabled in synthetic unit test"}

    return compute


def test_bounded_sample_selection_is_deterministic_and_caps_at_available_records() -> None:
    records = [_record(index) for index in range(8)]

    first = select_bounded_sample(records, max_samples=4, seed=17)
    repeated = select_bounded_sample(records, max_samples=4, seed=17)
    all_available = select_bounded_sample(records[:2], max_samples=4, seed=17)

    assert [record.row_locator for record in first] == [record.row_locator for record in repeated]
    assert len(first) == 4
    assert len(all_available) == 2


def test_provenance_source_and_reference_are_preserved_without_document_identity() -> None:
    record = _record(4)
    service = _FakeTranslationService()

    prediction = translate_paragraphs(
        [record], translation_service=service, source_lang="te", target_lang="en"
    )[0]

    assert prediction["dataset"] == "IndicDISCO-MT"
    assert prediction["revision"] == INDICDISCO_REVISION
    assert prediction["component"] == INDICDISCO_PARAGRAPH_COMPONENT
    assert prediction["split"] == "test"
    assert prediction["row_locator"] == record.row_locator
    assert prediction["source"] == record.source_text
    assert prediction["reference"] == record.reference_text
    assert prediction["source_language"] == "Telugu"
    assert prediction["target_language"] == "English"
    assert prediction["document_id"] is None
    assert prediction["context_used"] is False
    assert service.inputs == [("te", "en", "baseline", record.source_text)]


def test_metric_artifact_structure_stays_paragraph_only_and_unscored_in_fixture() -> None:
    records = [_record(0), _record(1)]
    service = _FakeTranslationService()
    calls: list[tuple[Any, ...]] = []
    sampled, predictions, metrics = evaluate_records(
        records,
        max_samples=5,
        seed=42,
        source_lang="te",
        target_lang="en",
        translation_service=service,
        metric_computer=_no_metric_compute(calls),
    )
    model_config = {
        "checkpoint": "synthetic-test-only",
        "generation_settings": {"max_new_tokens": 256, "max_length": 200},
    }
    artifact = build_artifact(
        predictions=predictions,
        metrics=metrics,
        dataset="IndicDISCO-MT",
        revision=INDICDISCO_REVISION,
        component=INDICDISCO_PARAGRAPH_COMPONENT,
        split="test",
        source_lang="te",
        target_lang="en",
        requested_max_samples=5,
        seed=42,
        model_name="nllb200",
        model_configuration=model_config,
        device="cpu",
        bytes_read=4096,
        candidate_row_count=len(records),
        exclusions=(),
    )

    assert len(sampled) == 2
    assert artifact["actual_sample_count"] == 2
    assert artifact["requested_max_samples"] == 5
    assert artifact["sample_row_locators"] == [record.row_locator for record in sampled]
    assert artifact["metric_configuration"]["aggregation"] == "corpus"
    assert artifact["metrics"]["corpus_bleu"]["status"] == "unavailable"
    assert artifact["metrics"]["corpus_chrf++"]["score"] is None
    assert artifact["model_configuration"]["generation_settings"] == {
        "max_new_tokens": 256,
        "max_length": 200,
    }
    assert artifact["rights_status"] == "unresolved"
    assert artifact["evaluation_granularity"] == "paragraph"
    assert artifact["document_level_evaluation"] is False
    assert artifact["context_used"] is False
    assert artifact["context_available"] is False
    assert artifact["sentence_alignment_available"] is False
    assert artifact["document_identity_available"] is False
    assert artifact["no_fabricated_document_ids"] is True
    assert artifact["no_inference"] is False
    assert artifact["metrics_computed"] is False
    assert [call[0] for call in calls] == ["bleu", "chrf++"]


def test_inference_failure_is_recorded_and_suppresses_corpus_metrics() -> None:
    records = [_record(0), _record(1)]
    service = _FakeTranslationService(fail_on={records[0].source_text})
    calls: list[tuple[Any, ...]] = []

    _, predictions, metrics = evaluate_records(
        records,
        max_samples=2,
        seed=0,
        source_lang="te",
        target_lang="en",
        translation_service=service,
        metric_computer=_no_metric_compute(calls),
    )

    assert predictions[0]["inference_status"] == "failed"
    assert predictions[0]["hypothesis"] is None
    assert "synthetic NLLB failure" in predictions[0]["inference_error"]
    assert predictions[1]["inference_status"] == "succeeded"
    assert predictions[1]["hypothesis"] is not None
    assert predictions[0]["metric_eligibility"]["paragraph_bleu"] is False
    assert metrics["corpus_bleu"]["status"] == "failed"
    assert metrics["corpus_chrf++"]["status"] == "failed"
    assert calls == []


def test_metric_configuration_names_existing_sacrebleu_settings() -> None:
    config = metric_configuration("en")

    assert config["bleu"]["implementation"] == "app.evaluation.bleu_metric.BLEUMetric via SacreBLEU"
    assert config["bleu"]["tokenizer"] == "13a"
    assert config["chrf++"]["char_order"] == 6
    assert config["chrf++"]["word_order"] == 2
    assert config["chrf++"]["beta"] == 2
