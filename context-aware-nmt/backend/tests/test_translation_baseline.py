"""Mocked unit tests for the sentence-level NLLB baseline."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from typing import Any

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.translation import (
    ModelUnavailableError,
    TranslationBackendError,
    TranslationContext,
    TranslatorRegistry,
)
from app.translation.nllb200 import FLORES_LANGUAGE_CODES, NLLB200Translator


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "JWT_SECRET_KEY": "test-secret",
        "POSTGRES_PASSWORD": "test-password",
    }
    values.update(overrides)
    return Settings(**values)


def test_translation_config_default_values() -> None:
    settings = _settings()

    assert settings.TRANSLATION_SOURCE_LANGUAGE == "en"
    assert settings.TRANSLATION_TARGET_LANGUAGE == "te"
    assert settings.TRANSLATION_MAX_INPUT_LENGTH == 512
    assert settings.TRANSLATION_MAX_OUTPUT_LENGTH == 256
    assert settings.MODEL_NLLB200_LOCAL_PATH is None


def test_translation_config_defaults_and_local_checkpoint_override() -> None:
    settings = _settings(
        MODEL_NLLB200_LOCAL_PATH="C:/models/nllb",
        TRANSLATION_SOURCE_LANGUAGE="te",
        TRANSLATION_TARGET_LANGUAGE="en",
        TRANSLATION_MAX_INPUT_LENGTH=384,
        TRANSLATION_MAX_OUTPUT_LENGTH=192,
    )

    assert settings.MODEL_NLLB200 == "facebook/nllb-200-distilled-600M"
    assert settings.MODEL_NLLB200_LOCAL_PATH == "C:/models/nllb"
    assert settings.TRANSLATION_SOURCE_LANGUAGE == "te"
    assert settings.TRANSLATION_TARGET_LANGUAGE == "en"
    assert settings.TRANSLATION_MAX_INPUT_LENGTH == 384
    assert settings.TRANSLATION_MAX_OUTPUT_LENGTH == 192
    assert settings.FORCE_DEVICE == "auto"
    assert settings.TRANSFORMERS_OFFLINE == 0


def test_translation_config_rejects_unsupported_default_language() -> None:
    with pytest.raises(ValidationError):
        _settings(TRANSLATION_SOURCE_LANGUAGE="fr")


def test_registry_registers_lazy_nllb_backend() -> None:
    translator_registry = TranslatorRegistry()
    translator_registry.populate()

    assert translator_registry.available() == ["nllb200"]
    translator = translator_registry.get("nllb200")
    assert isinstance(translator, NLLB200Translator)
    assert translator.loaded is False
    assert translator_registry.supports("nllb200", "en", "te")
    assert translator_registry.supports("nllb200", "te", "en")
    assert not translator_registry.supports("nllb200", "en", "hi")


class _FakeTensor:
    def __init__(self) -> None:
        self.device: str | None = None

    def to(self, device: str) -> _FakeTensor:
        self.device = device
        return self


class _FakeTokenizer:
    unk_token_id = 0

    def __init__(self) -> None:
        self.src_lang: str | None = None
        self.encoded_texts: list[str] = []
        self.max_length: int | None = None
        self.target_codes = {"eng_Latn": 1, "tel_Telu": 2}

    def __call__(self, sentences: list[str], **kwargs: Any) -> dict[str, _FakeTensor]:
        self.encoded_texts = sentences
        self.max_length = kwargs.get("max_length")
        return {"input_ids": _FakeTensor(), "attention_mask": _FakeTensor()}

    def convert_tokens_to_ids(self, language_code: str) -> int:
        return self.target_codes[language_code]

    def batch_decode(
        self, generated: list[str], *, skip_special_tokens: bool
    ) -> list[str]:
        assert skip_special_tokens is True
        return generated


class _FakeModel:
    def __init__(self) -> None:
        self.generation_kwargs: dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> list[str]:
        self.generation_kwargs = kwargs
        return ["హలో"]


class _FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()


def test_nllb_batch_translation_maps_language_codes_and_is_deterministic() -> None:
    settings = _settings(TRANSLATION_MAX_INPUT_LENGTH=384, TRANSLATION_MAX_OUTPUT_LENGTH=96)
    translator = NLLB200Translator(settings)
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    translator._tokenizer = tokenizer
    translator._model = model
    translator._torch = _FakeTorch()
    translator._device = "cpu"

    translations = translator.translate_batch(
        ["Hello", "   "],
        src_lang="en",
        tgt_lang="te",
    )

    assert translations == ["హలో", ""]
    assert tokenizer.src_lang == FLORES_LANGUAGE_CODES["en"]
    assert tokenizer.encoded_texts == ["Hello"]
    assert tokenizer.max_length == 384
    assert model.generation_kwargs["forced_bos_token_id"] == 2
    assert model.generation_kwargs["max_new_tokens"] == 96
    assert model.generation_kwargs["do_sample"] is False
    assert model.generation_kwargs["num_beams"] == 1
    assert model.generation_kwargs["num_beams"] == 1

    translator.translate_batch(["శుభోదయం"], src_lang="te", tgt_lang="en")
    assert tokenizer.src_lang == FLORES_LANGUAGE_CODES["te"]
    assert model.generation_kwargs["forced_bos_token_id"] == 1


def test_nllb_baseline_rejects_document_context() -> None:
    translator = NLLB200Translator(_settings())

    with pytest.raises(TranslationBackendError, match="does not accept document context"):
        translator.translate_batch(
            ["Hello"],
            src_lang="en",
            tgt_lang="te",
            context_groups=[[TranslationContext(source_sent_idx=0, score=1.0, snippet="prior")]],
        )


def test_auto_device_falls_back_to_cpu_when_cuda_is_unavailable() -> None:
    class NoCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    translator = NLLB200Translator(_settings(FORCE_DEVICE="auto"))
    assert translator._resolve_device(type("TorchStub", (), {"cuda": NoCuda})()) == "cpu"


def test_forced_cuda_reports_unavailable_device() -> None:
    class NoCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    translator = NLLB200Translator(_settings(FORCE_DEVICE="cuda"))
    with pytest.raises(ModelUnavailableError, match="CUDA is not available"):
        translator._resolve_device(type("TorchStub", (), {"cuda": NoCuda})())


def test_missing_transformers_returns_model_unavailable_without_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)
    translator = NLLB200Translator(_settings())

    with pytest.raises(ModelUnavailableError, match="requires PyTorch and Transformers"):
        translator.load()

    assert translator.loaded is False
