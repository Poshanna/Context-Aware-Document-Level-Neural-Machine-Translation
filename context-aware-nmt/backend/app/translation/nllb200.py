"""Lazy NLLB-200 sentence-level baseline translator."""
from __future__ import annotations

import threading
from typing import Any

from app.core.config import Settings, get_settings
from app.translation import (
        ModelUnavailableError,
        TranslationBackendError,
        TranslationContext,
        Translator,
        TranslatorInfo,
)


FLORES_LANGUAGE_CODES = {
        "en": "eng_Latn",
        "te": "tel_Telu",
}


class NLLB200Translator(Translator):
    info = TranslatorInfo(
        name="nllb200",
        max_seq_length=512,
        supported_directions={
            ("te", "en"), ("en", "te"),
        },
    )

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._load_lock = threading.RLock()
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def _resolve_device(self, torch: Any) -> str:
        configured_device = self.settings.FORCE_DEVICE
        if configured_device == "cuda":
            if not torch.cuda.is_available():
                raise ModelUnavailableError(
                    "CUDA was requested for NLLB-200, but CUDA is not available. "
                    "Set FORCE_DEVICE=auto or FORCE_DEVICE=cpu."
                )
            return "cuda"
        if configured_device == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        """Load the pretrained checkpoint on first use; safe to call repeatedly."""
        with self._load_lock:
            if self.loaded:
                return

            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:
                raise ModelUnavailableError(
                    "NLLB-200 requires PyTorch and Transformers. Install the backend "
                    "requirements before using the baseline translator."
                ) from exc

            device = self._resolve_device(torch)
            model_source = (
                self.settings.MODEL_NLLB200_LOCAL_PATH
                or self.settings.MODEL_NLLB200
            )
            local_files_only = bool(self.settings.TRANSFORMERS_OFFLINE)
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_source,
                    local_files_only=local_files_only,
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_source,
                    local_files_only=local_files_only,
                )
                model.to(device)
                model.eval()
                torch.manual_seed(self.settings.TRANSLATION_SEED)
                if device == "cuda":
                    torch.cuda.manual_seed_all(self.settings.TRANSLATION_SEED)
            except ModelUnavailableError:
                raise
            except Exception as exc:
                raise ModelUnavailableError(
                    "Could not load the configured NLLB-200 pretrained checkpoint. "
                    "Check model access/network availability, MODEL_NLLB200_LOCAL_PATH, "
                    "and TRANSFORMERS_OFFLINE."
                ) from exc

            self._torch = torch
            self._device = device
            self._tokenizer = tokenizer
            self._model = model

    def translate_batch(
        self,
        sentences: list[str],
        *,
        src_lang: str,
        tgt_lang: str,
        context_groups: list[list[TranslationContext]] | None = None,
    ) -> list[str]:
        if (src_lang, tgt_lang) not in self.info.supported_directions:
            raise TranslationBackendError(
                f"NLLB-200 does not support {src_lang}→{tgt_lang}."
            )
        if context_groups is not None:
            if len(context_groups) != len(sentences):
                raise TranslationBackendError(
                    "context_groups must contain one entry per input sentence."
                )
            if any(context_group for context_group in context_groups):
                raise TranslationBackendError(
                    "The baseline translator does not accept document context."
                )
        if any(not isinstance(sentence, str) for sentence in sentences):
            raise TranslationBackendError("Every input sentence must be a string.")
        if not sentences:
            return []

        active_positions = [
            index for index, sentence in enumerate(sentences) if sentence.strip()
        ]
        outputs = [""] * len(sentences)
        if not active_positions:
            return outputs

        self.load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._device is not None

        with self._load_lock:
            try:
                self._tokenizer.src_lang = FLORES_LANGUAGE_CODES[src_lang]
                encoded = self._tokenizer(
                    [sentences[index].strip() for index in active_positions],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.settings.TRANSLATION_MAX_INPUT_LENGTH,
                )
                encoded = {
                    key: value.to(self._device)
                    for key, value in encoded.items()
                }
                forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(
                    FLORES_LANGUAGE_CODES[tgt_lang]
                )
                if forced_bos_token_id == self._tokenizer.unk_token_id:
                    raise TranslationBackendError(
                        f"NLLB tokenizer has no language token for {tgt_lang}."
                    )
                with self._torch.inference_mode():
                    generated = self._model.generate(
                        **encoded,
                        forced_bos_token_id=forced_bos_token_id,
                        max_new_tokens=self.settings.TRANSLATION_MAX_OUTPUT_LENGTH,
                        do_sample=False,
                        num_beams=1,
                    )
                decoded = self._tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                )
            except TranslationBackendError:
                raise
            except Exception as exc:
                raise TranslationBackendError(
                    "NLLB-200 baseline translation failed."
                ) from exc

        for index, translation in zip(active_positions, decoded):
            outputs[index] = translation.strip()
        return outputs
