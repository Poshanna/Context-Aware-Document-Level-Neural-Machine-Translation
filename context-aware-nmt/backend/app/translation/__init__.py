"""Translation backend ABC + model registry.

All translation backends must conform to the :class:`Translator`
interface so that ``TranslationService`` can swap between IndicTrans2
and NLLB-200 (and future Hindi extensions) with a single registry key.

NOTE: No actual translation is performed in this skeleton — only the
interface + registry scaffolding exists.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar


class TranslationBackendError(RuntimeError):
    """Base exception for controlled translator backend failures."""


class ModelUnavailableError(TranslationBackendError):
    """Raised when a configured pretrained checkpoint cannot be loaded."""


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TranslationContext:
    """One context snippet paired with its retrieval metadata."""

    source_sent_idx: int
    score: float
    snippet: str


@dataclass
class TranslatorInfo:
    """Static capabilities reported by a translator.

    The service layer uses this for (a) validating language pairs before
    calling a backend, and (b) deciding context-budget truncation
    policies.
    """

    name: str
    max_seq_length: int
    supported_directions: set[tuple[str, str]]  # {(src, tgt), ...}
    prefers_prefix_context: bool = True  # True => prepend context before src


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class Translator(ABC):
    """Common interface for all neural translation backends."""

    info: ClassVar[TranslatorInfo]

    @abstractmethod
    def load(self) -> None:
        """Download weights (only if HF_HOME cache is empty) and warm up.

        Implementations MUST be idempotent.  The registry calls this once
        on first use; callers may also call it explicitly at startup to
        pay the load cost eagerly.
        """
        raise NotImplementedError

    @abstractmethod
    def translate_batch(
        self,
        sentences: list[str],
        *,
        src_lang: str,
        tgt_lang: str,
        context_groups: list[list[TranslationContext]] | None = None,
    ) -> list[str]:
        """Translate a batch of sentences.

        Parameters
        ----------
        sentences:
            One entry per source sentence.
        src_lang / tgt_lang:
            ISO-style codes like "te", "en", "hi".
        context_groups:
            If provided, same length as ``sentences``; ``context_groups[i]``
            is the ordered list of retrieved contexts for sentence ``i``.
            Implementations are responsible for truncating the combined
            source+context to ``info.max_seq_length``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TranslatorRegistry:
    """Keyed store for :class:`Translator` backends.

    Adding a new backend (e.g., a hypothetical custom Hindi one) requires
    exactly two changes:
      1. Create ``new_backend.py`` with a ``Translator`` subclass.
      2. Call :meth:`register` from ``registry.populate`` below.
    """

    def __init__(self) -> None:
        self._items: dict[str, type[Translator]] = {}
        self._instances: dict[str, Translator] = {}
        self._populated = False

    def register(self, name: str, cls: type[Translator]) -> None:
        self._items[name] = cls

    def available(self) -> list[str]:
        return sorted(self._items.keys())

    def supports(self, name: str, src: str, tgt: str) -> bool:
        if name not in self._items:
            return False
        return (src, tgt) in self._items[name].info.supported_directions

    def get(self, name: str, *, auto_load: bool = False) -> Translator:
        if name not in self._items:
            raise KeyError(
                f"Unknown translator '{name}'. Available: {self.available()}"
            )
        if name not in self._instances:
            self._instances[name] = self._items[name]()
        translator = self._instances[name]
        if auto_load:
            translator.load()
        return translator

    def populate(self) -> None:
        """Register built-in backends without importing model runtimes."""
        if self._populated:
            return
        # These imports are deferred so importing this package doesn't drag
        # torch/transformers into every process (e.g. alembic migrations).
        from app.translation.nllb200 import NLLB200Translator

        self.register("nllb200", NLLB200Translator)
        self._populated = True


registry = TranslatorRegistry()
# NOTE: In the skeleton we don't call registry.populate() yet because
# the concrete implementations will be added in Phase 3. Frontend and
# routers still work because validation is performed against string
# allow-lists. Populate will be called from the app lifespan once real
# implementations exist.
