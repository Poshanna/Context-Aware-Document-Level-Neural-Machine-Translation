"""Document-level context record data structures (interface only).

This module defines the :class:`ContextRecord` interface that later phases (5+
(Phase 5 context-retrieval pipeline will use to feed into the context-aware NMT
pipeline.  For Phase 3 we ONLY the data structures ONLY the interface only.

Samanantar is **sentence-level parallel corpus — consecutive rows in the tarball
are NOT guaranteed to be contiguous documents and we NEVER fabricate
document IDs or randomly group unrelated sentences together.  Therefore:

* ``document_id`` is always ``None`` for Samanantar-derived rows.
* ``context`` is always the empty list ``[]`` for Samanantar-derived rows.
* ``sentence_id`` equals the original :class:`ParallelRecord.id`.

The helper :func:`record_from_parallel` explicitly enforces these rules.  A later
Phase (5 builds concrete context-aware data using a separate document-level corpus
(such as Wikisource/TED talks/etc.) where real ``document_id`` are authentic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.preprocessing.dataset_loader import ParallelRecord


@dataclass(slots=True)
class ContextRecord:
    """One row ready to feed a context-aware NMT pipeline.

    Fields (matches spec FR-8.1):
        document_id — authentic document identifier from a document-level corpus,
            or ``None`` when the source is sentence-level only (Samanantar).
        sentence_id — 0-indexed position within the *original corpus (matches ParallelRecord.id).
        context — zero or more prior source sentences.  For sentence-level corpus
            (Samanantar) this is ALWAYS ``[]``; filled by the Phase 5 context
            retrieval.
        source_sentence — normalized source-language sentence.
        target_sentence — normalized target-language sentence.
    """

    document_id: str | None
    sentence_id: int
    context: list[str] = field(default_factory=list)
    source_sentence: str = ""
    target_sentence: str = ""


def record_from_parallel(pr: ParallelRecord) -> ContextRecord:
    """Convert a :class:`ParallelRecord` to a :class:`ContextRecord`.

    Strictly enforces:
        * ``document_id = None`` — Samanantar rows are sentence-level; never fabricated.
        * ``context = []`` — the Phase 5 will fill this.
        * ``sentence_id = pr.id`` — stable and deterministic.
    """

    return ContextRecord(
        document_id=None,
        sentence_id=pr.id,
        context=[],
        source_sentence=pr.source_text,
        target_sentence=pr.target_text,
    )
