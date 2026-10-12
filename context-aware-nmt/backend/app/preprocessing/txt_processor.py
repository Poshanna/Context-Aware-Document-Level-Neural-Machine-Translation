"""TXT file processor.

Phase 3 will implement paragraph splitting + sentence splitting.
"""
from __future__ import annotations

from pathlib import Path

from app.preprocessing import DocumentChunk, DocumentProcessor


class TXTProcessor(DocumentProcessor):
    name = "txt"
    accepted_extensions = ("txt",)

    def extract_chunks(self, source_path: Path) -> list[DocumentChunk]:  # pragma: no cover
        raise NotImplementedError("Phase 3 implementation.")

    def reassemble(  # pragma: no cover
        self,
        translated_chunks: list[DocumentChunk],
        output_path: Path,
        output_format: str = "txt",
    ) -> Path:
        raise NotImplementedError("Phase 3 implementation.")
