"""DOCX file processor based on python-docx.

Phase 3 will implement paragraph-level extraction with run preservation
so that bold/italic formatting can be roughly preserved on reassembly.
"""
from __future__ import annotations

from pathlib import Path

from app.preprocessing import DocumentChunk, DocumentProcessor


class DOCXProcessor(DocumentProcessor):
    name = "docx"
    accepted_extensions = ("docx",)

    def extract_chunks(self, source_path: Path) -> list[DocumentChunk]:  # pragma: no cover
        raise NotImplementedError("Phase 3 implementation.")

    def reassemble(  # pragma: no cover
        self,
        translated_chunks: list[DocumentChunk],
        output_path: Path,
        output_format: str = "txt",
    ) -> Path:
        raise NotImplementedError("Phase 3 implementation.")
