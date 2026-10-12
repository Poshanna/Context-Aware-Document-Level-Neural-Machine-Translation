"""PDF (text, non-scanned) file processor based on PyMuPDF.

Phase 3 will implement:
  * Per-page text blocks -> chunks with page + block index metadata.
  * Warning for documents with 2-column layouts (detection heuristic:
    check if block x-coordinates form two clusters).
"""
from __future__ import annotations

from pathlib import Path

from app.preprocessing import DocumentChunk, DocumentProcessor


class PDFProcessor(DocumentProcessor):
    name = "pdf"
    accepted_extensions = ("pdf",)

    def extract_chunks(self, source_path: Path) -> list[DocumentChunk]:  # pragma: no cover
        raise NotImplementedError("Phase 3 implementation.")

    def reassemble(  # pragma: no cover
        self,
        translated_chunks: list[DocumentChunk],
        output_path: Path,
        output_format: str = "txt",
    ) -> Path:
        raise NotImplementedError("Phase 3 implementation.")
