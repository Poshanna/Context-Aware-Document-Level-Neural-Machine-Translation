"""Document processing ABC + registry.

A :class:`DocumentProcessor` is responsible for two symmetric operations:

* **Extraction**: Given an on-disk file, return a linear list of
  :class:`DocumentChunk` objects preserving enough metadata to reassemble
  the output later (page number, paragraph id, doc order index).

* **Reassembly**: Given a list of translated chunks and a desired output
  format, write a translated file to disk and return its path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocumentChunk:
    """One unit of text processed during translation.

    For TXT files a chunk typically corresponds to one paragraph. For
    DOCX files a chunk corresponds to one paragraph run. For PDF files a
    chunk is a PyMuPDF block (with a page annotation).
    """

    id: int                          # 0-based linear order within the document
    text: str
    page: int | None = None
    paragraph_id: int | None = None
    doc_order: int = 0               # Stable sort key; same as id usually.
    meta: dict = field(default_factory=dict)


class DocumentProcessor(ABC):
    """Common interface for PDF/DOCX/TXT processors."""

    name: str
    accepted_extensions: tuple[str, ...]

    @abstractmethod
    def extract_chunks(self, source_path: Path) -> list[DocumentChunk]:
        """Read the file and split into translatable chunks.

        Raises ``ValueError`` if the file format is malformed or cannot
        be read.
        """
        raise NotImplementedError

    @abstractmethod
    def reassemble(
        self,
        translated_chunks: list[DocumentChunk],
        output_path: Path,
        output_format: str = "txt",
    ) -> Path:
        """Write ``translated_chunks`` back into a translated document.

        ``output_format`` is one of "txt", "docx".  PDF output is out of
        scope (PDF generation is a much harder problem than extraction);
        for PDF inputs the output is a DOCX with page markers in its
        text, or plain TXT.
        """
        raise NotImplementedError


class DocumentProcessorRegistry:
    """Select a suitable processor for a file extension."""

    def __init__(self) -> None:
        self._items: list[type[DocumentProcessor]] = []

    def register(self, cls: type[DocumentProcessor]) -> None:
        self._items.append(cls)

    def available(self) -> list[str]:
        return [c.name for c in self._items]

    def for_extension(self, ext: str) -> DocumentProcessor:
        ext = ext.lower().lstrip(".")
        for cls in self._items:
            if ext in {e.lower().lstrip(".") for e in cls.accepted_extensions}:
                return cls()
        raise KeyError(
            f"No processor for extension '{ext}'. "
            f"Registered extensions: {[e for c in self._items for e in c.accepted_extensions]}"
        )

    def populate(self) -> None:  # pragma: no cover - wire-up
        from app.preprocessing.txt_processor import TXTProcessor     # noqa: F401
        from app.preprocessing.pdf_processor import PDFProcessor     # noqa: F401
        from app.preprocessing.docx_processor import DOCXProcessor   # noqa: F401
        from app.preprocessing.sentence_splitter import SentenceSplitter  # noqa: F401
        # self.register(TXTProcessor)
        # self.register(PDFProcessor)
        # self.register(DOCXProcessor)


registry = DocumentProcessorRegistry()
