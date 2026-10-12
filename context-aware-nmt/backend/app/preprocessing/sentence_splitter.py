"""Deterministic, dependency-free sentence segmentation."""
from __future__ import annotations

_TERMINAL_PUNCTUATION = {".", "!", "?", "।", "。", "！", "？"}
_CLOSING_PUNCTUATION = {"\"", "'", "”", "’", "»", ")", "]", "}"}


class SentenceSplitter:
    def __init__(self, lang: str = "te") -> None:
        self.lang = lang

    def split(self, paragraph: str) -> list[str]:
        """Split on terminal punctuation followed by whitespace or document end."""
        text = paragraph.strip()
        if not text:
            return []

        sentences: list[str] = []
        sentence_start = 0
        index = 0
        while index < len(text):
            character = text[index]
            if character in _TERMINAL_PUNCTUATION:
                boundary_end = index + 1
                while (
                    boundary_end < len(text)
                    and text[boundary_end] in _TERMINAL_PUNCTUATION
                ):
                    boundary_end += 1
                while (
                    boundary_end < len(text)
                    and text[boundary_end] in _CLOSING_PUNCTUATION
                ):
                    boundary_end += 1
                if boundary_end == len(text) or text[boundary_end].isspace():
                    sentence = text[sentence_start:boundary_end].strip()
                    if sentence:
                        sentences.append(sentence)
                    while boundary_end < len(text) and text[boundary_end].isspace():
                        boundary_end += 1
                    sentence_start = boundary_end
                    index = boundary_end
                    continue
            elif character == "\n":
                sentence = text[sentence_start:index].strip()
                if sentence:
                    sentences.append(sentence)
                while index < len(text) and text[index].isspace():
                    index += 1
                sentence_start = index
                continue
            index += 1

        remaining = text[sentence_start:].strip()
        if remaining:
            sentences.append(remaining)
        return sentences
