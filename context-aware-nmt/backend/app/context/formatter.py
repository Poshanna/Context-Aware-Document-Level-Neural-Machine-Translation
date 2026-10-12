"""Deterministic plain-text formatting for context-aware model inputs."""

from __future__ import annotations


def format_context_input(
    context_sentences: list[str],
    current_sentence: str,
) -> str:
    """Format prior context separately from the sentence being translated."""
    context_block = "\n".join(context_sentences)
    if context_block:
        return f"CONTEXT:\n{context_block}\n\nCURRENT SENTENCE:\n{current_sentence}"
    return f"CONTEXT:\n\nCURRENT SENTENCE:\n{current_sentence}"
