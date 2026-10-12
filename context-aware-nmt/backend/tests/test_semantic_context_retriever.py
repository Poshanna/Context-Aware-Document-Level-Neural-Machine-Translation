import pytest

from app.services.semantic_context_retriever import (
    SemanticContextRetriever,
)


class FakeEmbeddingModel:
    """Deterministic fake embedding model for unit tests."""

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    ):
        vectors = []

        for index, _ in enumerate(texts):
            vector = [0.0, 0.0, 0.0]
            vector[index % 3] = 1.0
            vectors.append(vector)

        return vectors


def make_retriever(top_k=2):
    retriever = SemanticContextRetriever(
        top_k=top_k,
    )

    # Avoid downloading/loading a real embedding model in unit tests.
    retriever._model = FakeEmbeddingModel()

    return retriever


def test_empty_context_returns_empty():
    retriever = make_retriever()

    result = retriever.retrieve(
        current_sentence="Current sentence.",
        preceding_sentences=[],
    )

    assert result == []


def test_empty_current_sentence_rejected():
    retriever = make_retriever()

    with pytest.raises(ValueError):
        retriever.retrieve(
            current_sentence="",
            preceding_sentences=[
                {
                    "sentence_id": "1.1",
                    "text": "Previous sentence.",
                }
            ],
        )


def test_current_sentence_is_not_used_as_context():
    retriever = make_retriever(top_k=5)

    current = "Current sentence."

    result = retriever.retrieve(
        current_sentence=current,
        preceding_sentences=[
            {
                "sentence_id": "1.1",
                "text": current,
            },
            {
                "sentence_id": "2.1",
                "text": "Previous sentence.",
            },
        ],
    )

    assert all(
        candidate.text != current
        for candidate in result
    )


def test_top_k_is_respected():
    retriever = make_retriever(top_k=2)

    result = retriever.retrieve(
        current_sentence="Current sentence.",
        preceding_sentences=[
            {
                "sentence_id": "1.1",
                "text": "Sentence one.",
            },
            {
                "sentence_id": "2.1",
                "text": "Sentence two.",
            },
            {
                "sentence_id": "3.1",
                "text": "Sentence three.",
            },
            {
                "sentence_id": "4.1",
                "text": "Sentence four.",
            },
        ],
    )

    assert len(result) <= 2


def test_selected_context_preserves_document_order():
    retriever = make_retriever(top_k=3)

    result = retriever.retrieve(
        current_sentence="Current sentence.",
        preceding_sentences=[
            {
                "sentence_id": "1.1",
                "text": "Sentence one.",
            },
            {
                "sentence_id": "2.1",
                "text": "Sentence two.",
            },
            {
                "sentence_id": "3.1",
                "text": "Sentence three.",
            },
        ],
    )

    ids = [candidate.sentence_id for candidate in result]

    assert ids == sorted(
        ids,
        key=lambda value: int(value.split(".")[0]),
    )


def test_retrieve_texts_returns_text_only():
    retriever = make_retriever(top_k=2)

    result = retriever.retrieve_texts(
        current_sentence="Current sentence.",
        preceding_sentences=[
            {
                "sentence_id": "1.1",
                "text": "Sentence one.",
            },
            {
                "sentence_id": "2.1",
                "text": "Sentence two.",
            },
        ],
    )

    assert all(
        isinstance(item, str)
        for item in result
    )