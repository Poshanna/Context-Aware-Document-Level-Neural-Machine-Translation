# Document Context Pipeline

## Objective

The context pipeline makes preceding document sentences available alongside the current sentence. This is useful for translation phenomena such as pronoun resolution, terminology consistency, and discourse continuity. This phase establishes deterministic segmentation, causal context construction, formatting, and API plumbing; it does not establish improved translation quality.

## Three Distinct Stages

1. **Sentence-level baseline:** NLLB-200 translates each current sentence independently. Its pretrained checkpoint is not document-context-conditioned.
2. **Context-aware pipeline foundation (this phase):** deterministic recent context is retrieved and formatted for each sentence, while the same NLLB baseline still translates only the current sentence. The API reports context used and explicitly marks `translation_uses_context: false`.
3. **Future context-aware model:** a model trained or otherwise designed to consume the formatted context may use this input in a later phase. No fine-tuning or evaluation is included here.

## Architecture and Data Flow

1. The endpoint receives source/target language, document text, and an optional context window.
2. `SentenceSplitter` segments English/Telugu punctuation and newlines in order.
3. The `lastn` `ContextRetriever` selects only preceding sentences from the same submitted document.
4. `format_context_input` creates a stable `CONTEXT` / `CURRENT SENTENCE` representation.
5. The service calls the existing baseline translation service with only the current sentence. It does not pass formatted context into NLLB.
6. The response returns source sentence, translation, context sentences/indices, formatted input, backend name, and explicit metadata describing the limitation.

The retriever interface remains extensible; the registered Last-N strategy is positional and **not semantic retrieval**. No SBERT model is used.

## Context Window

The request's `context_window` controls how many immediately preceding sentences are included; it defaults to `CONTEXT_K` (3), and accepts 0 through 200. The window is causal: sentence `i` can use indices less than `i`, never future sentences. Context order is chronological, oldest to newest within the selected window. Empty documents produce empty sentence and translation lists; a single sentence has empty context.

## Formatting

For each sentence, the formatter produces:

```text
CONTEXT:
<previous sentences, one per line>

CURRENT SENTENCE:
<current sentence>
```

This is exposed for inspection and future model integration only. The current NLLB translator receives the current sentence alone.

## API Example

`POST /api/translate/document-context` (authenticated):

```json
{
  "source_lang": "en",
  "target_lang": "te",
  "text": "Ravi bought a book. He read it that night.",
  "context_window": 1
}
```

Response shape (translations are illustrative API fields, not claimed model output):

```json
{
  "data": {
    "status": "ok",
    "source_lang": "en",
    "target_lang": "te",
    "model_backend": "nllb200",
    "source_sentences": ["Ravi bought a book.", "He read it that night."],
    "translations": ["<baseline translation>", "<baseline translation>"],
    "sentences": [
      {
        "sentence_idx": 0,
        "source": "Ravi bought a book.",
        "translation": "<baseline translation>",
        "context": [],
        "context_indices": [],
        "formatted_input": "CONTEXT:\n\nCURRENT SENTENCE:\nRavi bought a book."
      },
      {
        "sentence_idx": 1,
        "source": "He read it that night.",
        "translation": "<baseline translation>",
        "context": ["Ravi bought a book."],
        "context_indices": [0],
        "formatted_input": "CONTEXT:\nRavi bought a book.\n\nCURRENT SENTENCE:\nHe read it that night."
      }
    ],
    "metadata": {
      "context_strategy": "lastn_nonsemantic",
      "context_window": 1,
      "context_is_causal": true,
      "translation_uses_context": false,
      "translation_model_type": "sentence_level_pretrained_checkpoint"
    }
  },
  "error": null
}
```

## Limitations

This endpoint requires a real or mocked available baseline backend. It does not promise context-conditioned translations, semantic context selection, document-level metrics, or accuracy improvements. NLLB's own sentence-level baseline remains distinct from the context representation created here. No document relationships are inferred from Samanantar rows; a document sent directly to this endpoint is treated as one user-supplied ordered text only.
