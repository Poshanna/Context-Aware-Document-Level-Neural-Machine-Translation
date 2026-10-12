# Context-Conditioned Translation Experiment

## 1. Baseline sentence-level translation

The verified baseline remains the pretrained sentence-level NLLB-200 model. Each current sentence is translated independently without any document-level conditioning. This is the control condition for the experiment.

## 2. Metadata-only context pipeline

The document-context pipeline retrieves recent preceding sentences using the deterministic last-N causal retriever. The context is computed and returned in the API response metadata, but the same sentence-level NLLB backend still receives only the current sentence.

This mode is intentionally called `metadata_only` and sets:

- `context_mode`: `metadata_only`
- `translation_uses_context`: `false`
- `context_used_for_translation`: `false`
- `translation_model_type`: `sentence_level_pretrained_checkpoint`

This preserves the verified baseline behaviour while making retrieval observability explicit.

## 3. Prefix-conditioned experiment

The experimental mode is not a native document-level NMT model. It is an input-conditioning approach around a sentence-level pretrained checkpoint.

The exact input format used for the model is:

```
CONTEXT:
<previous sentence 1>
<previous sentence 2>

CURRENT SENTENCE:
<current sentence>
```

This is deterministic and testable. The current sentence is always present. Only preceding sentences are included. No future-sentence leakage occurs.

## 4. Exact architecture

The architecture is:

1. sentence splitter
2. causal last-N retrieval
3. context formatter
4. optional prefix-conditioned input assembly
5. existing NLLB-200 backend call
6. translation result returned with metadata

This keeps the backend unchanged and isolates the experiment to input conditioning.

## 5. NLLB limitation

NLLB-200 is a sentence-level pretrained checkpoint. It is not a document-aware translation model and does not natively learn long-range document context in the way a document-level model would. The prefix-conditioned mode therefore should be described as an experimental input-conditioning method around a sentence-level checkpoint, not as native document-level machine translation.

## 6. Difference between retrieval and translation use

The project distinguishes between:

- retrieval: selecting prior sentences from the document
- translation use: whether those prior sentences are actually included in the model input

The document-context pipeline can retrieve context even when `metadata_only` is used. In that case, retrieval exists, but the backend input remains sentence-only and `translation_uses_context` is false.

## 7. Future path toward genuine context-aware modelling

A genuine context-aware translation system would need one of the following:

- a document-level model trained with preceding context on the source side
- a context-aware encoder-decoder architecture
- a fine-tuning stage with document-level supervision
- explicit alignment between source document context and target sentence generation

This project is intentionally not claiming such capability before it is measured and validated.

## 8. Limitations

- The backend remains a sentence-level pretrained checkpoint.
- Context is inserted as a formatted prefix, not as a learned document state.
- There is no claim of improved translation quality without an explicit evaluation run.
- This is an experiment for probing input conditioning only.

## 9. Reproducibility

The experiment records the following when it is used:

- model name: `nllb200`
- source language
- target language
- context window
- context mode (`metadata_only` or `prefix_conditioned`)
- generation configuration, when provided
- exact formatted model input for inspection

No secrets are required.

## 10. Comparison scope

This experiment is intended to isolate the effect of prefix-conditioned context on the exact same sentence-level backend. It does not replace the baseline, and it does not claim final quality gains without measured evaluation.
