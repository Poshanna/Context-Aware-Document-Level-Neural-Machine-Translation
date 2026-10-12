# Baseline NMT

## Objective

The baseline is sentence-level neural machine translation with no document context. It provides the non-contextual comparison pipeline for a later context-aware phase. This implementation uses pretrained checkpoints only; it does not train or fine-tune a model. No translation-quality metric is claimed here.

## Selected Model

NLLB-200 distilled 600M is the selected baseline fallback. Its public checkpoint is [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M). The configured checkpoint is `MODEL_NLLB200`; the default model key is `nllb200`.

IndicTrans2 remains the preferred project model when its checkpoint and required runtime/access are available. The configured IndicTrans2 checkpoint endpoints returned HTTP 401 in the inspected environment, and IndicTransToolkit is not installed, so it is not registered as a runnable backend here. NLLB-200 is public, but its weights are not currently cached locally. Check the checkpoint's current license and terms before use or redistribution.

## Supported Directions

The registered baseline supports English→Telugu (`en`→`te`) and Telugu→English (`te`→`en`). NLLB language-token mappings are `eng_Latn` and `tel_Telu`. Hindi and other directions are not registered in this Phase 4 baseline.

## Configuration

- `MODEL_NLLB200`: Hugging Face model ID; defaults to `facebook/nllb-200-distilled-600M`.
- `MODEL_NLLB200_LOCAL_PATH`: optional local checkpoint directory. When set, that path is used instead of the Hub ID.
- `TRANSLATION_SOURCE_LANGUAGE` / `TRANSLATION_TARGET_LANGUAGE`: smoke-script defaults; both can be overridden on the command line.
- `FORCE_DEVICE`: `auto` selects CUDA when available and CPU otherwise; `cpu` forces CPU; `cuda` requires available CUDA and returns a controlled error otherwise.
- `TRANSLATION_MAX_INPUT_LENGTH` / `TRANSLATION_MAX_OUTPUT_LENGTH`: tokenizer input limit and generation `max_new_tokens` limit.
- `TRANSLATION_SEED`: seed passed to PyTorch before generation. Decoding also uses `do_sample=False` and `num_beams=1`.
- `TRANSFORMERS_OFFLINE=1`: disable remote model resolution and require the checkpoint to be available locally.

Transformers and SentencePiece are already declared in `backend/requirements.txt`. No model weights are loaded at module import, test collection, or app startup. They load on the first real baseline translation request. With online mode enabled and no local checkpoint, Hugging Face Transformers may download the checkpoint at that time.

## Preprocessing Assumptions

The text endpoint performs the existing sentence segmentation and passes each sentence as a batch item. NLLB tokenization applies truncation at `TRANSLATION_MAX_INPUT_LENGTH`; output generation is bounded by `TRANSLATION_MAX_OUTPUT_LENGTH`. The service joins translated sentences with spaces. This is a sentence-level baseline and does not retrieve, prepend, or use prior-sentence context.

## Inference Flow

1. `POST /api/translate/text` validates the request and calls the translation service.
2. The service validates the model and language pair, then resolves the registered translator.
3. The NLLB tokenizer and model load lazily on the first request.
4. The tokenizer uses the source FLORES code; generation forces the target FLORES code.
5. Greedy deterministic generation returns sentence targets and the joined translated text.

A missing runtime, inaccessible checkpoint, unavailable offline files, or unavailable requested device produces a structured error. The service never substitutes a fake translation. The `context_aware` pipeline remains `not_implemented` in this phase.

## CPU and GPU

`FORCE_DEVICE=auto` uses CUDA if PyTorch reports it available and otherwise falls back to CPU. `FORCE_DEVICE=cuda` fails clearly when CUDA is unavailable. CPU inference is supported but can be substantially slower for this 600M-parameter checkpoint; no latency measurement has been made for this environment.

## Limitations

This stage does not translate documents, construct document relationships, retrieve context, or compute BLEU, chrF++, BERTScore, or COMET. Basic sentence splitting does not preserve all original whitespace/layout. NLLB output quality has not been evaluated in this project; no accuracy claim is made. The real checkpoint is not currently cached in the inspected environment, and this smoke test has not been run against it.

## Real Inference Smoke Test

Run from the repository root only when you are ready to load the pretrained checkpoint:

```powershell
python scripts/smoke_baseline_translation.py --source-lang en --target-lang te --text "Good morning."
python scripts/smoke_baseline_translation.py --source-lang te --target-lang en --text "శుభోదయం."
```

The script is not part of pytest and is not executed automatically. To require a local checkpoint, set `MODEL_NLLB200_LOCAL_PATH` and `TRANSFORMERS_OFFLINE=1` in `.env` before running it.
