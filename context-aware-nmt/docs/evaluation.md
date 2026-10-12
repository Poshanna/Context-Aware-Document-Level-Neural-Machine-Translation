# Quantitative Evaluation

## Architecture

`scripts/run_evaluation.py` reads only `data/processed/test.en-te.jsonl`; the source split is opened read-only. It uses deterministic reservoir sampling without replacement and records the seed, physical JSONL line indices, and source record IDs. Source and reference strings are preserved exactly as stored. The default is 50 examples; use 5 for a smoke evaluation. Sampling does not group or reorder records into documents.

The runner reuses the existing `TranslationService` and NLLB-200 pretrained checkpoint for the sentence-level baseline. Prefix-conditioned inference uses the existing context-conditioned service only for records with explicit context metadata: a non-empty `document_id`, a current `sentence_index`, and context entries carrying the same document ID and strictly earlier sentence indices. Entries from other documents, duplicate indices, and future/current sentence indices are excluded. The current processed Samanantar schema has none of these fields, so its examples receive `context_conditioned_not_available`; no context-conditioned translation is fabricated.

## Dataset Limitation

Samanantar test data is sentence-level. Adjacent records are not assumed to form documents and do not establish context relationships. The test split supports valid sentence-level baseline evaluation. A genuine document-level context evaluation requires a dataset with documented document boundaries and context, or a separately constructed and documented evaluation corpus. Prefix-conditioning isolated sentences is not genuine document-context evaluation.

## Phase 12B vs Phase 12C

- Phase 12B is the fixed 5-example smoke validation path: `--split test --seed 42 --max-samples 5`. It is the preserved smoke check for the existing evaluation bridge and artifact contract.
- Phase 12C is the larger bounded sentence-level baseline path, e.g. `--max-samples 25` or `--max-samples 50` with the same `test` split and default `seed=42`. It still uses the trusted sentence-level Samanantar adapter and does not infer document IDs or context from row adjacency.
- A 25- or 50-example run is still a bounded evaluation. It must not be presented as a final model-quality result, a benchmark claim, or a final quality estimate for deployment.

## Modes

- `baseline_sentence_level`: each source sentence is translated independently with no document context.
- `context_conditioned`: a translation is run only when explicit, same-document, prior context is present.
- `context_conditioned_not_available`: legitimate context is absent; the context-conditioned translation is `null` and its metrics are unavailable.

No future context is provided. The context window selects up to the most recent eligible prior indices.

## Metrics

BLEU is SacreBLEU **corpus BLEU**, not an average of sentence BLEU scores. It uses tokenizer `13a` for English targets and `none` for Telugu targets, with the library's unsmoothed corpus implementation. Consequently a very short corpus with no 4-gram may score zero even for an exact match; this is retained rather than smoothed to inflate small-run values. chrF++ is corpus-level SacreBLEU chrF with `char_order=6`, `word_order=2`, and `beta=2`; case, whitespace, and epsilon-smoothing behavior remain at library defaults. Input text receives no additional normalization.

BERTScore is opt-in via `--include-bertscore`, uses configured `BERT_SCORE_MODEL` (default `bert-base-multilingual-cased`), and may require its model to be present or downloaded. COMET is opt-in via `--include-comet`; it remains unavailable unless `ENABLE_COMET=1` and the optional `unbabel-comet` package/model are available. Optional metric outcomes are recorded as `computed`, `unavailable`, or `failed`; no metric is substituted silently.

Metrics are computed separately for baseline and context-conditioned predictions. With no legitimate context examples, context-conditioned metrics are unavailable. If any selected baseline translation fails, baseline metrics are marked failed rather than computed on a potentially biased partial subset.

## Reproducibility and Decoding

`--seed` (default 42) controls sampling and Python/NumPy/PyTorch seeds. Artifacts record language direction, test split path/count, sample indices and IDs, configured checkpoint, context strategy/window, device, timestamp, software versions, and generation settings. The existing NLLB backend uses `max_new_tokens=256` (configurable), `do_sample=False`, `num_beams=1`, and input truncation at its configured maximum. The locally cached checkpoint generation config is recorded when available. The checkpoint's `max_length` default is not changed; the backend's explicit `max_new_tokens` takes precedence when both are set.

## Artifacts

Each run creates `experiments/evaluation/<run-id>/` with:

- `summary.json`: metadata, selected set, limitations, and metric statuses/scores.
- `predictions.jsonl`: exact sources/references, translations, context availability, indices, and per-example status.
- `config.json`: run configuration and dataset/sampling information.
- `metrics.json`: separate metric results.
- `system_info.json`: device and software versions.
- `README.md`: concise run description and scientific caveat.

Generated predictions are local artifacts, not source data; do not commit large run artifacts.

## Running

From the project root, run the preserved Phase 12B five-example English→Telugu smoke evaluation:

```powershell
python scripts/run_samanantar_baseline.py --split test --source-lang en --target-lang te --max-samples 5 --seed 42
```

For Phase 12C, run a larger bounded sentence-level baseline sample such as 25 or 50 examples:

```powershell
python scripts/run_samanantar_baseline.py --split test --source-lang en --target-lang te --max-samples 25 --seed 42
```

The 25/50-example path is still a bounded evaluation. It is intended for broader smoke coverage and debugging, not for final model-quality reporting or benchmark publication. The script does not download datasets and fails clearly if the processed test split is absent. Basic runs calculate BLEU and chrF++ and do not invoke BERTScore or COMET. Each example invokes real NLLB inference; larger runs require adequate runtime, memory, and time. No model is fine-tuned.

## Limitations

The current Samanantar split does not provide document context, so its quantitative results compare sentence-level NLLB baseline predictions only. It cannot support a context-benefit claim. The context-conditioned mode requires explicit, independently documented document-context records; the runner does not infer these from adjacency. BLEU and chrF++ are lexical metrics and do not alone establish semantic adequacy.