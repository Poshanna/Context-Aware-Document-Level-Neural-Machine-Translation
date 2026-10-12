# Samanantar Trusted Adapter

## Role

`SamanantarPreparedAdapter` is a read-only Phase 12A boundary over the project's existing `data/processed/{train,val,test}.en-te.jsonl` files. It converts a bounded prefix of validated records to Phase 11 `sentence_level` records. It does not run preprocessing, write data, translate, calculate metrics, or modify the Phase 6/7 evaluator.

The prepared files' observed schema is exactly:

| Field | Required type | Adapter use |
|---|---|---|
| `id` | non-negative integer | Actual prepared-record identity; not a document ID |
| `source_text` | non-empty string | English source sentence |
| `target_text` | non-empty string | Telugu reference sentence |
| `source_language` | `en` | Validated source language |
| `target_language` | `te` | Validated target language |
| `data_source` | `samanantar` | Validated source label |

This schema was confirmed from the preprocessing writer and the first five metadata-only rows in each local split. Those rows contain no document, sentence-order, or context fields.

## Provenance Boundary

The adapter inherits `TrustedDatasetAdapter` and issues `verified_dataset` provenance only after validating the actual JSONL record. Provenance includes:

- Dataset name: `AI4Bharat Samanantar`.
- Split: `train`, `val`, or `test`.
- Record ID: the actual `id` in the processed JSONL.
- Source/reference locations: the local processed file path, physical line number, and field name.
- Alignment location: the same record's ID, language fields, and paired source/reference fields.
- Adapter name: `samanantar_prepared_jsonl_v1`.

The preprocessing report records the upstream Samanantar revision already used by this project: `ead34c5b22a0354d350e7dfdd6aece7df5a48f37`, with source URL `https://indicnlp.ai4bharat.org/samanantar/`, language pair `en-te`, and source/target codes `en`/`te`. This is stored in `provenance.upstream` and read by the adapter; the adapter does not hard-code or silently substitute the revision. It fails closed if the upstream block or its revision is missing or malformed.

The local preprocessing run is separate in `provenance.preprocessing`: `run_id`, `seed`, and a split-to-filename map for `train.en-te.jsonl`, `val.en-te.jsonl`, and `test.en-te.jsonl`. The adapter verifies these values against the existing report's timestamp, seed, and expected prepared filenames. `DatasetProvenance.dataset_revision` contains the upstream revision; `SamanantarSample.preprocessing_run_id` and `preprocessing_seed` identify the local transformation. Each record additionally identifies its prepared split, actual source `id`, file path, physical line, and source/reference field locations. No content checksum is claimed or invented.

The trusted adapter boundary is an application trust contract, not cryptographic verification. This implementation binds provenance to the actual prepared file path, row line, field names, source ID, split, upstream revision, and preprocessing run report. The local records are accepted only after the report identifies AI4Bharat Samanantar, the expected source URL and `en-te` pair, a well-formed upstream SHA, a consistent run ID/seed, and the exact expected filenames. No real-data adapter call can authorize a sentence if required record fields or report metadata are missing or inconsistent.

## Record and Exclusion Rules

Each accepted record has `track=sentence_level`, `current_text_unit=sentence`, `reference_unit=sentence`, `alignment_status=sentence_aligned`, and `reference_availability=available`. The current source/reference text is copied from `source_text`/`target_text` without repair. The record's `id` is preserved as `provenance.record_id` only; no document ID, sentence ID, or context position is created.

Context is always `context_unavailable`, with no preceding items, no strategy, and no context-conditioned translation. Samanantar is sentence-level. Adjacent rows are not related by this adapter and never become context.

Rows are excluded with deterministic reasons, including `blank_line`, `invalid_json`, `record_not_object`, `missing_or_invalid_record_id`, `duplicate_record_id`, `missing_or_invalid_source_text`, `missing_or_invalid_reference_text`, `invalid_language_pair`, `invalid_data_source`, `unexpected_document_context_fields`, `missing_or_unexpected_fields`, and `provenance_unavailable`. Malformed or unexpected data is not silently repaired. Each exclusion stores physical line number, record ID when valid, and reason.

## Split and Sampling

The adapter supports only the existing `train`, `val`, and `test` split filenames. It reads line-by-line and stops after at most `max_samples` physical JSONL lines; this is a deterministic prefix sample, not a random sample. `rows_inspected` includes malformed/blank lines within that bound. The preprocessing report supplies expected split counts, reported separately from the bounded sample count. A missing split or preprocessing report fails clearly; the adapter does not download or regenerate files.

## Relationship to Evaluation

The Phase 6/7 evaluator remains unchanged. It reads the existing sentence-level JSONL, invokes translation, computes its configured corpus metrics, and writes its existing artifacts. Phase 12A only constructs validated Phase 11 records with trusted provenance; it does not call the evaluator, compute BLEU/chrF++, or run a model. A later phase must explicitly integrate the adapter output and define corpus-level metric aggregation.

Phase 12B is the preserved 5-example smoke validation path. Phase 12C is the larger bounded sentence-level baseline path using the same trusted sentence-level adapter, with `--max-samples 25` or `--max-samples 50`. These are bounded validations only and are not a final model-quality result.

Phase 11 remains the separate methodology/report layer. Its metric gate requires verified provenance and matching sentence-level references. Synthetic test fixtures and the Phase 11 demo remain ineligible for research metrics; they do not pass through this real-data adapter.

## Limitations

- This adapter establishes only the sentence-level pair represented by one prepared row.
- It does **not** establish document-level context or relationships between rows.
- It does not validate translation quality, language identification beyond exact language fields, or upstream Samanantar rights.
- The local preprocessing-run identifier does not identify/pin the upstream corpus bytes.
- The adapter reads only the requested prefix; exclusions later in a split are not summarized unless `max_samples` reaches them.

## Read-Only Inspection CLI

From the repository root:

```powershell
python scripts\inspect_samanantar_adapter.py --split test --max-samples 5
```

The CLI prints dataset/local-run metadata, split, rows inspected, accepted/excluded counts, sorted exclusion-reason counts, and up to ten record summaries. It prints only text-presence flags and provenance locations/IDs, never sentence text. It has no dataset download, model, translation, or metric path.

Unit tests use only temporary synthetic JSONL/report fixtures through a test subclass. They do not read the real prepared splits.
