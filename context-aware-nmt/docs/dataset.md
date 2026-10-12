# Dataset Documentation

Dataset counts are pending because no Samanantar corpus is currently installed locally. The scripts never download the full corpus or fabricate statistics. Once a local corpus is available, run `python scripts/prepare_samanantar.py` and inspect `experiments/dataset_preprocessing_report.json`.

## 1. Dataset Name

The primary dataset is **AI4Bharat Samanantar**, a sentence-level parallel corpus for Indian languages. This Phase 3 pipeline covers English ↔ Telugu (`en` ↔ `te`). Hindi and other language pairs are not part of this dataset pipeline.

## 2. Dataset Source

Download the English-Telugu subset manually from the [AI4Bharat Samanantar site](https://indicnlp.ai4bharat.org/samanantar/). See the Samanantar paper by Ramesh et al. (2021), [Samanantar: The Largest Public Parallel Corpora for English-Indian Languages](https://arxiv.org/abs/2104.03224). Review the downloaded corpus's license and upstream source credits before redistribution.

The existing prepared en-te run records upstream revision `ead34c5b22a0354d350e7dfdd6aece7df5a48f37`. New preprocessing reports record this in a structured `provenance.upstream` block. The separate `provenance.preprocessing` block records the local run ID, seed, and prepared split filenames. No checksum is recorded unless one is actually computed and verified.

Scripts do not download or extract the corpus. Configure `SAMANANTAR_DATA_PATH` in `.env`, or pass `--data-path` to the preprocessing script. Relative paths are resolved from the project root.

## 3. License

Follow the license and attribution terms distributed with the exact Samanantar release and its upstream corpora. Do not infer redistribution permission from this documentation alone.

## 4. Language Pair

The supported directions are English→Telugu (`en`→`te`) and Telugu→English (`te`→`en`). Each preprocessing run handles one direction and writes direction-specific split files. Unrelated pairs are filtered when language metadata is present; they are never added to the English-Telugu output.

## Directory Structure

```text
data/
├── raw/        # Manually downloaded source corpus; not committed
├── processed/ # Generated split files; not committed
└── samples/   # Tiny committed smoke-test data only
```

The full Samanantar corpus must not be committed. The repository ignores
`data/raw/` and `data/processed/` while allowing small files in `data/samples/`.

## 5. Dataset Format

The loader supports local UTF-8 paired text files, JSONL, and TSV. No layout is downloaded automatically.

**Paired text:**

```text
<SAMANANTAR_DATA_PATH>/en-te/train.en
<SAMANANTAR_DATA_PATH>/en-te/train.te
```

Each line number is a pair. Flat `train.en` and `train.te` files are also accepted. If the files have different row counts, each unpaired trailing row is counted as malformed and skipped.

**JSONL:** the loader looks for `<pair>.jsonl`, `train.<pair>.jsonl`, or `train.jsonl`. Each non-empty line must be an object using `src`/`tgt`, `source_text`/`target_text`, or language-code fields such as `en`/`te`. Optional language metadata (`source_language`/`target_language`, `source_lang`/`target_lang`, or `src_lang`/`tgt_lang`) is used to filter and orient records. Pair-named files determine the direction for generic `src`/`tgt` fields. Malformed JSON and unsupported record shapes are counted and skipped.

**TSV:** pass a `.tsv` file directly, or place `en-te.tsv`, `train.en-te.tsv`, or `train.tsv` in the configured folder. Headered files may use `en`/`te`, `src`/`tgt`, or `source_text`/`target_text` columns. Headerless rows must have exactly two columns in the configured direction. Malformed rows are counted and skipped.

**Official Hugging Face Parquet:** the compatible language-specific layout is:

```text
data/raw/samanantar/
└── te/
	├── <shard-name-1>.parquet
	└── <shard-name-2>.parquet
```

The loader reads all Parquet shards under `te/` in sorted path order. Each
shard must have `src` and `tgt` columns; `idx` and `data_source` are optional.
For this official Telugu subset, `src` is English and `tgt` is Telugu. The
loader maps `idx` to the normalized `id` when it is a non-negative integer,
otherwise it uses deterministic row order. A `te`→`en` request swaps the text
columns. Existing `data_source` strings are preserved; missing values default
to `samanantar`. Parquet reading uses PyArrow, declared in
`backend/requirements.txt`.

If a downloaded release uses another format, inspect it and extend the loader deliberately; do not assume its row order represents documents.

## 6. Fields

Every normalized Samanantar record contains these fields:

| Field | Type | Description |
|---|---|---|
| `id` | integer | Deterministic input-row index; skipped malformed rows may leave gaps. |
| `id` | integer | Parquet `idx` when valid; otherwise deterministic input-row index. |
| `source_text` | string | NFC-normalized text with whitespace collapsed. |
| `target_text` | string | NFC-normalized text with whitespace collapsed. |
| `source_language` | string | `en` or `te`, matching the requested direction. |
| `target_language` | string | `en` or `te`, matching the requested direction. |
| `data_source` | string | `samanantar`. |
| `data_source` | string | Original non-empty Parquet value, or `samanantar` when absent. |

JSONL output is UTF-8 and preserves Telugu characters (`ensure_ascii=False`).

## 7. Preprocessing

`scripts/prepare_samanantar.py` runs these steps in order:

1. Load the local paired text, JSONL, or TSV input and count malformed rows.
2. Filter language metadata to the requested English-Telugu direction.
3. Normalize Unicode to NFC, replace zero-width characters/BOM/NBSP with spaces, collapse whitespace, and trim.
4. Remove null or empty source/target text.
5. Remove exact duplicate `(source_language, target_language, source_text, target_text)` pairs, keeping the first.
6. Flag highly length-imbalanced pairs in `malformed_flagged_count`; do not discard them solely for imbalance.
7. Remove either side shorter than `--min-chars` (default 2) or longer than `--max-chars` (default 1000).

Cleaning is deterministic and does not aggressively filter Telugu script or valid short sentences.

## 8. Filtering

| Rule | Default | Behavior |
|---|---:|---|
| Minimum characters per side | 2 | Shorter records are removed. |
| Maximum characters per side | 1000 | Longer records are removed. |
| Empty text | Always | Removed after normalization. |
| Exact duplicate pair | Always | Keep first occurrence. |
| Length imbalance ratio ≥ 0.95 | Flag only | Retained unless another filter removes it. |

The minimum and maximum are configurable through `--min-chars` and `--max-chars`.

## 9. Train / Validation / Test Split

Splitting occurs after cleaning and deduplication. A local `random.Random(seed)` shuffles a copy of the records; the global random state is not changed. The default proportions are 80% train, 10% validation, and 10% test. Rounded train/validation sizes are used and the test split receives the remainder. Duplicate pairs cannot cross splits because duplicates are removed first.

Files are named `train.en-te.jsonl`, `val.en-te.jsonl`, and `test.en-te.jsonl`; a Telugu→English run uses the corresponding `te-en` names.

### Dataset Validation and Synthetic Test Data

Run `python scripts/validate_dataset.py` for English→Telugu, or add `--src-lang te --tgt-lang en` for Telugu→English. Validation checks all required fields and types, malformed JSON/records, empty values, exact language pair, `data_source`, within-split and cross-split duplicate pairs, UTF-8/Unicode validity, basic length constraints, and Telugu-script presence (warning only). Exit code 0 means no issues, 1 means errors, and 2 means warnings only. The report is written to `experiments/dataset_validation_report.json`.

A small controlled test file can be created without Samanantar:

```bash
python scripts/create_sample_dataset.py --synthetic --size 3
```

This writes `data/samples/synthetic_test.jsonl`. Every record is labeled `data_source="synthetic_test"`. These records are test-only, are not Samanantar statistics or evaluation data, and contain no document IDs or relationships. To extract a small sample from a processed real split, use `python scripts/create_sample_dataset.py --size 50`.

## 10. Random Seed

`DATASET_SEED` defaults to 42. An explicit `--seed` overrides the environment/configuration value. The actual seed is recorded in the preprocessing report; it is independent of `EVAL_SEED`.

## 11. Data Leakage Prevention

Exact duplicates are removed before splitting. The validator checks duplicate pairs within each split and overlap between every pair of splits. Do not re-split cached outputs during later evaluation.

## 12. Limitations

Samanantar is a sentence-level parallel corpus. It does not establish that consecutive rows belong to the same document. Source corpora may include code-mixed content and near-duplicates; this pipeline normalizes whitespace and removes exact duplicates only.

## 13. Document-Level Context Limitation

**Samanantar is a sentence-level parallel corpus.** Consecutive rows are **not guaranteed to belong to the same document**. This phase does not create document IDs, group neighboring sentences, or claim Samanantar provides reliable document-level context. The existing context-record helper keeps `document_id=None` and `context=[]` for Samanantar records. A genuine document/context source is required in a later phase.

## 14. Reproducibility

1. Manually download and extract the English-Telugu subset under `data/raw/samanantar/`.
2. Set `SAMANANTAR_DATA_PATH=./data/raw/samanantar` and `DATASET_SEED=42` in `.env`, or provide `--data-path`.
3. Run `python scripts/prepare_samanantar.py --seed 42 --sample-size 50`.
4. Validate with `python scripts/validate_dataset.py` (add `--src-lang te --tgt-lang en` for the reverse direction).
5. Read measured counts from `experiments/dataset_preprocessing_report.json`; never copy or estimate counts from another run.

The report includes measured input, parsed/valid, malformed, language-filtered, cleaning-removal, duplicate, and train/validation/test counts, plus language pair, seed, and source dataset. `removed_count` is the difference between input rows and final cleaned records. No report is produced when the configured local corpus is missing.

**Local corpus status:** `data/raw/` currently contains no Samanantar data. Full-dataset processing is pending manual dataset availability; no corpus statistics have been generated.
