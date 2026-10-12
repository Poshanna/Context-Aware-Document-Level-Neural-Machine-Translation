# Phase 3 — Dataset Setup & Samanantar Preprocessing — Specification

**Status**: Approved

---

## 1. Problem

The Context-Aware Document-Level NMT research project has a working FastAPI
backend foundation (Phase 2 — 42/42 tests pass, 10/10 ACs pass, independent
review pass) but has **zero dataset pipeline implemented**. There is no
Samanantar loader, no preprocessing pipeline, no train/val/test split, no
sample generator, no validator, no preprocessing statistics, and no dataset
documentation.

Without this phase, the following upcoming phases have nothing to operate on:
- Phase 4 (ML inference): needs a test split to run BLEU/COMET/BERTScore against
- Phase 5 (Context-retrieval pipeline): needs a reproducible test split to
  measure Δ of context-aware vs baseline
- Thesis writing: needs honest, reproducible dataset counts per split

## 2. Users / Stakeholders

1. **Researcher / Thesis author** — needs `experiments/dataset_preprocessing_report.json` with honest, measured statistics to cite in the methodology chapter.
2. **Future implementation work (Phases 4–6)** — needs `data/processed/*.jsonl` split files to run evaluation and train (if later fine-tuning) against.
3. **Independent evaluator / examiners** — must be able to reproduce train/val/test splits from the same raw Samanantar tarball using only the repo + a documented seed.
4. **Academic honesty auditor** — must verify that NO split numbers are fabricated, NO document IDs are invented, and NO Hindi/Telugu cross-language pairs are falsely claimed.

## 3. Goals

1. **Samantar loader**: Implement a clean ABC + concrete loader that reads Samanantar en↔te records from a user-configurable local path and emits a normalized `{id, source_text, target_text, source_language, target_language, data_source}` record shape.
2. **Preprocessing pipeline**: Load → en↔te filter → drop nulls/empties → whitespace-normalize → drop exact source+target duplicates → length filter → 80/10/10 deterministic split → write JSONL.
3. **Sample generator**: `scripts/create_sample_dataset.py --size N` to produce a tiny, optionally-committable `data/samples/samanantar_sample_{en,te}.txt` parallel pair (never commit > 50 sentences).
4. **Main entry**: `scripts/prepare_samanantar.py --sample-size N --seed S` orchestrating the whole pipeline and writing `experiments/dataset_preprocessing_report.json` with ONLY measured, actually-computed integers (no placeholder numbers).
5. **Validator**: `scripts/validate_dataset.py` checks fields, empties, duplicates, invalid langs, split overlap, stats; returns a clear JSON report.
6. **Context data structures**: Define `{document_id, sentence_id, context: [], source_sentence, target_sentence}` shape as a Pydantic/dataclass interface. Explicitly DO NOT fill in document_id or context values for Samanantar rows (sentence-level only; Samanantar rows are not guaranteed contiguous).
7. **Dataset documentation**: `docs/dataset.md` covering name, source, license, format, fields, preprocessing, filters, split, seed, leakage prevention, limitations, reproducibility. Explicitly state if statistics are "pending local download".
8. **Pytest suite**: Tests for loader, normalization, empty/null filtering, duplicate removal, length filtering, deterministic splitting, leakage, sample generation, validation. Existing 42 backend tests MUST continue to pass (no Phase 2 regression).

## 4. Non-Goals / Explicit Do-Nots (§13 user directive)

- **DO NOT** implement IndicTrans2 / NLLB-200 inference.
- **DO NOT** implement model fine-tuning.
- **DO NOT** implement context retrieval or SBERT embeddings.
- **DO NOT** implement BLEU / chrF++ / BERTScore / COMET metrics.
- **DO NOT** build a frontend dataset UI.
- **DO NOT** auto-download the Samanantar tarball. Scripts MUST fail-fast with a clear error if the expected files under `SAMANANTAR_DATA_PATH` are missing.
- **DO NOT** place raw or processed Samanantar data under Git.
- **DO NOT** fabricate document IDs. Samanantar is sentence-level parallel data; no row order guarantees document cohesion.
- **DO NOT** randomly group unrelated Samanantar rows into "documents".
- **DO NOT** modify raw dataset files.

## 5. Functional Requirements (FR)

### FR-1 Dataset Configuration
- FR-1.1 Add `SAMANANTAR_DATA_PATH` to `Settings` (env-driven, no hardcoded Windows/Linux paths).
- FR-1.2 Add `SRC_DIR` / `PROCESSED_DIR` / `SAMPLES_DIR` helpers in Settings or in loader that default to `data/raw/`, `data/processed/`, `data/samples/` relative to project root.
- FR-1.3 Add `DATASET_SEED` (default: 42) — separate from `EVAL_SEED` so we can vary them independently.
- FR-1.4 Update `.env.example` with the three new env fields using placeholder defaults; NO real credentials.
- FR-1.5 Existing `.env` files (root + backend/) updated with these new placeholders so uvicorn still starts (no required SecretStr changes needed).

### FR-2 Samanantar Loader (Normalizer)
- FR-2.1 New file: `backend/app/preprocessing/dataset_loader.py`.
- FR-2.2 Defines a dataclass or Pydantic model `ParallelRecord` exactly with fields: `id` (int), `source_text` (str), `target_text` (str), `source_language` (str ∈ {en,te,hi}), `target_language` (str ∈ {en,te,hi}), `data_source` (str, default "samanantar").
- FR-2.3 `SamanantarLoader(data_path: Path | str, src_lang="en", tgt_lang="te")` class. Constructor validates path exists; else raises a clear `FileNotFoundError` with a message saying "Please download Samanantar en→te tarball and extract under SAMANANTAR_DATA_PATH per docs/dataset.md".
- FR-2.4 Loader supports two storage-layout strategies because Samanantar format is NOT assumed:
  - Strategy A — Parallel `.txt` files: `en-te/train.en` + `en-te/train.te` (one sentence per line, zip pairwise by line number).
  - Strategy B — `.jsonl` records: `{"src":"...","tgt":"..."}` or `{"en":"...","te":"..."}` per line.
  - Loader auto-detects which files actually exist under `data_path`; raises descriptive error if neither layout is present.
- FR-2.5 `.load() -> list[ParallelRecord]` with deterministic `id = 0..N-1` based on file order.
- FR-2.6 `.normalize(record: ParallelRecord) -> ParallelRecord` — whitespace normalization per FR-3.2 on both source + target; id preserved.

### FR-3 Data Cleaning
- FR-3.1 New module: `backend/app/preprocessing/dataset_cleaner.py` OR functions inside `dataset_loader.py` (dealer's choice — keep it single-file if < 250 LOC).
- FR-3.2 Whitespace normalize: strip leading/trailing, collapse internal runs of `\s+` to single space, normalize zero-width space U+200B and friends.
- FR-3.3 Remove null / empty source OR target: drop any record where `source_text.strip() == ""` OR `target_text.strip() == ""` or either is None/null.
- FR-3.4 Remove exact duplicate `(source_language, target_language, source_text, target_text)` pairs. Keep first occurrence.
- FR-3.5 Detect malformed: length-ratio |len(src)-len(tgt)| / max(len(src),len(tgt),1) > 0.95 flag as suspicious (only logged, not dropped).
- FR-3.6 Length filter defaults: min_chars=2, max_chars_source=1000, max_chars_target=1000; all configurable via kwargs.
- FR-3.7 Return a `CleaningReport` dataclass with actual measured ints: `input_count`, `null_removed_count`, `duplicate_removed_count`, `length_filtered_count`, `malformed_flagged_count`, `output_count`.

### FR-4 Train / Validation / Test Split
- FR-4.1 New module `backend/app/preprocessing/dataset_splitter.py` with `split_dataset(records, train=0.8, val=0.1, test=0.1, seed=42)` function.
- FR-4.2 Shuffle with `random.Random(seed).shuffle()` (NOT global random module state pollution) — local RNG instance only.
- FR-4.3 No split overlap: every pair `(src_normalized, tgt_normalized)` appears in exactly 0 or 1 splits. A pair that, after cleaning, duplicates an earlier one that ended up in a different split MUST be considered a duplicate and dropped OR assigned to the same split as its twin (dealer's choice — enforce by deduping BEFORE split).
- FR-4.4 Returns `Splits(train=[...], val=[...], test=[...])` dataclass.
- FR-4.5 Also writes to disk as JSONL: `data/processed/train.en-te.jsonl`, `val.en-te.jsonl`, `test.en-te.jsonl` when called via scripts (script writes; lib function only returns tuples).

### FR-5 Sample Generator Script
- FR-5.1 `scripts/create_sample_dataset.py` with argparse: `--size N`, `--source data/processed/train.en-te.jsonl` (default), `--output-dir data/samples`.
- FR-5.2 Takes the FIRST N rows of the source split (NOT random — deterministic sample order so committed samples are stable).
- FR-5.3 Writes two parallel files: `samanantar_sample.en.txt` + `samanantar_sample.te.txt` (one sentence per line).
- FR-5.4 Emits a summary printout: how many rows written, total source/target chars.
- FR-5.5 Rejects `--size > 10000` unless an explicit `--i-accept-the-large-size` flag is passed (project_rules.md §1.4 only allows tiny committed samples).

### FR-6 Main Preparation Script
- FR-6.1 `scripts/prepare_samanantar.py` with argparse: `--sample-size N` (default 0 = no sample), `--seed S` (default 42), `--data-path override` (falls back to Settings.SAMANANTAR_DATA_PATH).
- FR-6.2 Pipeline: Load → en↔te filter → Clean → Dedup (already inside Clean) → Length filter → Split → Write JSONL → (opt) Write sample → Write `experiments/dataset_preprocessing_report.json`.
- FR-6.3 Fail-fast if SAMANANTAR_DATA_PATH does not exist: print instructions to download, exit(1).
- FR-6.4 Report JSON MUST contain: raw_count, language_filtered_count, null_removed_count, duplicate_removed_count, length_filtered_count, malformed_flagged_count, final_count, train_count, validation_count, test_count, random_seed, generation_ts_utc, source_data_path_sha256_of_manifest (or "pending" if raw not hashed). Every integer field must be the actual measured int — NEVER a placeholder like 0 when the real run produced a different number.

### FR-7 Dataset Validation Script
- FR-7.1 `scripts/validate_dataset.py` — validates the processed JSONL split files.
- FR-7.2 Checks:
  1. Required fields present on every record.
  2. No empty `source_text` / `target_text`.
  3. No duplicate `(src,tgt)` pairs across splits (leakage check).
  4. `source_language ∈ {en,te,hi}` and `target_language ∈ {en,te,hi}`.
  5. Length statistics: min/max/mean chars for src/tgt per split.
  6. Overlap: count of records that appear in > 1 split (should be 0).
- FR-7.3 Output: stdout summary + `experiments/dataset_validation_report.json` with structured `{ok: bool, issues: [], summary: {train_count, val_count, test_count, ...}}`.
- FR-7.4 Exit code: 0 if ok, 1 if severe issues (missing fields, leakage), 2 on warnings only.

### FR-8 Context Interface (Document-Level Data Structures)
- FR-8.1 Define in `backend/app/preprocessing/context_record.py` a Pydantic/dataclass:
  ```python
  @dataclass
  class ContextRecord:
      document_id: str | None  # None for sentence-level-only corpora (Samanantar)
      sentence_id: int
      context: list[str]       # empty [] for now; filled in later phase by context retriever
      source_sentence: str
      target_sentence: str
  ```
- FR-8.2 Helper `record_from_parallel(pr: ParallelRecord) -> ContextRecord` sets `document_id=None`, `context=[]`, `sentence_id = pr.id`.
- FR-8.3 Unit test that a Samanantar-derived ContextRecord has `document_id is None` and `context == []` (never fabricate doc ids).

### FR-9 Dataset Documentation
- FR-9.1 `docs/dataset.md` contains: 1. Dataset name, 2. Dataset source (URL), 3. License (AI4Bharat MIT-like), 4. Language pair (en↔te + Hindi planned), 5. Format (parallel .txt OR jsonl), 6. Fields, 7. Preprocessing steps (ordered list), 8. Filtering rules (char limits, dedupe), 9. Split percentages, 10. Random seed default, 11. Data leakage prevention (dedup before split, no cross-split overlap), 12. Limitations (sentence-level only, no document order guarantees), 13. Document-level context limitation — Samanantar rows are NOT guaranteed to be contiguous documents so we leave document_id=None; document-level data must come from separate corpus later, 14. Reproducibility instructions (exact bash commands to reproduce splits from downloaded tarball).
- FR-9.2 If raw Samanantar has not been downloaded on THIS machine, every "count" in docs/dataset.md explicitly says "Pending local download — see `experiments/dataset_preprocessing_report.json` after running `python scripts/prepare_samanantar.py`".

### FR-10 Testing
- FR-10.1 New test file: `backend/tests/test_dataset_pipeline.py` with ≥ 20 tests covering:
  1. Loader: Parallel txt files (Strategy A) — 5-row synthetic fixture → ids 0..4.
  2. Loader: JSONL fixture (Strategy B) → same record count.
  3. Loader: Missing path → clear FileNotFoundError with "download Samanantar" substring.
  4. Loader: No valid layout → descriptive error.
  5. Normalize: Whitespace collapse, strip, zero-width-space removal.
  6. Cleaning: null/empty drop removes rows.
  7. Cleaning: duplicates removed (2 same pairs → keep 1).
  8. Cleaning: length filter removes short/long.
  9. Cleaning report ints add up correctly: input - (nulls+dups+length) ≈ output (allow for rounding of multi-stage order).
  10. Split: 80/10/10 % approximate counts on a 100-row fixture.
  11. Split: same seed twice → identical order (compare id lists).
  12. Split: seed 42 vs seed 43 → different order (statistically near-1 test).
  13. Split: no cross-split duplicate pairs (create 10 identical rows → assert total across splits ≤ 1 after dedupe).
  14. Leakage: after split with dedupe, `set((s,t))` across splits pairwise are disjoint.
  15. Sample script: `--size 10` creates two files with 10 lines each.
  16. Sample script: refuses size > 10000 without the flag.
  17. Validate script: good dataset → ok=True, exit 0.
  18. Validate script: empty text injected → bad field, reports issue.
  19. Validate script: duplicated pair across train+test → reports leakage, exit 1.
  20. ContextRecord: Samanantar-derived → document_id is None, context == [].
- FR-10.2 Existing 42 backend foundation tests still pass.
- FR-10.3 Total new tests ≥ 20; cumulative tests: 62+ passing.

## 6. Non-Functional Requirements (NFR)

- NFR-1 **Zero pandas dependency**: Prefer stdlib `json`, `pathlib`, `collections.Counter`, `random.Random()` over pandas — numpy already comes with torch so numpy arrays are OK, but pandas adds a 30–50 MB dep that is not needed for JSONL line counts.
- NFR-2 **Deterministic output**: Given same input files + same seed, byte-identical output JSONL files (ids and line order identical across runs; json.dumps with `sort_keys=True, ensure_ascii=False` so Telugu characters are readable).
- NFR-3 **Streaming-friendly**: Loader should iterate files line-by-line when possible (not load 50 GB at once — though for phase 3 in-memory is OK if the sample mode is the default; at least allow generator-based yielding).
- NFR-4 **Clear error messages**: Every `FileNotFoundError` / `ValueError` raised by scripts includes an actionable remedy (what file to download, to which path, which doc section to read).
- NFR-5 **Clean architecture boundaries (Phase 2 rubric 5/5 must not regress)**: `preprocessing/*.py` — NO FastAPI imports, NO `from app.api` imports. Scripts in `scripts/*.py` may import `from app.core.config import get_settings` (allowed).
- NFR-6 **Secrets**: NO new `SecretStr` required (this phase has no credentials). Grep of codebase must show no new API keys / tokens.

## 7. Constraints & Dependencies

- C-1: Python 3.11 (per backend baseline)
- C-2: stdlib + only what is already in requirements.txt (no new pip installs)
- C-3: `data/raw/` and `data/processed/` are gitignored per rules §6.2; scripts must work if those dirs contain nothing.
- C-4: Samanantar format: loader must NOT assume until files are on disk — Strategy A/B both supported with auto-detect and graceful failure.
- C-5: `experiments/dataset_preprocessing_report.json` path is committed (file itself gitignored by experiments/.gitkeep semantics? — up to current .gitignore; at minimum script must create parent dir if missing).

## 8. Assumptions

- A-1: The user will manually download Samanantar tarball from AI4Bharat website. Scripts will NOT do this.
- A-2: The Samanantar en↔te subset is either (a) a folder with `train.en` / `train.te` parallel text files OR (b) a `.jsonl` file. If the actual format differs, the loader auto-detect code will error with "layout not supported" and we will add Strategy C in a later update (out of scope for phase 3).
- A-3: For tests, we write tiny synthetic fixtures to `tmp_path` (pytest built-in) so NO real Samanantar data is ever touched by CI.

## 9. Open Questions

None — all ambiguity explicitly resolved: Strategy A/B auto-detect with fail-fast if neither; document_id=None for Samanantar; sample <50 committable; seed 42 default; JSONL output format.

---

## 10. Acceptance Criteria (AC)

### AC-1: Settings + .env.example extended
- **Type**: rule
- **Pass condition**: `Settings` has `SAMANANTAR_DATA_PATH: str | None = None` and `DATASET_SEED: int = 42` fields; `.env.example` L ≥ 1 new block of 2–4 lines showing `SAMANANTAR_DATA_PATH=./data/raw/samanantar`, `DATASET_SEED=42` placeholders.
- **Evidence source**: `config.py` field declarations + `.env.example` grep of new lines.

### AC-2: Samanantar loader normalizes records
- **Type**: rule
- **Pass condition**: Given two 5-line parallel `.en` + `.te` files in tmp_path, `SamanantarLoader(tmp_path).load()` returns list of 5 ParallelRecord with id ∈ {0..4}, source_language="en", target_language="te", data_source="samanantar", and record[2].source_text matches the 3rd line of the `.en` file exactly.
- **Evidence source**: `test_loader_strategy_a_parallel_txt` in test_dataset_pipeline.py.

### AC-3: Cleaning pipeline drops empties + dupes + shorts + longs correctly
- **Type**: rule
- **Pass condition**: Synthetic fixture with (null rows=2, duplicates=3 exact (src,tgt) pairs, 1 short 0-char src, 1 tgt with 1100 chars) → CleaningReport.null_removed_count=2, CleaningReport.duplicate_removed_count=3, CleaningReport.length_filtered_count=2.
- **Evidence source**: `test_cleaning_report_counts` + `test_cleaning_drops_empties_and_dupes`.

### AC-4: Split is deterministic and non-overlapping
- **Type**: rule
- **Pass condition**: (a) `split_dataset(X, seed=42)` run twice → identical id order; (b) 100-row fixture → train=80±2, val=10±2, test=10±2; (c) after dedup-before-split, `set(train_pairs) ∩ set(val_pairs) == set()` and same for train∩test, val∩test.
- **Evidence source**: `test_split_deterministic_same_seed`, `test_split_proportions_80_10_10`, `test_split_no_cross_pair_overlap`.

### AC-5: Main script writes report.json with ONLY measured ints
- **Type**: rule
- **Pass condition**: Run `python scripts/prepare_samanantar.py --data-path <tmp_fixture_100rows> --seed 42` → `experiments/dataset_preprocessing_report.json` parses, has keys raw_count=100, language_filtered_count, null_removed_count, duplicate_removed_count, length_filtered_count, final_count, train_count, validation_count, test_count, random_seed=42; every value is int, no value is hardcoded placeholder 0 unless the stage actually removed 0 rows.
- **Evidence source**: Test calls script via `subprocess.run` or function import, asserts dict values.

### AC-6: Validation script catches leakage + empty fields
- **Type**: rule
- **Pass condition**: (a) Good splits → `ok: true, exit 0`. (b) Inject one record with `source_text=""` → ok:false, issues[i].code=="empty_source_text". (c) Copy a test record also into train.jsonl → ok:false, issues[i].code=="cross_split_leak", exit 1.
- **Evidence source**: `test_validate_ok`, `test_validate_catches_empty`, `test_validate_catches_leakage`.

### AC-7: Sample generator respects size guardrails
- **Type**: rule
- **Pass condition**: (a) `--size 10` → exactly 10 lines in both output .txt files. (b) `--size 10001` → exit(2) or raises SystemExit with stderr containing "large-size" substring until `--i-accept-the-large-size` passed.
- **Evidence source**: `test_sample_creates_parallel_txt_files`, `test_sample_refuses_large_size`.

### AC-8: ContextRecord for Samanantar has no fabricated document_ids
- **Type**: rule
- **Pass condition**: `record_from_parallel(pr)` → `document_id is None`, `context == []`, `sentence_id == pr.id`. Test with 3 different ParallelRecords.
- **Evidence source**: `test_context_record_samanantar_has_null_document_id`.

### AC-9: docs/dataset.md exists and covers all 14 FR-9.1 sections
- **Type**: rule
- **Pass condition**: `docs/dataset.md` contains markdown headings for at least 12 of the 14 sections from FR-9.1; explicitly includes a paragraph mentioning "Samanantar is sentence-level parallel data; no document-order guarantees, so document_id is left None — we do not fabricate documents".
- **Evidence source**: `test_dataset_doc_sections_present` (grep for section headings) OR manual headings count check in review.

### AC-10: No Phase 2 regression. All foundation tests still pass.
- **Type**: rule
- **Pass condition**: `python -m pytest backend/tests/ -q` — 42 original tests still pass, ≥ 20 new tests pass. Total passed ≥ 62.
- **Evidence source**: Pytest terminal output.

### AC-11: Clean architecture boundaries preserved (rubric)
- **Type**: rubric
- **Dimension**: Preprocessing modules pure of FastAPI/web-app concerns
- **Scale**: 1–5. Threshold: ≥ 4.
- **Anchors**:
  1: `dataset_loader.py` imports `fastapi.Request/Response` or `from app.api.*`.
  3: Loader has no FastAPI imports but `prepare_samanantar.py` router-imports something.
  4: All preprocessing modules have 0 `app.api` or `fastapi` imports; scripts only import config/schemas minimally.
  5: Loader, cleaner, splitter, context_record dataclasses are 100% usable from a bare `python -c` with no app loaded; no FastAPI; no DB. Scripts use get_settings() for path but do not touch routers.
- **Evidence source**: Grep for `import fastapi` in preprocessing; import in scripts. Run `python -c "from app.preprocessing.dataset_loader import SamanantarLoader; from app.preprocessing.dataset_splitter import split_dataset"` → no traceback, no app import.
