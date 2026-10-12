# Phase 3 Tasks — Dataset Setup & Samanantar Preprocessing

Each task → Status (pending/in_progress/blocked/completed/cancelled) + TRs + Completion Evidence with absolute file links.

---

## Task 1: Extend Settings + .env.example for dataset configuration

**Priority**: high
**Depends on**: (none — foundations already complete)
**Status**: pending

### Description
Add dataset configuration fields to `Settings` (SAMANANTAR_DATA_PATH, DATASET_SEED) and update `.env.example` with placeholder values. Ensure existing `.env` files (root + backend/) also receive these new fields so uvicorn startup does not regress (no required fields added without defaults).

### Test Requirements (TRs)

#### TR 1.1: config.py has SAMANANTAR_DATA_PATH and DATASET_SEED fields
- **Type**: rule
- **Pass condition**: `Settings()` with no env → `samanantar_data_path is None` (or default relative path), `dataset_seed == 42`.
- **Evidence**: `test_dataset_settings_defaults` in test_dataset_pipeline.py.

#### TR 1.2: .env.example has dataset section with 2-4 new lines
- **Type**: rule
- **Pass condition**: Grep `.env.example` for `SAMANANTAR_DATA_PATH` → finds at least 1 line with placeholder default like `./data/raw/samanantar` (NOT a Windows absolute path). Grep for `DATASET_SEED=42` finds 1 line.
- **Evidence**: Manual grep output in review.

### Completion Evidence
- [ ] [config.py](file:///c:/Users/MEDHA%20TRUST/Downloads/Aiml/Major%20project/context-aware-nmt/backend/app/core/config.py) Settings class contains 2 new fields.
- [ ] [.env.example](file:///c:/Users/MEDHA%20TRUST/Downloads/Aiml/Major%20project/context-aware-nmt/.env.example) contains dataset section.
- [ ] Both [root .env](file:///c:/Users/MEDHA%20TRUST/Downloads/Aiml/Major%20project/context-aware-nmt/.env) and [backend .env](file:///c:/Users/MEDHA%20TRUST/Downloads/Aiml/Major%20project/context-aware-nmt/backend/.env) have the new fields (copied from .env.example).

---

## Task 2: Implement Samanantar Loader (ParallelRecord + Strategy A/B auto-detect)

**Priority**: high
**Depends on**: Task 1 (for data_path default)
**Status**: pending

### Description
Create `backend/app/preprocessing/dataset_loader.py` with:
- `ParallelRecord` dataclass (id, source_text, target_text, source_language, target_language, data_source)
- `SamanantarLoader(data_path, src_lang, tgt_lang)` constructor that validates path exists
- Auto-detect Strategy A (parallel en-te/train.en + train.te) vs Strategy B (en-te.jsonl)
- `.load() -> list[ParallelRecord]`
- `.normalize(pr) -> pr` (whitespace normalize)
- Error messages include "Please download Samanantar en->te tarball per docs/dataset.md"

### Test Requirements
#### TR 2.1: Strategy A — 5 parallel rows load with ids 0..4
- **Type**: rule
- **Pass condition**: test fixture writes `train.en` (5 lines) + `train.te` (5 lines) to `<tmp>/saman/en-te/`; `SamanantarLoader(tmp/"saman").load()` returns len=5; record[2].source_text == en_lines[2] exactly.
- **Evidence**: `test_loader_strategy_a_parallel_txt`

#### TR 2.2: Strategy B — JSONL records load correctly
- **Type**: rule
- **Pass condition**: `{"src":"...","tgt":"..."}` per line fixture → loaded ParallelRecords match.
- **Evidence**: `test_loader_strategy_b_jsonl`

#### TR 2.3: Missing path raises FileNotFoundError with actionable message
- **Type**: rule
- **Pass condition**: exc value contains "download Samanantar" substring.
- **Evidence**: `test_loader_missing_path_error`

#### TR 2.4: Empty dir with no layout → descriptive error
- **Type**: rule
- **Pass condition**: raises ValueError mentioning "Strategy A or B layout not found".
- **Evidence**: `test_loader_no_valid_layout`

#### TR 2.5: Normalize collapses whitespace + ZWSP removal
- **Type**: rule
- **Pass condition**: source="  hello\u200b   world  " → "hello world". target="   a\n\n\nb   " → "a b".
- **Evidence**: `test_normalize_whitespace_and_zwsp`

### Completion Evidence
- [ ] [dataset_loader.py](file:///c:/Users/MEDHA%20TRUST/Downloads/Aiml/Major%20project/context-aware-nmt/backend/app/preprocessing/dataset_loader.py) exists with SamanantarLoader + ParallelRecord.
- [ ] 5 tests pass (2.1 → 2.5).

---

## Task 3: Implement Data Cleaner with CleaningReport

**Priority**: high
**Depends on**: Task 2
**Status**: pending

### Description
Add cleaning functions (either inside dataset_loader.py as helpers or new dataset_cleaner.py if >250 LOC). Operations:
1. Drop null/empty src or tgt
2. Whitespace normalize (already in loader normalizer)
3. Drop exact (src_lang, tgt_lang, src_text, tgt_text) duplicates, keep first
4. Flag (don't drop) malformed via length-ratio 0.95+
5. Length filter: min_chars, max_chars configurable (default 2, 1000)
Return CleaningReport with counts: input, null_removed, duplicate_removed, length_filtered, malformed_flagged, output.

### Test Requirements
#### TR 3.1: Empties dropped count correct
- **Type**: rule
- **Pass condition**: 2 empty src + 1 empty tgt in 10-row fixture → null_removed_count == 3.
- **Evidence**: `test_cleaning_drops_empties`

#### TR 3.2: Duplicates dropped count correct
- **Type**: rule
- **Pass condition**: 3 identical (src,tgt) pairs among 10 → duplicate_removed_count == 2 (keeps 1).
- **Evidence**: `test_cleaning_drops_exact_duplicates`

#### TR 3.3: Length filter count correct
- **Type**: rule
- **Pass condition**: 1 src has 1 char (below min=2) + 1 tgt has 1100 chars (above max=1000) → length_filtered_count == 2.
- **Evidence**: `test_cleaning_length_filter`

#### TR 3.4: Counts add up (accounting for order)
- **Type**: rule
- **Pass condition**: input_count - (null + dup + length) == output_count (order: drop nulls first, then dedupe, then length — test fixture crafted so this equality holds).
- **Evidence**: `test_cleaning_report_counts_add_up`

#### TR 3.5: Malformed flagger runs (logged only)
- **Type**: rule
- **Pass condition**: 1 extremely lopsided pair (src=1char, tgt=100chars) → malformed_flagged_count >= 1.
- **Evidence**: `test_cleaning_malformed_flagged`

### Completion Evidence
- [ ] Cleaning functions + CleaningReport dataclass defined (in loader or new module).
- [ ] 5 tests pass (3.1 → 3.5).

---

## Task 4: Implement Deterministic Splitter (80/10/10, non-overlapping)

**Priority**: high
**Depends on**: Task 2
**Status**: pending

### Description
New module `backend/app/preprocessing/dataset_splitter.py`:
- `split_dataset(records, train=0.8, val=0.1, test=0.1, seed=42) -> Splits`
- Uses local `random.Random(seed)` (no global state)
- Returns 3 lists (train, val, test)
- (Optionally) utility `write_splits_jsonl(splits, out_dir)` — called by scripts

### Test Requirements
#### TR 4.1: Split proportions 80/10/10 approximate (100 rows)
- **Type**: rule
- **Pass condition**: train len ∈ {79,80,81}, val ∈ {9,10,11}, test ∈ {9,10,11} (±1 tolerance).
- **Evidence**: `test_split_proportions_80_10_10`

#### TR 4.2: Same seed twice → identical order
- **Type**: rule
- **Pass condition**: [r.id for r in run1.train] == [r.id for r in run2.train]
- **Evidence**: `test_split_deterministic_same_seed`

#### TR 4.3: Different seeds → different order (probabilistic near-1 test)
- **Type**: rule
- **Pass condition**: [r.id for r in run_seed42.train] != [r.id for r in run_seed43.train]
- **Evidence**: `test_split_seed_sensitivity`

#### TR 4.4: No cross-split duplicate pairs
- **Type**: rule
- **Pass condition**: pairs(train) ∩ pairs(val) == set(); pairs(train) ∩ pairs(test) == set(); pairs(val) ∩ pairs(test) == set() (run on a deduped fixture).
- **Evidence**: `test_split_no_cross_pair_overlap`

#### TR 4.5: Dedup + split together: 10 identical rows → total ≤ 1 across splits
- **Type**: rule
- **Pass condition**: len(train)+len(val)+len(test) == 1
- **Evidence**: `test_split_after_dedupe_merged_singles`

### Completion Evidence
- [ ] [dataset_splitter.py](file:///c:/Users/MEDHA%20TRUST/Downloads/Aiml/Major%20project/context-aware-nmt/backend/app/preprocessing/dataset_splitter.py) exists with split_dataset.
- [ ] 5 tests pass (4.1 → 4.5).

---

## Task 5: Implement ContextRecord interface (no document_id fabrication)

**Priority**: medium
**Depends on**: Task 2
**Status**: pending

### Description
New module `backend/app/preprocessing/context_record.py`:
- `ContextRecord(document_id: str|None, sentence_id: int, context: list[str], source_sentence, target_sentence)`
- `record_from_parallel(pr) -> ContextRecord` with document_id=None, context=[]

### Test Requirements
#### TR 5.1: Samanantar-derived ContextRecord has NO fabricated document_id
- **Type**: rule
- **Pass condition**: Assert `document_id is None`, `context == []`, `sentence_id == pr.id` for 3 records.
- **Evidence**: `test_context_record_samanantar_has_null_document_id`

### Completion Evidence
- [ ] [context_record.py](file:///c:/Users/MEDHA%20TRUST/Downloads/Aiml/Major%20project/context-aware-nmt/backend/app/preprocessing/context_record.py) exists.
- [ ] 1 test passes.

---

## Task 6: Sample Generator Script (`create_sample_dataset.py`)

**Priority**: medium
**Depends on**: Task 4 (uses processed JSONL)
**Status**: pending

### Description
`scripts/create_sample_dataset.py`:
- argparse: --size N, --source (default processed/train), --output-dir (samples/)
- Writes samanantar_sample.en.txt + samanantar_sample.te.txt (first N lines, NOT random)
- Rejects --size > 10000 unless --i-accept-the-large-size

### Test Requirements
#### TR 6.1: Creates 2 parallel .txt files with exactly N lines (N=10)
- **Type**: rule
- **Pass condition**: len(open(...).readlines()) == 10 for both files.
- **Evidence**: `test_sample_creates_parallel_txt_files`

#### TR 6.2: Refuses size > 10000 without flag
- **Type**: rule
- **Pass condition**: exits non-zero, stderr contains "large-size" substring.
- **Evidence**: `test_sample_refuses_large_size`

### Completion Evidence
- [ ] [create_sample_dataset.py](file:///c:/Users/MEDHA%20TRUST/Downloads/Aiml/Major%20project/context-aware-nmt/scripts/create_sample_dataset.py) exists.
- [ ] 2 tests pass.

---

## Task 7: Main Preparation Script (`prepare_samanantar.py`)

**Priority**: high
**Depends on**: Tasks 2, 3, 4, 6
**Status**: pending

### Description
`scripts/prepare_samanantar.py` orchestrates full pipeline:
- Load → filter en→te → Clean → Dedup → Length Filter → Split → Write JSONL → (opt) write sample via Task 6 → write experiments/dataset_preprocessing_report.json
- argparse: --sample-size (default 0), --seed (default 42), --data-path override
- Fail-fast if data_path missing (exit 1 with instructions)
- Report JSON keys: raw_count, language_filtered_count, null_removed_count, duplicate_removed_count, length_filtered_count, malformed_flagged_count, final_count, train_count, validation_count, test_count, random_seed, generation_ts_utc, source_data_path (or manifest hash)

### Test Requirements
#### TR 7.1: Script produces report.json with non-zero measured ints for all keys
- **Type**: rule
- **Pass condition**: Every key present; all counts are int; given a 100-row fixture with crafted empties/dupes/longs → report.raw_count == 100 and all other counts are measured (no hardcoded placeholder 0s unless stage actually removed 0).
- **Evidence**: `test_prepare_samanantar_writes_report_json`

#### TR 7.2: Script splits + writes JSONL (train/val/test files exist)
- **Type**: rule
- **Pass condition**: After run, `data/processed/train.en-te.jsonl`, `val.en-te.jsonl`, `test.en-te.jsonl` all exist under tmp_path outdir; line counts match report.
- **Evidence**: `test_prepare_samanantar_writes_jsonl_splits`

#### TR 7.3: Missing data path → fails fast with instructions
- **Type**: rule
- **Pass condition**: exit code 1; stdout contains "download" and "docs/dataset.md" substrings.
- **Evidence**: `test_prepare_samanantar_missing_path`

### Completion Evidence
- [ ] [prepare_samanantar.py](file:///c:/Users/MEDHA%20TRUST/Downloads/Aiml/Major%20project/context-aware-nmt/scripts/prepare_samanantar.py) exists.
- [ ] 3 tests pass.

---

## Task 8: Dataset Validation Script

**Priority**: medium
**Depends on**: Task 7 (needs processed splits)
**Status**: pending

### Description
`scripts/validate_dataset.py`:
- Loads train/val/test JSONL
- Checks 6 rules from FR-7
- Writes experiments/dataset_validation_report.json
- Exit 0 (ok), 1 (severe), 2 (warnings-only)

### Test Requirements
#### TR 8.1: Valid splits → ok:True, exit 0
- **Type**: rule
- **Pass condition**: Exit code 0, report.ok is True.
- **Evidence**: `test_validate_ok`

#### TR 8.2: Empty source injected → ok:False, issue code=="empty_source_text"
- **Type**: rule
- **Pass condition**: Issues list contains code=="empty_source_text".
- **Evidence**: `test_validate_catches_empty_source_text`

#### TR 8.3: Cross-split leakage → ok:False, issue code=="cross_split_leak", exit 1
- **Type**: rule
- **Pass condition**: exit 1, leakage issue in report.
- **Evidence**: `test_validate_catches_cross_split_leak`

#### TR 8.4: Invalid language label "fr" → caught
- **Type**: rule
- **Pass condition**: issue code=="invalid_language_label".
- **Evidence**: `test_validate_catches_invalid_language`

### Completion Evidence
- [ ] [validate_dataset.py](file:///c:/Users/MEDHA%20TRUST/Downloads/Aiml/Major%20project/context-aware-nmt/scripts/validate_dataset.py) exists.
- [ ] 4 tests pass.

---

## Task 9: Dataset Documentation (`docs/dataset.md`)

**Priority**: low
**Depends on**: (documentation can be written in parallel with any task)
**Status**: pending

### Description
Write `docs/dataset.md` with 14 sections from FR-9.1. Explicitly state "Samanantar is sentence-level parallel data — consecutive rows are NOT guaranteed to belong to the same document, so we NEVER fabricate document_id. Document-level context data is sourced from a separate corpus in Phase 5; we define data structures in this phase ONLY to reserve the interface." If full dataset not downloaded on THIS machine, every count explicitly says "Pending — run `python scripts/prepare_samanantar.py` then inspect experiments/dataset_preprocessing_report.json".

### Test Requirements
#### TR 9.1: 12 of 14 FR-9 sections present as markdown headings
- **Type**: rule
- **Pass condition**: grep for ≥12 heading patterns matches. Contains the exact sentence-level warning paragraph.
- **Evidence**: Reviewer inspection + (opt) `test_dataset_doc_mentions_context_limitation` in tests.

### Completion Evidence
- [ ] [dataset.md](file:///c:/Users/MEDHA%20TRUST/Downloads/Aiml/Major%20project/context-aware-nmt/docs/dataset.md) exists with ≥12 sections.

---

## Task 10: Pytest suite — all tests, no Phase 2 regression

**Priority**: high
**Depends on**: Tasks 2, 3, 4, 5, 6, 7, 8
**Status**: pending

### Description
All tests for Tasks 2–8 live in 1 file: `backend/tests/test_dataset_pipeline.py` with ≥ 20 tests. Run full `pytest backend/tests/` and verify original 42 foundation tests + new tests all pass (≥ 62 total).

### Test Requirements
#### TR 10.1: test_dataset_pipeline.py contains ≥ 20 tests
- **Type**: rule
- **Pass condition**: `pytest --collect-only backend/tests/test_dataset_pipeline.py | tail` → ≥ 20 collected.
- **Evidence**: pytest output in review.

#### TR 10.2: Original 42 foundation tests UNCHANGED and PASS
- **Type**: rule
- **Pass condition**: `pytest backend/tests/ -q` → no failures in test_health, test_history, test_languages, test_translate_text, test_translate_document, test_security, test_context_lastn.
- **Evidence**: Full pytest run output in Completion Evidence.

### Completion Evidence
- [ ] [test_dataset_pipeline.py](file:///c:/Users/MEDHA%20TRUST/Downloads/Aiml/Major%20project/context-aware-nmt/backend/tests/test_dataset_pipeline.py) exists.
- [ ] ≥ 62 tests total pass.

---

## Task 11: Independent Review Phase

**Priority**: high
**Depends on**: All Tasks 1–10 completed
**Status**: pending

### Description
Enter Review phase (read-only evidence gathering):
- Re-verify each AC independently (not same-person self-verification as final gate).
- Write `.trae/specs/review.md` with per-AC pass/fail + independent evidence.
- Score AC-11 rubric independently (≥4 required).
- Overall result: pass / fail / blocked.

### Completion Evidence
- [ ] [review.md](file:///c:/Users/MEDHA%20TRUST/Downloads/Aiml/Major%20project/.trae/specs/review.md) created/updated.
- [ ] Review result == pass.
