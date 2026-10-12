# PRALEKHA English-Telugu QA

## 1. Dataset and Revision

- Dataset: [AI4Bharat/PRALEKHA](https://huggingface.co/datasets/ai4bharat/Pralekha).
- Revision inspected: `4c16a270141891dd5c3e98f6f4f207f11bd11236`, the SHA pinned in [the Phase 9A audit](pralekha_audit.md).
- The Hugging Face dataset API reported the same repository SHA during this QA. The dev and test Parquet files were fetched by their revision-specific URLs and read in memory; no corpus files were saved to the workspace.
- Official background sources listed by the audit: [PRALEKHA paper](https://aclanthology.org/2025.ijcnlp-long.37/) and [AI4Bharat PRALEKHA repository](https://github.com/AI4Bharat/Pralekha).

## 2. Configurations and Splits

The English/Telugu routes reported by the Hub metadata are:

| Configuration | Split(s) | Rows | What was inspected |
|---|---|---:|---|
| `dev` | `eng_tel` | 1,000 | All records, pinned Parquet |
| `test` | `eng_tel` | 1,000 | All records, pinned Parquet |
| `train` | `eng_tel` | 101,109 | Split metadata only; training data not fetched |
| `alignable` | `eng`, `tel` | 298,111; 110,077 | Metadata counts and first 3 records of each language split |
| `unalignable` | `eng`, `tel` | 149,055; 55,038 | Split metadata only; records not fetched |

`dev/eng_tel`, `test/eng_tel`, and `train/eng_tel` are paired English-to-Telugu configurations. `alignable` and `unalignable` are separate language-specific document collections, not the MT evaluation schema. Counts for routes other than pinned dev/test are Hub size metadata observed while the repository SHA matched the pin; their data rows were not downloaded or audited here. Other language routes are outside this English/Telugu QA.

## 3. Schema

Pinned `dev/eng_tel` and `test/eng_tel` have the same four columns. All are non-nullable-by-observation string values in the Parquet schema; no identity or sentence fields are present.

| Column | Type | Observed role |
|---|---|---|
| `src_lang` | string | Source language code; `eng` on every inspected dev/test row |
| `tgt_lang` | string | Target language code; `tel` on every inspected dev/test row |
| `src_txt` | string | Source document text |
| `tgt_txt` | string | Paired target document text |

The sampled `alignable/eng` and `alignable/tel` schemas each contain `n_id`, `doc_id`, `lang`, and `text`, all strings. The `unalignable` and `train/eng_tel` row schemas were not independently inspected: Not established from the inspected data.

## 4. Record Counts

| Split | Records | Empty `src_txt` | Empty `tgt_txt` | `src_txt == tgt_txt` |
|---|---:|---:|---:|---:|
| `dev/eng_tel` | 1,000 | 0 | 0 | 0 |
| `test/eng_tel` | 1,000 | 0 | 0 | 2 |

The two test exact-copy rows are row indices 0 and 2. Counts above were computed over all records in the revision-pinned Parquet files.

## 5. Language-Direction QA

- Every dev and test row has `src_lang="eng"` and `tgt_lang="tel"`; no record has reversed direction labels.
- All 1,000 source strings in each split are Latin-letter-dominant under a character-script count. This supports the `eng` labels but is not a language-identification test. Whether every source is linguistically English: Not established from the inspected data.
- In each split, 998/1,000 targets are Telugu-script-dominant by the same heuristic. Dev rows 588 and 967 have more ASCII Latin letters than Telugu codepoints but also contain Telugu script; this alone does not establish that either target is in the wrong language.
- Test rows 0 and 2 are exact copies of their source strings and contain no Telugu codepoints in the target. They are observed English-to-English target anomalies, despite the `eng` to `tel` labels.
- A formal language identifier was not run. Script counts do not establish linguistic identity for mixed-script or transliterated text.

## 6. n_id / Document Identity QA

- `n_id` exists in the sampled `alignable/eng` and `alignable/tel` records. The first three observed values in each split match by value: `mkb-1`, `mkb-2`, and `mkb-3`. Each matched value has an English record and a Telugu record, and the corresponding sampled `doc_id` values differ by language.
- This establishes cross-language `n_id` matching for those three observed values, not uniqueness across the full alignable corpus. Because the same `n_id` occurs in English and Telugu records, it is not globally unique across the combined language records. Whether it is unique within each language split: Not established from the inspected data.
- Neither `n_id` nor `doc_id` exists in the pinned `dev/eng_tel` or `test/eng_tel` schema. Do not assign alignable IDs to evaluation rows by row order or content resemblance.
- `doc_id` exists in the sampled alignable schema; whether it exists in the evaluation data is answered no for the inspected pinned dev/test tables.

## 7. Document Granularity

Each dev/test record contains one paired source-text and target-text field, rather than sentence-level rows. The pinned rows are long, internally line-broken texts (source median length about 1,036 characters in dev and 1,211 in test). The Phase 9A audit's cited dataset documentation and inspected examples describe these as document pairs, including MKB/PIB material. The record unit is therefore document-level text, not a sentence pair.

Whether every record is the complete original publication or script, rather than a document excerpt: Not established from the inspected data. A record may contain paragraph-like sections, transcript or speaker material, and other internal formatting; it is not a provided paragraph or sentence segmentation table.

## 8. Document Length Statistics

Character counts include all characters in each raw text field. Percentiles use the nearest-rank 95th percentile. Line breaks are literal LF characters; blank-line gaps count consecutive LF characters with optional spaces/tabs between them. They are formatting statistics, not annotated paragraph counts.

| Split / field | Characters min / median / mean / p95 / max | LF count median / mean / max | Records with blank-line gap |
|---|---|---|---:|
| dev `src_txt` | 260 / 1,035.5 / 3,066.3 / 12,211 / 28,828 | 6 / 16.5 / 1,393 | 9/1,000 |
| dev `tgt_txt` | 264 / 1,201.5 / 3,134.5 / 11,953 / 31,973 | 6 / 16.5 / 1,401 | 18/1,000 |
| test `src_txt` | 136 / 1,211 / 3,572.7 / 14,050 / 58,764 | 6 / 14.7 / 504 | 5/1,000 |
| test `tgt_txt` | 120 / 1,325.5 / 3,583 / 13,307 / 49,867 | 6 / 14.4 / 504 | 5/1,000 |

Every inspected evaluation text has at least two LF characters. Equal source/target LF counts occur in 528/1,000 dev pairs and 532/1,000 test pairs. This is coarse structural similarity only: it does not show that line or paragraph boundaries correspond. Exact sentence counts and paragraph counts: Not established from the inspected data.

## 9. Alignment Availability

The paired dev/test row provides document-level source/reference pairing through `src_txt` and `tgt_txt` in the same record. There are no sentence IDs, paragraph IDs, timestamps, token offsets, or sentence-alignment fields. Sentence-level reference alignment: Not established from the inspected data; no explicit sentence alignment is provided by the inspected schema.

At the document level, both sides contain internal line breaks and have the same median LF count, so their broad text structure is comparable. Exact LF counts match in only 528/1,000 dev pairs and 532/1,000 test pairs; the punctuation-count heuristic matches in only 167 dev and 152 test pairs. Therefore segment-by-segment structural compatibility is not established, and these counts do not justify sentence or paragraph alignment.

Equal punctuation counts are not alignment evidence; punctuation marks do not provide sentence segmentation.

## 10. Data Anomalies

- Empty source or target strings: none in dev or test.
- Exact source-equals-target pairs: none in dev; two in test (row indices 0 and 2). These test targets are English source copies, not Telugu references.
- Exact duplicate source texts, exact duplicate target texts, and exact duplicate source/target pairs within each split: none observed.
- Wrong direction labels: none. Test's two copied targets are same-language source/target anomalies under the English/Telugu labels.
- Dev rows 588 and 967 are mixed-script target outliers under the character-count heuristic; Telugu characters are present, so the heuristic does not establish a language error. They merit review before evaluation.
- Source/target line-break totals are not consistently equal. This is not itself an error, since document translation can change formatting and sentence structure.

## 11. Evaluation Feasibility

- **Document-level evaluation:** Supported conditionally for `dev/eng_tel` and `test/eng_tel`: each record supplies one paired document text. The two copied test targets and the two mixed-script dev candidates require review and a documented inclusion decision before publishing results. No metrics were computed in this QA.
- **Sentence-level BLEU/chrF++:** Not supported as sentence-paired evaluation by the inspected data. No source/reference sentence alignment is supplied. Independently splitting both sides and pairing sentences by order would introduce an unsupported assumption.
- **Context-window evaluation:** Causal source context can be extracted after applying a documented sentence splitter to each source document: for a current source sentence, use only earlier source sentences from that same record, reset at the next record, and never include future source text or target/reference text. This enables a context-conditioned document-output experiment, but does not create reference sentence alignment or dataset-provided sentence IDs. Sentence-level context-conditioned scoring remains unsupported without alignment.
- The evaluation rows have no exposed document IDs. Keep the source split and original row index as a provenance locator if needed; it is not a dataset-supplied `document_id` and must not be presented as one.

## 12. Missing Information

For the requested evaluation record:

| Desired field | Availability |
|---|---|
| `document_id` | Missing from dev/test; `n_id` and `doc_id` cannot be transferred from `alignable` rows. |
| `sentence_id` | Not present. Any segmentation index would be newly derived, not a dataset sentence ID. |
| `source_sentence` | Not present as a field; it could be produced only by a documented source-side splitter. |
| `reference_translation` | Only the whole paired `tgt_txt` document is present; aligned reference sentences are not. |
| `previous_context_sentence_ids` | No sentence IDs exist. Previous source sentences can be extracted causally, but their IDs would be derived local positions, not dataset IDs. |
| `alignment` | No sentence/paragraph alignment annotation is present. |

Sentence-level English/Telugu language validation beyond the observed labels, scripts, and reviewed copied targets: Not established from the inspected data. Full alignable `n_id` uniqueness and complete-original-document status are also not established.

## 13. Recommended Next Step

Have a reviewer inspect test rows 0 and 2 and dev rows 588 and 967, then record a transparent keep/exclude decision before any evaluation. If clean document-level evaluation remains acceptable, use the pinned dev/test document pairs and report the missing IDs and the treatment of flagged rows. For sentence-level BLEU/chrF++ or sentence-level context evaluation, obtain verified sentence segmentation and source/reference alignment from the dataset maintainers or another explicitly aligned resource. Do not manufacture document IDs, sentence IDs, or alignments from row order.

**Execution record**

- Files created: `docs/pralekha_qa.md`.
- Files modified: none existing.
- Data downloaded: the pinned dev and test Parquet files (4,676,131 + 5,409,046 = 10,085,177 bytes) into memory for inspection; no dataset files were saved. Retrieved split metadata and only the first three alignable English and Telugu records for ID checks. No training, full alignable, or unalignable corpus shards were fetched.
- Commands/requests executed: Hugging Face dataset API SHA lookup; dataset-server `/splits`, `/size`, and bounded `/rows` requests; pinned Hub tree requests for dev/test and the alignable file list; revision-specific HTTP GETs for the two evaluation Parquet files; in-memory Python/PyArrow schema, language-script, duplicate, length, newline, and exact-copy checks. A repeated rows query was rate-limited and not used for results. `git status --short` could not run because the workspace path is not a Git repository.
- Project files changed: only `docs/pralekha_qa.md`; no backend, frontend, evaluation, Samanantar, TED2020, dataset, or test files changed.
- Tests run: none. This was read-only data QA and documentation only.
- Test result: not run for Phase 9B. The Phase 9A status of 137 passed, 2 warnings, 0 failures is the status supplied by the user; it was not rerun.
