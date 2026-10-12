# PRALEKHA structural validation (Phase 13B)

## Scope and method

This is a structural validation only. No translation model was run, no BLEU/chrF++ was computed, and no sentence or paragraph alignments were created.

- Official dataset: `ai4bharat/Pralekha`
- Pinned repository revision: `4c16a270141891dd5c3e98f6f4f207f11bd11236`
- Exact source file: `dev/eng_tel-00000-of-00001.parquet` at that revision
- Split inspected: `dev/eng_tel`
- Official metadata reports 1,000 rows in that split.
- Three real rows were structurally inspected: pinned-file row positions 0, 1, and 2. These positions are retrieval locators, not dataset-provided document IDs.

The revision was confirmed through the official Hub revision API and pinned dataset card/tree. The records were read from the official revision-specific Parquet URL, not a mirror. The official Datasets Server `/rows` API documentation lists no revision selector; exploratory viewer responses were therefore not treated as pinned-record evidence.

## Data access

The Parquet file is 4,676,131 bytes and contains one row group with all 1,000 records. Reading three records with PyArrow required fetching and decoding the complete row group. HTTP range requests transferred 4,741,663 bytes in total, including repeated metadata ranges. All data remained in process memory; no dataset, temporary, or derived data file was written to the workspace. Only the first three rows were summarized below; the remaining rows were not structurally reviewed.

The row-group layout limited how little data could be transferred from this pinned file. No other PRALEKHA shard, split, train data, alignable collection, or full corpus was fetched.

For endpoint capability checking, the official but revision-unpinned Datasets Server was also queried once for three rows and once through `/first-rows`; the preview returned rows 0-9 with long text cells truncated. Those responses were not used as evidence for the pinned findings. Their content overlapped the sampled indices, but is additional official-viewer data access beyond the pinned-shard transfer above.

## Observed schema and sample

The pinned Parquet schema is exactly:

| Field | Type | Row 0-2 observation |
|---|---|---|
| `src_lang` | string | `eng` on 3/3 rows |
| `tgt_lang` | string | `tel` on 3/3 rows |
| `src_txt` | string | Non-empty on 3/3 rows |
| `tgt_txt` | string | Non-empty on 3/3 rows |

Each row has a co-located English source document string and Telugu reference document string. Their presence in one `eng_tel` record supports **row-level document pairing**. The sampled character counts were source/target: row 0, 22,435/21,609; row 1, 23,142/21,648; row 2, 22,894/21,421.

The rows contain line breaks (source/target counts respectively: 41/48, 118/172, and 69/72). These are text formatting observations only. The schema does not say they are paragraph boundaries or aligned units.

## Identity and provenance results

| Requirement | Result for sampled rows |
|---|---|
| Dataset and pinned revision | Present in retrieval provenance for all 3 |
| Split | `dev/eng_tel`, present for all 3 |
| `n_id` | Absent from the schema and all 3 records |
| English document `doc_id` | Absent |
| Telugu document `doc_id` | Absent |
| Source/target language labels | Present and correct as `eng` / `tel` on 3/3 |
| Source/reference fields | `src_txt` / `tgt_txt`, non-empty on 3/3 |
| Record locator | Pinned Parquet row positions 0, 1, 2; not source-provided record IDs |
| Separate source and target document identity | Not available; only the paired row is exposed |
| Verified adapter provenance issued | No; records fail the existing adapter's identity contract |

The paired row and pinned row position locate the two text fields reproducibly, but they do not provide `n_id` or individual English/Telugu document IDs. Accordingly, **0/3 sampled rows are accepted as trusted PRALEKHA records by the Phase 13A adapter contract**. Exclusion reason for each: `n_id` and separate source/target document IDs are not present in the evaluation schema; a row position must not be relabeled as any of these IDs. The adapter also currently validates language values as `en`/`te`, while the official dataset labels are `eng`/`tel`; this is an additional integration mismatch, not a basis to rewrite the observed labels.

No verified provenance object was issued for these rows. An honest source/reference location can be described as the pinned Parquet path plus row position and field name, but missing dataset identity fields remain missing.

## Structure and alignment

| Structure / evidence | Count or status |
|---|---|
| Co-located source/reference document pairs | 3/3 rows |
| Dataset-provided `n_id` | 0/3 |
| Separate source/target document IDs | 0/3 |
| Explicit paragraph records, IDs, or ordering | 0/3; not in schema |
| Explicit sentence records, IDs, or ordering | 0/3; not in schema |
| Explicit other ordered units | None observed |
| Sentence alignment | Not provided; 0 eligible records |
| Paragraph alignment | Not provided; 0 eligible records |
| Document-level pairing | Present at row granularity for 3/3; not independently addressable by `n_id` or per-language document ID |

The text has an internal character sequence and newline formatting, but neither establishes ordered paragraph or sentence units for context construction. No document boundaries can be joined across rows. No `alignable` identifiers were transferred to these rows.

## Context and metric eligibility

- **Current unit:** only the complete row-level source document can be treated as a supplied unit. No smaller current unit is dataset-annotated.
- **Preceding units:** unavailable. There are no explicit ordered paragraphs/sentences or unit IDs.
- **Same-document causal context:** unavailable for this sampled representation. The adapter cannot construct legitimate `last_n` context from these records without deriving units, which this phase does not do.
- **Document-level reference granularity:** structurally present for 3/3 row pairs. Whole-document prediction/reference comparison is the only reference granularity represented by the sample.
- **Document BLEU/chrF++ eligibility:** structural pairing is present for 3/3, but **trusted project metric eligibility is 0/3** because required dataset identity/provenance is incomplete under the current adapter contract. No metrics were run.
- **Sentence BLEU/chrF++ eligibility:** 0/3. Sentence-level reference alignment is absent.
- **Paragraph-level metrics:** 0/3. Paragraph pairing/alignment is absent.

The structural presence of a full-document pair is not a translation-quality result and does not authorize a run through the current trusted adapter. The sample does not establish split-wide cleanliness or document-pair correctness beyond the co-located row fields.

## Exclusions and contract gap

All three inspected records are excluded from trusted adapter acceptance. Missing fields are the source-provided `n_id` and per-language document IDs; the official row index is only a pinned split locator. The language-code mismatch (`eng`/`tel` versus adapter-accepted `en`/`te`) also prevents direct validation through the existing Phase 13A helper.

This validation did not change the adapter or weaken its tests. Any future adapter integration must preserve the official ISO-3 labels, retain the pinned row locator as a locator rather than a document ID, and decide explicitly whether missing `n_id` is an exclusion or whether an approved provenance contract can accept paired row identity alone. It must not invent document IDs or derive context units from line breaks.

## Limitations

- Only rows 0-2 were summarized; no conclusions are made about all 1,000 dev records or the test split.
- The source schema supplies document-text pairs, not sentence/paragraph segmentation or alignment annotations.
- Row positions are tied to the pinned split and file; they are not source-provided document identifiers.
- Text script/content observations are not a formal language-identification evaluation.
- One-row-group Parquet layout forced full-shard byte transfer and decoding even though only three rows were reviewed.
- The earlier Phase 13A synthetic context builder demonstrates behavior for explicitly ordered synthetic units; it does not establish that this PRALEKHA evaluation schema supplies such units.

## Execution record

- Exact source: [official pinned PRALEKHA revision](https://huggingface.co/datasets/ai4bharat/Pralekha/tree/4c16a270141891dd5c3e98f6f4f207f11bd11236), file `dev/eng_tel-00000-of-00001.parquet`.
- Official metadata: repository SHA `4c16a270141891dd5c3e98f6f4f207f11bd11236`; `dev/eng_tel` has 1,000 examples; the pinned card metadata lists the four fields above.
- Real records structurally summarized: 3 (row positions 0-2) from `dev/eng_tel`.
- Data transfer: one 1,000-row dev shard row group, 4,741,663 HTTP range bytes; in memory only; no local data file.
- Endpoint capability probe: official revision-unpinned viewer returned up to 10 preview rows (truncated text); excluded from findings.
- No other split or corpus shard was downloaded.
- Model inference: none.
- Translation metrics: none.
- Git initialization: none.
