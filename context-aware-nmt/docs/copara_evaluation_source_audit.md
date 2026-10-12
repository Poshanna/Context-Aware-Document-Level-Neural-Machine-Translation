# CoPara evaluation-source audit (Phase 13C)

## Decision

**CoPara is not sufficient for the next context-aware evaluation phase in the exposed files.** It does provide English-Telugu passage pairs at the row level, so whole-passage outputs could in principle be compared with same-row Telugu passage references. But it does not expose a parent article identifier plus a within-article paragraph order, nor sentence spans/order. Therefore the dataset does not establish which passage precedes another for the same document, and this audit does not construct context from adjacent rows or split text into artificial aligned units.

## Sources and revisions

| Source | Exact version inspected | Relevant claim |
|---|---|---|
| Author repository, `ENikhil/CoPara` | Commit `7c92005159d0dd61c18f99f40b683e6877dc7f69`, dated 2024-09-15 | README calls the resource paragraph-level and reports 2,856 paragraph/passage pairs. `Paragraphs.csv` and `Articles.csv` are the two exposed representations. |
| Author `Data/Paragraphs.csv` | Same commit; Git blob `d1696b17f1e37478e029ae0ad7affdff775f7579`; 14,596,944 bytes | Passage-level CSV representation. |
| Author `Data/Articles.csv` | Same commit; Git blob `4676f232f12f6a7405cf35e5ae5ca6606ebdcccb`; 19,459,661 bytes | Article-level CSV representation. |
| AI4Bharat Indic NLP Catalog | `master` commit `7c608d6fddbecae59ddfd110cf4ff0bd14e4d2d0`, dated 2024-12-14 | Describes CoPara as long-context parallel corpora for four Dravidian languages and reports 2,586 passage pairs. |
| CoPara paper | Nikhil E, Mukund Choudhary, Radhika Mamidi, ACL Anthology `2023.dravidianlangtech-1.12` (2023; no data-repository revision stated on the paper page) | Abstract reports 2,856 paragraph/passage pairs. The project audit records the paper's paragraph construction as article alignment followed by paragraph-count comparison and positional correspondence when counts match. |

The author repository and paper report **2,856**; the catalog reports **2,586**. These are conflicting source claims. The author README/paper count is not silently substituted for the catalog count, and no full-file row count was performed here to reconcile them. The catalog is collaborative metadata, not an alternate release of the source data.

## Data access

- Representation inspected: `Data/Paragraphs.csv` and `Data/Articles.csv` at the pinned author-repository commit; neither has an evaluation split field.
- Bounded access: one `Range: bytes=0-65535` request per CSV, 65,536 bytes each, **131,072 bytes total**. The GitHub contents metadata provided each full file's byte size and blob SHA without retrieving the full CSVs.
- Records structurally reviewed: the first two records in each file, **4 records total**. The byte ranges include additional trailing CSV content; those additional rows were not reviewed. No CSV was saved to the workspace.
- No full corpus, complete CSV, model checkpoint, or derived data file was downloaded or persisted.

## Observed representation

Both CSV samples have the same 21 fields: `Index`; a text, token-count, sentence-count, and word-count field for each of `English`, `Kannada`, `Malayalam`, `Tamil`, and `Telugu`.

`Paragraphs.csv` supplies one multilingual passage/paragraph tuple per row. The first two sampled rows have non-empty English and Telugu text:

| `Index` | English characters | Telugu characters | English sentence count | Telugu sentence count |
|---:|---:|---:|---:|---:|
| 0 | 548 | 725 | 4 | 7 |
| 1 | 238 | 228 | 2 | 2 |

`Articles.csv` supplies article-level multilingual text rows. Its first two sampled English/Telugu fields were non-empty, with lengths 549/726 and 2,876/2,846 characters. These are observations from two records, not full-corpus statistics. The article and paragraph files each contain an `Index`; the sample/schema does not expose a shared `article_id` or an explicit paragraph-to-article crosswalk. Similar or equal indices across the separate files are not treated as a crosswalk.

The `Sentences` columns are integer counts, not sentence text, sentence IDs, spans, or alignment links. The unequal counts in paragraph row 0 (4 English, 7 Telugu) directly show that those counts cannot be used as one-to-one sentence alignment. Equal counts in row 1 do not establish alignment either.

## Eligibility assessment

| Question | Finding |
|---|---|
| Passage identity | Yes, at row granularity. `Index` is described in the author README as a zero-based index of all data; the same row contains the language tuple. Full-file uniqueness/continuity was not audited. |
| English-Telugu passage pairing | Yes, at row granularity. The author describes the rows as n-way aligned paragraph/passage tuples, and English and Telugu fields co-occur in each sampled row. The paper describes paragraph-position correspondence within aligned articles as part of construction. |
| Article/document identity for a passage row | Not exposed in the sampled `Paragraphs.csv` schema. There is no `article_id`, source issue ID, or paragraph ordinal field. `Articles.csv` is a separate article-level representation, not a demonstrated row mapping. |
| Ordered source/target units | The passage itself is the paired unit. `Index` is a global data index, not documented as an article-local order. The paper's paragraph-position construction claim does not give the sampled flat CSV row an exposed parent article key and paragraph position. No sentence units or sentence ordering fields are supplied. |
| Preceding context | Unavailable across passage rows. The data does not establish that adjacent global indices belong to the same article or are consecutive paragraphs. Within-passage sentence context would require deriving sentence units/spans; this phase does not invent those units. |
| Current evaluation unit | The complete `English` passage (or complete article when using `Articles.csv`) is an available unit. No smaller current unit is dataset-annotated in these files. |
| Reference at matching granularity | Yes for complete passage output versus the same row's `Telugu` passage; likewise article output versus an article-level `Telugu` field. No aligned Telugu sentence references are available. |
| BLEU/chrF++ | Passage-level corpus BLEU/chrF++ is structurally possible for whole-passage hypotheses against same-row passage references. Sentence-level BLEU/chrF++ is unsupported. A project run would still require verified provenance, explicit row locations, a documented held-out protocol, and suitable metric implementation. |
| Baseline vs context-conditioned | Not supported as a context-aware comparison from these files alone. A passage-only baseline can be defined, but there is no legitimate prior source passage chain for the context condition. |

The author paper describes a positional paragraph-alignment construction procedure: articles are aligned, then internal paragraph counts are compared; where counts match, corresponding paragraph positions are assumed congruent and retained. This is an author-described paragraph-level assumption, not sentence alignment. In the exposed flat paragraph CSV, the parent article and local paragraph position needed to reconstruct a causal sequence are not fields. No sentence or paragraph pieces were split or paired in this audit.

## Project contract and provenance

The project evaluator can represent `TextUnit.PASSAGE`, requires the current unit and reference to match granularity, and gates passage BLEU/chrF++ on verified provenance, a prediction, a reference location, alignment location, and passage-pair evidence. Context records additionally require a provenance-bound passage identity plus ordered, strictly earlier source units from the same passage; they reject cross-passage context. Thus a row-level `Index` and same-row language fields may support a passage-pair record after a trusted adapter verifies the exact upstream commit and row locations, but they do not authorize prior-passage context.

A future source adapter would need to preserve the exact author commit, CSV path/blob, `Index`, language columns, source/reference field locations, and evidence that both texts are members of the same author-described n-way passage tuple. Since no official train/dev/test split is exposed, any local holdout must be explicitly marked as a project-created split. It should be grouped at article/document level to avoid article leakage; the missing paragraph-to-article mapping currently prevents that grouping for `Paragraphs.csv`.

`context_conditioned_experiment.py` is only a placeholder and does not execute metrics. `run_context_comparison.py` uses a punctuation-based source splitter and runs inference; it is not a provenance-aware CoPara adapter and was not used. Its segmentation must not be described as CoPara alignment.

## Limitations and rights

- Only two initial records per representation were inspected; no complete CSV audit or dataset-wide QA was conducted.
- The 2,856/2,586 count discrepancy remains unresolved.
- The CSVs do not expose an official evaluation split or a demonstrated article-to-paragraph crosswalk.
- CoPara's paragraph correspondence depends on the authors' count/position assumption. Sentence-level correspondence and sentence alignment are absent.
- The author repository identifies a GPL-3.0 license, but the inspected sources do not establish a separate license for New India Samachar text. Confirm source-text reuse/redistribution permissions and attribution before handling or publishing passages.
- No model inference, translation, BLEU, chrF++, or other metric was run.

## Recommendation

**Do not select the exposed CoPara files as the next context-aware evaluation source yet.** They are a conditional candidate for a passage-level MT metric study, not a ready context-aware test set. The missing data property is an authoritative **parent article/document ID plus ordered paragraph identity** linking each passage tuple to its position in that same source document. A documented held-out article-level split or a safe way to create one without article leakage is also needed. Obtain that mapping/order from the authors or an official release before constructing preceding context; otherwise keep context eligibility unavailable. Do not infer relationships from global `Index`, split passage strings into alleged aligned sentences, or proceed to inference.
