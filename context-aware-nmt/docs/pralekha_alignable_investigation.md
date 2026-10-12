# PRALEKHA Alignable Resource Investigation

## 1. Dataset and Revision

- Dataset: [AI4Bharat/PRALEKHA](https://huggingface.co/datasets/ai4bharat/Pralekha).
- Revision: `4c16a270141891dd5c3e98f6f4f207f11bd11236`, pinned in [the Phase 9A audit](pralekha_audit.md). The Hub API reported the same SHA during this inspection.
- **Official documentation:** The pinned dataset card describes PRALEKHA as a document-level cross-lingual alignment resource. It defines fields for the alignable set and says `n_id` is the unique identifier for alignable document pairs.
- **Direct observation:** The pinned card feature metadata and bounded viewer samples agree on the `alignable` schema. Eight sampled IDs were joined between `eng` and `tel` by the exact `n_id` value, not by row position.
- **Official sources:** [Pinned dataset card](https://huggingface.co/datasets/ai4bharat/Pralekha/tree/4c16a270141891dd5c3e98f6f4f207f11bd11236), [PRALEKHA paper](https://aclanthology.org/2025.ijcnlp-long.37/), and [AI4Bharat repository](https://github.com/AI4Bharat/Pralekha).

## 2. Alignable Resource Location

- Hub configuration/subset: `alignable`.
- Language splits: `eng` and `tel` (ISO 639-3 language codes).
- Pinned Parquet files:

| Language split | File | Repository file size |
|---|---|---:|
| `eng` | `alignable/eng-00000-of-00003.parquet` | 194,711,644 bytes |
| `eng` | `alignable/eng-00001-of-00003.parquet` | 191,896,662 bytes |
| `eng` | `alignable/eng-00002-of-00003.parquet` | 192,147,685 bytes |
| `tel` | `alignable/tel-00000-of-00002.parquet` | 165,704,079 bytes |
| `tel` | `alignable/tel-00001-of-00002.parquet` | 163,611,649 bytes |

The shard paths and sizes come from the Hub tree at the pinned revision. No Parquet shard was downloaded. The five shard files total about 908 MB, so the investigation used only metadata and eight-row slices for each language.

## 3. Schema

The pinned dataset card feature metadata and the viewer schemas for both sampled language splits report these columns, each with type `string`:

| Column | Type | Meaning stated by official card | Directly observed |
|---|---|---|---|
| `n_id` | string | Unique identifier for an `alignable` document pair | Shared across the matched English and Telugu sample records |
| `doc_id` | string | Unique identifier for an individual document | Different value on each sampled English/Telugu member of a pair |
| `lang` | string | Document language, ISO 639-3 code | `eng` on English rows, `tel` on Telugu rows |
| `text` | string | Textual content of the document | Long, line-broken document text |

`src_txt` and `tgt_txt` are not columns in the alignable schema. They are source and target text fields in the separate paired MT configurations; the pinned dev/test representation uses `src_lang`, `tgt_lang`, `src_txt`, and `tgt_txt`. The alignable schema has no sentence ID, paragraph ID, offsets, token alignment, sentence alignment, or other explicit alignment field. Its alignment-related field is `n_id`.

## 4. Meaning of n_id

- **Official documentation:** `n_id` is the unique identifier for an alignable document pair. The card separately describes `doc_id` as the unique identifier for an individual document.
- **Direct observation:** In the first eight sampled records from each language split, the intersecting values `mkb-1` through `mkb-8` each identify a sampled `eng` record and a sampled `tel` record when matched by `n_id`. For all eight matches, the language-specific `doc_id` values differ.
- **Inference supported by both:** `n_id` operates at the document-pair level, while `doc_id` identifies one language-specific document. `n_id` does not identify a sentence or paragraph in the observed schema.
- **Scope limit:** The full alignable corpus was not scanned for duplicate keys or cardinality violations. Global `n_id` uniqueness is stated by the official card, but was not independently verified across every shard.

## 5. Document Relationship

The eight sample joins are explicit value matches, not assumptions based on adjacent rows:

| `n_id` | English `lang` | Telugu `lang` | Same `doc_id`? | Observed unit |
|---|---|---|---|---|
| `mkb-1` | `eng` | `tel` | No | Long MKB document text on each side |
| `mkb-2` | `eng` | `tel` | No | Long MKB document text on each side |
| `mkb-3` | `eng` | `tel` | No | Long MKB document text on each side |
| `mkb-4` | `eng` | `tel` | No | Long MKB document text on each side |
| `mkb-5` | `eng` | `tel` | No | Long MKB document text on each side |
| `mkb-6` | `eng` | `tel` | No | Long MKB document text on each side |
| `mkb-7` | `eng` | `tel` | No | Long MKB document text on each side |
| `mkb-8` | `eng` | `tel` | No | Long MKB document text on each side |

The English and Telugu text beginnings in these matched records are thematically and linguistically corresponding MKB openings. Combined with the official field definition, this is evidence of document-level pairing. It is not evidence of sentence-to-sentence links. The strings' completeness relative to the original scripts was not audited: Not established from the inspected data.

## 6. Sentence/Paragraph Relationship

- **Directly observed:** `text` is a raw string containing internal line feeds and punctuation, not a list of sentence or paragraph records. For the eight joined pairs, English/Telugu LF counts were equal in only two pairs (`mkb-1`, `mkb-6`); the other six differed. Blank-line separators also differed in several pairs.
- **Directly observed:** Sentence-like punctuation occurs in both texts, but counts differ across the sample. Punctuation is not a sentence segmentation annotation.
- **Official documentation:** The fields describe the text of individual documents; no sentence or paragraph segmentation/alignment fields are documented.
- **Conclusion:** Paragraph and sentence boundaries may be heuristically derived from the strings, but they are not provided as labeled parallel units. Sentence boundaries or paragraph correspondence: Not established from the inspected data.

## 7. Alignment Evidence

- **Document level:** Yes, document-level pairing is provided: `n_id` is documented as the pair identifier and the sampled English/Telugu records with the same ID contain corresponding document text.
- **Paragraph level:** No explicit paragraph links, paragraph IDs, or parallel paragraph records are present.
- **Sentence level:** No sentence IDs or sentence links are present. Sentence-level reference alignment is not established by the inspected alignable resource.
- **Token level:** No token offsets or token-alignment fields are present.
- **Usable alignment:** The resource provides usable document-pair alignment through `n_id`; it does not provide usable paragraph-, sentence-, or token-level alignment in the inspected representation.

Sentence order across the English and Telugu documents is not explicitly annotated. The examples show corresponding document openings, but do not establish that every English sentence has a Telugu sentence in the same order. Do not infer such links from the row order, punctuation, or matching line counts.

## 8. Alignment Cardinality

| Unit | Finding | Evidence scope |
|---|---|---|
| Document pair | One sampled English record and one sampled Telugu record per each of 8 matched `n_id` values; official card defines `n_id` as a unique pair identifier | Eight bounded sample joins plus official field definition; no full-corpus cardinality audit |
| One English sentence to one Telugu sentence | Not established | No sentence segmentation or sentence links |
| One-to-many sentences | Not established | No sentence alignment annotations |
| Many-to-one sentences | Not established | No sentence alignment annotations |
| Many-to-many sentences | Not established | No sentence alignment annotations |

No sentence-level cardinality can be quantified from this schema. Splitting each `text` and pairing segments by their index would invent alignment evidence and is not justified.

## 9. Context Reconstruction

A causal source-context sequence is technically reconstructible within an English document: choose a documented English sentence splitter, process sentences in their text order, and for each current sentence select only earlier source sentences belonging to the same `n_id`. Reset at the next document pair. This does not require target text, future sentences, row adjacency, or sentence alignment.

However, the resource does not contain dataset-provided `sentence_id` values or `previous_context_sentence_ids`. A splitter can provide sentence text and transient processing order, but those positions are derived and must not be represented as source dataset IDs. No sentence IDs or context records were created in this investigation.

## 10. Evaluation Compatibility

The existing project [evaluation architecture](evaluation.md) consumes sentence-level records; its context-conditioned path requires explicit same-document identity and strictly earlier sentence indices. The [context-conditioned experiment design](context_conditioned_experiment.md) uses the same source sentence with an optional preceding-source prefix. Raw alignable records are document-level and are not directly in that evaluation input format.

| Component | Compatibility finding |
|---|---|
| Baseline NLLB | Could translate source sentences from each English document without context, but the raw resource is not a ready-to-run evaluation dataset. A documented text segmentation and an output/reference scoring design would be needed. No model was run. |
| Prefix-conditioned NLLB | Causal preceding-source context can be formed within each English `n_id` document after a fixed splitter. The raw resource lacks sentence IDs and aligned reference sentences, so it cannot directly provide the runner's complete sentence-level context/reference records. No model was run. |
| BLEU / chrF++ | Could compare one generated full-document output against the Telugu document matched by `n_id`; this is document-level scoring, not sentence-level reference scoring. The current runner would need a compatible input path. No metric was computed. |
| Optional BERTScore / COMET | A document text pair can be presented to a document-level scorer in principle, but model input limits, chunking, truncation, and consistent treatment across systems would need to be validated. Sentence-aligned scoring is unsupported. No metric was computed. |

The card says MT dev/test documents are sampled from `alignable`, but the MT split rows do not expose the alignable `n_id`/`doc_id` crosswalk. Therefore this resource alone cannot establish which alignable records belong to the held-out dev/test sets or prove split independence. Do not treat the whole alignable collection as a clean independent test set on the basis of this inspection.

## 11. Limitations

- Only eight rows per language were sampled; the large alignable Parquet shards were not downloaded or exhaustively audited.
- Official documentation defines `n_id` as a unique document-pair identifier; full-corpus uniqueness and exactly-one-record-per-language cardinality were not independently checked.
- `doc_id` identifies individual documents according to the card; the algorithm or provenance of its observed 64-character values was not established.
- No sentence, paragraph, or token alignment annotations were found in the schema.
- Cross-language sentence order and one-to-one/one-to-many/many-to-one/many-to-many sentence relations cannot be determined from the document strings and `n_id` alone.
- The text contains line breaks and punctuation, but those do not define parallel segments.
- The dev/test MT records do not expose identifiers to crosswalk them to alignable rows; held-out membership and overlap cannot be checked here.
- A reproducible sentence-level reference record cannot be made without verified segmentation plus sentence alignment. Do not synthesize IDs or assign alignments by row order.

## 12. Recommendation

Use `n_id` only as the source-provided document-pair key for the alignable collection; retain each language's `doc_id` as the individual document identifier. The observed data and official definition establish document-level English/Telugu pairing, not sentence-level alignment. A reproducible document-level source/reference pair can be assembled by matching records on `n_id`, but the requested sentence-level record fields cannot all be populated faithfully from this resource.

For the existing project, do not use alignable records as a direct sentence-level BLEU/chrF++ test set or claim sentence-level context gains. If exploring a document-level baseline/prefix-conditioned comparison later, first define the source-only splitter, whole-document output construction, metric limits, and a split-leakage check. Request a sentence-aligned crosswalk or alignment annotations from the dataset authors if sentence-level scoring is required.

**Execution record**

- Files created: `docs/pralekha_alignable_investigation.md`.
- Files modified: none existing.
- Data downloaded: no Parquet shards or full corpus data. Retrieved pinned card/tree metadata and bounded viewer slices of 8 English plus 8 Telugu alignable records; text was inspected in memory only and no data files were saved.
- Commands/requests executed: Hub dataset SHA lookup; pinned Hub tree query for `alignable` shard paths; pinned raw `README.md` request; Datasets Server `/rows` requests for `alignable/eng` and `alignable/tel` with `offset=0&length=8`; read-only inspection of project evaluation documentation. The eight records were joined by their `n_id` values.
- Tests run: none. This was a read-only data/documentation investigation; no loader, evaluation dataset, translation, or metrics were produced.
