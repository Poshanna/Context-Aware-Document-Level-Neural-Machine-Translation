# IndicDISCO-MT source audit (Phase 14A)

## Summary and structural status

**Final structural status: partially confirmed.** The paper-linked release contains English-Telugu parallel paragraph records, a named `test` split, and narrower sentence/word-alignment benchmark components. The paragraph text preserves its natural within-paragraph sequence and the paired target is a full paragraph, so same-passage source context can be derived from the source string using a documented source-only sentence splitter and then evaluated after reassembling a paragraph output. This is not the same as a released sentence-context table or a document-level corpus with explicit article IDs and ordered paragraph units.

The release does **not** establish parent article/document identity or a cross-paragraph ordering/context chain. ProAlign and LexiAlign are phenomenon-specific alignment subsets; neither turns the release into a general document-level, ordered parallel corpus. Dataset-specific reuse terms were not found, so further real-data processing should wait for rights clarification.

## Official source and revision

- Paper: [IndicDISCO-MT: A Discourse-Centric Benchmark for Evaluating Discourse Phenomena in Indian Language Machine Translation](https://aclanthology.org/2026.eamt-1.15/), Hingrajiya et al., EAMT 2026, ACL Anthology ID `2026.eamt-1.15`, pages 190-204.
- Paper-linked dataset: [HimangY/IndicDISCO-MT](https://huggingface.co/datasets/HimangY/IndicDISCO-MT). This is the exact repository linked in paper footnote 1; no mirror was used.
- Repository snapshot: commit `13f4a8caa71b00859ee4235713aa253c3b2fe193`, last modified 2026-05-25. The repository exposes `main` only; the refs endpoint listed no tags. No semantic release version was found.
- The Hub repository has no README/dataset card. Its API metadata has no dataset license field; the repository tree has no dataset license file.

Sampled files at that pinned commit:

| Component / split | Exact file | File size | File blob OID |
|---|---|---:|---|
| `Indicdisco_MT`, Telugu test | `Indicdisco_MT/Tel/tel_indicdisco-mt_test.csv` | 58,968 bytes | `cada4e09897d97227d5a45abb91daebc4a569804` |
| `DiscoAlign`, Telugu test | `DiscoAlign/Tel/_tel_discoalign_test.csv` | 128,770 bytes | `5728476edb2cce4d2f538991b248438b901cc92d` |
| `ProAlign`, Telugu test | `ProAlign/Tel/tel_proalign_test.csv` | 57,434 bytes | `221ec37299d72a5c0663a2e7232b1cdb8f971992` |
| `LexiAlign`, Telugu test | `LexiAlign/Tel/tel_lexialign_test.csv` | 66,933 bytes | `bd320f6f2477c3db12905653ec694c970a422f93` |

The repository contains `train`, `dev`, and `test` files for each component/language. The paper describes 85 English paragraphs and about 1,030 sentences per language; that is a paper-reported corpus description, not a count independently computed over the release here.

## Minimal data access

- Data split inspected: Telugu `test` only, from the four files above.
- Access method: one HTTP byte-range request (`bytes=0-8191`) for each pinned file; **8,192 bytes per file, 32,768 data bytes total**. Responses were held in memory; no file was saved.
- Complete records structurally reviewed: one `Indicdisco_MT` paragraph pair, two `DiscoAlign` sentence-pair rows, two `ProAlign` sentence-pair rows, and one `LexiAlign` paragraph/alignment row. Partial bytes from additional rows were present in the requested ranges but were not treated as inspected records.
- The official paper PDF was fetched in memory for method/release information (2,816,442 bytes per fetch, three targeted text-extraction passes); it was not saved. Hub API metadata/tree requests were also made; response byte totals were not recorded.
- No full corpus, full CSV, training split, or model checkpoint was downloaded.

## Observed schemas and component semantics

### `Indicdisco_MT`

The sampled `tel_indicdisco-mt_test.csv` header is `source,target`. The sample has a Telugu paragraph in `source` and its English paragraph reference in `target` (sampled complete row: 709 Telugu characters and 624 English characters). No `id`, `document_id`, `paragraph_id`, sentence ID, source sentence span, or reference sentence span is present in this schema.

The paper describes the core corpus as 85 human-translated English paragraphs paired with paragraphs in eight Indian languages. It says the translation task is performed sentence-by-sentence by annotators who are instructed to consider full document context. The release stores paired paragraph strings, not the annotator's sentence-level input/context sequence. Here “document context” is the paper's description of the translation process; the tested CSV row itself is a paragraph-level example, not an identified full article/document.

### `DiscoAlign`

The sampled header is `sno,source,target,Corrected_Alignment`. `source` and `target` are Telugu/English sentence text; `Corrected_Alignment` contains word-level links. This is a sentence-pair/word-alignment representation. The first two sample rows both have `sno=76` and separate sentence strings. The paper describes DiscoAlign as source-to-target word alignment for the IndicDISCO-MT data.

### `ProAlign`

The sampled header is `sno,source,target,gold_pronoun`. Rows contain source/target sentence strings and selected personal-pronoun alignments (for example, source pronoun to English pronoun). The first two sample rows also both have `sno=76`, matching the corresponding initial DiscoAlign sentence pair sample.

### `LexiAlign`

The sampled header is `source,target,gold_lexical`. The first row contains a Telugu and English paragraph-sized text pair (709/624 characters in this sample) and selected lexical alignment links. No example/document identifier or order field appears in this component's header.

The paper says ProAlign is made by filtering DiscoAlign links to personal pronouns, and LexiAlign by filtering alignment links to lexical items such as nouns/proper nouns. These are targeted discourse-phenomenon alignment benchmarks, not general document-context input tables. ProAlign is sentence-pair based; LexiAlign retains paragraph-sized text in the sampled row but targets selected lexical links. Neither supplies a general ordered paragraph chain with preceding context.

## Structural assessment

| Property | Finding from paper and pinned release |
|---|---|
| English-Telugu pair | Present. The published task is Indian language to English; the released Telugu core row is `source=Telugu`, `target=English`. Swapping the paired fields could define the reverse direction, but the paper does not present it as a separately collected English-to-Telugu direction. |
| Explicit example/document identity | Core paragraph CSV: no ID column; the pinned file plus split and physical row position can locate an example, but is not a source-provided document ID. DiscoAlign/ProAlign expose `sno`, but the paper/release gives no schema definition for it; it repeats across sentence rows, so it is not treated here as a unique sentence ID or confirmed article ID. LexiAlign has no ID column in the sampled header. |
| Ordered source units | Paragraph text has its own textual order. The CSV does not expose sentence spans, sentence indices, or ordered prior-unit fields. ProAlign/DiscoAlign provide sentence records, but no explicit sentence ordinal/document sequence field was observed. |
| Ordered target/reference units | Same-row paragraph pairing is available in `Indicdisco_MT`; ProAlign/DiscoAlign provide same-row sentence pairs. No explicit aligned sentence sequence for the paragraph text is present in the core CSV. |
| Source-target alignment | Yes at row granularity for the core Telugu-English paragraph pair, as described by the paper. DiscoAlign gives sentence-pair word alignments; ProAlign gives selected pronoun alignments; LexiAlign gives selected lexical alignments. These labels do not imply a general document alignment graph. |
| Preceding discourse context | The full source paragraph supplies textual context within that paragraph. No separate context field or prior-sentence list is released in the inspected schema. A source-only splitter could derive sentence order within a paragraph, but that segmentation is not an upstream annotation. |
| Same-document causal context | Not confirmed across records or at article/document level: no parent document ID or ordered paragraph index is exposed. Within a single paragraph, causal source context is conditionally derivable from the paragraph's own text order; it must reset at that row boundary and must not use adjacent rows. |
| Reference at output granularity | Yes for a complete paragraph output against its same-row English paragraph reference (Telugu→English); for a ProAlign sentence output against its same-row English sentence text. No sentence reference is established for a sentence produced by splitting the core paragraph pair. |
| BLEU/chrF++ | Structurally possible at paragraph granularity for complete core paragraph outputs against the paired English paragraph, using the released `test` split and verified row provenance. Sentence-level BLEU/chrF++ is not justified for sentence pieces split from the core paragraph pair. ProAlign's sentence references are a selected pronoun subset, not a general MT test set; LexiAlign is selected for lexical links, not a general translation-quality sample. No metrics were calculated. |
| Reproducibility | A pinned Hub commit and named train/dev/test files are available. There is no semantic release tag or dataset card. Record the commit, exact path, split, and row locator; do not elevate row position or `sno` to a document ID without source definition. |
| Academic-use/rights | The paper PDF footer states the **paper** is CC BY-ND 4.0, while the ACL Anthology's general site notice says materials published since 2016 are CC BY 4.0. These statements differ; this audit does not resolve the paper-license discrepancy. Separately, the linked dataset repository exposes no dataset-specific license/card in the inspected metadata/tree. Neither paper statement establishes a license for the underlying translated data. Dataset reuse terms are unresolved; obtain author clarification before further use/redistribution. |

## Phase 14B suitability

**The core `Indicdisco_MT` paragraph-pair component can support a conservative Phase 14B adapter design for passage-level records**, subject to resolving dataset reuse terms before broader processing. A trusted adapter must preserve the exact commit, component/split/path, physical row locator, `source`/`target` locations, Telugu/English direction, and `unit_type=passage`. It must not invent a document ID or label the row locator as one.

ProAlign/DiscoAlign and LexiAlign would need separate task-specific schemas if used: sentence-pair alignment records with their actual annotation fields, and an explicit note that `sno` is opaque until authors document its meaning. They should not be silently flattened into a generic document-context dataset.

The release supports **within-paragraph context as a possible derived input strategy**, not released ordered context units. A later experiment could use a documented Telugu-aware source splitter only inside one paired paragraph, use earlier source sentences for the context condition, and reassemble model outputs to the paragraph granularity before comparing with the paragraph reference. That would be a project-derived segmentation policy, not sentence alignment supplied by IndicDISCO-MT. If Phase 14C requires dataset-provided sentence IDs/alignment or context across paragraphs in a document, this release is insufficient without author-supplied structure.

## Limitations

- Only a few initial test records were inspected; no split-wide audit or count validation was performed.
- The sample shows no explicit dataset document/article identity in the core file, and `sno` semantics are not documented by the dataset card/release.
- The paper's claim that annotators considered full document context describes collection/translation practice; it does not establish a released sequence of ordered sentence/context records.
- No source-target sentence alignment is present in `Indicdisco_MT`; ProAlign/DiscoAlign alignments are specific sentence/word-level annotations, not full parallel document structure.
- The paper's “85 paragraphs / about 1,030 sentences” is not a split-level count from this audit.
- The repo has train/dev/test files at the pinned commit, but the dataset has no card, semantic version, or dataset-specific license declaration.
- English→Telugu is the reverse orientation of the released Telugu→English pairs, not a separately described collection direction.

## Execution record

- Files created: `docs/document_level_source_audit.md`.
- Files modified: none; no adapters or evaluation code changed.
- Official source/revision: ACL Anthology `2026.eamt-1.15`; linked HF repo `HimangY/IndicDISCO-MT`, commit `13f4a8caa71b00859ee4235713aa253c3b2fe193`.
- Sampled split: Telugu `test` for `Indicdisco_MT`, `DiscoAlign`, `ProAlign`, and `LexiAlign`.
- Dataset bytes: 32,768 bytes (four 8 KiB range reads), in memory only. No full dataset or CSV downloaded/saved.
- Paper PDF: 2,816,442 bytes per fetch, read in memory for metadata/methodology; not saved.
- NLLB inference: none.
- BLEU/chrF++ or other metrics: none.
- Final structural status: **partially confirmed**.
