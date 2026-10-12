# CoPara Dataset Audit

## 1. Dataset Identity

- **Official name:** CoPara: The First Dravidian Paragraph-level n-way Aligned Corpus.
- **Official repository:** [ENikhil/CoPara](https://github.com/ENikhil/CoPara), inspected at commit `7c92005159d0dd61c18f99f40b683e6877dc7f69` on its `main` branch.
- **Official paper:** Nikhil E, Mukund Choudhary, and Radhika Mamidi, 2023, [CoPara: The First Dravidian Paragraph-level n-way Aligned Corpus](https://aclanthology.org/2023.dravidianlangtech-1.12/), pages 88–96.
- **Directly observed:** The repository contains `Data/Paragraphs.csv`, `Data/Articles.csv`, `README.md`, and `LICENSE`. The source files were inspected by bounded HTTP byte ranges; neither CSV was downloaded in full.

## 2. Official Sources

- [AI4Bharat Indic NLP Catalog](https://github.com/AI4Bharat/indicnlp_catalog), which links CoPara and describes it as long-context parallel data mined from New India Samachar.
- [CoPara repository at inspected commit](https://github.com/ENikhil/CoPara/tree/7c92005159d0dd61c18f99f40b683e6877dc7f69), including the [README](https://github.com/ENikhil/CoPara/blob/7c92005159d0dd61c18f99f40b683e6877dc7f69/README.md), [paragraph data](https://github.com/ENikhil/CoPara/blob/7c92005159d0dd61c18f99f40b683e6877dc7f69/Data/Paragraphs.csv), [article data](https://github.com/ENikhil/CoPara/blob/7c92005159d0dd61c18f99f40b683e6877dc7f69/Data/Articles.csv), and [repository license](https://github.com/ENikhil/CoPara/blob/7c92005159d0dd61c18f99f40b683e6877dc7f69/LICENSE).
- [ACL Anthology paper](https://aclanthology.org/2023.dravidianlangtech-1.12/), which describes the data creation, paragraph alignment, quality review, and NMT use.

## 3. Languages

**Official documentation:** CoPara is n-way aligned across five languages: English, Kannada, Malayalam, Tamil, and Telugu. The CSV header directly observed in `Paragraphs.csv` and `Articles.csv` contains one text column for each of these languages.

## 4. English-Telugu Availability

**Yes.** The n-way data includes both `English` and `Telugu` passage columns in the same records. The first two sampled rows contain non-empty values for both columns, and the paper describes each n-way tuple as spanning all five languages. Thus English-Telugu passage pairs are part of the resource.

## 5. Dataset Size

There is a source discrepancy:

- The author repository README and paper report **2,856 n-way aligned paragraph/passage tuples** across English and four Dravidian languages. Under the documented five-language n-way structure, the reported English-Telugu passage-pair count is 2,856.
- The AI4Bharat catalog entry instead says **2,586 passage pairs**. It does not explain that count or give a per-language breakdown.
- **Direct observation:** `Paragraphs.csv` uses a zero-based `Index`; sampled rows include indices 0 and 1, and a small range at the file tail contains index 2855. The complete CSV was not scanned, so row continuity and non-empty English/Telugu cells for every row were not independently verified.

Use **2,856 as the author-reported n-way count**, but record the catalog conflict. An authoritative correction to the catalog discrepancy is not established by the inspected sources.

The paper also reports 1,893 n-way aligned articles before paragraph splitting. `Articles.csv` is the article-level version; this figure is paper-reported, not a full-file row count performed for this audit.

## 6. Data Format and Schema

The repository distributes CSV files. The observed header is the same in `Paragraphs.csv` and `Articles.csv`:

| Columns | Logical content/type |
|---|---|
| `Index` | Zero-based row index; integer-valued in the inspected CSV rows |
| `English`, `Kannada`, `Malayalam`, `Tamil`, `Telugu` | Passage/article text; strings |
| `<Language> Tokens`, `<Language> Sentences`, `<Language> Words` for each of the five languages | Per-language count metadata; integer-valued in inspected rows |

Expanded, the 21 columns are: `Index`, `English`, `English Tokens`, `English Sentences`, `English Words`, `Kannada`, `Kannada Tokens`, `Kannada Sentences`, `Kannada Words`, `Malayalam`, `Malayalam Tokens`, `Malayalam Sentences`, `Malayalam Words`, `Tamil`, `Tamil Tokens`, `Tamil Sentences`, `Tamil Words`, `Telugu`, `Telugu Tokens`, `Telugu Sentences`, and `Telugu Words`.

No separate machine-readable schema or explicit CSV dtype declaration was found. The `Sentences` columns are counts, not sentence text arrays or sentence alignment annotations. There are no sentence IDs, offsets, token links, or sentence-alignment columns.

## 7. Passage/Document Structure

- `Paragraphs.csv` has one indexed row containing a passage/paragraph text field for each of the five languages. This makes the passage tuple boundary explicit at the CSV-record level.
- `Articles.csv` is the corresponding article-level aligned data, according to the README; its row boundary is explicit, but no sentence alignment is represented there either.
- **Official paper:** Authors first aligned articles using annotator-marked article breaks, then compared the number of internal paragraphs across languages. Where counts matched, corresponding paragraphs were assumed congruent and retained, including the one-paragraph case. Paragraphs over 512 tokens under the IndicBART tokenizer were filtered.
- **Inference:** The usable unit for long-context translation is a paired passage/paragraph (or, using `Articles.csv`, an article), not a sentence pair. Whether each article record is a complete issue or an entire magazine document: Not established from the inspected data.

## 8. Sentence Structure

Sentence boundary spans are not explicit. The text cells are passage strings and the `English Sentences`, `Telugu Sentences`, and analogous columns report only counts.

Direct sample evidence from `Paragraphs.csv`:

| `Index` | English sentence count | Telugu sentence count |
|---:|---:|---:|
| 0 | 4 | 7 |
| 1 | 2 | 2 |

The paper's corpus-level averages also differ: 3.8 English sentences per paragraph and 4.5 Telugu sentences per paragraph. Equal counts in one sample row do not imply sentence alignment; the unequal counts in another show that equal-count, one-to-one sentence correspondence cannot be assumed for every passage.

Sentence-level reference alignment is not established.

## 9. Alignment

- **Document/article level:** Article-level records are described as aligned across languages; 1,893 n-way articles are reported in the paper.
- **Passage/paragraph level:** Yes, the core 2,856 records are described as n-way aligned paragraph/passage tuples, including English and Telugu. The paper's paragraph procedure aligns paragraph positions by manual layout-based segmentation and accepts articles when internal paragraph counts match.
- **Sentence level:** No sentence-to-sentence mapping or paired sentence spans are present in the CSV schema or described as dataset annotations. Sentence-level reference alignment is not established.
- **Token level:** No token alignment is represented. Token counts are aggregate metadata only.
- **Order:** Paragraph position is part of the authors' paragraph-level alignment procedure. Sentence order correspondence inside aligned paragraphs is not annotated or verified by the inspected data.

The paper's alignment procedure is not purely automatic: annotators use visual layout cues and mark article/paragraph boundaries; equal paragraph counts are then treated as congruent positions. That is paragraph-level alignment, not evidence that sentences align one-to-one or that sentence-level alignments can be recovered.

## 10. Context Reconstruction

Previous source sentences can be reconstructed causally **within a passage** after applying a documented English sentence splitter: retain the source passage row boundary, preserve its text order, and provide only earlier source sentences plus the current source sentence. Do not use Telugu reference text or any future source sentence as model context.

This context is derived, not an annotation in CoPara. The data provides passage boundaries and sentence counts, but not sentence spans or stable sentence IDs. A deterministic splitter is required, and its boundaries could differ from the authors' sentence-counting procedure. Context reconstruction does not create sentence-level reference alignment.

## 11. Evaluation Compatibility

| Experiment/component | Finding |
|---|---|
| Baseline: current sentence → NLLB | Source sentences can be derived from each English passage and translated independently. CoPara does not supply a Telugu reference sentence for each derived source sentence, so sentence-level reference scoring is unsupported. |
| Context-conditioned: prior source sentences + current sentence → NLLB | A causal source prefix can be derived within each passage after sentence splitting. This supports a passage-level context experiment in principle, but the raw data is not directly in the project's sentence/context JSONL format. |
| BLEU and chrF++ against sentence references | No. There are no aligned Telugu reference sentences. Do not split both languages and pair by sentence index. |
| BLEU and chrF++ against passage references | Potentially yes: compare one generated passage output with the English/Telugu passage tuple. This is passage-level evaluation and requires a documented output-construction and text-normalization policy. No metrics were computed here. |
| Optional BERTScore and COMET | Potentially usable on passage text pairs, subject to each metric model's input limits and a consistent, documented truncation/chunking policy. No metrics were computed here. |

For baseline/context comparison, a defensible passage-level design could translate source sentences independently for the baseline, add only preceding source sentences for the context condition, join each condition's sentence outputs in source order, and score one joined prediction against the paired Telugu passage. This does not yield sentence-level BLEU/chrF++ and must be evaluated as passage-level output. It would require preprocessing and evaluation changes; no loader or dataset was created in this audit.

## 12. License and Provenance

- **Repository license:** GitHub identifies the repository as GPL-3.0 and the repository includes a GNU GPL v3 `LICENSE` file. The README does not state a separate license specifically for the CSV corpus text.
- **Source content:** The paper says CoPara uses New India Samachar, a magazine published by India's Ministry of Information and Broadcasting, and selected 15 issues published in 2022, each represented across English and four Dravidian language editions. The catalog likewise describes the passages as mined from New India Samachar.
- **Construction:** The paper reports six annotators using magazine layout cues to segment and align the language editions. Google Cloud Vision OCR was used for text extraction; annotators refined OCR errors and formatting. The paper additionally reports artificial cross-lingual similarity checks and human alignment-quality ratings.
- **Translation status:** The data consists of multilingual magazine editions aligned through CoPara's pipeline; the sources inspected do not state who translated the magazine content or whether every Telugu passage is a human translation of the English passage. Translation provenance: Not established from the inspected data.
- **Reuse caution:** A repository-level GPL-3.0 label does not by itself clarify rights in the underlying magazine text. No source-text redistribution permission or separate data license was established. Confirm permissions and attribution requirements with the authors/source before redistributing passages or derived text.

## 13. Risks and Limitations

- The AI4Bharat catalog count (2,586) conflicts with the authors' README and paper (2,856). The paper/repository count is supported by the sampled terminal index 2855, but the full CSV was not audited for gaps or missing language cells.
- The published resource is passage/paragraph-aligned, not sentence-aligned. `Sentences` are counts only; they cannot pair a source sentence with a Telugu reference sentence.
- Sentence-level one-to-one, one-to-many, many-to-one, and many-to-many alignment types are all unknown. Passage-level one-to-one positional tuples are the documented unit; do not promote them to sentence alignment.
- The authors' paragraph method assumes that equal internal paragraph counts indicate congruent paragraph positions. This is an explicit alignment heuristic, not sentence-level validation.
- The paper uses OCR and describes human cleanup; residual OCR, formatting, and alignment noise remain possible. The paper reports human/artificial quality assessment, but this audit did not independently validate quality across the dataset.
- Source material is limited to sampled 2022 New India Samachar issues and their multilingual editions; domain and publication coverage are narrow.
- No independent benchmark split or held-out document protocol was established from the inspected files. The paper reports sampling 5% of the corpus for its own NMT evaluation; that does not itself define a reusable sentence-aligned test set for this project.
- Repository GPL-3.0 status does not settle the underlying magazine text's redistribution terms.

## 14. Recommendation

CoPara is a **conditional candidate for an academic passage-level English-Telugu long-context experiment**: it provides corresponding English and Telugu passage strings, a reported 2,856 five-language n-way passage tuples, and passage boundaries that allow source-only prior context to be constructed after documented sentence splitting. A fair baseline-versus-prefix-conditioned experiment could score complete generated passages against the paired Telugu passage using passage-level BLEU/chrF++ and optionally document-level BERTScore/COMET with validated length handling.

It is **not a sentence-aligned evaluation corpus**. Sentence-level BLEU/chrF++ against paired Telugu sentence references, per-sentence translation accuracy claims, and sentence-level one-to-many alignment analyses are unsupported. Before adoption, reconcile the 2,586/2,856 catalog discrepancy, establish a held-out evaluation protocol, and confirm text reuse terms. Do not present derived sentence splits as dataset sentence alignments.

**Execution record**

- Files created: `docs/copara_audit.md`.
- Files modified: none existing.
- Data downloaded: No full corpus files. Read 100,003 bytes total from bounded HTTP ranges across `Paragraphs.csv` and `Articles.csv`; no CSV data was saved. Also fetched the repository README/license, metadata, and the 488,341-byte ACL paper PDF in memory; no files were written to the workspace.
- Commands/requests executed: GitHub API requests for repository metadata, current commit, and tree; pinned raw README/LICENSE GETs; bounded `Range` GETs for both CSV headers/initial rows and the `Paragraphs.csv` tail; ACL Anthology paper PDF GET and local in-memory text extraction; AI4Bharat catalog search. No shell commands changed project state.
- Tests run: none. This was a read-only source/data audit; no model inference or translation metrics were run.
