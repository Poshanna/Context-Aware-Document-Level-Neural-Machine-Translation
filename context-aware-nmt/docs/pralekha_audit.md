# PRALEKHA English-Telugu Evaluation Audit

## 1. Dataset Identity

PRALEKHA is an AI4Bharat document-level multilingual corpus and cross-lingual document-alignment benchmark. It covers English (`eng`) and Telugu (`tel`) among 12 languages. The documented domains are Indian Press Information Bureau (PIB) news bulletins and Mann Ki Baat (MKB) podcast scripts; the dataset card describes the data as human-written or human-verified.

The dataset is relevant to this project because its MT development and test examples are document pairs, not unrelated sentence rows. It is not, by itself, a sentence-aligned evaluation corpus.

## 2. Official Sources

- [PRALEKHA Hugging Face dataset and card](https://huggingface.co/datasets/ai4bharat/Pralekha)
- [PRALEKHA GitHub repository](https://github.com/AI4Bharat/Pralekha)
- [PRALEKHA paper, ACL Anthology](https://aclanthology.org/2025.ijcnlp-long.37/)
- [PRALEKHA paper, arXiv](https://arxiv.org/abs/2411.19096)
- [Hugging Face dataset API metadata](https://huggingface.co/api/datasets/ai4bharat/Pralekha)
- [Hugging Face Datasets Server split metadata](https://datasets-server.huggingface.co/splits?dataset=ai4bharat/Pralekha)

The Hub API reported repository revision `4c16a270141891dd5c3e98f6f4f207f11bd11236` during this audit. Pin this SHA for reproducible retrieval; the `main` branch is mutable.

## 3. License

The Hugging Face dataset card declares the **dataset** under CC BY 4.0. The GitHub repository also displays an MIT license for its repository/code; do not confuse that code license with the dataset license. Attribute PRALEKHA/AI4Bharat, cite the paper, retain the dataset license notice, and identify any modifications when redistributing derived material. The card identifies PIB and Mann Ki Baat as source domains and asks users to contact the maintainers about copyright concerns. The dataset-level license declaration should not be treated as a substitute for checking any source-specific restrictions before redistributing source text or derived artifacts.

Citation:

```bibtex
@inproceedings{suryanarayanan-etal-2025-pralekha,
  title = "{PRALEKHA}: Cross-Lingual Document Alignment for {I}ndic Languages",
  author = "Suryanarayanan, Sanjay and Song, Haiyue and Khan, Mohammed Safi Ur Rahman and Kunchukuttan, Anoop and Dabre, Raj",
  booktitle = "Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics",
  year = "2025",
  pages = "662--676",
  url = "https://aclanthology.org/2025.ijcnlp-long.37/"
}
```

## 4. Available English-Telugu Subsets

The English-to-Telugu document-pair subset is `dev` or `test`, with split name `eng_tel` in each. The Hub metadata reports 1,000 examples in each split. Sampled rows use `src_lang="eng"` and `tgt_lang="tel"`. The `train/eng_tel` split is also listed, but this audit did not inspect or download it.

`alignable` is a separate document collection, with language splits `eng` and `tel`. Its records are useful for studying document identity and cross-language pairing; they are not the MT dev/test schema.

## 5. Actual Data Schema

Observed fields and meanings:

| Subset | Observed fields | Interpretation |
|---|---|---|
| `alignable/eng`, `alignable/tel` | `n_id`, `doc_id`, `lang`, `text` | `n_id` identifies a paired document across languages; `doc_id` identifies one language-specific document; `lang` is the ISO 639-3 language code; `text` is its content. |
| `dev/eng_tel`, `test/eng_tel` | `src_lang`, `tgt_lang`, `src_txt`, `tgt_txt` | Paired source and target document strings, with language codes. |

The current card prose calls the pair fields `src_text` and `tgt_text`, but the current Hub feature metadata and sampled rows use **`src_txt` and `tgt_txt`**. Implementations should inspect the loaded dataset features instead of assuming the prose field names.

No `n_id` or `doc_id` field was present in the sampled dev/test rows. Do not transfer alignable IDs to these rows based on content resemblance or row order.

## 6. Document-Level Structure

The sampled `alignable` English and Telugu row at `row_idx=0` both have `n_id="mkb-1"`, but have different `doc_id` values and language codes (`eng` and `tel`). Their `text` fields contain long, continuous MKB episode content, establishing that these sampled `text` values are documents rather than individual sentences.

Sampled `dev/eng_tel` and `test/eng_tel` records likewise contain long paired MKB episode text in `src_txt` and `tgt_txt`. The dev example includes dialogue/transcript material and audio markers within the episode. These are whole-document examples with internal paragraph/newline structure; they are not sentence-level rows.

## 7. Sentence/Paragraph Structure

Document strings contain newlines and paragraph-like breaks. Their paragraphing is not a parallel alignment annotation: boundaries, spacing, and segmentation differ between English and Telugu. The sampled Telugu material also includes dialogue, quoted material, and transcript/audio markers. Neither schema exposes sentence IDs, paragraph IDs, timestamps, or links between individual source and target sentences.

Sentence boundaries can be heuristically reconstructed from punctuation and line breaks, but punctuation-only segmentation is not guaranteed reliable for abbreviations, quotations, speaker turns, and language-specific text. A deterministic, Telugu-aware splitter with a documented version and rules is needed for sentence-level model inputs. It must not be represented as dataset-provided segmentation.

## 8. Alignment Evidence

The explicit cross-language evidence in the sampled `alignable` records is the shared `n_id`: English and Telugu both use `mkb-1`; their individual `doc_id`s are distinct. This supports matching those two document records, not matching their sentences.

In `dev/eng_tel` and `test/eng_tel`, the source and target fields occur together in one record and are explicitly labelled by `src_lang`/`tgt_lang`. That is sufficient document-pair evidence for document-level evaluation. It does not supply sentence alignment. No consecutive-row grouping or synthetic document IDs are needed or justified.

## 9. Context Extraction Feasibility

Same-document prior context is feasible at the document level: segment a single `src_txt` string into source sentences in order, and for each current sentence provide only a fixed number of preceding source sentences from that same row. Reset context at each document boundary and never pass future source sentences or reference target text to the model.

This supports a controlled comparison between:

- **Baseline, sentence-only NLLB-200:** translate each segmented source sentence with no preceding context.
- **Context-aware, prefix-conditioned NLLB-200:** use the same segmentation and decoding settings, adding a fixed, documented source prefix from earlier sentences in the same document.

Keep the checkpoint, language codes, decoding, sentence splitter, context-window size, separators, and truncation policy constant or explicitly recorded. No inference or metric values were produced in this audit.

## 10. Dev/Test Suitability

**Conditional yes:** `dev/eng_tel` and `test/eng_tel` can support English-to-Telugu document-level evaluation, and the same paired records can be reversed for Telugu-to-English by swapping input/reference roles. The latter is a reverse-direction evaluation of the same documents, not an independently sampled reverse-direction benchmark.

Compare complete generated documents with their paired reference documents using corpus-level BLEU and chrF++ (for SacreBLEU, record the full signature and use `CHRF(word_order=2)` for chrF++). Neither metric requires sentence-to-sentence reference alignment when complete document outputs are compared. Preserve one prediction/reference pair per source document and report both directions and both model conditions separately.

Optional BERTScore or COMET may be reported only with the exact model/checkpoint, library version, and scoring procedure recorded. Long documents may exceed metric-model input limits; any chunking or truncation must be identical across systems, documented, and distinguished from whole-document scoring.

The sampled test rows require QA before adoption: `row_idx=2` was returned with English content in both `src_txt` and `tgt_txt` (the target appeared to duplicate the source). Treat this as a concrete suspected bad pair, not evidence about the prevalence of the issue. Verify the entire 1,000-row test split and dev split for language mismatch, identical source/target content, empty/near-empty text, and suspicious length ratios before publishing results. Freeze the resulting inclusion/exclusion manifest and report exclusions; do not silently repair or drop rows.

The card says dev/test documents are sampled from `alignable`, but the sampled MT rows do not carry their `n_id`/`doc_id` identifiers. Thus the design is documented, while per-row identity and split leakage cannot be audited from these fields alone. Hashes can be used as reproducibility checks, not as fabricated dataset IDs.

## 11. Risks and Limitations

- No sentence alignment is provided. Sentence-level BLEU/chrF++ comparisons between individual source and target sentences would be unsupported and potentially misleading.
- Source and target sentence/paragraph boundaries may not correspond one-to-one. Translation can merge, split, reorder, omit, or add material.
- Punctuation-based segmentation can make the context-aware result depend on the splitter. Publish the exact splitter/version and keep it fixed across conditions.
- Sampled dev/test rows have no explicit document IDs. Use the pinned dataset revision, split, original row index, and SHA-256 of the raw source/target strings in a local manifest; do not expose the text in that manifest unless necessary.
- One sampled test pair appears to have English in the Telugu target field. Dataset-wide quality checks are needed before treating reported aggregate scores as clean.
- These documents include different MKB episodes and PIB bulletins and may vary greatly in length and content. Report document counts, length distributions, and per-domain/source observations where identifiable; do not infer domains from adjacency.
- BERTScore/COMET document limits and Telugu metric quality require validation. They are supplementary, not prerequisites for a reproducible BLEU/chrF++ comparison.
- CC BY 4.0 is the dataset card's declaration; check source-specific attribution or reuse terms before redistributing texts or artifacts.

## 12. Recommendation

Proceed with PRALEKHA as a **conditional document-level evaluation candidate**, using only the published `dev/eng_tel` and `test/eng_tel` pairs and pinning the Hugging Face revision. Evaluate whole-document outputs for BLEU and chrF++; add reverse-direction runs by swapping each pair. For the context-aware condition, derive causal prior-source context only within the same document and compare against a sentence-only baseline using identical segmentation and decoding.

Before final adoption, run a split-wide, text-free QA report for both 1,000-row sets, investigate the sampled English-to-English target, and freeze a row-index/content-hash manifest with explicit exclusions. Do not claim sentence-level reference alignment or use `alignable` row adjacency to create one. If a clean, independently verified evaluation subset is a requirement, contact the dataset authors because the MT split has no exposed document IDs for direct traceability.

## 13. Exact Reproduction Steps

The inspection used metadata and a maximum of three rows per requested split; no Parquet shard or full dataset was downloaded. The observed Hub revision was `4c16a270141891dd5c3e98f6f4f207f11bd11236`.

The metadata and sample requests were HTTP GETs to these exact URLs:

```text
https://huggingface.co/api/datasets/ai4bharat/Pralekha
https://datasets-server.huggingface.co/splits?dataset=ai4bharat/Pralekha
https://datasets-server.huggingface.co/rows?dataset=ai4bharat/Pralekha&config=alignable&split=eng&offset=0&length=3
https://datasets-server.huggingface.co/rows?dataset=ai4bharat/Pralekha&config=alignable&split=tel&offset=0&length=3
https://datasets-server.huggingface.co/rows?dataset=ai4bharat/Pralekha&config=dev&split=eng_tel&offset=0&length=3
https://datasets-server.huggingface.co/rows?dataset=ai4bharat/Pralekha&config=test&split=eng_tel&offset=0&length=3
https://datasets-server.huggingface.co/first-rows?dataset=ai4bharat/Pralekha&config=alignable&split=eng
https://datasets-server.huggingface.co/first-rows?dataset=ai4bharat/Pralekha&config=alignable&split=tel
https://datasets-server.huggingface.co/first-rows?dataset=ai4bharat/Pralekha&config=dev&split=eng_tel
https://datasets-server.huggingface.co/first-rows?dataset=ai4bharat/Pralekha&config=test&split=eng_tel
```

Equivalent PowerShell inspection commands:

```powershell
$base = "https://datasets-server.huggingface.co"
Invoke-RestMethod "$base/splits?dataset=ai4bharat/Pralekha"
Invoke-RestMethod "$base/rows?dataset=ai4bharat/Pralekha&config=alignable&split=eng&offset=0&length=3"
Invoke-RestMethod "$base/rows?dataset=ai4bharat/Pralekha&config=alignable&split=tel&offset=0&length=3"
Invoke-RestMethod "$base/rows?dataset=ai4bharat/Pralekha&config=dev&split=eng_tel&offset=0&length=3"
Invoke-RestMethod "$base/rows?dataset=ai4bharat/Pralekha&config=test&split=eng_tel&offset=0&length=3"
```

For a pinned local reproduction with Hugging Face `datasets` installed (this loads the two small evaluation splits, not the large training data):

```python
from datasets import load_dataset

revision = "4c16a270141891dd5c3e98f6f4f207f11bd11236"
for config in ("dev", "test"):
    rows = load_dataset(
        "ai4bharat/Pralekha",
        config,
        split="eng_tel",
        revision=revision,
        streaming=True,
    )
    print(config, rows.features)
    for index, row in enumerate(rows):
        print(index, row["src_lang"], row["tgt_lang"])
        print("source chars:", len(row["src_txt"]), "target chars:", len(row["tgt_txt"]))
        if index == 2:
            break
```

This audit ran no project test suite. No translation or evaluation metrics were computed.
