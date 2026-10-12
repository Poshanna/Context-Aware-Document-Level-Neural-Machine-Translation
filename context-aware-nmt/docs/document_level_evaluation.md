# Document-Level Evaluation Methodology

## Purpose and Scope

Phase 11 separates sentence-reference evaluation from document/passage context records. It does not load datasets, call translation models, calculate metrics, or claim that context improves translation. `backend/app/evaluation/document_evaluation.py` is an isolated standard-library layer. The Phase 6/7 runner, NLLB implementation, Samanantar preprocessing, and TED2020 loader remain unchanged.

The current model remains a sentence-level **pretrained checkpoint**. Prefix conditioning is an experiment around that checkpoint, not a new document-aware architecture.

## Compatibility Boundary

| Existing Phase 6/7 evaluator | Phase 11 layer |
|---|---|
| Reads the existing sentence-level JSONL test representation | Defines provenance-aware sentence/document/passage records |
| Runs baseline and eligible context-conditioned translation | Stores supplied outputs and context-use status; does not run a model |
| Calculates corpus BLEU/chrF++ and optional metrics | Computes eligibility only; does not calculate any metric |
| Writes the existing evaluation artifacts | Builds an in-memory, deterministic audit/report object |

There is no adapter from Phase 11 records to the existing runner, and Phase 11 is not a drop-in replacement. A later integration must be a separately reviewed boundary and must not infer document membership from row order.

## Provenance and Trusted Adapter

`DatasetProvenance` distinguishes `synthetic`, `verified_dataset`, `verified_annotation`, and `unverified`. Synthetic and unverified objects are made through explicit factories. Verified objects cannot be directly constructed; only a `TrustedDatasetAdapter` implementation can issue them. The structured verified provenance records dataset name, pinned revision, split, source record ID, document/passage unit ID and type when present, source/reference/alignment locations, and the adapter name.

This is **not cryptographic verification**. Python code can be modified or a purported adapter can be untrustworthy. The trust boundary is reviewed adapter code: it must construct provenance from the record actually loaded, preserve the upstream row/location and revision, and only claim reference/alignment locations that exist in the source or a separately verified annotation. Free-form source strings, evidence enums, and caller-chosen IDs alone do not authorize metrics. An arbitrary matching ID is not proof that two rows share a document.

No real dataset adapter is implemented in Phase 11. There is consequently no route from raw dataset rows to verified provenance in this phase. Synthetic fixtures cannot set dataset name/revision/record locations, and `metric_eligibility` suppresses metrics for both synthetic and unverified provenance even when a synthetic reference is present. `synthetic_fixture` alignment evidence is accepted only with synthetic provenance and is never research-metric eligible.

For verified records, the adapter must provide non-empty `dataset_name`, immutable `dataset_revision`, `split`, `record_id`, and `source_location`. It supplies `reference_location` only for a loaded reference and `alignment_location` only where the claimed pairing/alignment is actually established. For document/passage context, the provenance `unit_id` and `unit_type` must match the evaluation record. Every prior-context item carries its own provenance and must identify the same dataset revision, split, adapter, and unit.

## Track Contracts

| Track | Current text unit | Identity/context | Available reference contract |
|---|---|---|---|
| `sentence_level` | Sentence only | No document/passage ID or context; baseline only | If available, must be a sentence reference with sentence-alignment evidence |
| `document_context` | Sentence input or complete document input | Must carry a provenance-bound document ID/type; context prefixes are allowed only for sentence inputs | Reference must match the current text unit. A document pair does not become a sentence reference |
| `passage_context` | Sentence input or complete passage input | Must carry a provenance-bound passage ID/type; context prefixes are allowed only for sentence inputs | Reference must match the current text unit. A passage pair does not become a sentence reference |

Available reference text must have exactly the same `TextUnit` as `current_text`; a mismatch is rejected rather than converted or merely relabeled. Sentence-level references require sentence-alignment evidence. Document and passage references require their corresponding pair evidence. An unavailable reference has no text or reference unit. If the source contains a document/passage pair but not a reference for the current sentence, that sentence record marks its reference unavailable; a separate whole-document/passage record may retain a same-granularity reference.

## Context Validation

Context state is explicit: `context_unavailable`, `context_available_not_used`, or `context_supplied_to_model`, with a separate checked boolean indicating whether it was actually supplied. `context_strategy` is currently the explicit `last_n` strategy. Context positions are zero-based, contiguous local sentence-order positions, not invented dataset sentence IDs.

For `last_n`, a sentence at position `i` with window `k` must carry exactly the available positions from `max(0, i-k)` through `i-1`, in order. The record rejects future/current context, duplicate or out-of-order positions, another unit ID/type, another dataset/revision/split/adapter, and context without provenance-bound unit identity. The report builder rejects duplicate current positions and repeated verified source record IDs. No semantic retrieval or sentence splitting is implemented here; a future adapter/splitter must establish the positions and pass the actual source spans.

## Audit Reports and Metric Gates

Each report item retains source/target language, track, full structured provenance and locations, unit ID/type, current position and text, reference text/availability/unit, alignment status/evidence, context entries/strategy/window/status/use, both baseline and context-conditioned translations, all metric eligibility decisions, and any qualitative observation or exclusion reason. The same item supplies the current text to both conditions. Reports sort by track, dataset/revision/split, unit ID, unit position, and source record ID; original input ordinal is the final deterministic tie-breaker and is not a dataset ID. JSON serialization follows the dataclass field order and stable list ordering.

`metric_eligibility` is a gate only. It requires verified dataset/annotation provenance, an available reference at the same granularity as the current output, matching alignment evidence, and a translation for the requested condition. It never calls BLEU, chrF++, BERTScore, COMET, or a model. Sentence BLEU/chrF++ eligibility is per item, but the project's BLEU methodology is corpus-level: later execution must aggregate eligible same-scope examples as a corpus and must not average sentence BLEU values. Optional BERTScore/COMET also require exact text-unit matches and a separate, consistent policy for model length limits.

Suppressed metric decisions include explicit reasons. Records with no eligible reference metric are listed for qualitative-only review; excluded records retain their explicit exclusion reason in both the item and exclusion summary. Qualitative observations are not quality scores and cannot support an improvement claim.

## Dataset Evidence and Appropriate Use

### Samanantar

The current processed representation contains source/reference sentence pairs and supports sentence-level quantitative evaluation for valid pairs. It does not establish reliable document relationships. Do not group adjacent rows into documents. Samanantar can populate `sentence_level`; document context requires a separately verified document/context source.

### PRALEKHA

The exposed `dev/eng_tel` and `test/eng_tel` files pair English and Telugu document strings but expose neither sentence alignment nor document IDs. They can support document-context records only after an adapter preserves legitimate document identity; they do not support sentence-level BLEU/chrF++ from those evaluation files. The separate `alignable` resource's `n_id` is a document-pair ID, not sentence alignment, and cannot be transferred to evaluation rows by order.

### CoPara

CoPara provides English-Telugu paragraph/passage pairs with passage boundaries. Its sentence-count columns are counts, not sentence spans or correspondences. A passage record must retain `TextUnit.PASSAGE` and its passage reference; do not split the pair and pair sentences by position. Passage-level metrics are eligible only for complete passage predictions against the paired passage reference and verified provenance. Sentence-level scoring requires separately verified sentence alignment. Reconcile the catalog's 2,586 count with the author-reported 2,856, and clarify New India Samachar text reuse terms before redistribution.

### TED2020

TED2020 remains unused while data rights are unresolved. Do not download or process it until permissions are clarified. Its existing loader is unchanged and isolated; a future adapter must preserve its explicit XML alignment and rights metadata rather than bypass them.

## Future Adapter Procedure

A future adapter must be reviewed independently and must:

1. Load only user-provided data under the applicable license; do not download a dataset automatically.
2. Pin and record the dataset revision and split.
3. Create verified provenance from actual loaded record IDs and source/reference/alignment locations; never mint document IDs or alignment labels.
4. Set `unit_id`/`unit_type` only from an observed upstream field or explicit annotation.
5. Preserve sentence ordering from documented segmentation and attach provenance to every context source unit.
6. Mark reference unavailable whenever the source sentence lacks an aligned reference at the requested granularity.

The adapter is a trusted application boundary, not a proof that the upstream dataset itself is correct. Human audit remains necessary for provenance and rights.

## Synthetic Demonstration

The CLI requires `--demo`, accepts no data path, and uses factory-created synthetic provenance only. Its records remain metric-ineligible, even when they contain constructed references. The CLI does not access files, datasets, models, or metric libraries.

```powershell
python scripts/run_document_evaluation.py --demo
```

Unit tests use only synthetic fixtures. Phase 11 currently has no verified real-data adapter and is not ready to authorize real-data metric computation.