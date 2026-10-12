# IndicDISCO-MT paragraph adapter (Phase 14B)

## Scope

This is **paragraph-level infrastructure**, not document-level context evaluation. It adapts an already-loaded row from the core `Indicdisco_MT` component only. It does not fetch files, read local dataset files, split paragraphs, run a model, build context, or calculate metrics.

The adapter is pinned to:

- Dataset: `IndicDISCO-MT`
- Official repository: `HimangY/IndicDISCO-MT`
- Revision: `13f4a8caa71b00859ee4235713aa253c3b2fe193`
- Component: `Indicdisco_MT`
- Supported split names: `train`, `dev`, `test`
- Observed direction: Telugu source to English reference
- Observed row fields: exactly `source`, `target`

A caller must pass the values and provenance locations from a row loaded from that pinned release. The adapter fails closed for another revision, component, language direction, missing row locator/location, empty text, or a row with an unexpected schema.

## Record contract

Each accepted row preserves the complete source and reference paragraph exactly as supplied, with:

- `unit_type = paragraph`
- `alignment = paragraph_pair` (same-row source/reference pairing only)
- `document_id = None`
- `ordering = None`
- `cross_row_context = unavailable`
- `previous_paragraph_context = unavailable`
- `same_document_context = unavailable`
- `within_paragraph_sentence_context = unavailable`
- `rights_status = unresolved`

The `row_locator` is stored as a record locator in provenance. It is not a source-provided document ID and is never promoted to one. Shared `DatasetProvenance` carries dataset, pinned revision, split, row locator, source/reference/alignment locations, and verified status, while leaving `unit_id` and `unit_type` unset so no document or paragraph ID is minted.

## Metric gates

`metric_eligibility` is structural only; it never computes a score.

- Paragraph BLEU: structurally eligible against the same-row English paragraph reference.
- Paragraph chrF++: structurally eligible against the same-row English paragraph reference.
- Sentence BLEU: unavailable; the core paragraph rows do not provide sentence references/alignment.
- Sentence chrF++: unavailable for the same reason.
- Document-context BLEU/chrF++: unavailable; parent-document identity, cross-row order, and document context are not established.

Paragraph metric eligibility is not authorization to use or redistribute the underlying text. Rights remain unresolved, and no metric has been calculated.

## Context boundary

The adapter never constructs cross-paragraph context. `reject_cross_row_context` rejects any non-empty set of preceding rows because the release does not establish that rows share a parent document or their order within one. It does not split a paragraph into sentences: within-paragraph sentence context is also marked unavailable in this paragraph-scoring contract. Any future sentence-context design would require a separately reviewed segmentation/input contract and must retain paragraph-level references/output granularity unless actual sentence-aligned references become available.

## Rights and limitations

Phase 14A found no dataset-specific license/card in the linked Hub repository. The paper PDF states CC BY-ND 4.0 for the paper, while the ACL Anthology's general site notice describes post-2016 ACL materials as CC BY 4.0; that paper-license discrepancy is unresolved. Neither statement establishes reuse rights for the dataset. The adapter does not authorize redistribution; obtain dataset-specific permission/terms before real-data use beyond the audit sample.

The adapter has only been exercised with synthetic test fixtures. No real IndicDISCO-MT file was processed by this implementation, no NLLB inference was run, and no BLEU/chrF++ metric was computed.

## Next gate

Before any real-data structural validation or evaluation, resolve dataset rights and define a real loader that preserves the pinned commit, component, split, exact row locator, source/reference locations, and observed Telugu-to-English labels. Continue to keep document identity, ordering, and all context states unavailable unless authoritative dataset fields establish them.

## Phase 14C: Bounded Real-Data Structural Validation

Run `scripts/validate_indicdisco_structure.py` only with explicit source/revision/component/split/language arguments. Remote input is accepted only when it is the exact Hugging Face URL for the pinned commit and selected core component/split. The validator fetches byte ranges in small chunks, with defaults of at most 3 complete rows and 16 KiB; the absolute row-reading cap is 16 KiB. An explicit local file is accepted as pinned only after the validator fetches that file's size and Git blob OID from the exact revision's official Hugging Face tree, checks the local file size, and streams the local bytes through the Git blob hash algorithm for an exact OID comparison. The locally calculated object ID is only compared to the official upstream OID; it is not treated as a new upstream checksum. If official metadata is unavailable or the size/OID differs, the validator fails closed. This verifies one local split file, not the corpus as a whole.

Example for the pinned official Telugu test split:

```powershell
python scripts/validate_indicdisco_structure.py `
	--source "https://huggingface.co/datasets/HimangY/IndicDISCO-MT/resolve/13f4a8caa71b00859ee4235713aa253c3b2fe193/Indicdisco_MT/Tel/tel_indicdisco-mt_test.csv" `
	--revision 13f4a8caa71b00859ee4235713aa253c3b2fe193 `
	--component Indicdisco_MT `
	--split test `
	--source-language Telugu `
	--target-language English `
	--max-rows 3 `
	--max-bytes 16384
```

Each run writes a metadata-only JSON report under `experiments/evaluation/indicdisco_structure/`. Paragraph text is not copied into the report; accepted records include character counts and explicit source/reference locations. A complete record is accepted only if the source is the exact pinned URL or the local file's Git blob OID and size match official pinned metadata, the schema is exactly `source,target`, both paragraphs are non-empty, provenance points to the pinned revision/split/row, and adapter validation passes. An incomplete CSV tail is reported and never counted as an accepted record.

The actual bounded test-split run inspected row locator `...tel_indicdisco-mt_test.csv#row=0` from the pinned revision. It read 16,384 bytes, accepted 1 complete same-row paragraph pair, and excluded 0 complete rows. A partial trailing record was detected at the byte limit. The report marks document identity, ordering, sentence alignment, same-document context, previous-paragraph context, and within-paragraph sentence context unavailable. Paragraph BLEU/chrF++ are structurally eligible only; sentence and document-context metric eligibility are false. These are eligibility flags, not computed metrics.

For the existing local `data/raw/indicdisco/Indicdisco_MT/Tel/tel_indicdisco-mt_test.csv`, the official pinned tree reports size 58,968 bytes and Git blob OID `cada4e09897d97227d5a45abb91daebc4a569804`. Its local Git blob object ID matched that published OID. The local file therefore passes verification against the pinned file metadata; this comparison does not use a locally invented value as the upstream checksum and does not confer rights to redistribute the dataset.

The validator does not infer IDs from row numbers, use adjacent rows as context, split paragraphs into sentence evaluation records, or access ProAlign/LexiAlign/DiscoAlign. Every report states `rights_status="unresolved"`, `no_inference=true`, and `no_metrics_computed=true`. This validation does not resolve licensing or authorize redistribution.

## Phase 14D: Bounded Paragraph Evaluation

IndicDISCO-MT paragraph evaluation is paragraph-level evaluation and does not establish document-level/context-aware translation performance.

The optional command `scripts/run_indicdisco_paragraph_evaluation.py` requires an explicit source URL or local CSV path, pinned revision, component, split, and Telugu-to-English language codes. Remote input must be the exact official URL for revision `13f4a8caa71b00859ee4235713aa253c3b2fe193`. A local file is eligible only after its size and Git blob OID match metadata fetched from that exact pinned Hub tree; otherwise it fails closed. It reuses the bounded structural reader and the trusted paragraph adapter, samples only accepted same-row paragraph pairs from the bounded prefix, and writes artifacts under `experiments/evaluation/indicdisco_paragraph/`. It never fetches the full dataset or turns row locators into document IDs.

Invoking the command performs real NLLB baseline inference and corpus BLEU/chrF++ using the existing translation service and metric registry. The service receives the complete paragraph under its existing baseline behavior (including its internal sentence segmentation); the resulting text is reassembled and compared at paragraph granularity. No context is supplied, and the result is not a context-aware evaluation. Generation configuration is not overridden: the existing service's `max_new_tokens=256` behavior is retained, and `max_length=200` remains the checkpoint default rather than a new per-call generation option. An individual inference failure is recorded without a fabricated hypothesis; following the existing evaluator's complete-sample policy, corpus metrics are suppressed if any selected baseline hypothesis fails.

Artifacts retain row-level provenance and the complete source, reference, and hypothesis text for each selected record, plus metric/model configuration, device, rights status, and explicit false flags for document identity, context, and sentence alignment. Rights remain `unresolved`; the command does not authorize redistribution. The command is not run automatically by tests or project setup. Tests use synthetic paragraph records, fake translation services, and a no-score metric stub; they do not access the official dataset, invoke NLLB, or calculate metrics.
