# DocHPLT evaluation adapter

## Why DocHPLT is included

DocHPLT provides a real, document-structured English-to-Telugu dataset with explicit source and target document IDs, sentence IDs, and alignment metadata. This project needs a document-aware evaluation source that preserves actual document boundaries and alignment evidence instead of inferring them from neighboring rows.

The adapter is intentionally strict: it accepts a single verified shard, checks the actual schema and language pair, and only admits rows that satisfy the same-document, 1-to-1, high-confidence alignment contract required for clean evaluation.

## Actual schema used by the audited shard

The audited shard is:

- `data/raw/dochplt/en-te/train-00000-of-00030.parquet`
- dataset revision: `76a591ec3c014d0ac4193a3cbdfc12689616a8d3`
- language pair: `en-te`

The row schema in the real parquet is:

```python
{
    "src_doc_id": "string",
    "tgt_doc_id": "string",
    "lang_pair": "string",
    "src_doc": {
        "ids": ["string", ...],
        "sentences": ["string", ...],
    },
    "tgt_doc": {
        "ids": ["string", ...],
        "sentences": ["string", ...],
    },
    "alignment": [
        {
            "src": ["string"],
            "tgt": ["string"],
            "aligner-score": float,
            "bicleaner-score": float,
            "bifixer-score": float,
        },
        ...
    ],
}
```

This is the exact structure that the adapter validates with PyArrow rather than pandas, which avoids the Windows DLL issue in this environment.

## Actual audit findings from the real shard

The verified real shard audit reported:

- Documents: 12,996
- Language pair: `en-te`
- Source documents with more than one sentence: 12,996
- Target documents with more than one sentence: 12,986
- Total source sentences: 574,607
- Total target sentences: 340,985
- Total alignment groups: 118,724
- Documents with alignment: 12,996
- Documents with no alignment: 0

Alignment breakdown:

- 1-to-1: 87,711
- 1-to-many: 2,660
- many-to-1: 24,594
- many-to-many: 3,759
- invalid references: 0

For 1-to-1 alignments with preceding source context:

- aligner-score >= 0.50: 50,431
- aligner-score >= 0.60: 36,783
- aligner-score >= 0.70: 29,195
- aligner-score >= 0.80: 18,692
- aligner-score >= 0.90: 12,284

The project threshold is fixed at:

- `ALIGNER_SCORE_THRESHOLD = 0.70`

This threshold is not changed automatically and is preserved in the adapter and provenance metadata.

## Why explicit alignment matters

The adapter does not assume that sentence numbers match across source and target documents. A source sentence ID like `2.1` may align to target ID `11.1`. The real alignment data in DocHPLT is explicit and authoritative, so the adapter checks the actual `src` and `tgt` lists and resolves those IDs to the precise source and target sentence text before creating any example.

This prevents false assumptions such as `source 5.1 -> target 5.1` when the alignment metadata says otherwise.

## Why only 1-to-1 alignments are used

The clean evaluation subset deliberately uses only 1-to-1 alignments. This keeps the evaluation contract unambiguous and avoids building a synthetic record from a many-to-many, 1-to-many, or many-to-1 mapping. Non-1-to-1 alignments are counted and rejected, rather than guessed at or silently discarded.

## Why aligner-score >= 0.70 is used

The audit and the project threshold both fix `aligner-score >= 0.70` as the minimal acceptable confidence for a clean evaluation example. This keeps the subset aligned with the validated threshold and preserves the project’s fail-closed rules.

## Why preceding same-document context is required

For each selected example, the source sentence must have at least one preceding source sentence in the same source document. The adapter requires same-document context, never adjacent-row context, and never cross-document context. This is stored in provenance and in the example payload as `context_sentence_ids` and `context_sentences`.

## Why adjacent Parquet rows are not treated as context

The adapter never infers context from row order in the parquet file. It uses only the actual source document’s sentence IDs and the explicit preceding IDs within that document. This avoids making hidden assumptions about row adjacency, document segmentation, or ordering that are not part of the trusted dataset contract.

## Why this is a held-out evaluation subset, not an official test set

The adapter is built from a training split shard and is labelled as a deterministic held-out evaluation subset from the DocHPLT training corpus. It does not claim to be an official DocHPLT test set, and it preserves the explicit provenance fields for the dataset name, config, split, revision, shard path, threshold, context window, and seed.

## Exact pinned metadata

- Dataset: `DocHPLT`
- Dataset config: `en-te`
- Split: `train`
- Revision: `76a591ec3c014d0ac4193a3cbdfc12689616a8d3`
- Source shard: `en-te/train-00000-of-00030.parquet`
- Fixed threshold: `0.70`
- Default context window: `2`

## Deterministic sampling

The adapter supports:

- `max_samples`
- `seed`
- `context_window`
- `aligner_score_threshold`

Sampling is stable and reproducible: the same shard, seed, max-sample count, threshold, and context window yield the same examples. The adapter deliberately records the seed and does not pretend the subset is a benchmark test split.

## Context-window construction

For a source sentence `S5` with a context window of `2`, the context is the immediate preceding source sentences from the same document:

- `S3`
- `S4`

The selected current sentence `S5` is never duplicated in the context, future sentences are never included, and document boundaries are never crossed.

Example payload:

```python
{
    "document_id": "...",
    "src_doc_id": "...",
    "tgt_doc_id": "...",
    "context_sentence_ids": ["3.1", "4.1"],
    "context_sentences": ["...", "..."],
    "source_sentence_id": "5.1",
    "source_sentence": "...",
    "target_sentence_id": "8.1",
    "target_reference": "...",
    "alignment_score": 0.83,
    "lang_pair": "en-te",
}
```

## Limitations

This adapter currently has explicit constraints:

- It uses a single en-te shard.
- It is built from the training split rather than an official DocHPLT test split.
- It accepts only 1-to-1 alignments.
- It remains a prefix/context-conditioned evaluation foundation rather than a native document-level translator.
- It depends on the quality of the DocHPLT alignment metadata itself.

The implementation respects these limitations and does not claim a context-aware improvement without a real baseline and context-conditioned evaluation run.
