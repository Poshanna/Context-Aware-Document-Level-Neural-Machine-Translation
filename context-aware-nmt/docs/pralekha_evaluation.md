# PRALEKHA evaluation foundation (Phase 13A)

This document describes the conservative Phase 13A foundation for document/passage-aware evaluation based on the PRALEKHA dataset audit. This is a design and adapter phase only; it does not load the full corpus, run a real model, or claim a scored result.

## Goal

Create a trusted, fail-closed evaluation contract for PRALEKHA that preserves only the evidence the dataset actually exposes:

- dataset name and revision metadata when available,
- source/reference document pairing when explicit,
- same-document causal context only,
- no fabricated sentence alignment,
- no claim of sentence-level metric support without verified sentence alignment.

## Conservative rules

1. We do not assume sentence-level alignment exists in the public PRALEKHA rows.
2. We only permit context that is same-document, causal, and earlier-only.
3. We require explicit document/passage identity and provenance before any context-aware evaluation record is accepted.
4. We reject duplicate unit IDs, duplicate orderings within a document, cross-document context, and future leakage.
5. Sentence-level BLEU/chrF++ stays unavailable unless a true sentence-alignment contract is later introduced.

## Provenance contract

The adapter in `backend/app/evaluation/pralekha_adapter.py` validates the following metadata before accepting a PRALEKHA record:

- dataset name must be `AI4Bharat PRALEKHA`,
- split must match the loaded split,
- dataset revision must be present if pinned,
- record ID must be present,
- source location and alignment location must be present,
- document/passage identity must be preserved if available,
- provenance must be verified dataset provenance, not unverified or synthetic.

## Unit semantics

The design distinguishes between:

- `PraLekhaDocumentPair`: a source-target document pair with explicit dataset provenance and pair id,
- `PraLekhaDocumentUnit`: an ordered unit within a document, for same-document causal context only,
- causal context: preceding units from the same document only, ordered by `unit_order`.

This intentionally prevents a document-aware evaluation from silently crossing dataset or document boundaries.

## Metric gate

The phase does not compute metrics. It only returns a conservative eligibility gate.

- sentence-level metrics require sentence alignment evidence and available reference text,
- document-level metrics require document pairing and available reference text,
- cross-document or unverifiable provenance automatically blocks eligible context-aware scoring.

## Phase 13A status

This phase is intentionally isolated and non-evaluative:

- no real PRALEKHA corpus download,
- no model execution,
- no produced translation scores,
- no claim of full document-context quality beyond the synthetic validation contract.

The next step, after this foundation, is to connect real PRALEKHA records to the same fail-closed provenance and context rules only when the dataset exposes the required alignment metadata.
