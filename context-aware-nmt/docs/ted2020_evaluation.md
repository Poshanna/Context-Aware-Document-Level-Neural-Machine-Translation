# TED2020 Document-Level Evaluation Input

## Purpose and Scope

The TED2020 loader is a separate, local-only reader for an English-Telugu OPUS
TED2020 v1 XML release. It does not download data, alter Samanantar, run model
inference, or change the Phase 7 evaluation runner. `load_ted2020()` returns
one-to-one aligned records and a validation report. The command-line validator
prints or writes statistics only; it does not write corpus text or records.

TED2020 reuse is not authorized by this tool. The OPUS TED2020 release points
to TED's usage policy, which restricts use. Resolve applicable TED and TED
Translators terms and obtain rights-holder/legal clarification before using,
transforming, or redistributing text. The loader does not make a rights
determination.

## Local Inputs

Obtain and unpack an OPUS TED2020 v1 release manually under the applicable
permissions. The loader accepts:

- The English-Telugu `cesAlign` XML file (`en-te.xml` or gzip-compressed XML).
- An English input as an extracted directory, ZIP archive, or single XML file.
- A Telugu input in the same forms.

For archives, language documents are read in place from ZIP members; they are
not extracted by the loader. The OPUS alignment file refers to documents using
paths like `en/ted2020-1183.xml.gz`, while release archives may contain plain
`TED2020/xml/en/ted2020-1183.xml` members. Resolution matches the referenced
filename against archive/directory members and reports unresolved or ambiguous
references.

Example with local paths:

```powershell
python scripts/validate_ted2020.py `
  --alignment C:\data\TED2020\xml\en-te.xml.gz `
  --english C:\data\TED2020\xml\en.zip `
  --telugu C:\data\TED2020\xml\te.zip `
  --context-window 3
```

The script does not fetch missing inputs. Optional `--report-output report.json`
writes only JSON statistics, not corpus text.

## XML Parsing and Alignment Policy

The bilingual alignment XML is expected to have a `cesAlign` root. Each
`linkGrp` must identify an English `fromDoc`, Telugu `toDoc`, and sentence-level
`targType`. The matching language XML contains ordered `s` sentence elements
with document-local IDs. A link's `xtargets` value contains source IDs and
reference IDs separated by `;`.

Only links with exactly one source ID and one reference ID are emitted as the
primary evaluation records. One-to-many, many-to-one, and many-to-many links
are counted as excluded; the loader does not duplicate, concatenate, or guess
translation text to force them into a single-sentence pair. Missing sentence
references, duplicate sentence IDs, malformed links, unresolved document
references, and sentence-count mismatches appear in the report.

Sentence IDs are document-local. The `document_id` is derived from the verified
`ted2020-<id>` filename convention. `sentence_id` is the English source `s`
element ID. `alignment.source_span` and `alignment.reference_span` retain both
IDs, because English and Telugu IDs need not match. The sentence text is
assembled from nested XML text nodes with whitespace normalized between tokens.

## Context Construction

For each emitted one-to-one record, `previous_context_sentence_ids` contains up
to the configured number of immediately preceding English sentence IDs from
the same document, in XML order. The current sentence and future sentences are
never included. Context is generated from the document's ordered English
sentence list, not from bilingual link order. Window `0` emits no context.

The loader does not invent phenomenon labels: `phenomena` is empty. Human
annotation would be a separate documented step.

## Provenance and Validation

The record's `provenance_id` is deterministic and includes corpus/release,
document ID, and both alignment IDs. Retain a separate manifest with input
file paths, release/version, retrieval date, file checksums, original
`fromDoc`/`toDoc` paths, alignment-file version, context-window setting, and
the validation report. Keep that manifest free of copied TED text unless its
terms permit otherwise.

Validation checks document path conventions and language pairing, sentence-ID
uniqueness, link syntax/cardinality, source/target reference existence,
document resolution, sentence counts, and generated context ordering. A
sentence-count mismatch is reported but is not by itself a parsing failure,
because valid segmentations may differ. `valid` is false when malformed or
unresolved references are found. Review the report and excluded alignment
counts before using any returned records.

## Rights and Corpus Hygiene

OPUS's TED2020 release says to respect the TED Talks Usage Policy and that the
dataset carries its original-source license. Public accessibility is not a
grant of separate reuse permission. In particular, the loader cannot determine
whether a transformed or redistributed evaluation subset is permitted. Obtain
clarification before creating or publishing such a subset.

Do not commit TED2020 source/reference text, derived evaluation records, or
generated predictions. Synthetic test fixtures contain only artificial text.