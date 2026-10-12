"""Load and validate a locally supplied OPUS TED2020 English-Telugu release.

The loader never downloads data. It reads an alignment XML file and English and
Telugu XML directories or ZIP archives, and returns only valid one-to-one
alignment records for the primary evaluation subset.
"""

from __future__ import annotations

import gzip
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree as ET

_DOCUMENT_ID_PATTERN = re.compile(r"(?:^|/)ted2020-(\d+)\.xml(?:\.gz)?$")
_CARDINALITIES = ("one_to_one", "one_to_many", "many_to_one", "many_to_many")


@dataclass
class TED2020LoadResult:
    """Evaluation records and corpus-text-free validation statistics."""

    records: list[dict[str, Any]]
    report: dict[str, Any]


def extract_document_id(document_reference: str) -> str:
    """Extract the release-local talk ID from an OPUS TED2020 XML path."""
    normalized = document_reference.replace("\\", "/")
    match = _DOCUMENT_ID_PATTERN.search(normalized)
    if match is None:
        raise ValueError(f"Unrecognized TED2020 document reference: {document_reference!r}")
    return match.group(1)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _read_xml_bytes(path: Path) -> bytes:
    payload = path.read_bytes()
    if payload.startswith(b"\x1f\x8b"):
        return gzip.decompress(payload)
    return payload


def _parse_xml(payload: bytes, label: str) -> ET.Element:
    try:
        return ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ValueError(f"Malformed XML in {label}: {exc}") from exc


class _DocumentSource:
    """Read XML members from an extracted directory or ZIP, without extracting."""

    def __init__(self, source: Path | str, language: str) -> None:
        self.path = Path(source)
        self.language = language
        if not self.path.exists():
            raise FileNotFoundError(f"TED2020 {language} input does not exist: {self.path}")
        self._zip: zipfile.ZipFile | None = None
        self._files: list[str] = []
        if self.path.is_dir():
            self._files = [p.relative_to(self.path).as_posix() for p in self.path.rglob("*.xml*")]
        elif zipfile.is_zipfile(self.path):
            self._zip = zipfile.ZipFile(self.path)
            self._files = self._zip.namelist()
        elif self.path.is_file():
            self._files = [self.path.name]
        else:
            raise ValueError(f"TED2020 {language} input must be a directory, XML file, or ZIP archive")

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def _find_member(self, document_reference: str) -> str | None:
        reference = document_reference.replace("\\", "/").lstrip("/")
        name = PurePosixPath(reference).name
        candidates = [name]
        if name.endswith(".gz"):
            candidates.insert(0, name[:-3])

        matches: set[str] = set()
        for member in self._files:
            normalized = member.replace("\\", "/").lstrip("/")
            for candidate in candidates:
                if normalized == reference or normalized.endswith("/" + candidate):
                    matches.add(member)
        if len(matches) == 1:
            return next(iter(matches))
        return None

    def read_document(self, document_reference: str) -> bytes | None:
        member = self._find_member(document_reference)
        if member is None:
            return None
        if self._zip is not None:
            payload = self._zip.read(member)
        elif self.path.is_dir():
            payload = (self.path / Path(member)).read_bytes()
        else:
            payload = self.path.read_bytes()
        if payload.startswith(b"\x1f\x8b"):
            payload = gzip.decompress(payload)
        return payload


def _extract_sentences(
    root: ET.Element,
) -> tuple[list[tuple[str, str]], list[str], int]:
    sentences: list[tuple[str, str]] = []
    duplicates: list[str] = []
    missing_ids = 0
    seen: set[str] = set()
    for element in root.iter():
        if _local_name(element.tag) != "s":
            continue
        sentence_id = element.get("id")
        if not sentence_id:
            missing_ids += 1
            continue
        if sentence_id in seen:
            duplicates.append(sentence_id)
            continue
        seen.add(sentence_id)
        words = [child for child in element.iter() if _local_name(child.tag) == "w"]
        if words:
            text = " ".join(
                " ".join("".join(word.itertext()).split())
                for word in words
            ).strip()
        else:
            text = " ".join("".join(element.itertext()).split())
        sentences.append((sentence_id, text))
    return sentences, duplicates, missing_ids


def _parse_targets(value: str | None) -> tuple[list[str], list[str]] | None:
    if value is None or value.count(";") != 1:
        return None
    source_part, target_part = value.split(";", 1)
    source_ids = source_part.split()
    target_ids = target_part.split()
    if not source_ids or not target_ids:
        return None
    return source_ids, target_ids


def _alignment_type(source_ids: list[str], target_ids: list[str]) -> str:
    if len(source_ids) == 1 and len(target_ids) == 1:
        return "one_to_one"
    if len(source_ids) == 1:
        return "one_to_many"
    if len(target_ids) == 1:
        return "many_to_one"
    return "many_to_many"


def _new_report(context_window: int) -> dict[str, Any]:
    return {
        "corpus": "TED2020",
        "release": "v1",
        "source_language": "en",
        "target_language": "te",
        "context_window": context_window,
        "documents": 0,
        "links": 0,
        "alignment_counts": {name: 0 for name in _CARDINALITIES},
        "excluded_non_one_to_one": 0,
        "missing_source_references": 0,
        "missing_target_references": 0,
        "malformed_links": 0,
        "sentence_count_mismatches": 0,
        "duplicate_sentence_ids": 0,
        "missing_sentence_ids": 0,
        "unresolved_document_references": 0,
        "unresolved_document_reference_details": [],
        "malformed_documents": 0,
        "invalid_context_links": 0,
        "valid": True,
    }


def load_ted2020(
    alignment_path: Path | str,
    english_source: Path | str,
    telugu_source: Path | str,
    *,
    context_window: int = 0,
) -> TED2020LoadResult:
    """Read local release inputs and return one-to-one English-Telugu records.

    ``english_source`` and ``telugu_source`` may each be an extracted directory,
    a ZIP archive from the OPUS release, or a single document XML file. The
    alignment path may be plain XML or gzip-compressed XML.
    """
    if (
        not isinstance(context_window, int)
        or isinstance(context_window, bool)
        or context_window < 0
    ):
        raise ValueError("context_window must be a non-negative integer")
    alignment_file = Path(alignment_path)
    if not alignment_file.is_file():
        raise FileNotFoundError(f"TED2020 alignment XML does not exist: {alignment_file}")

    report = _new_report(context_window)
    en_source = _DocumentSource(english_source, "English")
    te_source = _DocumentSource(telugu_source, "Telugu")
    document_cache: dict[tuple[str, str], tuple[list[tuple[str, str]], set[str]] | None] = {}
    grouped_links: dict[str, list[tuple[list[str], list[str], str | None]]] = {}
    group_references: dict[str, tuple[str, str]] = {}
    duplicate_total = 0
    missing_id_total = 0

    def get_document(
        language: str, reference: str
    ) -> tuple[list[tuple[str, str]], set[str]] | None:
        nonlocal duplicate_total, missing_id_total
        key = (language, reference)
        if key in document_cache:
            return document_cache[key]
        source = en_source if language == "en" else te_source
        payload = source.read_document(reference)
        if payload is None:
            report["unresolved_document_references"] += 1
            report["unresolved_document_reference_details"].append(reference)
            document_cache[key] = None
            return None
        try:
            root = _parse_xml(payload, reference)
            sentences, duplicates, missing_ids = _extract_sentences(root)
        except (ValueError, OSError, EOFError):
            report["malformed_documents"] += 1
            document_cache[key] = None
            return None
        duplicate_total += len(duplicates)
        missing_id_total += missing_ids
        document_cache[key] = (sentences, set(duplicates))
        return document_cache[key]

    try:
        root = _parse_xml(_read_xml_bytes(alignment_file), str(alignment_file))
        if _local_name(root.tag) != "cesAlign":
            raise ValueError("Expected cesAlign root in bilingual alignment XML")

        for group in root.iter():
            if _local_name(group.tag) != "linkGrp":
                continue
            from_doc = group.get("fromDoc")
            to_doc = group.get("toDoc")
            try:
                source_doc_id = extract_document_id(from_doc or "")
                target_doc_id = extract_document_id(to_doc or "")
            except ValueError:
                report["unresolved_document_references"] += 1
                report["unresolved_document_reference_details"].extend(
                    ref for ref in (from_doc, to_doc) if ref
                )
                report["malformed_links"] += sum(
                    1 for child in group if _local_name(child.tag) == "link"
                )
                continue
            if (
                not (from_doc or "").replace("\\", "/").startswith("en/")
                or not (to_doc or "").replace("\\", "/").startswith("te/")
                or source_doc_id != target_doc_id
            ):
                report["malformed_links"] += sum(
                    1 for child in group if _local_name(child.tag) == "link"
                )
                continue

            document_id = source_doc_id
            group_references[document_id] = (from_doc or "", to_doc or "")
            grouped_links.setdefault(document_id, [])
            for link in group:
                if _local_name(link.tag) != "link":
                    continue
                report["links"] += 1
                targets = _parse_targets(link.get("xtargets"))
                if targets is None:
                    report["malformed_links"] += 1
                    continue
                source_ids, target_ids = targets
                kind = _alignment_type(source_ids, target_ids)
                report["alignment_counts"][kind] += 1
                link_id = link.get("id")
                grouped_links[document_id].append((source_ids, target_ids, link_id))

        report["documents"] = len(group_references)
        records: list[dict[str, Any]] = []
        for document_id in sorted(group_references):
            from_doc, to_doc = group_references[document_id]
            source_document = get_document("en", from_doc)
            target_document = get_document("te", to_doc)
            if source_document is None or target_document is None:
                continue
            source_sentences, source_duplicates = source_document
            target_sentences, target_duplicates = target_document
            source_map = dict(source_sentences)
            target_map = dict(target_sentences)
            source_order = {sentence_id: index for index, (sentence_id, _) in enumerate(source_sentences)}
            if len(source_sentences) != len(target_sentences):
                report["sentence_count_mismatches"] += 1
            report["duplicate_sentence_ids"] += len(source_duplicates) + len(target_duplicates)

            candidates: list[tuple[int, str, str, str, str]] = []
            used_source_ids: set[str] = set()
            used_target_ids: set[str] = set()
            for source_ids, target_ids, link_id in grouped_links[document_id]:
                if len(source_ids) != 1 or len(target_ids) != 1:
                    report["excluded_non_one_to_one"] += 1
                    continue
                source_id, target_id = source_ids[0], target_ids[0]
                has_missing_reference = False
                if source_id not in source_map:
                    report["missing_source_references"] += 1
                    has_missing_reference = True
                if target_id not in target_map:
                    report["missing_target_references"] += 1
                    has_missing_reference = True
                if has_missing_reference:
                    continue
                if source_id in source_duplicates or target_id in target_duplicates:
                    report["malformed_links"] += 1
                    continue
                if source_id in used_source_ids or target_id in used_target_ids:
                    report["malformed_links"] += 1
                    continue
                used_source_ids.add(source_id)
                used_target_ids.add(target_id)
                candidates.append(
                    (source_order[source_id], source_id, target_id, source_map[source_id], target_map[target_id])
                )

            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            for order_index, source_id, target_id, source_text, target_text in candidates:
                prior_ids = [sentence_id for sentence_id, _ in source_sentences[:order_index]]
                if context_window:
                    prior_ids = prior_ids[-context_window:]
                else:
                    prior_ids = []
                if any(source_order.get(context_id, order_index) >= order_index for context_id in prior_ids):
                    report["invalid_context_links"] += 1
                    prior_ids = [
                        context_id for context_id in prior_ids
                        if source_order.get(context_id, order_index) < order_index
                    ]
                if source_id in prior_ids:
                    report["invalid_context_links"] += 1
                    prior_ids = [context_id for context_id in prior_ids if context_id != source_id]
                records.append(
                    {
                        "document_id": document_id,
                        "sentence_id": source_id,
                        "source_sentence": source_text,
                        "reference_translation": target_text,
                        "previous_context_sentence_ids": prior_ids,
                        "alignment": {
                            "source_span": [source_id],
                            "reference_span": [target_id],
                            "method": "xml-alignment",
                        },
                        "phenomena": [],
                        "provenance_id": f"TED2020-v1:{document_id}:{source_id}:{target_id}",
                    }
                )

        report["duplicate_sentence_ids"] = duplicate_total
        report["missing_sentence_ids"] = missing_id_total
        report["included_one_to_one_records"] = len(records)
        report["valid"] = not any(
            report[key]
            for key in (
                "missing_source_references",
                "missing_target_references",
                "malformed_links",
                "duplicate_sentence_ids",
                "missing_sentence_ids",
                "unresolved_document_references",
                "malformed_documents",
                "invalid_context_links",
            )
        )
        return TED2020LoadResult(records=records, report=report)
    finally:
        en_source.close()
        te_source.close()


def write_report(report: dict[str, Any], output: Path | str | None = None) -> None:
    """Print or write JSON statistics; the report contains no corpus text."""
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if output is None:
        print(serialized)
        return
    Path(output).write_text(serialized + "\n", encoding="utf-8")