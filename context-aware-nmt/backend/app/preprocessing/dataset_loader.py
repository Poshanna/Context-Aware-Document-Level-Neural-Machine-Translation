"""Samanantar parallel-corpus loader + basic preprocessing cleaner.

Supports local paired text, JSONL, TSV, and Parquet layouts (auto-detected):

* **Strategy A — parallel text files**
  ``<data_path>/<pair>/train.<src_lang>``
  ``<data_path>/<pair>/train.<tgt_lang>``
  One sentence per line; records are zipped by line number.

* **Strategy B — JSONL records**
  ``<data_path>/<pair>.jsonl`` or ``<data_path>/train.<pair>.jsonl``
  Each line is an object with keys either ``{"src","tgt"}`` or
  ``{"<src_lang>","<tgt_lang>"}``.

* **Strategy C — TSV records**
    Headered or two-column TSV input.

* **Strategy D — official Samanantar Parquet shards**
    ``<data_path>/te/*.parquet`` with ``src`` and ``tgt`` columns, where the
    official Telugu subset stores English in ``src`` and Telugu in ``tgt``.

This module is 100% FastAPI-free — no imports from ``app.api`` or Web layers.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from itertools import chain, zip_longest
from pathlib import Path
from typing import Iterable, Iterator

log = logging.getLogger(__name__)

_VALID_LANGS = {"en", "te", "hi"}

_WS_RE = re.compile(r"\s+")
_ZWSP_CHARS = ("\u200b", "\u200c", "\u200d", "\ufeff", "\u00a0")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ParallelRecord:
    """One normalized parallel sentence pair (spec FR-2.2 required fields)."""

    id: int
    source_text: str
    target_text: str
    source_language: str
    target_language: str
    data_source: str = "samanantar"


@dataclass(slots=True)
class CleaningReport:
    """Measured counts produced by :func:`clean_records`.

    Every integer is the actual number of rows removed / kept in that stage;
    never a placeholder zero when the stage actually removed rows.
    """

    input_count: int
    null_removed_count: int = 0
    duplicate_removed_count: int = 0
    length_filtered_count: int = 0
    malformed_flagged_count: int = 0
    output_count: int = 0


@dataclass(slots=True)
class LoadingReport:
    """Measured input outcomes before cleaning."""

    input_count: int = 0
    malformed_count: int = 0
    language_filtered_count: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pair_dir(src_lang: str, tgt_lang: str) -> str:
    return f"{src_lang}-{tgt_lang}"


def normalize_text(text: str | None) -> str:
    """Collapse whitespace, strip ZWSP/BOM, Unicode NFC form."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    for zw in _ZWSP_CHARS:
        text = text.replace(zw, " ")
    text = unicodedata.normalize("NFC", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class SamanantarLoader:
    """Load English-Telugu (or other pair) records from a local folder.

    Raises ``FileNotFoundError`` with remediation text when ``data_path``
    does not exist on disk.
    """

    def __init__(
        self,
        data_path: Path | str,
        src_lang: str = "en",
        tgt_lang: str = "te",
    ) -> None:
        if src_lang not in _VALID_LANGS or tgt_lang not in _VALID_LANGS:
            raise ValueError(
                f"Invalid language pair {src_lang!r} <-> {tgt_lang!r}. "
                f"Valid codes: {sorted(_VALID_LANGS)!r}."
            )
        if src_lang == tgt_lang:
            raise ValueError("Source and target language must differ.")

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.pair = _pair_dir(src_lang, tgt_lang)
        self._input_pair = (src_lang, tgt_lang)
        self.data_path = Path(data_path)
        self.loading_report = LoadingReport()
        self._text_files: tuple[Path, Path] | None = None
        self._parquet_files: list[Path] | None = None

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"SAMANANTAR_DATA_PATH={self.data_path} does not exist. "
                "Please download the Samanantar en->te tarball manually and "
                "extract it under the path referenced by SAMANANTAR_DATA_PATH per "
                "docs/dataset.md, then re-run this script. Scripts do NOT "
                "auto-download the dataset."
            )

    # -- Layout detection -------------------------------------------------

    def _detect_layout(self) -> str:
        if self.data_path.is_file() and self.data_path.suffix.lower() == ".parquet":
            self._parquet_files = [self.data_path]
            return "strategy_d"
        if self.data_path.is_file() and self.data_path.suffix.lower() == ".tsv":
            self._set_input_pair_from_name(self.data_path.name)
            return f"strategy_c::{self.data_path}"

        reverse_pair = _pair_dir(self.tgt_lang, self.src_lang)
        for pair_name in dict.fromkeys((self.pair, reverse_pair)):
            pair_dir = self.data_path / pair_name
            src_file = pair_dir / f"train.{self.src_lang}"
            tgt_file = pair_dir / f"train.{self.tgt_lang}"
            if src_file.is_file() and tgt_file.is_file():
                self._text_files = (src_file, tgt_file)
                return "strategy_a"

        src_flat = self.data_path / f"train.{self.src_lang}"
        tgt_flat = self.data_path / f"train.{self.tgt_lang}"
        if src_flat.is_file() and tgt_flat.is_file():
            self._text_files = (src_flat, tgt_flat)
            return "strategy_a_flat"

        jsonl_candidates = [
            (self.data_path / f"{self.pair}.jsonl", (self.src_lang, self.tgt_lang)),
            (self.data_path / f"train.{self.pair}.jsonl", (self.src_lang, self.tgt_lang)),
            (self.data_path / f"{reverse_pair}.jsonl", (self.tgt_lang, self.src_lang)),
            (self.data_path / f"train.{reverse_pair}.jsonl", (self.tgt_lang, self.src_lang)),
            (self.data_path / "train.jsonl", (self.src_lang, self.tgt_lang)),
        ]
        for c, pair in jsonl_candidates:
            if c.is_file():
                self._input_pair = pair
                return f"strategy_b::{c}"

        tsv_candidates = [
            (self.data_path / f"{self.pair}.tsv", (self.src_lang, self.tgt_lang)),
            (self.data_path / f"train.{self.pair}.tsv", (self.src_lang, self.tgt_lang)),
            (self.data_path / f"{reverse_pair}.tsv", (self.tgt_lang, self.src_lang)),
            (self.data_path / f"train.{reverse_pair}.tsv", (self.tgt_lang, self.src_lang)),
            (self.data_path / "train.tsv", (self.src_lang, self.tgt_lang)),
            (self.data_path / self.pair / "train.tsv", (self.src_lang, self.tgt_lang)),
            (self.data_path / reverse_pair / "train.tsv", (self.tgt_lang, self.src_lang)),
        ]
        for candidate, pair in tsv_candidates:
            if candidate.is_file():
                self._input_pair = pair
                return f"strategy_c::{candidate}"

        official_te_dir = self.data_path / "te"
        if official_te_dir.is_dir():
            parquet_files = sorted(official_te_dir.rglob("*.parquet"))
        elif self.data_path.name.lower() == "te":
            parquet_files = sorted(self.data_path.rglob("*.parquet"))
        else:
            parquet_files = sorted(self.data_path.glob("*.parquet"))
        if parquet_files:
            self._parquet_files = parquet_files
            return "strategy_d"

        pair_q = self.data_path / self.pair / f"train.{self.src_lang}"
        flat_q = self.data_path / f"train.{self.src_lang}"
        jsonl_q1 = self.data_path / f"{self.pair}.jsonl"
        jsonl_q2 = self.data_path / "train.jsonl"
        raise ValueError(
            "Could not auto-detect Samanantar storage layout under "
            f"{self.data_path}.\n"
            "  Strategy A (parallel txt) expected locations:\n"
            f"    - {pair_q}\n"
            f"    - {flat_q}\n"
            "  Strategy B (jsonl) expected locations:\n"
            f"    - {jsonl_q1}\n"
            f"    - {jsonl_q2}\n"
            "  Strategy C (TSV) expected locations include:\n"
            f"    - {self.data_path / 'train.tsv'}\n"
            "  Strategy D (Parquet) expected location:\n"
            f"    - {self.data_path / 'te' / '*.parquet'}\n"
            "If the actual tarball format differs, please extend "
            "`_detect_layout` in `dataset_loader.py`."
        )

    def _set_input_pair_from_name(self, name: str) -> None:
        reverse_pair = _pair_dir(self.tgt_lang, self.src_lang)
        if reverse_pair in name.lower():
            self._input_pair = (self.tgt_lang, self.src_lang)
        elif self.pair in name.lower():
            self._input_pair = (self.src_lang, self.tgt_lang)

    # -- Loaders per layout ----------------------------------------------

    def _iter_strategy_a(
        self, src_file: Path, tgt_file: Path
    ) -> Iterator[ParallelRecord]:
        row_id = 0
        with src_file.open("r", encoding="utf-8") as sf, tgt_file.open(
            "r", encoding="utf-8"
        ) as tf:
            for src_line, tgt_line in zip_longest(sf, tf):
                self.loading_report.input_count += 1
                if src_line is None or tgt_line is None:
                    self.loading_report.malformed_count += 1
                    row_id += 1
                    continue
                yield ParallelRecord(
                    id=row_id,
                    source_text=src_line.rstrip("\r\n"),
                    target_text=tgt_line.rstrip("\r\n"),
                    source_language=self.src_lang,
                    target_language=self.tgt_lang,
                    data_source="samanantar",
                )
                row_id += 1

    def _extract_json_pair(self, obj: dict, record_id: int) -> ParallelRecord | None:
        source_lang = obj.get("source_language", obj.get("source_lang", obj.get("src_lang")))
        target_lang = obj.get("target_language", obj.get("target_lang", obj.get("tgt_lang")))
        reverse = self._input_pair == (self.tgt_lang, self.src_lang)
        if source_lang is not None or target_lang is not None:
            if (
                not isinstance(source_lang, str)
                or not isinstance(target_lang, str)
                or not source_lang
                or not target_lang
            ):
                self.loading_report.malformed_count += 1
                return None
            if (source_lang, target_lang) == (self.src_lang, self.tgt_lang):
                pass
            elif (source_lang, target_lang) == (self.tgt_lang, self.src_lang):
                reverse = True
            else:
                self.loading_report.language_filtered_count += 1
                return None

        if self.src_lang in obj and self.tgt_lang in obj:
            src, tgt = obj[self.src_lang], obj[self.tgt_lang]
        elif "src" in obj and "tgt" in obj:
            src, tgt = obj["src"], obj["tgt"]
            if reverse:
                src, tgt = tgt, src
        elif "source_text" in obj and "target_text" in obj:
            src, tgt = obj["source_text"], obj["target_text"]
            if reverse:
                src, tgt = tgt, src
        else:
            self.loading_report.malformed_count += 1
            return None

        if src is not None and not isinstance(src, str):
            self.loading_report.malformed_count += 1
            return None
        if tgt is not None and not isinstance(tgt, str):
            self.loading_report.malformed_count += 1
            return None
        return ParallelRecord(
            id=record_id,
            source_text=src or "",
            target_text=tgt or "",
            source_language=self.src_lang,
            target_language=self.tgt_lang,
            data_source="samanantar",
        )

    def _iter_strategy_b(self, jsonl_file: Path) -> Iterator[ParallelRecord]:
        record_id = 0
        with jsonl_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                self.loading_report.input_count += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    log.warning(
                        "Skipping malformed JSONL record #%d: %s", record_id, exc
                    )
                    self.loading_report.malformed_count += 1
                    record_id += 1
                    continue
                if not isinstance(obj, dict):
                    self.loading_report.malformed_count += 1
                    record_id += 1
                    continue
                record = self._extract_json_pair(obj, record_id)
                if record is not None:
                    yield record
                record_id += 1

    def _iter_strategy_c(self, tsv_file: Path) -> Iterator[ParallelRecord]:
        with tsv_file.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = csv.reader(fh, delimiter="\t")
            first_row = next(rows, None)
            if first_row is None:
                return

            normalized_header = [cell.strip().lower() for cell in first_row]
            header_keys = {
                "src",
                "tgt",
                "src_lang",
                "tgt_lang",
                "source_lang",
                "target_lang",
                "source_text",
                "target_text",
                "source_language",
                "target_language",
                "en",
                "te",
            }
            has_header = bool(set(normalized_header) & header_keys)
            if has_header:
                header = normalized_header
                data_rows = enumerate(rows)
            else:
                header = []
                data_rows = enumerate(chain([first_row], rows))

            for row_id, row in data_rows:
                self.loading_report.input_count += 1
                if has_header:
                    if len(row) != len(header):
                        self.loading_report.malformed_count += 1
                        continue
                    values = dict(zip(header, row))
                    source_lang = values.get(
                        "source_language", values.get("source_lang", values.get("src_lang"))
                    )
                    target_lang = values.get(
                        "target_language", values.get("target_lang", values.get("tgt_lang"))
                    )
                    reverse = self._input_pair == (self.tgt_lang, self.src_lang)
                    if source_lang or target_lang:
                        if (source_lang, target_lang) == (self.tgt_lang, self.src_lang):
                            reverse = True
                        elif (source_lang, target_lang) != (self.src_lang, self.tgt_lang):
                            if not source_lang or not target_lang:
                                self.loading_report.malformed_count += 1
                            else:
                                self.loading_report.language_filtered_count += 1
                            continue
                    if self.src_lang in values and self.tgt_lang in values:
                        src, tgt = values[self.src_lang], values[self.tgt_lang]
                    elif "src" in values and "tgt" in values:
                        src, tgt = values["src"], values["tgt"]
                        if reverse:
                            src, tgt = tgt, src
                    elif "source_text" in values and "target_text" in values:
                        src, tgt = values["source_text"], values["target_text"]
                        if reverse:
                            src, tgt = tgt, src
                    else:
                        self.loading_report.malformed_count += 1
                        continue
                elif len(row) == 2:
                    src, tgt = row
                    if self._input_pair == (self.tgt_lang, self.src_lang):
                        src, tgt = tgt, src
                else:
                    self.loading_report.malformed_count += 1
                    continue
                yield ParallelRecord(
                    id=row_id,
                    source_text=src,
                    target_text=tgt,
                    source_language=self.src_lang,
                    target_language=self.tgt_lang,
                    data_source="samanantar",
                )

    def _iter_strategy_d(self) -> Iterator[ParallelRecord]:
        if (self.src_lang, self.tgt_lang) not in {("en", "te"), ("te", "en")}:
            raise ValueError("Samanantar Parquet input supports only en-te or te-en.")
        if self._parquet_files is None:
            raise RuntimeError("Parquet files were not discovered before loading.")
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError(
                "Parquet input requires PyArrow. Install backend requirements with "
                "`pip install -r backend/requirements.txt`."
            ) from exc

        record_id = 0
        for parquet_path in self._parquet_files:
            parquet_file = parquet.ParquetFile(parquet_path)
            column_names = set(parquet_file.schema_arrow.names)
            missing_columns = {"src", "tgt"} - column_names
            if missing_columns:
                raise ValueError(
                    f"Parquet shard {parquet_path} is missing required column(s): "
                    f"{', '.join(sorted(missing_columns))}."
                )
            columns = ["src", "tgt"]
            if "idx" in column_names:
                columns.append("idx")
            if "data_source" in column_names:
                columns.append("data_source")

            for batch in parquet_file.iter_batches(columns=columns):
                for row in batch.to_pylist():
                    self.loading_report.input_count += 1
                    source_text = row.get("src")
                    target_text = row.get("tgt")
                    source_id = row.get("idx")
                    data_source = row.get("data_source", "samanantar")
                    if (
                        source_text is not None
                        and not isinstance(source_text, str)
                    ) or (target_text is not None and not isinstance(target_text, str)):
                        self.loading_report.malformed_count += 1
                        record_id += 1
                        continue
                    if data_source is None:
                        data_source = "samanantar"
                    elif not isinstance(data_source, str):
                        self.loading_report.malformed_count += 1
                        record_id += 1
                        continue

                    if (self.src_lang, self.tgt_lang) == ("te", "en"):
                        source_text, target_text = target_text, source_text
                    normalized_id = (
                        source_id
                        if isinstance(source_id, int)
                        and not isinstance(source_id, bool)
                        and source_id >= 0
                        else record_id
                    )
                    yield ParallelRecord(
                        id=normalized_id,
                        source_text=source_text or "",
                        target_text=target_text or "",
                        source_language=self.src_lang,
                        target_language=self.tgt_lang,
                        data_source=data_source,
                    )
                    record_id += 1

    # -- Public API -----------------------------------------------------

    def iter_records(self) -> Iterator[ParallelRecord]:
        """Stream records without any preprocessing."""
        self.loading_report = LoadingReport()
        layout = self._detect_layout()
        if layout in {"strategy_a", "strategy_a_flat"}:
            assert self._text_files is not None
            src, tgt = self._text_files
            yield from self._iter_strategy_a(src, tgt)
        elif layout.startswith("strategy_b::"):
            jsonl_path = Path(layout.split("::", 1)[1])
            yield from self._iter_strategy_b(jsonl_path)
        elif layout.startswith("strategy_c::"):
            tsv_path = Path(layout.split("::", 1)[1])
            yield from self._iter_strategy_c(tsv_path)
        elif layout == "strategy_d":
            yield from self._iter_strategy_d()
        else:  # pragma: no cover — _detect_layout raises
            raise AssertionError(layout)

    def load(self) -> list[ParallelRecord]:
        """Materialize :meth:`iter_records` into a list."""
        return list(self.iter_records())

    # -- Normalization ---------------------------------------------------

    def normalize(self, record: ParallelRecord) -> ParallelRecord:
        """Return a shallow copy with whitespace-normalized text; id preserved."""
        return ParallelRecord(
            id=record.id,
            source_text=normalize_text(record.source_text),
            target_text=normalize_text(record.target_text),
            source_language=record.source_language,
            target_language=record.target_language,
            data_source=record.data_source,
        )


# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------


def clean_records(
    records: Iterable[ParallelRecord],
    *,
    min_chars: int = 2,
    max_chars_source: int = 1000,
    max_chars_target: int = 1000,
    malformed_length_ratio: float = 0.95,
) -> tuple[list[ParallelRecord], CleaningReport]:
    """Ordered preprocessing pipeline: normalize → drop empties → dedupe → length filter.

    Returns (cleaned_records_list, CleaningReport).
    """

    seen: "OrderedDict[tuple[str, str, str, str], None]" = OrderedDict()
    report = CleaningReport(input_count=0)

    raw_list = list(records)
    report.input_count = len(raw_list)

    # Stage 1: normalize + drop null/empty source OR target
    after_null: list[ParallelRecord] = []
    for r in raw_list:
        nr = ParallelRecord(
            id=r.id,
            source_text=normalize_text(r.source_text),
            target_text=normalize_text(r.target_text),
            source_language=r.source_language,
            target_language=r.target_language,
            data_source=r.data_source,
        )
        if nr.source_text == "" or nr.target_text == "":
            report.null_removed_count += 1
            continue
        after_null.append(nr)

    # Stage 2: exact (src_lang, tgt_lang, src_text, tgt_text) dedupe — keep first
    after_dedupe: list[ParallelRecord] = []
    for r in after_null:
        key = (r.source_language, r.target_language, r.source_text, r.target_text)
        if key in seen:
            report.duplicate_removed_count += 1
            continue
        seen[key] = None
        after_dedupe.append(r)

    # Stage 3: length filter + malformed flag
    after_len: list[ParallelRecord] = []
    for r in after_dedupe:
        ls = len(r.source_text)
        lt = len(r.target_text)
        if (
            ls < min_chars
            or lt < min_chars
            or ls > max_chars_source
            or lt > max_chars_target
        ):
            report.length_filtered_count += 1
            continue
        longest = max(ls, lt, 1)
        ratio = abs(ls - lt) / longest
        if ratio >= malformed_length_ratio:
            report.malformed_flagged_count += 1
        after_len.append(r)

    report.output_count = len(after_len)
    return after_len, report
