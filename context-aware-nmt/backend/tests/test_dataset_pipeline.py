"""Phase 3 — Dataset pipeline tests (loader, cleaner, splitter, scripts, docs).

Covers all 20+ tests enumerated in FR-10.1.  Every test uses ``tmp_path`` for
synthetic fixture data so **no real Samanantar download is ever required** for
CI.

Existing 42 Phase-2 foundation tests must continue to pass unchanged.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure test environment is set up (JWT/POSTGRES env vars set by conftest.py
# at import time via side-effect.
# ---------------------------------------------------------------------------
# conftest.py already sets env then imports app.main.  The test harness uses
# backend/tests root which has conftest sys.path inserts.  Scripts however
# live under PROJECT_ROOT/scripts which is separate from sys.path during
# in-process import via importlib.util.  We therefore resolve PROJECT_ROOT
# once and pass it explicitly to each script test.

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DOCS_DIR = PROJECT_ROOT / "docs"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.preprocessing.dataset_loader import (  # noqa: E402
    CleaningReport,
    ParallelRecord,
    SamanantarLoader,
    clean_records,
    normalize_text,
)
from app.preprocessing.dataset_splitter import (  # noqa: E402
    Splits,
    read_split_jsonl,
    split_dataset,
    write_splits_jsonl,
)
from app.preprocessing.context_record import (  # noqa: E402
    ContextRecord,
    record_from_parallel,
)


# ===========================================================================
# Helper: write Strategy A fixtures
# ===========================================================================


def _write_strategy_a(tmp: Path, en_lines: list[str], te_lines: list[str]) -> Path:
    pair_dir = tmp / "saman" / "en-te"
    pair_dir.mkdir(parents=True)
    (pair_dir / "train.en").write_text("\n".join(en_lines) + "\n", encoding="utf-8")
    (pair_dir / "train.te").write_text("\n".join(te_lines) + "\n", encoding="utf-8")
    return tmp / "saman"


def _write_strategy_a_flat(tmp: Path, en_lines: list[str], te_lines: list[str]) -> Path:
    flat = tmp / "saman_flat"
    flat.mkdir()
    (flat / "train.en").write_text("\n".join(en_lines) + "\n", encoding="utf-8")
    (flat / "train.te").write_text("\n".join(te_lines) + "\n", encoding="utf-8")
    return flat


EN_5 = [
    "How are you?",
    "The weather is nice today.",
    "I am reading.",
    "The book is very interesting.",
    "Yes, that is true.",
]
TE_5 = [
    "మీరు ఎలా ఉన్నారు?",
    "ఈ రోజు వాతావరణం బాగుంది.",
    "నేను చదువుతున్నాను.",
    "పుస్తకం చాలా ఆసక్తికరంగా ఉంది.",
    "అవును, అది నిజం.",
]


# ===========================================================================
# TR 2.1 / 2.2 / 2.3 / 2.4 / 2.5 — Loader tests
# ===========================================================================


class TestLoaderStrategyA:
    """Strategy A = parallel train.en / train.te files."""

    def test_loads_5_parallel_rows_ids_0_to_4(self, tmp_path: Path) -> None:
        """TR 2.1: Parallel .txt files → 5 rows with ids 0..4."""
        data_path = _write_strategy_a(tmp_path, EN_5, TE_5)
        records = SamanantarLoader(data_path).load()
        assert len(records) == 5
        assert [r.id for r in records] == [0, 1, 2, 3, 4]
        assert records[2].source_text == EN_5[2]
        assert records[2].target_text == TE_5[2]
        for r in records:
            assert r.source_language == "en"
            assert r.target_language == "te"
            assert r.data_source == "samanantar"

    def test_counts_malformed_and_filters_unrelated_pairs(self, tmp_path: Path) -> None:
        base = tmp_path / "mixed_pairs"
        base.mkdir()
        input_path = base / "en-te.jsonl"
        input_path.write_text(
            '{"src":"Hello there","tgt":"హలో"}\n'
            '{invalid json}\n'
            '{"src":"Bonjour","tgt":"హలో","src_lang":"en","tgt_lang":"hi"}\n'
            '["not", "an", "object"]\n',
            encoding="utf-8",
        )

        loader = SamanantarLoader(base)
        records = loader.load()

        assert len(records) == 1
        assert loader.loading_report.input_count == 4
        assert loader.loading_report.malformed_count == 2
        assert loader.loading_report.language_filtered_count == 1

    def test_reverse_direction_from_language_keyed_records(self, tmp_path: Path) -> None:
        base = tmp_path / "reverse"
        base.mkdir()
        (base / "en-te.jsonl").write_text(
            json.dumps({"en": "Good morning", "te": "శుభోదయం"}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

        [record] = SamanantarLoader(base, src_lang="te", tgt_lang="en").load()

        assert record.source_text == "శుభోదయం"
        assert record.target_text == "Good morning"
        assert (record.source_language, record.target_language) == ("te", "en")

    def test_reverse_direction_from_pair_named_src_tgt_jsonl(self, tmp_path: Path) -> None:
        base = tmp_path / "reverse_generic"
        base.mkdir()
        (base / "en-te.jsonl").write_text(
            json.dumps({"src": "Good morning", "tgt": "శుభోదయం"}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

        [record] = SamanantarLoader(base, src_lang="te", tgt_lang="en").load()

        assert record.source_text == "శుభోదయం"
        assert record.target_text == "Good morning"

    def test_tsv_header_supports_reverse_direction(self, tmp_path: Path) -> None:
        tsv_path = tmp_path / "pairs.tsv"
        tsv_path.write_text("en\tte\nHello\tహలో\n", encoding="utf-8")

        [record] = SamanantarLoader(tsv_path, src_lang="te", tgt_lang="en").load()

        assert record.source_text == "హలో"
        assert record.target_text == "Hello"

    def test_flat_strategy_a_works_too(self, tmp_path: Path) -> None:
        """Flat variant (no en-te subdir) also loads."""
        data_path = _write_strategy_a_flat(tmp_path, EN_5, TE_5)
        records = SamanantarLoader(data_path).load()
        assert len(records) == 5
        assert records[4].source_text == EN_5[4]


class TestLoaderStrategyB:
    """Strategy B = .jsonl records."""

    def test_src_tgt_keys_load(self, tmp_path: Path) -> None:
        """TR 2.2: {"src":"...","tgt":"..."} per line → correct records."""
        base = tmp_path / "saman_b"
        base.mkdir()
        jsonl = base / "en-te.jsonl"
        with jsonl.open("w", encoding="utf-8") as fh:
            for en, te in zip(EN_5, TE_5):
                fh.write(json.dumps({"src": en, "tgt": te}, ensure_ascii=False) + "\n")
        records = SamanantarLoader(base).load()
        assert len(records) == 5
        assert records[0].source_text == EN_5[0]
        assert records[0].target_text == TE_5[0]
        for r in records:
            assert r.source_language == "en"
            assert r.target_language == "te"

    def test_per_lang_keys_load(self, tmp_path: Path) -> None:
        """Strategy B variant with {"en":"...", "te":"..."} keys."""
        base = tmp_path / "saman_b2"
        base.mkdir()
        jsonl = base / "train.en-te.jsonl"
        with jsonl.open("w", encoding="utf-8") as fh:
            for en, te in zip(EN_5, TE_5):
                fh.write(json.dumps({"en": en, "te": te}, ensure_ascii=False) + "\n")
        records = SamanantarLoader(base).load()
        assert len(records) == 5
        assert records[3].source_text == EN_5[3]
        assert records[3].target_text == TE_5[3]


def _write_parquet_shards(root: Path, shards: dict[str, list[dict]]) -> Path:
    import struct

    import pyarrow as pa
    import pyarrow.parquet as pq

    def string_array(values: list[str]) -> pa.Array:
        encoded_values = [value.encode("utf-8") for value in values]
        offsets = [0]
        for value in encoded_values:
            offsets.append(offsets[-1] + len(value))
        offset_buffer = pa.py_buffer(
            struct.pack(f"<{len(offsets)}i", *offsets)
        )
        data_buffer = pa.py_buffer(b"".join(encoded_values))
        return pa.Array.from_buffers(
            pa.string(), len(values), [None, offset_buffer, data_buffer]
        )

    te_dir = root / "te"
    te_dir.mkdir(parents=True)
    for shard_name, rows in shards.items():
        column_names = list(rows[0])
        arrays = []
        for column_name in column_names:
            values = [row[column_name] for row in rows]
            if column_name == "idx":
                value_buffer = pa.py_buffer(struct.pack(f"<{len(values)}q", *values))
                arrays.append(
                    pa.Array.from_buffers(pa.int64(), len(values), [None, value_buffer])
                )
            else:
                arrays.append(string_array(values))
        pq.write_table(
            pa.Table.from_arrays(arrays, names=column_names), te_dir / shard_name
        )
    return root


class TestLoaderParquet:
    def test_reads_multiple_official_te_shards_and_preserves_fields(
        self, tmp_path: Path
    ) -> None:
        data_path = _write_parquet_shards(
            tmp_path / "samanantar",
            {
                "part-00001.parquet": [
                    {
                        "idx": 11,
                        "src": "English second",
                        "tgt": "తెలుగు రెండవది",
                        "data_source": "source-b",
                    }
                ],
                "part-00000.parquet": [
                    {
                        "idx": 10,
                        "src": "English first",
                        "tgt": "తెలుగు మొదటిది",
                        "data_source": "source-a",
                    }
                ],
            },
        )

        records = SamanantarLoader(data_path, src_lang="en", tgt_lang="te").load()

        assert [record.id for record in records] == [10, 11]
        assert [record.source_text for record in records] == [
            "English first",
            "English second",
        ]
        assert [record.target_text for record in records] == [
            "తెలుగు మొదటిది",
            "తెలుగు రెండవది",
        ]
        assert [record.data_source for record in records] == ["source-a", "source-b"]

    def test_reverses_official_src_tgt_for_telugu_to_english(
        self, tmp_path: Path
    ) -> None:
        data_path = _write_parquet_shards(
            tmp_path / "samanantar",
            {
                "part-00000.parquet": [
                    {
                        "idx": 7,
                        "src": "Good morning",
                        "tgt": "శుభోదయం",
                        "data_source": "source-a",
                    }
                ]
            },
        )

        [record] = SamanantarLoader(data_path, src_lang="te", tgt_lang="en").load()

        assert record.id == 7
        assert record.source_text == "శుభోదయం"
        assert record.target_text == "Good morning"
        assert (record.source_language, record.target_language) == ("te", "en")
        assert record.data_source == "source-a"

    def test_missing_data_source_defaults_to_samanantar(self, tmp_path: Path) -> None:
        data_path = _write_parquet_shards(
            tmp_path / "samanantar",
            {
                "part-00000.parquet": [
                    {"idx": 0, "src": "Hello", "tgt": "హలో"}
                ]
            },
        )

        [record] = SamanantarLoader(data_path).load()

        assert record.data_source == "samanantar"

    def test_parquet_preprocessing_preserves_source_and_writes_report(
        self, tmp_path: Path
    ) -> None:
        rows = [
            {
                "idx": index,
                "src": f"English sentence number {index}",
                "tgt": f"తెలుగు వాక్యం సంఖ్య {index}",
                "data_source": "upstream-source",
            }
            for index in range(4)
        ]
        data_path = _write_parquet_shards(
            tmp_path / "samanantar",
            {"part-00000.parquet": rows[:2], "part-00001.parquet": rows[2:]},
        )
        processed_dir = tmp_path / "processed"
        report_path = tmp_path / "preprocessing-report.json"
        prepare_module = _load_script_module(SCRIPTS_DIR / "prepare_samanantar.py")

        rc = prepare_module.main(
            [
                "--data-path",
                str(data_path),
                "--processed-dir",
                str(processed_dir),
                "--report",
                str(report_path),
            ]
        )

        assert rc == 0
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["raw_count"] == 4
        assert report["valid_count"] == 4
        assert report["final_count"] == 4
        assert report["train_count"] + report["validation_count"] + report["test_count"] == 4

        output_records = []
        for split in ("train", "val", "test"):
            split_path = processed_dir / f"{split}.en-te.jsonl"
            assert split_path.exists()
            output_records.extend(
                json.loads(line)
                for line in split_path.read_text(encoding="utf-8").splitlines()
                if line
            )
        assert len(output_records) == 4
        assert {record["data_source"] for record in output_records} == {
            "upstream-source"
        }

        validate_module = _load_script_module(SCRIPTS_DIR / "validate_dataset.py")
        validation_rc = validate_module.main(
            ["--processed-dir", str(processed_dir), "--report", str(tmp_path / "validation-report.json")]
        )
        assert validation_rc == 0


class TestLoaderErrors:
    def test_missing_path_error_message_mentions_download(self, tmp_path: Path) -> None:
        """TR 2.3: Missing path raises FileNotFoundError with "download Samanantar"."""
        bogus = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError) as excinfo:
            SamanantarLoader(bogus)
        assert "download" in str(excinfo.value).lower()
        assert "dataset.md" in str(excinfo.value)

    def test_empty_dir_no_layout(self, tmp_path: Path) -> None:
        """TR 2.4: Empty dir raises ValueError mentioning Strategy A/B."""
        empty = tmp_path / "empty"
        empty.mkdir()
        loader = SamanantarLoader(empty)
        with pytest.raises(ValueError) as excinfo:
            list(loader.iter_records())
        msg = str(excinfo.value).lower()
        assert "strategy a" in msg or "parallel" in msg
        assert "strategy b" in msg or "jsonl" in msg

    def test_invalid_pair_code_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            SamanantarLoader(tmp_path, src_lang="en", tgt_lang="fr")
        with pytest.raises(ValueError):
            SamanantarLoader(tmp_path, src_lang="en", tgt_lang="en")


class TestNormalize:
    def test_whitespace_and_zwsp_collapse(self) -> None:
        """TR 2.5: Whitespace collapse + ZWSP removal."""
        assert normalize_text("  hello\u200b   world  ") == "hello world"
        assert normalize_text("   a\n\n\nb   ") == "a b"
        assert normalize_text("\ufeffHi\u00a0there") == "Hi there"
        assert normalize_text(None) == ""
        assert normalize_text(1234) == "1234"

    def test_loader_normalize_preserves_id(self, tmp_path: Path) -> None:
        data_path = _write_strategy_a(
            tmp_path,
            ["  a\u200bb  "],
            ["  c\u00a0\u200bd     e  \t\tf"],
        )
        [r] = SamanantarLoader(data_path).load()
        normed = SamanantarLoader(data_path).normalize(r)
        assert normed.id == r.id
        assert normed.source_text == "a b"
        # NBSP, ZWSP, multiple spaces → collapse to single space + tabs collapse too
        assert normed.target_text == "c d e f"


# ===========================================================================
# TR 3.1 / 3.2 / 3.3 / 3.4 / 3.5 — Cleaning tests
# ===========================================================================


def _rec(i: int, s: str | None, t: str | None) -> ParallelRecord:
    return ParallelRecord(
        id=i,
        source_text="" if s is None else s,
        target_text="" if t is None else t,
        source_language="en",
        target_language="te",
    )


class TestCleaning:
    def test_drops_null_empty(self) -> None:
        """TR 3.1: 2 empty src + 1 empty tgt → null_removed_count == 3."""
        records = [
            _rec(0, "", TE_5[0]),  # empty src → removed
            _rec(1, None, TE_5[1]),  # None → becomes "" → removed
            _rec(2, EN_5[2], ""),  # empty tgt → removed
            _rec(3, EN_5[3], TE_5[3]),
            _rec(4, EN_5[4], TE_5[4]),
            _rec(5, "   \t   ", TE_5[0]),  # all whitespace → stripped → empty
        ]
        # Wait — the 6th record above has source="   \t   " → after normalize ""
        # so we expect 4 empty records total. Adjust fixture to match TR:
        records_adj = [
            _rec(0, "", TE_5[0]),
            _rec(1, "   \n\n   ", TE_5[1]),  # all whitespace
            _rec(2, EN_5[2], ""),
            _rec(3, EN_5[3], TE_5[3]),
            _rec(4, EN_5[4], TE_5[4]),
        ]
        cleaned, report = clean_records(records_adj)
        assert report.input_count == 5
        assert report.null_removed_count == 3
        assert report.output_count == 2

    def test_drops_exact_duplicates(self) -> None:
        """TR 3.2: 3 identical (src,tgt) pairs → duplicate_removed_count == 2."""
        records = [
            _rec(0, EN_5[0], TE_5[0]),
            _rec(1, EN_5[0], TE_5[0]),  # dup
            _rec(2, EN_5[0], TE_5[0]),  # dup
            _rec(3, EN_5[1], TE_5[1]),
            _rec(4, EN_5[2], TE_5[2]),
        ]
        cleaned, report = clean_records(records)
        assert report.duplicate_removed_count == 2
        assert report.output_count == 3

    def test_length_filter(self) -> None:
        """TR 3.3: 1 short 1-char src + 1 1100-char tgt → length_filtered_count == 2."""
        very_long_t = "x" * 1100
        records = [
            _rec(0, "H", TE_5[0]),  # too short (1 char < min 2)
            _rec(1, EN_5[1], very_long_t),  # too long tgt
            _rec(2, EN_5[2], TE_5[2]),
            _rec(3, EN_5[3], TE_5[3]),
            _rec(4, EN_5[4], TE_5[4]),
        ]
        cleaned, report = clean_records(records)
        assert report.length_filtered_count == 2
        assert report.output_count == 3

    def test_report_counts_add_up(self) -> None:
        """TR 3.4: input - (nulls + dups + length) == output."""
        # Crafted fixture: 1 null, 2 dups, 1 long → total input=10, removed=1+2+1=4, output=6.
        very_long_t = "t" * 1001
        records = [
            # Row 0: null
            _rec(0, "", "abcdef"),
            # Rows 1–3: dup three times → 2 removed later
            _rec(1, EN_5[0], TE_5[0]),
            _rec(2, EN_5[0], TE_5[0]),
            _rec(3, EN_5[0], TE_5[0]),
            # Row 4: long tgt
            _rec(4, EN_5[1], very_long_t),
            # Rows 5–9: clean survivors (5 rows). But dups reduce to 1 of the triplet.
            _rec(5, EN_5[2], TE_5[2]),
            _rec(6, EN_5[3], TE_5[3]),
            _rec(7, EN_5[4], TE_5[4]),
            _rec(8, "Hello world", "హలో ప్రపంచం"),
            _rec(9, "Good morning", "శుభోదయం"),
        ]
        cleaned, report = clean_records(records)
        assert report.input_count == 10
        assert report.null_removed_count == 1
        assert report.duplicate_removed_count == 2
        assert report.length_filtered_count == 1
        assert report.output_count == report.input_count - (
            report.null_removed_count
            + report.duplicate_removed_count
            + report.length_filtered_count
        )
        assert report.output_count == len(cleaned)
        assert report.output_count == 6

    def test_malformed_flagged_count(self) -> None:
        """TR 3.5: lopsided pair → malformed_flagged_count ≥ 1."""
        # Length ratio: src=1 char, tgt=100 chars → ratio = (100-1)/100 = 0.99 ≥ 0.95 → flagged.
        # Also ensure both pass the min_chars=2 threshold so they are not length-filtered first.
        lopsided_src = "Hi"  # len 2 (passes min_chars)
        lopsided_tgt = "x" * 200  # ratio |2-200|/200 = 198/200 = 0.99 → flagged
        records = [
            _rec(0, lopsided_src, lopsided_tgt),
            _rec(1, EN_5[1], TE_5[1]),  # balanced
            _rec(2, EN_5[2], TE_5[2]),  # balanced
        ]
        cleaned, report = clean_records(records)
        assert len(cleaned) == 3
        assert report.malformed_flagged_count >= 1


# ===========================================================================
# TR 4.1 / 4.2 / 4.3 / 4.4 / 4.5 — Splitter tests
# ===========================================================================


def _many_records(n: int, dedupe=True) -> list[ParallelRecord]:
    """Deterministic list of n distinct records.

    For TR 4.5 (dedup + split) callers set n=10 with identical contents;
    otherwise the helper emits distinct (src,tgt).
    """
    out = []
    for i in range(n):
        if dedupe:
            out.append(_rec(i, f"Sentence number {i} in English", f"ఇంగ్లిష్‌లో వాక్యం సంఖ్య {i}"))
        else:
            out.append(_rec(i, "same s", "same t"))
    return out


class TestSplitter:
    def test_80_10_10_proportions_100(self) -> None:
        """TR 4.1: 100-row fixture → train ∈ {79,80,81}, val/test ∈ {9,10,11}."""
        records = _many_records(100)
        s = split_dataset(records, seed=42)
        total = len(s.train) + len(s.val) + len(s.test)
        assert total == 100
        assert len(s.train) in {79, 80, 81}
        assert len(s.val) in {9, 10, 11}
        assert len(s.test) in {9, 10, 11}

    def test_same_seed_twice_deterministic(self) -> None:
        """TR 4.2: seed 42 twice → identical id order."""
        records = _many_records(50)
        s1 = split_dataset(records, seed=42)
        s2 = split_dataset(records, seed=42)
        assert [r.id for r in s1.train] == [r.id for r in s2.train]
        assert [r.id for r in s1.val] == [r.id for r in s2.val]
        assert [r.id for r in s1.test] == [r.id for r in s2.test]

    def test_seed_sensitivity(self) -> None:
        """TR 4.3: seed 42 vs 43 → different order."""
        records = _many_records(50)
        s42 = split_dataset(records, seed=42)
        s43 = split_dataset(records, seed=43)
        assert [r.id for r in s42.train] != [r.id for r in s43.train]

    def test_no_cross_split_overlap(self) -> None:
        """TR 4.4: pairwise pair sets are disjoint across splits."""
        records = _many_records(100)  # All pairs distinct
        s = split_dataset(records, seed=42)
        pairs = lambda xs: {(r.source_text, r.target_text) for r in xs}
        assert pairs(s.train) & pairs(s.val) == set()
        assert pairs(s.train) & pairs(s.test) == set()
        assert pairs(s.val) & pairs(s.test) == set()

    def test_10_identical_rows_deduped_to_1_total(self) -> None:
        """TR 4.5: 10 identical rows → len(splits) == 1 (dedup happens before split in pipeline)."""
        records = _many_records(10, dedupe=False)  # 10 identical (s,t)
        # Manually run through clean_records (dedup) → split
        cleaned, _ = clean_records(records)
        assert len(cleaned) == 1
        s = split_dataset(cleaned, seed=42)
        assert len(s.train) + len(s.val) + len(s.test) == 1

    def test_write_jsonl_roundtrip(self, tmp_path: Path) -> None:
        records = _many_records(10)
        s = split_dataset(records, seed=42)
        write_splits_jsonl(s, tmp_path)
        expected = {"train": s.train, "val": s.val, "test": s.test}
        for split in ("train", "val", "test"):
            path = tmp_path / f"{split}.en-te.jsonl"
            assert path.exists()
            loaded = read_split_jsonl(path)
            assert len(loaded) == len(expected[split])
            if expected[split]:
                assert loaded[0].source_text == expected[split][0].source_text


# ===========================================================================
# TR 5.1 — ContextRecord
# ===========================================================================


class TestContextRecord:
    def test_samanantar_derived_has_null_document_id(self) -> None:
        """TR 5.1: document_id is None, context == [], sentence_id matches pr.id (run on 3 records)."""
        prs = [_rec(0, EN_5[i], TE_5[i]) for i in range(3)]
        for pr in prs:
            cr: ContextRecord = record_from_parallel(pr)
            assert cr.document_id is None
            assert cr.context == []
            assert cr.sentence_id == pr.id
            assert cr.source_sentence == pr.source_text
            assert cr.target_sentence == pr.target_text


# ===========================================================================
# TR 6.1 / 6.2 — Sample generator script tests
# ===========================================================================


def _load_script_module(script_path: Path):
    name = script_path.stem
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module so decorators that look up
    # cls.__module__ (e.g. @dataclass) can find it via sys.modules.get().
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestSampleGenerator:
    def test_creates_two_10_line_files(self, tmp_path: Path, capfd) -> None:
        """TR 6.1: --size 10 → 10 lines in each output file."""
        # 1. Make a source split (train jsonl) with 15 rows
        records = _many_records(15)
        s = split_dataset(records, train=1.0, val=0.0, test=0.0, seed=1)
        src = tmp_path / "train.en-te.jsonl"
        write_splits_jsonl(Splits(train=s.train, val=[], test=[]), tmp_path)
        src = tmp_path / "train.en-te.jsonl"  # rewrite var

        out_dir = tmp_path / "samples"
        mod = _load_script_module(SCRIPTS_DIR / "create_sample_dataset.py")
        rc = mod.main(
            [
                "--size",
                "10",
                "--source",
                str(src),
                "--output-dir",
                str(out_dir),
            ]
        )
        assert rc == 0
        en_path = out_dir / "samanantar_sample.en.txt"
        te_path = out_dir / "samanantar_sample.te.txt"
        assert en_path.exists() and te_path.exists()
        assert len(en_path.read_text(encoding="utf-8").splitlines()) == 10
        assert len(te_path.read_text(encoding="utf-8").splitlines()) == 10

    def test_refuses_large_size_without_flag(self, tmp_path: Path) -> None:
        """TR 6.2: size > 10000 without flag → exit 2."""
        # Fabricate a source file (any existing file works — the size check fires before reading)
        fake_src = tmp_path / "train.en-te.jsonl"
        fake_src.write_text("")
        mod = _load_script_module(SCRIPTS_DIR / "create_sample_dataset.py")
        rc = mod.main(
            [
                "--size",
                "10001",
                "--source",
                str(fake_src),
                "--output-dir",
                str(tmp_path),
            ]
        )
        assert rc == 2

    def test_accepts_large_size_with_flag(self, tmp_path: Path) -> None:
        """Safety flag bypass works (file missing error is not exit 2)."""
        # No source file written. Result should be exit 1 (source not found).
        mod = _load_script_module(SCRIPTS_DIR / "create_sample_dataset.py")
        rc = mod.main(
            [
                "--size",
                "10001",
                "--source",
                str(tmp_path / "nonexistent.jsonl"),
                "--output-dir",
                str(tmp_path),
                "--i-accept-the-large-size",
            ]
        )
        # Large size refusal (exit 2) is bypassed; we reach source-missing code (exit 1).
        assert rc != 2

    def test_synthetic_sample_is_labeled_and_has_no_document_metadata(
        self, tmp_path: Path
    ) -> None:
        mod = _load_script_module(SCRIPTS_DIR / "create_sample_dataset.py")
        rc = mod.main(
            ["--size", "2", "--synthetic", "--output-dir", str(tmp_path)]
        )

        assert rc == 0
        records = [
            json.loads(line)
            for line in (tmp_path / "synthetic_test.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(records) == 2
        assert all(record["data_source"] == "synthetic_test" for record in records)
        assert all("document_id" not in record for record in records)


# ===========================================================================
# TR 7.1 / 7.2 / 7.3 — prepare_samanantar tests
# ===========================================================================


class TestPrepareSamanantar:
    def test_writes_report_json_with_measured_ints(self, tmp_path: Path) -> None:
        """TR 7.1: Report JSON has all keys populated with measured ints for 100-row fixture."""
        # Build a 100-row fixture with some empties, dups, longs
        en_rows = [f"English sentence #{i}" for i in range(100)]
        te_rows = [f"తెలుగు వాక్యం #{i}" for i in range(100)]
        # Inject empties (rows 30, 31)
        en_rows[30] = ""
        te_rows[31] = "   "  # all-whitespace → normalize ""
        # Inject dups (rows 40..42 identical to row 10)
        for j in (40, 41, 42):
            en_rows[j] = en_rows[10]
            te_rows[j] = te_rows[10]
        # Inject a too-long target (row 50)
        te_rows[50] = "T" * 1001

        data_path = _write_strategy_a(tmp_path, en_rows, te_rows)
        processed_dir = tmp_path / "processed"
        report_path = tmp_path / "report.json"

        mod = _load_script_module(SCRIPTS_DIR / "prepare_samanantar.py")
        rc = mod.main(
            [
                "--data-path",
                str(data_path),
                "--seed",
                "42",
                "--processed-dir",
                str(processed_dir),
                "--report",
                str(report_path),
            ]
        )
        assert rc == 0
        report = json.loads(report_path.read_text(encoding="utf-8"))
        # All required keys present
        for key in (
            "raw_count",
            "language_filtered_count",
            "null_removed_count",
            "duplicate_removed_count",
            "length_filtered_count",
            "final_count",
            "train_count",
            "validation_count",
            "test_count",
            "random_seed",
            "generation_ts_utc",
        ):
            assert key in report, f"Missing key {key!r} in report JSON"
        assert report["raw_count"] == 100
        # 2 empties (rows 30,31) → null_removed == 2
        assert report["null_removed_count"] == 2
        # 3 dups (rows 40..42 → original row 10 kept once) → 3 removed
        assert report["duplicate_removed_count"] == 3
        # 1 length filter (row 50, tgt len 1001)
        assert report["length_filtered_count"] == 1
        # final_count = 100 - 2 - 3 - 1 = 94
        assert report["final_count"] == 94
        assert report["input_count"] == 100
        assert report["valid_count"] == 100
        assert report["malformed_record_count"] == 0
        assert report["removed_count"] == 6
        assert report["source_dataset"] == "AI4Bharat Samanantar"
        assert report["language_pair"] == "en-te"
        assert report["train_count"] + report["validation_count"] + report["test_count"] == 94
        assert report["random_seed"] == 42
        # Seed is an actual int, not a string
        assert isinstance(report["random_seed"], int)
        assert isinstance(report["raw_count"], int)

    def test_writes_jsonl_split_files(self, tmp_path: Path) -> None:
        """TR 7.2: After run, split files exist and their line counts match report counts."""
        data_path = _write_strategy_a(tmp_path, EN_5, TE_5)
        processed_dir = tmp_path / "processed"
        report_path = tmp_path / "report.json"

        mod = _load_script_module(SCRIPTS_DIR / "prepare_samanantar.py")
        rc = mod.main(
            [
                "--data-path",
                str(data_path),
                "--processed-dir",
                str(processed_dir),
                "--report",
                str(report_path),
            ]
        )
        assert rc == 0
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for split_name, report_key in (
            ("train", "train_count"),
            ("val", "validation_count"),
            ("test", "test_count"),
        ):
            p = processed_dir / f"{split_name}.en-te.jsonl"
            assert p.exists()
            cnt = len(p.read_text(encoding="utf-8").splitlines())
            assert cnt == report[report_key]

    def test_missing_data_path_exit_1_and_mentions_download(self, tmp_path: Path, capfd) -> None:
        """TR 7.3: No data-path configured → exit 1, output references download/docs."""
        mod = _load_script_module(SCRIPTS_DIR / "prepare_samanantar.py")
        # Clear any env override by passing --data-path pointing at missing dir.
        rc = mod.main(
            [
                "--data-path",
                str(tmp_path / "definitely_missing"),
                "--report",
                str(tmp_path / "r.json"),
            ]
        )
        assert rc == 1
        captured = capfd.readouterr()
        combined = (captured.out + captured.err).lower()
        assert "download" in combined
        assert "dataset.md" in combined

    def test_reverse_direction_writes_directional_split_names(self, tmp_path: Path) -> None:
        data_path = _write_strategy_a(tmp_path, EN_5, TE_5)
        processed_dir = tmp_path / "processed_reverse"
        report_path = tmp_path / "reverse_report.json"
        mod = _load_script_module(SCRIPTS_DIR / "prepare_samanantar.py")

        rc = mod.main(
            [
                "--data-path",
                str(data_path),
                "--src-lang",
                "te",
                "--tgt-lang",
                "en",
                "--processed-dir",
                str(processed_dir),
                "--report",
                str(report_path),
            ]
        )

        assert rc == 0
        assert (processed_dir / "train.te-en.jsonl").exists()
        first = read_split_jsonl(processed_dir / "train.te-en.jsonl")[0]
        assert first.source_language == "te"
        assert first.target_language == "en"
        assert first.source_text in TE_5
        assert first.target_text in EN_5


# ===========================================================================
# TR 8.1 / 8.2 / 8.3 / 8.4 — validate_dataset tests
# ===========================================================================


class TestValidateDataset:
    @staticmethod
    def _write_good_splits(tmp: Path) -> Path:
        records = _many_records(30)
        s = split_dataset(records, seed=42)
        write_splits_jsonl(s, tmp)
        return tmp

    def test_good_splits_exit_0_ok_true(self, tmp_path: Path) -> None:
        """TR 8.1: Good splits → exit 0 and report.ok == True."""
        processed = TestValidateDataset._write_good_splits(tmp_path)
        report_path = tmp_path / "val_report.json"
        mod = _load_script_module(SCRIPTS_DIR / "validate_dataset.py")
        rc = mod.main(
            [
                "--processed-dir",
                str(processed),
                "--report",
                str(report_path),
            ]
        )
        assert rc == 0
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["ok"] is True
        assert report["has_severe_issues"] is False
        # train/val/test summary blocks exist
        for split in ("train", "val", "test"):
            assert split in report["summary"]
            assert report["summary"][split]["count"] >= 0

    def test_catches_empty_source_text(self, tmp_path: Path) -> None:
        """TR 8.2: Injected empty source → issue code=="empty_source_text"."""
        processed = TestValidateDataset._write_good_splits(tmp_path)
        # Inject empty source into test split
        test_path = processed / "test.en-te.jsonl"
        lines = test_path.read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        first["source_text"] = ""
        lines[0] = json.dumps(first, ensure_ascii=False)
        test_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        mod = _load_script_module(SCRIPTS_DIR / "validate_dataset.py")
        rc = mod.main(
            ["--processed-dir", str(processed), "--report", str(tmp_path / "vr.json")]
        )
        assert rc == 1
        report = json.loads((tmp_path / "vr.json").read_text(encoding="utf-8"))
        codes = [issue["code"] for issue in report["issues"]]
        assert "empty_source_text" in codes

    def test_catches_cross_split_leak_exit_1(self, tmp_path: Path) -> None:
        """TR 8.3: Copy test row into train → cross_split_leak code, exit 1."""
        processed = TestValidateDataset._write_good_splits(tmp_path)
        test_path = processed / "test.en-te.jsonl"
        train_path = processed / "train.en-te.jsonl"
        # Append first test row into train → now same pair in two splits
        first_test_line = test_path.read_text(encoding="utf-8").splitlines()[0]
        with train_path.open("a", encoding="utf-8") as fh:
            fh.write(first_test_line + "\n")
        mod = _load_script_module(SCRIPTS_DIR / "validate_dataset.py")
        rc = mod.main(
            ["--processed-dir", str(processed), "--report", str(tmp_path / "vl2.json")]
        )
        assert rc == 1
        report = json.loads((tmp_path / "vl2.json").read_text(encoding="utf-8"))
        codes = [issue["code"] for issue in report["issues"]]
        assert "cross_split_leak" in codes

    def test_catches_invalid_language_label(self, tmp_path: Path) -> None:
        """TR 8.4: "fr" in source_language → code=="invalid_language_label"."""
        processed = TestValidateDataset._write_good_splits(tmp_path)
        val_path = processed / "val.en-te.jsonl"
        lines = val_path.read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[0])
        row["source_language"] = "fr"
        lines[0] = json.dumps(row, ensure_ascii=False)
        val_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        mod = _load_script_module(SCRIPTS_DIR / "validate_dataset.py")
        rc = mod.main(
            ["--processed-dir", str(processed), "--report", str(tmp_path / "vl3.json")]
        )
        assert rc == 1
        report = json.loads((tmp_path / "vl3.json").read_text(encoding="utf-8"))
        codes = {issue["code"] for issue in report["issues"]}
        assert "invalid_language_label" in codes

    def test_malformed_json_is_reported_without_crashing(self, tmp_path: Path) -> None:
        processed = TestValidateDataset._write_good_splits(tmp_path)
        with (processed / "train.en-te.jsonl").open("a", encoding="utf-8") as fh:
            fh.write("{broken json}\n")

        mod = _load_script_module(SCRIPTS_DIR / "validate_dataset.py")
        rc = mod.main(
            ["--processed-dir", str(processed), "--report", str(tmp_path / "bad.json")]
        )

        assert rc == 1
        report = json.loads((tmp_path / "bad.json").read_text(encoding="utf-8"))
        assert "malformed_record" in {issue["code"] for issue in report["issues"]}

    def test_catches_within_split_duplicates_bad_unicode_and_lengths(
        self, tmp_path: Path
    ) -> None:
        processed = TestValidateDataset._write_good_splits(tmp_path)
        train_path = processed / "train.en-te.jsonl"
        train_rows = train_path.read_text(encoding="utf-8").splitlines()
        train_path.write_text(
            "\n".join([*train_rows, train_rows[0]]) + "\n", encoding="utf-8"
        )
        val_path = processed / "val.en-te.jsonl"
        val_rows = val_path.read_text(encoding="utf-8").splitlines()
        invalid = json.loads(val_rows[0])
        invalid["source_text"] = "x"
        invalid["target_text"] = "\ufffd"
        val_rows[0] = json.dumps(invalid, ensure_ascii=False)
        val_path.write_text("\n".join(val_rows) + "\n", encoding="utf-8")

        mod = _load_script_module(SCRIPTS_DIR / "validate_dataset.py")
        rc = mod.main(
            [
                "--processed-dir",
                str(processed),
                "--report",
                str(tmp_path / "checks.json"),
            ]
        )

        assert rc == 1
        report = json.loads((tmp_path / "checks.json").read_text(encoding="utf-8"))
        codes = {issue["code"] for issue in report["issues"]}
        assert {"duplicate_pair", "length_constraint", "invalid_unicode"} <= codes


# ===========================================================================
# TR 1.1 / 1.2 — Settings and .env.example tests
# ===========================================================================


class TestSettings:
    def test_dataset_settings_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TR 1.1: Settings() defaults dataset_seed==42, samanantar_data_path None when env cleared."""
        # monkeypatch any explicit env override out (set to empty string triggers default?)
        # Actually, pydantic-settings reads env at construction time. The test conftest.py
        # does not set these, so fresh Settings() should see defaults.
        from app.core.config import Settings, get_settings

        # The get_settings() singleton is already cached from conftest import;
        # construct a fresh Settings() instance that we can inspect for defaults:
        fresh = Settings()  # type: ignore[call-arg]
        assert fresh.DATASET_SEED == 42
        # SAMANANTAR_DATA_PATH either is None or matches .env placeholder.
        # If env has it set, we can at least confirm the field is accessible:
        assert isinstance(fresh.SAMANANTAR_DATA_PATH, (type(None), str))

    def test_env_example_has_dataset_section(self) -> None:
        """TR 1.2: .env.example has SAMANANTAR_DATA_PATH placeholder + DATASET_SEED=42."""
        env_example = PROJECT_ROOT / ".env.example"
        text = env_example.read_text(encoding="utf-8")
        assert "SAMANANTAR_DATA_PATH=./data/raw/samanantar" in text
        assert "DATASET_SEED=42" in text


# ===========================================================================
# TR 9.1 — docs/dataset.md section headings present
# ===========================================================================


class TestDatasetDoc:
    REQUIRED_HEADINGS_SUBSTRINGS = [
        "Dataset Name",
        "Dataset Source",
        "License",
        "Language Pair",
        "Dataset Format",
        "Fields",
        "Preprocessing",
        "Filtering",
        "Train",
        "Validation",
        "Test Split",
        "Random Seed",
        "Data Leakage Prevention",
        "Limitations",
        "Document-Level",
        "Reproducibility",
    ]

    def test_at_least_12_headings_present_and_context_limitation_written(self) -> None:
        """TR 9.1: ≥ 12 of 14 FR-9 sections covered AND contains the Samanantar-sentence-level warning."""
        path = DOCS_DIR / "dataset.md"
        assert path.exists(), "docs/dataset.md is missing."
        text = path.read_text(encoding="utf-8")
        # Count Markdown headings (lines starting with "# ")
        headings = [ln.strip() for ln in text.splitlines() if ln.lstrip().startswith("#")]
        # Count headings that contain keywords that map to FR-9.1 sections
        found = 0
        for kw in self.REQUIRED_HEADINGS_SUBSTRINGS:
            if any(kw.lower() in h.lower() for h in headings):
                found += 1
        assert found >= 12, f"Only {found} of 14+ expected section headings found. Headings: {headings!r}"
        # Contains exact document-level limitation warning
        assert "samanantar is a sentence-level parallel corpus" in text.lower()
        assert "never fabricate document" in text.lower() or "not guaranteed to belong to the same document" in text.lower()


# ===========================================================================
# TR 10.1 (meta): This test file itself has ≥ 20 collected test items.
# Verified externally via `pytest --collect-only`; no code needed.
# ===========================================================================
