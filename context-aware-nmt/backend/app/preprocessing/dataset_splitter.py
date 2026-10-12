"""Deterministic 80/10/10 train/validation/test splitter.

Uses a local :class:`random.Random` instance seeded with ``DATASET_SEED`` (default
42) so that it never mutates the global ``random`` module state — two identical
calls with the same seed + inputs produce byte-identical output.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from app.preprocessing.dataset_loader import ParallelRecord

T = TypeVar("T")


@dataclass(slots=True)
class Splits:
    """Three disjoint record lists."""

    train: list[ParallelRecord] = field(default_factory=list)
    val: list[ParallelRecord] = field(default_factory=list)
    test: list[ParallelRecord] = field(default_factory=list)


def split_dataset(
    records: list[ParallelRecord],
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
) -> Splits:
    """Split ``records`` into three disjoint, deterministic buckets.

    * Buckets sum exactly to ``len(records)`` (no record lost).
    * Uses a local ``random.Random(seed)`` so no global state is touched.
    * Shuffles a copy — the input list is **not** mutated.
    * Duplicate records (if any) remain in whichever bucket their shuffled
      index lands — callers should run :func:`clean_records` first so dedupe
      happens **before** splitting (prevents cross-split leakage of identical
      pairs).

    Proportions are approximate (integer rounding may shift 1–2 records between
    buckets) but always cover every record exactly once.
    """

    if abs((train + val + test) - 1.0) > 1e-6:
        raise ValueError(
            f"train+val+test must sum to 1.0 (got {train}+{val}+{test}={train+val+test})."
        )
    if not (0.0 <= train <= 1.0 and 0.0 <= val <= 1.0 and 0.0 <= test <= 1.0):
        raise ValueError(
            "train, val, test must each be in the inclusive range [0.0, 1.0]."
        )
    if train == 0.0:
        raise ValueError("train split proportion cannot be 0.0 (at least one train record required).")

    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(round(n * train))
    val_end = train_end + int(round(n * val))
    # test takes the remainder to guarantee total==n (absorbs rounding error)

    return Splits(
        train=shuffled[:train_end],
        val=shuffled[train_end:val_end],
        test=shuffled[val_end:],
    )


def _split_filenames(language_pair: tuple[str, str]) -> dict[str, str]:
    pair = f"{language_pair[0]}-{language_pair[1]}"
    return {
        "train": f"train.{pair}.jsonl",
        "val": f"val.{pair}.jsonl",
        "test": f"test.{pair}.jsonl",
    }


def _record_to_dict(r: ParallelRecord) -> dict:
    return {
        "id": r.id,
        "source_text": r.source_text,
        "target_text": r.target_text,
        "source_language": r.source_language,
        "target_language": r.target_language,
        "data_source": r.data_source,
    }


def write_splits_jsonl(
    splits: Splits,
    out_dir: Path | str,
    language_pair: tuple[str, str] = ("en", "te"),
) -> dict[str, int]:
    """Write the three splits as JSONL and return per-split line counts.

    ``json.dumps`` uses ``sort_keys=True, ensure_ascii=False`` so Telugu
    characters are stored verbatim and files are byte-identical across runs
    given the same records.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    filenames = _split_filenames(language_pair)
    for split_name, records in (
        ("train", splits.train),
        ("val", splits.val),
        ("test", splits.test),
    ):
        p = out_dir / filenames[split_name]
        written = 0
        with p.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(
                    json.dumps(_record_to_dict(r), sort_keys=True, ensure_ascii=False)
                    + "\n"
                )
                written += 1
        counts[split_name] = written
    return counts


def read_split_jsonl(path: Path | str) -> list[ParallelRecord]:
    """Inverse of :func:`write_splits_jsonl` — load one split file."""
    path = Path(path)
    records: list[ParallelRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                ParallelRecord(
                    id=int(obj.get("id", 0)),
                    source_text=str(obj.get("source_text", "")),
                    target_text=str(obj.get("target_text", "")),
                    source_language=str(obj.get("source_language", "en")),
                    target_language=str(obj.get("target_language", "te")),
                    data_source=str(obj.get("data_source", "samanantar")),
                )
            )
    return records
