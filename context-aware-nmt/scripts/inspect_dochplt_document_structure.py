from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dochplt"
    / "en-te"
    / "train-00000-of-00030.parquet"
)


def describe(value, name, depth=0):
    indent = " " * depth

    print()
    print(f"{indent}{name}")
    print(f"{indent}{'-' * len(name)}")
    print(f"{indent}Python type: {type(value)}")

    if isinstance(value, (bytes, bytearray)):
        print(f"{indent}Length: {len(value)} bytes")
        print(f"{indent}First 200 bytes:")
        print(repr(value[:200]))
        return

    if isinstance(value, str):
        print(f"{indent}Length: {len(value)} characters")
        print(f"{indent}First 2000 characters:")
        print(repr(value[:2000]))
        return

    if isinstance(value, dict):
        print(f"{indent}Dictionary keys: {list(value.keys())}")

        for key, item in list(value.items())[:10]:
            describe(item, f"key={key!r}", depth + 2)

        return

    if isinstance(value, (list, tuple)):
        print(f"{indent}Length: {len(value)}")

        for index, item in enumerate(value[:5]):
            describe(item, f"item[{index}]", depth + 2)

        return

    print(f"{indent}repr:")
    print(f"{indent}{repr(value)[:2000]}")


def main():
    print("=" * 80)
    print("DocHPLT Raw Document Structure Inspection")
    print("=" * 80)

    print()
    print(f"Input file:")
    print(INPUT_FILE)

    table = pq.read_table(INPUT_FILE)

    print()
    print(f"Rows: {table.num_rows}")
    print(f"Columns: {table.column_names}")

    row = 0

    print()
    print("=" * 80)
    print(f"INSPECTING ROW {row}")
    print("=" * 80)

    src_doc = table["src_doc"][row].as_py()
    tgt_doc = table["tgt_doc"][row].as_py()
    alignment = table["alignment"][row].as_py()

    print()
    print("SOURCE DOCUMENT")
    print("=" * 80)

    describe(src_doc, "src_doc")

    print()
    print("TARGET DOCUMENT")
    print("=" * 80)

    describe(tgt_doc, "tgt_doc")

    print()
    print("ALIGNMENT")
    print("=" * 80)

    describe(alignment, "alignment")

    print()
    print("=" * 80)
    print("JSON-SERIALIZABLE SNAPSHOT")
    print("=" * 80)

    def safe(value):
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except TypeError:
            return repr(value)

    snapshot = {
        "src_doc": safe(src_doc),
        "tgt_doc": safe(tgt_doc),
        "alignment": safe(alignment),
    }

    print(
        json.dumps(
            snapshot,
            ensure_ascii=False,
            indent=2,
            default=str,
        )[:10000]
    )


if __name__ == "__main__":
    main()