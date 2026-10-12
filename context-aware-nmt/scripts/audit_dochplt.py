import pyarrow.parquet as pq

PARQUET_PATH = "data/raw/dochplt/en-te/train-00000-of-00030.parquet"
THRESHOLD = 0.70

print("Loading DocHPLT shard...")
rows = pq.read_table(PARQUET_PATH).to_pylist()

checked = 0
ordered = 0
unique_ids = 0
aligned_ok = 0

for row in rows:
    checked += 1

    src_ids = row["src_doc"]["ids"]
    tgt_ids = row["tgt_doc"]["ids"]

    # Check source sentence IDs are in document order.
    try:
        source_numbers = [int(x.split(".")[0]) for x in src_ids]
        if all(
            x <= y
            for x, y in zip(source_numbers, source_numbers[1:])
        ):
            ordered += 1
    except (ValueError, IndexError):
        pass

    # Check sentence IDs are unique.
    if (
        len(src_ids) == len(set(src_ids))
        and len(tgt_ids) == len(set(tgt_ids))
    ):
        unique_ids += 1

    # Validate alignments above the selected threshold.
    valid = True

    for alignment in row["alignment"]:
        if alignment["aligner-score"] >= THRESHOLD:
            src = alignment["src"]
            tgt = alignment["tgt"]

            if not (
                len(src) == 1
                and len(tgt) == 1
                and src[0] in src_ids
                and tgt[0] in tgt_ids
            ):
                valid = False
                break

    if valid:
        aligned_ok += 1


print()
print("=" * 60)
print("DocHPLT STRUCTURAL AUDIT")
print("=" * 60)
print(f"Documents checked: {checked}")
print(f"Source IDs numerically ordered: {ordered}")
print(f"Documents with unique source+target IDs: {unique_ids}")
print(
    f"Documents with valid >= {THRESHOLD:.2f} "
    f"alignment references: {aligned_ok}"
)
print("=" * 60)