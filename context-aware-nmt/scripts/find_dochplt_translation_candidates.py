import json
import re
from pathlib import Path

import pyarrow.parquet as pq


SHARD = Path(
    "data/raw/dochplt/en-te/train-00000-of-00030.parquet"
)

OUTPUT = Path(
    "experiments/evaluation/dochplt/"
    "dochplt_translation_candidates.json"
)

THRESHOLD = 0.70
CONTEXT_WINDOW = 2

TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")
LATIN_RE = re.compile(r"[A-Za-z]")


def has_telugu(text: str) -> bool:
    return bool(TELUGU_RE.search(text))


def has_latin(text: str) -> bool:
    return bool(LATIN_RE.search(text))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def is_metadata_like(text: str) -> bool:
    t = text.lower()

    patterns = [
        r"\bfacebook comments\b",
        r"\bpost comments\b",
        r"\bsubscribe to\b",
        r"\bjavascript enabled\b",
        r"\bour email is\b",
        r"\bcomments \(atom\)",
        r"\bhttp://",
        r"\bhttps://",
        r"\bcopyright\b",
        r"\bdisclaimer\b",
    ]

    return any(re.search(p, t) for p in patterns)


def sentence_map(doc):
    ids = doc["ids"]
    sentences = doc["sentences"]

    return {
        str(sentence_id): str(sentence)
        for sentence_id, sentence in zip(ids, sentences)
    }


def preceding_context(source_ids, source_map, current_id):
    try:
        position = source_ids.index(current_id)
    except ValueError:
        return []

    previous_ids = source_ids[
        max(0, position - CONTEXT_WINDOW):position
    ]

    return [
        {
            "id": str(sentence_id),
            "text": source_map[str(sentence_id)],
        }
        for sentence_id in previous_ids
    ]


def main():
    if not SHARD.exists():
        raise FileNotFoundError(
            f"Missing shard: {SHARD}"
        )

    table = pq.read_table(SHARD)

    rows = table.to_pylist()

    candidates = []

    stats = {
        "documents": len(rows),
        "alignment_groups": 0,
        "one_to_one": 0,
        "score_pass": 0,
        "source_latin": 0,
        "target_telugu": 0,
        "context_available": 0,
        "metadata_rejected": 0,
        "source_equals_target_rejected": 0,
        "candidates": 0,
    }

    for row_index, row in enumerate(rows):

        src_doc = row["src_doc"]
        tgt_doc = row["tgt_doc"]

        source_ids = [str(x) for x in src_doc["ids"]]
        target_ids = [str(x) for x in tgt_doc["ids"]]

        source_map = sentence_map(src_doc)
        target_map = sentence_map(tgt_doc)

        source_id_set = set(source_ids)
        target_id_set = set(target_ids)

        alignments = row["alignment"] or []

        stats["alignment_groups"] += len(alignments)

        for alignment in alignments:

            src_refs = [
                str(x)
                for x in (alignment.get("src") or [])
            ]

            tgt_refs = [
                str(x)
                for x in (alignment.get("tgt") or [])
            ]

            score = alignment.get("aligner-score")

            # --------------------------------------------------
            # Strict 1-to-1 alignment
            # --------------------------------------------------

            if len(src_refs) != 1 or len(tgt_refs) != 1:
                continue

            stats["one_to_one"] += 1

            source_id = src_refs[0]
            target_id = tgt_refs[0]

            if source_id not in source_id_set:
                continue

            if target_id not in target_id_set:
                continue

            if score is None or float(score) < THRESHOLD:
                continue

            stats["score_pass"] += 1

            source = source_map[source_id].strip()
            target = target_map[target_id].strip()

            if not source or not target:
                continue

            # --------------------------------------------------
            # Language/script sanity
            # --------------------------------------------------

            if not has_latin(source):
                continue

            stats["source_latin"] += 1

            if not has_telugu(target):
                continue

            stats["target_telugu"] += 1

            # --------------------------------------------------
            # Remove obvious metadata
            # --------------------------------------------------

            if is_metadata_like(source) or is_metadata_like(target):
                stats["metadata_rejected"] += 1
                continue

            # --------------------------------------------------
            # Source and target should not be identical
            # --------------------------------------------------

            if normalize(source) == normalize(target):
                stats["source_equals_target_rejected"] += 1
                continue

            # --------------------------------------------------
            # Must have preceding same-document context
            # --------------------------------------------------

            context = preceding_context(
                source_ids,
                source_map,
                source_id,
            )

            if not context:
                continue

            stats["context_available"] += 1

            # --------------------------------------------------
            # Candidate
            # --------------------------------------------------

            candidates.append(
                {
                    "row_index": row_index,
                    "source_document_id": row["src_doc_id"],
                    "target_document_id": row["tgt_doc_id"],
                    "lang_pair": row["lang_pair"],
                    "source_sentence_id": source_id,
                    "target_sentence_id": target_id,
                    "source_sentence": source,
                    "target_reference": target,
                    "alignment_score": float(score),
                    "context": context,
                }
            )

    output = {
        "dataset": "DocHPLT",
        "config": "en-te",
        "split": "train",
        "source_shard": str(SHARD),
        "alignment_threshold": THRESHOLD,
        "context_window": CONTEXT_WINDOW,
        "selection_policy": {
            "alignment": "strict 1-to-1",
            "minimum_alignment_score": THRESHOLD,
            "source_requires_latin_script": True,
            "target_requires_telugu_script": True,
            "exclude_metadata_like_text": True,
            "exclude_source_equals_target": True,
            "require_preceding_same_document_context": True,
        },
        "statistics": stats,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 90)
    print("DocHPLT Full-Shard Translation Candidate Scan")
    print("=" * 90)

    for key, value in stats.items():
        print(f"{key:32}: {value}")

    print()
    print(f"Candidate count             : {len(candidates)}")
    print(f"Saved                       : {OUTPUT}")
    print("=" * 90)


if __name__ == "__main__":
    main()