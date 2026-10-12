import json
import math
import re
from collections import Counter
from pathlib import Path


INPUT = Path(
    "experiments/evaluation/dochplt/"
    "dochplt_translation_candidates.json"
)

OUTPUT = Path(
    "experiments/evaluation/dochplt/"
    "context_quality_candidates.json"
)

TOP_N = 200


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than",
    "to", "of", "in", "on", "for", "from", "with", "by", "at",
    "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "they",
    "them", "their", "he", "she", "his", "her", "we", "you",
    "your", "our", "my", "me", "i", "about", "into", "over",
    "after", "before", "during", "also", "very", "can", "will",
    "would", "should", "could", "may", "might", "do", "does",
    "did", "have", "has", "had", "not", "no"
}


def tokenize(text):
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [
        token
        for token in tokens
        if token not in STOPWORDS and len(token) > 2
    ]


def cosine_similarity(text_a, text_b):
    a = Counter(tokenize(text_a))
    b = Counter(tokenize(text_b))

    if not a or not b:
        return 0.0

    common = set(a) & set(b)

    numerator = sum(a[x] * b[x] for x in common)

    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return numerator / (norm_a * norm_b)


def main():

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    with INPUT.open("r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = data["candidates"]

    scored = []

    for candidate in candidates:

        source = candidate["source_sentence"]

        context_text = " ".join(
            item["text"]
            for item in candidate["context"]
        )

        similarity = cosine_similarity(
            source,
            context_text
        )

        scored.append(
            {
                **candidate,
                "context_relevance_score": similarity,
            }
        )

    # Highest context relevance first.
    # Alignment score is used as a deterministic tie-breaker.
    scored.sort(
        key=lambda x: (
            x["context_relevance_score"],
            x["alignment_score"],
        ),
        reverse=True,
    )

    selected = scored[:TOP_N]

    output = {
        "dataset": "DocHPLT",
        "config": "en-te",
        "source_candidate_count": len(candidates),
        "selected_count": len(selected),
        "selection_policy": {
            "ranking": (
                "lexical cosine similarity between preceding "
                "same-document context and current source sentence"
            ),
            "tie_breaker": "alignment score",
            "selection_count": TOP_N,
            "target_reference_not_used_for_selection": True,
        },
        "candidates": selected,
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
    print("DocHPLT Context-Quality Candidate Selection")
    print("=" * 90)
    print(f"Input candidates : {len(candidates)}")
    print(f"Selected         : {len(selected)}")
    print()

    print("Top 20 candidates:")
    print()

    for i, candidate in enumerate(selected[:20], 1):
        print(f"--- Candidate {i} ---")
        print(
            f"Context relevance : "
            f"{candidate['context_relevance_score']:.4f}"
        )
        print(
            f"Alignment score   : "
            f"{candidate['alignment_score']:.4f}"
        )
        print(
            f"Source            : "
            f"{candidate['source_sentence']}"
        )
        print(
            f"Target            : "
            f"{candidate['target_reference']}"
        )
        print(
            f"Context           : "
            f"{[x['text'] for x in candidate['context']]}"
        )
        print()

    print(f"Saved: {OUTPUT}")
    print("=" * 90)


if __name__ == "__main__":
    main()