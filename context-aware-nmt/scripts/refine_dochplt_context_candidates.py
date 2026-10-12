import json
import re
from collections import Counter
from pathlib import Path


INPUT = Path(
    "experiments/evaluation/dochplt/"
    "dochplt_translation_candidates.json"
)

OUTPUT = Path(
    "experiments/evaluation/dochplt/"
    "refined_context_candidates.json"
)

TOP_N = 100


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than",
    "to", "of", "in", "on", "for", "from", "with", "by", "at",
    "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "they",
    "them", "their", "he", "she", "his", "her", "we", "you",
    "your", "our", "my", "me", "i", "about", "into", "over",
    "after", "before", "during", "also", "very", "can", "will",
    "would", "should", "could", "may", "might", "do", "does",
    "did", "have", "has", "had"
}


def normalize(text):
    return re.sub(r"\s+", " ", text.lower().strip())


def tokens(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [
        w for w in words
        if w not in STOPWORDS and len(w) >= 3
    ]


def similarity(source, context):
    source_tokens = Counter(tokens(source))
    context_tokens = Counter(tokens(context))

    if not source_tokens or not context_tokens:
        return 0.0

    intersection = set(source_tokens) & set(context_tokens)

    numerator = sum(
        source_tokens[w] * context_tokens[w]
        for w in intersection
    )

    source_norm = sum(
        value * value
        for value in source_tokens.values()
    ) ** 0.5

    context_norm = sum(
        value * value
        for value in context_tokens.values()
    ) ** 0.5

    if source_norm == 0 or context_norm == 0:
        return 0.0

    return numerator / (source_norm * context_norm)


def is_trivial(source, context_sentences):
    source_norm = normalize(source)

    # Reject context that simply repeats the current sentence.
    for sentence in context_sentences:
        if normalize(sentence) == source_norm:
            return True

    # Reject extremely short/title-like context.
    meaningful = [
        sentence
        for sentence in context_sentences
        if len(tokens(sentence)) >= 3
    ]

    if not meaningful:
        return True

    # Reject excessive repetition.
    normalized = [normalize(x) for x in context_sentences]

    if len(set(normalized)) < len(normalized):
        return True

    return False


def main():

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    with INPUT.open("r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = data["candidates"]

    refined = []
    rejected = Counter()
    seen = set()

    for candidate in candidates:

        source = candidate["source_sentence"]

        context_sentences = [
            item["text"]
            for item in candidate["context"]
        ]

        # -----------------------------------------------
        # Trivial/repetitive context
        # -----------------------------------------------

        if is_trivial(source, context_sentences):
            rejected["trivial_or_repeated_context"] += 1
            continue

        # -----------------------------------------------
        # Duplicate candidate detection
        # -----------------------------------------------

        key = (
            candidate["source_document_id"],
            candidate["source_sentence_id"],
            candidate["target_sentence_id"],
            normalize(source),
            tuple(normalize(x) for x in context_sentences),
        )

        if key in seen:
            rejected["duplicate_candidate"] += 1
            continue

        seen.add(key)

        # -----------------------------------------------
        # Context relevance
        # -----------------------------------------------

        context_text = " ".join(context_sentences)

        score = similarity(
            source,
            context_text,
        )

        # Require some lexical relationship.
        if score <= 0.05:
            rejected["very_low_context_similarity"] += 1
            continue

        # Reject near-perfect repetition.
        if score >= 0.995:
            rejected["near_duplicate_context"] += 1
            continue

        refined.append(
            {
                **candidate,
                "context_relevance_score": score,
            }
        )

    # Prefer meaningful context relevance, but use alignment
    # only as a deterministic secondary criterion.
    refined.sort(
        key=lambda x: (
            x["context_relevance_score"],
            x["alignment_score"],
            x["source_document_id"],
            x["source_sentence_id"],
        ),
        reverse=True,
    )

    selected = refined[:TOP_N]

    output = {
        "dataset": "DocHPLT",
        "config": "en-te",
        "input_candidate_count": len(candidates),
        "refined_candidate_count": len(refined),
        "selected_count": len(selected),
        "selection_policy": {
            "minimum_alignment_score": 0.70,
            "require_preceding_same_document_context": True,
            "reject_repeated_current_sentence": True,
            "reject_repeated_context": True,
            "reject_duplicate_candidates": True,
            "minimum_context_similarity": 0.05,
            "maximum_context_similarity": 0.995,
            "target_reference_not_used_for_selection": True,
        },
        "rejection_counts": dict(rejected),
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
    print("DocHPLT Refined Context Candidate Selection")
    print("=" * 90)

    print(f"Input candidates   : {len(candidates)}")
    print(f"Refined candidates : {len(refined)}")
    print(f"Selected           : {len(selected)}")

    print()
    print("Rejection counts:")

    for reason, count in rejected.items():
        print(f"  {reason:35}: {count}")

    print()
    print("Top 20 refined candidates:")
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