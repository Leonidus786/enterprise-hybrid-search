from .validate import finite_number

def rrf(rankings, k=60):
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank).
    - a repeated id inside one ranking counts only at its best (first) rank
    - ties are broken by first-seen order (deterministic)"""
    finite_number("rrf constant k", k)
    if k <= 0:
        raise ValueError("rrf constant k must be positive")
    scores = {}
    for ranking in rankings:
        seen = set()
        for rank, doc in enumerate(ranking, 1):
            if doc in seen:
                continue
            seen.add(doc)
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])  # stable sort keeps first-seen on ties
