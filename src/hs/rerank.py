import math
from .validate import positive_int

class Reranker:
    """score_fn: list[(query, text)] -> list[float] (same shape as CrossEncoder.predict).
    Ties keep the incoming (fusion) order."""
    def __init__(self, score_fn):
        self.score_fn = score_fn

    def rerank(self, query, chunks, top_n=10):
        positive_int("top_n", top_n)
        if not chunks:
            return []
        scores = list(self.score_fn([(query, c.text) for c in chunks]))
        if len(scores) != len(chunks):
            raise ValueError(f"scorer returned {len(scores)} scores for {len(chunks)} pairs")
        if not all(math.isfinite(s) for s in scores):
            raise ValueError("scorer returned non-finite scores")
        ranked = sorted(zip(scores, chunks), key=lambda x: -x[0])
        return [c for _, c in ranked[:top_n]]
