import numpy as np
from .validate import positive_int

def _normalize(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)

class DenseIndex:
    """Exact cosine search. embed_fn: list[str] -> ndarray (n, dim).
    query_prefix: prepended to queries only (e.g. BGE's retrieval instruction)."""
    def __init__(self, chunks, embed_fn, query_prefix=""):
        self.chunks = list(chunks)
        self.embed_fn = embed_fn
        self.query_prefix = query_prefix
        self.matrix = None
        if self.chunks:  # some embedders return a 1-D empty array for [] input
            m = np.asarray(embed_fn([c.text for c in self.chunks]), dtype=np.float32)
            if m.ndim != 2 or m.shape[0] != len(self.chunks):
                raise ValueError(f"embedder returned shape {m.shape} for {len(self.chunks)} chunks")
            if not np.isfinite(m).all():
                raise ValueError("embedder returned non-finite values")
            self.matrix = _normalize(m)

    def search(self, query, k=10):
        positive_int("k", k)
        if self.matrix is None:
            return []
        q = np.asarray(self.embed_fn([self.query_prefix + query]), dtype=np.float32)
        if q.shape != (1, self.matrix.shape[1]):
            raise ValueError(f"query embedding shape {q.shape} != (1, {self.matrix.shape[1]})")
        if not np.isfinite(q).all():
            raise ValueError("embedder returned non-finite values for the query")
        sims = self.matrix @ _normalize(q)[0]
        order = np.argsort(-sims)[:k]
        return [(self.chunks[i].chunk_id, float(sims[i])) for i in order]
