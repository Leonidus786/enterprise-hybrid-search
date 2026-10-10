from .dense import DenseIndex
from .fusion import rrf
from .rerank import Reranker
from .sparse import SparseIndex
from .validate import positive_int

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")

class HybridPipeline:
    def __init__(self, chunks, embed_fn, score_fn=None, candidate_k=50, rrf_k=60, query_prefix=""):
        positive_int("candidate_k", candidate_k)
        self.chunks = list(chunks)
        self._by_id = {c.chunk_id: c for c in self.chunks}
        if len(self._by_id) != len(self.chunks):
            raise ValueError("duplicate chunk_id in corpus")
        self.sparse = SparseIndex(self.chunks)
        self.dense = DenseIndex(self.chunks, embed_fn, query_prefix=query_prefix)
        self.reranker = Reranker(score_fn) if score_fn else None
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    @property
    def chunk_ids(self):
        return set(self._by_id)

    @property
    def doc_ids(self):
        return {c.doc_id for c in self.chunks}

    def doc_of(self, chunk_id):
        return self._by_id[chunk_id].doc_id

    def search_docs(self, query, mode="hybrid", k=10):
        """Top-k UNIQUE parent doc ids. Ranks every chunk first, then dedupes, so many
        strong chunks of one doc cannot crowd other docs out of the top-k.
        (hybrid_rerank only sees candidate_k fused chunks, so it can return fewer than k docs.)"""
        positive_int("k", k)
        if not self.chunks:
            return []
        seen, docs = set(), []
        for cid in self.search(query, mode=mode, k=len(self.chunks)):
            d = self._by_id[cid].doc_id
            if d not in seen:
                seen.add(d)
                docs.append(d)
                if len(docs) == k:
                    break
        return docs

    def search(self, query, mode="hybrid", k=10):
        """Returns ranked chunk_ids."""
        positive_int("k", k)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "hybrid_rerank" and self.reranker is None:
            raise ValueError("hybrid_rerank needs score_fn")
        n = max(k, self.candidate_k)
        if mode == "bm25":
            return [i for i, _ in self.sparse.search(query, k)]
        if mode == "dense":
            return [i for i, _ in self.dense.search(query, k)]
        sp = [i for i, _ in self.sparse.search(query, n)]
        de = [i for i, _ in self.dense.search(query, n)]
        fused = [i for i, _ in rrf([sp, de], self.rrf_k)]
        if mode == "hybrid":
            return fused[:k]
        cands = [self._by_id[i] for i in fused[:self.candidate_k]]
        return [c.chunk_id for c in self.reranker.rerank(query, cands, top_n=k)]
