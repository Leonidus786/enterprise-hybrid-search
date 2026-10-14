import time

from .dense import DenseIndex
from .fusion import rrf
from .rerank import Reranker
from .sparse import SparseIndex
from .validate import positive_int


MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


class HybridPipeline:
    def __init__(
        self,
        chunks,
        embed_fn,
        score_fn=None,
        candidate_k=50,
        rrf_k=60,
        query_prefix="",
    ):
        positive_int("candidate_k", candidate_k)
        positive_int("rrf_k", rrf_k)

        self.chunks = list(chunks)
        self._by_id = {
            chunk.chunk_id: chunk for chunk in self.chunks
        }

        if len(self._by_id) != len(self.chunks):
            raise ValueError("duplicate chunk_id in corpus")

        self.sparse = SparseIndex(self.chunks)
        self.dense = DenseIndex(
            self.chunks,
            embed_fn,
            query_prefix=query_prefix,
        )

        self.reranker = Reranker(score_fn) if score_fn else None
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

        # Instrumentation from the most recent search.
        self.last_timing_ms = {
            "retrieval": 0.0,
            "rerank": 0.0,
        }

        # Candidate chunk IDs BEFORE reranking.
        self.last_candidate_ids = []

    @property
    def chunk_ids(self):
        return set(self._by_id)

    @property
    def doc_ids(self):
        return {chunk.doc_id for chunk in self.chunks}

    def doc_of(self, chunk_id):
        return self._by_id[chunk_id].doc_id

    def search_docs(self, query, mode="hybrid", k=10):
        """Return up to k unique parent document IDs.

        Rank chunks first and then deduplicate by parent document.
        For hybrid_rerank, preserve the fused ranking outside the
        reranked candidate pool.
        """
        positive_int("k", k)

        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

        if not self.chunks:
            return []

        seen = set()
        docs = []

        # Search the full chunk ranking so duplicate chunks belonging
        # to the same document do not prematurely limit the result set.
        ranked_chunks = self.search(
            query,
            mode=mode,
            k=len(self.chunks),
        )

        for chunk_id in ranked_chunks:
            doc_id = self._by_id[chunk_id].doc_id

            if doc_id not in seen:
                seen.add(doc_id)
                docs.append(doc_id)

                if len(docs) >= k:
                    break

        return docs

    def search(self, query, mode="hybrid", k=10):
        """Return ranked chunk IDs, up to k results.

        last_candidate_ids contains the candidate chunks before reranking.
        last_timing_ms contains retrieval and reranker component timings.
        """
        positive_int("k", k)

        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

        if mode == "hybrid_rerank" and self.reranker is None:
            raise ValueError("hybrid_rerank needs score_fn")

        self.last_timing_ms = {
            "retrieval": 0.0,
            "rerank": 0.0,
        }
        self.last_candidate_ids = []

        if not self.chunks:
            return []

        # Sparse-only retrieval.
        if mode == "bm25":
            start = time.perf_counter()
            results = [
                chunk_id
                for chunk_id, _ in self.sparse.search(query, k)
            ]

            self.last_timing_ms["retrieval"] = (
                time.perf_counter() - start
            ) * 1000

            self.last_candidate_ids = results[:self.candidate_k]
            return results

        # Dense-only retrieval.
        if mode == "dense":
            start = time.perf_counter()
            results = [
                chunk_id
                for chunk_id, _ in self.dense.search(query, k)
            ]

            self.last_timing_ms["retrieval"] = (
                time.perf_counter() - start
            ) * 1000

            self.last_candidate_ids = results[:self.candidate_k]
            return results

        # Hybrid candidate generation: BM25 + dense + RRF.
        depth = min(
            len(self.chunks),
            max(k, self.candidate_k),
        )

        start = time.perf_counter()

        sparse_results = [
            chunk_id
            for chunk_id, _ in self.sparse.search(query, depth)
        ]
        dense_results = [
            chunk_id
            for chunk_id, _ in self.dense.search(query, depth)
        ]

        fused = [
            chunk_id
            for chunk_id, _ in rrf(
                [sparse_results, dense_results],
                self.rrf_k,
            )
        ]

        self.last_timing_ms["retrieval"] = (
            time.perf_counter() - start
        ) * 1000

        # Store the pre-reranking candidate pool for evaluation.
        self.last_candidate_ids = fused[:self.candidate_k]

        if mode == "hybrid":
            return fused[:k]

        # Only rerank the first candidate_k fused chunks.
        candidate_ids = list(self.last_candidate_ids)
        candidates = [
            self._by_id[chunk_id]
            for chunk_id in candidate_ids
        ]

        if not candidates:
            return fused[:k]

        start = time.perf_counter()

        reranked_chunks = self.reranker.rerank(
            query,
            candidates,
            top_n=len(candidates),
        )

        self.last_timing_ms["rerank"] = (
            time.perf_counter() - start
        ) * 1000

        reranked_ids = [
            chunk.chunk_id for chunk in reranked_chunks
        ]
        reranked_set = set(reranked_ids)

        # Preserve any candidate omitted by the reranker.
        unreturned_candidates = [
            chunk_id
            for chunk_id in candidate_ids
            if chunk_id not in reranked_set
        ]

        # Do not discard the rest of the original fused ranking.
        untouched_tail = fused[len(candidate_ids):]

        combined = (
            reranked_ids
            + unreturned_candidates
            + untouched_tail
        )

        return combined[:k]
