import json
import re
import time
from collections.abc import Mapping

from .dense import DenseIndex
from .fusion import rrf
from .rerank import Reranker
from .sparse import SparseIndex
from .validate import positive_int


MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")

_IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")


def _same_json_mapping(left, right):
    """Compare JSON mappings without Python bool/int equality aliasing."""
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    try:
        left_json = json.dumps(
            dict(left), sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False,
        )
        right_json = json.dumps(
            dict(right), sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False,
        )
    except (TypeError, ValueError):
        return False
    return left_json == right_json


def _verified_query_embedding_identity(embed_fn):
    """Return model identity only when the query embedder proves it."""
    if not callable(embed_fn):
        raise TypeError("query_embed_fn must be callable")

    if hasattr(embed_fn, "verified_embedding_identity"):
        identity = embed_fn.verified_embedding_identity
        if identity is None:
            raise ValueError(
                "query embedder does not expose a verified embedding identity"
            )
    else:
        metadata = getattr(embed_fn, "model_metadata", None)
        config = getattr(embed_fn, "embedding_config", None)
        if not isinstance(metadata, Mapping):
            raise ValueError("query embedder is missing verified model metadata")
        if metadata.get("revision_verified") is not True:
            raise ValueError("query embedder immutable model revision is unverified")
        identity = {
            "model_id": metadata.get("model_id"),
            "model_revision": metadata.get("resolved_revision"),
            "config": config,
        }

    if not isinstance(identity, Mapping):
        raise ValueError("query embedder identity must be a mapping")
    model_id = identity.get("model_id")
    revision = identity.get("model_revision")
    config = identity.get("config")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("query embedder identity has no valid model_id")
    if not isinstance(revision, str) or not _IMMUTABLE_REVISION_RE.fullmatch(revision):
        raise ValueError("query embedder identity has no verified immutable revision")
    if not isinstance(config, Mapping):
        raise ValueError("query embedder identity has no declared embedding_config")
    return {
        "model_id": model_id,
        "model_revision": revision.lower(),
        "config": dict(config),
    }


class HybridPipeline:
    def __init__(
        self,
        chunks,
        embed_fn,
        score_fn=None,
        candidate_k=50,
        rrf_k=60,
        query_prefix="",
        query_embed_fn=None,
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
            query_embed_fn=query_embed_fn,
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

    @classmethod
    def from_snapshot(
        cls,
        snapshot,
        *,
        query_embed_fn,
        query_prefix="",
        score_fn=None,
        candidate_k=50,
        rrf_k=60,
    ):
        """Build a searchable pipeline from a validated persistent snapshot.

        The persisted matrix is reused as-is. The query embedder must prove the
        model ID, immutable revision and vector-affecting configuration stored
        in the manifest. Query-prefix compatibility and the supported BM25
        configuration are checked before constructing either retrieval index.
        """
        from .index_store import LoadedIndexSnapshot

        if not isinstance(snapshot, LoadedIndexSnapshot):
            raise TypeError("snapshot must be a LoadedIndexSnapshot")
        positive_int("candidate_k", candidate_k)
        positive_int("rrf_k", rrf_k)
        if not isinstance(query_prefix, str):
            raise TypeError("query_prefix must be a string")

        manifest = snapshot.manifest
        embedding = manifest.get("embedding")
        if not isinstance(embedding, Mapping):
            raise ValueError("snapshot manifest is missing embedding identity")
        stored_config = embedding.get("config")
        if not isinstance(stored_config, Mapping):
            raise ValueError("snapshot manifest has an invalid embedding config")

        config_without_prefix = dict(stored_config)
        stored_query_prefix = config_without_prefix.pop("query_prefix", "")
        if (
            not isinstance(stored_query_prefix, str)
            or query_prefix != stored_query_prefix
        ):
            raise ValueError(
                "query_prefix does not match the persisted embedding configuration"
            )

        identity = _verified_query_embedding_identity(query_embed_fn)
        stored_revision = embedding.get("model_revision")
        if (
            identity["model_id"] != embedding.get("model_id")
            or not isinstance(stored_revision, str)
            or identity["model_revision"] != stored_revision.lower()
        ):
            raise ValueError(
                "query embedder model ID or immutable revision does not match snapshot"
            )
        if not _same_json_mapping(identity["config"], config_without_prefix):
            raise ValueError(
                "query embedder embedding configuration does not match snapshot"
            )

        vectors_config = manifest.get("vectors")
        if not isinstance(vectors_config, Mapping):
            raise ValueError("snapshot manifest is missing vector metadata")
        configured_dimension = config_without_prefix.get("dimension")
        if (
            configured_dimension is not None
            and configured_dimension != vectors_config.get("dimension")
        ):
            raise ValueError(
                "embedding configuration dimension does not match snapshot vectors"
            )

        sparse_config = manifest.get("sparse")
        if not isinstance(sparse_config, Mapping):
            raise ValueError("snapshot manifest has an invalid sparse configuration")
        sparse_keys = set(sparse_config)
        if sparse_keys not in (
            {"algorithm", "version"},
            {"algorithm", "version", "k1", "b"},
        ):
            raise ValueError("unsupported BM25 sparse configuration fields")
        version = sparse_config.get("version")
        if (
            sparse_config.get("algorithm") != "bm25"
            or isinstance(version, bool)
            or not isinstance(version, int)
            or version != 1
        ):
            raise ValueError("unsupported sparse index algorithm or version")
        k1 = sparse_config.get("k1", 1.5)
        b = sparse_config.get("b", 0.75)

        chunks = list(snapshot.chunks)
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if len(by_id) != len(chunks):
            raise ValueError("duplicate chunk_id in snapshot corpus")

        # Do not call __init__: it would embed all document chunks again.
        pipeline = cls.__new__(cls)
        pipeline.chunks = chunks
        pipeline._by_id = by_id
        pipeline.sparse = SparseIndex(chunks, k1=k1, b=b)
        pipeline.dense = DenseIndex.from_snapshot(
            snapshot,
            query_embed_fn=query_embed_fn,
            query_prefix=query_prefix,
        )
        pipeline.reranker = Reranker(score_fn) if score_fn else None
        pipeline.candidate_k = candidate_k
        pipeline.rrf_k = rrf_k
        pipeline.last_timing_ms = {"retrieval": 0.0, "rerank": 0.0}
        pipeline.last_candidate_ids = []
        return pipeline

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
