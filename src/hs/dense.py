import numpy as np
from .validate import positive_int

def _normalize(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)

class DenseIndex:
    """Exact cosine search over chunk embeddings.

    ``embed_fn`` embeds document chunks during ordinary construction.
    ``normalized_embedding_matrix`` accepts an already validated, row-aligned
    matrix from a persistent index and skips document embedding generation.
    ``query_embed_fn`` still embeds each query; ``query_prefix`` is prepended
    to queries only (e.g. BGE's retrieval instruction).
    """
    def __init__(
        self,
        chunks,
        embed_fn=None,
        query_prefix="",
        query_embed_fn=None,
        normalized_embedding_matrix=None,
    ):
        self.chunks = list(chunks)
        if embed_fn is not None and not callable(embed_fn):
            raise TypeError("embed_fn must be callable or None")
        if query_embed_fn is not None and not callable(query_embed_fn):
            raise TypeError("query_embed_fn must be callable or None")

        self.embed_fn = embed_fn
        self.query_embed_fn = embed_fn if query_embed_fn is None else query_embed_fn
        self.query_prefix = query_prefix
        self.matrix = None

        if normalized_embedding_matrix is not None:
            if not self.chunks:
                raise ValueError("cannot provide embeddings for an empty corpus")
            if self.query_embed_fn is None:
                raise ValueError(
                    "query_embed_fn or embed_fn is required with precomputed embeddings"
                )

            try:
                matrix = np.asarray(normalized_embedding_matrix, dtype=np.float32)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "precomputed embedding matrix must be a numeric 2D array"
                ) from exc

            if (
                matrix.ndim != 2
                or matrix.shape[0] != len(self.chunks)
                or matrix.shape[1] == 0
            ):
                raise ValueError(
                    "precomputed embedding matrix must have shape "
                    f"({len(self.chunks)}, dimension>0); got {matrix.shape}"
                )
            if not np.isfinite(matrix).all():
                raise ValueError(
                    "precomputed embedding matrix must contain only finite values"
                )

            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                norms = np.linalg.norm(matrix, axis=1)
            if not np.isfinite(norms).all():
                raise ValueError("precomputed embedding matrix contains invalid norms")
            nonzero_rows = np.any(matrix != 0, axis=1)
            if np.any(nonzero_rows & (norms == 0)):
                raise ValueError(
                    "precomputed embedding matrix contains non-zero rows with zero norm"
                )
            if nonzero_rows.any() and not np.allclose(
                norms[nonzero_rows], 1.0, rtol=1e-5, atol=1e-6
            ):
                raise ValueError(
                    "precomputed embedding matrix must be L2-normalized"
                )

            # The loader returns a read-only C-contiguous float32 matrix. Keep
            # that view when possible: this path must not re-embed or renormalize
            # persisted document vectors.
            self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)

        elif self.chunks:  # some embedders return a 1-D empty array for [] input
            if self.embed_fn is None:
                raise ValueError(
                    "embed_fn is required when precomputed embeddings are not supplied"
                )
            if self.query_embed_fn is None:
                raise ValueError("query_embed_fn or embed_fn is required")

            m = np.asarray(
                self.embed_fn([c.text for c in self.chunks]), dtype=np.float32
            )
            if m.ndim != 2 or m.shape[0] != len(self.chunks):
                raise ValueError(f"embedder returned shape {m.shape} for {len(self.chunks)} chunks")
            if not np.isfinite(m).all():
                raise ValueError("embedder returned non-finite values")
            self.matrix = _normalize(m)

    @classmethod
    def from_snapshot(
        cls,
        snapshot,
        *,
        query_embed_fn,
        query_prefix="",
    ):
        """Construct from one validated snapshot to keep chunks and rows aligned."""
        from .index_store import LoadedIndexSnapshot

        if not isinstance(snapshot, LoadedIndexSnapshot):
            raise TypeError("snapshot must be a LoadedIndexSnapshot")
        return cls(
            snapshot.chunks,
            query_prefix=query_prefix,
            query_embed_fn=query_embed_fn,
            normalized_embedding_matrix=snapshot.vectors,
        )

    def search(self, query, k=10):
        positive_int("k", k)
        if self.matrix is None:
            return []
        q = np.asarray(self.query_embed_fn([self.query_prefix + query]), dtype=np.float32)
        if q.shape != (1, self.matrix.shape[1]):
            raise ValueError(f"query embedding shape {q.shape} != (1, {self.matrix.shape[1]})")
        if not np.isfinite(q).all():
            raise ValueError("embedder returned non-finite values for the query")
        sims = self.matrix @ _normalize(q)[0]
        order = np.argsort(-sims)[:k]
        return [(self.chunks[i].chunk_id, float(sims[i])) for i in order]
