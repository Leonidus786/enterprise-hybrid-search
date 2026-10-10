# Architecture

    query ─┬─> SparseIndex (BM25) ──┐
           │                        ├─> RRF fusion ─> top candidate_k chunks ─> Reranker (cross-encoder) ─> top-k
           └─> DenseIndex (cosine) ─┘

| Module | Responsibility |
|---|---|
| chunking | text -> overlapping word-window `Chunk(chunk_id, doc_id, text)` |
| sparse | BM25 (Lucene idf) over chunks |
| dense | exact cosine search; embedder injected; optional query prefix (BGE) |
| fusion | Reciprocal Rank Fusion (rank-based, scale-free) |
| rerank | cross-encoder re-scoring of fused candidates; scorer injected |
| pipeline | 4 modes: bm25 / dense / hybrid / hybrid_rerank; `search` (chunks), `search_docs` (unique docs) |
| metrics / evaluate | P@K, R@K, MRR, nDCG@K, latency p50/p95; doc-level and chunk-level protocols |
| models | real sentence-transformers wrappers (lazy import) |

Rules: models are injected (tests need no downloads); ranking logic never lives in IO/CLI code.

Validation (all covered by tests, shared helpers in `validate.py`): count params (`k`, `top_n`, `candidate_k`) must be real
positive ints; float params (`k1`, `b`, RRF constant) must be finite; embedder / scorer output shape and finiteness are
checked; corpus and query files are schema-checked with line numbers; relevance labels must exist in the corpus.
Not validated: text quality, language, duplicate *content* across docs, embedder semantic correctness.
