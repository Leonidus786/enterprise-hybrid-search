# Decisions (ADRs)
Status values: PROVISIONAL (no measured evidence yet) / CONFIRMED (evidence filled in from YOUR benchmark).
Never write a number here you did not measure.

## ADR-001 Fusion: RRF vs raw score addition  [PROVISIONAL]
- Problem: BM25 scores (unbounded) and cosine (-1..1) are on different scales.
- Options: raw sum; min-max normalised weighted sum; RRF.
- Decision: RRF, score(d) = sum_i 1 / (k + rank_i(d)), k = 60.
- Why: needs no score calibration; robust when one retriever is weak on a query.
- Limitation: ignores score magnitude (a huge BM25 margin counts the same as a tiny one).
- Evidence: TODO bm25 / dense / hybrid MRR + nDCG@10 on my labeled queries.
- Revisit if: weighted fusion beats RRF on my query set.

## ADR-002 BM25 formula  [CONFIRMED by hand-computed unit tests]
- idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))   (Lucene variant, always > 0)
- score(d,q) = sum_t idf(t) * f * (k1 + 1) / (f + k1 * (1 - b + b * |d| / avgdl)),  k1 = 1.5, b = 0.75
- Why Lucene idf: the classic Robertson idf hits 0 / goes negative when a term is in >= half the docs,
  which silently dropped valid matches in a small corpus (found by a failing test).
- Tokenizer: lowercase, \w+ . Limitation: no stemming / stopwords / multilingual handling.

## ADR-003 Evaluation protocol: doc-level vs chunk-level  [CONFIRMED by regression test]
- Bug found in review: taking top-k CHUNKS then deduping to docs lets one doc with many strong chunks
  crowd every other doc out of the top-k (relevant doc B never seen -> recall 0.0).
- Decision: doc-level eval ranks all chunks, dedupes to unique docs, then cuts to k (`search_docs`).
  Chunk-level eval is a separate protocol (`level="chunk"`) with chunk-id labels. Never mix them.
- Limitation: hybrid_rerank only sees `candidate_k` fused chunks, so it can return < k docs.

## ADR-004 Dense search: exact NumPy cosine  [PROVISIONAL]
- Fine for <= ~100k chunks (brute force). Beyond that: FAISS HNSW / Qdrant. Not needed for the benchmark.

## ADR-005 Dependency injection of embedder and cross-encoder  [CONFIRMED]
- Tests run offline in milliseconds with fakes; real models only in `models.py` / `scripts/run_eval.py`.
- Cost: fake-model results say nothing about retrieval quality. Only real-model runs count as evidence.

## ADR-006 BGE query instruction  [PROVISIONAL]
- bge-*-en-v1.5 retrieval uses a query-side prefix ("Represent this sentence for searching relevant passages: ").
- Decision: prefix queries only, never documents (`query_prefix`). Evidence: TODO compare with/without on my queries.

## ADR-007 candidate_k sensitivity  [PROVISIONAL]
- hybrid_rerank only reranks `candidate_k` fused chunks (default 50). If many of them belong to one doc, doc-level
  eval can return < k docs, so its numbers depend on candidate_k.
- Experiment to run: `python scripts/run_eval.py --candidate-k 20|50|100|200`, record MRR/nDCG and p95 latency for each.
- Evidence: TODO (the cost/quality curve is the interview answer to "why 50?").
