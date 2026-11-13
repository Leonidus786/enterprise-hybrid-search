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

## ADR-007 Candidate depth sensitivity  [PROVISIONAL]
- Problem: hybrid_rerank reranks only candidate_k fused candidates; results depend on candidate depth.
- Decision: Keep candidate_k=50 as the current default for the optional reranker, but do not claim that 50 is optimal. hybrid_rerank remains opt-in and excluded from default evaluation strategies.
- Experiment status: COMPLETE for K=10, 20, and 50 on all 300 SciFact queries.
- Quality evidence: reranking did not improve aggregate MRR@10 or nDCG@10 at any tested depth. At K=20, Recall@10 was 0.84444 versus 0.83306 for Hybrid, while MRR@10 and nDCG@10 were lower. K=10 recall was unchanged; K=50 reranked recall was lower.
- Candidate Recall@K increased with candidate depth, but these depth-specific metrics are not interchangeable with Recall@100.
- Dedicated latency evidence: five trials, ten excluded warmups, 30 queries per trial, 150 samples per strategy, rotating strategy order, and uncached query embeddings. Within-run total P50 slowdown ratios were approximately 13.65x at K=10, 20.96x at K=20, and 49.7x at K=50.
- Qualification: K=10 used a cold document-embedding cache; K=20 and K=50 used a warm cache. Compare strategies within each depth only. No statistical-significance or cross-depth latency-trend claim is established.
- Model revisions matched cached local snapshots; loaded-model commit introspection was not independently verified. K=50 quality provenance recorded a dirty worktree and must retain that qualification.
- K=100 and K=200 were not tested. Revisit candidate depth only with target-corpus and target-hardware evidence showing a useful quality-cost trade-off.

## ADR-008 Stable ranking tie-breaking  [CONFIRMED]
- Problem: equal scores need deterministic ordering without changing ranking scores.
- Decision: Dense ranking uses stable argsort. Sparse ranking preserves input order for equal scores, RRF preserves first-seen order, and reranking preserves incoming order for equal scores. Do not introduce an arbitrary chunk-ID tie-breaker.
- Why: persistent lifecycle operations establish canonical document order; chunk generation is sequential; insert and update reconstruct sorted document and chunk order while aligning vector rows; delete preserves retained chunk order; snapshot serialization and loading preserve the stored chunk sequence.
- Boundary: the low-level build_index_snapshot function preserves caller order. Direct callers remain responsible for supplying deterministic input order.
- Evidence: four equal-score regression tests passed; final full suite passed 595 tests.

## ADR-009 Evaluation and latency configuration freeze  [CONFIRMED]
- Supported evaluation strategies are bm25, dense, hybrid, and hybrid_rerank. Default evaluation strategies are bm25, dense, and hybrid; reranking remains opt-in.
- The canonical quality-evaluation protocol is BEIR SciFact test, document-level, no chunking, and no persistent evaluation cache.
- The latency manifest records all 19 run_latency parser defaults, including pinned model revisions. Its required output path is excluded because it has no default.
- Evaluation CLI revision defaults remain None; the dedicated latency runner defaults to pinned embedding and reranker revisions. These are intentionally distinct contracts.
- Evidence: configuration contract tests passed; final full suite passed 595 tests.
