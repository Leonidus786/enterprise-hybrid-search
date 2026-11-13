

# Phase 3 - Evaluation Acceptance

**Status:** IN PROGRESS  
**Objective:** Validate retrieval quality, statistical comparisons, failure patterns, latency, and reproducibility.

## 1. Existing foundations

Already implemented:
- Per-query evaluation and aggregate consistency checks.
- Query/relevance fingerprints and paired bootstrap.
- Bonferroni multiple-comparison correction.
- Benchmark provenance and source/configuration fingerprints.
- Ranking traces and candidate-depth sensitivity experiments.

Phase 3 is evidence and acceptance work, not a rewrite.

## 2. Final quality evaluation

Protocol:
- Dataset: BEIR SciFact test.
- Corpus: 5,183 documents.
- Queries: all 300.
- Level: document.
- Chunking: disabled.
- k=10, retrieval_k=100, candidate_k=50.
- Cache: disabled.
- Strategies: bm25, dense, hybrid, hybrid_rerank.
- Embedding revision: 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
- Reranker revision: 233902d25c440f23af6f7d6e94d2946bac0bee0a

Acceptance:
- 300/300 queries per strategy.
- Aggregate/per-query consistency passes.
- Same query/relevance fingerprint across strategies.
- Input hashes and model revision metadata recorded.
- Git state and source/configuration fingerprints reviewed.
- No unexplained missing query rows or provenance mismatches.

Proposed artifact: results/phase3_acceptance_final_v1.json

## 3. Statistical comparison family

Use configs/analysis_family.json.

- BM25 vs Dense.
- Dense vs Hybrid.
- Hybrid vs Hybrid+Reranker.
- Primary metrics: nDCG@10 and Recall@100.
- Six hypotheses.
- Bonferroni family alpha: 0.05.
- Resamples: 10,000.
- Seed: 42.
- MRR@10: descriptive only.

Important: hybrid-vs-dense results were previously inspected. This analysis is not preregistered and must not be described as a blind confirmatory test.

Report effect sizes, nominal intervals, adjusted intervals, and wins/losses/ties. Statistical significance does not establish practical importance.

## 4. Failure analysis

Inspect:
- Hybrid losses against Dense.
- Queries affected by reranking.
- Zero-recall and low-recall queries.
- Candidate recall versus final ranking quality.
- Missing relevant documents and rank changes.
- Document-level deduplication effects.
- Per-query outliers and malformed records.

Use query IDs and saved traces. Avoid unnecessarily exposing raw query text.

## 5. Dedicated latency evaluation

Use scripts/run_latency.py.

- At least five trials.
- Ten warmups excluded.
- Rotating strategy order.
- Uncached query embeddings.
- Record document-cache state and hardware.
- Report total, retrieval, and reranking P50/P95/P99.
- Use consistent cache conditions for strategy comparisons.
- Do not mix evaluation-run timing with dedicated latency claims.

Preserve the raw latency JSON artifact.

## 6. Final acceptance report

Create docs/EVALUATION_ACCEPTANCE.md with:
- Dataset and model provenance.
- Exact evaluation protocol.
- Aggregate and per-query results.
- Paired comparisons and correction.
- Failure analysis.
- Repeated latency results.
- Reproducibility commands.
- Limitations and unresolved issues.

## 7. Exit criteria

Phase 3 closes only when:
1. Final quality and latency artifacts are retained.
2. Provenance and all 300-query consistency checks pass.
3. All six declared comparisons are reported.
4. Failure analysis is complete.
5. The report is reproducible from recorded commands.

**Boundary:** Phase 3 does not establish enterprise readiness, tenant isolation, ACL enforcement, production scalability, or customer effectiveness. Grounded RAG and application delivery remain subsequent phases.



















