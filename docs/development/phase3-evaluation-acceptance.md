# Phase 3 — Evaluation Acceptance Protocol

**Status:** FROZEN — execution pending
**Objective:** Establish a reproducible retrieval-quality baseline, investigate failure patterns, measure dedicated query latency, and preserve verifiable provenance.

## 1. Scope and interpretation

Phase 3 is an evaluation and acceptance phase, not a retrieval-system rewrite.

Its objective is to validate the committed Phase 2 implementation using a fixed SciFact configuration.

Important limitations:

- SciFact has 5,183 documents and 300 test queries.
- Historical results from this test set have already been inspected.
- Therefore, the fresh run is a reproducibility and baseline refresh, not an untouched confirmatory experiment.
- No new retrieval tuning decisions should be justified solely by repeatedly examining this test set.
- Independent domain validation belongs to the subsequent validation programme.

Existing foundations:

- Per-query evaluation and aggregate consistency checks.
- Query/relevance fingerprints.
- Paired bootstrap comparisons.
- Bonferroni multiple-comparison correction.
- Benchmark provenance and source/configuration fingerprints.
- Ranking traces and previous candidate-depth sensitivity experiments.

## 2. Fixed dataset and model configuration

Dataset:

- Benchmark: BEIR SciFact test.
- Corpus: 5,183 documents.
- Queries: all 300.
- Level: document.
- Chunking: disabled; one searchable item per document.
- Corpus file: `data/scifact_test/corpus.jsonl`
- Queries file: `data/scifact_test/queries.json`

Expected input SHA-256:

- Corpus: `91af725e167cfbfcb272e77ee272233e255718b8c515e08a36cfa734c520fb97`
- Queries: `89f248368661e4fad34f07b388939c5149ed69e77911deec48e32a85bc069489`

Embedding:

- Model: `BAAI/bge-small-en-v1.5`
- Revision: `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`

Reranker:

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Revision: `233902d25c440f23af6f7d6e94d2946bac0bee0a`

Shared settings:

- Reranker batch size: 8.
- Reranker device: default/automatic; actual runtime metadata must be retained.
- Reranker maximum length: model default.
- BGE query prefix: enabled.
- Fake models: disabled.
- Raw query text: not included in the result artifact.
- Strategies: `bm25`, `dense`, `hybrid`, `hybrid_rerank`.

## 3. Quality evaluation

Configuration:

| Parameter | Value |
|---|---:|
| Queries | 300 |
| k | 10 |
| retrieval_k | 100 |
| candidate_k | 50 |
| Level | doc |
| Chunking | disabled |
| Embedding cache | disabled |
| Strategies | 4 |
| Expected strategy-query records | 1,200 |

Interpretation:

- k=10 is the top-10 metric cutoff.
- retrieval_k=100 is the document-ranking depth used for Recall@100.
- candidate_k=50 is the pre-reranking candidate pool.
- The evaluator searches to `max(retrieval_k, candidate_k)`.
- In document-level mode, `search_docs()` returns unique parent documents.

Run from `E:\ehs-phase3-acceptance` using the committed worktree code and the existing project virtual environment.

Exact command:

```cmd
E:\enterprise-hybrid-search\.venv\Scripts\python.exe scripts\run_eval.py --corpus data\scifact_test\corpus.jsonl --queries data\scifact_test\queries.json --level doc --no-chunk --k 10 --retrieval-k 100 --candidate-k 50 --no-cache --embedding-revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a --reranker-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a --reranker-batch-size 8 --strategies bm25 dense hybrid hybrid_rerank --output E:\ehs-phase3-results\phase3_acceptance_final_v1.json
```

Artifact:

`E:\ehs-phase3-results\phase3_acceptance_final_v1.json`

Output handling:

- Quality results are written using `Path.write_text()`.
- An existing output file can be overwritten.
- Verify the artifact path does not exist before starting.
- The parent result directory is created automatically.
- Use an absolute external output path so the evaluation worktree remains clean.

Acceptance:

- All four strategies evaluate all 300 queries.
- Exactly 1,200 strategy-query score records are present.
- Aggregate/per-query consistency passes.
- Query/relevance fingerprints match across strategies.
- Expected corpus and query-file hashes match.
- Pinned model revisions are verified in metadata.
- Git commit and source/configuration provenance are recorded.
- No unexplained missing query rows or provenance mismatches.

The evaluation-run timing is diagnostic only. Dedicated latency claims must come from Section 5.

## 4. Statistical comparison family

Source configuration:

`configs/analysis_family.json`

Declared comparisons:

1. BM25 vs Dense.
2. Dense vs Hybrid.
3. Hybrid vs Hybrid + Reranker.

Primary metrics:

- nDCG@10.
- Recall@100.

Statistical configuration:

- Six hypotheses.
- 10,000 paired bootstrap resamples.
- Random seed: 42.
- Family-wise alpha: 0.05.
- Bonferroni correction.
- MRR@10: descriptive only.

Exact command:

```cmd
E:\enterprise-hybrid-search\.venv\Scripts\python.exe scripts\compare_runs.py --a E:\ehs-phase3-results\phase3_acceptance_final_v1.json --pair bm25 dense --pair dense hybrid --pair hybrid hybrid_rerank --metrics ndcg@10 recall@100 --resamples 10000 --seed 42 --family-alpha 0.05
```

Report:

- Absolute paired effect sizes.
- Nominal confidence intervals.
- Bonferroni-adjusted confidence intervals.
- Query-level wins, losses and ties.
- Practical significance separately from statistical significance.

Supplementary diagnostic — HitRate@10:

- Derive it separately for each strategy from the quality artifact `per_query` records.
- For each query, Hit@10 is 1 when per-query `recall@10` is greater than zero; otherwise it is 0.
- HitRate@10 is the mean of Hit@10 across all evaluated queries. The expected denominator is 300.
- Report it as a supplementary diagnostic, not as a primary metric or part of the six-hypothesis statistical family.
- No evaluator or statistical-comparison implementation change is required.

Important limitations:

Hybrid-vs-Dense outcomes were previously inspected. The fresh analysis must not be described as a blind preregistered confirmatory test.

The comparison command prints its results. Preserve the console output in the final acceptance report.

## 5. Dedicated latency evaluation

Protocol:

| Parameter | Value |
|---|---:|
| Queries | 300 |
| Trials | 5 |
| Strategies | 4 |
| Timed samples per strategy | 1,500 |
| Total timed samples | 6,000 |
| Warm-up queries per strategy | 10 |
| Total warm-up invocations | 40 |
| Result depth | 100 documents |
| candidate_k | 50 |
| Level | doc |
| Chunking | disabled |

Execution semantics:

- The first ten queries are used to warm up every selected strategy.
- This happens once before the five timed trials.
- Warm-up calls are excluded from returned samples.
- Strategy order rotates deterministically across trials.
- Query order remains stable.
- Every trial evaluates all 300 queries for all four strategies.
- P50, P95 and P99 are calculated for total, retrieval and reranking times.

Cache policy:

- Document embeddings use `CachedEmbedder`.
- Query embeddings are not disk-cached.
- Document embeddings are loaded or generated while constructing the pipeline.
- Index construction occurs before `run_trials()`.
- Timed samples therefore represent query execution on an already constructed in-memory index.
- Cold index-build time is not included in these latency measurements.
- The document-cache hit/miss statistics must be retained.
- Use a dedicated, initially nonexistent cache directory.

The quality benchmark disables the document embedding cache, whereas the latency runner enables it. This is intentional and must be stated when comparing the artifacts.

Exact command:

```cmd
E:\enterprise-hybrid-search\.venv\Scripts\python.exe scripts\run_latency.py --corpus data\scifact_test\corpus.jsonl --queries data\scifact_test\queries.json --level doc --no-chunk --k 100 --candidate-k 50 --reranker-batch-size 8 --trials 5 --warmup-queries 10 --strategies bm25 dense hybrid hybrid_rerank --embedding-revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a --reranker-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a --cache-dir E:\ehs-phase3-cache-v1 --output E:\ehs-phase3-results\scifact_doclevel_latency_300q_5trials_v2.json
```

Artifact:

`E:\ehs-phase3-results\scifact_doclevel_latency_300q_5trials_v2.json`

Important:

- The latency runner refuses to start on a dirty Git working tree.
- The output path must not already exist.
- The dedicated cache directory must be initially absent for this protocol run.
- Both output and cache paths are outside the evaluation worktree.
- Preserve raw samples, model metadata, hardware metadata, cache statistics and Git provenance.

Acceptance:

- Exactly 1,500 samples per strategy.
- Exactly 6,000 timed samples total.
- Five completed trials.
- Forty warm-up calls excluded.
- Protocol validity reported as true.
- All three timing stages summarized.
- Hardware and runtime package versions retained.

## 6. Failure and ranking analysis

Inspect all 300 queries using the quality artifact and saved ranking traces.

Classify:

- Zero Recall@100.
- Positive Recall@100 but zero Recall@10.
- Low Recall@10.
- Hybrid losses versus Dense.
- Queries helped and harmed by reranking.
- Relevant documents absent from the candidate pool.
- Relevant documents present in candidates but ranked poorly.
- Document-level deduplication effects.
- Material ranking changes and outliers.

For each category report:

- Query count.
- Representative query IDs.
- Relevant document IDs.
- Candidate and final ranks.
- Whether failure occurred during retrieval, fusion or reranking.

Avoid unnecessary exposure of raw query text.

This is diagnostic analysis, not permission to repeatedly tune parameters against the inspected SciFact test set.

## 7. RRF and reranker interpretation

Document the actual observed behaviour, not assumptions.

RRF investigation:

- Compare sparse and dense rankings.
- Inspect consensus versus specialist documents.
- Attribute ranking losses to candidate generation and fusion.
- Separate candidate coverage failures from fusion ordering failures.

Reranker investigation:

- Compare pre-reranking and final ranks.
- Identify candidate recall limitations.
- Record relevance gains and regressions.
- Compare reranking time against total query latency.
- Evaluate cost/benefit, not nDCG alone.

Further parameter sweeps and model selection must use development/validation data, not repeated tuning on the inspected SciFact test set.

## 8. Provenance and reproducibility

Retain and review:

- Git commit SHA.
- Git dirty state.
- Source/configuration fingerprints.
- Corpus and query hashes.
- Query/relevance fingerprint.
- Requested and resolved model revisions.
- Python and package versions.
- Hardware metadata.
- Cache policy and cache statistics.
- Evaluation configuration.
- Exact commands.

Never replace or silently edit a completed evidence artifact. Use a new versioned output path for a new run.

If a run fails or an artifact is incomplete, preserve it for investigation and use a new artifact name for any rerun.

## 9. Final acceptance report

Create:

`docs/EVALUATION_ACCEPTANCE.md`

Include:

- Exact protocol and reproduction commands.
- Dataset and model provenance.
- Four-strategy aggregate results.
- Per-query consistency checks.
- Six paired comparisons.
- Full-query failure classification.
- RRF and reranker findings.
- Dedicated repeated latency results.
- Cache and hardware details.
- Statistical and practical limitations.
- Unresolved issues and recommended development experiments.

## 10. Exit criteria

Phase 3 closes only when:

1. Fresh quality artifact is retained.
2. All four strategies have 300 evaluated queries.
3. All 1,200 strategy-query records pass consistency checks.
4. Dataset and model provenance are verified.
5. All six declared statistical comparisons are reported.
6. All 300 queries receive failure classification.
7. RRF and reranker investigations are documented.
8. Dedicated latency artifact contains 6,000 timed samples.
9. Five-trial latency protocol is valid.
10. Final acceptance report is reproducible from recorded commands.

## 11. Boundary and next phase

Phase 3 does not establish:

- Independent domain validation.
- Universal search superiority.
- Grounded RAG answer quality.
- Claim-level citation correctness.
- Tenant isolation or ACL enforcement.
- Production scalability.
- Customer effectiveness.

These require separate evaluation and product-development stages.

The next validation programme must establish a new domain-specific dataset with human-verified relevance labels and an untouched holdout.

**Phase 3 objective: establish a trustworthy baseline, understand failures, preserve reproducible evidence, and make justified decisions about what to improve next.**
