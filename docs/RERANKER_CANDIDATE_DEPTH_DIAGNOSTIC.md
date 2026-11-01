## Cross-Encoder Candidate-Depth Diagnostic (Exploratory)

**Run date:** 2026-11-01
**Dataset:** BEIR SciFact test, document-level, no chunking
**Scope:** First 30 of 300 queries; `k=10`, `retrieval_k=100`; one evaluation pass per query and strategy
**Strategies:** `hybrid` and `hybrid_rerank`
**Models:** `BAAI/bge-small-en-v1.5` revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; `cross-encoder/ms-marco-MiniLM-L-6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`

This is a **30-query exploratory candidate-depth experiment**, not the full held-out evaluation and not the dedicated repeated-trial latency benchmark. The three runs use the same first 30 queries with candidate depths 10, 20 and 50. Each run reported per-query consistency PASS for 60 strategy-query records.

| Candidate K | Candidate Recall@K | Hybrid Recall@10 | Reranked Recall@10 | Hybrid MRR@10 | Reranked MRR@10 | Δ MRR@10 | Hybrid nDCG@10 | Reranked nDCG@10 | Δ nDCG@10 | Reranker P50 / P95 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.843 | 0.843 | 0.843 | 0.781 | 0.679 | -0.102 | 0.787 | 0.721 | -0.066 | 1,166.90 / 1,363.10 |
| 20 | 0.893 | 0.843 | 0.853 | 0.781 | 0.681 | -0.100 | 0.787 | 0.723 | -0.064 | 2,576.71 / 3,147.21 |
| 50 | 0.967 | 0.843 | 0.820 | 0.781 | 0.675 | -0.106 | 0.787 | 0.711 | -0.076 | 7,279.13 / 9,428.28 |

| Candidate K | Hybrid total P50 / P95 (ms) | Hybrid+reranker total P50 / P95 (ms) | Reranker P50 (ms) |
|---:|---:|---:|---:|
| 10 | 40.24 / 59.00 | 1,226.86 / 1,423.38 | 1,166.90 |
| 20 | 30.96 / 45.22 | 2,635.99 / 3,222.66 | 2,576.71 |
| 50 | 57.41 / 96.20 | 7,371.85 / 9,538.53 | 7,279.13 |

### Interpretation

- Candidate Recall@K rises as the candidate pool grows: 0.843 at K=10, 0.893 at K=20 and 0.967 at K=50.
- On this 30-query subset, reranking lowers MRR@10 and nDCG@10 at all three candidate depths.
- Recall@10 is unchanged at K=10, improves by 0.010 at K=20, and falls by 0.023 at K=50. The small K=20 recall gain does not offset the lower MRR/nDCG and the large inference cost.
- Reranker median latency increases from about 1.17 s at K=10 to 2.58 s at K=20 and 7.28 s at K=50 on the recorded CPU environment.
- Current decision: keep Hybrid as the default and Cross-Encoder reranking disabled by default. Do not remove the reranker until its scoring path and optimization options have been profiled.

### Methodology and limitations

- `--max-queries 30` selects the first 30 queries; this is not a random or independently held-out sample. Do not generalize the results to all 300 queries or other domains.
- These were single-pass `run_eval.py` latency summaries, not five repeated trials. Do not merge these latency numbers into the dedicated latency benchmark table.
- The run used the default persistent embedding cache (`.cache/embeddings`); each run reported 5,245 cache hits and zero misses. Retrieval latency is therefore not a cold/uncached comparison.
- Model loading is outside the per-query search timer; the scorer is initialized before evaluation. These numbers represent query-time inference/search, not application cold start.
- The results do not establish statistical significance. Use the full per-query outputs for paired analysis before making broader quality claims.

### Reproduction commands

```cmd
uv run python scripts\run_eval.py --corpus data\scifact_test\corpus.jsonl --queries data\scifact_test\queries.json --k 10 --retrieval-k 100 --candidate-k 10 --level doc --no-chunk --max-queries 30 --strategies hybrid hybrid_rerank --embedding-revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a --reranker-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a --output results\rerank_candidate_k10_30q_v1.json

uv run python scripts\run_eval.py --corpus data\scifact_test\corpus.jsonl --queries data\scifact_test\queries.json --k 10 --retrieval-k 100 --candidate-k 20 --level doc --no-chunk --max-queries 30 --strategies hybrid hybrid_rerank --embedding-revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a --reranker-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a --output results\rerank_candidate_k20_30q_v1.json

uv run python scripts\run_eval.py --corpus data\scifact_test\corpus.jsonl --queries data\scifact_test\queries.json --k 10 --retrieval-k 100 --candidate-k 50 --level doc --no-chunk --max-queries 30 --strategies hybrid hybrid_rerank --embedding-revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a --reranker-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a --output results\rerank_candidate_k50_30q_v1.json
```

**Artifact policy:** keep the three raw JSON files in the local `results/` directory (currently Git-ignored) and attach them to a versioned GitHub Release or publish them through a deliberate benchmark-artifact policy. Do not force-add the whole `results/` directory.
