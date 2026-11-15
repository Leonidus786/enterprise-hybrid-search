# Phase 3 — Evaluation and Retrieval Quality Acceptance

**Status:** Evidence assembled; full regression passed; final diff review and owner acceptance pending.

**Evaluation worktree:** `E:\ehs-phase3-acceptance`

**Repository result directory:** `results/`

**Evaluation commit:** `0f3b7c03f2c1afd4e49c8d9a0b6a54e4ba23cbca`
**Dataset:** BEIR SciFact test; document-level; 5,183 documents; 300 queries

## 1. Executive summary

Phase 3 establishes a reproducible retrieval-quality baseline for BM25, dense retrieval, Hybrid (BM25 + dense + RRF), and Hybrid + MS MARCO Cross-Encoder reranking. It also documents query-level failure classes, RRF diagnostics, candidate-depth sensitivity, reranker truncation, repeated latency, model/runtime provenance, and the next independent-validation design.

**Current decision:** Hybrid remains the default retrieval strategy. The MS MARCO Cross-Encoder remains opt-in/default OFF: its observed top-10 quality point estimates are lower than Hybrid, Recall@100 and Candidate Recall@50 are unchanged, and repeated CPU latency is materially higher. MedCPT is a promising exploratory challenger on the same fixed 50-candidate pools, but it is not selected as a production default because this comparison uses previously inspected SciFact queries and lacks independent-domain and dedicated serving-latency validation.

This report does not claim independent confirmatory quality, production deployment, universal search superiority, grounded-answer correctness, citation correctness, or enterprise security readiness.

## 2. Scope, protocol and dataset integrity

The frozen execution protocol is `docs/development/phase3-evaluation-acceptance.md` (SHA-256 `cb3f608123bdfdb98a59f5e34331ca1f29773574c5af868742b9a67f2db3c009`). The quality artifact records:

- Corpus: `data\scifact_test\corpus.jsonl`; 5,183 searchable documents; SHA-256 `91af725e167cfbfcb272e77ee272233e255718b8c515e08a36cfa734c520fb97`.
- Queries: `data\scifact_test\queries.json`; 300 evaluated queries; file SHA-256 `89f248368661e4fad34f07b388939c5149ed69e77911deec48e32a85bc069489`.
- Query/relevance fingerprint: `8674ce8b992faba06794d0bec7cb806a99373eecc5466372aaf829741ec2f738`.
- Document-level retrieval; chunking disabled; `k=10`, `retrieval_k=100`, `candidate_k=50`.
- Quality run Git commit `0f3b7c03f2c1afd4e49c8d9a0b6a54e4ba23cbca`; saved artifact says working tree dirty = `False`.

The 300 SciFact test queries had been inspected previously. The fresh run is therefore a reproducibility and baseline refresh, **not** an untouched confirmatory test. Repeated inspection of this set must not be used as the sole basis for further tuning.

### Metric definitions

The implementation in `src/hs/metrics.py` is authoritative:

- Precision@10: relevant results in the first 10 divided by 10.
- Recall@10 and Recall@100: relevant results retrieved in the respective cutoff divided by the number of relevant documents.
- MRR@10: reciprocal rank of the first relevant result within the first 10.
- nDCG@10: **binary-relevance** nDCG, not graded nDCG.
- Candidate Recall@50: fraction of relevant documents present in the candidate pool of 50.
- HitRate@10 below is supplementary/descriptive: proportion of queries with at least one relevant result in top 10.

## 3. Four-strategy quality results

| Strategy | Precision@10 | Recall@10 | Recall@100 | Candidate Recall@50 | MRR@10 | nDCG@10 | HitRate@10* |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.0867 | 0.7816 | 0.8859 | 0.8704 | 0.6340 | 0.6646 | 0.8000 |
| dense | 0.0937 | 0.8296 | 0.9433 | 0.9283 | 0.6725 | 0.7057 | 0.8400 |
| hybrid | 0.0933 | 0.8331 | 0.9617 | 0.9406 | 0.6765 | 0.7103 | 0.8467 |
| hybrid_rerank | 0.0920 | 0.8211 | 0.9617 | 0.9406 | 0.6568 | 0.6901 | 0.8367 |

*HitRate@10 is derived from per-query `recall@10 > 0`; it is not part of the six-hypothesis statistical family.*

Hybrid and Hybrid + reranker tie for the highest observed Recall@100 and Candidate Recall@50. Hybrid has better observed top-10 ranking metrics than the reranked variant. Its nDCG@10 is only slightly above Dense in point estimate; the frozen paired comparison does not establish a statistically significant Dense-to-Hybrid nDCG improvement after multiplicity correction.

## 4. Paired statistical comparisons — frozen six-test family

The primary statistical family contains six paired comparisons: BM25 vs Dense, Dense vs Hybrid, and Hybrid vs Hybrid + reranker, each for nDCG@10 and Recall@100. The analysis uses 10,000 paired query bootstrap resamples, seed 42, and Bonferroni-adjusted confidence intervals (99.1667% confidence). MRR is descriptive, not part of this six-test family.

Delta is `strategy b - strategy a`. W/L/T are query-level wins/losses/ties for b versus a.

| Comparison | Metric | Mean delta | Nominal 95% CI | Bonferroni-adjusted CI | W/L/T | Significant after adjustment? |
|---|---|---:|---:|---:|---:|---|
| bm25 → dense | ndcg@10 | +0.04106 | [+0.00441, +0.07793] | [-0.00831, +0.09131] | 82/54/164 | No |
| bm25 → dense | recall@100 | +0.05744 | [+0.02133, +0.09445] | [+0.00930, +0.10689] | 28/8/264 | Yes |
| dense → hybrid | ndcg@10 | +0.00462 | [-0.02338, +0.03289] | [-0.03496, +0.04268] | 59/55/186 | No |
| dense → hybrid | recall@100 | +0.01833 | [-0.00333, +0.04000] | [-0.01000, +0.05000] | 9/3/288 | No |
| hybrid → hybrid_rerank | ndcg@10 | -0.02021 | [-0.04962, +0.00953] | [-0.06089, +0.01891] | 55/62/183 | No |
| hybrid → hybrid_rerank | recall@100 | +0.00000 | [+0.00000, +0.00000] | [+0.00000, +0.00000] | 0/0/300 | No |

Only BM25 → Dense Recall@100 excludes zero after the frozen correction. The other five comparisons do not establish an adjusted statistically significant difference. This does not imply that strategies are equivalent; it means the available sample and uncertainty intervals do not establish the tested effect at the declared family-wise error rate.

## 5. All-query failure analysis

Every strategy-query record is assigned to exactly one mutually exclusive class. Counts below sum to 300 for each strategy. Definitions are preserved in `phase3_failure_analysis_v1.json`.

| Strategy | Top-10 hit | Relevant in candidate pool, missed top 10 | Candidate pool miss, relevant in top 100 | No relevant top-100 hit | Total |
|---|---:|---:|---:|---:|---:|
| bm25 | 240 | 24 | 4 | 32 | 300 |
| dense | 252 | 28 | 4 | 16 | 300 |
| hybrid | 254 | 29 | 6 | 11 | 300 |
| hybrid_rerank | 251 | 32 | 6 | 11 | 300 |

For Hybrid, six queries have relevant documents in ranked positions 51–100 but outside the 50-candidate pool: `1, 421, 870, 914, 1175, 1191`. Eleven queries have no relevant document in the top 100: `437, 502, 560, 716, 887, 913, 975, 1110, 1199, 1213, 1332`. The same 50-candidate pool is used by Hybrid and Hybrid + reranker, so reranking cannot repair a candidate-generation miss. These cases require retrieval/candidate-generation improvements, not a different ordering of the same candidates.

Independent recomputation of the stored ranking traces previously matched all 1,200 query-level recall values; candidate IDs were verified as a subset of their corresponding top-100 rankings. The RRF diagnostic checks are described next.

## 6. RRF and candidate-depth investigation

### RRF diagnostic

The saved RRF diagnostic covers 11 selected failure-analysis queries (`437, 502, 560, 716, 887, 913, 975, 1110, 1199, 1213, 1332`), with RRF k = 60. The corrected comparison verifies:

- Raw RRF ranking equals the `hybrid` (pre-rerank) top-100 trace for all 11 diagnostic queries.
- Candidate IDs equal the Hybrid candidate pool for all 11 queries.
- The raw RRF ranking differs from `hybrid_rerank` for all 11 queries; this is the expected effect of the subsequent Cross-Encoder stage, not an RRF mismatch.

This is an 11-query diagnostic, not a claim that every sparse and dense component ranking was separately retained and reconstructed for all 300 queries. The main quality artifact still contains the full 300-query final rankings and candidate traces.

### Candidate-depth sensitivity (Phase 2 frozen evidence)

The candidate-depth study is preserved in `docs/development/phase2-freeze.md` (SHA-256 `d210bb009be1e559855afe0b2c93d011a2136b585a28fcaac9ec9d3fdb87d62f`). It is a **sensitivity study, not a pure reranker-only ablation**, because changing candidate depth also changes the upstream candidate pool. The frozen results are reproduced below:

## Candidate-depth experiment

The experiment is a **candidate-depth sensitivity study**, not a pure reranker-only ablation: changing candidate depth also changes the upstream candidate pool. Each quality run evaluated 300/300 SciFact queries at document level with `k=10` and `retrieval_k=100`, using pinned embedding and reranker revisions.

### Quality observations

| Candidate K | Strategy | Recall@10 | MRR@10 | nDCG@10 | Candidate Recall@K |
|---:|---|---:|---:|---:|---:|
| 10 | Hybrid | 0.833 | 0.677 | 0.710 | 0.833 |
| 10 | Hybrid + reranker | 0.833 | 0.663 | 0.698 | 0.833 |
| 20 | Hybrid | 0.83306 | 0.67655 | 0.71030 | 0.89022 |
| 20 | Hybrid + reranker | 0.84444 | 0.66319 | 0.70037 | 0.89022 |
| 50 | Hybrid | 0.833 | 0.677 | 0.710 | 0.941 |
| 50 | Hybrid + reranker | 0.821 | 0.657 | 0.690 | 0.941 |

**Interpretation:**

- Reranking did not improve aggregate MRR@10 or nDCG@10 at K=10, 20, or 50.
- K=20 showed a small observed Recall@10 increase, while MRR@10 and nDCG@10 decreased.
- K=10 showed unchanged Recall@10 and lower MRR/nDCG after reranking.
- K=50 showed lower Recall@10, MRR@10, and nDCG@10 after reranking.
- Candidate Recall@K is depth-specific. The K=10, K=20, and K=50 values are not interchangeable with one another or with Recall@100.
- These are observed aggregate results, not proof of statistical significance or generalization to other corpora.

### Dedicated latency observations

The dedicated protocol used 5 trials, 10 warmups excluded from samples, 30 queries per trial (150 timed samples per strategy), rotating strategy order, and uncached query embeddings.

| Candidate K | Hybrid total P50 | Hybrid + reranker total P50 | Approximate P50 ratio |
|---:|---:|---:|---:|
| 10 | 150.17 ms | 2,048.87 ms | 13.65× |
| 20 | 80.58 ms | 1,688.91 ms | 20.96× |
| 50 | 78.29 ms | 3,891.20 ms | 49.7× |

**Latency qualification:** K=10 used a cold document-embedding cache, whereas K=20 and K=50 used a warm document-embedding cache. Compare Hybrid and Hybrid+reranker only within the same K run. Do not infer a candidate-depth latency trend by comparing absolute values across K.

### Model and provenance qualification

The embedding revision was `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; the reranker revision was `233902d25c440f23af6f7d6e94d2946bac0bee0a`. The available verification status was a match to the local cached snapshot (`verified_cached_snapshot_match`), not an independently exposed loaded-model commit hash.

The K=50 quality artifact recorded `working_tree_dirty=true`. The known tracked-source equivalence check found normalized content equivalent to the benchmark commit; raw hash differences included CRLF/LF line-ending normalization. This artifact must not be described as a clean-worktree run. K=10/K=20 quality and the dedicated latency runs were produced from clean detached worktrees.

Interpretation: at K=10, 20 and 50, reranking did not improve aggregate MRR@10 or nDCG@10. K=20 had a small observed Recall@10 increase but lower MRR/nDCG. K=50 showed lower Recall@10, MRR and nDCG. These are descriptive aggregate results, not statistical proof or evidence of generalization.

The Phase 2 freeze records that its K=50 quality artifact had `working_tree_dirty=true`; a normalized tracked-source equivalence check was documented, including CRLF/LF normalization. It must not be represented as a clean-worktree benchmark. The K=10/K=20 quality and dedicated latency runs were recorded as clean detached worktrees.

## 7. Cross-Encoder truncation investigation

The truncation analyses are derived from saved tokenization/ranking evidence; `inference_performed=false` in these audits. They are descriptive and **do not establish a causal effect** of truncation on ranking.

- 15,000 query-document pairs audited; maximum pair length 512; `longest_first` truncation; 2,197 pairs truncated, all on the document side; 260,742 tokens dropped.
- Raw pair-token length: P50 366, P90 550, P95 625, P99 851, maximum 1,970.
- Across 300 queries, truncated candidate counts: zero for 13 queries, 1–5 for 145, and 6+ for 142.
- Association between truncated-candidate count and nDCG delta was weak and not statistically significant (Spearman rho -0.0829, p=0.152); MRR association was rho -0.1021, p=0.0774.
- Among 320 relevant query-document pairs, 55 were truncated: 9 improved, 10 worsened, 36 unchanged. In top-10 outcomes for these truncated relevant pairs: 1 gained, 3 lost, 48 retained; the remaining pairs were not top-10 outcomes.

These findings justify investigating input-length handling, but not claiming truncation caused the observed reranker regression. Any mitigation must be evaluated on development/independent validation data.

## 8. Dedicated repeated latency evaluation

Canonical latency artifact: `phase3_latency_doclevel_v1.json`. This is the **document-level, no-chunking** run. The separate `phase3_latency_final_v1.json`, if present, is a chunked experiment and is not used as the canonical latency result here.

Protocol: 5 trials × 300 queries × 4 strategies = 6,000 samples; 1,500 samples per strategy; 10 warm-up queries per strategy excluded from timed samples; strategy order rotates by trial; query-embedding cache disabled; document embeddings cached. The artifact marks the protocol valid.

| Strategy | Samples | Total P50 (ms) | Total P95 (ms) | Total P99 (ms) | Retrieval P50 (ms) | Rerank P50 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 1500 | 35.91 | 74.44 | 100.13 | 35.75 | 0.00 |
| dense | 1500 | 54.31 | 96.20 | 144.98 | 54.15 | 0.00 |
| hybrid | 1500 | 106.66 | 186.10 | 273.54 | 106.38 | 0.00 |
| hybrid_rerank | 1500 | 4915.33 | 8512.31 | 9599.96 | 101.14 | 4800.62 |

Hybrid + MS MARCO reranker total latency is approximately 46.08× Hybrid at p50, 45.74× at p95, and 35.10× at p99. Its reranker-stage p50 is approximately 4,800.62 ms. These are local CPU measurements, not a deployed-service SLO. The outer total timer is authoritative; retrieval/rerank component timers are diagnostic and may not sum exactly to total because of timer boundaries and residual work.

## 9. Reranker decision and MedCPT exploratory comparison

### MS MARCO Cross-Encoder decision

Decision artifact: `phase3_reranker_decision_v1.json`. Keep the MS MARCO Cross-Encoder **default OFF** and retain it only as an optional experiment. All four observed top-10 point estimates (Precision@10, Recall@10, MRR@10, nDCG@10) are lower than Hybrid; Recall@100 and Candidate Recall@50 are unchanged; the paired nDCG comparison is not statistically significant after the frozen six-test correction; and CPU latency is substantially higher. The evidence does not prove a statistically significant quality decline, but it does not demonstrate a quality benefit that justifies the measured latency cost.

### MedCPT versus MS MARCO on fixed candidates

MedCPT A/B is exploratory/non-confirmatory. It reranks the same 50 Hybrid candidate IDs for each of the 300 previously inspected SciFact queries; it does not improve candidate generation. The paired statistics below use the exact evaluator definitions (including binary nDCG@10), 10,000 query-paired bootstrap resamples, seed 20261115, and Bonferroni correction across four metrics (98.75% confidence intervals).

| Metric | MS MARCO | MedCPT | Delta | Nominal 95% CI | Bonferroni-adjusted 98.75% CI | W/L/T |
|---|---:|---:|---:|---:|---:|---:|
| precision@10 | 0.0920 | 0.1013 | +0.00933 | [+0.00533, +0.01333] | [+0.00433, +0.01500] | 32/5/263 |
| recall@10 | 0.8211 | 0.9029 | +0.08178 | [+0.04511, +0.12011] | [+0.03733, +0.13122] | 32/5/263 |
| mrr@10 | 0.6568 | 0.7679 | +0.11116 | [+0.07881, +0.14527] | [+0.06945, +0.15495] | 82/22/196 |
| ndcg@10 | 0.6901 | 0.7961 | +0.10597 | [+0.07712, +0.13668] | [+0.06864, +0.14468] | 89/24/187 |

Mean Candidate Recall@50 was 0.94056 for the fixed candidate pool. All four adjusted intervals exclude zero in this exploratory sample. This is a promising ranking signal, not independent confirmation or a production selection. MedCPT inference took 7396.02 seconds in the recorded run; dedicated repeated serving latency and compute-cost evaluation remains required before adoption.

## 10. Model, runtime and provenance reconciliation

The quality and canonical latency runs share:

- Corpus SHA-256 `91af725e167cfbfcb272e77ee272233e255718b8c515e08a36cfa734c520fb97`.
- Query/relevance SHA-256 `89f248368661e4fad34f07b388939c5149ed69e77911deec48e32a85bc069489`.
- Query/relevance fingerprint `8674ce8b992faba06794d0bec7cb806a99373eecc5466372aaf829741ec2f738`.
- Git commit `0f3b7c03f2c1afd4e49c8d9a0b6a54e4ba23cbca`, with both artifacts recording clean worktrees.
- Identical source-code fingerprint across 27 files: `62c558d16779321dd003b66e15d66d1e899ee9f299ad42f40c57de137bf96c33`.

Model revisions:

| Model | Requested/resolved revision | Verification |
|---|---|---|
| BAAI/bge-small-en-v1.5 | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | `verified_cached_snapshot_match` |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | `233902d25c440f23af6f7d6e94d2946bac0bee0a` | `verified_cached_snapshot_match` |

Revision status is a match to the local cached snapshot. `verified_from_loaded_model=false`; therefore this report does not claim loaded-model object attestation.

Runtime recorded by the latency artifact: processor `Intel64 Family 6 Model 140 Stepping 1, GenuineIntel`, logical CPUs `8`, CUDA available `False`; package versions: `{"huggingface-hub": "1.33.0", "numpy": "2.5.3", "sentence-transformers": "6.1.0", "torch": "2.14.1", "transformers": "5.19.0"}`.

**Prefix limitation:** quality records `bge_prefix_enabled=true`; latency does not record this flag or the exact invocation. Exact prefix parity between historical quality and latency runs is unverified. The provenance reconciliation explicitly retains this limitation.

**Cache qualification:** quality and latency configurations are not identical. The latency run intentionally disables query-embedding cache and enables document-embedding cache. Interpret latency under this measured configuration; do not imply identical cache behavior between quality and latency runs.

## 11. Independent validation design — Step 3.13

Design document: `docs/development/independent-validation-design.md` (SHA-256 `e354dc9ebca953ff7d181938468c98c7fbb2ff4d57ab9d114027f802ce41391e`). Status: **DESIGN DRAFT — pending domain selection and preregistration**.

The design specifies an approved domain corpus, query-family separation, deduplication/leakage controls, human double annotation and adjudication, a predeclared binary-nDCG primary metric, coverage guardrails, paired statistics, pilot-based sample-size determination, latency/cost gates, and a separately governed untouched holdout. It does **not** claim that the corpus has been collected, labels adjudicated, sample size powered, or an independent test executed.

Required decisions before preregistration include target domain/use cases, corpus owner and permissions, query sampling frame, annotation/adjudication resources, sample sizes, practical effect and non-inferiority thresholds, latency SLO, and independent holdout custody.

## 12. Limitations and non-claims

1. SciFact's 300 test queries were previously inspected. The refresh is reproducible but not untouched confirmatory evidence.
2. Results are specific to this corpus, query set, relevance labels, model snapshots and local runtime; they do not establish universal search superiority.
3. The 11-query RRF component diagnostic is selected, not an all-query component-rank reconstruction.
4. Truncation analyses are observational/descriptive; causal impact is unproven.
5. Model revisions match local cached snapshots; loaded model objects were not independently attested.
6. Historical latency artifact does not record BGE-prefix invocation; exact prefix parity is unverified.
7. Latency is local CPU benchmark evidence, not a deployed-service SLO or cloud cost estimate.
8. MedCPT comparison is exploratory on previously inspected queries and has no independent-domain confirmation or dedicated repeated serving-latency result.
9. Independent domain data, human-verified labels and untouched holdout remain future work.
10. Phase 3 does not evaluate generated-answer correctness, faithfulness, claim-level citation accuracy, abstention, prompt-injection resistance, authentication, tenant isolation, ACL enforcement or disaster recovery.

## 13. Acceptance status

| Step | Status | Evidence / remaining qualification |
|---|---|---|
| 3.1–3.7 | Complete | Frozen protocol, dataset checks, 300-query quality run, four strategies, consistency and six paired comparisons. |
| 3.8 | Built | `phase3_failure_analysis_v1.json`; 1,200 mutually exclusive classifications validated. |
| 3.9 | Verified / consolidated | 11-query RRF trace verified; K=10/20/50 sensitivity evidence preserved in Phase 2 freeze. |
| 3.10 | Built | Reranker decision artifact; MS MARCO default OFF; MedCPT remains exploratory. |
| 3.11 | Accepted with limitations | 6,000-sample document-level latency artifact; protocol marked valid. |
| 3.12 | Built | Dataset, Git, source and model metadata reconciled; prefix and loaded-model attestation limitations explicit. |
| 3.13 | Designed | Independent-validation design draft; domain selection and preregistration pending. |
| 3.14 | Report generated; regression passed | All 15 artifacts packaged; 595 tests passed; final diff review and owner acceptance pending. |

**Phase 3 exit recommendation:** evaluation, analysis, evidence packaging, report generation, isolated reproduction and full regression are complete. Final acceptance remains pending complete Git diff review and explicit project-owner approval. No partial Phase 3 commit is proposed.

## 14. Artifact manifest

SHA-256 values below are computed from the result files present when this report is generated.

| Artifact | SHA-256 |
|---|---|
| `phase3_acceptance_final_v1.json` | `3357b12b389452cc80aa81a09a83616d5d3668ae0c9e3b27a1accd7a505c9972` |
| `phase3_latency_doclevel_v1.json` | `5f0900511a695355e8058a3183f86155b284c3ca735701ef8eb9069a5fd5249e` |
| `phase3_statistical_comparisons_full_precision_v1.json` | `b6669b30f2cd531bcf4955f39f148bfec36517e1e0daf917effc9a0228035d53` |
| `phase3_rrf_diagnostic_v1.json` | `5d775d9a374b5f1757e9663a8fa33ecd179037564be1cea372104c2b933f1728` |
| `phase3_failure_analysis_v1.json` | `66fbfdb8b904401bee11d981ec60ae6734df4f01880cce37c0f0ff22e8ee19e5` |
| `phase3_medcpt_ab_300q_v1.json` | `aa0c5bf9d89f398a795d05ef1e49e1e8ac1a8e384bdb9b212d0ee714ff3cf575` |
| `phase3_medcpt_statistical_comparisons_v1.json` | `09e8afa610d41c5a2e1da3b13506a065c7bca96a794abffc91210fce3577fca1` |
| `phase3_reranker_decision_v1.json` | `8a3eab8809f056187c8013f580db626b9cf258f9c7f8f1120e1931babf4ec763` |
| `phase3_provenance_reconciliation_v1.json` | `7d38e4b80460247962a3b31479df35be5f3cf2eafe9e8a9b34360853e81808d3` |
| `phase3_reranker_truncation_audit_v1.json` | `91628608a9b1487bf3b40254eb0001d15c75f18f4763f2ac5be491ea2e9a29bf` |
| `phase3_reranker_truncation_query_association_v1.json` | `a65c1942212fa3d641a11a1746d820b2118ca5ea066af3828f1bac253052ee36` |
| `phase3_reranker_relevant_truncation_impact_v1.json` | `48e9102c3cf5dbb3ac4ed34a0378fb684d1487e7b1d78063e949474f2bb30ab5` |
| `phase3_latency_final_v1.json` | `4d1d10f4b2a18e8da0f7b9888c09faea2ca2dc464419447d6b087dd8c615dfae` |
| `phase3_medcpt_screening_10q_v1.json` | `73eb05e29b8b275785504d6c4aa2d22184001218a31ff088f56d574f4564bcc0` |
| `phase3_statistical_comparisons_v1.txt` | `97904f8a19e62235cc9c464eec347554b5f460a2118938d199dd7223379bc696` |

All 15 preserved result artifacts are versioned in this repository under `results/`. `phase3_latency_doclevel_v1.json` is the canonical document-level latency evaluation; `phase3_latency_final_v1.json` is a separate chunked experiment, not the canonical latency result. The rounded statistical text file is supplementary; the full-precision JSON is authoritative. Regenerate this report with `python scripts/build_phase3_acceptance_report.py` from the repository root.

---

**Report generated from preserved artifacts.** Regenerate with `python scripts/build_phase3_acceptance_report.py` from the acceptance worktree after any source artifact changes; review the Git diff before accepting.
