# Phase 2 — Retrieval Reliability and Configuration Freeze

**Status:** Implementation and regression validation complete; included in the Phase 2 closure commit
**Freeze date:** 2026-11-13
**Project level:** L3 — professional retrieval/indexing prototype

## Scope and outcome

Phase 2 hardened ranking behavior, measured candidate-depth sensitivity, verified deduplication invariants, and synchronized the configuration manifest with the evaluation and latency runners. The implementation scope is frozen. No further Phase 2 feature work should be added without a demonstrated regression or a new, explicitly approved requirement.

| Milestone | Status | Evidence / qualification |
|---|---|---|
| 2.1 Lazy CrossEncoder loading | Complete · pushed | Commit `2082727` |
| 2.2 Reranker default OFF | Complete · pushed | Commit `5af3452` |
| 2.3 Pre/post-rerank ranking trace integration | Tested; included in the Phase 2 closure commit | 15 targeted tests passed; test file included in the seven-file Phase 2 closure set |
| 2.4 Candidate-depth sensitivity | Complete | K=10, 20, 50 quality evaluations over 300 queries each and dedicated repeated latency runs |
| 2.5 Stable tie-breaking | Implemented and tested; included in the Phase 2 closure commit | One-line stable dense sort; 4 targeted tests passed; test file included in the seven-file Phase 2 closure set |
| 2.6 Deduplication verification | Complete | Existing duplicate-ID, parent-document, RRF, evaluation, and lifecycle tests reviewed; no unjustified production changes added |
| 2.7 Configuration freeze | Implemented and tested; included in the Phase 2 closure commit | Manifest synchronized; 6 configuration tests passed |

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

## Stable tie-breaking decision

Dense ranking now uses `np.argsort(-sims, kind="stable")`. Sparse sorting, RRF first-seen ordering, and reranker incoming-order tie behavior are preserved and covered by equal-score regression tests.

Persistent index ordering supports this contract:

- Rebuild sorts documents by `doc_id`; chunk generation walks documents and their word windows in sequence.
- Insert and update reconstruct chunks in sorted document order and reorder vector rows to match chunk IDs.
- Delete retains the existing chunk sequence and selects the corresponding vector rows.
- Snapshot serialization preserves caller-supplied record order; snapshot loading reconstructs chunks from JSONL in that same order.
- The low-level `build_index_snapshot()` intentionally preserves caller order. Canonical ordering is established by the higher-level lifecycle operations; arbitrary callers of the low-level builder remain responsible for supplying deterministic order.

Decision: preserve the established incoming order on ties. Do not introduce an arbitrary chunk-ID tie-breaker or rewrite other ranking components.

## Deduplication decision

The audit found existing coverage for duplicate document/chunk IDs in pipeline, rebuild, build, insert, RRF, retrieval, evaluation, and index lifecycle paths. No production defect was established, so no broad rewrite or test-count-only additions were made.

One narrow coverage limitation remains: the defensive `duplicate chunk ID after update` guard has no direct branch-specific regression test. Normal update flow removes the old document's chunks before generating replacements; the audit did not establish a reachable defect. Keep this as a documented test gap, not a claimed bug.

## Configuration freeze

`configs/default.json` now distinguishes:

- `strategies`: all supported evaluation modes, including `hybrid_rerank`.
- `default_strategies`: `bm25`, `dense`, and `hybrid`; reranking remains opt-in.
- `canonical_protocol`: SciFact test corpus/query paths, document-level mode, no chunking, no persistent evaluation cache, and the default BGE query prefix.
- `latency_defaults`: all 19 defaults captured from `run_latency.build_parser()`; the required `--output` argument is deliberately excluded because it has no default.

The configuration tests also lock model IDs, pinned latency revisions, adapter parameter defaults, evaluation runner's unpinned revision defaults, and the canonical protocol's CLI representation. Existing model-adapter tests remain the source for actual adapter invocation behavior; those tests were not duplicated.

## Final validation

| Check | Result |
|---|---:|
| Stable tie-breaking targeted tests | 4 passed |
| Configuration contract tests | 6 passed |
| Ranking-trace integration tests | 15 passed |
| Final full test suite, after the final manifest assertion | **595 passed in 9.06 s** |
| `git diff --check` | Passed immediately before staging and commit review |

## Git and release state

The Phase 2 implementation and validation scope is frozen. The seven-file local closure set is `configs/default.json`, `docs/DECISIONS.md`, `docs/development/phase2-freeze.md`, `src/hs/dense.py`, `tests/test_configs.py`, `tests/test_ranking_trace_integration.py`, and `tests/test_stable_tie_breaking.py`. Unrelated untracked files (`3`, `None`, `docs/development/phase3-evaluation-acceptance.md`, and the three `phase2_*_patch.py` helpers) are intentionally preserved and excluded.

The selective local commit closes the pending Phase 2 Git changes. This does not imply a push, release tag, or deployment; none is claimed here.

## Engineering scorecard

**Current:** L3 — professional retrieval/indexing prototype.
**Why:** persistent versioned index lifecycle; reproducible retrieval evaluation/provenance foundations; ranking traces, stable tie behavior, candidate-depth evidence, and 595 passing tests.
**Next level:** L4 requires an accepted evaluation report and a user-facing grounded Search + RAG application with citations, abstention, API, demo, CI, deployment, and security validation. Phase 2 does not establish production or enterprise readiness.
