# PATCH_README_Step_3_evaluation_integration_patch

## Purpose

This patch adds query-level evaluation records and paired-bootstrap comparisons to `enterprise-hybrid-search`. Aggregate metrics alone can hide which queries changed; per-query values allow the same queries to be compared and confidence intervals to be estimated from paired score differences.

## What the Step 3 patch contains

- `src/hs/per_query.py` — per-query ranking-score helpers and the public binary-relevance `score_rankings()` API.
- `src/hs/evaluate.py` — adds `evaluate_with_details()` while preserving the legacy aggregate-only return shape from `evaluate()`.
- `src/hs/compare.py` — result JSON validation, aggregate/per-query consistency checks, paired bootstrap confidence intervals, and query-fingerprint comparison checks where fingerprints are available in both runs.
- `scripts/run_eval.py` — saves aggregate and per-query metrics, records a canonical fingerprint of evaluated query text and relevance labels, and checks that per-query means reconcile with aggregate metrics before saving.
- `scripts/compare_runs.py` — compares strategies from one saved run or compares two saved runs using paired query-level bootstrap intervals.
- `tests/test_evaluate_per_query.py` — regression tests for compatibility, graded nDCG, consistency, and mismatched query fingerprints.
- `docs/development/step3-evaluation-integration.md` — step-specific implementation notes.

## Result format and interpretation

The original Step 3 patch writes schema version `2` when per-query records are present. The JSON includes:

- `metrics`: aggregate ranking metrics and aggregate latency summaries by strategy.
- `per_query`: metric values keyed by strategy and query ID.
- `dataset.query_fingerprint`: a SHA-256 fingerprint of the evaluated query IDs/text and relevance labels (including grades when supplied).

Aggregate ranking scores and per-query ranking scores use the same evaluation logic. Latency percentiles remain aggregate-only; do not interpret their absence from `per_query` as a consistency failure.

The integrated evaluator preserves the existing relevance behavior: document-level queries with a non-empty `grades` mapping use graded nDCG; otherwise nDCG uses binary relevance. Precision, recall, and MRR use the query's `relevant` IDs. The legacy metric key `mrr` is retained.

A query fingerprint checks logical query/relevance-set identity; it does not prove that the labels themselves are correct or that a benchmark is representative.

## Apply instructions

1. Make sure you are applying this patch to the Step 2/baseline version for which it was prepared.
2. Extract the ZIP into the repository root, preserving the archive's directory structure.
3. Review any replacement prompt carefully before allowing overwrite. This archive includes `src/hs/compare.py`, `src/hs/evaluate.py`, and `scripts/run_eval.py`, which may have been extended by later Step 4 ranking-trace and Step 5 provenance/hardening work.
4. **Do not blindly apply this historical Step 3 ZIP over a repository that already contains the later Step 4/Step 5 implementation.** Doing so can replace newer comparison/provenance/trace functionality and downgrade the result schema from version 4 to version 2. On a later implementation, use this README as documentation and retain the newer source files, or merge changes intentionally.
5. Keep existing real benchmark result files; do not replace them with smoke-test output.

## Verify the Step 3 patch on its intended baseline

Run these commands from the repository root in Windows CMD:

```cmd
uv run python -m pytest -q tests\test_per_query.py tests\test_compare.py tests\test_evaluate_per_query.py
uv run python -m pytest -q
uv run python scripts\run_eval.py --fake --no-cache --output results\step3-smoke.json
uv run python -c "import json; p=json.load(open('results/step3-smoke.json', encoding='utf-8')); print('schema', p['schema_version']); print('modes', list(p['per_query'])); print('queries/mode', {k: len(v) for k, v in p['per_query'].items()})"
```

Expected smoke-test characteristics: schema `2`; strategies normally include `bm25`, `dense`, `hybrid`, and `hybrid_rerank`; `per_query` contains query-level scores for the evaluated queries; and the CLI reports that aggregate/per-query consistency passed. Exact fake-model metric values are not a benchmark target.

`--fake` is for an offline integration smoke test only. Its metrics are not evidence of retrieval quality, generalization, or production latency.

## Compare strategies with paired bootstrap intervals

Example: compare hybrid against dense within one saved file:

```cmd
uv run python scripts\compare_runs.py --a results\step3-smoke.json --pair dense hybrid --metrics ndcg@10 recall@100
```

Example: compare two saved files, where the evaluated query IDs match:

```cmd
uv run python scripts\compare_runs.py --a results\run-a.json --b results\run-b.json --pair dense hybrid --metrics ndcg@10 recall@100
```

The reported mean delta is `B - A`; the interval is a paired bootstrap confidence interval over query-level deltas. A comparison requires matching query IDs. Where both files contain query fingerprints, differing fingerprints are rejected. Use real, protocol-compatible benchmark runs for research conclusions.

## Latest repository validation reported by the user

These results were reported after the subsequent Step 5 hardening work, not from running the isolated historical Step 3 ZIP:

- Targeted tests: **60 passed**.
- Full suite: **271 passed**.
- Fake CLI smoke test completed and saved `results/step5-hardening-smoke.json`.
- The saved result reported schema `4`, corpus/query file SHA-256 values, query and source fingerprints, and `reproducibility_status.status = fake_models_smoke_test_not_benchmark_evidence`.
- Model metadata identified the embedding and reranker as fake models with `revision_status = not_applicable_fake_model`.

These checks support that the reported current repository's Step 5 smoke path completed. They do not establish that the historical Step 3 ZIP can safely overwrite later files, nor do fake-model results establish benchmark quality. Preserve the current schema v4 implementation when working on the latest project state.

## Scope and limitations

This step supplies per-query scores and paired comparisons. It does not by itself prove dataset validity, label quality, exact model reproducibility, fair uncached latency, repeated-trial latency confidence intervals, cross-dataset generalization, or production readiness. Those require separately controlled experiments and provenance checks.
