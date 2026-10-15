# Step 3 — Multiple-comparison correction

## Why this patch exists

The prior report compared three retrieval-system pairs across three metrics: nine hypotheses. Reading each nominal 95% bootstrap interval as a separate significance decision can overstate evidence across the family.

## What changes

- `scripts/compare_runs.py` preserves each comparison's nominal 95% percentile-bootstrap confidence interval.
- It also computes a Bonferroni-adjusted interval at confidence `1 - family_alpha / number_of_tests` (default family alpha 0.05).
- The primary `Significant after adjustment?` label is based on the adjusted interval excluding zero.
- The report states family size and adjusted confidence level explicitly.
- `--family-alpha` controls the family-wide alpha and must be between zero and one.

## Caveat

The correction applies to all pair/metric combinations requested in one CLI invocation. Bonferroni is conservative. The intervals are percentile-bootstrap intervals, so their finite-sample coverage is approximate; this adjustment does not validate the benchmark protocol. For final claims, predeclare nDCG@10 as the primary metric and treat Recall@100 and MRR as secondary/exploratory unless the evaluation plan states otherwise.

## Verify

```cmd
uv run python -m pytest -q tests\test_compare.py tests\test_compare_adjustment.py
uv run python scripts\compare_runs.py --a results\scifact_test_doc_level_step3.json --pair bm25 dense --pair dense hybrid --pair hybrid hybrid_rerank --metrics ndcg@10 recall@100 mrr --resamples 10000 --seed 42
```
