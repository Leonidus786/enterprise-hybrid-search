# Step 3 — Per-query evaluation and paired comparisons

## Why this change exists

Before this step, `run_eval.py` saved only aggregate metrics. An aggregate such as `nDCG@10 = 0.710` says what the average was, but not which queries improved or whether hybrid and dense were better on the same individual queries. Without aligned per-query values, a paired confidence interval cannot be calculated defensibly.

This patch keeps the existing `evaluate()` return shape for compatibility and adds `evaluate_with_details()` for the CLI. Saved result JSON schema version 2 contains:

- `metrics`: aggregate metrics per strategy (including latency percentiles).
- `per_query`: query-level ranking metrics per strategy.
- `dataset.query_fingerprint`: SHA-256 fingerprint of the evaluated query text and relevance labels, to reject comparisons when the evaluated query/relevance set differs.

Aggregate and per-query ranking scores are calculated through the same scoring helper. `run_eval.py` checks their means before saving. Latency percentiles remain aggregate-only; it would be incorrect to compare a p50/p95 latency value against the arithmetic mean of per-query values.

## Relevance and nDCG

The standalone `score_rankings()` API follows the Step 3 test contract and uses binary relevance. The integrated evaluator preserves existing semantics: document-level queries with a non-empty `grades` mapping use graded nDCG; other cases use binary nDCG. Precision, recall, and MRR continue to use the query's `relevant` IDs. MRR is limited to the top `k` and retains the legacy JSON key `mrr`.

## Files in this change

- `src/hs/per_query.py` — per-query scoring helper and public binary `score_rankings()` function.
- `src/hs/compare.py` — JSON validation, aggregate consistency checking and paired bootstrap; validates query fingerprints when both runs provide them.
- `src/hs/evaluate.py` — adds `evaluate_with_details()` while preserving `evaluate()`'s existing result shape.
- `scripts/compare_runs.py` — compares saved runs with paired bootstrap confidence intervals.
- `scripts/run_eval.py` — emits query-wise scores, query fingerprint and schema version 2 JSON; checks consistency before saving.
- `tests/test_evaluate_per_query.py` — regression tests for aggregate consistency, graded nDCG, API compatibility and mismatched query fingerprints.

## Apply in Windows File Explorer

Extract this ZIP into `E:\enterprise-hybrid-search`, preserving folder paths. Allow replacement of `src\hs\evaluate.py` and `scripts\run_eval.py` only after confirming they are the intended project files. The other paths add or update the Step 3 modules and this note.

## Verify in CMD

From `E:\enterprise-hybrid-search`:

```cmd
uv run python -m pytest -q tests\test_per_query.py tests\test_compare.py tests\test_evaluate_per_query.py
uv run python -m pytest -q
uv run python scripts\run_eval.py --fake --no-cache --output results\step3-smoke.json
uv run python -c "import json; p=json.load(open('results/step3-smoke.json', encoding='utf-8')); print('schema', p['schema_version']); print('modes', list(p['per_query'])); print('queries/mode', {k: len(v) for k,v in p['per_query'].items()})"
```

Fake-model output is an integration smoke test only, not benchmark evidence. Do not replace SciFact result files with smoke-test results.

## Step 1 and Step 2 documentation

No separate README is required for every step. Keep this note under `docs/development/`; document earlier steps only if they have meaningful decisions, commands, artifacts or acceptance evidence that is not already recorded in the main project handoff/tracker.
