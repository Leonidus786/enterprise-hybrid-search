# Step 4 — Per-query Ranking Trace

## Purpose

Persist the IDs returned for each query and the pre-reranking candidate IDs, so benchmark differences can be investigated query by query rather than relying only on aggregate scores.

## Result schema

Step 3 result schema advances to version 3. Each `per_query.<strategy>.<query_id>` record keeps the existing numeric metric keys and adds `_trace`:

- `schema_version`: trace schema version.
- `level`: `doc` or `chunk`; indicates ID space.
- `ranked_ids`: ranked IDs returned by the evaluator's search call (up to search depth).
- `candidate_ids`: pre-reranking candidate IDs mapped to the evaluation ID space.
- `relevant_ids`: qrel IDs for this query.
- `grades`: graded qrels when present for document-level evaluation.
- `query_text_sha256`: SHA-256 of query text.
- counts for returned ranking, candidate pool, and relevant IDs.
- `query_text`: included only with `--include-query-text`.

This trace stores IDs, not underlying BM25, dense, RRF or cross-encoder numeric retrieval scores. The current pipeline returns IDs from its public search methods; capturing exact component scores would be a separate instrumentation task.

## Privacy

Raw query text is excluded by default. It can be opted into for an approved dataset with `--include-query-text`. Do not enable that option for confidential queries unless the result artifact has the required access controls.

## Verification

Run:

```cmd
uv run python -m pytest -q tests\test_ranking_trace.py tests\test_per_query.py tests\test_compare.py tests\test_evaluate_per_query.py tests\test_compare_adjustment.py
uv run python -m pytest -q
uv run python scripts\run_eval.py --fake --no-cache --output results\step4-trace-smoke.json
```

Inspect `ranking_trace` metadata and one `per_query.<strategy>.<query_id>._trace` record in the resulting JSON. Then run a real benchmark to create fresh schema v3 output. Historical schema v2 files remain unchanged and will not contain rankings.
