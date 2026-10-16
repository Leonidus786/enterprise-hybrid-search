Step 4 — Raw per-query ranking trace

Extract this ZIP into E:\enterprise-hybrid-search, preserving folder structure.

Files replaced:
  src\hs\evaluate.py
  scripts\run_eval.py
  scripts\compare_runs.py

New files:
  tests\test_ranking_trace.py
  docs\development\step4-ranking-trace.md

Safety:
- Keep your existing backup until all tests pass.
- Existing numeric per-query metric keys stay in place; `_trace` is added alongside them.
- `compare_runs.py` is based on the Bonferroni-correction version, and default metric discovery explicitly excludes trace metadata.
- Query text is not stored unless `--include-query-text` is specified.
- Trace records ranked IDs, candidate IDs, relevance IDs, optional graded qrels, ID space, and a query-text hash. Current pipeline interfaces do not expose raw component retrieval scores, so those are not claimed as included.

CMD verification:
  uv run python -m pytest -q tests\test_ranking_trace.py tests\test_per_query.py tests\test_compare.py tests\test_evaluate_per_query.py tests\test_compare_adjustment.py
  uv run python -m pytest -q
  uv run python scripts\run_eval.py --fake --no-cache --output results\step4-trace-smoke.json

Do not overwrite existing benchmark JSON files. New evaluations should use new output names.
