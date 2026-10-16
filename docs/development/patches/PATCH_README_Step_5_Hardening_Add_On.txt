Step 5 Hardening Add-on — Protocol Fail-Closed + Provenance Clarity

Apply AFTER the Step 5 Provenance Patch, at the repository root.
This add-on replaces:
  src/hs/compare.py
  src/hs/provenance.py
  scripts/run_eval.py
  tests/test_provenance.py

And adds:
  tests/test_compare_protocol.py

What it changes:
1. Schema v4 cross-run comparisons fail closed when required dataset/config/provenance fields are missing.
2. Compares corpus identity, canonical query/qrels fingerprint, model IDs/revisions, and benchmark-source fingerprint.
3. Latency cross-run comparisons require matching cache settings and matching recorded runtime fields.
4. Cross-file comparisons reject unresolved model revisions; within-one-file mode comparisons remain possible.
5. Fake-model smoke runs are explicitly labelled "fake_models_smoke_test_not_benchmark_evidence" and never count fake labels as resolved model commits.
6. Query/relevance fingerprint is stable if the same uniquely identified queries are reordered in the input file. Per-query results still retain their own saved ordering.

Validation command:
  uv run python -m pytest -q tests\test_provenance.py tests\test_compare_protocol.py tests\test_ranking_trace.py tests\test_per_query.py tests\test_compare.py tests\test_evaluate_per_query.py tests\test_compare_adjustment.py
  uv run python -m pytest -q

Important limit:
This add-on does not replace a full clean-clone smoke test. The validation workspace used during preparation contains NotImplementedError stubs for the loader/pipeline components, so the CLI smoke test cannot run here. Run the fake CLI smoke on the complete project checkout after applying both patches.

Review note:
Schema v4 strict comparisons are intentionally stricter. Runs made with different relevant source code, model revisions, corpus identity, query/qrels identity, retrieval settings, or latency runtime should not silently be treated as an apples-to-apples comparison. To compare across an intentional code change, design and document that as a separate controlled experiment rather than disabling checks ad hoc.


Developer documentation: `docs/development/step5-hardening.md` covers the schema-v4 comparison rules, apply order, validation commands, interpretation, and limitations.
