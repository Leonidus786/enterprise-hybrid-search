# Step 5 — Protocol comparison hardening

## Purpose

Make saved benchmark comparisons fail closed when a schema-v4 run does not carry enough evidence to establish that two runs are comparable. The goal is to prevent an apparently valid confidence interval from being reported for incompatible or insufficiently identified runs.

This is a comparison/provenance hardening add-on. It does **not** prove that a benchmark is scientifically valid, that relevance labels are correct, or that a run can be reproduced on a clean clone.

## Apply order

1. Start from the complete `enterprise-hybrid-search` repository.
2. Apply the Step 5 provenance patch first.
3. Extract this add-on at the repository root, preserving the paths in the archive and allowing the listed files to be replaced.
4. Review the changed files before committing. Do not overwrite saved benchmark result JSON files.

## Files changed

Replaces:

- `src/hs/compare.py`
- `src/hs/provenance.py`
- `scripts/run_eval.py`
- `tests/test_provenance.py`

Adds:

- `tests/test_compare_protocol.py`

## Behaviour

### Schema and dataset identity

For schema version 4 or later, comparisons check that both results use the same schema and have the required dataset identity/count fields. Corpus SHA-256, query-file SHA-256, and the canonical query/relevance fingerprint must be present and available. The logical fingerprint is compared for equality; the raw query-file hashes are retained as provenance and may differ if the same logical query/label records are serialized in a different order.

### Configuration and code identity

Relevant retrieval settings must be present and consistent, including ranking cutoff, retrieval/candidate depth, evaluation level, chunking, fake-model flag, model IDs, prefix setting, and requested model revisions. The source-code fingerprint must be available and match between schema-v4 runs.

### Model revisions

Model identifiers and revision metadata are checked. A model revision that cannot be resolved remains explicitly unknown; the code must not guess a commit. Separate-file comparisons reject unresolved model revisions because model identity cannot be established. Comparisons between strategies within the same saved run can still be useful because those strategies share the run's loaded models.

### Latency comparisons

In addition to relevant retrieval settings, latency comparisons require matching cache configuration and matching recorded runtime fields: Python version, platform, machine, processor, logical CPU count, and relevant package versions. This helps avoid comparing unlike environments, but it is not a substitute for a controlled repeated-trial latency protocol.

### Fake-model smoke runs

`--fake` is an offline functional smoke test, not benchmark evidence. Its provenance status is labelled `fake_models_smoke_test_not_benchmark_evidence`; fake identifiers are not represented as resolved real model commits.

### Older result files

Legacy files retain best-effort comparison behaviour where the stricter schema-v4 evidence is not available. Do not interpret a successful legacy comparison as equivalent to passing all schema-v4 checks.

## Validation on Windows (CMD)

Run targeted tests:

```cmd
uv run python -m pytest -q tests\test_provenance.py tests\test_compare_protocol.py tests\test_ranking_trace.py tests\test_per_query.py tests\test_compare.py tests\test_evaluate_per_query.py tests\test_compare_adjustment.py
```

Run the complete test suite:

```cmd
uv run python -m pytest -q
```

Run a fake-model CLI smoke test on the **complete project checkout**:

```cmd
uv run python scripts\run_eval.py --fake --no-cache --output results\step5-hardening-smoke.json
```

The smoke command should finish successfully and write a schema-v4 JSON file. It validates plumbing only. Do not cite fake-model metrics as retrieval-performance results.

Inspect key fields:

```cmd
uv run python -c "import json; p=json.load(open('results/step5-hardening-smoke.json', encoding='utf-8')); print('schema:', p['schema_version']); print('corpus sha256:', p['dataset']['corpus_sha256']); print('query-file sha256:', p['dataset']['queries_file_sha256']); print('query fingerprint:', p['dataset']['query_fingerprint']); print('source fingerprint:', p['provenance']['source_code']['sha256']); print('reproducibility:', p['provenance']['reproducibility_status']['status']); print('models:', p['provenance']['models'])"
```

Expected: schema version `4`, non-empty hashes when the default sample inputs and source files exist, and a fake-model status explicitly saying this is not benchmark evidence. If any are missing, investigate instead of calling the run reproducible.

## Interpretation and limits

- SHA-256 establishes file identity, not dataset correctness, licensing, or label quality.
- Source fingerprints cover the source files selected by the provenance helper; they are not a container digest or a complete locked build environment.
- Runtime metadata is descriptive. Matching metadata does not guarantee identical latency across operating systems, load levels, CPU governors, or hardware conditions.
- One timing sample per query is not a repeated-trial latency study. Warm-up, repeats, cache policy, concurrency/load, and p50/p95/p99 reporting need their own explicit protocol.
- A passing unit suite is not a clean-clone reproduction. Verify the full fake CLI smoke test and three-command reproduction from a fresh checkout before claiming end-to-end reproducibility.
- Do not disable the strict checks just to force a comparison through. If code or model changes are intentional, record them as a controlled experiment and explain the comparison design.
