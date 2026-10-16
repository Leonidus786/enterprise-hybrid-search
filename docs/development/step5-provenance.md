# Step 5 — Benchmark provenance

## Purpose

Make each saved benchmark JSON auditable and easier to reproduce by capturing input-file hashes, the canonical query/relevance fingerprint, model metadata, selected source-file hashes, Git state, runtime/package versions, machine metadata, and a configuration fingerprint.

## Changes

- `src/hs/provenance.py`: streaming SHA-256 helpers and provenance envelope construction.
- `src/hs/models.py`: optional Hugging Face revision arguments and best-effort extraction of the resolved immutable commit hash from loaded model configuration/tokenizer metadata.
- `scripts/run_eval.py`: schema version 4 for per-query result files, corpus/query file checksums, config fingerprint, model revision options, and provenance metadata.
- `tests/test_provenance.py`: deterministic hashing, missing-input reporting, source/config fingerprint and model revision tests.

## CMD validation

```cmd
uv run python -m pytest -q tests\test_provenance.py tests\test_ranking_trace.py tests\test_per_query.py tests\test_compare.py tests\test_evaluate_per_query.py tests\test_compare_adjustment.py
uv run python -m pytest -q
uv run python scripts\run_eval.py --fake --no-cache --output results\step5-provenance-smoke.json
```

Then inspect the generated JSON:

```cmd
uv run python -c "import json; p=json.load(open('results/step5-provenance-smoke.json', encoding='utf-8')); print('schema:', p['schema_version']); print('corpus sha256:', p['dataset']['corpus_sha256']); print('queries sha256:', p['dataset']['queries_file_sha256']); print('config fingerprint:', p['provenance']['config_fingerprint_sha256']); print('git:', p['provenance']['git']); print('reproducibility:', p['provenance']['reproducibility_status']); print('models:', p['provenance']['models'])"
```

## Important limitations

- Model revision extraction is best-effort and offline. If the loaded model does not expose a commit hash, the revision is recorded as unknown. Do not describe the run as fully model-reproducible while this field is unknown.
- To pin a model explicitly, pass an immutable Hugging Face commit through `--embedding-revision` and/or `--reranker-revision`. Branch names such as `main` are not immutable.
- SHA-256 proves file identity, not dataset licensing, relevance-label correctness, or scientific validity.
- Source hashes cover Python modules under `src/hs`, the benchmark and comparison CLIs, and project dependency manifests; they are not a full container/environment lock.
- Hardware metadata is descriptive and is not a performance guarantee.
- Never store raw query text by default. `--include-query-text` is opt-in and may expose sensitive material.
