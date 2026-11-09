# Enterprise Hybrid Search

A reproducible Python retrieval-engineering project that combines **BM25 keyword search**, **dense semantic retrieval**, and **Reciprocal Rank Fusion (RRF)**, with an optional **Cross-Encoder reranking** stage.

The project focuses on making retrieval behavior measurable and inspectable: deterministic document and chunk identities, model-revision provenance, persistent index snapshots, per-query ranking traces, ranking metrics, latency measurements, and regression tests.

> **Project scope:** This is a retrieval and persistent-index engineering prototype with a tested local document lifecycle. It is **not yet** a hosted enterprise search product, grounded RAG application, or production platform.

<p align="center">
  <img src="assets/enterprise-hybrid-search-social-card.png" alt="Enterprise Hybrid Search: BM25 and dense retrieval combined with RRF and optional Cross-Encoder reranking" width="100%" />
</p>

## Project status

**Latest local verification: 10 November 2026**

| Area | Current status |
| --- | --- |
| Retrieval | BM25 sparse retrieval, exact Dense cosine search, RRF fusion, and optional Cross-Encoder reranking are implemented. |
| Ingestion | JSONL document loading, optional source/title/metadata fields, revision-aware embedding caching, deterministic document/chunk identities, and duplicate document-ID validation are implemented. |
| Persistent indexing | Immutable versioned snapshots use `documents.jsonl`, `chunks.jsonl`, `vectors.npy`, a validated manifest, and artifact checksums. |
| Snapshot lifecycle | Snapshot loading, persisted DenseIndex initialization, HybridPipeline reconstruction, atomic activation, and explicit rollback are implemented and tested. |
| Full rebuild | Deterministic full-corpus rebuild sorts documents by `doc_id`, regenerates chunks, and activates only after snapshot validation. |
| Incremental lifecycle | Insert, update, and delete APIs are implemented. Insert/update embed only affected document chunks; unaffected vectors are reused. |
| Idempotency | Rebuild/insert/update/delete request fingerprints are stored in operation-aware manifest schema v2. Exact active retries, conflicting requests, and inactive-version protection are tested. |
| Failure recovery | Failure-path tests cover embedding/build, artifact-write, version-promotion, pointer-replacement, invalid staged snapshot, corrupt-active rollback, and orphan-staging isolation. Automatic crash recovery and orphan cleanup are not implemented. Immutable versions are retained; automatic pruning is not implemented. |
| Evaluation | Document- and chunk-level evaluation, per-query traces, provenance, statistical comparison, and latency tooling are implemented. |
| Tests | **578 tests passed** in the local Windows test run on 10 November 2026 (`11.12s`). This is local regression evidence, not external CI or deployment evidence. |
| Product surface | Hosted API, grounded RAG answer generation, authentication, multi-tenant access control, and a complete interviewer-facing demo are not yet implemented. |

### Latest test evidence

The following local Windows regression gates were observed on 10 November 2026:

| Regression gate | Result |
| --- | --- |
| Activation test module | 21 passed in 2.17s |
| Phase 1 indexing suite (9 modules) | 174 passed in 8.30s |
| Full project suite | **578 passed in 11.12s** |

These suites overlap; their test counts must not be added together. `git diff --check` completed without whitespace errors. Git emitted only CRLF-to-LF normalization warnings for `src/hs/index_store.py` and `tests/test_index_rebuild.py.

These are local regression results, not evidence of external CI, clean-clone installation, deployment, load testing, or production readiness.

**Acceptance status:** Steps 5.1–5.6 have implementation and regression-test evidence. Steps 5.5 and 5.6 are accepted locally. Step 5.7 and final Phase 1 acceptance remain pending.
## What the project includes

- **BM25 sparse retrieval:** Token-based retrieval with Lucene-style inverse document frequency, term-frequency saturation, and document-length normalisation.
- **Dense retrieval:** Exact cosine-similarity search using NumPy and injected embedding functions.
- **Hybrid retrieval:** Reciprocal Rank Fusion combines sparse and dense rank positions without directly adding scores from different scales.
- **Optional reranking:** A Sentence Transformers Cross-Encoder can rescore a bounded candidate pool.
- **Chunking:** Word-window chunking with configurable size and overlap, plus a no-chunk/whole-document mode.
- **Metadata-aware ingestion:** JSONL documents with required document IDs and text, plus optional source, title, and metadata fields.
- **Deterministic identity:** SHA-256 document and chunk revision identities, including supported document metadata, with explicit duplicate document-ID rejection.
- **Embedding cache:** Persistent per-text NumPy vectors with revision-aware cache identity, configuration-aware isolation, corruption checks, and atomic file replacement.
- **Persistent searchable index:** Immutable versioned snapshots containing document records, chunk records, normalized dense vectors, and a versioned manifest.
- **Index integrity:** Artifact checksums, strict manifest validation, embedding identity checks, vector validation, and corpus fingerprints.
- **Atomic activation:** Validated snapshots are promoted to immutable versions and published through an atomic `CURRENT.json` pointer. Explicit rollback to retained versions is supported.
- **Deterministic rebuild:** Full-corpus rebuild API with canonical document ordering and regenerated chunks and vectors.
- **Incremental mutations:** Insert, update, and delete create new immutable snapshots. Insert/update embed only affected document chunks; vectors for unchanged chunks are reused.
- **Idempotent mutations:** Canonical request fingerprints distinguish exact retries from conflicts and prevent an inactive immutable version from being silently reactivated.
- **Evaluation:** Precision@K, Recall@K, MRR, nDCG@K, candidate recall, per-query traces, and latency reporting.
- **Provenance:** Corpus/query fingerprints, model revision metadata, runtime information, and Git state in saved evaluation artifacts.
- **Tests:** Unit, edge-case, regression, schema, persistence, mutation, and integration tests.

## Retrieval flow

```text
                         JSONL corpus
                              |
                              v
                         Text chunking
                              |
                 +------------+------------+
                 |                         |
                 v                         v
            BM25 sparse              Dense cosine
             retrieval                retrieval
                 |                         |
                 +------------+------------+
                              |
                              v
                         RRF rank fusion
                              |
                              v
                  Optional Cross-Encoder
                        reranking
                              |
                              v
                        Ranked results
                              |
                              v
                Evaluation and per-query traces
```

The Cross-Encoder is an optional ranking stage. The recorded SciFact experiments did **not** establish a ranking-metric improvement over Hybrid retrieval, while CPU inference added substantial latency. Run it deliberately and measure it on the target corpus and hardware before adopting it.

## Persistent index architecture

The persistent index uses immutable, versioned snapshots.

```text
index_root/
├── CURRENT.json
├── staging/
└── versions/
    └── <index_id>/
        ├── manifest.json
        ├── documents.jsonl
        ├── chunks.jsonl
        └── vectors.npy
```

- `documents.jsonl` stores validated source documents and their supported metadata.
- `chunks.jsonl` stores deterministic chunk records and revision identities.
- `vectors.npy` stores the normalized `float32` embedding matrix, aligned by chunk order.
- `manifest.json` records corpus fingerprints, embedding identity, chunking configuration, sparse configuration, vector dimensions, and artifact checksums. Operation-aware snapshots also record the operation type and request fingerprint.
- `CURRENT.json` identifies the active immutable snapshot and records previous-version information used by rollback.

The loader validates the manifest, artifact checksums, record counts, corpus identity, chunk consistency, and vector matrix before exposing a snapshot.

A persisted `HybridPipeline` reconstructs BM25 from stored chunks and initializes DenseIndex from precomputed vectors. Loading an existing snapshot does not require re-embedding its documents.

### Manifest compatibility and idempotency

- Generic and legacy snapshots using manifest **schema v1** remain readable.
- Mutation snapshots use manifest **schema v2** and include strict operation metadata.
- The operation metadata contains the operation name (`rebuild`, `insert`, `update`, or `delete`) and a SHA-256 request fingerprint.
- Request fingerprints are computed from canonical JSON. `created_at_utc` is snapshot metadata and is not part of request identity.
- An exact retry returns the existing snapshot only while that version remains active.
- A different request using the same active version ID raises an idempotency conflict.
- An existing inactive version is not reactivated by an operation retry; it remains immutable and raises `FileExistsError`.
- The operation metadata is covered by the manifest checksum referenced by `CURRENT.json`.

### Lifecycle operations

| Operation | Expected behavior |
| --- | --- |
| Rebuild | Sort the input corpus deterministically, regenerate chunks and vectors, and activate a new immutable snapshot. An exact active retry returns the existing snapshot without embedding again. |
| Insert | Embed only the inserted document's chunks and reuse existing vectors. An exact active retry does not embed again. |
| Update | Replace one document revision, embed its new chunks, and reuse vectors for unchanged documents. An exact active retry does not embed again. |
| Delete | Remove the selected document and its vector rows while preserving unaffected vectors. An exact active retry returns the existing snapshot without reactivation. |

The mutation APIs create new immutable versions rather than overwriting existing snapshots. Activation currently assumes a **single writer**. Cross-process writer locking is not implemented; callers must serialize concurrent writers.

## Quick start

### Requirements

- Python 3.10 or newer (the latest recorded local test run used Python 3.12.10).
- [`uv`](https://docs.astral.sh/uv/).
- Optional model dependencies and sufficient RAM/disk space for Sentence Transformers models.

### 1. Install development dependencies

```cmd
uv sync --extra dev
```

For real embedding and Cross-Encoder evaluation, install the model extra too:

```cmd
uv sync --extra models --extra dev
```

### 2. Run the test suite

```cmd
uv run python -m pytest -q
```

Latest recorded local result (10 November 2026):

```text
578 passed in 11.12s
```

Test counts and runtime can change as the repository evolves and depend on the local environment. This result is not a substitute for running the suite on the exact checkout being evaluated.

### 3. Run an offline smoke test

```cmd
uv run python scripts/run_eval.py --fake
```

The fake embedding and scoring functions allow the retrieval and evaluation flow to run without downloading models. Fake-model metrics are **smoke-test output only**: do not use them as evidence of retrieval quality or as a comparison between real retrieval models.

### 4. Run a real-model evaluation

```cmd
uv run python scripts/run_eval.py
```

The first run may download the configured models. With no `--strategies` argument, the current CLI evaluates all four strategies (`bm25`, `dense`, `hybrid`, and `hybrid_rerank`). To exclude the reranked strategy from the evaluation, pass an explicit strategy list such as:

```cmd
uv run python scripts/run_eval.py --strategies bm25 dense hybrid
```

The recorded project documentation notes that the real-model runner initializes the Cross-Encoder adapter before evaluation even when that strategy is excluded. Lazy loading it only when requested remains a retrieval-refinement item to verify and address.

The recorded model revisions are:

| Component | Model | Pinned revision in the recorded SciFact run |
| --- | --- | --- |
| Embeddings | `BAAI/bge-small-en-v1.5` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `233902d25c440f23af6f7d6e94d2946bac0bee0a` |

The CLI supports explicit model revisions. See the pinned evaluation command below for a reproducible command shape.

## Input data format

The current corpus loader reads JSONL. Each line represents one document. The required fields are `doc_id` and `text`; `source`, `title`, and `metadata` are optional.

```json
{
  "doc_id": "example_doc",
  "text": "Text to index goes here.",
  "source": "internal-handbook",
  "title": "Example document",
  "metadata": {
    "department": "engineering"
  }
}
```

The query file is a JSON array. Each query has an ID, query text, and relevant document IDs:

```json
[
  {
    "id": "q1",
    "query": "How do I find the relevant passage?",
    "relevant": ["example_doc"]
  }
]
```

For document-level evaluation, `relevant` contains parent document IDs. For chunk-level evaluation, use the chunk-ID space expected by the chunk-level protocol. The bundled `data/sample/` corpus and labels are a small starter dataset, not a representative enterprise benchmark.

## Useful evaluation commands

### Offline smoke test

```cmd
uv run python scripts/run_eval.py --fake
```

### Compare retrieval without reranking

```cmd
uv run python scripts/run_eval.py --strategies bm25 dense hybrid
```

### Explicitly evaluate Hybrid retrieval and reranking

```cmd
uv run python scripts/run_eval.py --strategies hybrid hybrid_rerank --candidate-k 10
```

### Evaluate chunk IDs instead of parent document IDs

```cmd
uv run python scripts/run_eval.py --level chunk
```

### Evaluate each document as one searchable item

```cmd
uv run python scripts/run_eval.py --no-chunk --level doc
```

### Pin model revisions and save a result artifact

```cmd
uv run python scripts/run_eval.py ^
  --corpus data/scifact_test/corpus.jsonl ^
  --queries data/scifact_test/queries.json ^
  --k 10 ^
  --retrieval-k 100 ^
  --candidate-k 50 ^
  --level doc ^
  --no-chunk ^
  --embedding-revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a ^
  --reranker-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a ^
  --output results/scifact_doclevel_pinned_v2.json
```

The line-continuation character `^` is for Windows CMD. On shells that use a different continuation syntax, put the command on one line or adjust the separators.

### Cross-Encoder inference options

The evaluation CLI supports the following Cross-Encoder settings. The options affect reranking when `hybrid_rerank` is evaluated:

| Option | Purpose | Current default |
| --- | --- | --- |
| `--reranker-batch-size` | Number of query/document pairs per inference batch | `8` |
| `--reranker-device` | Explicit inference device, such as `cpu` or `cuda` | Automatic (`None`) |
| `--reranker-max-length` | Override the model's maximum input length | Model setting (`None`) |

Example for a CPU run:

```cmd
uv run python scripts/run_eval.py --strategies hybrid hybrid_rerank --candidate-k 10 --reranker-batch-size 8 --reranker-device cpu
```

Batch size 8 is a **provisional, exploratory CPU-oriented default**, not a universal optimum. The profiling sample was small and showed timing variability. Re-profile on the hardware and workload that matter before selecting deployment settings.

### Latency benchmark tooling

The repository includes a dedicated latency runner and an exploratory Cross-Encoder batch-size profiler:

```cmd
uv run python scripts/run_latency.py --help
uv run python scripts/profile_reranker_batch_sizes.py --help
```

Check the script help for the current arguments and defaults before running a benchmark. Keep model loading, warm-up, cache behavior, query selection, trial count, hardware, and strategy ordering explicit in any reported latency result.

## Evaluation methodology

The evaluator supports document-level and chunk-level ranking protocols. It can save aggregate metrics together with per-query traces, input fingerprints, model metadata, runtime details, and the Git state observed during the run.

Metrics used in the project include:

- **Precision@K:** Fraction of the top-K results that are relevant.
- **Recall@K:** Fraction of relevant items retrieved in the top K.
- **MRR:** Mean reciprocal rank of the first relevant result, averaged over queries.
- **nDCG@K:** Ranking quality with position-sensitive gain; graded labels can be used where available.
- **Candidate Recall:** Fraction of relevant items present in the candidate pool passed to later stages.
- **Latency percentiles:** P50/P95/P99 where the benchmark protocol records them.

Document-level and chunk-level results are different evaluation protocols. Do not combine their metrics or confidence intervals in one comparison. A small query set is useful for debugging, but it is not a substitute for a representative held-out evaluation set.

## Recorded benchmark: SciFact document-level retrieval

The recorded pinned-model evaluation used the BEIR SciFact test split:

<details>
<summary>Benchmark scope and model configuration</summary>

- Dataset: BEIR SciFact test split.
- Evaluated queries: 300 / 300.
- Corpus: 5,183 unique documents and 5,183 searchable items.
- Evaluation level: document.
- Chunking: disabled (one searchable item per document).
- `k=10`, `retrieval_k=100`, `candidate_k=50`.
- Embedding model: `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.
- Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`, revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
- Model revisions were verified against the loaded model/local cached snapshot in the recorded run.
- Embedding cache: disabled for this canonical evaluation.

</details>

| Strategy | Precision@10 | Recall@10 | Recall@100 | Candidate Recall@50 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.0867 | 0.7816 | 0.8859 | 0.8704 | 0.6340 | 0.6646 |
| Dense | 0.0937 | 0.8296 | 0.9433 | 0.9283 | 0.6725 | 0.7057 |
| Hybrid (BM25 + Dense, RRF) | 0.0933 | 0.8331 | 0.9617 | 0.9406 | 0.6765 | 0.7103 |
| Hybrid + Cross-Encoder | 0.0920 | 0.8211 | 0.9617 | 0.9406 | 0.6568 | 0.6901 |

### Interpretation

- Hybrid retrieval had the highest displayed Recall@10, Recall@100, Candidate Recall@50, MRR@10, and nDCG@10 among these four recorded rows. The paired-bootstrap analysis conducted for this experiment did **not** establish the Dense-versus-Hybrid differences as statistically significant.
- In this run, the Cross-Encoder row was lower than Hybrid on Precision@10, Recall@10, MRR@10, and nDCG@10. It did not demonstrate a ranking-quality improvement on this dataset/configuration.
- These measurements describe one pinned SciFact evaluation. They do not establish performance on private enterprise documents, other domains, multilingual/Hinglish queries, concurrent workloads, or larger corpora.
- The evaluation latency fields in this result artifact are not the dedicated repeated-latency benchmark reported below and should not be presented as production latency.

## Dedicated repeated latency benchmark

A separate latency run used 30 distinct SciFact queries, five measured trials per strategy (150 samples per strategy), and 10 warm-up queries excluded from the measured samples. Model loading was outside the timed samples. Query embedding cache was disabled, document embedding cache was enabled, and strategy order rotated. The recorded environment was CPU-only (8 logical CPU cores, 4 PyTorch threads; CUDA unavailable).

| Strategy | Total latency P50 (ms) | P95 (ms) | P99 (ms) |
| --- | ---: | ---: | ---: |
| BM25 | 60.67 | 123.02 | 179.31 |
| Dense | 130.22 | 314.60 | 471.96 |
| Hybrid | 188.75 | 417.10 | 518.09 |
| Hybrid + Cross-Encoder | 8,845.14 | 12,718.23 | 15,545.64 |

For the recorded reranking run, the Cross-Encoder component itself had P50 latency of approximately **8,667.56 ms** and P95 of approximately **12,446.21 ms**. On this CPU environment, the hybrid-plus-reranker total P50 was about **46.9 times** the Hybrid P50. This is a measurement of the specific benchmark setup, not a universal multiplier or a production service SLA.

## Reranker candidate-depth and batch-size experiments

The candidate-depth diagnostic compares `candidate_k` values of 10, 20, and 50 on the first 30 of 300 SciFact queries. Each setting used one evaluation pass per query and strategy. Candidate recall increased as the pool grew, but the reranker did not improve the displayed ranking metrics and its CPU cost increased substantially with candidate depth.

| Candidate K | Hybrid Candidate Recall@K | Reranked MRR@10 | Reranked nDCG@10 | Reranker component P50 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.843 | 0.679 | 0.721 | 1,166.90 |
| 20 | 0.893 | 0.681 | 0.723 | 2,576.71 |
| 50 | 0.967 | 0.675 | 0.711 | 7,279.13 |

These are exploratory 30-query, single-pass comparisons—not repeated latency percentiles and not a final statistical claim. The full diagnostic, including the paired Hybrid comparison and end-to-end timings, is in [`docs/RERANKER_CANDIDATE_DEPTH_DIAGNOSTIC.md`](docs/RERANKER_CANDIDATE_DEPTH_DIAGNOSTIC.md).

A later 30-query `candidate_k=10`, batch-size-8 evaluation recorded Hybrid P50 of **29.75 ms** and Hybrid+Rerank P50 of **918.51 ms**. Ranking values were unchanged from the earlier candidate-K=10 run (reranked MRR@10 `0.679`, nDCG@10 `0.721`). This was a single evaluation pass, not the dedicated repeated-latency protocol; it should not be used to claim a validated speed-up over the earlier run.

### Current reranker decision

Keep Cross-Encoder reranking available as an **optional experiment**, and compare it with the Hybrid baseline on the target dataset before using it. The current evidence does not justify treating it as a default quality improvement. Small profiling runs are insufficient to select a universally optimal batch size.

## Design decisions and trade-offs

- **Why RRF?** BM25 scores and cosine similarities are on different scales. RRF combines rank positions without requiring score calibration.
- **Why exact cosine search?** It is simple to inspect and test. Exact matrix search scans the stored vectors; very large corpora may require an ANN index after workload-specific evaluation.
- **Why inject model functions?** Retrieval tests can run offline with deterministic functions, and model implementations can be replaced without rewriting ranking logic.
- **Why separate document and chunk evaluation?** Multiple chunks from one parent document can crowd other documents out. Document-level evaluation deduplicates to parent IDs; chunk-level evaluation measures chunk rankings.
- **Why keep reranking optional?** The recorded SciFact runs did not show a ranking gain, while CPU inference introduced significant additional latency.
- **Why immutable snapshots?** A new version can be built and validated before the active pointer changes. Failed builds should not expose incomplete snapshot contents as the active index.
- **Why preserve precomputed vectors?** Insert/update/delete should not re-embed unchanged documents, and re-normalizing already-validated rows can introduce unnecessary floating-point changes.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/DECISIONS.md`](docs/DECISIONS.md) for implementation details and engineering trade-offs.

## Current limitations

- The current document loader supports JSONL documents with `doc_id` and `text`, plus optional `source`, `title`, and `metadata`. A standalone TXT/PDF upload workflow is not implemented.
- Chunking supports configurable word-window chunks and whole-document mode. Token-aware or character-bounded chunking has not been implemented.
- Failure-injection tests cover handled build/activation failures and orphan-staging isolation. They do not establish automatic process-crash recovery or orphan cleanup. All immutable versions are currently retained; automatic pruning and configurable retention are not implemented.
- Activation uses a documented single-writer assumption; cross-process locking is not provided.
- The evaluation CLI evaluates all four strategies when no explicit `--strategies` list is supplied. The Cross-Encoder adapter may still be initialized even when reranking is excluded from the requested strategy list; lazy loading remains to be verified and improved.
- The Cross-Encoder has not demonstrated a ranking-quality improvement over Hybrid in the recorded SciFact experiments. Its CPU inference cost is substantial; treat it as an optional experiment.
- The hosted search API, grounded RAG answer generation, authentication, tenant isolation, source-level ACL enforcement, monitoring, and complete interviewer-facing demo are not implemented.
- Exact dense retrieval scans the stored vectors. Large-corpus scalability, production concurrency, and service-level latency have not been validated.
- The SciFact benchmark uses English scientific text and is not representative of every enterprise retrieval domain.
- No `LICENSE` file was confirmed in the supplied project documentation. Verify the repository before public release; until a license is selected and added, do not assume the project grants permission to reuse or redistribute the code.

## Roadmap

The roadmap is ordered to finish and freeze the local retrieval/index lifecycle before adding product infrastructure.

### Phase 1 — Close persistent indexing

1. **Step 5.6: failure recovery and immutable-version safety — accepted.** Tests cover embedding/build failure, artifact-write failure, version-promotion failure, pointer-replacement failure, invalid staged snapshots, rollback from a corrupt active snapshot, and orphan-staging isolation. Immutable versions are retained; automatic pruning, automatic crash recovery, and orphan cleanup are not implemented.
2. **Step 5.7: complete lifecycle regression.** Test rebuild → insert → update → delete → fresh-process reload → search, including corpus integrity, vector-row alignment, deterministic ordering, and rollback behavior.
3. **Phase 1 acceptance review.** Review the complete diff, run `git diff --check`, focused mutation tests, and the full suite. Document the single-writer limitation and recovery/retention semantics before calling Phase 1 frozen.
4. **Ingestion backlog.** Verify standalone TXT loading, typed timestamp metadata, malformed-document handling, source provenance, and character/token-based chunk limits. These items are not marked complete.

### Phase 2 — Retrieval refinement

- Verify Cross-Encoder default-OFF behavior and lazy model loading.
- Persist or expose pre-rerank and post-rerank result IDs for traceability.
- Compare candidate depths such as K=10, 20, and 50 on a suitable evaluation set.
- Verify deterministic tie-breaking, duplicate handling, and source provenance.
- Freeze model revisions and retrieval configuration with evidence-backed defaults.

### Phase 3 — Evaluation and benchmarking

- Investigate the top 10–20 retrieval failures.
- Report query-embedding, BM25, Dense, RRF, reranking, and total stage latency separately.
- Verify metric definitions and implementations, including MRR@K, MAP@K, and HitRate@K if those metrics are adopted.
- Record hardware/software provenance and ensure model loading is outside steady-state timing.
- Create an independent untouched evaluation set with 100+ diverse, human-reviewed domain queries.
- Preserve raw results, per-query traces, dataset fingerprints, and model/configuration revisions.

### Phase 4 — Grounded RAG

- Add a provider-neutral LLM interface, deterministic fake provider, and one real provider adapter.
- Build an evidence-aware context selector with explicit token budgeting.
- Return citations tied to actual document and chunk identifiers.
- Abstain when retrieved evidence is insufficient.
- Version prompts and treat retrieved documents as untrusted input.
- Add prompt-injection tests and an independent RAG evaluation set for answer correctness, groundedness, citation quality, relevance, and abstention.

### Phase 5 — FastAPI service

Keep retrieval/RAG logic independent of the HTTP transport and expose:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Process health |
| GET | `/ready` | Dependency/model readiness |
| POST | `/v1/search` | Hybrid retrieval |
| POST | `/v1/compare` | BM25 / Dense / Hybrid comparison |
| POST | `/v1/answer` | Grounded RAG answer |

Add request/response schemas, validation, stable error contracts, startup initialization, timeouts, request limits, structured logs, request IDs, and stage timings.

### Phase 6 — Interviewer-facing demo

Build a browser demo where an interviewer can submit a query, compare BM25/Dense/Hybrid rankings, inspect source metadata and ranking changes, view grounded answers with citations, see abstention when evidence is insufficient, and inspect latency. Select UI technology only after the API contract is stable.

### Phases 7–8 — CI, deployment, and portfolio release

- Verify a clean clone and install on the supported Python versions.
- Add CI for tests, linting, and dependency/security checks.
- Add Docker and documented local startup.
- Deploy a public demo/API and run remote health, readiness, search, and answer smoke tests.
- Document logs, request limits, secrets, rollback, and operational constraints.
- Complete the architecture diagram, ADRs, `LEARN.md`, benchmark methodology, limitations, reproduction steps, and demo walkthrough.
- Publish a versioned release with durable benchmark artifacts.

### Phases 9–10 — Enterprise readiness and customer validation

These are separate from a public portfolio demo. Before serving confidential customer data, implement and validate authentication, authorization, tenant isolation, document-level ACLs, SSO/OIDC where required, encryption/key management, retention/deletion, audit logs, rate limits, monitoring, backup/restore, load testing, and a security assessment. A customer pilot then requires an agreed corpus, a human-judged relevance set, acceptance metrics, and a measured customer outcome.

## Repository documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/DECISIONS.md`](docs/DECISIONS.md)
- [`docs/RERANKER_CANDIDATE_DEPTH_DIAGNOSTIC.md`](docs/RERANKER_CANDIDATE_DEPTH_DIAGNOSTIC.md)
- [`docs/index.html`](docs/index.html)
- [`docs/SOCIAL_POSTS.md`](docs/SOCIAL_POSTS.md)

## Reproducibility notes

The benchmark values in this README are transcribed from the recorded evaluation and latency artifacts described in the supplied project documentation. The `results/` directory was not confirmed as tracked or available to a fresh clone in the supplied repository context. Keep raw artifacts in a durable, versioned location before presenting the benchmark as independently reproducible.

When publishing a new benchmark, retain corpus and query fingerprints, model IDs and immutable revisions, cache configuration, chunking protocol, candidate depth, hardware/software environment, warm-up policy, trial count, and per-query traces. Do not mix document-level and chunk-level evaluation results or treat fake-model smoke tests as retrieval-quality evidence.

---

Feedback on retrieval methodology, evaluation design, fusion, and reranking is welcome.
