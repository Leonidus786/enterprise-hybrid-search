# Enterprise Hybrid Search

**A practical Python project for combining keyword search, semantic search, and reranking.**

Enterprise Hybrid Search explores a common retrieval problem: keyword search is good at finding exact terms, IDs, and error codes, while dense retrieval can find related meaning even when the wording differs. This project runs both retrievers, combines their rankings with Reciprocal Rank Fusion (RRF), and can optionally rerank the candidates with a cross-encoder.

> **Current status:** The retrieval pipeline and evaluation CLI are implemented, with 130 passing baseline tests. Fake-model evaluation is for smoke testing only. A real-model evaluation has run on the small sample dataset, but a representative external benchmark has not yet been validated. A hosted search API and complete browser UI are still pending.

<p align="center">
  <img src="assets/enterprise-hybrid-search-social-card.png" alt="Enterprise Hybrid Search: BM25 and dense retrieval combined with RRF and optional reranking" width="100%" />
</p>

## What it includes

- **Text chunking:** word-window chunks with configurable size and overlap.
- **BM25 sparse search:** exact token matching with Lucene-style IDF, term-frequency saturation, and document-length normalisation.
- **Dense search:** exact cosine-similarity search with NumPy; the embedding function is injected, so tests do not need a model download.
- **Hybrid ranking:** Reciprocal Rank Fusion (RRF) combines sparse and dense rankings without adding scores from different scales directly.
- **Optional reranking:** a cross-encoder scores a smaller set of fused candidates.
- **Evaluation:** Precision@K, Recall@K, MRR, nDCG@K, and p50/p95 latency, with separate document-level and chunk-level evaluation paths.
- **Input validation:** checks for corpus/query schemas, duplicate IDs, invalid parameters, unexpected model output shapes, and non-finite scores.
- **Automated tests:** unit, edge-case, and end-to-end tests; GitHub Actions CI is configured.

## Retrieval flow

```text
JSONL documents
      |
      v
 word-window chunking
      |
      +----------------------------+
      |                            |
      v                            v
 BM25 sparse search        Dense cosine search
      |                            |
      +-------------+--------------+
                    |
                    v
           RRF rank fusion
                    |
                    v
       Optional cross-encoder rerank
                    |
                    v
          Ranked chunk results
                    |
                    v
      Evaluation: P@K, R@K, MRR,
             nDCG@K, p50/p95
```

Read more: [Architecture](docs/ARCHITECTURE.md) · [Engineering decisions](docs/DECISIONS.md) · [Project overview page](https://Leonidus786.github.io/enterprise-hybrid-search/).

## Quick start

Python 3.10 or newer is required. The commands below use [`uv`](https://docs.astral.sh/uv/) and work from the project root.

### 1. Install the project and development dependencies

```bash
uv sync --extra dev
```

### 2. Run the tests

```bash
uv run python -m pytest -q
```

The current local baseline is **130 passing tests**. Runtime depends on the machine and environment.

### 3. Run the offline evaluation smoke test

```bash
uv run python scripts/run_eval.py --fake
```

This runs the BM25, dense, hybrid, and hybrid-plus-reranking paths using deterministic fake functions. It is useful for checking that the pipeline and metrics execute end to end. **Do not use these fake-model scores to claim retrieval quality or compare real models.**

### 4. Run with real embedding and reranking models

Install the optional model dependencies:

```bash
uv sync --extra models --extra dev
```

Then run:

```bash
uv run python scripts/run_eval.py
```

The first real-model run downloads the configured Sentence Transformers models and may take time and disk space. By default, the script uses:

- Embeddings: `BAAI/bge-small-en-v1.5`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

A real-model evaluation has completed on the small sample corpus, but these results are not a representative benchmark and do not establish general retrieval quality. Review the data and relevance labels before treating the output as a benchmark. For CPU/GPU memory constraints, see the [Sentence Transformers documentation](https://www.sbert.net/).

### Install with pip instead

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/run_eval.py --fake
```

For real-model evaluation, install the model extra too:

```bash
python -m pip install -e ".[models,dev]"
python scripts/run_eval.py
```

## Evaluation data

The repository includes a small example corpus of **14 documents** at `data/sample/corpus.jsonl` and **10 labeled queries** at `data/sample/queries.json`. This is a learning/evaluation starter set, not a representative enterprise dataset.

With the default `--k 10`, the printed report includes **Recall@10** (shown as `recall@10`), Precision@10, MRR, nDCG@10, and p50/p95 latency. These are metric names, not a published real-model result.

Each corpus JSONL row has this shape:

```json
{ "doc_id": "example_doc", "text": "Text to index goes here." }
```

Each query entry has this shape:

```json
{
  "id": "q1",
  "query": "How do I find the relevant passage?",
  "relevant": ["example_doc"]
}
```

The `relevant` values are document IDs for document-level evaluation. Use chunk IDs only with the chunk-level protocol. The data is a small starter set, not a representative enterprise benchmark. For meaningful claims, build a representative corpus, label relevant results, and report the real-model results with the model names, settings, dataset, and latency environment.

### Useful CLI options

```bash
# Change the evaluation cutoff
uv run python scripts/run_eval.py --k 5 --fake

# Change how many fused chunks are passed to the reranker
uv run python scripts/run_eval.py --candidate-k 100 --fake

# Evaluate chunk IDs rather than unique parent documents
uv run python scripts/run_eval.py --level chunk --fake

# Use real models but disable the BGE query-side instruction
uv run python scripts/run_eval.py --no-bge-prefix
```

The `--fake` flag is for pipeline smoke tests only. The `--no-bge-prefix` option applies to real-model runs; it is not a claim that one prefix setting is universally better.

## Design notes

- **Why RRF?** BM25 scores and cosine similarities live on different scales. RRF combines rank positions without requiring score calibration.
- **Why exact cosine search?** It keeps the implementation small and straightforward to test. Very large corpora may need an approximate nearest-neighbour index or vector database.
- **Why inject models?** Tests stay fast and offline, and embedding/reranking implementations can be swapped without rewriting retrieval logic.
- **Why separate document-level and chunk-level evaluation?** Ranking chunks and then deduplicating can allow many chunks from one document to crowd other documents out. The evaluation code keeps these protocols separate.

These are engineering choices, not claims that the current defaults are optimal. See [DECISIONS.md](docs/DECISIONS.md) for trade-offs and experiments still to run.

## Current limitations

- No hosted API, authentication, persistent vector database, or complete browser search UI is included yet.
- Ingestion currently reads JSONL documents and chunks their text. It does **not** currently provide direct PDF/TXT upload UI.
- The example corpus and labels are small and should not be treated as an enterprise benchmark.
- Fake-model scores are not evidence of real retrieval quality.
- Exact dense search scans all stored vectors; large-corpus deployment will require performance testing and may need ANN/vector-database support.
- A `LICENSE` file has not been added yet. Until a license is chosen and included, do not assume others have permission to reuse or redistribute the code.

## Roadmap

1. Run and document real-model benchmarks on a more representative, labeled corpus.
2. Study candidate-pool size versus reranking latency and retrieval quality.
3. Add a small API and user-facing search demo.
4. Add document ingestion, persistence, deployment, and access controls only after the retrieval baseline is measured.

## Project links

- [Architecture](docs/ARCHITECTURE.md)
- [Engineering decisions](docs/DECISIONS.md)
- [Static project page and toy RRF explainer](docs/index.html)
- [Social launch copy](docs/SOCIAL_POSTS.md)

If you try the project, feedback on evaluation design, fusion, or reranking is welcome. The next update will focus on real-model measurements rather than unverified performance claims.

<!-- BEGIN GENERATED HYBRID SEARCH BENCHMARK -->
## Benchmark results

**Dataset:** SciFact test | **Queries:** 300 / 300 | **Corpus documents:** 5183 | **Searchable items:** 5183.
**Protocol:** doc-level; chunking configuration recorded in JSON | **K:** 10 | **Retrieval K:** 100 | **Candidate K:** 50.

| Strategy | Precision@10 | Recall@10 | Recall@100 | Candidate Recall@50 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.0867 | 0.7816 | 0.8859 | 0.8704 | 0.6340 | 0.6646 |
| Dense | 0.0937 | 0.8296 | 0.9433 | 0.9283 | 0.6725 | 0.7057 |
| Hybrid (BM25 + Dense, RRF) | 0.0933 | 0.8331 | 0.9617 | 0.9406 | 0.6765 | 0.7103 |
| Hybrid + Cross-Encoder | 0.0920 | 0.8211 | 0.9617 | 0.9406 | 0.6568 | 0.6901 |

### Reproducibility metadata
- Embedding model: `BAAI/bge-small-en-v1.5`; revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; revision verified: `True`.
- Reranker model: `cross-encoder/ms-marco-MiniLM-L-6-v2`; revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`; revision verified: `True`.
- Persistent embedding cache setting: `disabled`.
- Git commit: `3afc66099fe2cf5ec2d7c218d562ef3ca82c6cdb`; working-tree state recorded as `clean`.
- Source artifact: `results/scifact_doclevel_pinned_v2.json`.

### Interpretation and limitations
- These are observed results for this SciFact configuration, not evidence of state-of-the-art performance or enterprise-domain effectiveness.
- Keep document-level and chunked experiments in separate tables; do not combine their metrics or confidence intervals.
- The Cross-Encoder result is an experiment, not the default recommendation: on the recorded run it did not improve the displayed ranking metrics and added substantial CPU inference time. It should remain optional/off by default until it demonstrates a measured gain on the target corpus.
- The latency values in an evaluation JSON are not a production latency claim. Publish P50/P95 only from the dedicated repeated, warm-model, interleaved latency protocol with uncached query embeddings and recorded hardware.
- The benchmark uses English scientific text. It does not establish performance on private enterprise documents, other domains, multilingual/Hinglish queries, concurrent traffic, or million-document corpora.
- This repository is a retrieval/evaluation toolkit prototype. Do not describe it as enterprise-ready or production-ready until authentication, tenant isolation, source ACL enforcement, deletion, monitoring, recovery, and security controls have been implemented and tested.

> Numbers are generated from the result JSON. Do not hand-edit the table; rerun the generator after creating a new benchmark artifact.
<!-- END GENERATED HYBRID SEARCH BENCHMARK -->
