# Problem Statement: Enterprise Hybrid Search and Reranking

## 1. Background

Enterprise knowledge is often distributed across technical documentation, internal guides, runbooks, support notes, and other text documents. Users rarely search using the exact wording found in those documents. They may describe a symptom, use a synonym, or know only part of an error message. A retrieval system based only on exact keyword matching can miss semantically relevant documents, while a dense-vector-only system can miss exact identifiers, product names, error codes, and other lexical signals.

## 2. Problem

Build and evaluate a reproducible retrieval pipeline that combines lexical and semantic search to return relevant documents for a natural-language query. The system must also support an optional second-stage reranker and provide a consistent evaluation harness so the retrieval strategies can be compared on the same labeled query set.

The core engineering question is:

**Does combining BM25 keyword retrieval with dense embedding retrieval, using Reciprocal Rank Fusion (RRF), and optionally applying a cross-encoder reranker improve retrieval quality enough to justify its additional latency and complexity on a representative, labeled corpus?**

## 3. Objectives

1. Implement BM25 sparse retrieval for exact-term and lexical matching.
2. Implement dense retrieval using sentence embeddings and cosine similarity.
3. Combine sparse and dense result rankings with Reciprocal Rank Fusion (RRF), without assuming their raw scores are directly comparable.
4. Optionally rerank a candidate pool with a cross-encoder that scores query-document pairs.
5. Support document-level and chunk-level retrieval/evaluation, keeping their interpretation explicit.
6. Validate corpus records, query records, relevance labels, and configuration parameters with clear errors.
7. Compare retrieval strategies using a repeatable benchmark and report quality and latency metrics.

## 4. Intended users and use cases

The intended users are engineers building or evaluating search for internal knowledge bases, technical documentation, troubleshooting guides, and other text-heavy collections. Example queries include finding a solution to a technical symptom, retrieving a document containing a precise error code, or finding relevant content when the query uses different wording from the source text.

## 5. Proposed solution

The pipeline supports four strategies:

- **BM25:** lexical retrieval based on query-term matching and document statistics.
- **Dense:** semantic retrieval using normalized embeddings and cosine similarity.
- **Hybrid (RRF):** combine the rankings from BM25 and dense retrieval using Reciprocal Rank Fusion.
- **Hybrid + reranker:** retrieve a candidate pool with hybrid search, then rerank its query-document pairs with a cross-encoder.

The project also includes data-loading validation, chunking, evaluation metrics, command-line evaluation, unit tests, and documented design decisions.

## 6. Input and output contract

### Corpus

The corpus is a JSONL file with one document per line:

```json
{"doc_id": "example_doc", "text": "Document text goes here."}
```

Each `doc_id` must be non-empty and unique. Each `text` value must be a string. Documents may be split into overlapping word-window chunks for retrieval.

### Labeled queries

The query file is a JSON array. Each record contains a unique query ID, query text, and a non-empty list of relevant document IDs:

```json
{
  "id": "q1",
  "query": "How do I diagnose this error?",
  "relevant": ["example_doc"]
}
```

Relevant IDs must exist in the corpus. Labels should be reviewed by a human; they are the reference standard for evaluating retrieval quality.

### Output

For a query and selected strategy, the system returns ranked chunk IDs or unique document IDs, depending on the chosen retrieval/evaluation level. The evaluation command reports metrics for each strategy over the same queries.

## 7. Evaluation plan

Evaluate all strategies on the same corpus, query set, relevance labels, and cutoff K. Report:

- **Precision@K:** fraction of the top-K results judged relevant.
- **Recall@K:** fraction of known relevant items retrieved in the top K.
- **MRR:** reciprocal rank of the first relevant result, averaged across queries.
- **nDCG@K:** ranking quality with the current binary relevance labels.
- **Latency p50 and p95:** median and 95th-percentile query latency, measured after a warm-up run.

Run both document-level and chunk-level evaluation when useful, and state clearly which level each result represents. Tune the reranker candidate-pool size as a separate experiment, since it affects both recall opportunity and latency.

The bundled sample dataset is a smoke-test fixture, not evidence of enterprise search quality. A meaningful initial benchmark should use a representative corpus and at least 50 carefully labeled queries as a practical starting point; more queries and coverage of different query types are preferable when possible. Record model names, versions, environment, data version, K, candidate-pool size, and run configuration with every benchmark.

## 8. Success criteria

The project is successful when:

1. Unit and regression tests pass on a clean environment.
2. Invalid data and configuration fail early with understandable error messages.
3. The same benchmark can be rerun with recorded settings and produce the same metric definitions and comparable results.
4. All four strategies are evaluated on identical labels and data.
5. The results quantify the quality/latency trade-off rather than claiming hybrid search is automatically better.
6. Any conclusions about real retrieval quality are based on real embedding/reranker models and a representative labeled dataset—not fake-model smoke-test numbers.

## 9. Constraints and current limitations

- Dense retrieval currently uses exact in-memory vector search; large-scale approximate-nearest-neighbor indexing and persistence are outside the current baseline.
- Real model downloads and real-model quality/performance must be verified in the target environment.
- The bundled dataset is small and synthetic/toy-sized; its scores must not be generalized to production workloads.
- API serving, authentication, multi-tenancy, monitoring, production deployment, and large-corpus scalability are not considered complete unless implemented and separately tested.
- Cross-encoder reranking adds compute and latency and only reranks the retrieved candidate pool; relevant items missing from that pool cannot be recovered by reranking alone.

## 10. Deliverables

- Modular Python package for chunking, sparse retrieval, dense retrieval, fusion, reranking, and evaluation.
- Automated unit and regression tests.
- Command-line benchmark for fake/offline smoke tests and real-model runs.
- Architecture and decision records explaining key design choices and known limitations.
- A benchmark report with labeled data, configuration, retrieval metrics, latency, and a discussion of trade-offs.

## 11. Out of scope for the initial milestone

A hosted search API, UI, authentication/authorization, multi-tenant isolation, distributed indexing, automated relevance labeling, and claims of production-grade throughput are out of scope for the initial retrieval/evaluation milestone. These can be added after retrieval quality and system constraints have been measured.
