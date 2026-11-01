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
