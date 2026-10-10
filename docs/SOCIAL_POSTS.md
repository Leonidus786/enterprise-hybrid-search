# Social launch posts

Before posting, replace `Leonidus786` in the GitHub URL with the actual GitHub username. Do not use the placeholder URL publicly.

Repository URL: `https://github.com/Leonidus786/enterprise-hybrid-search`

## LinkedIn

I’ve been building a project called **Enterprise Hybrid Search** to understand a practical problem in information retrieval: keyword search is great for exact terms, but it can miss a relevant passage when the wording is different.

The project combines:

- **BM25** for keyword-based retrieval
- **Dense cosine search** for embedding-based retrieval
- **Reciprocal Rank Fusion (RRF)** to combine rankings
- **Optional cross-encoder reranking** for a second-stage score
- Evaluation with Precision@K, Recall@K, MRR, nDCG@K, and p50/p95 latency

I’ve also added chunking, input and model-output validation, separate document-level and chunk-level evaluation, and tests for edge cases. The current test suite passes **130 tests** on my local setup.

One important note: the offline evaluation uses fake models to test the pipeline. I haven’t verified real-model retrieval quality yet, so I’m not presenting those scores as proof of performance. That’s the next step, followed by a simple user-facing demo.

Code and project notes: https://github.com/Leonidus786/enterprise-hybrid-search

I’ll share updates as I measure real-model results and improve the project. If you work with search, RAG, or retrieval evaluation, I’d genuinely appreciate feedback on the design and evaluation approach.

#Python #InformationRetrieval #RAG #MachineLearning

## X / Twitter — main post

I’m building Enterprise Hybrid Search: BM25 + dense embeddings, combined with RRF and optional cross-encoder reranking. The pipeline has 130 passing tests; real-model quality evaluation is still next.

Code: https://github.com/Leonidus786/enterprise-hybrid-search

#Python #RAG

## X / Twitter — optional thread

**Post 2**

So far: word-window chunking, BM25 with Lucene-style IDF, exact cosine search, RRF fusion, optional reranking, and evaluation for Precision@K, Recall@K, MRR, nDCG@K, and p50/p95 latency. Tests use deterministic fake models so they run quickly without downloads.

**Post 3**

The important caveat: fake-model scores are only a smoke test, not proof that retrieval is good. Next I’m validating BGE + MiniLM on labeled queries, then I’ll work toward a small live search demo. Feedback welcome: https://github.com/Leonidus786/enterprise-hybrid-search

## Suggested first comment on LinkedIn

The project page includes a small interactive RRF explainer. It demonstrates how ranked lists are fused using sample rankings; it is not a hosted endpoint for the Python search engine. I’ll update the project as the real-model benchmark and live demo take shape.
