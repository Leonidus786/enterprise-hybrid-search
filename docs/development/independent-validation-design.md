# Independent Domain Validation Design

**Status:** DESIGN DRAFT — pending domain selection and preregistration

**Scope:** Phase 3, Step 3.13. This document specifies the next independent validation programme. It does not claim that an independent dataset has been collected or evaluated.

## 1. Objective

Establish an independent, domain-specific retrieval evaluation dataset with human-verified relevance labels and an untouched final query holdout. The existing SciFact evaluation remains a reproducibility baseline and is not the independent validation dataset.

## 2. Dataset requirements

- Obtain a legally approved, versioned document corpus representative of the intended business domain. Record its source, owner, access permissions, collection date, document count, deduplication policy and SHA-256 fingerprint.
- Source queries from genuine information needs and representative user workflows. Record query provenance and remove personal or confidential information where required.
- Do not reuse the inspected SciFact queries, labels or benchmark-derived query formulations as independent test queries.
- Freeze document identities, corpus version, metadata and query/relevance fingerprints before evaluation.
- Preserve a documented data dictionary and dataset-card style description, including exclusions and known coverage gaps.

## 3. Query separation and leakage controls

- Partition query families, user tasks or sessions into development, validation and final holdout groups. Closely related paraphrases and duplicate intents must remain in the same partition.
- The same frozen searchable corpus may be used across query partitions for a document-retrieval task; query and relevance-label separation is mandatory. Any temporal or unseen-document generalization claim requires a separately defined corpus split.
- Deduplicate exact queries and review near-duplicate intents before the split is finalized. Record the method and reviewed exclusions.
- Keep final holdout queries and relevance labels under an independent custodian. Developers and model selectors must not inspect them before configuration freeze.
- If holdout queries or labels are inspected for tuning, retire that holdout from confirmatory use and obtain a new untouched holdout.

## 4. Human relevance judgments

- Prepare a written annotation guide with query intent, document relevance criteria, examples, edge cases and treatment of ambiguous or unanswerable queries.
- Use binary relevance labels for the primary metrics to preserve compatibility with the current evaluator. A relevant document must materially satisfy the information need, not merely share topical vocabulary.
- Have two assessors label each query-document judgment independently and without seeing system identity, ranking position or model scores.
- Adjudicate disagreements through a designated domain expert. Preserve original labels, adjudicated labels and adjudication rationale.
- Report raw agreement and an appropriate chance-corrected agreement statistic. Review low-agreement topics and revise the annotation guide using development data only.
- Graded labels may be collected as a supplementary layer, but graded nDCG must remain separately named and must not be confused with the binary nDCG used in the existing evaluation.

## 5. Evaluation protocol and metrics

- Freeze corpus, queries, relevance labels, retrieval configuration, model identifiers and revisions, code commit, dependencies and hardware before opening the holdout.
- Use binary nDCG@10 as the primary ranking metric.
- Report Recall@100 and Candidate Recall@50 as coverage guardrails.
- Report Precision@10, Recall@10, MRR@10 and HitRate@10 as secondary descriptive metrics.
- Compare predeclared retrieval strategies on identical queries and corpus snapshots. Keep candidate generation and reranking effects separately measurable.
- Use paired query-level statistical comparisons with effect sizes and confidence intervals. Declare the hypothesis family, resampling method, seed, confidence level and multiplicity correction before evaluation.
- Use a development-only pilot to estimate variance and determine the required holdout sample size through a documented power or precision analysis. Do not assume that 300 queries is automatically sufficient.
- Define minimum practical improvement, non-inferiority margins and coverage guardrails before holdout evaluation. Statistical significance alone does not establish product value.

## 6. Latency and operational measurement

- Measure on hardware representative of the intended deployment environment.
- Predeclare concurrency, warm-up, cache policy, batch size, timeout and repeated-trial protocol.
- Record total, retrieval and reranking latency distributions, including p50, p95 and p99.
- Define the acceptable latency SLO and compute-cost budget with the intended product owner before unblinding the holdout.
- Report local benchmark latency separately from deployed-service latency.

## 7. Acceptance gates

- Dataset provenance, permissions, corpus fingerprint and query fingerprint are complete.
- Query-family separation, deduplication and holdout access controls pass review.
- Human annotation and adjudication are complete, with agreement reported.
- Sample size, primary metric, practical effect threshold, statistical procedure and latency SLO are preregistered.
- All compared configurations are frozen before holdout evaluation.
- Quality, coverage, latency and cost meet the predeclared domain-specific gates. Any failed gate is reported rather than silently relaxed.
- The final report includes all queries, failures, exclusions, uncertainty intervals and known limitations.

## 8. RAG evaluation remains separate

Retrieval validation does not establish answer correctness, faithfulness, citation accuracy or safe abstention. Those outcomes require a separate answer-level dataset and human-reviewed evaluation protocol during the Grounded RAG phase.

## 9. Decisions still required before preregistration

- Intended business domain and representative use cases.
- Corpus owner, approved source, access permissions and data retention policy.
- Query population, sampling frame and collection method.
- Annotation team, adjudication owner and annotation budget.
- Development and holdout sample sizes, justified by pilot variance and power or precision analysis.
- Primary comparison, minimum practically important effect and non-inferiority margins.
- Required candidate coverage and latency SLO.
- Independent holdout custodian and access procedure.

## 10. Source and scope

This design implements the independent-validation requirement in the frozen Phase 3 evaluation protocol. It is a design specification, not evidence of dataset collection, human annotation or independent evaluation.

Frozen protocol SHA-256: `cb3f608123bdfdb98a59f5e34331ca1f29773574c5af868742b9a67f2db3c009`
