import math
import time

from .metrics import (
    graded_ndcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .validate import positive_int


def percentile(values, p):
    """Linearly interpolated percentile, where p is in [0, 100]."""
    if not values:
        raise ValueError("no values")

    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")

    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    lo = math.floor(position)
    hi = math.ceil(position)

    return ordered[lo] + (
        ordered[hi] - ordered[lo]
    ) * (position - lo)


def _candidate_results(pipeline, ranked, level, candidate_k):
    """Get the pre-reranking candidate pool in the evaluation ID space.

    Pipeline candidate IDs are chunk IDs. For document-level evaluation,
    convert them to unique parent document IDs without changing their order.
    """
    candidate_ids = getattr(
        pipeline,
        "last_candidate_ids",
        None,
    )

    # Compatibility with simple pipeline implementations that do not
    # expose their pre-reranking candidate set.
    if candidate_ids is None:
        return ranked[:candidate_k]

    if level == "chunk":
        return list(candidate_ids)

    if not hasattr(pipeline, "doc_of"):
        return ranked[:candidate_k]

    documents = []
    seen = set()

    for chunk_id in candidate_ids:
        doc_id = pipeline.doc_of(chunk_id)

        if doc_id not in seen:
            seen.add(doc_id)
            documents.append(doc_id)

    return documents


def evaluate(
    pipeline,
    queries,
    modes,
    k=10,
    warmup=True,
    level="doc",
    retrieval_k=100,
):
    """Evaluate retrieval quality, candidate recall, and latency.

    Ranking metrics:
      - Precision@k
      - Recall@k
      - Recall@retrieval_k
      - Candidate Recall@candidate_k (before reranking)
      - MRR@k
      - nDCG@k

    Latency metrics:
      - latency_p50_ms / latency_p95_ms: end-to-end search latency
      - retrieval_latency_p50_ms / p95: retrieval component
      - rerank_latency_p50_ms / p95: cross-encoder component

    Component latency is read from pipeline.last_timing_ms. For a pipeline
    without component instrumentation, those component timings are zero.
    """
    positive_int("k", k)
    positive_int("retrieval_k", retrieval_k)

    if retrieval_k < k:
        raise ValueError(
            "retrieval_k must be greater than or equal to k"
        )

    if level not in ("doc", "chunk"):
        raise ValueError('level must be "doc" or "chunk"')

    if not queries:
        raise ValueError("no queries to evaluate")

    modes = list(modes)

    if not modes:
        raise ValueError("no modes to evaluate")

    candidate_k = getattr(
        pipeline,
        "candidate_k",
        retrieval_k,
    )
    positive_int("candidate_k", candidate_k)

    known = (
        pipeline.doc_ids
        if level == "doc"
        else pipeline.chunk_ids
    )

    # Validate every relevance label before performing searches.
    for query_item in queries:
        relevant = query_item.get("relevant")

        if not relevant:
            raise ValueError(
                f"query {query_item.get('id')} has no relevance labels"
            )

        missing = set(relevant) - known

        if missing:
            raise ValueError(
                f"query {query_item.get('id')}: labels not in corpus "
                f"({level} level): {sorted(missing)}"
            )

    search = (
        pipeline.search_docs
        if level == "doc"
        else pipeline.search
    )

    # Ensure both retrieval depth and candidate depth are available.
    search_depth = max(retrieval_k, candidate_k)

    output = {}

    for mode in modes:
        if warmup:
            search(
                queries[0]["query"],
                mode=mode,
                k=search_depth,
            )

        precision_sum = 0.0
        recall_k_sum = 0.0
        recall_retrieval_sum = 0.0
        candidate_recall_sum = 0.0
        mrr_sum = 0.0
        ndcg_sum = 0.0

        total_latencies = []
        retrieval_latencies = []
        rerank_latencies = []

        for query_item in queries:
            relevant = set(query_item["relevant"])

            start = time.perf_counter()

            ranked = search(
                query_item["query"],
                mode=mode,
                k=search_depth,
            )

            total_ms = (
                time.perf_counter() - start
            ) * 1000

            total_latencies.append(total_ms)

            # Read timing components recorded by HybridPipeline.
            component_times = getattr(
                pipeline,
                "last_timing_ms",
                {},
            )

            retrieval_latencies.append(
                float(component_times.get("retrieval", 0.0))
            )
            rerank_latencies.append(
                float(component_times.get("rerank", 0.0))
            )

            top_k = ranked[:k]

            precision_sum += precision_at_k(
                top_k,
                relevant,
                k,
            )
            recall_k_sum += recall_at_k(
                top_k,
                relevant,
                k,
            )
            mrr_sum += reciprocal_rank(
                top_k,
                relevant,
            )

            grades = query_item.get("grades")

            if level == "doc" and grades:
                ndcg_sum += graded_ndcg_at_k(
                    top_k,
                    grades,
                    k,
                )
            else:
                ndcg_sum += ndcg_at_k(
                    top_k,
                    relevant,
                    k,
                )

            # Recall at the requested final retrieval depth.
            recall_retrieval_sum += recall_at_k(
                ranked[:retrieval_k],
                relevant,
                retrieval_k,
            )

            # Recall in the candidate set before any reranking.
            candidates = _candidate_results(
                pipeline,
                ranked,
                level,
                candidate_k,
            )

            candidate_recall_sum += recall_at_k(
                candidates,
                relevant,
                candidate_k,
            )

        count = len(queries)

        output[mode] = {
            f"precision@{k}": precision_sum / count,
            f"recall@{k}": recall_k_sum / count,
            f"recall@{retrieval_k}": (
                recall_retrieval_sum / count
            ),
            f"candidate_recall@{candidate_k}": (
                candidate_recall_sum / count
            ),
            # Compatibility: MRR is calculated on the top-k ranking.
            "mrr": mrr_sum / count,
            f"ndcg@{k}": ndcg_sum / count,

            # End-to-end search latency.
            "latency_p50_ms": percentile(
                total_latencies, 50
            ),
            "latency_p95_ms": percentile(
                total_latencies, 95
            ),

            # Component latency breakdown.
            "retrieval_latency_p50_ms": percentile(
                retrieval_latencies, 50
            ),
            "retrieval_latency_p95_ms": percentile(
                retrieval_latencies, 95
            ),
            "rerank_latency_p50_ms": percentile(
                rerank_latencies, 50
            ),
            "rerank_latency_p95_ms": percentile(
                rerank_latencies, 95
            ),
        }

    return output
