import hashlib
import math
import time

from .per_query import _score_one_ranking
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
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _candidate_results(pipeline, ranked, level, candidate_k):
    """Get pre-reranking candidates in the evaluation ID space.

    Candidate IDs are usually chunk IDs. At document level, map chunk IDs to
    unique parent documents while preserving ranking order.
    """
    candidate_ids = getattr(pipeline, "last_candidate_ids", None)
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


def _evaluate(
    pipeline, queries, modes, k, warmup, level, retrieval_k,
    collect_per_query, include_query_text=False,
):
    positive_int("k", k)
    positive_int("retrieval_k", retrieval_k)
    if retrieval_k < k:
        raise ValueError("retrieval_k must be greater than or equal to k")
    if level not in ("doc", "chunk"):
        raise ValueError('level must be "doc" or "chunk"')
    if not queries:
        raise ValueError("no queries to evaluate")

    modes = list(modes)
    if not modes:
        raise ValueError("no modes to evaluate")

    candidate_k = getattr(pipeline, "candidate_k", retrieval_k)
    positive_int("candidate_k", candidate_k)
    known = pipeline.doc_ids if level == "doc" else pipeline.chunk_ids

    # Validate relevance labels before performing any searches.
    seen_query_ids = set()
    for query_item in queries:
        relevant = query_item.get("relevant")
        if not relevant:
            raise ValueError(f"query {query_item.get('id')} has no relevance labels")
        missing = set(relevant) - known
        if missing:
            raise ValueError(
                f"query {query_item.get('id')}: labels not in corpus "
                f"({level} level): {sorted(missing)}"
            )
        if collect_per_query:
            qid = query_item.get("id")
            if not isinstance(qid, str) or not qid.strip():
                raise ValueError("per-query output requires each query to have a non-empty string id")
            if qid in seen_query_ids:
                raise ValueError(f"duplicate query id {qid!r}")
            seen_query_ids.add(qid)

    search = pipeline.search_docs if level == "doc" else pipeline.search
    search_depth = max(retrieval_k, candidate_k)
    aggregate_output = {}
    per_query_output = {}

    for mode in modes:
        if warmup:
            search(queries[0]["query"], mode=mode, k=search_depth)

        totals = {
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            f"recall@{retrieval_k}": 0.0,
            f"candidate_recall@{candidate_k}": 0.0,
            "mrr": 0.0,
            f"ndcg@{k}": 0.0,
        }
        total_latencies = []
        retrieval_latencies = []
        rerank_latencies = []
        mode_per_query = {}

        for query_item in queries:
            start = time.perf_counter()
            ranked = search(query_item["query"], mode=mode, k=search_depth)
            total_ms = (time.perf_counter() - start) * 1000
            total_latencies.append(total_ms)

            component_times = getattr(pipeline, "last_timing_ms", {})
            retrieval_latencies.append(float(component_times.get("retrieval", 0.0)))
            rerank_latencies.append(float(component_times.get("rerank", 0.0)))

            candidates = _candidate_results(pipeline, ranked, level, candidate_k)
            scores = _score_one_ranking(
                ranked,
                query_item,
                k=k,
                retrieval_k=retrieval_k,
                level=level,
                candidate_ranking=candidates,
                candidate_k=candidate_k,
            )
            for metric_name in totals:
                totals[metric_name] += scores[metric_name]

            if collect_per_query:
                # The per-query scores and aggregate sums use the exact same
                # calculation path, preventing metric-definition drift.
                # Trace IDs use the same evaluation-level ID space as metrics:
                # parent document IDs for level="doc", chunk IDs for level="chunk".
                query_text = query_item["query"]
                trace = {
                    "schema_version": 1,
                    "level": level,
                    "query_text_sha256": hashlib.sha256(
                        query_text.encode("utf-8")
                    ).hexdigest(),
                    "ranked_ids": list(ranked),
                    "candidate_ids": list(candidates),
                    "relevant_ids": sorted(set(query_item["relevant"])),
                    "returned_count": len(ranked),
                    "candidate_count": len(candidates),
                    "relevant_count": len(set(query_item["relevant"])),
                }
                grades = query_item.get("grades")
                if level == "doc" and grades:
                    trace["grades"] = dict(grades)
                if include_query_text:
                    trace["query_text"] = query_text
                mode_per_query[query_item["id"]] = {
                    **scores,
                    "_trace": trace,
                }

        count = len(queries)
        result = {metric_name: total / count for metric_name, total in totals.items()}
        result.update({
            "latency_p50_ms": percentile(total_latencies, 50),
            "latency_p95_ms": percentile(total_latencies, 95),
            "retrieval_latency_p50_ms": percentile(retrieval_latencies, 50),
            "retrieval_latency_p95_ms": percentile(retrieval_latencies, 95),
            "rerank_latency_p50_ms": percentile(rerank_latencies, 50),
            "rerank_latency_p95_ms": percentile(rerank_latencies, 95),
        })
        aggregate_output[mode] = result
        if collect_per_query:
            per_query_output[mode] = mode_per_query

    return aggregate_output, per_query_output


def evaluate(
    pipeline,
    queries,
    modes,
    k=10,
    warmup=True,
    level="doc",
    retrieval_k=100,
):
    """Evaluate retrieval metrics and latency, preserving the original API.

    Returns ``{mode: {aggregate metric: value}}``. MRR remains top-k MRR under
    the backwards-compatible key ``mrr``. Use ``evaluate_with_details`` when
    the caller also needs per-query scores for reproducible comparisons.
    """
    aggregate, _ = _evaluate(
        pipeline, queries, modes, k, warmup, level, retrieval_k,
        collect_per_query=False,
        include_query_text=False,
    )
    return aggregate


def evaluate_with_details(
    pipeline,
    queries,
    modes,
    k=10,
    warmup=True,
    level="doc",
    retrieval_k=100,
    include_query_text=False,
):
    """Return aggregate metrics, per-query scores, and an audit trace.

    Shape: ``{"metrics": {mode: {...}}, "per_query": {mode: {qid: {...}}}``.
    Each query record includes metric values and a ``_trace`` object containing
    ranked IDs, pre-rerank candidate IDs, labels, and query-text hash. Exact
    query text is omitted unless ``include_query_text=True`` because queries
    can contain sensitive information. Latency stays aggregate-only.
    """
    aggregate, per_query = _evaluate(
        pipeline, queries, modes, k, warmup, level, retrieval_k,
        collect_per_query=True,
        include_query_text=include_query_text,
    )
    return {"metrics": aggregate, "per_query": per_query}
