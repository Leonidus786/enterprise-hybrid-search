"""Integration coverage for pre- and post-reranking evaluation traces."""

import pytest

from hs.chunking import Chunk
from hs.evaluate import evaluate_with_details
from hs.pipeline import HybridPipeline


CHUNKS = [
    Chunk("doc-a#0", "doc-a", "alpha semantic retrieval and search"),
    Chunk("doc-b#0", "doc-b", "beta lexical indexing and ranking"),
    Chunk("doc-c#0", "doc-c", "gamma dense vector retrieval"),
]
QUERY_TEXT = "alpha retrieval"


def _reverse_candidate_order(pairs):
    """Return increasing scores so Reranker reverses the incoming order."""
    return [float(index) for index, _ in enumerate(pairs)]


def _to_doc_ids(pipeline, chunk_ids):
    return [pipeline.doc_of(chunk_id) for chunk_id in chunk_ids]


def _run_trace(embed, level, *, rerank):
    pipeline = HybridPipeline(
        CHUNKS,
        embed,
        score_fn=_reverse_candidate_order if rerank else None,
        candidate_k=len(CHUNKS),
    )

    # Capture the actual pipeline rankings before evaluation. Retrieval is
    # deterministic for this fixture, so the evaluation run must match them.
    pre_rerank_chunks = pipeline.search(
        QUERY_TEXT, mode="hybrid", k=len(CHUNKS)
    )
    pre_rerank_candidates = list(pipeline.last_candidate_ids)

    if rerank:
        final_chunks = pipeline.search(
            QUERY_TEXT, mode="hybrid_rerank", k=len(CHUNKS)
        )
        mode = "hybrid_rerank"
    else:
        final_chunks = pre_rerank_chunks
        mode = "hybrid"

    if level == "chunk":
        expected_candidates = pre_rerank_candidates
        expected_ranked = list(final_chunks)
        relevant_ids = [chunk.chunk_id for chunk in CHUNKS]
    else:
        expected_candidates = _to_doc_ids(pipeline, pre_rerank_candidates)
        expected_ranked = _to_doc_ids(pipeline, final_chunks)
        relevant_ids = [chunk.doc_id for chunk in CHUNKS]

    result = evaluate_with_details(
        pipeline,
        [{"id": "q1", "query": QUERY_TEXT, "relevant": relevant_ids}],
        [mode],
        k=len(CHUNKS),
        retrieval_k=len(CHUNKS),
        warmup=False,
        level=level,
    )
    trace = result["per_query"][mode]["q1"]["_trace"]
    return pipeline, result, trace, expected_candidates, expected_ranked


@pytest.mark.parametrize("level", ["chunk", "doc"])
def test_real_pipeline_trace_separates_pre_and_post_rerank_order(embed, level):
    pipeline, result, trace, expected_candidates, expected_ranked = _run_trace(
        embed, level, rerank=True
    )

    assert trace["candidate_ids"] == expected_candidates
    assert trace["ranked_ids"] == expected_ranked
    assert trace["candidate_ids"] == list(reversed(trace["ranked_ids"]))
    assert trace["candidate_ids"] != trace["ranked_ids"]
    assert result["metrics"]["hybrid_rerank"]["rerank_latency_p50_ms"] >= 0.0
    assert pipeline.last_timing_ms["rerank"] >= 0.0


@pytest.mark.parametrize("level", ["chunk", "doc"])
def test_real_pipeline_without_reranker_traces_hybrid_order(embed, level):
    _, result, trace, expected_candidates, expected_ranked = _run_trace(
        embed, level, rerank=False
    )

    assert trace["candidate_ids"] == expected_candidates
    assert trace["ranked_ids"] == expected_ranked
    assert trace["candidate_ids"] == trace["ranked_ids"]
    assert result["metrics"]["hybrid"]["rerank_latency_p50_ms"] == 0.0
