from hs.evaluate import evaluate_with_details
from hs.per_query import score_rankings


def test_mrr_cutoff_alias_preserves_legacy_key():
    result = score_rankings(
        rankings={"q1": ["d1", "d2", "d3"]},
        queries=[{"id": "q1", "relevant": ["d2"]}],
        k=2,
        retrieval_k=3,
    )["q1"]

    assert result["mrr"] == 0.5
    assert result["mrr@2"] == 0.5
    assert result["mrr"] == result["mrr@2"]


class FakePipeline:
    doc_ids = {"d1", "d2", "d3"}
    chunk_ids = {"d1", "d2", "d3"}
    candidate_k = 3
    last_candidate_ids = None
    last_timing_ms = {}

    def search_docs(self, query, mode, k):
        return ["d1", "d2", "d3"][:k]


def test_evaluator_exposes_mrr_alias_in_aggregate_and_per_query():
    result = evaluate_with_details(
        FakePipeline(),
        [{"id": "q1", "query": "test", "relevant": ["d2"]}],
        ["bm25"],
        k=2,
        retrieval_k=3,
    )

    aggregate = result["metrics"]["bm25"]
    per_query = result["per_query"]["bm25"]["q1"]

    assert aggregate["mrr"] == aggregate["mrr@2"] == 0.5
    assert per_query["mrr"] == per_query["mrr@2"] == 0.5
