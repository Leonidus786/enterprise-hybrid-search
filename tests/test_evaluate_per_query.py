import pytest

from hs.compare import check_consistency
from hs.evaluate import evaluate, evaluate_with_details


class FixedPipeline:
    candidate_k = 2
    doc_ids = {"d1", "d2"}
    chunk_ids = {"d1#0", "d2#0"}
    last_timing_ms = {"retrieval": 1.0, "rerank": 0.0}

    def __init__(self):
        self.last_candidate_ids = []

    @staticmethod
    def doc_of(chunk_id):
        return chunk_id.split("#", 1)[0]

    def search_docs(self, query, mode="bm25", k=10):
        ranking = ["d2", "d1"] if query == "q1" else ["d1", "d2"]
        self.last_candidate_ids = [f"{doc_id}#0" for doc_id in ranking]
        return ranking[:k]

    def search(self, query, mode="bm25", k=10):
        ranking = ["d2#0", "d1#0"] if query == "q1" else ["d1#0", "d2#0"]
        self.last_candidate_ids = ranking
        return ranking[:k]


QUERIES = [
    {
        "id": "q1",
        "query": "q1",
        "relevant": ["d1", "d2"],
        "grades": {"d1": 3, "d2": 1},
    },
    {
        "id": "q2",
        "query": "q2",
        "relevant": ["d1", "d2"],
        "grades": {"d1": 1, "d2": 3},
    },
]


def test_aggregate_metrics_reconcile_with_per_query_metrics_and_preserve_graded_ndcg():
    details = evaluate_with_details(
        FixedPipeline(), QUERIES, ["bm25"], k=2, retrieval_k=2, warmup=False
    )

    assert set(details) == {"metrics", "per_query"}
    assert set(details["per_query"]["bm25"]) == {"q1", "q2"}
    # q1 places the grade-1 document before the grade-3 document; graded nDCG
    # therefore penalises this ranking (binary nDCG would be 1.0).
    assert details["per_query"]["bm25"]["q1"]["ndcg@2"] < 1.0
    assert check_consistency(details) == []


def test_legacy_evaluate_keeps_its_aggregate_return_shape():
    result = evaluate(FixedPipeline(), QUERIES, ["bm25"], k=2, retrieval_k=2, warmup=False)
    assert set(result) == {"bm25"}
    assert "per_query" not in result
    assert "latency_p50_ms" in result["bm25"]
    assert "ndcg@2" in result["bm25"]


def test_comparison_rejects_different_query_fingerprints():
    from hs.compare import compare

    base = {
        "dataset": {"searchable_items": 2, "unique_documents": 2, "total_queries": 2,
                    "evaluated_queries": 2, "query_fingerprint": "abc"},
        "config": {"k": 2, "retrieval_k": 2, "candidate_k": 2, "level": "doc",
                    "chunking": {"enabled": False, "size": None, "overlap": 0},
                    "fake_models": False, "embedding_model": "model-a", "reranker_model": "reranker-a",
                    "bge_prefix_enabled": True},
        "metrics": {"m": {"ndcg@2": 0.5}},
        "per_query": {"m": {"q1": {"ndcg@2": 0.5}, "q2": {"ndcg@2": 0.5}}},
    }
    other = {**base, "dataset": {**base["dataset"], "query_fingerprint": "different"}}
    with pytest.raises(ValueError, match="query_fingerprint"):
        compare(base, other, "m", "m", "ndcg@2", n_resamples=100)
