import importlib.util
from pathlib import Path

from hs.compare import check_consistency
from hs.evaluate import evaluate_with_details


class TracePipeline:
    candidate_k = 2
    doc_ids = {"d1", "d2", "d3"}
    chunk_ids = {"d1#0", "d2#0", "d3#0"}
    last_timing_ms = {"retrieval": 1.0, "rerank": 0.0}

    def __init__(self):
        self.last_candidate_ids = []

    @staticmethod
    def doc_of(chunk_id):
        return chunk_id.split("#", 1)[0]

    def search_docs(self, query, mode="bm25", k=10):
        ranking = ["d2", "d1", "d3"] if query == "question one" else ["d1", "d3", "d2"]
        self.last_candidate_ids = [f"{doc_id}#0" for doc_id in ranking[:2]]
        return ranking[:k]

    def search(self, query, mode="bm25", k=10):
        ranking = ["d2#0", "d1#0", "d3#0"] if query == "question one" else ["d1#0", "d3#0", "d2#0"]
        self.last_candidate_ids = ranking[:2]
        return ranking[:k]


QUERIES = [
    {"id": "q1", "query": "question one", "relevant": ["d1"], "grades": {"d1": 3}},
    {"id": "q2", "query": "question two", "relevant": ["d3"], "grades": {"d3": 2}},
]


def test_trace_captures_rankings_candidates_labels_and_hash_without_query_text():
    result = evaluate_with_details(
        TracePipeline(), QUERIES, ["bm25"], k=2, retrieval_k=3, warmup=False
    )
    row = result["per_query"]["bm25"]["q1"]
    trace = row["_trace"]
    assert trace["schema_version"] == 1
    assert trace["level"] == "doc"
    assert trace["ranked_ids"] == ["d2", "d1", "d3"]
    assert trace["candidate_ids"] == ["d2", "d1"]
    assert trace["relevant_ids"] == ["d1"]
    assert trace["grades"] == {"d1": 3}
    assert trace["returned_count"] == 3
    assert len(trace["query_text_sha256"]) == 64
    assert "query_text" not in trace
    assert check_consistency(result) == []


def test_query_text_is_only_in_trace_when_explicitly_enabled():
    result = evaluate_with_details(
        TracePipeline(), QUERIES, ["bm25"], k=2, retrieval_k=3,
        warmup=False, include_query_text=True,
    )
    assert result["per_query"]["bm25"]["q1"]["_trace"]["query_text"] == "question one"


def test_chunk_level_trace_keeps_chunk_ids():
    # Chunk-level qrels use chunk IDs directly; no parent-document conversion is expected.
    chunk_queries = [
        {"id": "cq1", "query": "question one", "relevant": ["d1#0"]},
    ]
    result = evaluate_with_details(
        TracePipeline(), chunk_queries, ["bm25"], k=2, retrieval_k=3,
        warmup=False, level="chunk",
    )
    trace = result["per_query"]["bm25"]["cq1"]["_trace"]
    assert trace["level"] == "chunk"
    assert trace["ranked_ids"] == ["d2#0", "d1#0", "d3#0"]
    assert trace["candidate_ids"] == ["d2#0", "d1#0"]


def test_default_metric_discovery_ignores_nested_trace():
    script = Path(__file__).resolve().parents[1] / "scripts" / "compare_runs.py"
    spec = importlib.util.spec_from_file_location("trace_compare_runs", script)
    module = importlib.util.module_from_spec(spec)
    # compare_runs imports hs; src is inserted by project test configuration.
    spec.loader.exec_module(module)
    result = {
        "metrics": {"dense": {"ndcg@2": 0.5, "recall@2": 0.6}},
        "per_query": {"dense": {
            "q1": {"ndcg@2": 0.4, "recall@2": 0.5, "_trace": {"ranked_ids": ["d1"]}},
            "q2": {"ndcg@2": 0.6, "recall@2": 0.7, "_trace": {"ranked_ids": ["d2"]}},
        }},
    }
    assert module._available_metrics(result, result, "dense", "dense") == ["ndcg@2", "recall@2"]
