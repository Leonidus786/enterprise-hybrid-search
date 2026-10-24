import numpy as np
import pytest

STRATEGIES = ["bm25", "dense", "hybrid", "hybrid_rerank"]


def test_trial_count_rejects_fewer_than_five():
    from scripts.run_latency import validate_trials

    with pytest.raises(ValueError, match="5"):
        validate_trials(4)


def test_trial_count_accepts_five():
    from scripts.run_latency import validate_trials

    validate_trials(5)


def test_strategy_order_rotates_deterministically():
    from scripts.run_latency import rotated_order

    assert rotated_order(STRATEGIES, 0) == STRATEGIES
    assert rotated_order(STRATEGIES, 1) == [
        "dense", "hybrid", "hybrid_rerank", "bm25"
    ]
    assert rotated_order(STRATEGIES, 4) == STRATEGIES


def test_latency_summary_reports_p50_p95_p99():
    from scripts.run_latency import summarize_latencies

    result = summarize_latencies([1, 2, 3, 4, 5])

    assert result["count"] == 5
    assert result["p50_ms"] == pytest.approx(3.0)
    assert result["p95_ms"] == pytest.approx(4.8)
    assert result["p99_ms"] == pytest.approx(4.96)


class FakePipeline:
    doc_ids = {"d1"}
    chunk_ids = {"c1"}
    candidate_k = 2

    def __init__(self):
        self.calls = []
        self.last_timing_ms = {}

    def search_docs(self, query, mode, k):
        self.calls.append((mode, query))
        self.last_timing_ms = {
            "retrieval": 1.25,
            "rerank": 0.5 if mode == "hybrid_rerank" else 0.0,
        }
        return ["d1"]

    def search(self, query, mode, k):
        self.calls.append((mode, query))
        self.last_timing_ms = {
            "retrieval": 1.25,
            "rerank": 0.5 if mode == "hybrid_rerank" else 0.0,
        }
        return ["c1"]


def test_warmups_are_excluded_from_measured_samples():
    from scripts.run_latency import run_trials

    pipeline = FakePipeline()
    queries = [
        {"id": "q1", "query": "first query"},
        {"id": "q2", "query": "second query"},
    ]
    strategies = ["bm25", "dense"]

    samples = run_trials(
        pipeline,
        queries,
        strategies,
        trials=5,
        warmup_queries=1,
        level="doc",
        k=2,
    )

    assert len(samples) == 5 * len(queries) * len(strategies)
    assert {sample["trial"] for sample in samples} == {0, 1, 2, 3, 4}
    assert {sample["query_id"] for sample in samples} == {"q1", "q2"}
    assert all(sample["total_ms"] >= 0 for sample in samples)
    assert all(sample["retrieval_ms"] == 1.25 for sample in samples)
    assert all(
        sample["rerank_ms"] == (0.5 if sample["strategy"] == "hybrid_rerank" else 0.0)
        for sample in samples
    )
    # Two warm-up calls plus 20 measured calls; warm-ups aren't samples.
    assert len(pipeline.calls) == 2 + len(samples)


def test_strategy_order_interleaves_and_rotates_between_trials():
    from scripts.run_latency import run_trials

    pipeline = FakePipeline()
    queries = [
        {"id": "q1", "query": "first query"},
        {"id": "q2", "query": "second query"},
    ]
    strategies = ["bm25", "dense"]

    run_trials(
        pipeline,
        queries,
        strategies,
        trials=5,
        warmup_queries=1,
        level="doc",
        k=2,
    )

    measured_calls = pipeline.calls[2:]

    assert measured_calls[:4] == [
        ("bm25", "first query"),
        ("dense", "first query"),
        ("bm25", "second query"),
        ("dense", "second query"),
    ]
    assert measured_calls[4:8] == [
        ("dense", "first query"),
        ("bm25", "first query"),
        ("dense", "second query"),
        ("bm25", "second query"),
    ]


def test_dense_index_can_use_uncached_query_embedder():
    from hs.dense import DenseIndex
    from types import SimpleNamespace

    document_calls = []
    query_calls = []

    def document_embed(texts):
        document_calls.extend(texts)
        return np.tile(
            np.array([[1.0, 0.0]], dtype=np.float32),
            (len(texts), 1),
        )

    def query_embed(texts):
        query_calls.extend(texts)
        return np.tile(
            np.array([[1.0, 0.0]], dtype=np.float32),
            (len(texts), 1),
        )

    chunks = [
        SimpleNamespace(chunk_id="c1", doc_id="d1", text="document text")
    ]

    index = DenseIndex(
        chunks,
        document_embed,
        query_prefix="Q: ",
        query_embed_fn=query_embed,
    )
    index.search("test query", k=1)

    assert document_calls == ["document text"]
    assert query_calls == ["Q: test query"]