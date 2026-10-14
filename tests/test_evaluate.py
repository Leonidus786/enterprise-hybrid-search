import pytest
from hs.chunking import Chunk
from hs.pipeline import HybridPipeline
from hs.evaluate import evaluate

CH = [
    Chunk("d1#0", "d1", "kubernetes pods scheduling and nodes"),
    Chunk("d1#1", "d1", "kubernetes services and ingress"),
    Chunk("d2#0", "d2", "baking sourdough bread at home"),
    Chunk("d3#0", "d3", "gradient descent optimises neural networks"),
]
QUERIES = [
    {"id": "q1", "query": "sourdough bread", "relevant": ["d2"]},
    {"id": "q2", "query": "neural networks gradient", "relevant": ["d3"]},
]


def test_evaluate_returns_metrics_per_mode(embed, score):
    p = HybridPipeline(CH, embed, score_fn=score)
    res = evaluate(p, QUERIES, ["bm25", "dense",
                   "hybrid", "hybrid_rerank"], k=3)
    assert set(res) == {"bm25", "dense", "hybrid", "hybrid_rerank"}
    assert set(res["bm25"]) == {
        "precision@3",
        "recall@3",
        "recall@100",
        "candidate_recall@50",
        "mrr",
        "ndcg@3",
        "latency_p50_ms",
        "latency_p95_ms",
        "retrieval_latency_p50_ms",
        "retrieval_latency_p95_ms",
        "rerank_latency_p50_ms",
        "rerank_latency_p95_ms",
    }


def test_evaluate_scores_perfect_when_every_query_hits_first(embed):
    p = HybridPipeline(CH, embed)
    res = evaluate(p, QUERIES, ["bm25"], k=3)
    assert res["bm25"]["mrr"] == pytest.approx(1.0)
    assert res["bm25"]["recall@3"] == pytest.approx(1.0)


def test_evaluate_dedupes_chunks_to_docs(embed):
    p = HybridPipeline(CH, embed)
    q = [{"id": "q", "query": "kubernetes", "relevant": ["d1"]}]
    res = evaluate(p, q, ["bm25"], k=3)
    # two d1 chunks collapse to one doc, so precision@3 = 1/3, not 2/3
    assert res["bm25"]["precision@3"] == pytest.approx(1 / 3)
