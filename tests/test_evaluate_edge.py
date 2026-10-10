import pytest
from hs.chunking import Chunk
from hs.evaluate import evaluate, percentile
from hs.pipeline import HybridPipeline

CH = [Chunk("d1#0", "d1", "alpha beta"), Chunk("d2#0", "d2", "gamma delta")]
Q = [{"id": "q1", "query": "alpha", "relevant": ["d1"]},
     {"id": "q2", "query": "gamma", "relevant": ["d2"]}]

def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([10], 95) == 10
    assert percentile([1, 2, 3, 4, 5], 100) == 5

def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)

def test_query_without_relevance_labels_is_rejected(embed):
    p = HybridPipeline(CH, embed)
    with pytest.raises(ValueError, match="q9"):
        evaluate(p, [{"id": "q9", "query": "alpha", "relevant": []}], ["bm25"], k=3)

def test_no_queries_is_rejected(embed):
    with pytest.raises(ValueError):
        evaluate(HybridPipeline(CH, embed), [], ["bm25"], k=3)

def test_warmup_search_is_run_once_and_not_timed_or_scored(embed):
    calls = []
    def spy(pairs):
        calls.append(len(pairs))
        return [1.0] * len(pairs)
    p = HybridPipeline(CH, embed, score_fn=spy)
    evaluate(p, Q, ["hybrid_rerank"], k=2)
    assert len(calls) == len(Q) + 1
    calls.clear()
    evaluate(p, Q, ["hybrid_rerank"], k=2, warmup=False)
    assert len(calls) == len(Q)

def test_p95_latency_not_below_p50(embed):
    res = evaluate(HybridPipeline(CH, embed), Q, ["bm25"], k=2)
    assert res["bm25"]["latency_p95_ms"] >= res["bm25"]["latency_p50_ms"] >= 0
