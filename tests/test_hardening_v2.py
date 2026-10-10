import numpy as np
import pytest
from hs.chunking import Chunk
from hs.dense import DenseIndex
from hs.evaluate import evaluate
from hs.fusion import rrf
from hs.metrics import ndcg_at_k, recall_at_k
from hs.pipeline import HybridPipeline
from hs.rerank import Reranker
from hs.sparse import SparseIndex

# doc A owns 4 strong chunks; doc B has 1 weaker chunk; C is filler.
MULTI = [Chunk(f"A#{i}", "A", "alpha alpha alpha") for i in range(4)] + [
    Chunk("B#0", "B", "alpha beta"),
    Chunk("C#0", "C", "gamma delta"),
]

# ---- metrics: k contract must be consistent across all @k metrics
@pytest.mark.parametrize("k", [0, -1])
def test_recall_rejects_nonpositive_k(k):
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, k)

@pytest.mark.parametrize("k", [0, -1])
def test_ndcg_rejects_nonpositive_k(k):
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a"}, k)

# ---- fusion
@pytest.mark.parametrize("k", [0, -5])
def test_rrf_rejects_nonpositive_constant(k):
    with pytest.raises(ValueError):
        rrf([["a"]], k=k)

def test_rrf_counts_a_repeated_id_once_per_ranking():
    assert dict(rrf([["a", "a"]], k=60))["a"] == pytest.approx(1 / 61)

def test_rrf_ties_break_by_first_seen_order():
    assert [d for d, _ in rrf([["x", "y"], ["y", "x"]])] == ["x", "y"]
    assert [d for d, _ in rrf([["y", "x"], ["x", "y"]])] == ["y", "x"]

# ---- rerank
def test_rerank_rejects_nonpositive_top_n(score):
    with pytest.raises(ValueError):
        Reranker(score).rerank("q", [Chunk("a#0", "a", "x")], top_n=0)

def test_rerank_rejects_wrong_number_of_scores():
    with pytest.raises(ValueError):
        Reranker(lambda pairs: [1.0]).rerank("q", MULTI[:3], top_n=2)

def test_rerank_rejects_nan_scores():
    with pytest.raises(ValueError):
        Reranker(lambda pairs: [float("nan")] * len(pairs)).rerank("q", MULTI[:2], top_n=2)

# ---- dense
def test_dense_rejects_wrong_row_count():
    with pytest.raises(ValueError):
        DenseIndex(MULTI, lambda texts: np.ones((1, 4)))

def test_dense_rejects_non_finite_embeddings():
    with pytest.raises(ValueError):
        DenseIndex(MULTI, lambda texts: np.full((len(texts), 4), np.nan))

def test_dense_rejects_query_dimension_mismatch():
    def emb(texts):
        return np.ones((len(texts), 4 if len(texts) > 1 else 3))
    idx = DenseIndex(MULTI, emb)
    with pytest.raises(ValueError):
        idx.search("q", k=1)

def test_query_prefix_is_applied_to_queries_only(embed):
    calls = []
    def rec(texts):
        calls.append(list(texts))
        return embed(texts)
    idx = DenseIndex(MULTI, rec, query_prefix="Query: ")
    idx.search("hello", k=1)
    assert calls[0][0] == "alpha alpha alpha"      # documents: no prefix
    assert calls[-1] == ["Query: hello"]            # query: prefixed

# ---- sparse params
@pytest.mark.parametrize("kw", [{"k1": -1.0}, {"b": 1.5}, {"b": -0.1}])
def test_sparse_rejects_invalid_bm25_params(kw):
    with pytest.raises(ValueError):
        SparseIndex(MULTI, **kw)

# ---- pipeline
def test_pipeline_rejects_nonpositive_candidate_k(embed):
    with pytest.raises(ValueError):
        HybridPipeline(MULTI, embed, candidate_k=0)

def test_search_docs_returns_unique_docs_in_rank_order(embed):
    p = HybridPipeline(MULTI, embed)
    assert p.search_docs("alpha", mode="bm25", k=2) == ["A", "B"]

# ---- evaluate: the doc-level top-K regression
def test_doc_level_eval_sees_k_unique_docs_not_k_chunks(embed):
    p = HybridPipeline(MULTI, embed)
    q = [{"id": "q", "query": "alpha", "relevant": ["B"]}]
    res = evaluate(p, q, ["bm25"], k=2)["bm25"]
    # old behaviour: top-2 chunks are both A -> B never seen -> recall 0.0
    assert res["recall@2"] == pytest.approx(1.0)
    assert res["mrr"] == pytest.approx(0.5)

def test_chunk_level_eval_uses_chunk_ids(embed):
    p = HybridPipeline(MULTI, embed)
    q = [{"id": "q", "query": "alpha", "relevant": ["B#0"]}]
    res = evaluate(p, q, ["bm25"], k=5, level="chunk")["bm25"]
    assert res["recall@5"] == pytest.approx(1.0)
    assert res["mrr"] == pytest.approx(0.2)

def test_unknown_level_rejected(embed):
    with pytest.raises(ValueError):
        evaluate(HybridPipeline(MULTI, embed), [{"id": "q", "query": "a", "relevant": ["A"]}],
                 ["bm25"], k=2, level="paragraph")

def test_relevance_label_pointing_at_missing_doc_is_rejected(embed):
    p = HybridPipeline(MULTI, embed)
    with pytest.raises(ValueError):
        evaluate(p, [{"id": "q", "query": "alpha", "relevant": ["NOPE"]}], ["bm25"], k=2)
