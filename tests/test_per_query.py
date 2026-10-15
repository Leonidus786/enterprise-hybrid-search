"""Step 3a: per-query metrics from saved rankings (pure function, no models).

Contract (hs/per_query.py):
    score_rankings(rankings, queries, k=10, retrieval_k=100) -> {qid: {metric: value}}
      rankings : {qid: [ranked ids, best first]}      queries : [{"id", "query", "relevant": [...]}]
      keys per query: precision@k, recall@k, recall@{retrieval_k}, mrr, ndcg@k
      precision/recall@k/mrr/ndcg use ranking[:k]  (MRR is MRR@k!);  recall@{retrieval_k} uses ranking[:retrieval_k]
      binary relevance.
"""
import math

import pytest
from hs.per_query import score_rankings

QUERIES = [
    {"id": "q1", "query": "x", "relevant": ["a", "b"]},
    {"id": "q2", "query": "y", "relevant": ["c"]},
]


def test_perfect_ranking_scores_one():
    out = score_rankings({"q1": ["a", "b", "z"], "q2": ["c", "z"]}, QUERIES, k=2, retrieval_k=3)
    assert out["q1"]["ndcg@2"] == pytest.approx(1.0)
    assert out["q1"]["recall@2"] == 1.0 and out["q1"]["mrr"] == 1.0 and out["q1"]["precision@2"] == 1.0
    assert out["q2"]["mrr"] == 1.0 and out["q2"]["recall@3"] == 1.0


def test_exact_values_for_a_known_ranking():
    out = score_rankings({"q1": ["x", "a", "y", "b"], "q2": ["c"]}, QUERIES, k=2, retrieval_k=4)["q1"]
    d = 1 / math.log2(3)
    assert out["precision@2"] == pytest.approx(0.5)
    assert out["recall@2"] == pytest.approx(0.5)
    assert out["recall@4"] == pytest.approx(1.0)
    assert out["mrr"] == pytest.approx(0.5)
    assert out["ndcg@2"] == pytest.approx(d / (1 + d))


def test_mrr_is_cut_at_k_even_when_retrieval_is_deeper():
    out = score_rankings({"q1": ["x", "y", "a"], "q2": ["c"]}, QUERIES, k=2, retrieval_k=5)["q1"]
    assert out["mrr"] == 0.0 and out["recall@2"] == 0.0     # a is at rank 3: invisible at k=2
    assert out["recall@5"] == pytest.approx(0.5)            # ...but recall@5 sees it


def test_retrieval_depth_truncates_recall():
    out = score_rankings({"q1": ["x"] * 0 + ["x1", "x2", "x3", "x4", "x5", "a"], "q2": ["c"]},
                         QUERIES, k=2, retrieval_k=5)["q1"]
    assert out["recall@5"] == 0.0                           # a sits at rank 6


def test_exact_key_set():
    out = score_rankings({"q1": ["a"], "q2": ["c"]}, QUERIES, k=2, retrieval_k=5)["q1"]
    assert set(out) == {"precision@2", "recall@2", "recall@5", "mrr", "ndcg@2"}


def test_missing_ranking_names_the_query():
    with pytest.raises(ValueError, match="q2"):
        score_rankings({"q1": ["a"]}, QUERIES)


def test_extra_rankings_are_ignored():
    out = score_rankings({"q1": ["a"], "q2": ["c"], "q99": ["zzz"]}, QUERIES, k=1, retrieval_k=2)
    assert set(out) == {"q1", "q2"}


def test_duplicate_ids_in_a_ranking_raise():
    with pytest.raises(ValueError):
        score_rankings({"q1": ["a", "a"], "q2": ["c"]}, QUERIES, k=2, retrieval_k=2)


@pytest.mark.parametrize("k,r", [(0, 5), (-1, 5), (5, 3), (1.5, 5), (2, True)])
def test_bad_cutoffs_raise(k, r):
    with pytest.raises(ValueError):
        score_rankings({"q1": ["a"], "q2": ["c"]}, QUERIES, k=k, retrieval_k=r)


def test_query_without_labels_raises():
    q = [{"id": "q1", "query": "x", "relevant": []}]
    with pytest.raises(ValueError, match="q1"):
        score_rankings({"q1": ["a"]}, q)
