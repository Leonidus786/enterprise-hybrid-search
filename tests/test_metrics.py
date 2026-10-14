import math
import pytest
from hs.metrics import precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k, graded_ndcg_at_k


def test_precision_at_k_counts_relevant_in_top_k():
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5


def test_precision_divides_by_k_even_if_ranking_shorter():
    assert precision_at_k(["a"], {"a"}, 4) == 0.25


def test_precision_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, 0)


def test_recall_at_k_fraction_of_relevant_found():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5


def test_recall_is_zero_when_no_relevant_labeled():
    assert recall_at_k(["a"], set(), 5) == 0.0


def test_reciprocal_rank_uses_first_relevant_hit():
    assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5


def test_reciprocal_rank_zero_when_no_hit():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3) == pytest.approx(1.0)


def test_ndcg_penalises_lower_rank():
    expected = (1 / math.log2(3)) / 1.0
    assert ndcg_at_k(["x", "a"], {"a"}, 2) == pytest.approx(expected)


def test_ndcg_zero_when_no_relevant_labeled():
    assert ndcg_at_k(["a"], set(), 5) == 0.0


def test_graded_ndcg_rewards_better_relevant_ranking():
    grades = {"high": 2, "low": 1}

    best = graded_ndcg_at_k(["high", "low"], grades, 2)
    worse = graded_ndcg_at_k(["low", "high"], grades, 2)

    assert best == 1.0
    assert worse < best


def test_graded_ndcg_with_empty_grades_is_zero():
    assert graded_ndcg_at_k(["d1"], {}, 10) == 0.0
