import pytest
from hs.metrics import precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k

@pytest.mark.parametrize("fn", [
    lambda r: precision_at_k(r, {"a"}, 2),
    lambda r: recall_at_k(r, {"a"}, 2),
    lambda r: reciprocal_rank(r, {"a"}),
    lambda r: ndcg_at_k(r, {"a"}, 2),
])
def test_ranking_with_duplicate_ids_is_rejected(fn):
    with pytest.raises(ValueError):
        fn(["a", "a"])

def test_ndcg_k_larger_than_ranking_is_fine():
    assert ndcg_at_k(["a"], {"a"}, 10) == pytest.approx(1.0)

def test_empty_ranking_scores_zero():
    assert recall_at_k([], {"a"}, 3) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0
    assert ndcg_at_k([], {"a"}, 3) == 0.0
