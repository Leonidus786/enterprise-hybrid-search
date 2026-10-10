import pytest
from hs.fusion import rrf

def test_rrf_rewards_docs_ranked_well_in_both_lists():
    fused = rrf([["a", "b", "c"], ["b", "c", "a"]], k=60)
    assert [d for d, _ in fused] == ["b", "a", "c"]

def test_rrf_includes_docs_present_in_only_one_list():
    fused = dict(rrf([["a"], ["b"]], k=60))
    assert set(fused) == {"a", "b"}

def test_rrf_score_formula():
    fused = dict(rrf([["a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)

def test_rrf_empty_input():
    assert rrf([], k=60) == []
