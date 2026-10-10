import math
import pytest
from hs.chunking import Chunk
from hs.sparse import SparseIndex

def mk(*texts):
    return [Chunk(f"d{i}#0", f"d{i}", t) for i, t in enumerate(texts)]

def test_bm25_score_matches_hand_computed_value():
    # N=3, df("b")=1 -> idf = ln(1 + 2.5/1.5) = 0.98083
    # d0: tf=1, dl=2, avgdl=7/3 -> denom = 1 + 1.5*(0.25 + 0.75*2/(7/3)) = 2.339286
    # score = idf * tf*(k1+1)/denom = 0.98083 * 2.5 / 2.339286 = 1.04821
    idx = SparseIndex(mk("a b", "a c", "d e f"))
    (cid, score), = idx.search("b", k=5)
    assert cid == "d0#0"
    assert score == pytest.approx(1.0482, abs=1e-3)

def test_idf_matches_lucene_formula():
    idx = SparseIndex(mk("a b", "a c", "d e f"))
    assert idx.idf["a"] == pytest.approx(math.log(1 + (3 - 2 + 0.5) / (2 + 0.5)))
    assert idx.idf["b"] == pytest.approx(math.log(1 + (3 - 1 + 0.5) / (1 + 0.5)))

def test_shorter_doc_outranks_longer_doc_with_same_term_frequency():
    idx = SparseIndex(mk("kubernetes", "kubernetes plus many other filler words here", "unrelated text"))
    hits = idx.search("kubernetes", k=3)
    assert [c for c, _ in hits] == ["d0#0", "d1#0"]

def test_term_frequency_saturates():
    idx = SparseIndex(mk("kube", " ".join(["kube"] * 10), "other words"))
    s = dict(idx.search("kube", k=3))
    assert s["d0#0"] < s["d1#0"] < 10 * s["d0#0"]

def test_term_in_half_the_corpus_still_scores():
    idx = SparseIndex(mk("kube a", "kube b", "c d", "e f"))
    assert {c for c, _ in idx.search("kube", k=4)} == {"d0#0", "d1#0"}

def test_empty_query_returns_nothing():
    assert SparseIndex(mk("a b")).search("", k=3) == []

def test_punctuation_only_query_returns_nothing():
    assert SparseIndex(mk("a b")).search("?!...", k=3) == []

def test_empty_corpus_returns_nothing():
    assert SparseIndex([]).search("anything", k=3) == []

def test_all_empty_chunks_do_not_crash():
    assert SparseIndex(mk("", "  ")).search("x", k=3) == []

@pytest.mark.parametrize("k", [0, -1])
def test_nonpositive_k_raises(k):
    with pytest.raises(ValueError):
        SparseIndex(mk("a")).search("a", k=k)
