from hs.chunking import Chunk
from hs.rerank import Reranker

def test_rerank_reorders_by_score_fn(score):
    cands = [
        Chunk("a#0", "a", "unrelated words here"),
        Chunk("b#0", "b", "python retry logic"),
    ]
    out = Reranker(score).rerank("python retry", cands, top_n=2)
    assert [c.chunk_id for c in out] == ["b#0", "a#0"]

def test_rerank_truncates_to_top_n(score):
    cands = [Chunk(f"{i}#0", str(i), "x y z") for i in range(5)]
    assert len(Reranker(score).rerank("x", cands, top_n=2)) == 2

def test_rerank_empty_candidates(score):
    assert Reranker(score).rerank("q", [], top_n=3) == []
