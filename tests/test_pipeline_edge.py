import pytest
from hs.chunking import Chunk
from hs.pipeline import HybridPipeline

def test_duplicate_chunk_ids_are_rejected(embed):
    dup = [Chunk("x#0", "x", "a b"), Chunk("x#0", "x", "c d")]
    with pytest.raises(ValueError):
        HybridPipeline(dup, embed)

@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_nonpositive_k_raises_in_every_mode(embed, mode):
    p = HybridPipeline([Chunk("x#0", "x", "a b")], embed)
    with pytest.raises(ValueError):
        p.search("a", mode=mode, k=0)

def test_empty_query_does_not_crash_any_mode(embed, score):
    p = HybridPipeline([Chunk("x#0", "x", "a b")], embed, score_fn=score)
    for mode in ("bm25", "dense", "hybrid", "hybrid_rerank"):
        p.search("", mode=mode, k=3)
