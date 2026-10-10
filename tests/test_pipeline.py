import pytest
from hs.chunking import Chunk
from hs.pipeline import HybridPipeline

CH = [
    Chunk("d1#0", "d1", "kubernetes pods scheduling and nodes"),
    Chunk("d1#1", "d1", "kubernetes services and ingress"),
    Chunk("d2#0", "d2", "baking sourdough bread at home"),
    Chunk("d3#0", "d3", "gradient descent optimises neural networks"),
]

@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_each_mode_returns_relevant_chunk_first(mode, embed):
    p = HybridPipeline(CH, embed)
    assert p.search("sourdough bread", mode=mode, k=3)[0] == "d2#0"

def test_hybrid_rerank_uses_cross_encoder(embed, score):
    p = HybridPipeline(CH, embed, score_fn=score)
    assert p.search("gradient descent", mode="hybrid_rerank", k=2)[0] == "d3#0"

def test_hybrid_rerank_without_score_fn_raises(embed):
    p = HybridPipeline(CH, embed)
    with pytest.raises(ValueError):
        p.search("x", mode="hybrid_rerank")

def test_unknown_mode_raises(embed):
    with pytest.raises(ValueError):
        HybridPipeline(CH, embed).search("x", mode="magic")

def test_doc_id_lookup(embed):
    assert HybridPipeline(CH, embed).doc_of("d1#1") == "d1"
