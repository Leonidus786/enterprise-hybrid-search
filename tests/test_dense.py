from hs.chunking import Chunk
from hs.dense import DenseIndex

CH = [
    Chunk("d1#0", "d1", "kubernetes pods scheduling and nodes"),
    Chunk("d2#0", "d2", "baking sourdough bread at home"),
    Chunk("d3#0", "d3", "gradient descent optimises neural networks"),
]

def test_nearest_chunk_shares_vocabulary(embed):
    hits = DenseIndex(CH, embed).search("sourdough bread", k=1)
    assert hits[0][0] == "d2#0"

def test_scores_are_cosine_bounded(embed):
    hits = DenseIndex(CH, embed).search("neural networks", k=3)
    assert all(-1.0001 <= s <= 1.0001 for _, s in hits)

def test_results_sorted_descending(embed):
    hits = DenseIndex(CH, embed).search("pods nodes", k=3)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)

def test_k_larger_than_corpus_is_clamped(embed):
    assert len(DenseIndex(CH, embed).search("bread", k=50)) == 3
