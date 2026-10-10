from hs.chunking import Chunk
from hs.sparse import SparseIndex

CH = [
    Chunk("d1#0", "d1", "kubernetes pods scheduling and nodes"),
    Chunk("d2#0", "d2", "baking sourdough bread at home"),
    Chunk("d3#0", "d3", "gradient descent optimises neural networks"),
]

def test_exact_keyword_ranks_matching_chunk_first():
    hits = SparseIndex(CH).search("kubernetes", k=3)
    assert hits[0][0] == "d1#0"

def test_query_with_no_known_tokens_returns_nothing():
    assert SparseIndex(CH).search("zzzzunknown", k=3) == []

def test_respects_k():
    assert len(SparseIndex(CH).search("bread kubernetes gradient", k=2)) == 2

def test_search_is_case_insensitive():
    assert SparseIndex(CH).search("KUBERNETES", k=1)[0][0] == "d1#0"
