import numpy as np
import pytest
from hs.chunking import Chunk
from hs.dense import DenseIndex

CH = [Chunk("d0#0", "d0", "alpha beta"), Chunk("d1#0", "d1", "gamma delta")]

def test_empty_corpus_returns_nothing_even_if_embedder_returns_1d_empty():
    idx = DenseIndex([], lambda texts: np.zeros((0,)))
    assert idx.search("x", k=3) == []

def test_zero_vector_query_does_not_produce_nan(embed):
    hits = DenseIndex(CH, embed).search("", k=2)
    assert all(np.isfinite(s) for _, s in hits)

@pytest.mark.parametrize("k", [0, -1])
def test_nonpositive_k_raises(embed, k):
    with pytest.raises(ValueError):
        DenseIndex(CH, embed).search("alpha", k=k)
