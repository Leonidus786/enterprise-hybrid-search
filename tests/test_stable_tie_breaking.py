
import numpy as np

from hs.chunking import Chunk
from hs.dense import DenseIndex
from hs.fusion import rrf
from hs.rerank import Reranker
from hs.sparse import SparseIndex


def test_dense_equal_scores_preserve_input_order():
    levels = [0, 3, 2, 1, 1, 3, 0, 2, 0, 0, 2, 3, 2, 3, 2, 3]

    vectors = {
        "query": [1.0, 0.0],
        "level-0": [0.0, 1.0],
        "level-1": [0.8, 0.6],
        "level-2": [0.6, 0.8],
        "level-3": [1.0, 0.0],
    }

    chunks = [
        Chunk(
            f"doc-{i:02d}#0",
            f"doc-{i:02d}",
            f"level-{level}",
        )
        for i, level in enumerate(levels)
    ]

    def embed(texts):
        return np.asarray(
            [vectors[text] for text in texts],
            dtype=np.float32,
        )

    index = DenseIndex(chunks, embed)

    expected_indices = (
        1, 5, 11, 13, 15,
        3, 4,
        2, 7, 10, 12, 14,
        0, 6, 8, 9,
    )
    expected_ids = [chunks[i].chunk_id for i in expected_indices]

    first = index.search("query", k=len(chunks))
    second = index.search("query", k=len(chunks))

    assert [chunk_id for chunk_id, _ in first] == expected_ids
    assert [chunk_id for chunk_id, _ in second] == expected_ids


def test_sparse_equal_scores_preserve_input_order():
    chunks = [
        Chunk("z#0", "z", "needle common text"),
        Chunk("a#0", "a", "needle common text"),
        Chunk("m#0", "m", "needle common text"),
    ]

    hits = SparseIndex(chunks).search("needle", k=3)

    assert [chunk_id for chunk_id, _ in hits] == [
        "z#0",
        "a#0",
        "m#0",
    ]


def test_rrf_equal_scores_preserve_first_seen_order():
    fused = rrf(
        [
            ["z", "a"],
            ["a", "z"],
        ]
    )

    assert [doc_id for doc_id, _ in fused] == ["z", "a"]


def test_reranker_equal_scores_preserve_input_order():
    chunks = [
        Chunk("z#0", "z", "first candidate"),
        Chunk("a#0", "a", "second candidate"),
        Chunk("m#0", "m", "third candidate"),
    ]

    def equal_score(pairs):
        return [0.5] * len(pairs)

    ranked = Reranker(equal_score).rerank(
        "query",
        chunks,
        top_n=3,
    )

    assert [chunk.chunk_id for chunk in ranked] == [
        "z#0",
        "a#0",
        "m#0",
    ]
