from __future__ import annotations

import numpy as np
import pytest

from hs.chunking import Chunk, chunk_text
from hs.dense import DenseIndex
from hs.documents import Document
from hs.index_store import build_index_snapshot, load_index_snapshot


@pytest.fixture
def chunks():
    return [
        Chunk("doc-a#0", "doc-a", "alpha"),
        Chunk("doc-b#0", "doc-b", "beta"),
        Chunk("doc-c#0", "doc-c", "gamma"),
        Chunk("doc-d#0", "doc-d", "delta"),
    ]


def _document_embeddings(texts):
    vectors = {
        "alpha": [3.0, 4.0, 0.0],
        "beta": [1.0, 0.0, 0.0],
        "gamma": [0.0, 0.0, 0.0],
        "delta": [0.0, 1.0, 0.0],
    }
    return np.asarray([vectors[text] for text in texts], dtype=np.float32)


def _query_embedding(texts):
    assert len(texts) == 1
    assert texts[0] == "query:alpha"
    return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)


def test_precomputed_matrix_matches_fresh_dense_index_rankings_and_scores(chunks):
    fresh = DenseIndex(
        chunks,
        _document_embeddings,
        query_prefix="query:",
        query_embed_fn=_query_embedding,
    )
    stored_matrix = fresh.matrix.copy()
    stored_matrix.setflags(write=False)
    loaded = DenseIndex(
        chunks,
        query_prefix="query:",
        query_embed_fn=_query_embedding,
        normalized_embedding_matrix=stored_matrix,
    )

    assert loaded.matrix.dtype == np.float32
    assert loaded.matrix.flags.c_contiguous
    assert not loaded.matrix.flags.writeable
    assert np.array_equal(loaded.matrix, stored_matrix)
    assert loaded.search("alpha", k=4) == fresh.search("alpha", k=4)


def test_precomputed_matrix_does_not_call_document_embedder(chunks):
    stored_matrix = _document_embeddings([chunk.text for chunk in chunks])
    norms = np.linalg.norm(stored_matrix, axis=1, keepdims=True)
    stored_matrix = stored_matrix / np.where(norms == 0, 1, norms)
    calls = []

    def forbidden_document_embedder(texts):
        calls.append(list(texts))
        raise AssertionError("document embeddings must not be regenerated")

    index = DenseIndex(
        chunks,
        forbidden_document_embedder,
        query_prefix="query:",
        query_embed_fn=_query_embedding,
        normalized_embedding_matrix=stored_matrix,
    )

    assert calls == []
    index.search("alpha", k=2)
    assert calls == []


def test_precomputed_matrix_can_be_loaded_without_document_embed_fn(chunks):
    stored_matrix = _document_embeddings([chunk.text for chunk in chunks])
    norms = np.linalg.norm(stored_matrix, axis=1, keepdims=True)
    stored_matrix = stored_matrix / np.where(norms == 0, 1, norms)

    index = DenseIndex(
        chunks,
        query_prefix="query:",
        query_embed_fn=_query_embedding,
        normalized_embedding_matrix=stored_matrix,
    )

    assert [chunk_id for chunk_id, _ in index.search("alpha", k=2)] == [
        chunk_id for chunk_id, _ in DenseIndex(
            chunks,
            _document_embeddings,
            query_prefix="query:",
            query_embed_fn=_query_embedding,
        ).search("alpha", k=2)
    ]


def test_precomputed_matrix_keeps_zero_rows_valid(chunks):
    matrix = _document_embeddings([chunk.text for chunk in chunks])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.where(norms == 0, 1, norms)

    index = DenseIndex(
        chunks,
        query_prefix="query:",
        query_embed_fn=_query_embedding,
        normalized_embedding_matrix=normalized,
    )

    assert np.array_equal(index.matrix[2], np.zeros(3, dtype=np.float32))


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.ones((3, 3), dtype=np.float32), "shape"),
        (np.ones((4, 0), dtype=np.float32), "shape"),
        (np.ones(4, dtype=np.float32), "shape"),
        (np.asarray([[1.0, np.nan, 0.0]] * 4, dtype=np.float32), "finite"),
        (np.asarray([[2.0, 0.0, 0.0]] * 4, dtype=np.float32), "L2-normalized"),
        (
            np.full((4, 3), np.finfo(np.float32).max, dtype=np.float32),
            "invalid norms",
        ),
        (
            np.tile(
                np.asarray(
                    [np.nextafter(np.float32(0), np.float32(1)), 0.0, 0.0],
                    dtype=np.float32,
                ),
                (4, 1),
            ),
            "zero norm",
        ),
    ],
)
def test_precomputed_matrix_rejects_invalid_values(chunks, matrix, message):
    with pytest.raises(ValueError, match=message):
        DenseIndex(
            chunks,
            query_embed_fn=_query_embedding,
            normalized_embedding_matrix=matrix,
        )


def test_precomputed_matrix_requires_query_embedder_when_no_embed_fn(chunks):
    with pytest.raises(ValueError, match="query_embed_fn or embed_fn is required"):
        DenseIndex(
            chunks,
            normalized_embedding_matrix=np.ones((4, 3), dtype=np.float32)
            / np.sqrt(3.0),
        )


def test_precomputed_matrix_rejects_empty_corpus():
    with pytest.raises(ValueError, match="empty corpus"):
        DenseIndex(
            [],
            query_embed_fn=_query_embedding,
            normalized_embedding_matrix=np.empty((0, 3), dtype=np.float32),
        )


def test_nonempty_corpus_still_requires_document_embedder(chunks):
    with pytest.raises(ValueError, match="embed_fn is required"):
        DenseIndex(chunks, query_embed_fn=_query_embedding)


def test_existing_empty_index_behavior_is_preserved():
    index = DenseIndex([], None)
    assert index.matrix is None
    assert index.search("anything") == []


def test_precomputed_matrix_requires_callable_query_embedder(chunks):
    matrix = _document_embeddings([chunk.text for chunk in chunks])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.where(norms == 0, 1, norms)
    with pytest.raises(TypeError, match="query_embed_fn must be callable"):
        DenseIndex(
            chunks,
            query_embed_fn="not-callable",
            normalized_embedding_matrix=normalized,
        )


def test_loaded_snapshot_initializes_dense_index_without_document_reembedding(tmp_path):
    documents = [
        Document("doc-1", "alpha beta"),
        Document("doc-2", "gamma delta"),
    ]
    chunks = []
    for document in documents:
        chunks.extend(
            chunk_text(
                document.doc_id,
                document.text,
                size=1,
                overlap=0,
                source=document.source,
                title=document.title,
                metadata=document.metadata,
                document_revision_sha256=document.revision_sha256,
            )
        )

    document_embed_calls = []

    def build_embedder(texts):
        document_embed_calls.append(list(texts))
        return _document_embeddings(texts)

    snapshot_dir = tmp_path / "snapshot"
    build_index_snapshot(
        snapshot_dir,
        index_id="dense-integration-test",
        documents=documents,
        chunks=chunks,
        embed_fn=build_embedder,
        embedding_model_id="test/fake-embedder",
        embedding_model_revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        embedding_config={
            "adapter": "test-deterministic-embedder-v1",
            "normalize_embeddings": True,
        },
        sparse_config={"algorithm": "bm25", "version": 1},
        created_at_utc="2026-11-02T00:00:00+00:00",
    )
    loaded = load_index_snapshot(snapshot_dir)
    assert len(document_embed_calls) == 1

    dense = DenseIndex.from_snapshot(
        loaded,
        query_prefix="query:",
        query_embed_fn=_query_embedding,
    )

    assert dense.matrix is loaded.vectors
    assert not dense.matrix.flags.writeable
    assert document_embed_calls == [[chunk.text for chunk in chunks]]

    fresh = DenseIndex(
        loaded.chunks,
        _document_embeddings,
        query_prefix="query:",
        query_embed_fn=_query_embedding,
    )
    persisted_results = dense.search("alpha", k=4)
    fresh_results = fresh.search("alpha", k=4)
    assert [item[0] for item in persisted_results] == [
        item[0] for item in fresh_results
    ]
    assert [item[1] for item in persisted_results] == pytest.approx(
        [item[1] for item in fresh_results]
    )
    assert document_embed_calls == [[chunk.text for chunk in chunks]]


def test_from_snapshot_rejects_unvalidated_objects():
    with pytest.raises(TypeError, match="LoadedIndexSnapshot"):
        DenseIndex.from_snapshot(object(), query_embed_fn=_query_embedding)
