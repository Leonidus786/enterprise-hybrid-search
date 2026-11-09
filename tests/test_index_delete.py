
from __future__ import annotations

import hashlib

import numpy as np
import pytest

import hs.index_store as store
from hs.chunking import chunk_text
from hs.documents import Document


MODEL_ID = "test/delete-embedder"
MODEL_REVISION = "b" * 40

DOCUMENT_EMBEDDING_CONFIG = {
    "adapter": "test-deterministic-embedder-v1",
    "normalize_embeddings": True,
    "dimension": 16,
}

STORED_EMBEDDING_CONFIG = {
    **DOCUMENT_EMBEDDING_CONFIG,
    "query_prefix": "query:",
}

SPARSE_CONFIG = {
    "algorithm": "bm25",
    "version": 1,
    "k1": 1.5,
    "b": 0.75,
}

CREATED_AT = "2026-11-08T00:00:00+00:00"
CHUNK_SIZE = 2
CHUNK_OVERLAP = 0


def _vector_for(text: str) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (
        np.frombuffer(digest[:16], dtype=np.uint8).astype(np.float32)
        + 1.0
    )


def _embedder(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 16), dtype=np.float32)

    return np.stack([_vector_for(text) for text in texts], axis=0)


_embedder.model_metadata = {
    "model_id": MODEL_ID,
    "resolved_revision": MODEL_REVISION,
    "revision_verified": True,
}

_embedder.embedding_config = dict(DOCUMENT_EMBEDDING_CONFIG)


def _chunks_for(document: Document) -> list:
    return chunk_text(
        document.doc_id,
        document.text,
        size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
        source=document.source,
        title=document.title,
        metadata=document.metadata,
        document_revision_sha256=document.revision_sha256,
    )


def _seed_active_index(root, *, documents=None):
    if documents is None:
        documents = [
            Document(
                "doc-c",
                "charlie delta echo foxtrot",
                source="c.txt",
                title="C document",
                metadata={"revision": 1},
            ),
            Document(
                "doc-a",
                "alpha bravo",
                source="a.txt",
                title="A document",
            ),
        ]

    chunks = [
        chunk
        for document in documents
        for chunk in _chunks_for(document)
    ]

    return store.build_and_activate_index_snapshot(
        root,
        index_id="idx-before-delete",
        documents=documents,
        chunks=chunks,
        embed_fn=_embedder,
        embedding_model_id=MODEL_ID,
        embedding_model_revision=MODEL_REVISION,
        embedding_config=STORED_EMBEDDING_CONFIG,
        sparse_config=SPARSE_CONFIG,
        created_at_utc=CREATED_AT,
    )


def _delete(root, *, index_id, doc_id):
    return store.delete_document_and_activate_index_snapshot(
        root,
        index_id=index_id,
        doc_id=doc_id,
    )


def test_delete_removes_document_and_chunks_preserves_vectors_and_activates_snapshot(
    tmp_path,
):
    root = tmp_path / "indexes"
    before = _seed_active_index(root)

    untouched_vectors = {
        chunk.chunk_id: before.vectors[row].copy()
        for row, chunk in enumerate(before.chunks)
        if chunk.doc_id == "doc-a"
    }

    deleted = _delete(
        root,
        index_id="idx-after-delete",
        doc_id="doc-c",
    )

    # Deleted document and all its chunks are absent.
    assert [item.doc_id for item in deleted.documents] == ["doc-a"]
    assert [chunk.chunk_id for chunk in deleted.chunks] == ["doc-a#0"]
    assert all(chunk.doc_id != "doc-c" for chunk in deleted.chunks)

    # Remaining document vector is preserved exactly.
    assert set(untouched_vectors) == {"doc-a#0"}

    np.testing.assert_array_equal(
        deleted.vectors[0],
        untouched_vectors["doc-a#0"],
    )

    # Index configuration remains unchanged.
    assert deleted.manifest["embedding"] == before.manifest["embedding"]
    assert deleted.manifest["chunking"] == before.manifest["chunking"]
    assert deleted.manifest["sparse"] == before.manifest["sparse"]

    assert deleted.manifest["corpus"]["document_count"] == 1
    assert deleted.manifest["corpus"]["chunk_count"] == 1

    # New immutable snapshot is active; previous snapshot remains available.
    pointer = store._read_current_pointer(root)

    assert pointer["index_id"] == "idx-after-delete"
    assert pointer["previous_index_id"] == "idx-before-delete"

    assert (root / "versions" / "idx-before-delete").is_dir()
    assert (root / "versions" / "idx-after-delete").is_dir()

    # Reloaded snapshot matches persisted vectors and chunk ordering.
    reloaded = store.load_current_index_snapshot(root)

    assert [item.doc_id for item in reloaded.documents] == ["doc-a"]
    assert [chunk.chunk_id for chunk in reloaded.chunks] == ["doc-a#0"]

    np.testing.assert_array_equal(reloaded.vectors, deleted.vectors)


def test_delete_rejects_unknown_document_without_changing_active_snapshot(
    tmp_path,
):
    root = tmp_path / "indexes"
    _seed_active_index(root)

    pointer_before = (root / "CURRENT.json").read_bytes()
    versions_before = sorted(
        path.name for path in (root / "versions").iterdir()
    )

    with pytest.raises(
        ValueError,
        match="(not found|does not exist|unknown)",
    ):
        _delete(
            root,
            index_id="idx-unknown-delete",
            doc_id="doc-unknown",
        )

    assert (root / "CURRENT.json").read_bytes() == pointer_before

    assert sorted(
        path.name for path in (root / "versions").iterdir()
    ) == versions_before

    assert list((root / "staging").iterdir()) == []


def test_delete_rejects_removal_of_last_searchable_document(
    tmp_path,
):
    root = tmp_path / "indexes"

    _seed_active_index(
        root,
        documents=[
            Document(
                "doc-only",
                "alpha bravo",
                source="only.txt",
            )
        ],
    )

    pointer_before = (root / "CURRENT.json").read_bytes()
    versions_before = sorted(
        path.name for path in (root / "versions").iterdir()
    )

    with pytest.raises(
        ValueError,
        match="(searchable|last document|empty index)",
    ):
        _delete(
            root,
            index_id="idx-empty-delete",
            doc_id="doc-only",
        )

    assert (root / "CURRENT.json").read_bytes() == pointer_before

    current = store.load_current_index_snapshot(root)

    assert current.manifest["index_id"] == "idx-before-delete"

    assert sorted(
        path.name for path in (root / "versions").iterdir()
    ) == versions_before

    assert list((root / "staging").iterdir()) == []


def test_delete_can_remove_empty_document_when_other_searchable_content_remains(
    tmp_path,
):
    root = tmp_path / "indexes"

    documents = [
        Document(
            "doc-empty",
            " \n\t ",
            source="empty.txt",
        ),
        Document(
            "doc-a",
            "alpha bravo",
            source="a.txt",
        ),
    ]

    before = _seed_active_index(root, documents=documents)

    assert {item.doc_id for item in before.documents} == {
        "doc-empty",
        "doc-a",
    }

    assert [chunk.doc_id for chunk in before.chunks] == ["doc-a"]

    old_vector = before.vectors[0].copy()

    deleted = _delete(
        root,
        index_id="idx-after-empty-delete",
        doc_id="doc-empty",
    )

    assert [item.doc_id for item in deleted.documents] == ["doc-a"]
    assert [chunk.chunk_id for chunk in deleted.chunks] == ["doc-a#0"]

    np.testing.assert_array_equal(deleted.vectors[0], old_vector)

    assert deleted.manifest["corpus"]["document_count"] == 1
    assert deleted.manifest["corpus"]["chunk_count"] == 1
