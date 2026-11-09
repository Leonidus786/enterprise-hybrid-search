
from __future__ import annotations

import hashlib

import numpy as np
import pytest

import hs.index_store as store
from hs.chunking import chunk_text
from hs.documents import Document


MODEL_ID = "test/idempotency-embedder"
MODEL_REVISION = "c" * 40

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


def _embedder(calls: list[list[str]] | None = None):
    def embed(texts: list[str]) -> np.ndarray:
        if calls is not None:
            calls.append(list(texts))

        if not texts:
            return np.empty((0, 16), dtype=np.float32)

        return np.stack([_vector_for(text) for text in texts], axis=0)

    embed.model_metadata = {
        "model_id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "revision_verified": True,
    }
    embed.embedding_config = dict(DOCUMENT_EMBEDDING_CONFIG)

    return embed


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


def _documents() -> list[Document]:
    return [
        Document(
            "doc-a",
            "alpha bravo",
            source="a.txt",
            title="A document",
        ),
        Document(
            "doc-c",
            "charlie delta echo foxtrot",
            source="c-v1.txt",
            title="C original",
            metadata={"revision": 1},
        ),
    ]


def _seed_active_index(root):
    documents = _documents()
    chunks = [
        chunk
        for document in documents
        for chunk in _chunks_for(document)
    ]

    return store.build_and_activate_index_snapshot(
        root,
        index_id="idx-before-idempotency",
        documents=documents,
        chunks=chunks,
        embed_fn=_embedder(),
        embedding_model_id=MODEL_ID,
        embedding_model_revision=MODEL_REVISION,
        embedding_config=STORED_EMBEDDING_CONFIG,
        sparse_config=SPARSE_CONFIG,
        created_at_utc=CREATED_AT,
    )


def _rebuild(root, *, index_id, documents, embed_fn):
    return store.rebuild_and_activate_index_snapshot(
        root,
        index_id=index_id,
        documents=documents,
        embed_fn=embed_fn,
        embedding_model_id=MODEL_ID,
        embedding_model_revision=MODEL_REVISION,
        embedding_config=STORED_EMBEDDING_CONFIG,
        sparse_config=SPARSE_CONFIG,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def _insert(root, *, index_id, document, embed_fn):
    return store.insert_document_and_activate_index_snapshot(
        root,
        index_id=index_id,
        document=document,
        embed_fn=embed_fn,
    )


def _update(root, *, index_id, document, embed_fn):
    return store.update_document_and_activate_index_snapshot(
        root,
        index_id=index_id,
        document=document,
        embed_fn=embed_fn,
    )


def _delete(root, *, index_id, doc_id):
    return store.delete_document_and_activate_index_snapshot(
        root,
        index_id=index_id,
        doc_id=doc_id,
    )


def _filesystem_state(root):
    pointer = (root / "CURRENT.json").read_bytes()
    versions = tuple(
        sorted(path.name for path in (root / "versions").iterdir())
    )
    return pointer, versions


def _assert_same_snapshot(first, retry):
    assert retry.path == first.path
    assert retry.manifest == first.manifest
    assert retry.documents == first.documents
    assert retry.chunks == first.chunks

    np.testing.assert_array_equal(
        retry.vectors,
        first.vectors,
    )


def _assert_retry_did_not_mutate(root, state_before):
    assert _filesystem_state(root) == state_before
    assert list((root / "staging").iterdir()) == []


def test_rebuild_retry_returns_same_snapshot_without_embedding(tmp_path):
    root = tmp_path / "indexes"
    documents = _documents()

    first_calls: list[list[str]] = []
    first = _rebuild(
        root,
        index_id="idx-rebuild-idempotent",
        documents=documents,
        embed_fn=_embedder(first_calls),
    )

    assert len(first_calls) == 1

    state_before = _filesystem_state(root)

    retry_calls: list[list[str]] = []
    retry = _rebuild(
        root,
        index_id="idx-rebuild-idempotent",
        documents=documents,
        embed_fn=_embedder(retry_calls),
    )

    assert retry_calls == []
    _assert_same_snapshot(first, retry)
    _assert_retry_did_not_mutate(root, state_before)


def test_rebuild_rejects_different_corpus_for_active_version_id(tmp_path):
    root = tmp_path / "indexes"

    first = _rebuild(
        root,
        index_id="idx-rebuild-conflict",
        documents=_documents(),
        embed_fn=_embedder(),
    )

    changed_documents = [
        Document(
            "doc-a",
            "alpha bravo changed",
            source="a.txt",
            title="A document",
        ),
        _documents()[1],
    ]

    state_before = _filesystem_state(root)
    retry_calls: list[list[str]] = []

    with pytest.raises(ValueError, match="idempotency conflict"):
        _rebuild(
            root,
            index_id="idx-rebuild-conflict",
            documents=changed_documents,
            embed_fn=_embedder(retry_calls),
        )

    assert retry_calls == []
    assert store.load_current_index_snapshot(root).path == first.path
    _assert_retry_did_not_mutate(root, state_before)


def test_insert_retry_returns_same_snapshot_without_embedding(tmp_path):
    root = tmp_path / "indexes"
    _seed_active_index(root)

    document = Document(
        "doc-b",
        "golf hotel india",
        source="b.txt",
        title="B document",
    )

    first_calls: list[list[str]] = []
    first = _insert(
        root,
        index_id="idx-insert-idempotent",
        document=document,
        embed_fn=_embedder(first_calls),
    )

    assert len(first_calls) == 1

    state_before = _filesystem_state(root)

    retry_calls: list[list[str]] = []
    retry = _insert(
        root,
        index_id="idx-insert-idempotent",
        document=document,
        embed_fn=_embedder(retry_calls),
    )

    assert retry_calls == []
    _assert_same_snapshot(first, retry)
    _assert_retry_did_not_mutate(root, state_before)


def test_insert_rejects_different_document_revision_for_same_version_id(
    tmp_path,
):
    root = tmp_path / "indexes"
    _seed_active_index(root)

    first_document = Document("doc-b", "golf hotel india")

    first = _insert(
        root,
        index_id="idx-insert-conflict",
        document=first_document,
        embed_fn=_embedder(),
    )

    changed_document = Document("doc-b", "different replacement content")

    state_before = _filesystem_state(root)
    retry_calls: list[list[str]] = []

    with pytest.raises(ValueError, match="idempotency conflict"):
        _insert(
            root,
            index_id="idx-insert-conflict",
            document=changed_document,
            embed_fn=_embedder(retry_calls),
        )

    assert retry_calls == []
    assert store.load_current_index_snapshot(root).path == first.path
    _assert_retry_did_not_mutate(root, state_before)


def test_update_retry_returns_same_snapshot_without_embedding(tmp_path):
    root = tmp_path / "indexes"
    _seed_active_index(root)

    document = Document(
        "doc-c",
        "golf hotel india juliet",
        source="c-v2.txt",
        title="C updated",
        metadata={"revision": 2},
    )

    first_calls: list[list[str]] = []
    first = _update(
        root,
        index_id="idx-update-idempotent",
        document=document,
        embed_fn=_embedder(first_calls),
    )

    assert len(first_calls) == 1

    state_before = _filesystem_state(root)

    retry_calls: list[list[str]] = []
    retry = _update(
        root,
        index_id="idx-update-idempotent",
        document=document,
        embed_fn=_embedder(retry_calls),
    )

    assert retry_calls == []
    _assert_same_snapshot(first, retry)
    _assert_retry_did_not_mutate(root, state_before)


def test_update_rejects_different_revision_for_same_version_id(tmp_path):
    root = tmp_path / "indexes"
    _seed_active_index(root)

    first_document = Document(
        "doc-c",
        "golf hotel india juliet",
        metadata={"revision": 2},
    )

    first = _update(
        root,
        index_id="idx-update-conflict",
        document=first_document,
        embed_fn=_embedder(),
    )

    changed_document = Document(
        "doc-c",
        "completely different content",
        metadata={"revision": 3},
    )

    state_before = _filesystem_state(root)
    retry_calls: list[list[str]] = []

    with pytest.raises(ValueError, match="idempotency conflict"):
        _update(
            root,
            index_id="idx-update-conflict",
            document=changed_document,
            embed_fn=_embedder(retry_calls),
        )

    assert retry_calls == []
    assert store.load_current_index_snapshot(root).path == first.path
    _assert_retry_did_not_mutate(root, state_before)


def test_delete_retry_returns_same_snapshot_without_reactivation(tmp_path):
    root = tmp_path / "indexes"
    _seed_active_index(root)

    first = _delete(
        root,
        index_id="idx-delete-idempotent",
        doc_id="doc-c",
    )

    state_before = _filesystem_state(root)

    retry = _delete(
        root,
        index_id="idx-delete-idempotent",
        doc_id="doc-c",
    )

    _assert_same_snapshot(first, retry)
    _assert_retry_did_not_mutate(root, state_before)


def test_delete_rejects_different_document_for_same_version_id(tmp_path):
    root = tmp_path / "indexes"
    _seed_active_index(root)

    first = _delete(
        root,
        index_id="idx-delete-conflict",
        doc_id="doc-c",
    )

    state_before = _filesystem_state(root)

    with pytest.raises(ValueError, match="idempotency conflict"):
        _delete(
            root,
            index_id="idx-delete-conflict",
            doc_id="doc-a",
        )

    assert store.load_current_index_snapshot(root).path == first.path
    _assert_retry_did_not_mutate(root, state_before)


def test_inactive_immutable_version_cannot_be_reactivated_by_retry(tmp_path):
    root = tmp_path / "indexes"
    _seed_active_index(root)

    first_document = Document("doc-b", "golf hotel")
    second_document = Document("doc-d", "kilo lima")

    first = _insert(
        root,
        index_id="idx-first-insert",
        document=first_document,
        embed_fn=_embedder(),
    )

    _insert(
        root,
        index_id="idx-second-insert",
        document=second_document,
        embed_fn=_embedder(),
    )

    state_before = _filesystem_state(root)

    with pytest.raises(FileExistsError):
        _insert(
            root,
            index_id="idx-first-insert",
            document=first_document,
            embed_fn=_embedder(),
        )

    current = store.load_current_index_snapshot(root)

    assert current.manifest["index_id"] == "idx-second-insert"
    assert current.path != first.path
    _assert_retry_did_not_mutate(root, state_before)
