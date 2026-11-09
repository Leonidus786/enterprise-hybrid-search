
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import hs.index_store as store
from hs.documents import Document


MODEL_ID = "test/full-rebuild-embedder"
MODEL_REVISION = "a" * 40

EMBEDDING_CONFIG = {
    "adapter": "sha256-deterministic-test-v1",
    "normalize_embeddings": True,
    "dimension": 8,
}

SPARSE_CONFIG = {
    "algorithm": "bm25",
    "version": 1,
    "k1": 1.5,
    "b": 0.75,
}

CREATED_AT = "2026-11-05T00:00:00+00:00"


def _documents():
    return [
        Document(
            "doc-z",
            "omega psi chi phi",
            source="z.txt",
            title="Z document",
        ),
        Document(
            "doc-empty",
            "   ",
            source="empty.txt",
        ),
        Document(
            "doc-a",
            "alpha beta gamma delta",
            source="a.txt",
            title="A document",
            metadata={"department": "search"},
        ),
        Document(
            "doc-duplicate-content",
            "alpha beta gamma delta",
            source="duplicate.txt",
            metadata={"department": "search"},
        ),
    ]


def _embedder(calls=None):
    def embed(texts):
        if calls is not None:
            calls.append(list(texts))

        rows = []

        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            rows.append([float(value + 1) for value in digest[:8]])

        return np.asarray(rows, dtype=np.float32)

    # Attach verified identity to the callable embedder.
    # These lines belong OUTSIDE the nested embed() function.
    embed.model_metadata = {
        "model_id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "revision_verified": True,
    }
    embed.embedding_config = dict(EMBEDDING_CONFIG)

    return embed


def _rebuild(
    index_root,
    index_id,
    documents,
    embed_fn,
    *,
    chunk_size=3,
    chunk_overlap=1,
):
    return store.rebuild_and_activate_index_snapshot(
        index_root,
        index_id=index_id,
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_fn=embed_fn,
        embedding_model_id=MODEL_ID,
        embedding_model_revision=MODEL_REVISION,
        embedding_config=EMBEDDING_CONFIG,
        sparse_config=SPARSE_CONFIG,
        created_at_utc=CREATED_AT,
    )


def test_full_rebuild_is_deterministic_across_input_order(tmp_path):
    root = tmp_path / "indexes"

    documents = _documents()
    first_calls = []
    second_calls = []

    first = _rebuild(
        root,
        "idx-full-001",
        documents,
        _embedder(first_calls),
    )

    second = _rebuild(
        root,
        "idx-full-002",
        list(reversed(documents)),
        _embedder(second_calls),
    )

    expected_document_ids = [
        "doc-a",
        "doc-duplicate-content",
        "doc-empty",
        "doc-z",
    ]

    assert [doc.doc_id for doc in first.documents] == expected_document_ids
    assert [doc.doc_id for doc in second.documents] == expected_document_ids

    # Empty documents remain in the document collection but are not chunked.
    assert "doc-empty" not in {chunk.doc_id for chunk in first.chunks}

    # Identical content under different document IDs must remain distinct.
    first_duplicate = first.documents[0]
    second_duplicate = first.documents[1]

    assert first_duplicate.content_sha256 == second_duplicate.content_sha256
    assert first_duplicate.revision_sha256 != second_duplicate.revision_sha256

    assert (
        [chunk.chunk_id for chunk in first.chunks]
        == [chunk.chunk_id for chunk in second.chunks]
    )

    assert (
        [chunk.revision_sha256 for chunk in first.chunks]
        == [chunk.revision_sha256 for chunk in second.chunks]
    )

    assert (
        first.manifest["corpus"]["fingerprint_sha256"]
        == second.manifest["corpus"]["fingerprint_sha256"]
    )

    assert first.manifest["chunking"] == second.manifest["chunking"]
    assert first.manifest["embedding"] == second.manifest["embedding"]
    assert first.manifest["sparse"] == second.manifest["sparse"]

    # Document, chunk and vector artifact contents must be reproducible.
    for artifact in store.REQUIRED_ARTIFACTS:
        first_bytes = (first.path / artifact).read_bytes()
        second_bytes = (second.path / artifact).read_bytes()

        assert first_bytes == second_bytes

    np.testing.assert_array_equal(first.vectors, second.vectors)

    # Embedding order must also be canonical.
    assert first_calls == second_calls
    assert first_calls == [[chunk.text for chunk in first.chunks]]

    # A second build activates the new version without deleting the old one.
    assert first.path.is_dir()
    assert second.path.is_dir()

    current = store.load_current_index_snapshot(root)
    assert current.manifest["index_id"] == "idx-full-002"

    pointer = json.loads((root / "CURRENT.json").read_text(encoding="utf-8"))
    assert pointer["previous_index_id"] == "idx-full-001"


def test_failed_full_rebuild_preserves_previous_active_snapshot(tmp_path):
    root = tmp_path / "indexes"

    original = _rebuild(
        root,
        "idx-good",
        _documents(),
        _embedder(),
    )

    pointer_before = (root / "CURRENT.json").read_bytes()

    def broken_embedder(texts):
        raise OSError("simulated model failure")

    broken_embedder.model_metadata = {
        "model_id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "revision_verified": True,
    }


    broken_embedder.embedding_config = dict(EMBEDDING_CONFIG)

    with pytest.raises(RuntimeError, match="embedding generation failed"):
            _rebuild(
                root,
                "idx-failed",
                _documents(),
                broken_embedder,
            )

    assert (root / "CURRENT.json").read_bytes() == pointer_before

    current = store.load_current_index_snapshot(root)

    assert current.manifest["index_id"] == "idx-good"
    assert original.path.is_dir()
    assert not (root / "versions" / "idx-failed").exists()
    assert list((root / "staging").iterdir()) == []


def test_duplicate_document_ids_are_rejected_before_embedding(tmp_path):
    root = tmp_path / "indexes"
    calls = []

    documents = [
        Document("duplicate", "first content"),
        Document("duplicate", "second content"),
    ]

    with pytest.raises(ValueError, match="duplicate document ID"):
        _rebuild(
            root,
            "idx-invalid",
            documents,
            _embedder(calls),
        )

    assert calls == []
    assert not (root / "CURRENT.json").exists()
    assert not (root / "versions").exists()


def test_all_empty_documents_are_rejected_before_embedding(tmp_path):
    root = tmp_path / "indexes"
    calls = []

    documents = [
        Document("empty-one", "   "),
        Document("empty-two", "\n\t"),
    ]

    with pytest.raises(ValueError, match="without searchable chunks"):
        _rebuild(
            root,
            "idx-empty",
            documents,
            _embedder(calls),
        )

    assert calls == []
    assert not (root / "CURRENT.json").exists()
    assert not (root / "versions").exists()
    assert not (root / "staging").exists()


def test_whole_document_chunking_is_supported(tmp_path):
    root = tmp_path / "indexes"

    documents = [
        Document("doc-whole", "  one   two\nthree  "),
    ]

    snapshot = _rebuild(
        root,
        "idx-whole",
        documents,
        _embedder(),
        chunk_size=None,
        chunk_overlap=0,
    )

    assert snapshot.manifest["chunking"] == {
        "algorithm": "whole_document",
        "version": 1,
    }

    assert len(snapshot.chunks) == 1
    assert snapshot.chunks[0].text == "one two three"


@pytest.mark.parametrize(
    "chunk_size, chunk_overlap",
    [
        (0, 0),
        (-1, 0),
        (3, 3),
        (True, 0),
        (3, True),
    ],
)
def test_invalid_chunking_configuration_is_rejected_before_embedding(
    tmp_path,
    chunk_size,
    chunk_overlap,
):
    root = tmp_path / "indexes"
    calls = []

    with pytest.raises(ValueError):
        _rebuild(
            root,
            "idx-invalid-chunking",
            _documents(),
            _embedder(calls),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    assert calls == []
    assert not (root / "CURRENT.json").exists()


@pytest.mark.parametrize(
    "overrides, expected_message",
    [
        (
            {"embedding_model_id": "test/wrong-embedder"},
            "declared embedding model ID",
        ),
        (
            {"embedding_model_revision": "b" * 40},
            "declared embedding revision",
        ),
        (
            {"embedding_config": {**EMBEDDING_CONFIG, "dimension": 9}},
            "declared document embedding config",
        ),
    ],
)
def test_rebuild_rejects_unverified_declared_identity_before_embedding(
    tmp_path,
    overrides,
    expected_message,
):
    root = tmp_path / "indexes"
    calls = []

    kwargs = {
        "index_id": "idx-invalid-identity",
        "documents": _documents(),
        "embed_fn": _embedder(calls),
        "embedding_model_id": MODEL_ID,
        "embedding_model_revision": MODEL_REVISION,
        "embedding_config": EMBEDDING_CONFIG,
        "sparse_config": SPARSE_CONFIG,
        "created_at_utc": CREATED_AT,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=expected_message):
        store.rebuild_and_activate_index_snapshot(root, **kwargs)

    # Identity mismatch must be rejected before embedding inference.
    assert calls == []

    # No index directories or active pointer should be created.
    assert not (root / "CURRENT.json").exists()
    assert not (root / "staging").exists()
    assert not (root / "versions").exists()
