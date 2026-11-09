
from __future__ import annotations

import hashlib

import numpy as np
import pytest

import hs.index_store as store
from hs.chunking import chunk_text
from hs.documents import Document


MODEL_ID = "test/update-embedder"
MODEL_REVISION = "a" * 40

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

CREATED_AT = "2026-11-06T00:00:00+00:00"
CHUNK_SIZE = 2
CHUNK_OVERLAP = 0


def _vector_for(text: str, *, dimension: int = 16) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (
        np.frombuffer(digest[:dimension], dtype=np.uint8)
        .astype(np.float32)
        + 1.0
    )


def _embedder(
    calls: list[list[str]] | None = None,
    *,
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
    config: dict | None = None,
    output_dimension: int = 16,
    revision_verified: bool = True,
):
    def embed(texts: list[str]) -> np.ndarray:
        if calls is not None:
            calls.append(list(texts))

        if not texts:
            return np.empty(
                (0, output_dimension),
                dtype=np.float32,
            )

        return np.stack(
            [
                _vector_for(text, dimension=output_dimension)
                for text in texts
            ],
            axis=0,
        )

    embed.model_metadata = {
        "model_id": model_id,
        "resolved_revision": revision,
        "revision_verified": revision_verified,
    }

    embed.embedding_config = dict(
        DOCUMENT_EMBEDDING_CONFIG if config is None else config
    )

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


def _seed_active_index(root):
    """Create an active index containing two searchable documents."""
    documents = [
        Document(
            "doc-c",
            "charlie delta echo foxtrot",
            source="c-v1.txt",
            title="C original",
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

    snapshot = store.build_and_activate_index_snapshot(
        root,
        index_id="idx-before-update",
        documents=documents,
        chunks=chunks,
        embed_fn=_embedder(),
        embedding_model_id=MODEL_ID,
        embedding_model_revision=MODEL_REVISION,
        embedding_config=STORED_EMBEDDING_CONFIG,
        sparse_config=SPARSE_CONFIG,
        created_at_utc=CREATED_AT,
    )

    return snapshot


def _seed_single_document_index(root):
    """Create an index whose only searchable document may be emptied."""
    document = Document(
        "doc-only",
        "alpha bravo",
        source="only.txt",
    )
    chunks = _chunks_for(document)

    return store.build_and_activate_index_snapshot(
        root,
        index_id="idx-single-document",
        documents=[document],
        chunks=chunks,
        embed_fn=_embedder(),
        embedding_model_id=MODEL_ID,
        embedding_model_revision=MODEL_REVISION,
        embedding_config=STORED_EMBEDDING_CONFIG,
        sparse_config=SPARSE_CONFIG,
        created_at_utc=CREATED_AT,
    )


def _update(root, *, index_id, document, embed_fn):
    return store.update_document_and_activate_index_snapshot(
        root,
        index_id=index_id,
        document=document,
        embed_fn=embed_fn,
    )


def test_update_replaces_revision_embeds_only_new_chunks_and_preserves_other_vectors(
    tmp_path,
):
    root = tmp_path / "indexes"
    before = _seed_active_index(root)

    old_document = next(
        item for item in before.documents
        if item.doc_id == "doc-c"
    )

    old_untouched_vectors = {
        chunk.chunk_id: before.vectors[row].copy()
        for row, chunk in enumerate(before.chunks)
        if chunk.doc_id != "doc-c"
    }

    updated_document = Document(
        "doc-c",
        "golf hotel india juliet kilo lima",
        source="c-v2.txt",
        title="C updated",
        metadata={
            "revision": 2,
            "department": "engineering",
        },
    )

    assert updated_document.revision_sha256 != old_document.revision_sha256

    expected_new_chunks = _chunks_for(updated_document)
    calls: list[list[str]] = []
    embedder = _embedder(calls)

    updated = _update(
        root,
        index_id="idx-after-update",
        document=updated_document,
        embed_fn=embedder,
    )

    expected_texts = [
        chunk.text for chunk in expected_new_chunks
    ]

    # Only the updated document's chunks are embedded.
    assert calls == [expected_texts]

    # Document ordering is canonical and the document occurs only once.
    assert [item.doc_id for item in updated.documents] == [
        "doc-a",
        "doc-c",
    ]

    updated_docs = [
        item for item in updated.documents
        if item.doc_id == "doc-c"
    ]
    assert len(updated_docs) == 1
    assert updated_docs[0].revision_sha256 == updated_document.revision_sha256
    assert updated_docs[0].source == "c-v2.txt"
    assert updated_docs[0].title == "C updated"
    assert updated_docs[0].metadata == {
        "revision": 2,
        "department": "engineering",
    }

    # No stale document content or stale document revision remains.
    updated_chunks = [
        chunk for chunk in updated.chunks
        if chunk.doc_id == "doc-c"
    ]

    assert [chunk.chunk_id for chunk in updated_chunks] == [
        "doc-c#0",
        "doc-c#1",
        "doc-c#2",
    ]

    assert [chunk.text for chunk in updated_chunks] == expected_texts

    assert all(
        chunk.document_revision_sha256 == updated_document.revision_sha256
        for chunk in updated_chunks
    )

    assert all(chunk.source == "c-v2.txt" for chunk in updated_chunks)
    assert all(chunk.title == "C updated" for chunk in updated_chunks)

    # Unchanged documents retain exactly their original vectors.
    for row, chunk in enumerate(updated.chunks):
        if chunk.chunk_id in old_untouched_vectors:
            np.testing.assert_array_equal(
                updated.vectors[row],
                old_untouched_vectors[chunk.chunk_id],
            )

    # Updated document vectors are normalized.
    updated_rows = [
        row
        for row, chunk in enumerate(updated.chunks)
        if chunk.doc_id == "doc-c"
    ]

    norms = np.linalg.norm(updated.vectors[updated_rows], axis=1)

    np.testing.assert_allclose(
        norms,
        np.ones(len(updated_rows)),
        rtol=1e-5,
        atol=1e-6,
    )

    # Existing index configuration is preserved.
    assert updated.manifest["embedding"] == before.manifest["embedding"]
    assert updated.manifest["chunking"] == before.manifest["chunking"]
    assert updated.manifest["sparse"] == before.manifest["sparse"]

    assert updated.manifest["corpus"]["document_count"] == 2
    assert updated.manifest["corpus"]["chunk_count"] == 4

    # New snapshot activated; old immutable snapshot remains available.
    pointer = store._read_current_pointer(root)

    assert pointer["index_id"] == "idx-after-update"
    assert pointer["previous_index_id"] == "idx-before-update"

    assert (root / "versions" / "idx-before-update").is_dir()
    assert (root / "versions" / "idx-after-update").is_dir()

    # Persisted state must match in-memory state after reload.
    reloaded = store.load_current_index_snapshot(root)

    assert [chunk.chunk_id for chunk in reloaded.chunks] == [
        chunk.chunk_id for chunk in updated.chunks
    ]

    np.testing.assert_array_equal(
        reloaded.vectors,
        updated.vectors,
    )


def test_update_rejects_unknown_document_id_before_embedding_or_activation(
    tmp_path,
):
    root = tmp_path / "indexes"
    _seed_active_index(root)

    pointer_before = (root / "CURRENT.json").read_bytes()
    versions_before = sorted(
        path.name for path in (root / "versions").iterdir()
    )

    calls: list[list[str]] = []

    with pytest.raises(ValueError, match="(not found|does not exist|unknown)"):
        _update(
            root,
            index_id="idx-unknown-update",
            document=Document("doc-unknown", "new text"),
            embed_fn=_embedder(calls),
        )

    assert calls == []
    assert (root / "CURRENT.json").read_bytes() == pointer_before

    assert sorted(
        path.name for path in (root / "versions").iterdir()
    ) == versions_before


def test_update_to_empty_document_retains_record_and_preserves_other_documents(
    tmp_path,
):
    root = tmp_path / "indexes"
    before = _seed_active_index(root)

    old_untouched_vectors = {
        chunk.chunk_id: before.vectors[row].copy()
        for row, chunk in enumerate(before.chunks)
        if chunk.doc_id == "doc-a"
    }

    calls: list[list[str]] = []

    empty_document = Document(
        "doc-c",
        " \n\t ",
        source="c-empty.txt",
        title="C empty",
        metadata={"revision": 2},
    )

    updated = _update(
        root,
        index_id="idx-empty-update",
        document=empty_document,
        embed_fn=_embedder(calls),
    )

    assert calls == []

    assert "doc-c" in {
        document.doc_id for document in updated.documents
    }

    assert "doc-c" not in {
        chunk.doc_id for chunk in updated.chunks
    }

    assert [chunk.chunk_id for chunk in updated.chunks] == ["doc-a#0"]

    for row, chunk in enumerate(updated.chunks):
        np.testing.assert_array_equal(
            updated.vectors[row],
            old_untouched_vectors[chunk.chunk_id],
        )

    assert updated.manifest["corpus"]["document_count"] == 2
    assert updated.manifest["corpus"]["chunk_count"] == 1


@pytest.mark.parametrize(
    "embedder_kwargs",
    [
        {"model_id": "test/different-model"},
        {"revision": "b" * 40},
        {
            "config": {
                **DOCUMENT_EMBEDDING_CONFIG,
                "normalize_embeddings": False,
            }
        },
    ],
    ids=[
        "different-model",
        "different-immutable-revision",
        "different-vector-configuration",
    ],
)
def test_update_rejects_incompatible_embedder_before_embedding(
    tmp_path,
    embedder_kwargs,
):
    root = tmp_path / "indexes"
    _seed_active_index(root)

    pointer_before = (root / "CURRENT.json").read_bytes()
    calls: list[list[str]] = []

    with pytest.raises(ValueError):
        _update(
            root,
            index_id="idx-incompatible-update",
            document=Document("doc-c", "new content here"),
            embed_fn=_embedder(calls, **embedder_kwargs),
        )

    assert calls == []
    assert (root / "CURRENT.json").read_bytes() == pointer_before
    assert not (
        root / "versions" / "idx-incompatible-update"
    ).exists()


def test_update_rejects_unverified_model_revision_before_embedding(
    tmp_path,
):
    root = tmp_path / "indexes"
    _seed_active_index(root)

    pointer_before = (root / "CURRENT.json").read_bytes()
    calls: list[list[str]] = []

    with pytest.raises(ValueError):
        _update(
            root,
            index_id="idx-unverified-update",
            document=Document("doc-c", "new content here"),
            embed_fn=_embedder(
                calls,
                revision_verified=False,
            ),
        )

    assert calls == []
    assert (root / "CURRENT.json").read_bytes() == pointer_before


def test_update_rejects_vector_dimension_mismatch_without_activation(
    tmp_path,
):
    root = tmp_path / "indexes"
    _seed_active_index(root)

    pointer_before = (root / "CURRENT.json").read_bytes()
    calls: list[list[str]] = []

    updated_document = Document(
        "doc-c",
        "golf hotel india",
    )

    with pytest.raises(ValueError, match="(dimension|shape|match)"):
        _update(
            root,
            index_id="idx-wrong-dimension-update",
            document=updated_document,
            embed_fn=_embedder(
                calls,
                output_dimension=8,
            ),
        )

    assert calls == [
        [chunk.text for chunk in _chunks_for(updated_document)]
    ]

    assert (root / "CURRENT.json").read_bytes() == pointer_before

    assert not (
        root / "versions" / "idx-wrong-dimension-update"
    ).exists()


def test_embedding_failure_preserves_previous_active_snapshot(
    tmp_path,
):
    root = tmp_path / "indexes"
    before = _seed_active_index(root)

    pointer_before = (root / "CURRENT.json").read_bytes()
    calls: list[list[str]] = []

    def broken_embedder(texts: list[str]) -> np.ndarray:
        calls.append(list(texts))
        raise OSError("simulated embedding failure")

    broken_embedder.model_metadata = {
        "model_id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "revision_verified": True,
    }

    broken_embedder.embedding_config = dict(
        DOCUMENT_EMBEDDING_CONFIG
    )

    with pytest.raises(RuntimeError, match="embedding generation failed"):
        _update(
            root,
            index_id="idx-failed-update",
            document=Document("doc-c", "golf hotel india"),
            embed_fn=broken_embedder,
        )

    assert calls == [["golf hotel", "india"]]

    assert (root / "CURRENT.json").read_bytes() == pointer_before

    current = store.load_current_index_snapshot(root)

    assert current.manifest["index_id"] == before.manifest["index_id"]

    assert not (root / "versions" / "idx-failed-update").exists()

    assert list((root / "staging").iterdir()) == []


def test_update_rejects_non_document_input_without_embedding(
    tmp_path,
):
    root = tmp_path / "indexes"
    _seed_active_index(root)

    pointer_before = (root / "CURRENT.json").read_bytes()
    calls: list[list[str]] = []

    with pytest.raises(TypeError, match="Document"):
        _update(
            root,
            index_id="idx-invalid-update",
            document={
                "doc_id": "doc-c",
                "text": "replacement text",
            },
            embed_fn=_embedder(calls),
        )

    assert calls == []
    assert (root / "CURRENT.json").read_bytes() == pointer_before


def test_update_cannot_remove_last_searchable_document(
    tmp_path,
):
    root = tmp_path / "indexes"
    _seed_single_document_index(root)

    pointer_before = (root / "CURRENT.json").read_bytes()
    calls: list[list[str]] = []

    with pytest.raises(ValueError):
        _update(
            root,
            index_id="idx-all-empty-update",
            document=Document("doc-only", " \n\t "),
            embed_fn=_embedder(calls),
        )

    assert calls == []
    assert (root / "CURRENT.json").read_bytes() == pointer_before

    current = store.load_current_index_snapshot(root)

    assert current.manifest["index_id"] == "idx-single-document"
    assert list((root / "staging").iterdir()) == []
