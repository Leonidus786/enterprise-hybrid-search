from __future__ import annotations

import hashlib

import numpy as np
import pytest

import hs.index_store as store
from hs.cache import CachedEmbedder
from hs.chunking import chunk_text
from hs.documents import Document


MODEL_ID = "test/insert-embedder"
MODEL_REVISION = "a" * 40
DOCUMENT_EMBEDDING_CONFIG = {
    "adapter": "test-deterministic-embedder-v1",
    "normalize_embeddings": True,
    "dimension": 16,
}
# The persisted manifest may also retain the query-side prefix. The document
# embedder identity deliberately excludes it, matching HybridPipeline's
# existing compatibility contract.
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
    return np.frombuffer(digest[:dimension], dtype=np.uint8).astype(np.float32) + 1.0


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
            return np.empty((0, output_dimension), dtype=np.float32)
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
    # Deliberately use non-canonical input order. Insert must canonicalize the
    # new snapshot and remap old vectors by chunk_id, not preserve row position.
    documents = [
        Document(
            "doc-c",
            "charlie delta echo foxtrot",
            source="c.txt",
            title="C document",
        ),
        Document(
            "doc-a",
            "alpha bravo",
            source="a.txt",
            title="A document",
        ),
    ]
    chunks = [
        chunk for document in documents for chunk in _chunks_for(document)]
    build_calls: list[list[str]] = []

    snapshot = store.build_and_activate_index_snapshot(
        root,
        index_id="idx-before-insert",
        documents=documents,
        chunks=chunks,
        embed_fn=_embedder(build_calls),
        embedding_model_id=MODEL_ID,
        embedding_model_revision=MODEL_REVISION,
        embedding_config=STORED_EMBEDDING_CONFIG,
        sparse_config=SPARSE_CONFIG,
        created_at_utc=CREATED_AT,
    )
    assert len(build_calls) == 1
    assert build_calls[0] == [chunk.text for chunk in chunks]
    return snapshot


def _insert(root, *, index_id, document, embed_fn):
    return store.insert_document_and_activate_index_snapshot(
        root,
        index_id=index_id,
        document=document,
        embed_fn=embed_fn,
    )


def test_insert_embeds_only_new_chunks_and_reuses_existing_vectors_in_canonical_order(
    tmp_path,
):
    root = tmp_path / "indexes"
    before = _seed_active_index(root)
    old_vectors_by_chunk_id = {
        chunk.chunk_id: before.vectors[row].copy()
        for row, chunk in enumerate(before.chunks)
    }

    new_document = Document(
        "doc-b",
        "bravo charlie delta echo",
        source="b.txt",
        title="B document",
        metadata={"department": "search"},
    )
    expected_new_chunks = _chunks_for(new_document)
    base_calls: list[list[str]] = []
    base_embedder = _embedder(base_calls)
    cached_embedder = CachedEmbedder(
        base_embedder,
        tmp_path / "embedding-cache",
        MODEL_ID,
        model_revision=MODEL_REVISION,
        embedding_config=DOCUMENT_EMBEDDING_CONFIG,
    )

    inserted = _insert(
        root,
        index_id="idx-after-insert",
        document=new_document,
        embed_fn=cached_embedder,
    )

    expected_new_texts = [chunk.text for chunk in expected_new_chunks]
    assert base_calls == [expected_new_texts]

    assert [document.doc_id for document in inserted.documents] == [
        "doc-a",
        "doc-b",
        "doc-c",
    ]
    assert [chunk.doc_id for chunk in inserted.chunks] == [
        "doc-a",
        "doc-b",
        "doc-b",
        "doc-c",
        "doc-c",
    ]
    assert [chunk.chunk_id for chunk in inserted.chunks] == [
        "doc-a#0",
        "doc-b#0",
        "doc-b#1",
        "doc-c#0",
        "doc-c#1",
    ]

    # Every pre-existing chunk must retain exactly the same stored vector,
    # even though canonical ordering changed its row position.
    for row, chunk in enumerate(inserted.chunks):
        if chunk.chunk_id in old_vectors_by_chunk_id:
            np.testing.assert_array_equal(
                inserted.vectors[row],
                old_vectors_by_chunk_id[chunk.chunk_id],
            )

    new_chunk_ids = {chunk.chunk_id for chunk in expected_new_chunks}
    new_rows = [
        row
        for row, chunk in enumerate(inserted.chunks)
        if chunk.chunk_id in new_chunk_ids
    ]
    assert len(new_rows) == len(expected_new_chunks)
    new_norms = np.linalg.norm(inserted.vectors[new_rows], axis=1)
    np.testing.assert_allclose(new_norms, np.ones(
        len(new_rows)), rtol=1e-5, atol=1e-6)

    assert inserted.manifest["embedding"] == before.manifest["embedding"]
    assert inserted.manifest["chunking"] == before.manifest["chunking"]
    assert inserted.manifest["sparse"] == before.manifest["sparse"]
    assert inserted.manifest["corpus"]["document_count"] == 3
    assert inserted.manifest["corpus"]["chunk_count"] == 5

    pointer = store._read_current_pointer(root)
    assert pointer["index_id"] == "idx-after-insert"
    assert pointer["previous_index_id"] == "idx-before-insert"
    assert (root / "versions" / "idx-before-insert").is_dir()
    assert (root / "versions" / "idx-after-insert").is_dir()

    reloaded = store.load_current_index_snapshot(root)
    assert [chunk.chunk_id for chunk in reloaded.chunks] == [
        chunk.chunk_id for chunk in inserted.chunks
    ]
    np.testing.assert_array_equal(reloaded.vectors, inserted.vectors)


def test_insert_of_empty_document_retains_document_without_embedding(tmp_path):
    root = tmp_path / "indexes"
    before = _seed_active_index(root)
    old_vectors_by_chunk_id = {
        chunk.chunk_id: before.vectors[row].copy()
        for row, chunk in enumerate(before.chunks)
    }
    calls: list[list[str]] = []
    embedder = _embedder(calls)

    empty_document = Document("doc-0-empty", "  \n\t", source="empty.txt")
    inserted = _insert(
        root,
        index_id="idx-with-empty-document",
        document=empty_document,
        embed_fn=embedder,
    )

    assert calls == []
    assert "doc-0-empty" in {document.doc_id for document in inserted.documents}
    assert "doc-0-empty" not in {chunk.doc_id for chunk in inserted.chunks}
    assert len(inserted.documents) == len(before.documents) + 1
    assert len(inserted.chunks) == len(before.chunks)

    for row, chunk in enumerate(inserted.chunks):
        np.testing.assert_array_equal(
            inserted.vectors[row], old_vectors_by_chunk_id[chunk.chunk_id]
        )


def test_duplicate_document_id_is_rejected_before_embedding_and_activation(tmp_path):
    root = tmp_path / "indexes"
    _seed_active_index(root)
    pointer_before = (root / "CURRENT.json").read_bytes()
    versions_before = sorted(
        path.name for path in (root / "versions").iterdir())
    calls: list[list[str]] = []

    with pytest.raises(ValueError, match="duplicate document ID"):
        _insert(
            root,
            index_id="idx-duplicate-insert",
            document=Document("doc-a", "different content"),
            embed_fn=_embedder(calls),
        )

    assert calls == []
    assert (root / "CURRENT.json").read_bytes() == pointer_before
    assert sorted(path.name for path in (
        root / "versions").iterdir()) == versions_before


@pytest.mark.parametrize(
    "embedder_kwargs",
    [
        {"model_id": "test/other-model"},
        {"revision": "b" * 40},
        {"config": {**DOCUMENT_EMBEDDING_CONFIG, "normalize_embeddings": False}},
    ],
    ids=["different-model-id", "different-immutable-revision",
         "different-config-same-dimension"],
)
def test_insert_rejects_incompatible_embedder_before_embedding_or_activation(
    tmp_path,
    embedder_kwargs,
):
    root = tmp_path / "indexes"
    _seed_active_index(root)
    pointer_before = (root / "CURRENT.json").read_bytes()
    calls: list[list[str]] = []

    with pytest.raises(ValueError, match="(incompatible|does not match|identity|revision)"):
        _insert(
            root,
            index_id="idx-incompatible-insert",
            document=Document("doc-b", "bravo charlie"),
            embed_fn=_embedder(calls, **embedder_kwargs),
        )

    assert calls == []
    assert (root / "CURRENT.json").read_bytes() == pointer_before
    assert not (root / "versions" / "idx-incompatible-insert").exists()


def test_insert_rejects_unverified_model_revision_before_embedding(tmp_path):
    root = tmp_path / "indexes"
    _seed_active_index(root)
    pointer_before = (root / "CURRENT.json").read_bytes()
    calls: list[list[str]] = []

    with pytest.raises(ValueError, match="(unverified|verified|identity)"):
        _insert(
            root,
            index_id="idx-unverified-insert",
            document=Document("doc-b", "bravo charlie"),
            embed_fn=_embedder(calls, revision_verified=False),
        )

    assert calls == []
    assert (root / "CURRENT.json").read_bytes() == pointer_before


def test_insert_rejects_new_vector_dimension_mismatch_without_activation(tmp_path):
    root = tmp_path / "indexes"
    _seed_active_index(root)
    pointer_before = (root / "CURRENT.json").read_bytes()
    calls: list[list[str]] = []

    with pytest.raises(ValueError, match="(dimension|shape|match)"):
        _insert(
            root,
            index_id="idx-wrong-dimension",
            document=Document("doc-b", "bravo charlie"),
            embed_fn=_embedder(calls, output_dimension=8),
        )

    assert calls == [["bravo charlie"]]
    assert (root / "CURRENT.json").read_bytes() == pointer_before
    assert not (root / "versions" / "idx-wrong-dimension").exists()


def test_embedding_failure_preserves_previous_active_snapshot(tmp_path):
    root = tmp_path / "indexes"
    before = _seed_active_index(root)
    pointer_before = (root / "CURRENT.json").read_bytes()
    calls: list[list[str]] = []

    def broken_embedder(texts: list[str]) -> np.ndarray:
        calls.append(list(texts))
        raise OSError("simulated model failure")

    broken_embedder.model_metadata = {
        "model_id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "revision_verified": True,
    }
    broken_embedder.embedding_config = dict(DOCUMENT_EMBEDDING_CONFIG)

    with pytest.raises(RuntimeError, match="embedding generation failed"):
        _insert(
            root,
            index_id="idx-failed-insert",
            document=Document("doc-b", "bravo charlie"),
            embed_fn=broken_embedder,
        )

    assert calls == [["bravo charlie"]]
    assert (root / "CURRENT.json").read_bytes() == pointer_before
    current = store.load_current_index_snapshot(root)
    assert current.manifest["index_id"] == before.manifest["index_id"]
    assert not (root / "versions" / "idx-failed-insert").exists()
    assert list((root / "staging").iterdir()) == []


def test_insert_rejects_non_document_input_without_embedding(tmp_path):
    root = tmp_path / "indexes"
    _seed_active_index(root)
    pointer_before = (root / "CURRENT.json").read_bytes()
    calls: list[list[str]] = []

    with pytest.raises(TypeError, match="Document"):
        _insert(
            root,
            index_id="idx-invalid-document",
            document={"doc_id": "doc-b", "text": "bravo charlie"},
            embed_fn=_embedder(calls),
        )

    assert calls == []
    assert (root / "CURRENT.json").read_bytes() == pointer_before
