"""Step 4.2: reproducible persistent index snapshot construction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import hs.index_store as store
from hs.chunking import chunk_text
from hs.documents import Document
from hs.index_store import (
    CHUNKS_ARTIFACT,
    DOCUMENTS_ARTIFACT,
    VECTORS_ARTIFACT,
    artifact_metadata,
    build_index_snapshot,
    chunk_from_record,
    decode_jsonl,
    document_from_record,
)

MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
EMBEDDING_CONFIG = {
    "adapter": "test-deterministic-embedder-v1",
    "normalize_embeddings": True,
    "dimension": 3,
}
SPARSE_CONFIG = {
    "algorithm": "bm25",
    "version": 1,
    "k1": 1.5,
    "b": 0.75,
}
CREATED_AT = "2026-11-01T12:00:00+00:00"


@pytest.fixture
def corpus():
    documents = [
        Document(
            "doc-1",
            "alpha beta gamma delta",
            source="manual.txt",
            title="Manual",
            metadata={"department": "engineering", "labels": ["search", "rag"]},
        ),
        # Empty documents remain in documents.jsonl, although they have no chunks.
        Document("doc-empty", "   ", source="empty.txt"),
    ]
    chunks = chunk_text(
        documents[0].doc_id,
        documents[0].text,
        size=2,
        overlap=0,
        source=documents[0].source,
        title=documents[0].title,
        metadata=documents[0].metadata,
        document_revision_sha256=documents[0].revision_sha256,
    )
    return documents, chunks


def make_embedder(rows=None):
    calls = []

    def embed(texts):
        calls.append(list(texts))
        if rows is not None:
            return np.asarray(rows)
        return np.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)

    return embed, calls


def build(path: Path, documents, chunks, embed_fn, **overrides):
    args = {
        "index_id": "idx-build-test-001",
        "documents": documents,
        "chunks": chunks,
        "embed_fn": embed_fn,
        "embedding_model_id": "test/fake-embedder",
        "embedding_model_revision": MODEL_REVISION,
        "embedding_config": EMBEDDING_CONFIG,
        "sparse_config": SPARSE_CONFIG,
        "created_at_utc": CREATED_AT,
    }
    args.update(overrides)
    return build_index_snapshot(path, **args)


def test_build_writes_snapshot_artifacts_and_manifest(tmp_path, corpus):
    documents, chunks = corpus
    embed_fn, calls = make_embedder()
    snapshot = tmp_path / "indexes" / "idx-build-test-001"

    manifest = build(snapshot, documents, chunks, embed_fn)

    assert snapshot.is_dir()
    assert set(path.name for path in snapshot.iterdir()) == {
        "manifest.json",
        DOCUMENTS_ARTIFACT,
        CHUNKS_ARTIFACT,
        VECTORS_ARTIFACT,
    }
    assert calls == [[chunk.text for chunk in chunks]]
    assert manifest["corpus"]["document_count"] == 2
    assert manifest["corpus"]["chunk_count"] == 2
    assert manifest["vectors"] == {
        "rows": 2,
        "dimension": 3,
        "dtype": "float32",
        "normalized": True,
    }

    stored_documents = decode_jsonl((snapshot / DOCUMENTS_ARTIFACT).read_bytes())
    stored_chunks = decode_jsonl((snapshot / CHUNKS_ARTIFACT).read_bytes())
    assert len(stored_documents) == 2
    assert len(stored_chunks) == 2
    assert document_from_record(stored_documents[1]).text == "   "
    assert [chunk_from_record(row).chunk_id for row in stored_chunks] == [
        chunk.chunk_id for chunk in chunks
    ]

    manifest_on_disk = json.loads((snapshot / "manifest.json").read_text("utf-8"))
    assert manifest_on_disk == manifest
    assert (snapshot / "manifest.json").read_bytes().endswith(b"\n")


def test_vector_rows_are_float32_normalized_and_aligned_to_chunk_order(
    tmp_path, corpus
):
    documents, chunks = corpus
    embed_fn, calls = make_embedder()
    snapshot = tmp_path / "snapshot"

    build(snapshot, documents, chunks, embed_fn)

    vectors = np.load(snapshot / VECTORS_ARTIFACT, allow_pickle=False)
    assert vectors.dtype == np.float32
    assert vectors.flags.c_contiguous
    np.testing.assert_allclose(vectors[0], [0.6, 0.8, 0.0])
    # Match the existing DenseIndex policy: zero vectors remain zero.
    np.testing.assert_array_equal(vectors[1], [0.0, 0.0, 0.0])
    assert len(calls[0]) == len(chunks)


def test_artifact_checksums_and_sizes_match_written_bytes(tmp_path, corpus):
    documents, chunks = corpus
    embed_fn, _ = make_embedder()
    snapshot = tmp_path / "snapshot"

    manifest = build(snapshot, documents, chunks, embed_fn)

    for name in (DOCUMENTS_ARTIFACT, CHUNKS_ARTIFACT, VECTORS_ARTIFACT):
        path = snapshot / name
        expected = artifact_metadata(path)
        assert manifest["artifacts"][name]["sha256"] == expected["sha256"]
        assert manifest["artifacts"][name]["size_bytes"] == expected["size_bytes"]

    vectors_bytes = (snapshot / VECTORS_ARTIFACT).read_bytes()
    assert hashlib.sha256(vectors_bytes).hexdigest() == manifest["artifacts"][
        VECTORS_ARTIFACT
    ]["sha256"]


def test_manifest_records_chunking_and_embedding_configs(tmp_path, corpus):
    documents, chunks = corpus
    embed_fn, _ = make_embedder()

    manifest = build(tmp_path / "snapshot", documents, chunks, embed_fn)

    assert manifest["chunking"] == {
        "algorithm": "word_window",
        "version": 1,
        "size": 2,
        "overlap": 0,
    }
    assert manifest["embedding"] == {
        "model_id": "test/fake-embedder",
        "model_revision": MODEL_REVISION,
        "config": EMBEDDING_CONFIG,
    }
    assert manifest["sparse"] == SPARSE_CONFIG


def test_corpus_fingerprint_and_chunk_order_are_deterministic(tmp_path, corpus):
    documents, chunks = corpus
    first_embed, _ = make_embedder()
    second_embed, _ = make_embedder()

    first = build(
        tmp_path / "first", documents, chunks, first_embed,
        index_id="idx-first",
    )
    second = build(
        tmp_path / "second", documents, chunks, second_embed,
        index_id="idx-second",
    )

    assert first["corpus"]["fingerprint_sha256"] == second[
        "corpus"
    ]["fingerprint_sha256"]
    first_vectors = np.load(tmp_path / "first" / VECTORS_ARTIFACT, allow_pickle=False)
    second_vectors = np.load(tmp_path / "second" / VECTORS_ARTIFACT, allow_pickle=False)
    np.testing.assert_array_equal(first_vectors, second_vectors)


def test_destination_must_not_exist_and_embedder_is_not_called(tmp_path, corpus):
    documents, chunks = corpus
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    embed_fn, calls = make_embedder()

    with pytest.raises(FileExistsError, match="already exists"):
        build(snapshot, documents, chunks, embed_fn)

    assert calls == []
    assert list(snapshot.iterdir()) == []


@pytest.mark.parametrize("documents,chunks", [([], []), ([Document("d", "text")], [])])
def test_empty_corpus_or_no_searchable_chunks_is_rejected(
    tmp_path, documents, chunks
):
    embed_fn, calls = make_embedder()

    with pytest.raises(
        ValueError, match="empty document set|without searchable chunks"
    ):
        build(tmp_path / "snapshot", documents, chunks, embed_fn)

    assert calls == []
    assert not (tmp_path / "snapshot").exists()


def test_duplicate_document_ids_are_rejected(tmp_path, corpus):
    documents, chunks = corpus
    duplicate_documents = [documents[0], documents[0]]
    embed_fn, calls = make_embedder()

    with pytest.raises(ValueError, match="duplicate document ID"):
        build(tmp_path / "snapshot", duplicate_documents, chunks, embed_fn)

    assert calls == []


def test_duplicate_chunk_ids_are_rejected(tmp_path, corpus):
    documents, chunks = corpus
    duplicate_chunks = [chunks[0], chunks[0]]
    embed_fn, calls = make_embedder()

    with pytest.raises(ValueError, match="duplicate chunk ID"):
        build(tmp_path / "snapshot", documents, duplicate_chunks, embed_fn)

    assert calls == []


def test_chunk_with_unknown_document_is_rejected(tmp_path, corpus):
    documents, chunks = corpus
    bad = chunk_text(
        "missing-doc",
        "alpha beta",
        size=2,
        overlap=0,
        document_revision_sha256="a" * 64,
    )
    embed_fn, calls = make_embedder()

    with pytest.raises(ValueError, match="unknown document"):
        build(tmp_path / "snapshot", documents, bad, embed_fn)

    assert calls == []


def test_stale_document_revision_is_rejected(tmp_path, corpus):
    documents, chunks = corpus
    stale = chunk_text(
        documents[0].doc_id,
        documents[0].text,
        size=2,
        overlap=0,
        source=documents[0].source,
        title=documents[0].title,
        metadata=documents[0].metadata,
        document_revision_sha256="a" * 64,
    )
    embed_fn, calls = make_embedder()

    with pytest.raises(ValueError, match="parent document revision"):
        build(tmp_path / "snapshot", documents, stale, embed_fn)

    assert calls == []


def test_missing_or_reordered_chunks_are_rejected(tmp_path, corpus):
    documents, chunks = corpus
    embed_fn, calls = make_embedder()

    with pytest.raises(ValueError, match="do not match the ordered documents"):
        build(tmp_path / "snapshot", documents, chunks[:1], embed_fn)

    with pytest.raises(ValueError, match="do not match the ordered documents"):
        build(tmp_path / "snapshot-2", documents, list(reversed(chunks)), embed_fn)

    assert calls == []


def test_inconsistent_chunking_configs_are_rejected(tmp_path, corpus):
    documents, chunks = corpus
    first = chunks[0]
    from dataclasses import replace

    changed = replace(
        chunks[1],
        chunking_config={"algorithm": "whole_document", "version": 1},
    )
    embed_fn, calls = make_embedder()

    with pytest.raises(ValueError, match="same chunking_config"):
        build(tmp_path / "snapshot", documents, [first, changed], embed_fn)

    assert calls == []


def test_explicit_chunking_config_must_match_chunks(tmp_path, corpus):
    documents, chunks = corpus
    embed_fn, calls = make_embedder()

    with pytest.raises(ValueError, match="does not match"):
        build(
            tmp_path / "snapshot",
            documents,
            chunks,
            embed_fn,
            chunking_config={"algorithm": "whole_document", "version": 1},
        )

    assert calls == []


@pytest.mark.parametrize(
    "embedding_model_revision",
    ["main", "latest", "not-a-commit", "a" * 39, "g" * 40],
)
def test_mutable_or_invalid_model_revision_is_rejected(
    tmp_path, corpus, embedding_model_revision
):
    documents, chunks = corpus
    embed_fn, calls = make_embedder()

    with pytest.raises(ValueError, match="immutable hexadecimal commit"):
        build(
            tmp_path / "snapshot",
            documents,
            chunks,
            embed_fn,
            embedding_model_revision=embedding_model_revision,
        )

    assert calls == []


@pytest.mark.parametrize("bad_config", [{}, None, {"normalize": float("nan")}])
def test_invalid_embedding_config_is_rejected_before_embedding(
    tmp_path, corpus, bad_config
):
    documents, chunks = corpus
    embed_fn, calls = make_embedder()

    with pytest.raises((ValueError, TypeError)):
        build(
            tmp_path / "snapshot", documents, chunks, embed_fn,
            embedding_config=bad_config,
        )

    assert calls == []


@pytest.mark.parametrize("bad_config", [{}, None])
def test_invalid_sparse_config_is_rejected_before_embedding(
    tmp_path, corpus, bad_config
):
    documents, chunks = corpus
    embed_fn, calls = make_embedder()

    with pytest.raises((ValueError, TypeError)):
        build(
            tmp_path / "snapshot", documents, chunks, embed_fn,
            sparse_config=bad_config,
        )

    assert calls == []


@pytest.mark.parametrize(
    "bad_vectors, expected_message",
    [
        ([[1.0, 2.0]], "shape"),
        ([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "shape"),
        ([[1.0, 2.0, float("nan")], [1.0, 2.0, 3.0]], "finite"),
        ([[1.0, 2.0, float("inf")], [1.0, 2.0, 3.0]], "finite"),
        (["not numeric", "still not numeric"], "numeric 2D"),
        ([[], []], "shape"),
    ],
)
def test_invalid_embedding_outputs_are_rejected_and_not_persisted(
    tmp_path, corpus, bad_vectors, expected_message
):
    documents, chunks = corpus
    embed_fn, calls = make_embedder(rows=bad_vectors)
    snapshot = tmp_path / "snapshot"

    with pytest.raises(ValueError, match=expected_message):
        build(snapshot, documents, chunks, embed_fn)

    assert len(calls) == 1
    assert not snapshot.exists()


def test_embedder_exception_is_reported_and_does_not_create_snapshot(
    tmp_path, corpus
):
    documents, chunks = corpus

    def broken_embedder(texts):
        raise OSError("model unavailable")

    with pytest.raises(RuntimeError, match="embedding generation failed") as exc:
        build(tmp_path / "snapshot", documents, chunks, broken_embedder)

    assert isinstance(exc.value.__cause__, OSError)
    assert not (tmp_path / "snapshot").exists()


def test_failed_artifact_write_cleans_up_incomplete_snapshot(
    tmp_path, corpus, monkeypatch
):
    documents, chunks = corpus
    embed_fn, _ = make_embedder()
    snapshot = tmp_path / "snapshot"

    def fail_artifact_metadata(path):
        raise OSError("simulated disk read failure")

    monkeypatch.setattr(store, "artifact_metadata", fail_artifact_metadata)
    with pytest.raises(OSError, match="simulated disk read failure"):
        build(snapshot, documents, chunks, embed_fn)

    assert not snapshot.exists()


def test_builder_rejects_unsupported_index_id_before_embedding(tmp_path, corpus):
    documents, chunks = corpus
    embed_fn, calls = make_embedder()

    with pytest.raises(ValueError, match="unsafe path characters"):
        build(
            tmp_path / "snapshot", documents, chunks, embed_fn,
            index_id="../escape",
        )

    assert calls == []


def test_whole_document_chunking_configuration_is_supported(tmp_path):
    document = Document("doc-whole", "one two three")
    chunks = chunk_text(
        document.doc_id,
        document.text,
        size=None,
        source=document.source,
        title=document.title,
        metadata=document.metadata,
        document_revision_sha256=document.revision_sha256,
    )
    embed_fn, _ = make_embedder(rows=[[1.0, 0.0, 0.0]])

    manifest = build(
        tmp_path / "snapshot", [document], chunks, embed_fn,
        index_id="idx-whole-document",
    )

    assert manifest["chunking"] == {
        "algorithm": "whole_document",
        "version": 1,
    }


@pytest.mark.parametrize(
    "row, expected_message",
    [
        ([3.0e38, 3.0e38, 3.0e38], "norms overflow"),
        ([1.0e-45, 0.0, 0.0], "norm underflowed"),
    ],
)
def test_embedding_norm_overflow_and_underflow_are_rejected(
    tmp_path, corpus, row, expected_message
):
    documents, chunks = corpus
    embed_fn, _ = make_embedder(rows=[row, [1.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match=expected_message):
        build(tmp_path / "snapshot", documents, chunks, embed_fn)

    assert not (tmp_path / "snapshot").exists()


def test_non_callable_embedder_is_rejected_before_any_write(tmp_path, corpus):
    documents, chunks = corpus

    with pytest.raises(TypeError, match="embed_fn must be callable"):
        build(tmp_path / "snapshot", documents, chunks, None)

    assert not (tmp_path / "snapshot").exists()


def test_chunk_metadata_must_match_source_document(tmp_path, corpus):
    documents, chunks = corpus
    from dataclasses import replace

    mismatched = [
        chunks[0],
        replace(chunks[1], metadata={"department": "finance"}),
    ]
    embed_fn, calls = make_embedder()

    with pytest.raises(ValueError, match="do not match the ordered documents"):
        build(tmp_path / "snapshot", documents, mismatched, embed_fn)

    assert calls == []


def test_unsupported_chunking_algorithm_is_rejected(tmp_path, corpus):
    documents, chunks = corpus
    from dataclasses import replace

    unsupported = [
        replace(
            chunk,
            chunking_config={"algorithm": "sentence_window", "version": 1},
        )
        for chunk in chunks
    ]
    embed_fn, calls = make_embedder()

    with pytest.raises(ValueError, match="unsupported chunking"):
        build(tmp_path / "snapshot", documents, unsupported, embed_fn)

    assert calls == []
