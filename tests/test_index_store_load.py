"""Step 4.3: strict, integrity-checked index snapshot loading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hs.chunking import chunk_text
from hs.documents import Document
from hs.index_store import (
    CHUNKS_ARTIFACT,
    DOCUMENTS_ARTIFACT,
    VECTORS_ARTIFACT,
    artifact_metadata,
    build_index_snapshot,
    canonical_json,
    decode_jsonl,
    load_index_snapshot,
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
def built_snapshot(tmp_path):
    documents = [
        Document(
            "doc-1",
            "alpha beta gamma delta",
            source="manual.txt",
            title="Manual",
            metadata={"department": "engineering", "labels": ["search", "rag"]},
        ),
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
    embed_calls = []

    def embed(texts):
        embed_calls.append(list(texts))
        return np.asarray(
            [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
        )

    snapshot = tmp_path / "snapshot"
    build_index_snapshot(
        snapshot,
        index_id="idx-load-test-001",
        documents=documents,
        chunks=chunks,
        embed_fn=embed,
        embedding_model_id="test/fake-embedder",
        embedding_model_revision=MODEL_REVISION,
        embedding_config=EMBEDDING_CONFIG,
        sparse_config=SPARSE_CONFIG,
        created_at_utc=CREATED_AT,
    )
    return snapshot, embed_calls


def _read_manifest(snapshot: Path) -> dict:
    return json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(snapshot: Path, manifest: dict) -> None:
    payload = (canonical_json(manifest) + "\n").encode("utf-8")
    (snapshot / "manifest.json").write_bytes(payload)


def _refresh_artifact_manifest(snapshot: Path, name: str) -> dict:
    manifest = _read_manifest(snapshot)
    manifest["artifacts"][name].update(artifact_metadata(snapshot / name))
    _write_manifest(snapshot, manifest)
    return manifest


def test_load_round_trips_snapshot_without_reembedding(built_snapshot):
    snapshot, embed_calls = built_snapshot
    calls_before_load = list(embed_calls)

    loaded = load_index_snapshot(
        snapshot,
        expected_embedding={
            "model_id": "test/fake-embedder",
            "model_revision": MODEL_REVISION,
            "config": EMBEDDING_CONFIG,
        },
    )

    assert embed_calls == calls_before_load
    assert loaded.manifest["index_id"] == "idx-load-test-001"
    assert [item.doc_id for item in loaded.documents] == ["doc-1", "doc-empty"]
    assert [item.chunk_id for item in loaded.chunks] == ["doc-1#0", "doc-1#1"]
    assert loaded.vectors.dtype == np.float32
    assert loaded.vectors.shape == (2, 3)
    assert not loaded.vectors.flags.writeable
    np.testing.assert_allclose(loaded.vectors[0], [0.6, 0.8, 0.0])
    np.testing.assert_array_equal(loaded.vectors[1], [0.0, 0.0, 0.0])


def test_loader_rejects_embedding_model_id_mismatch(built_snapshot):
    snapshot, _ = built_snapshot
    with pytest.raises(ValueError, match="incompatible"):
        load_index_snapshot(
            snapshot,
            expected_embedding={
                "model_id": "different/model",
                "model_revision": MODEL_REVISION,
                "config": EMBEDDING_CONFIG,
            },
        )


def test_loader_rejects_embedding_revision_mismatch(built_snapshot):
    snapshot, _ = built_snapshot
    with pytest.raises(ValueError, match="incompatible"):
        load_index_snapshot(
            snapshot,
            expected_embedding={
                "model_id": "test/fake-embedder",
                "model_revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
                "config": EMBEDDING_CONFIG,
            },
        )


def test_loader_rejects_embedding_config_mismatch(built_snapshot):
    snapshot, _ = built_snapshot
    changed_config = {**EMBEDDING_CONFIG, "normalize_embeddings": False}
    with pytest.raises(ValueError, match="incompatible"):
        load_index_snapshot(
            snapshot,
            expected_embedding={
                "model_id": "test/fake-embedder",
                "model_revision": MODEL_REVISION,
                "config": changed_config,
            },
        )


@pytest.mark.parametrize(
    "artifact_name",
    [DOCUMENTS_ARTIFACT, CHUNKS_ARTIFACT, VECTORS_ARTIFACT],
)
def test_loader_rejects_artifact_checksum_mismatch(built_snapshot, artifact_name):
    snapshot, _ = built_snapshot
    path = snapshot / artifact_name
    payload = path.read_bytes()
    path.write_bytes(payload + b"tampered")

    with pytest.raises(ValueError, match="(size|SHA-256) mismatch"):
        load_index_snapshot(snapshot)


@pytest.mark.parametrize(
    "artifact_name",
    [DOCUMENTS_ARTIFACT, CHUNKS_ARTIFACT, VECTORS_ARTIFACT],
)
def test_loader_rejects_missing_artifact(built_snapshot, artifact_name):
    snapshot, _ = built_snapshot
    (snapshot / artifact_name).unlink()

    with pytest.raises(ValueError, match="layout mismatch"):
        load_index_snapshot(snapshot)


def test_loader_rejects_unexpected_snapshot_files(built_snapshot):
    snapshot, _ = built_snapshot
    (snapshot / "notes.txt").write_text("not an index artifact", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        load_index_snapshot(snapshot)


def test_loader_rejects_manifest_with_duplicate_keys(built_snapshot):
    snapshot, _ = built_snapshot
    manifest_path = snapshot / "manifest.json"
    raw = manifest_path.read_bytes()
    tampered = raw.replace(
        b'"format":', b'"format": "wrong", "format":', 1
    )
    assert tampered != raw
    # Keep the canonical LF bytes: Path.write_text() translates newlines to
    # CRLF on Windows, which would test line-ending rejection instead.
    manifest_path.write_bytes(tampered)

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_index_snapshot(snapshot)


def test_loader_rejects_manifest_with_crlf_line_endings(built_snapshot):
    snapshot, _ = built_snapshot
    manifest_path = snapshot / "manifest.json"
    raw = manifest_path.read_bytes()
    assert raw.endswith(b"\n") and not raw.endswith(b"\r\n")
    manifest_path.write_bytes(raw[:-1] + b"\r\n")

    with pytest.raises(ValueError, match="must use LF, not CRLF"):
        load_index_snapshot(snapshot)


def test_loader_rejects_manifest_unknown_schema(built_snapshot):
    snapshot, _ = built_snapshot
    manifest = _read_manifest(snapshot)
    manifest["schema_version"] = 99
    _write_manifest(snapshot, manifest)

    with pytest.raises(ValueError, match="schema version"):
        load_index_snapshot(snapshot)


def test_loader_rejects_document_record_fingerprint_mismatch(built_snapshot):
    snapshot, _ = built_snapshot
    path = snapshot / DOCUMENTS_ARTIFACT
    records = decode_jsonl(path.read_bytes())
    records[0]["text"] = "tampered source text"
    path.write_bytes(b"".join((canonical_json(row) + "\n").encode() for row in records))
    _refresh_artifact_manifest(snapshot, DOCUMENTS_ARTIFACT)

    with pytest.raises(ValueError, match="content_sha256 mismatch"):
        load_index_snapshot(snapshot)


def test_loader_rejects_chunk_record_fingerprint_mismatch(built_snapshot):
    snapshot, _ = built_snapshot
    path = snapshot / CHUNKS_ARTIFACT
    records = decode_jsonl(path.read_bytes())
    records[0]["text"] = "tampered chunk text"
    path.write_bytes(b"".join((canonical_json(row) + "\n").encode() for row in records))
    _refresh_artifact_manifest(snapshot, CHUNKS_ARTIFACT)

    with pytest.raises(ValueError, match="content_sha256 mismatch"):
        load_index_snapshot(snapshot)


def test_loader_rejects_corpus_fingerprint_mismatch(built_snapshot):
    snapshot, _ = built_snapshot
    manifest = _read_manifest(snapshot)
    manifest["corpus"]["fingerprint_sha256"] = "0" * 64
    _write_manifest(snapshot, manifest)

    with pytest.raises(ValueError, match="corpus fingerprint"):
        load_index_snapshot(snapshot)


def test_loader_rejects_non_float32_vectors_even_with_updated_checksum(built_snapshot):
    snapshot, _ = built_snapshot
    vectors = np.load(snapshot / VECTORS_ARTIFACT, allow_pickle=False).astype(np.float64)
    with (snapshot / VECTORS_ARTIFACT).open("wb") as handle:
        np.save(handle, vectors, allow_pickle=False)
    manifest = _refresh_artifact_manifest(snapshot, VECTORS_ARTIFACT)
    manifest["artifacts"][VECTORS_ARTIFACT]["dtype"] = "float32"
    _write_manifest(snapshot, manifest)

    with pytest.raises(ValueError, match="dtype must be float32"):
        load_index_snapshot(snapshot)


def test_loader_rejects_non_normalized_vectors_even_with_updated_checksum(built_snapshot):
    snapshot, _ = built_snapshot
    vectors = np.load(snapshot / VECTORS_ARTIFACT, allow_pickle=False)
    vectors[0] = [3.0, 4.0, 1.0]
    with (snapshot / VECTORS_ARTIFACT).open("wb") as handle:
        np.save(handle, vectors, allow_pickle=False)
    _refresh_artifact_manifest(snapshot, VECTORS_ARTIFACT)

    with pytest.raises(ValueError, match="non-normalized"):
        load_index_snapshot(snapshot)


def test_loader_rejects_non_finite_vectors_even_with_updated_checksum(built_snapshot):
    snapshot, _ = built_snapshot
    vectors = np.load(snapshot / VECTORS_ARTIFACT, allow_pickle=False)
    vectors[0, 0] = np.nan
    with (snapshot / VECTORS_ARTIFACT).open("wb") as handle:
        np.save(handle, vectors, allow_pickle=False)
    _refresh_artifact_manifest(snapshot, VECTORS_ARTIFACT)

    with pytest.raises(ValueError, match="non-finite"):
        load_index_snapshot(snapshot)


def test_loader_rejects_vector_shape_inconsistent_with_manifest(built_snapshot):
    snapshot, _ = built_snapshot
    vectors = np.load(snapshot / VECTORS_ARTIFACT, allow_pickle=False)
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    with (snapshot / VECTORS_ARTIFACT).open("wb") as handle:
        np.save(handle, vectors, allow_pickle=False)
    manifest = _read_manifest(snapshot)
    manifest["artifacts"][VECTORS_ARTIFACT].update(artifact_metadata(snapshot / VECTORS_ARTIFACT))
    _write_manifest(snapshot, manifest)

    with pytest.raises(ValueError, match="shape"):
        load_index_snapshot(snapshot)


def test_loader_rejects_corrupt_npy_after_checksum_is_updated(built_snapshot):
    snapshot, _ = built_snapshot
    (snapshot / VECTORS_ARTIFACT).write_bytes(b"not a numpy file")
    _refresh_artifact_manifest(snapshot, VECTORS_ARTIFACT)

    with pytest.raises(ValueError, match="corrupt or invalid"):
        load_index_snapshot(snapshot)


def test_loader_requires_snapshot_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="directory"):
        load_index_snapshot(tmp_path / "does-not-exist")


def test_decode_jsonl_rejects_duplicate_object_keys():
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        decode_jsonl(b'{"doc_id":"first","doc_id":"second"}\n')


def test_decode_jsonl_rejects_non_standard_nan():
    with pytest.raises(ValueError, match="non-standard JSON numeric constant"):
        decode_jsonl(b'{"score":NaN}\n')


def test_loaded_vector_matrix_is_read_only(built_snapshot):
    snapshot, _ = built_snapshot
    loaded = load_index_snapshot(snapshot)
    with pytest.raises(ValueError):
        loaded.vectors[0, 0] = 9.0


def test_loader_rejects_duplicate_keys_inside_document_jsonl(built_snapshot):
    snapshot, _ = built_snapshot
    path = snapshot / DOCUMENTS_ARTIFACT
    raw = path.read_text(encoding="utf-8")
    path.write_text(
        raw.replace('"doc_id":', '"doc_id": "shadow", "doc_id":', 1),
        encoding="utf-8",
        newline="",
    )
    _refresh_artifact_manifest(snapshot, DOCUMENTS_ARTIFACT)

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_index_snapshot(snapshot)


def test_decode_jsonl_rejects_crlf_line_endings():
    with pytest.raises(ValueError, match="LF, not CRLF"):
        decode_jsonl(b'{"doc_id":"one"}\r\n')


@pytest.mark.parametrize(
    "payload",
    [
        b'{ "doc_id":"one"}\n',
        b'{"text":"x","doc_id":"one"}\n',
    ],
)
def test_decode_jsonl_rejects_non_canonical_json(payload):
    with pytest.raises(ValueError, match="not canonically encoded"):
        decode_jsonl(payload)
