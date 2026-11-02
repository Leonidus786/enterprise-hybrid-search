"""Step 4.1: persistent index serialization and manifest contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hs.chunking import Chunk
from hs.documents import Document
from hs.index_store import (
    CHUNKS_ARTIFACT,
    DOCUMENTS_ARTIFACT,
    INDEX_FORMAT,
    INDEX_SCHEMA_VERSION,
    VECTORS_ARTIFACT,
    artifact_metadata,
    build_manifest,
    corpus_fingerprint,
    chunk_from_record,
    chunk_to_record,
    decode_jsonl,
    document_from_record,
    document_to_record,
    encode_jsonl,
    validate_manifest,
)


@pytest.fixture
def sample_artifacts():
    return {
        DOCUMENTS_ARTIFACT: {
            "sha256": "a" * 64, "size_bytes": 200, "record_count": 2,
        },
        CHUNKS_ARTIFACT: {
            "sha256": "b" * 64, "size_bytes": 400, "record_count": 3,
        },
        VECTORS_ARTIFACT: {
            "sha256": "c" * 64, "size_bytes": 1200,
            "shape": [3, 100], "dtype": "float32",
        },
    }


def build_valid_manifest(artifacts):
    return build_manifest(
        index_id="idx-2026-11-01-a1",
        embedding_model_id="BAAI/bge-small-en-v1.5",
        embedding_model_revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        embedding_config={"normalize_embeddings": True, "batch_size": 64},
        chunking_config={"algorithm": "word_window", "version": 1, "size": 200, "overlap": 40},
        sparse_config={"algorithm": "bm25", "version": 1, "k1": 1.5, "b": 0.75},
        document_count=2,
        chunk_count=3,
        vector_dimension=100,
        corpus_fingerprint_sha256="d" * 64,
        artifacts=artifacts,
        created_at_utc="2026-11-01T12:00:00+00:00",
    )


def test_document_record_round_trip_preserves_unicode_and_identity():
    original = Document(
        "policy-हिन्दी", "नमस्ते दुनिया — retrieval 搜索 🔎",
        source="स्रोत/文件.txt", title="भाषा दस्तावेज़",
        metadata={"labels": ["हिन्दी", "中文", "🔎"], "version": 3},
    )
    restored = document_from_record(document_to_record(original))
    assert restored == original
    assert restored.content_sha256 == original.content_sha256
    assert restored.revision_sha256 == original.revision_sha256


def test_document_deserializer_rejects_tampered_content_fingerprint():
    record = document_to_record(Document("d1", "original"))
    record["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content_sha256 mismatch"):
        document_from_record(record)


def test_document_deserializer_rejects_unknown_or_non_integer_record_schema():
    record = document_to_record(Document("d1", "body"))
    record["record_schema_version"] = 999
    with pytest.raises(ValueError, match="schema version"):
        document_from_record(record)
    record["record_schema_version"] = 1.0
    with pytest.raises(ValueError, match="schema version"):
        document_from_record(record)


def test_chunk_record_round_trip_preserves_revision_and_chunking_config():
    original = Chunk(
        "d1#0", "d1", "alpha beta gamma", source="manual.pdf", title="Manual",
        metadata={"department": "engineering"},
        document_revision_sha256="e" * 64,
        chunking_config={"algorithm": "word_window", "version": 1, "size": 3, "overlap": 0},
    )
    restored = chunk_from_record(chunk_to_record(original))
    assert restored == original
    assert restored.source == original.source
    assert restored.metadata == original.metadata
    assert restored.document_revision_sha256 == original.document_revision_sha256
    assert restored.content_sha256 == original.content_sha256
    assert restored.revision_sha256 == original.revision_sha256


def test_chunk_deserializer_rejects_tampered_revision_fingerprint():
    record = chunk_to_record(Chunk("d1#0", "d1", "body"))
    record["revision_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="revision_sha256 mismatch"):
        chunk_from_record(record)


def test_chunk_deserializer_rejects_boolean_record_schema_version():
    record = chunk_to_record(Chunk("d1#0", "d1", "body"))
    record["record_schema_version"] = True
    with pytest.raises(ValueError, match="schema version"):
        chunk_from_record(record)


def test_jsonl_encoding_is_canonical_utf8_and_ends_with_newline():
    records = [{"z": "हिन्दी\u2028line-break-safe", "a": 1}, {"b": True, "a": None}]
    payload = encode_jsonl(records)
    assert payload.endswith(b"\n")
    assert "हिन्दी".encode("utf-8") in payload
    assert payload == encode_jsonl(records)
    assert decode_jsonl(payload) == records
    assert payload.splitlines()[0].startswith(b'{"a":1,"z":')


def test_empty_jsonl_is_empty_bytes_and_empty_records():
    assert encode_jsonl([]) == b""
    assert decode_jsonl(b"") == []


@pytest.mark.parametrize("payload, message", [
    (b'{"x":1}', "end with a newline"),
    (b'{bad json}\n', "invalid JSONL"),
    (b'[]\n', "must be an object"),
    (b'{"x":1}\n\n', "blank line"),
    (b'\xff\n', "valid UTF-8"),
])
def test_jsonl_decoder_rejects_malformed_payloads(payload, message):
    with pytest.raises(ValueError, match=message):
        decode_jsonl(payload)


def test_jsonl_encoder_rejects_non_mapping_and_non_json_rows():
    with pytest.raises(TypeError, match="mapping"):
        encode_jsonl([["not", "an", "object"]])
    with pytest.raises(ValueError, match="JSON-compatible"):
        encode_jsonl([{"score": float("nan")}])


def test_corpus_fingerprint_is_stable_and_sensitive_to_order_and_revisions():
    doc_a = Document("a", "alpha")
    doc_b = Document("b", "beta")
    chunk_a = Chunk("a#0", "a", "alpha")
    chunk_b = Chunk("b#0", "b", "beta")
    first = corpus_fingerprint([doc_a, doc_b], [chunk_a, chunk_b])
    assert first == corpus_fingerprint([doc_a, doc_b], [chunk_a, chunk_b])
    assert first != corpus_fingerprint([doc_b, doc_a], [chunk_a, chunk_b])
    changed = Document("a", "alpha", metadata={"revision": 2})
    assert first != corpus_fingerprint([changed, doc_b], [chunk_a, chunk_b])


def test_corpus_fingerprint_rejects_wrong_record_types():
    with pytest.raises(TypeError, match="Document"):
        corpus_fingerprint(["not a document"], [])
    with pytest.raises(TypeError, match="Chunk"):
        corpus_fingerprint([], ["not a chunk"])


def test_artifact_metadata_computes_streamed_sha256_and_size(tmp_path: Path):
    artifact = tmp_path / "records.jsonl"
    payload = "दस्तावेज़\n".encode("utf-8")
    artifact.write_bytes(payload)
    assert artifact_metadata(artifact) == {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def test_manifest_contains_versioned_storage_contract(sample_artifacts):
    manifest = build_valid_manifest(sample_artifacts)
    assert manifest["format"] == INDEX_FORMAT
    assert manifest["schema_version"] == INDEX_SCHEMA_VERSION == 1
    assert manifest["corpus"] == {
        "document_count": 2, "chunk_count": 3, "fingerprint_sha256": "d" * 64,
    }
    assert manifest["vectors"] == {
        "rows": 3, "dimension": 100, "dtype": "float32", "normalized": True,
    }
    assert set(manifest["artifacts"]) == {
        DOCUMENTS_ARTIFACT, CHUNKS_ARTIFACT, VECTORS_ARTIFACT,
    }


def test_manifest_round_trip_validation_returns_detached_json_safe_copy(sample_artifacts):
    manifest = build_valid_manifest(sample_artifacts)
    validated = validate_manifest(manifest)
    assert validated == manifest
    validated["embedding"]["config"]["batch_size"] = 8
    assert manifest["embedding"]["config"]["batch_size"] == 64


def test_manifest_rejects_model_revision_or_config_mismatch(sample_artifacts):
    manifest = build_valid_manifest(sample_artifacts)
    expected = {
        "model_id": "BAAI/bge-small-en-v1.5",
        "model_revision": "0" * 40,
        "config": {"normalize_embeddings": True, "batch_size": 64},
    }
    with pytest.raises(ValueError, match="incompatible"):
        validate_manifest(manifest, expected_embedding=expected)


def test_manifest_accepts_matching_embedding_identity(sample_artifacts):
    manifest = build_valid_manifest(sample_artifacts)
    expected = {
        "model_id": "BAAI/bge-small-en-v1.5",
        "model_revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "config": {"batch_size": 64, "normalize_embeddings": True},
    }
    assert validate_manifest(manifest, expected_embedding=expected) == manifest


def test_manifest_rejects_unknown_or_non_integer_schema_version(sample_artifacts):
    manifest = build_valid_manifest(sample_artifacts)
    manifest["schema_version"] = 2
    with pytest.raises(ValueError, match="schema version"):
        validate_manifest(manifest)
    manifest["schema_version"] = 1.0
    with pytest.raises(ValueError, match="schema version"):
        validate_manifest(manifest)


def test_manifest_rejects_missing_or_extra_artifact(sample_artifacts):
    missing = dict(sample_artifacts)
    missing.pop(VECTORS_ARTIFACT)
    with pytest.raises(ValueError, match="exactly the required artifacts"):
        build_valid_manifest(missing)
    extra = dict(sample_artifacts, **{"unexpected.bin": {"sha256": "f" * 64, "size_bytes": 1}})
    with pytest.raises(ValueError, match="exactly the required artifacts"):
        build_valid_manifest(extra)


def test_manifest_rejects_artifact_record_count_mismatch(sample_artifacts):
    sample_artifacts[CHUNKS_ARTIFACT]["record_count"] = 4
    with pytest.raises(ValueError, match="record_count does not match"):
        build_valid_manifest(sample_artifacts)


def test_manifest_rejects_vector_shape_mismatch(sample_artifacts):
    sample_artifacts[VECTORS_ARTIFACT]["shape"] = [3, 99]
    with pytest.raises(ValueError, match="shape does not match"):
        build_valid_manifest(sample_artifacts)


def test_manifest_rejects_mutable_or_invalid_model_revision(sample_artifacts):
    manifest = build_valid_manifest(sample_artifacts)
    manifest["embedding"]["model_revision"] = "main"
    with pytest.raises(ValueError, match="immutable hexadecimal commit"):
        validate_manifest(manifest)


def test_manifest_rejects_corrupt_checksum_and_non_float32_dtype(sample_artifacts):
    sample_artifacts[DOCUMENTS_ARTIFACT]["sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="SHA-256"):
        build_valid_manifest(sample_artifacts)

    sample_artifacts[DOCUMENTS_ARTIFACT]["sha256"] = "a" * 64
    sample_artifacts[VECTORS_ARTIFACT]["dtype"] = "float64"
    with pytest.raises(ValueError, match="dtype must be float32"):
        build_valid_manifest(sample_artifacts)


@pytest.mark.parametrize("index_id", ["../escape", "..", "", "bad/name", "a" * 129, "CON", "NUL.txt"])
def test_manifest_rejects_unsafe_index_ids(sample_artifacts, index_id):
    with pytest.raises(ValueError, match="index_id"):
        manifest = build_valid_manifest(sample_artifacts)
        manifest["index_id"] = index_id
        validate_manifest(manifest)


def test_manifest_rejects_boolean_counts_and_unzoned_timestamps(sample_artifacts):
    manifest = build_valid_manifest(sample_artifacts)
    manifest["corpus"]["document_count"] = True
    with pytest.raises(ValueError, match="document_count"):
        validate_manifest(manifest)

    manifest = build_valid_manifest(sample_artifacts)
    manifest["created_at_utc"] = "2026-11-01T12:00:00"
    with pytest.raises(ValueError, match="UTC timezone"):
        validate_manifest(manifest)
