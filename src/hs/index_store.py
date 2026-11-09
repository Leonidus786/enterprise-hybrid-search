"""Build and validate portable, versioned hybrid-search index snapshots.

This module owns serialization contracts, snapshot construction, artifact
integrity metadata, and manifest validation. Snapshot loading and atomic
activation are implemented in later index-store steps.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

from .chunking import Chunk, chunk_text
from .documents import Document, normalize_metadata
from .identity import canonical_json, sha256_json

INDEX_FORMAT = "enterprise-hybrid-search-index"
INDEX_SCHEMA_VERSION = 1
RECORD_SCHEMA_VERSION = 1
DOCUMENTS_ARTIFACT = "documents.jsonl"
CHUNKS_ARTIFACT = "chunks.jsonl"
VECTORS_ARTIFACT = "vectors.npy"
REQUIRED_ARTIFACTS = (
    DOCUMENTS_ARTIFACT,
    CHUNKS_ARTIFACT,
    VECTORS_ARTIFACT,
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INDEX_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
_OPERATION_SCHEMA_VERSION = 2
_OPERATION_TYPES = frozenset({"rebuild", "insert", "update", "delete"})


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must contain valid Unicode") from exc
    return value.strip()


def _require_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _validate_operation_metadata(value: object) -> dict[str, str]:
    """Validate the request identity recorded by a mutation snapshot."""
    if not isinstance(value, Mapping) or set(value) != {
        "operation",
        "request_fingerprint_sha256",
    }:
        raise ValueError("manifest.operation has an invalid schema")

    operation = _require_nonempty_string(
        value["operation"], "operation.operation"
    )
    if operation not in _OPERATION_TYPES:
        raise ValueError("operation.operation is unsupported")

    fingerprint = _require_sha256(
        value["request_fingerprint_sha256"],
        "operation.request_fingerprint_sha256",
    )

    return {
        "operation": operation,
        "request_fingerprint_sha256": fingerprint,
    }


def _json_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    try:
        # canonical_json rejects NaN/Infinity and non-JSON Python objects.
        return json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain JSON-compatible values") from exc


def document_to_record(document: Document) -> dict[str, Any]:
    """Serialize a validated Document with integrity-checkable fingerprints."""
    if not isinstance(document, Document):
        raise TypeError("document must be a Document")
    return {
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "doc_id": document.doc_id,
        "text": document.text,
        "source": document.source,
        "title": document.title,
        "metadata": normalize_metadata(document.metadata),
        "content_sha256": document.content_sha256,
        "revision_sha256": document.revision_sha256,
    }


def document_from_record(record: Mapping[str, Any]) -> Document:
    """Deserialize a document record and reject stale/tampered fingerprints."""
    expected = {
        "record_schema_version", "doc_id", "text", "source", "title",
        "metadata", "content_sha256", "revision_sha256",
    }
    if not isinstance(record, Mapping) or set(record) != expected:
        raise ValueError("document record has missing or unsupported fields")
    if (
        isinstance(record["record_schema_version"], bool)
        or not isinstance(record["record_schema_version"], int)
        or record["record_schema_version"] != RECORD_SCHEMA_VERSION
    ):
        raise ValueError("unsupported document record schema version")
    document = Document(
        doc_id=record["doc_id"],
        text=record["text"],
        source=record["source"],
        title=record["title"],
        metadata=record["metadata"],
    )
    if record["content_sha256"] != document.content_sha256:
        raise ValueError("document content_sha256 mismatch")
    if record["revision_sha256"] != document.revision_sha256:
        raise ValueError("document revision_sha256 mismatch")
    return document


def chunk_to_record(chunk: Chunk) -> dict[str, Any]:
    """Serialize a Chunk while retaining its parent/config revision identity."""
    if not isinstance(chunk, Chunk):
        raise TypeError("chunk must be a Chunk")
    return {
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "source": chunk.source,
        "title": chunk.title,
        "metadata": normalize_metadata(chunk.metadata),
        "document_revision_sha256": chunk.document_revision_sha256,
        "chunking_config": normalize_metadata(chunk.chunking_config),
        "content_sha256": chunk.content_sha256,
        "revision_sha256": chunk.revision_sha256,
    }


def chunk_from_record(record: Mapping[str, Any]) -> Chunk:
    """Deserialize a chunk and reject stale/tampered fingerprints."""
    expected = {
        "record_schema_version", "chunk_id", "doc_id", "text", "source",
        "title", "metadata", "document_revision_sha256", "chunking_config",
        "content_sha256", "revision_sha256",
    }
    if not isinstance(record, Mapping) or set(record) != expected:
        raise ValueError("chunk record has missing or unsupported fields")
    if (
        isinstance(record["record_schema_version"], bool)
        or not isinstance(record["record_schema_version"], int)
        or record["record_schema_version"] != RECORD_SCHEMA_VERSION
    ):
        raise ValueError("unsupported chunk record schema version")
    chunk = Chunk(
        chunk_id=record["chunk_id"],
        doc_id=record["doc_id"],
        text=record["text"],
        source=record["source"],
        title=record["title"],
        metadata=record["metadata"],
        document_revision_sha256=record["document_revision_sha256"],
        chunking_config=record["chunking_config"],
    )
    if record["content_sha256"] != chunk.content_sha256:
        raise ValueError("chunk content_sha256 mismatch")
    if record["revision_sha256"] != chunk.revision_sha256:
        raise ValueError("chunk revision_sha256 mismatch")
    return chunk


def encode_jsonl(records: Iterable[Mapping[str, Any]]) -> bytes:
    """Encode mappings as canonical UTF-8 JSONL with one trailing LF per row."""
    encoded: list[str] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise TypeError(f"JSONL record #{index} must be a mapping")
        try:
            encoded.append(canonical_json(dict(record)))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"JSONL record #{index} is not JSON-compatible") from exc
    if not encoded:
        return b""
    return ("\n".join(encoded) + "\n").encode("utf-8")


class _DuplicateJSONKeyError(ValueError):
    """Raised when one JSON object contains the same key more than once."""


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(
        f"non-standard JSON numeric constant is not allowed: {value}")


def _decode_json_object(line: str, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            line,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {context}: {exc.msg}") from exc
    except _DuplicateJSONKeyError as exc:
        raise ValueError(f"invalid JSON in {context}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"invalid JSON in {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON value in {context} must be an object")
    try:
        canonical = canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"JSON value in {context} is not JSON-compatible") from exc
    if canonical != line:
        raise ValueError(f"JSON value in {context} is not canonically encoded")
    return value


def decode_jsonl(payload: bytes | str) -> list[dict[str, Any]]:
    """Decode strict JSONL; duplicate keys, blank lines and non-objects fail."""
    if isinstance(payload, bytes):
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("JSONL payload must be valid UTF-8") from exc
    elif isinstance(payload, str):
        content = payload
        try:
            content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("JSONL payload must be valid UTF-8") from exc
    else:
        raise TypeError("JSONL payload must be bytes or str")

    if not content:
        return []
    if not content.endswith("\n"):
        raise ValueError("JSONL payload must end with a newline")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(content[:-1].split("\n"), start=1):
        if line.endswith("\r"):
            raise ValueError(
                "JSONL payload must use LF, not CRLF, line endings")
        if not line.strip():
            raise ValueError(
                f"JSONL contains a blank line at line {line_number}")
        try:
            rows.append(
                _decode_json_object(
                    line, context=f"JSONL record at line {line_number}")
            )
        except ValueError as exc:
            raise ValueError(
                f"invalid JSONL record at line {line_number}: {exc}"
            ) from exc
    return rows


def artifact_metadata(path: str | Path) -> dict[str, Any]:
    """Return SHA-256 and byte size for a file, streaming large artifacts."""
    artifact_path = Path(path)
    digest = hashlib.sha256()
    size_bytes = 0
    with artifact_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size_bytes += len(block)
    return {"sha256": digest.hexdigest(), "size_bytes": size_bytes}


def corpus_fingerprint(
    documents: Iterable[Document], chunks: Iterable[Chunk]
) -> str:
    """Fingerprint ordered document/chunk revisions used by one index build.

    Ordering is significant because the dense vector matrix is positionally
    aligned with the ordered chunk records.
    """
    document_rows = list(documents)
    chunk_rows = list(chunks)
    if any(not isinstance(item, Document) for item in document_rows):
        raise TypeError("documents must contain only Document instances")
    if any(not isinstance(item, Chunk) for item in chunk_rows):
        raise TypeError("chunks must contain only Chunk instances")
    return sha256_json(
        {
            "identity_type": "hs.index_corpus",
            "identity_schema_version": 1,
            "document_revisions": [item.revision_sha256 for item in document_rows],
            "chunk_revisions": [item.revision_sha256 for item in chunk_rows],
        }
    )


def _validate_artifact_record(
    record: object,
    name: str,
    *,
    expected_record_count: int | None = None,
    expected_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"artifact metadata for {name} must be an object")
    result = dict(record)
    _require_sha256(result.get("sha256"), f"artifacts.{name}.sha256")
    _require_int(result.get("size_bytes"), f"artifacts.{name}.size_bytes")

    if expected_record_count is not None:
        if set(result) != {"sha256", "size_bytes", "record_count"}:
            raise ValueError(f"artifacts.{name} has an invalid schema")
        count = _require_int(
            result["record_count"], f"artifacts.{name}.record_count"
        )
        if count != expected_record_count:
            raise ValueError(
                f"artifacts.{name}.record_count does not match corpus")
    elif expected_shape is not None:
        if set(result) != {"sha256", "size_bytes", "shape", "dtype"}:
            raise ValueError(f"artifacts.{name} has an invalid schema")
        shape = result["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in shape)
        ):
            raise ValueError(
                f"artifacts.{name}.shape must be a two-integer list")
        if tuple(shape) != expected_shape:
            raise ValueError(
                f"artifacts.{name}.shape does not match vectors metadata")
        if result["dtype"] != "float32":
            raise ValueError(f"artifacts.{name}.dtype must be float32")
    return result


def build_manifest(
    *,
    index_id: str,
    embedding_model_id: str,
    embedding_model_revision: str,
    embedding_config: Mapping[str, Any],
    chunking_config: Mapping[str, Any],
    sparse_config: Mapping[str, Any],
    document_count: int,
    chunk_count: int,
    vector_dimension: int,
    corpus_fingerprint_sha256: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    created_at_utc: str | None = None,
    operation_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate a v1 snapshot or operation-aware v2 snapshot."""
    if created_at_utc is None:
        created_at_utc = datetime.now(timezone.utc).isoformat()
    if not isinstance(artifacts, Mapping):
        raise ValueError("artifacts must be a mapping")
    normalized_operation = (
        _validate_operation_metadata(operation_metadata)
        if operation_metadata is not None
        else None
    )

    manifest = {
        "format": INDEX_FORMAT,
        "schema_version": (
            _OPERATION_SCHEMA_VERSION
            if normalized_operation is not None
            else INDEX_SCHEMA_VERSION
        ),
        "index_id": index_id,
        "created_at_utc": created_at_utc,
        "corpus": {
            "document_count": document_count,
            "chunk_count": chunk_count,
            "fingerprint_sha256": corpus_fingerprint_sha256,
        },
        "embedding": {
            "model_id": embedding_model_id,
            "model_revision": embedding_model_revision,
            "config": _json_mapping(embedding_config, "embedding_config"),
        },
        "vectors": {
            "rows": chunk_count,
            "dimension": vector_dimension,
            "dtype": "float32",
            "normalized": True,
        },
        "chunking": _json_mapping(chunking_config, "chunking_config"),
        "sparse": _json_mapping(sparse_config, "sparse_config"),
        "artifacts": dict(artifacts),
    }
    if normalized_operation is not None:
        manifest["operation"] = normalized_operation
    return validate_manifest(manifest)


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_embedding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate manifest v1/v2 strictly and return an independent JSON-safe copy."""
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a JSON object")
    if "schema_version" not in manifest:
        raise ValueError("manifest has missing or unsupported fields")

    version = manifest["schema_version"]

    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version not in {INDEX_SCHEMA_VERSION, _OPERATION_SCHEMA_VERSION}
    ):
        raise ValueError("unsupported index schema version")

    required = {
        "format", "schema_version", "index_id", "created_at_utc", "corpus",
        "embedding", "vectors", "chunking", "sparse", "artifacts",
    }

    if version == _OPERATION_SCHEMA_VERSION:
        required.add("operation")

    if set(manifest) != required:
        raise ValueError("manifest has missing or unsupported fields")

    if manifest["format"] != INDEX_FORMAT:
        raise ValueError("unsupported index format")

    index_id = _require_nonempty_string(manifest["index_id"], "index_id")
    if not _INDEX_ID_RE.fullmatch(index_id):
        raise ValueError("index_id contains unsafe path characters")
    windows_device_name = index_id.split(".", 1)[0].upper()
    if windows_device_name in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(
        r"(?:COM|LPT)[1-9]", windows_device_name
    ):
        raise ValueError("index_id is a reserved filesystem name")

    created_at = _require_nonempty_string(
        manifest["created_at_utc"], "created_at_utc"
    )
    try:
        parsed_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "created_at_utc must be an ISO-8601 timestamp") from exc
    if parsed_time.tzinfo is None or parsed_time.utcoffset() != timedelta(0):
        raise ValueError("created_at_utc must include a UTC timezone")

    corpus = manifest["corpus"]
    if not isinstance(corpus, Mapping) or set(corpus) != {
        "document_count", "chunk_count", "fingerprint_sha256"
    }:
        raise ValueError("manifest.corpus has an invalid schema")
    document_count = _require_int(
        corpus["document_count"], "corpus.document_count", minimum=1)
    chunk_count = _require_int(
        corpus["chunk_count"], "corpus.chunk_count", minimum=1)
    _require_sha256(corpus["fingerprint_sha256"], "corpus.fingerprint_sha256")

    embedding = manifest["embedding"]
    if not isinstance(embedding, Mapping) or set(embedding) != {
        "model_id", "model_revision", "config"
    }:
        raise ValueError("manifest.embedding has an invalid schema")
    model_revision = _require_nonempty_string(
        embedding["model_revision"], "embedding.model_revision"
    )
    if not _REVISION_RE.fullmatch(model_revision):
        raise ValueError(
            "embedding.model_revision must be an immutable hexadecimal commit")
    normalized_embedding = {
        "model_id": _require_nonempty_string(embedding["model_id"], "embedding.model_id"),
        "model_revision": model_revision,
        "config": _json_mapping(embedding["config"], "embedding.config"),
    }
    if expected_embedding is not None:
        if not isinstance(expected_embedding, Mapping):
            raise ValueError("expected_embedding must be a JSON object")
        normalized_expected = {
            "model_id": _require_nonempty_string(
                expected_embedding.get(
                    "model_id"), "expected_embedding.model_id"
            ),
            "model_revision": _require_nonempty_string(
                expected_embedding.get("model_revision"),
                "expected_embedding.model_revision",
            ),
            "config": _json_mapping(
                expected_embedding.get("config"), "expected_embedding.config"
            ),
        }
        if canonical_json(normalized_embedding) != canonical_json(normalized_expected):
            raise ValueError(
                "index embedding identity is incompatible with requested model")

    vectors = manifest["vectors"]
    if not isinstance(vectors, Mapping) or set(vectors) != {
        "rows", "dimension", "dtype", "normalized"
    }:
        raise ValueError("manifest.vectors has an invalid schema")
    rows = _require_int(vectors["rows"], "vectors.rows", minimum=1)
    dimension = _require_int(
        vectors["dimension"], "vectors.dimension", minimum=1)
    if rows != chunk_count:
        raise ValueError("vectors.rows must equal corpus.chunk_count")
    if vectors["dtype"] != "float32":
        raise ValueError("vectors.dtype must be float32")
    if vectors["normalized"] is not True:
        raise ValueError("vectors.normalized must be true")

    chunking = _json_mapping(manifest["chunking"], "chunking")
    sparse = _json_mapping(manifest["sparse"], "sparse")
    if not chunking or not sparse:
        raise ValueError(
            "chunking and sparse configurations must not be empty")

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(REQUIRED_ARTIFACTS):
        raise ValueError(
            "manifest.artifacts must contain exactly the required artifacts")
    normalized_artifacts = {
        DOCUMENTS_ARTIFACT: _validate_artifact_record(
            artifacts[DOCUMENTS_ARTIFACT], DOCUMENTS_ARTIFACT,
            expected_record_count=document_count,
        ),
        CHUNKS_ARTIFACT: _validate_artifact_record(
            artifacts[CHUNKS_ARTIFACT], CHUNKS_ARTIFACT,
            expected_record_count=chunk_count,
        ),
        VECTORS_ARTIFACT: _validate_artifact_record(
            artifacts[VECTORS_ARTIFACT], VECTORS_ARTIFACT,
            expected_shape=(chunk_count, dimension),
        ),
    }

    normalized = {
        "format": INDEX_FORMAT,
        "schema_version": version,
        "index_id": index_id,
        "created_at_utc": created_at,
        "corpus": {
            "document_count": document_count,
            "chunk_count": chunk_count,
            "fingerprint_sha256": corpus["fingerprint_sha256"],
        },
        "embedding": normalized_embedding,
        "vectors": {
            "rows": rows,
            "dimension": dimension,
            "dtype": "float32",
            "normalized": True,
        },
        "chunking": chunking,
        "sparse": sparse,
        "artifacts": normalized_artifacts,
    }
    if version == _OPERATION_SCHEMA_VERSION:
        normalized["operation"] = _validate_operation_metadata(
            manifest["operation"]
        )
    # Ensure manifest is fully JSON serializable and detached from caller objects.
    return json.loads(canonical_json(normalized))


def _index_chunking_config(chunks: list[Chunk]) -> dict[str, Any]:
    """Return the one chunking configuration represented by all chunks."""
    if not chunks:
        raise ValueError(
            "cannot build a persistent index without searchable chunks")

    configs = [normalize_metadata(chunk.chunking_config) for chunk in chunks]
    if any(not config for config in configs):
        raise ValueError(
            "every persisted chunk must declare its chunking_config")

    encoded = {canonical_json(config) for config in configs}
    if len(encoded) != 1:
        raise ValueError(
            "all chunks in one index must use the same chunking_config")
    return configs[0]


def _chunking_parameters(config: Mapping[str, Any]) -> tuple[int | None, int]:
    """Translate the supported, versioned chunk config into chunk_text args."""
    algorithm = config.get("algorithm")
    version = config.get("version")
    if isinstance(version, bool) or version != 1:
        raise ValueError("unsupported chunking algorithm version")

    if algorithm == "whole_document" and set(config) == {"algorithm", "version"}:
        return None, 0

    if algorithm == "word_window" and set(config) == {
        "algorithm", "version", "size", "overlap"
    }:
        size = config["size"]
        overlap = config["overlap"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or isinstance(overlap, bool)
            or not isinstance(overlap, int)
            or overlap < 0
            or overlap >= size
        ):
            raise ValueError("invalid word_window chunking parameters")
        return size, overlap

    raise ValueError("unsupported chunking algorithm/configuration")


def _validate_build_corpus(
    documents: list[Document],
    chunks: list[Chunk],
    *,
    chunking_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate document/chunk identity and reproduce the declared chunk set."""
    if not documents:
        raise ValueError(
            "cannot build a persistent index from an empty document set")
    if not chunks:
        raise ValueError(
            "cannot build a persistent index without searchable chunks")
    if any(not isinstance(document, Document) for document in documents):
        raise TypeError("documents must contain only Document instances")
    if any(not isinstance(chunk, Chunk) for chunk in chunks):
        raise TypeError("chunks must contain only Chunk instances")

    document_by_id: dict[str, Document] = {}
    for document in documents:
        if document.doc_id in document_by_id:
            raise ValueError(f"duplicate document ID: {document.doc_id!r}")
        document_by_id[document.doc_id] = document

    seen_chunk_ids: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen_chunk_ids:
            raise ValueError(f"duplicate chunk ID: {chunk.chunk_id!r}")
        seen_chunk_ids.add(chunk.chunk_id)
        document = document_by_id.get(chunk.doc_id)
        if document is None:
            raise ValueError(
                f"chunk {chunk.chunk_id!r} references unknown document {chunk.doc_id!r}"
            )
        if chunk.document_revision_sha256 != document.revision_sha256:
            raise ValueError(
                f"chunk {chunk.chunk_id!r} does not match its parent document revision"
            )

    inferred_config = _index_chunking_config(chunks)
    if chunking_config is not None:
        supplied_config = _json_mapping(chunking_config, "chunking_config")
        if canonical_json(supplied_config) != canonical_json(inferred_config):
            raise ValueError(
                "chunking_config does not match the persisted chunks")

    size, overlap = _chunking_parameters(inferred_config)

    # Re-run the versioned deterministic chunker and compare serialized rows.
    # This catches omitted, reordered, stale, or metadata-mismatched chunks
    # before embedding work and persistence begin.
    expected_chunks: list[Chunk] = []
    for document in documents:
        expected_chunks.extend(
            chunk_text(
                document.doc_id,
                document.text,
                size,
                overlap,
                source=document.source,
                title=document.title,
                metadata=document.metadata,
                document_revision_sha256=document.revision_sha256,
            )
        )

    expected_records = [
        canonical_json(chunk_to_record(chunk)) for chunk in expected_chunks
    ]
    actual_records = [canonical_json(chunk_to_record(chunk))
                      for chunk in chunks]
    if actual_records != expected_records:
        raise ValueError(
            "chunks do not match the ordered documents and declared "
            "chunking configuration"
        )
    return inferred_config


def _normalize_embedding_matrix(raw: object, *, expected_rows: int) -> np.ndarray:
    """Validate and L2-normalize embeddings using DenseIndex's zero-row policy."""
    try:
        matrix = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("embedder must return a numeric 2D matrix") from exc

    if (
        matrix.ndim != 2
        or matrix.shape[0] != expected_rows
        or matrix.shape[1] == 0
    ):
        raise ValueError(
            "embedder must return a matrix with shape "
            f"({expected_rows}, dimension>0); got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings must contain only finite values")

    # Match src/hs/dense.py::_normalize exactly: non-zero rows become unit
    # length and zero rows remain zero rather than producing NaN values.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all():
        raise ValueError("embedding vector norms overflow float32")
    underflowed = np.any(matrix != 0, axis=1) & (norms[:, 0] == 0)
    if underflowed.any():
        raise ValueError("non-zero embedding vector norm underflowed float32")

    normalized = matrix / np.where(norms == 0, 1, norms)
    if not np.isfinite(normalized).all():
        raise ValueError(
            "normalized embeddings must contain only finite values")
    return np.ascontiguousarray(normalized, dtype=np.float32)


def _write_bytes_durable(path: Path, payload: bytes) -> None:
    """Create one immutable artifact file and flush its contents to disk."""
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_precomputed_embedding_matrix(
    raw: object,
    *,
    expected_rows: int,
) -> np.ndarray:
    """Validate an already-normalized matrix without changing its values.

    Incremental index operations reuse rows from a previously validated
    snapshot. Re-normalizing those rows can introduce tiny float32 changes,
    so this path validates normalization and preserves supplied values exactly.
    """
    try:
        matrix = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "precomputed embeddings must be a numeric 2D matrix") from exc

    if (
        matrix.ndim != 2
        or matrix.shape[0] != expected_rows
        or matrix.shape[1] == 0
    ):
        raise ValueError(
            "precomputed embeddings must have shape "
            f"({expected_rows}, dimension>0); got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(
            "precomputed embeddings must contain only finite values")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError("precomputed embeddings contain invalid vector norms")
    nonzero_rows = np.any(matrix != 0, axis=1)
    if np.any(nonzero_rows & (norms == 0)):
        raise ValueError(
            "precomputed embeddings contain non-zero rows with zero norm")
    if nonzero_rows.any() and not np.allclose(
        norms[nonzero_rows], 1.0, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("precomputed embeddings must be L2-normalized")

    return np.ascontiguousarray(matrix, dtype=np.float32)


def _write_vectors_durable(path: Path, vectors: np.ndarray) -> None:
    """Write a NumPy matrix without pickle and flush it to disk."""
    with path.open("xb") as handle:
        np.save(handle, vectors, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def build_index_snapshot(
    output_dir: str | Path,
    *,
    index_id: str,
    documents: Iterable[Document],
    chunks: Iterable[Chunk],
    embed_fn: Callable[[list[str]], object] | None,
    embedding_model_id: str,
    embedding_model_revision: str,
    embedding_config: Mapping[str, Any],
    sparse_config: Mapping[str, Any],
    chunking_config: Mapping[str, Any] | None = None,
    created_at_utc: str | None = None,
    precomputed_vectors: object | None = None,
    operation_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a new on-disk index snapshot and return its validated manifest.

    The destination must not already exist. Document and chunk records retain
    caller order; vector row ``i`` is generated from chunk row ``i``. The
    manifest is written last, so an interrupted build cannot look complete to
    the future loader. This function does not activate a snapshot or implement
    ``CURRENT.json`` switching.
    """
    if precomputed_vectors is None:
        if not callable(embed_fn):
            raise TypeError("embed_fn must be callable")
    elif embed_fn is not None and not callable(embed_fn):
        raise TypeError("embed_fn must be callable or None")

    document_rows = list(documents)
    chunk_rows = list(chunks)
    normalized_chunking_config = _validate_build_corpus(
        document_rows,
        chunk_rows,
        chunking_config=chunking_config,
    )

    # Validate configuration and identity before running a potentially costly
    # model call. build_manifest performs the canonical schema checks later too.
    normalized_embedding_config = _json_mapping(
        embedding_config, "embedding_config"
    )
    normalized_sparse_config = _json_mapping(sparse_config, "sparse_config")
    if not normalized_embedding_config:
        raise ValueError("embedding_config must not be empty")
    if not normalized_sparse_config:
        raise ValueError("sparse_config must not be empty")
    _require_nonempty_string(embedding_model_id, "embedding_model_id")
    revision = _require_nonempty_string(
        embedding_model_revision, "embedding_model_revision"
    )
    if not _REVISION_RE.fullmatch(revision):
        raise ValueError(
            "embedding_model_revision must be an immutable hexadecimal commit"
        )
    _require_nonempty_string(index_id, "index_id")
    if not _INDEX_ID_RE.fullmatch(index_id):
        raise ValueError("index_id contains unsafe path characters")
    device_name = index_id.split(".", 1)[0].upper()
    if device_name in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(
        r"(?:COM|LPT)[1-9]", device_name
    ):
        raise ValueError("index_id is a reserved filesystem name")

    target = Path(output_dir)
    if target.exists() or target.is_symlink():
        raise FileExistsError(
            f"index snapshot destination already exists: {target}")

    if precomputed_vectors is None:
        texts = [chunk.text for chunk in chunk_rows]
        try:
            raw_vectors = embed_fn(texts)
        except Exception as exc:
            raise RuntimeError(
                "embedding generation failed while building index") from exc
        vectors = _normalize_embedding_matrix(
            raw_vectors, expected_rows=len(chunk_rows)
        )
    else:
        vectors = _validate_precomputed_embedding_matrix(
            precomputed_vectors,
            expected_rows=len(chunk_rows),
        )

    document_payload = encode_jsonl(
        document_to_record(document) for document in document_rows
    )
    chunk_payload = encode_jsonl(chunk_to_record(chunk)
                                 for chunk in chunk_rows)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)

    try:
        documents_path = target / DOCUMENTS_ARTIFACT
        chunks_path = target / CHUNKS_ARTIFACT
        vectors_path = target / VECTORS_ARTIFACT
        _write_bytes_durable(documents_path, document_payload)
        _write_bytes_durable(chunks_path, chunk_payload)
        _write_vectors_durable(vectors_path, vectors)

        artifacts = {
            DOCUMENTS_ARTIFACT: {
                **artifact_metadata(documents_path),
                "record_count": len(document_rows),
            },
            CHUNKS_ARTIFACT: {
                **artifact_metadata(chunks_path),
                "record_count": len(chunk_rows),
            },
            VECTORS_ARTIFACT: {
                **artifact_metadata(vectors_path),
                "shape": list(vectors.shape),
                "dtype": "float32",
            },
        }
        manifest = build_manifest(
            index_id=index_id,
            embedding_model_id=embedding_model_id,
            embedding_model_revision=revision,
            embedding_config=normalized_embedding_config,
            chunking_config=normalized_chunking_config,
            sparse_config=normalized_sparse_config,
            document_count=len(document_rows),
            chunk_count=len(chunk_rows),
            vector_dimension=vectors.shape[1],
            corpus_fingerprint_sha256=corpus_fingerprint(
                document_rows, chunk_rows
            ),
            artifacts=artifacts,
            created_at_utc=created_at_utc,
            operation_metadata=operation_metadata,

        )
        manifest_payload = (canonical_json(manifest) + "\n").encode("utf-8")
        _write_bytes_durable(target / "manifest.json", manifest_payload)
        return manifest
    except Exception:
        # A failed in-process build is cleaned up. A hard process/host stop may
        # leave an orphan directory, but it has no manifest and is never active.
        import shutil

        shutil.rmtree(target, ignore_errors=True)
        raise


@dataclass(frozen=True)
class LoadedIndexSnapshot:
    """Validated snapshot records and their row-aligned dense vectors."""

    path: Path
    manifest: dict[str, Any]
    documents: tuple[Document, ...]
    chunks: tuple[Chunk, ...]
    vectors: np.ndarray


def _read_manifest_file(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(
            f"missing or unsafe index manifest: {path.name}")
    try:
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("manifest.json exceeds the 1 MiB safety limit")
        payload = path.read_bytes()
        content = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("manifest.json must be valid UTF-8") from exc
    except OSError as exc:
        raise OSError(f"cannot read index manifest: {exc}") from exc

    if not content.endswith("\n"):
        raise ValueError("manifest.json must end with a newline")
    if content.endswith("\r\n"):
        raise ValueError("manifest.json must use LF, not CRLF, line endings")
    manifest = _decode_json_object(content[:-1], context="manifest.json")
    return validate_manifest(manifest)


def _verify_snapshot_layout(snapshot_dir: Path) -> None:
    if snapshot_dir.is_symlink() or not snapshot_dir.is_dir():
        raise FileNotFoundError(
            "index snapshot directory does not exist or is unsafe")

    expected_names = {"manifest.json", *REQUIRED_ARTIFACTS}
    try:
        entries = list(snapshot_dir.iterdir())
    except OSError as exc:
        raise OSError(f"cannot list index snapshot directory: {exc}") from exc
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ValueError(
            f"snapshot layout mismatch; missing={missing}, unexpected={unexpected}"
        )
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(
                f"snapshot artifact must be a regular file: {entry.name}")


def _verify_artifact_bytes(snapshot_dir: Path, manifest: Mapping[str, Any]) -> None:
    for name in REQUIRED_ARTIFACTS:
        path = snapshot_dir / name
        expected = manifest["artifacts"][name]
        try:
            actual = artifact_metadata(path)
        except OSError as exc:
            raise OSError(
                f"cannot read snapshot artifact {name}: {exc}") from exc
        if actual["size_bytes"] != expected["size_bytes"]:
            raise ValueError(f"artifact size mismatch: {name}")
        if actual["sha256"] != expected["sha256"]:
            raise ValueError(f"artifact SHA-256 mismatch: {name}")


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read JSONL incrementally to avoid holding a second full byte copy."""
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise ValueError(
                        f"{path.name} must end every record with an LF newline"
                    )
                raw = line[:-1]
                if raw.endswith("\r"):
                    raise ValueError(
                        f"{path.name} must use LF, not CRLF, line endings")
                if not raw.strip():
                    raise ValueError(
                        f"{path.name} contains a blank line at line {line_number}"
                    )
                records.append(
                    _decode_json_object(
                        raw, context=f"{path.name} line {line_number}")
                )
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name} must be valid UTF-8") from exc
    return records


def _validate_loaded_vectors(path: Path, manifest: Mapping[str, Any]) -> np.ndarray:
    try:
        vectors = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError, TypeError) as exc:
        raise ValueError(f"vectors.npy is corrupt or invalid: {exc}") from exc

    expected = manifest["vectors"]
    expected_shape = (expected["rows"], expected["dimension"])
    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
        raise ValueError("vectors.npy must contain a two-dimensional ndarray")
    if vectors.shape != expected_shape:
        raise ValueError(
            f"vectors.npy shape {vectors.shape} does not match {expected_shape}"
        )
    if vectors.dtype != np.dtype(np.float32):
        raise ValueError("vectors.npy dtype must be float32")
    if not vectors.flags.c_contiguous:
        raise ValueError("vectors.npy must be C-contiguous")
    if not np.isfinite(vectors).all():
        raise ValueError("vectors.npy contains non-finite values")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError(
            "vectors.npy contains vector norms that overflow float32")
    nonzero_rows = np.any(vectors != 0, axis=1)
    if np.any(nonzero_rows & (norms == 0)):
        raise ValueError("vectors.npy contains non-zero rows with zero norm")
    if nonzero_rows.any() and not np.allclose(
        norms[nonzero_rows], 1.0, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("vectors.npy contains non-normalized non-zero rows")

    vectors.setflags(write=False)
    return vectors


def load_index_snapshot(
    snapshot_dir: str | Path,
    *,
    expected_embedding: Mapping[str, Any] | None = None,
) -> LoadedIndexSnapshot:
    """Load a snapshot only after validating its manifest and every artifact.

    No embedding function is accepted or invoked here: stored document vectors
    are loaded directly. The caller still needs a compatible query embedder for
    dense search, which is integrated with DenseIndex in the next step.
    """
    path = Path(snapshot_dir)
    _verify_snapshot_layout(path)
    manifest = _read_manifest_file(path / "manifest.json")
    if expected_embedding is not None:
        manifest = validate_manifest(
            manifest, expected_embedding=expected_embedding)

    _verify_artifact_bytes(path, manifest)

    document_records = _read_jsonl_records(path / DOCUMENTS_ARTIFACT)
    chunk_records = _read_jsonl_records(path / CHUNKS_ARTIFACT)
    if len(document_records) != manifest["corpus"]["document_count"]:
        raise ValueError(
            "documents.jsonl record count does not match manifest")
    if len(chunk_records) != manifest["corpus"]["chunk_count"]:
        raise ValueError("chunks.jsonl record count does not match manifest")

    documents = tuple(document_from_record(record)
                      for record in document_records)
    chunks = tuple(chunk_from_record(record) for record in chunk_records)
    _validate_build_corpus(
        list(documents),
        list(chunks),
        chunking_config=manifest["chunking"],
    )

    actual_fingerprint = corpus_fingerprint(documents, chunks)
    if actual_fingerprint != manifest["corpus"]["fingerprint_sha256"]:
        raise ValueError("corpus fingerprint does not match manifest")

    vectors = _validate_loaded_vectors(path / VECTORS_ARTIFACT, manifest)
    return LoadedIndexSnapshot(
        path=path,
        manifest=manifest,
        documents=documents,
        chunks=chunks,
        vectors=vectors,
    )


_CURRENT_FORMAT = "enterprise-hybrid-search-current"
_CURRENT_SCHEMA_VERSION = 1
_CURRENT_ARTIFACT = "CURRENT.json"
_CURRENT_MAX_BYTES = 64 * 1024


def _validate_index_id(value: object, name: str = "index_id") -> str:
    index_id = _require_nonempty_string(value, name)
    if not _INDEX_ID_RE.fullmatch(index_id):
        raise ValueError(f"{name} contains unsafe path characters")

    if index_id.endswith("."):
        raise ValueError(f"{name} must not end with a period")

    device_name = index_id.split(".", 1)[0].upper()
    if device_name in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(
        r"(?:COM|LPT)[1-9]", device_name
    ):
        raise ValueError(f"{name} is a reserved filesystem name")
    return index_id


def _index_directories(index_root: str | Path) -> tuple[Path, Path, Path]:
    """Create and return the index root, staging and versions directories."""
    root = Path(index_root)
    if root.is_symlink():
        raise ValueError("index_root must not be a symbolic link")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError("index_root must be a directory")

    staging = root / "staging"
    versions = root / "versions"
    for directory, label in ((staging, "staging"), (versions, "versions")):
        if directory.is_symlink():
            raise ValueError(
                f"index {label} directory must not be a symbolic link")
        directory.mkdir(exist_ok=True)
        if not directory.is_dir():
            raise ValueError(f"index {label} path must be a directory")
    return root, staging, versions


def _validate_current_pointer(pointer: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the strict CURRENT.json contract and return a detached mapping."""
    if not isinstance(pointer, Mapping):
        raise ValueError("CURRENT.json must contain a JSON object")
    required = {
        "format",
        "schema_version",
        "index_id",
        "manifest_sha256",
        "previous_index_id",
        "previous_manifest_sha256",
    }
    if set(pointer) != required:
        raise ValueError("CURRENT.json has missing or unsupported fields")
    if pointer["format"] != _CURRENT_FORMAT:
        raise ValueError("unsupported CURRENT.json format")
    version = pointer["schema_version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != _CURRENT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported CURRENT.json schema version")

    index_id = _validate_index_id(pointer["index_id"], "CURRENT.json index_id")
    manifest_sha256 = _require_sha256(
        pointer["manifest_sha256"], "CURRENT.json manifest_sha256"
    )
    previous_index_id = pointer["previous_index_id"]
    previous_manifest_sha256 = pointer["previous_manifest_sha256"]
    if previous_index_id is None and previous_manifest_sha256 is None:
        pass
    elif previous_index_id is None or previous_manifest_sha256 is None:
        raise ValueError("CURRENT.json previous index identity is incomplete")
    else:
        previous_index_id = _validate_index_id(
            previous_index_id, "CURRENT.json previous_index_id"
        )
        previous_manifest_sha256 = _require_sha256(
            previous_manifest_sha256,
            "CURRENT.json previous_manifest_sha256",
        )
        if previous_index_id == index_id:
            raise ValueError(
                "CURRENT.json cannot point to itself as the previous index")

    return {
        "format": _CURRENT_FORMAT,
        "schema_version": _CURRENT_SCHEMA_VERSION,
        "index_id": index_id,
        "manifest_sha256": manifest_sha256,
        "previous_index_id": previous_index_id,
        "previous_manifest_sha256": previous_manifest_sha256,
    }


def _read_current_pointer(index_root: str | Path) -> dict[str, Any] | None:
    """Read CURRENT.json strictly; return None only when it is truly absent."""
    root = Path(index_root)
    if root.is_symlink():
        raise ValueError("index_root must not be a symbolic link")
    if root.exists() and not root.is_dir():
        raise ValueError("index_root must be a directory")
    path = root / _CURRENT_ARTIFACT
    if path.is_symlink():
        raise ValueError("CURRENT.json must not be a symbolic link")
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError("CURRENT.json must be a regular file")
    try:
        if path.stat().st_size > _CURRENT_MAX_BYTES:
            raise ValueError("CURRENT.json exceeds the 64 KiB safety limit")
        payload = path.read_bytes()
    except OSError as exc:
        raise OSError(f"cannot read CURRENT.json: {exc}") from exc
    try:
        content = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("CURRENT.json must be valid UTF-8") from exc
    if not content.endswith("\n"):
        raise ValueError("CURRENT.json must end with a newline")
    if content.endswith("\r\n"):
        raise ValueError("CURRENT.json must use LF, not CRLF, line endings")
    record = _decode_json_object(content[:-1], context="CURRENT.json")
    return _validate_current_pointer(record)


def _write_current_pointer(index_root: str | Path, pointer: Mapping[str, Any]) -> None:
    """Flush a temporary pointer and atomically replace CURRENT.json."""
    root = Path(index_root)
    normalized = _validate_current_pointer(pointer)
    payload = (canonical_json(normalized) + "\n").encode("utf-8")
    fd: int | None = None
    temporary_path: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=".CURRENT.", suffix=".tmp", dir=root
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(fd, "wb") as handle:
            fd = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, root / _CURRENT_ARTIFACT)
    except Exception:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _load_pointer_target(
    index_root: str | Path,
    pointer: Mapping[str, Any],
    *,
    expected_embedding: Mapping[str, Any] | None = None,
) -> LoadedIndexSnapshot:
    """Load the referenced immutable version and verify its manifest identity."""
    root = Path(index_root)
    versions = root / "versions"
    if versions.is_symlink() or not versions.is_dir():
        raise FileNotFoundError(
            "index versions directory does not exist or is unsafe")
    index_id = _validate_index_id(pointer["index_id"])
    version_dir = versions / index_id
    if version_dir.is_symlink() or not version_dir.is_dir():
        raise FileNotFoundError(
            f"active index version is missing or unsafe: {index_id}")
    manifest_path = version_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError("active index manifest is missing or unsafe")
    actual_manifest_sha256 = artifact_metadata(manifest_path)["sha256"]
    if actual_manifest_sha256 != pointer["manifest_sha256"]:
        raise ValueError(
            "CURRENT.json manifest checksum does not match active snapshot")
    loaded = load_index_snapshot(
        version_dir, expected_embedding=expected_embedding)
    if loaded.manifest["index_id"] != index_id:
        raise ValueError(
            "CURRENT.json index_id does not match snapshot manifest")
    return loaded


def _build_operation_metadata(
    operation: str,
    request: Mapping[str, Any],
) -> dict[str, str]:
    """Create a deterministic fingerprint for one mutation request."""
    normalized_operation = _require_nonempty_string(
        operation, "operation"
    )
    if normalized_operation not in _OPERATION_TYPES:
        raise ValueError("operation is unsupported")

    normalized_request = _json_mapping(
        request, "operation request"
    )

    fingerprint = sha256_json(
        {
            "identity_type": "hs.index_operation_request",
            "identity_schema_version": 1,
            "operation": normalized_operation,
            "request": normalized_request,
        }
    )

    return _validate_operation_metadata(
        {
            "operation": normalized_operation,
            "request_fingerprint_sha256": fingerprint,
        }
    )


def _load_existing_operation_snapshot(
    index_root: str | Path,
    index_id: str,
    operation_metadata: Mapping[str, Any],
) -> LoadedIndexSnapshot | None:
    """Return an exact active retry, or reject an existing immutable version."""
    normalized_index_id = _validate_index_id(index_id)
    normalized_metadata = _validate_operation_metadata(
        operation_metadata
    )

    version_dir = (
        Path(index_root) / "versions" / normalized_index_id
    )

    if not version_dir.exists() and not version_dir.is_symlink():
        return None

    current_pointer = _read_current_pointer(index_root)

    # An existing inactive version must never be reactivated by a retry.
    if (
        current_pointer is None
        or current_pointer["index_id"] != normalized_index_id
    ):
        raise FileExistsError(
            f"immutable index version already exists: {normalized_index_id}"
        )

    snapshot = _load_pointer_target(
        index_root,
        current_pointer,
    )

    stored_metadata = snapshot.manifest.get("operation")

    if (
        stored_metadata is None
        or canonical_json(stored_metadata)
        != canonical_json(normalized_metadata)
    ):
        raise ValueError(
            f"idempotency conflict for index version: {normalized_index_id}"
        )

    return snapshot


def load_current_index_snapshot(
    index_root: str | Path,
    *,
    expected_embedding: Mapping[str, Any] | None = None,
) -> LoadedIndexSnapshot:
    """Load the active version only when CURRENT.json and its snapshot agree."""
    pointer = _read_current_pointer(index_root)
    if pointer is None:
        raise FileNotFoundError(
            "CURRENT.json does not exist; no index is active")
    return _load_pointer_target(
        index_root, pointer, expected_embedding=expected_embedding
    )


def activate_index_snapshot(
    index_root: str | Path,
    staged_snapshot_dir: str | Path,
    *,
    expected_embedding: Mapping[str, Any] | None = None,
) -> LoadedIndexSnapshot:
    """Validate and promote one staged snapshot, then atomically switch CURRENT.

    ``staged_snapshot_dir`` must be a directory below ``index_root/staging``.
    It is validated before promotion. Snapshot versions are immutable: an
    existing ``versions/<index_id>`` directory is never overwritten.

    The pointer is written to a sibling temporary file, flushed, and replaced
    with ``os.replace``. If pointer replacement fails, the previous CURRENT.json
    remains untouched; the newly promoted version is retained but inactive.

    Activation is a single-writer operation. Concurrent writers must be
    serialized by the caller; this file-based implementation does not provide
    a cross-process lock.
    """
    root, staging, versions = _index_directories(index_root)
    staged_path = Path(staged_snapshot_dir)
    if staged_path.is_symlink() or not staged_path.is_dir():
        raise FileNotFoundError(
            "staged snapshot directory does not exist or is unsafe")

    staging_resolved = staging.resolve(strict=True)
    staged_resolved = staged_path.resolve(strict=True)
    try:
        staged_resolved.relative_to(staging_resolved)
    except ValueError as exc:
        raise ValueError(
            "staged snapshot must be located inside index_root/staging") from exc
    if staged_resolved == staging_resolved:
        raise ValueError(
            "staged snapshot must be a child of the staging directory")

    # Validate every artifact before any version promotion or active-pointer change.
    staged_snapshot = load_index_snapshot(
        staged_resolved, expected_embedding=expected_embedding
    )
    index_id = _validate_index_id(staged_snapshot.manifest["index_id"])
    version_dir = versions / index_id
    if version_dir.exists() or version_dir.is_symlink():
        raise FileExistsError(
            f"immutable index version already exists: {index_id}")

    current_pointer = _read_current_pointer(root)
    previous_index_id: str | None = None
    previous_manifest_sha256: str | None = None
    if current_pointer is not None:
        # Do not replace an invalid active pointer and silently lose the last
        # known-good version from the rollback chain.
        current_snapshot = _load_pointer_target(root, current_pointer)
        previous_index_id = current_snapshot.manifest["index_id"]
        previous_manifest_sha256 = current_pointer["manifest_sha256"]

    # Staging and versions are siblings under index_root, so this rename stays
    # on the same filesystem. Destination is guaranteed not to exist above.
    os.replace(staged_resolved, version_dir)

    # Re-read from the immutable destination before publishing the pointer.
    promoted_snapshot = load_index_snapshot(
        version_dir, expected_embedding=expected_embedding
    )
    manifest_sha256 = artifact_metadata(
        version_dir / "manifest.json")["sha256"]
    pointer = {
        "format": _CURRENT_FORMAT,
        "schema_version": _CURRENT_SCHEMA_VERSION,
        "index_id": index_id,
        "manifest_sha256": manifest_sha256,
        "previous_index_id": previous_index_id,
        "previous_manifest_sha256": previous_manifest_sha256,
    }
    _write_current_pointer(root, pointer)

    # The containing temporary staging directory may now be empty. Removing it
    # is best-effort and must not turn a successful activation into a failure.
    try:
        staged_resolved.parent.rmdir()
    except OSError:
        pass
    return promoted_snapshot


def rollback_index_snapshot(
    index_root: str | Path,
    *,
    index_id: str | None = None,
    expected_embedding: Mapping[str, Any] | None = None,
) -> LoadedIndexSnapshot:
    """Reactivate the previous (or explicitly selected) retained index version.

    With no ``index_id``, the previous version recorded in CURRENT.json is
    selected. Supplying ``index_id`` permits an operator to reactivate any
    retained immutable version, including a version promoted before a failed
    pointer switch.
    """
    root = Path(index_root)
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError("index_root does not exist or is unsafe")
    versions = root / "versions"
    if versions.is_symlink() or not versions.is_dir():
        raise FileNotFoundError(
            "index versions directory does not exist or is unsafe")

    current_pointer = _read_current_pointer(root)
    target_id = index_id
    if target_id is None:
        if current_pointer is None or current_pointer["previous_index_id"] is None:
            raise ValueError(
                "no previous index version is available for rollback")
        target_id = current_pointer["previous_index_id"]
    target_id = _validate_index_id(target_id)
    if current_pointer is not None and target_id == current_pointer["index_id"]:
        raise ValueError("target index version is already active")

    version_dir = versions / target_id
    if version_dir.is_symlink() or not version_dir.is_dir():
        raise FileNotFoundError(
            f"retained index version does not exist: {target_id}")
    target_snapshot = load_index_snapshot(
        version_dir, expected_embedding=expected_embedding
    )
    if target_snapshot.manifest["index_id"] != target_id:
        raise ValueError(
            "requested index_id does not match retained snapshot manifest")
    target_manifest_sha256 = artifact_metadata(
        version_dir / "manifest.json")["sha256"]

    if (
        current_pointer is not None
        and target_id == current_pointer["previous_index_id"]
        and target_manifest_sha256 != current_pointer["previous_manifest_sha256"]
    ):
        raise ValueError(
            "previous index manifest checksum does not match CURRENT.json")

    previous_index_id = None
    previous_manifest_sha256 = None
    if current_pointer is not None:
        # Rollback is a recovery path: do not require the currently active
        # snapshot to be healthy before switching away from it. Preserve its
        # pointer identity as the new previous target, even when its artifacts
        # are corrupt or missing. The selected rollback target was validated
        # above and is the only snapshot that must be loadable here.
        previous_index_id = current_pointer["index_id"]
        previous_manifest_sha256 = current_pointer["manifest_sha256"]

    _write_current_pointer(
        root,
        {
            "format": _CURRENT_FORMAT,
            "schema_version": _CURRENT_SCHEMA_VERSION,
            "index_id": target_id,
            "manifest_sha256": target_manifest_sha256,
            "previous_index_id": previous_index_id,
            "previous_manifest_sha256": previous_manifest_sha256,
        },
    )
    return target_snapshot


def build_and_activate_index_snapshot(
    index_root: str | Path,
    *,
    index_id: str,
    documents: Iterable[Document],
    chunks: Iterable[Chunk],
    embed_fn: Callable[[list[str]], object] | None,
    embedding_model_id: str,
    embedding_model_revision: str,
    embedding_config: Mapping[str, Any],
    sparse_config: Mapping[str, Any],
    chunking_config: Mapping[str, Any] | None = None,
    created_at_utc: str | None = None,
    expected_embedding: Mapping[str, Any] | None = None,
    precomputed_vectors: object | None = None,
    operation_metadata: Mapping[str, Any] | None = None,
) -> LoadedIndexSnapshot:
    """Build in staging, validate, promote and activate a new immutable version.

    A unique temporary directory under ``index_root/staging`` isolates each
    build. The staging parent is removed best-effort after success or failure;
    a hard process/host stop may leave an inactive staging orphan, never a
    CURRENT.json pointer to an incomplete snapshot.
    """
    normalized_index_id = _validate_index_id(index_id)
    root, staging, versions = _index_directories(index_root)
    destination = versions / normalized_index_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"immutable index version already exists: {normalized_index_id}")
    stage_parent = Path(
        tempfile.mkdtemp(prefix=f".{normalized_index_id}.", dir=staging)
    )
    staged_snapshot_dir = stage_parent / "snapshot"
    try:
        build_index_snapshot(
            staged_snapshot_dir,
            index_id=normalized_index_id,
            documents=documents,
            chunks=chunks,
            embed_fn=embed_fn,
            embedding_model_id=embedding_model_id,
            embedding_model_revision=embedding_model_revision,
            embedding_config=embedding_config,
            sparse_config=sparse_config,
            chunking_config=chunking_config,
            created_at_utc=created_at_utc,
            precomputed_vectors=precomputed_vectors,
            operation_metadata=operation_metadata,
        )
        return activate_index_snapshot(
            root,
            staged_snapshot_dir,
            expected_embedding=expected_embedding,
        )
    finally:
        import shutil

        shutil.rmtree(stage_parent, ignore_errors=True)


def rebuild_and_activate_index_snapshot(
    index_root: str | Path,
    *,
    index_id: str,
    documents: Iterable[Document],
    embed_fn: Callable[[list[str]], object],
    embedding_model_id: str,
    embedding_model_revision: str,
    embedding_config: Mapping[str, Any],
    sparse_config: Mapping[str, Any],
    chunk_size: int | None = 200,
    chunk_overlap: int = 40,
    created_at_utc: str | None = None,
    expected_embedding: Mapping[str, Any] | None = None,
) -> LoadedIndexSnapshot:
    """Rebuild and activate a complete index in canonical document order.

    Documents are materialized once, validated, sorted by doc_id and chunked
    from scratch. The existing staged builder owns persistence, validation
    and atomic activation.
    """
    normalized_index_id = _validate_index_id(index_id)

    if not callable(embed_fn):
        raise TypeError("embed_fn must be callable")

    if (
        isinstance(chunk_overlap, bool)
        or not isinstance(chunk_overlap, int)
        or chunk_overlap < 0
    ):
        raise ValueError("chunk_overlap must be a non-negative integer")

    if chunk_size is None:
        if chunk_overlap != 0:
            raise ValueError(
                "chunk_overlap must be 0 when chunk_size is None"
            )
    else:
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size <= 0
        ):
            raise ValueError(
                "chunk_size must be a positive integer or None"
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

    # Materialize the input iterable exactly once.
    document_rows = list(documents)

    if not document_rows:
        raise ValueError(
            "cannot build a persistent index from an empty document set"
        )

    if any(
        not isinstance(document, Document)
        for document in document_rows
    ):
        raise TypeError(
            "documents must contain only Document instances"
        )

    # Reject duplicate identities before generating chunks or embeddings.
    seen_doc_ids: set[str] = set()

    for document in document_rows:
        if document.doc_id in seen_doc_ids:
            raise ValueError(
                f"duplicate document ID: {document.doc_id!r}"
            )
        seen_doc_ids.add(document.doc_id)

    # Canonical order makes the complete rebuild independent of input order.
    document_rows.sort(key=lambda document: document.doc_id)
    # Normalize build inputs before checking whether this is an exact retry.
    normalized_embedding_config = _json_mapping(
        embedding_config, "embedding_config"
    )
    normalized_sparse_config = _json_mapping(
        sparse_config, "sparse_config"
    )

    if not normalized_embedding_config:
        raise ValueError("embedding_config must not be empty")
    if not normalized_sparse_config:
        raise ValueError("sparse_config must not be empty")

    normalized_model_id = _require_nonempty_string(
        embedding_model_id, "embedding_model_id"
    )
    normalized_model_revision = _require_nonempty_string(
        embedding_model_revision, "embedding_model_revision"
    )

    if not _REVISION_RE.fullmatch(normalized_model_revision):
        raise ValueError(
            "embedding_model_revision must be an immutable hexadecimal commit"
        )

        # Verify that the embedder identity matches the declared build identity.
    # This reads metadata only; it does not invoke embedding inference.
    verified_identity = _verified_document_embedder_identity(embed_fn)

    if verified_identity["model_id"] != normalized_model_id:
        raise ValueError(
            "declared embedding model ID does not match verified embedder"
        )

    if verified_identity["model_revision"] != normalized_model_revision.lower():
        raise ValueError(
            "declared embedding revision does not match verified embedder"
        )

        # query_prefix belongs to query-side configuration, not document encoding.
    document_embedding_config = dict(normalized_embedding_config)
    query_prefix = document_embedding_config.pop("query_prefix", "")

    if not isinstance(query_prefix, str):
        raise ValueError(
            "embedding_config.query_prefix must be a string"
        )

    if canonical_json(verified_identity["config"]) != canonical_json(
        document_embedding_config
    ):
        raise ValueError(
            "declared document embedding config does not match verified embedder"
        )

    operation_metadata = _build_operation_metadata(
        "rebuild",
        {
            "documents": [
                {
                    "doc_id": document.doc_id,
                    "revision_sha256": document.revision_sha256,
                }
                for document in document_rows
            ],
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model_id": verified_identity["model_id"],
            "embedding_model_revision": verified_identity["model_revision"],
            "embedding_config": normalized_embedding_config,
            "sparse_config": normalized_sparse_config,
        },
    )

    existing_snapshot = _load_existing_operation_snapshot(
        index_root,
        normalized_index_id,
        operation_metadata,
    )

    if existing_snapshot is not None:
        return existing_snapshot
    # Always regenerate the complete chunk collection.
    chunk_rows: list[Chunk] = []

    for document in document_rows:
        chunk_rows.extend(
            chunk_text(
                document.doc_id,
                document.text,
                size=chunk_size,
                overlap=chunk_overlap,
                source=document.source,
                title=document.title,
                metadata=document.metadata,
                document_revision_sha256=document.revision_sha256,
            )
        )

    # Empty documents are retained, but an index needs searchable chunks.
    if not chunk_rows:
        raise ValueError(
            "cannot build a persistent index without searchable chunks"
        )

    return build_and_activate_index_snapshot(
        index_root,
        index_id=normalized_index_id,
        documents=document_rows,
        chunks=chunk_rows,
        embed_fn=embed_fn,
        embedding_model_id=verified_identity["model_id"],
        embedding_model_revision=verified_identity["model_revision"],
        embedding_config=normalized_embedding_config,
        sparse_config=sparse_config,
        chunking_config=None,
        created_at_utc=created_at_utc,
        expected_embedding=expected_embedding,
        operation_metadata=operation_metadata,
    )


def _verified_document_embedder_identity(
    embed_fn: Callable[[list[str]], object],
) -> dict[str, Any]:
    """Return a model identity proven by the embedder, not caller declarations."""
    if not callable(embed_fn):
        raise TypeError("embed_fn must be callable")

    if hasattr(embed_fn, "verified_embedding_identity"):
        identity = embed_fn.verified_embedding_identity
        if identity is None:
            raise ValueError(
                "embed_fn does not expose a verified embedding identity")
    else:
        metadata = getattr(embed_fn, "model_metadata", None)
        config = getattr(embed_fn, "embedding_config", None)
        if not isinstance(metadata, Mapping):
            raise ValueError("embed_fn is missing verified model metadata")
        if metadata.get("revision_verified") is not True:
            raise ValueError("embed_fn immutable model revision is unverified")
        identity = {
            "model_id": metadata.get("model_id"),
            "model_revision": metadata.get("resolved_revision"),
            "config": config,
        }

    if not isinstance(identity, Mapping):
        raise ValueError("embed_fn identity must be a mapping")
    model_id = _require_nonempty_string(
        identity.get("model_id"), "embed_fn.model_id")
    revision = _require_nonempty_string(
        identity.get("model_revision"), "embed_fn.model_revision")
    if not _REVISION_RE.fullmatch(revision):
        raise ValueError(
            "embed_fn model revision must be a verified immutable commit")
    config = _json_mapping(identity.get("config"), "embed_fn.embedding_config")
    if not config:
        raise ValueError("embed_fn embedding_config must not be empty")
    return {
        "model_id": model_id,
        "model_revision": revision.lower(),
        "config": config,
    }


def _validate_insert_embedding_compatibility(
    snapshot: LoadedIndexSnapshot,
    embedder_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate document-embedder identity against the active snapshot."""
    embedding = snapshot.manifest.get("embedding")
    if not isinstance(embedding, Mapping):
        raise ValueError("active snapshot is missing embedding identity")
    stored_config = _json_mapping(
        embedding.get("config"), "snapshot.embedding.config")
    stored_query_prefix = stored_config.pop("query_prefix", "")
    if not isinstance(stored_query_prefix, str):
        raise ValueError("snapshot query_prefix must be a string")

    stored_model_id = _require_nonempty_string(
        embedding.get("model_id"), "snapshot.embedding.model_id")
    stored_revision = _require_nonempty_string(
        embedding.get("model_revision"), "snapshot.embedding.model_revision")
    if (
        embedder_identity["model_id"] != stored_model_id
        or embedder_identity["model_revision"] != stored_revision.lower()
    ):
        raise ValueError(
            "document embedder model ID or immutable revision does not match snapshot")
    if canonical_json(embedder_identity["config"]) != canonical_json(stored_config):
        raise ValueError(
            "document embedder configuration does not match snapshot")

    vector_metadata = snapshot.manifest.get("vectors")
    if not isinstance(vector_metadata, Mapping):
        raise ValueError("active snapshot is missing vector metadata")
    stored_dimension = vector_metadata.get("dimension")
    configured_dimension = stored_config.get("dimension")
    if configured_dimension is not None and (
        isinstance(configured_dimension, bool)
        or not isinstance(configured_dimension, int)
        or configured_dimension <= 0
        or configured_dimension != stored_dimension
    ):
        raise ValueError(
            "embedding configuration dimension does not match snapshot vectors")

    # Preserve the complete stored configuration, including query_prefix.
    return {
        "model_id": stored_model_id,
        "model_revision": stored_revision,
        "config": _json_mapping(
            embedding.get("config"), "snapshot.embedding.config"),
    }


def insert_document_and_activate_index_snapshot(
    index_root: str | Path,
    *,
    index_id: str,
    document: Document,
    embed_fn: Callable[[list[str]], object],
    created_at_utc: str | None = None,
) -> LoadedIndexSnapshot:
    """Insert one new document while reusing every existing dense vector.

    Only chunks belonging to the inserted document are embedded. Existing
    vectors are remapped by chunk ID into canonical document/chunk order and
    written into a new immutable snapshot before atomic activation.

    Like snapshot activation, this operation assumes a single writer; it does
    not provide cross-process writer locking.
    """
    normalized_index_id = _validate_index_id(index_id)
    if not isinstance(document, Document):
        raise TypeError("document must be a Document")
    if not callable(embed_fn):
        raise TypeError("embed_fn must be callable")

    identity = _verified_document_embedder_identity(embed_fn)

    operation_metadata = _build_operation_metadata(
        "insert",
        {
            "document_revision_sha256": document.revision_sha256,
            "embedding": identity,
        },
    )

    existing_snapshot = _load_existing_operation_snapshot(
        index_root,
        normalized_index_id,
        operation_metadata,
    )

    if existing_snapshot is not None:
        return existing_snapshot

    active = load_current_index_snapshot(index_root)
    if any(existing.doc_id == document.doc_id for existing in active.documents):
        raise ValueError(f"duplicate document ID: {document.doc_id!r}")

    version_dir = Path(index_root) / "versions" / normalized_index_id
    if version_dir.exists() or version_dir.is_symlink():
        raise FileExistsError(
            f"immutable index version already exists: {normalized_index_id}")

    expected_embedding = _validate_insert_embedding_compatibility(
        active, identity)

    chunk_size, chunk_overlap = _chunking_parameters(
        active.manifest["chunking"])
    new_chunks = chunk_text(
        document.doc_id,
        document.text,
        size=chunk_size,
        overlap=chunk_overlap,
        source=document.source,
        title=document.title,
        metadata=document.metadata,
        document_revision_sha256=document.revision_sha256,
    )

    new_vectors = np.empty((0, active.vectors.shape[1]), dtype=np.float32)
    if new_chunks:
        try:
            raw_new_vectors = embed_fn([chunk.text for chunk in new_chunks])
        except Exception as exc:
            raise RuntimeError(
                "embedding generation failed while inserting document") from exc
        new_vectors = _normalize_embedding_matrix(
            raw_new_vectors, expected_rows=len(new_chunks))
        if new_vectors.shape[1] != active.vectors.shape[1]:
            raise ValueError(
                "new document embedding dimension does not match existing vectors")

    documents = sorted(
        [*active.documents, document], key=lambda item: item.doc_id)
    chunks_by_document: dict[str, list[Chunk]] = {
        item.doc_id: [] for item in documents
    }
    vector_by_chunk_id: dict[str, np.ndarray] = {
        chunk.chunk_id: active.vectors[row]
        for row, chunk in enumerate(active.chunks)
    }
    for chunk in active.chunks:
        chunks_by_document[chunk.doc_id].append(chunk)

    for row, chunk in enumerate(new_chunks):
        if chunk.chunk_id in vector_by_chunk_id:
            raise ValueError(
                f"duplicate chunk ID after insert: {chunk.chunk_id!r}")
        vector_by_chunk_id[chunk.chunk_id] = new_vectors[row]
        chunks_by_document[document.doc_id].append(chunk)

    chunks = [
        chunk
        for item in documents
        for chunk in chunks_by_document[item.doc_id]
    ]
    if not chunks:
        raise ValueError(
            "cannot insert into an index without searchable chunks")

    try:
        ordered_vectors = np.stack(
            [vector_by_chunk_id[chunk.chunk_id] for chunk in chunks], axis=0
        ).astype(np.float32, copy=False)
    except KeyError as exc:
        raise ValueError(
            f"missing vector for chunk {exc.args[0]!r} during insert") from exc

    return build_and_activate_index_snapshot(
        index_root,
        index_id=normalized_index_id,
        documents=documents,
        chunks=chunks,
        embed_fn=None,
        embedding_model_id=expected_embedding["model_id"],
        embedding_model_revision=expected_embedding["model_revision"],
        embedding_config=expected_embedding["config"],
        sparse_config=active.manifest["sparse"],
        chunking_config=active.manifest["chunking"],
        created_at_utc=created_at_utc,
        expected_embedding=expected_embedding,
        precomputed_vectors=ordered_vectors,
        operation_metadata=operation_metadata,
    )


def update_document_and_activate_index_snapshot(
    index_root: str | Path,
    *,
    index_id: str,
    document: Document,
    embed_fn: Callable[[list[str]], object],
    created_at_utc: str | None = None,
) -> LoadedIndexSnapshot:
    """Replace one existing document in a new immutable index snapshot.

    Only the replacement document's chunks are embedded. Vectors belonging
    to all other documents are reused exactly as stored in the active
    validated snapshot.

    The complete replacement is built and activated as one snapshot.
    Like the other index mutation operations, this assumes a single writer
    and does not provide cross-process writer locking.
    """
    normalized_index_id = _validate_index_id(index_id)

    if not isinstance(document, Document):
        raise TypeError("document must be a Document")

    if not callable(embed_fn):
        raise TypeError("embed_fn must be callable")

    identity = _verified_document_embedder_identity(embed_fn)
    operation_metadata = _build_operation_metadata(
        "update",
        {
            "document_revision_sha256": document.revision_sha256,
            "embedding": identity,
        },
    )

    existing_snapshot = _load_existing_operation_snapshot(
        index_root,
        normalized_index_id,
        operation_metadata,
    )
    if existing_snapshot is not None:
        return existing_snapshot

    active = load_current_index_snapshot(index_root)

    existing_document_ids = {
        item.doc_id for item in active.documents
    }

    if document.doc_id not in existing_document_ids:
        raise ValueError(
            f"document ID {document.doc_id!r} does not exist in active index"
        )

    version_dir = (
        Path(index_root) / "versions" / normalized_index_id
    )

    if version_dir.exists() or version_dir.is_symlink():
        raise FileExistsError(
            f"immutable index version already exists: {normalized_index_id}"
        )

    expected_embedding = _validate_insert_embedding_compatibility(
        active,
        identity,
    )

    chunk_size, chunk_overlap = _chunking_parameters(
        active.manifest["chunking"]
    )

    new_chunks = chunk_text(
        document.doc_id,
        document.text,
        size=chunk_size,
        overlap=chunk_overlap,
        source=document.source,
        title=document.title,
        metadata=document.metadata,
        document_revision_sha256=document.revision_sha256,
    )

    # Preserve all chunks except those belonging to the replaced document.
    retained_chunks = [
        chunk
        for chunk in active.chunks
        if chunk.doc_id != document.doc_id
    ]

    # The index must still contain searchable content after the update.
    # Reject this before invoking the embedding model.
    if not retained_chunks and not new_chunks:
        raise ValueError(
            "update would leave the index without searchable chunks"
        )

    new_vectors = np.empty(
        (0, active.vectors.shape[1]),
        dtype=np.float32,
    )

    if new_chunks:
        try:
            raw_new_vectors = embed_fn(
                [chunk.text for chunk in new_chunks]
            )
        except Exception as exc:
            raise RuntimeError(
                "embedding generation failed while updating document"
            ) from exc

        new_vectors = _normalize_embedding_matrix(
            raw_new_vectors,
            expected_rows=len(new_chunks),
        )

        if new_vectors.shape[1] != active.vectors.shape[1]:
            raise ValueError(
                "updated document embedding dimension does not "
                "match existing vectors"
            )

    # Replace the original document record and restore deterministic order.
    documents = [
        document if item.doc_id == document.doc_id else item
        for item in active.documents
    ]
    documents.sort(key=lambda item: item.doc_id)

    chunks_by_document: dict[str, list[Chunk]] = {
        item.doc_id: [] for item in documents
    }

    # Only unchanged chunks retain their original persisted vectors.
    vector_by_chunk_id: dict[str, np.ndarray] = {
        chunk.chunk_id: active.vectors[row]
        for row, chunk in enumerate(active.chunks)
        if chunk.doc_id != document.doc_id
    }

    for chunk in retained_chunks:
        chunks_by_document[chunk.doc_id].append(chunk)

    for row, chunk in enumerate(new_chunks):
        if chunk.chunk_id in vector_by_chunk_id:
            raise ValueError(
                f"duplicate chunk ID after update: {chunk.chunk_id!r}"
            )

        vector_by_chunk_id[chunk.chunk_id] = new_vectors[row]
        chunks_by_document[document.doc_id].append(chunk)

    # Construct canonical document/chunk ordering.
    chunks = [
        chunk
        for item in documents
        for chunk in chunks_by_document[item.doc_id]
    ]

    if not chunks:
        raise ValueError(
            "updated index must contain at least one searchable chunk"
        )

    try:
        ordered_vectors = np.stack(
            [
                vector_by_chunk_id[chunk.chunk_id]
                for chunk in chunks
            ],
            axis=0,
        ).astype(np.float32, copy=False)
    except KeyError as exc:
        raise ValueError(
            f"missing vector for chunk {exc.args[0]!r} during update"
        ) from exc

    return build_and_activate_index_snapshot(
        index_root,
        index_id=normalized_index_id,
        documents=documents,
        chunks=chunks,
        embed_fn=None,
        embedding_model_id=expected_embedding["model_id"],
        embedding_model_revision=expected_embedding["model_revision"],
        embedding_config=expected_embedding["config"],
        sparse_config=active.manifest["sparse"],
        chunking_config=active.manifest["chunking"],
        created_at_utc=created_at_utc,
        expected_embedding=expected_embedding,
        precomputed_vectors=ordered_vectors,
        operation_metadata=operation_metadata,
    )


def delete_document_and_activate_index_snapshot(
    index_root: str | Path,
    *,
    index_id: str,
    doc_id: str,
    created_at_utc: str | None = None,
) -> LoadedIndexSnapshot:
    """Delete one document using a new immutable index snapshot.

    Remaining documents, chunks, and their persisted vectors are reused
    without invoking an embedding model.

    The previous snapshot remains available for rollback. Like the other
    index mutation operations, this assumes a single writer and does not
    provide cross-process writer locking.
    """
    normalized_index_id = _validate_index_id(index_id)
    normalized_doc_id = _require_nonempty_string(doc_id, "doc_id")

    operation_metadata = _build_operation_metadata(
        "delete",
        {"doc_id": normalized_doc_id},
    )
    existing_snapshot = _load_existing_operation_snapshot(
        index_root,
        normalized_index_id,
        operation_metadata,
    )
    if existing_snapshot is not None:
        return existing_snapshot

    active = load_current_index_snapshot(index_root)

    existing_document_ids = {
        item.doc_id for item in active.documents
    }

    if normalized_doc_id not in existing_document_ids:
        raise ValueError(
            f"document ID {normalized_doc_id!r} does not exist in active index"
        )

    version_dir = (
        Path(index_root) / "versions" / normalized_index_id
    )

    if version_dir.exists() or version_dir.is_symlink():
        raise FileExistsError(
            f"immutable index version already exists: {normalized_index_id}"
        )

    # Retain every document except the requested deletion target.
    documents = [
        document
        for document in active.documents
        if document.doc_id != normalized_doc_id
    ]

    # Select chunks and their corresponding persisted vector rows together.
    retained_rows = [
        row
        for row, chunk in enumerate(active.chunks)
        if chunk.doc_id != normalized_doc_id
    ]

    chunks = [
        active.chunks[row]
        for row in retained_rows
    ]

    # Do not activate an index without searchable content.
    if not chunks:
        raise ValueError(
            "delete would leave the index without searchable chunks"
        )

    # Advanced indexing preserves the exact values and row alignment.
    # No embedding model invocation is required.
    ordered_vectors = active.vectors[retained_rows].astype(
        np.float32,
        copy=False,
    )

    embedding = active.manifest["embedding"]

    return build_and_activate_index_snapshot(
        index_root,
        index_id=normalized_index_id,
        documents=documents,
        chunks=chunks,
        embed_fn=None,
        embedding_model_id=embedding["model_id"],
        embedding_model_revision=embedding["model_revision"],
        embedding_config=embedding["config"],
        sparse_config=active.manifest["sparse"],
        chunking_config=active.manifest["chunking"],
        created_at_utc=created_at_utc,
        expected_embedding=embedding,
        precomputed_vectors=ordered_vectors,
        operation_metadata=operation_metadata,
    )
