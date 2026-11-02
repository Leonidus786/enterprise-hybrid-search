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


def _json_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    try:
        # canonical_json rejects NaN/Infinity and non-JSON Python objects.
        return json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain JSON-compatible values") from exc


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
            raise ValueError(f"JSONL record #{index} is not JSON-compatible") from exc
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
    raise ValueError(f"non-standard JSON numeric constant is not allowed: {value}")


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
        raise ValueError(f"JSON value in {context} is not JSON-compatible") from exc
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
            raise ValueError("JSONL payload must use LF, not CRLF, line endings")
        if not line.strip():
            raise ValueError(f"JSONL contains a blank line at line {line_number}")
        try:
            rows.append(
                _decode_json_object(line, context=f"JSONL record at line {line_number}")
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
            raise ValueError(f"artifacts.{name}.record_count does not match corpus")
    elif expected_shape is not None:
        if set(result) != {"sha256", "size_bytes", "shape", "dtype"}:
            raise ValueError(f"artifacts.{name} has an invalid schema")
        shape = result["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in shape)
        ):
            raise ValueError(f"artifacts.{name}.shape must be a two-integer list")
        if tuple(shape) != expected_shape:
            raise ValueError(f"artifacts.{name}.shape does not match vectors metadata")
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
) -> dict[str, Any]:
    """Build and validate manifest schema v1 from already-written artifacts."""
    if created_at_utc is None:
        created_at_utc = datetime.now(timezone.utc).isoformat()
    if not isinstance(artifacts, Mapping):
        raise ValueError("artifacts must be a mapping")
    manifest = {
        "format": INDEX_FORMAT,
        "schema_version": INDEX_SCHEMA_VERSION,
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
    return validate_manifest(manifest)


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_embedding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate manifest v1 strictly and return an independent JSON-safe copy."""
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a JSON object")
    required = {
        "format", "schema_version", "index_id", "created_at_utc", "corpus",
        "embedding", "vectors", "chunking", "sparse", "artifacts",
    }
    if set(manifest) != required:
        raise ValueError("manifest has missing or unsupported fields")
    if manifest["format"] != INDEX_FORMAT:
        raise ValueError("unsupported index format")
    version = manifest["schema_version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != INDEX_SCHEMA_VERSION
    ):
        raise ValueError("unsupported index schema version")

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
        raise ValueError("created_at_utc must be an ISO-8601 timestamp") from exc
    if parsed_time.tzinfo is None or parsed_time.utcoffset() != timedelta(0):
        raise ValueError("created_at_utc must include a UTC timezone")

    corpus = manifest["corpus"]
    if not isinstance(corpus, Mapping) or set(corpus) != {
        "document_count", "chunk_count", "fingerprint_sha256"
    }:
        raise ValueError("manifest.corpus has an invalid schema")
    document_count = _require_int(corpus["document_count"], "corpus.document_count", minimum=1)
    chunk_count = _require_int(corpus["chunk_count"], "corpus.chunk_count", minimum=1)
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
        raise ValueError("embedding.model_revision must be an immutable hexadecimal commit")
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
                expected_embedding.get("model_id"), "expected_embedding.model_id"
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
            raise ValueError("index embedding identity is incompatible with requested model")

    vectors = manifest["vectors"]
    if not isinstance(vectors, Mapping) or set(vectors) != {
        "rows", "dimension", "dtype", "normalized"
    }:
        raise ValueError("manifest.vectors has an invalid schema")
    rows = _require_int(vectors["rows"], "vectors.rows", minimum=1)
    dimension = _require_int(vectors["dimension"], "vectors.dimension", minimum=1)
    if rows != chunk_count:
        raise ValueError("vectors.rows must equal corpus.chunk_count")
    if vectors["dtype"] != "float32":
        raise ValueError("vectors.dtype must be float32")
    if vectors["normalized"] is not True:
        raise ValueError("vectors.normalized must be true")

    chunking = _json_mapping(manifest["chunking"], "chunking")
    sparse = _json_mapping(manifest["sparse"], "sparse")
    if not chunking or not sparse:
        raise ValueError("chunking and sparse configurations must not be empty")

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(REQUIRED_ARTIFACTS):
        raise ValueError("manifest.artifacts must contain exactly the required artifacts")
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
        "schema_version": INDEX_SCHEMA_VERSION,
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
    # Ensure manifest is fully JSON serializable and detached from caller objects.
    return json.loads(canonical_json(normalized))


def _index_chunking_config(chunks: list[Chunk]) -> dict[str, Any]:
    """Return the one chunking configuration represented by all chunks."""
    if not chunks:
        raise ValueError("cannot build a persistent index without searchable chunks")

    configs = [normalize_metadata(chunk.chunking_config) for chunk in chunks]
    if any(not config for config in configs):
        raise ValueError("every persisted chunk must declare its chunking_config")

    encoded = {canonical_json(config) for config in configs}
    if len(encoded) != 1:
        raise ValueError("all chunks in one index must use the same chunking_config")
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
        raise ValueError("cannot build a persistent index from an empty document set")
    if not chunks:
        raise ValueError("cannot build a persistent index without searchable chunks")
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
            raise ValueError("chunking_config does not match the persisted chunks")

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
    actual_records = [canonical_json(chunk_to_record(chunk)) for chunk in chunks]
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
        raise ValueError("normalized embeddings must contain only finite values")
    return np.ascontiguousarray(normalized, dtype=np.float32)


def _write_bytes_durable(path: Path, payload: bytes) -> None:
    """Create one immutable artifact file and flush its contents to disk."""
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


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
    embed_fn: Callable[[list[str]], object],
    embedding_model_id: str,
    embedding_model_revision: str,
    embedding_config: Mapping[str, Any],
    sparse_config: Mapping[str, Any],
    chunking_config: Mapping[str, Any] | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a new on-disk index snapshot and return its validated manifest.

    The destination must not already exist. Document and chunk records retain
    caller order; vector row ``i`` is generated from chunk row ``i``. The
    manifest is written last, so an interrupted build cannot look complete to
    the future loader. This function does not activate a snapshot or implement
    ``CURRENT.json`` switching.
    """
    if not callable(embed_fn):
        raise TypeError("embed_fn must be callable")

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
        raise FileExistsError(f"index snapshot destination already exists: {target}")

    texts = [chunk.text for chunk in chunk_rows]
    try:
        raw_vectors = embed_fn(texts)
    except Exception as exc:
        raise RuntimeError("embedding generation failed while building index") from exc
    vectors = _normalize_embedding_matrix(
        raw_vectors, expected_rows=len(chunk_rows)
    )

    document_payload = encode_jsonl(
        document_to_record(document) for document in document_rows
    )
    chunk_payload = encode_jsonl(chunk_to_record(chunk) for chunk in chunk_rows)

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
        raise FileNotFoundError(f"missing or unsafe index manifest: {path.name}")
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
        raise FileNotFoundError("index snapshot directory does not exist or is unsafe")

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
            raise ValueError(f"snapshot artifact must be a regular file: {entry.name}")


def _verify_artifact_bytes(snapshot_dir: Path, manifest: Mapping[str, Any]) -> None:
    for name in REQUIRED_ARTIFACTS:
        path = snapshot_dir / name
        expected = manifest["artifacts"][name]
        try:
            actual = artifact_metadata(path)
        except OSError as exc:
            raise OSError(f"cannot read snapshot artifact {name}: {exc}") from exc
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
                    raise ValueError(f"{path.name} must use LF, not CRLF, line endings")
                if not raw.strip():
                    raise ValueError(
                        f"{path.name} contains a blank line at line {line_number}"
                    )
                records.append(
                    _decode_json_object(raw, context=f"{path.name} line {line_number}")
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
        raise ValueError("vectors.npy contains vector norms that overflow float32")
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
        manifest = validate_manifest(manifest, expected_embedding=expected_embedding)

    _verify_artifact_bytes(path, manifest)

    document_records = _read_jsonl_records(path / DOCUMENTS_ARTIFACT)
    chunk_records = _read_jsonl_records(path / CHUNKS_ARTIFACT)
    if len(document_records) != manifest["corpus"]["document_count"]:
        raise ValueError("documents.jsonl record count does not match manifest")
    if len(chunk_records) != manifest["corpus"]["chunk_count"]:
        raise ValueError("chunks.jsonl record count does not match manifest")

    documents = tuple(document_from_record(record) for record in document_records)
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
