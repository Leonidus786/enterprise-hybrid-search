from dataclasses import dataclass, field
from typing import Any

from .documents import normalize_metadata, normalize_optional_text
from .identity import IDENTITY_SCHEMA_VERSION, sha256_json, sha256_text


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    source: str | None = field(default=None, compare=False, hash=False)
    title: str | None = field(default=None, compare=False, hash=False)
    metadata: dict[str, Any] = field(
        default_factory=dict, compare=False, hash=False
    )
    document_revision_sha256: str | None = field(
        default=None, compare=False, hash=False
    )
    chunking_config: dict[str, Any] = field(
        default_factory=dict, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        # Keep the original first three positional fields and equality semantics.
        if self.document_revision_sha256 is not None:
            revision = self.document_revision_sha256
            if (
                not isinstance(revision, str)
                or len(revision) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in revision)
            ):
                raise ValueError("document_revision_sha256 must be a SHA-256 digest or None")
            object.__setattr__(self, "document_revision_sha256", revision.lower())

        object.__setattr__(
            self, "source", normalize_optional_text(self.source, field_name="source")
        )
        object.__setattr__(
            self, "title", normalize_optional_text(self.title, field_name="title")
        )
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))
        object.__setattr__(
            self, "chunking_config", normalize_metadata(self.chunking_config)
        )

    @property
    def content_sha256(self) -> str:
        """SHA-256 of the exact normalized chunk text."""
        return sha256_text(self.text)

    @property
    def revision_sha256(self) -> str:
        """SHA-256 including document revision and chunking configuration."""
        return sha256_json(
            {
                "identity_type": "hs.chunk",
                "identity_schema_version": IDENTITY_SCHEMA_VERSION,
                "doc_id": self.doc_id,
                "chunk_id": self.chunk_id,
                "document_revision_sha256": self.document_revision_sha256,
                "content_sha256": self.content_sha256,
                "source": self.source,
                "title": self.title,
                "metadata": normalize_metadata(self.metadata),
                "chunking_config": normalize_metadata(self.chunking_config),
            }
        )


def chunk_text(
    doc_id,
    text,
    size=200,
    overlap=40,
    *,
    source=None,
    title=None,
    metadata=None,
    document_revision_sha256=None,
):
    """Word-window chunking with inherited metadata and stable revision hashes."""
    if size is None:
        normalized_text = " ".join(text.split())
        if not normalized_text:
            return []
        chunking_config = {"algorithm": "whole_document", "version": 1}
        return [
            Chunk(
                chunk_id=f"{doc_id}#0",
                doc_id=doc_id,
                text=normalized_text,
                source=source,
                title=title,
                metadata={} if metadata is None else metadata,
                document_revision_sha256=document_revision_sha256,
                chunking_config=chunking_config,
            )
        ]

    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("need size > 0 and 0 <= overlap < size")

    chunking_config = {
        "algorithm": "word_window",
        "version": 1,
        "size": size,
        "overlap": overlap,
    }
    words = text.split()
    chunks, i, n, step = [], 0, 0, size - overlap
    while i < len(words):
        chunks.append(
            Chunk(
                f"{doc_id}#{n}",
                doc_id,
                " ".join(words[i:i + size]),
                source=source,
                title=title,
                metadata={} if metadata is None else metadata,
                document_revision_sha256=document_revision_sha256,
                chunking_config=chunking_config,
            )
        )
        n += 1
        if i + size >= len(words):
            break
        i += step
    return chunks
