"""Validated document records and JSON-compatible metadata helpers."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
from typing import Any

from .identity import IDENTITY_SCHEMA_VERSION, sha256_json, sha256_text


def normalize_optional_text(value: object, *, field_name: str) -> str | None:
    """Normalize an optional descriptive field; blank strings become None."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f'"{field_name}" must be a string or null')
    normalized = value.strip()
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f'"{field_name}" must contain valid Unicode text') from exc
    return normalized or None


def _normalize_json_value(value: Any, *, path: str) -> Any:
    """Copy and validate one value using the JSON data model."""
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(f"{path} contains invalid Unicode text") from exc
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or Infinity")
        return value

    if isinstance(value, list):
        return [
            _normalize_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{path} keys must be non-empty strings")
            try:
                key.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(f"{path} contains an invalid Unicode key") from exc
            normalized[key] = _normalize_json_value(item, path=f"{path}.{key}")
        return normalized

    raise ValueError(
        f"{path} contains unsupported value type {type(value).__name__}; "
        "metadata must use JSON-compatible values"
    )


def normalize_metadata(value: object) -> dict[str, Any]:
    """Return an independent JSON metadata copy; ``None`` becomes an empty object."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError('"metadata" must be an object')

    normalized = _normalize_json_value(value, path="metadata")
    try:
        json.dumps(
            normalized,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError('"metadata" must contain valid JSON-compatible values') from exc
    return normalized


@dataclass(frozen=True)
class Document:
    """A source document before chunking, with deterministic SHA-256 identities.

    ``doc_id`` and ``text`` remain the legacy required fields. Optional source,
    title and JSON-compatible metadata are included in the revision fingerprint.
    """

    doc_id: str
    text: str
    source: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.doc_id, str) or not self.doc_id.strip():
            raise ValueError('"doc_id" must be a non-empty string')
        if not isinstance(self.text, str):
            raise ValueError('"text" must be a string')
        try:
            self.doc_id.encode("utf-8")
            self.text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("document ID and text must contain valid Unicode") from exc

        object.__setattr__(
            self, "source", normalize_optional_text(self.source, field_name="source")
        )
        object.__setattr__(
            self, "title", normalize_optional_text(self.title, field_name="title")
        )
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))

    @property
    def content_sha256(self) -> str:
        """SHA-256 of exact source text; whitespace is intentionally significant."""
        return sha256_text(self.text)

    @property
    def revision_sha256(self) -> str:
        """SHA-256 of document identity, content and normalized metadata snapshot."""
        return sha256_json(
            {
                "identity_type": "hs.document",
                "identity_schema_version": IDENTITY_SCHEMA_VERSION,
                "doc_id": self.doc_id,
                "content_sha256": self.content_sha256,
                "source": self.source,
                "title": self.title,
                "metadata": normalize_metadata(self.metadata),
            }
        )
