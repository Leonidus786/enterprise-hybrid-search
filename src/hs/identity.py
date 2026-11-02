"""Deterministic SHA-256 identities for ingested documents and chunks."""
from __future__ import annotations

import hashlib
import json
from typing import Any

IDENTITY_SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data deterministically for fingerprinting."""
    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        serialized.encode("utf-8")
        return serialized
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("identity payload must contain valid JSON-compatible values") from exc


def sha256_text(text: str) -> str:
    """Return the SHA-256 digest of exact UTF-8 text (no normalization)."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    try:
        payload = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("text must contain valid Unicode") from exc
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    """Return the SHA-256 digest of a canonical JSON representation."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
