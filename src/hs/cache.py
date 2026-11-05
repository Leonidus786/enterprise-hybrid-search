
"""Persistent, model-specific cache for text embeddings."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np


class CachedEmbedder:
    """Cache individual text embeddings while preserving input row order.

    The historical three-argument constructor remains supported. Persistent
    model caches should also provide a resolved immutable revision and every
    vector-affecting embedding setting through ``embedding_config``.
    """

    def __init__(
        self,
        embed_fn: Callable[[list[str]], np.ndarray],
        cache_dir: str | Path,
        model_id: str,
        *,
        model_revision: str | None = None,
        embedding_config: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if not callable(embed_fn):
            raise TypeError("embed_fn must be callable")
        if model_revision is not None and (
            not isinstance(model_revision, str) or not model_revision.strip()
        ):
            raise ValueError("model_revision must be a non-empty string or None")
        if embedding_config is not None and not isinstance(embedding_config, Mapping):
            raise TypeError("embedding_config must be a mapping or None")

        # Serialize config once so nested mappings use a deterministic identity.
        # Reject NaN/Infinity because they are not valid, portable JSON values.
        try:
            config_json = (
                json.dumps(
                    dict(embedding_config),
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                if embedding_config is not None
                else None
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("embedding_config must contain JSON-safe values") from exc

        self.embed_fn = embed_fn
        self.cache_dir = Path(cache_dir)
        self.model_id = model_id
        self.model_revision = model_revision.strip() if model_revision is not None else None
        self.embedding_config = json.loads(config_json) if config_json is not None else None

        # The original three-argument constructor retains its historical key
        # format. New callers get an isolated namespace for every model revision
        # and vector-affecting configuration.
        self._cache_identity = None
        if self.model_revision is not None or config_json is not None:
            self._cache_identity = json.dumps(
                {
                    "cache_schema_version": 2,
                    "model_id": self.model_id,
                    "model_revision": self.model_revision,
                    "embedding_config": self.embedding_config,
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )

        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict[str, int]:
        """Return cache hit/miss counts accumulated across calls."""
        return {"hits": self._hits, "misses": self._misses}

    @property
    def verified_embedding_identity(self) -> dict[str, object] | None:
        """Expose identity only when the wrapped model and cache config agree.

        Constructor arguments are declarations. Require verified model-loader
        metadata and a matching configuration attached by the actual embedder.
        """
        metadata = getattr(self.embed_fn, "model_metadata", None)
        actual_config = getattr(self.embed_fn, "embedding_config", None)
        if not isinstance(metadata, Mapping):
            return None
        if metadata.get("revision_verified") is not True:
            return None
        if metadata.get("model_id") != self.model_id:
            return None
        revision = metadata.get("resolved_revision")
        if (
            not isinstance(revision, str)
            or not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision)
            or self.model_revision is None
            or revision.lower() != self.model_revision.lower()
        ):
            return None
        if not isinstance(actual_config, Mapping) or self.embedding_config is None:
            return None
        try:
            actual_json = json.dumps(
                dict(actual_config), sort_keys=True, ensure_ascii=False,
                separators=(",", ":"), allow_nan=False,
            )
            declared_json = json.dumps(
                self.embedding_config, sort_keys=True, ensure_ascii=False,
                separators=(",", ":"), allow_nan=False,
            )
        except (TypeError, ValueError):
            return None
        if actual_json != declared_json:
            return None
        return {
            "model_id": self.model_id,
            "model_revision": revision.lower(),
            "config": dict(self.embedding_config),
        }

    def _path_for(self, text: str) -> Path:
        # Legacy callers retain the original key format. Revision-aware callers
        # hash the complete, canonical model/config identity with the exact text.
        if self._cache_identity is None:
            payload = f"{self.model_id}\0{text}".encode("utf-8")
        else:
            payload = self._cache_identity.encode("utf-8") + b"\0" + text.encode("utf-8")
        key = hashlib.sha256(payload).hexdigest()
        return self.cache_dir / f"{key}.npy"

    @staticmethod
    def _remove(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Cleanup failure should not mask the original cache problem.
            pass

    def _read(self, path: Path) -> np.ndarray | None:
        if not path.is_file():
            return None

        try:
            vector = np.asarray(
                np.load(path, allow_pickle=False),
                dtype=np.float32,
            )

            if (
                vector.ndim != 1
                or vector.size == 0
                or not np.isfinite(vector).all()
            ):
                raise ValueError("Invalid cached embedding")

            return vector

        except (OSError, ValueError, EOFError, TypeError):
            # Ignore and remove corrupted or invalid cache entries.
            self._remove(path)
            return None

    @staticmethod
    def _validate_batch(
        raw: object,
        expected_rows: int,
    ) -> np.ndarray:
        try:
            matrix = np.asarray(raw, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Embedder must return a numeric 2D matrix"
            ) from exc

        if (
            matrix.ndim != 2
            or matrix.shape[0] != expected_rows
            or matrix.shape[1] == 0
        ):
            raise ValueError(
                "Embedding matrix must have shape "
                f"({expected_rows}, dimension>0); got {matrix.shape}"
            )

        if not np.isfinite(matrix).all():
            raise ValueError("Embeddings must contain only finite values")

        return matrix

    def _embed(self, texts: list[str]) -> np.ndarray:
        raw = self.embed_fn(texts)
        return self._validate_batch(raw, expected_rows=len(texts))

    def _write(self, path: Path, vector: np.ndarray) -> None:
        """Atomically write one embedding to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)

        fd, temp_name = tempfile.mkstemp(
            prefix="embedding-",
            suffix=".tmp",
            dir=path.parent,
        )
        temp_path = Path(temp_name)

        try:
            with os.fdopen(fd, "wb") as handle:
                np.save(handle, vector, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temp_path, path)

        finally:
            self._remove(temp_path)

    def __call__(self, texts: Iterable[str]) -> np.ndarray:
        requested = list(texts)

        if any(not isinstance(text, str) for text in requested):
            raise TypeError("All texts must be strings")

        if not requested:
            return np.empty((0, 0), dtype=np.float32)

        # Preserve first-seen order and avoid embedding duplicates twice.
        unique_texts = list(dict.fromkeys(requested))

        cached: dict[str, np.ndarray] = {}
        missing: list[str] = []

        for text in unique_texts:
            vector = self._read(self._path_for(text))

            if vector is None:
                missing.append(text)
            else:
                cached[text] = vector

        # Cached vectors with inconsistent dimensions cannot be combined.
        dimensions = {vector.shape[0] for vector in cached.values()}

        if len(dimensions) > 1:
            for text in cached:
                self._remove(self._path_for(text))

            cached.clear()
            missing = unique_texts.copy()

        new_vectors: dict[str, np.ndarray] = {}

        if missing:
            matrix = self._embed(missing)

            # Check the newly computed vectors against cached dimensions.
            cached_dimensions = {
                vector.shape[0] for vector in cached.values()
            }

            if cached_dimensions and matrix.shape[1] not in cached_dimensions:
                # The stored vectors no longer match this embedder's output.
                for text in cached:
                    self._remove(self._path_for(text))

                cached.clear()
                missing = unique_texts.copy()
                matrix = self._embed(missing)

            new_vectors = {
                text: matrix[index].copy()
                for index, text in enumerate(missing)
            }

        # Count each unique requested text once per successful call.
        self._hits += len(cached)
        self._misses += len(missing)

        # Validate the entire embedding batch before writing any new entries.
        for text, vector in new_vectors.items():
            self._write(self._path_for(text), vector)

        vectors = {**cached, **new_vectors}

        # Reconstruct the exact requested order, including duplicate rows.
        return np.stack(
            [vectors[text] for text in requested],
            axis=0,
        ).astype(np.float32, copy=False)
