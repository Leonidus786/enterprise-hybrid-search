
"""Persistent, model-specific cache for text embeddings."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np


class CachedEmbedder:
    """Cache individual text embeddings while preserving input row order."""

    def __init__(
        self,
        embed_fn: Callable[[list[str]], np.ndarray],
        cache_dir: str | Path,
        model_id: str,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if not callable(embed_fn):
            raise TypeError("embed_fn must be callable")

        self.embed_fn = embed_fn
        self.cache_dir = Path(cache_dir)
        self.model_id = model_id
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict[str, int]:
        """Return cache hit/miss counts accumulated across calls."""
        return {"hits": self._hits, "misses": self._misses}

    def _path_for(self, text: str) -> Path:
        # Both model identity and exact text determine the cache key.
        key = hashlib.sha256(
            f"{self.model_id}\0{text}".encode("utf-8")
        ).hexdigest()
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
