"""Real model loaders with optional immutable Hugging Face revisions.

Model loading remains lazy so unit tests do not require torch or HF downloads.
"""
from __future__ import annotations

import re
from typing import Any

_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")


def _extract_resolved_revision(model: Any) -> str | None:
    """Best-effort extraction of the actual model snapshot commit, offline."""
    objects: list[Any] = [model]
    inner_model = getattr(model, "model", None)
    if inner_model is not None:
        objects.append(inner_model)
    modules = getattr(model, "_modules", None)
    if isinstance(modules, dict) or hasattr(modules, "values"):
        try:
            objects.extend(list(modules.values()))
        except (AttributeError, TypeError):
            pass

    configs: list[Any] = []
    tokenizers: list[Any] = []
    for obj in objects:
        if obj is None:
            continue
        config = getattr(obj, "config", None)
        if config is not None:
            configs.append(config)
        auto_model = getattr(obj, "auto_model", None)
        if auto_model is not None and getattr(auto_model, "config", None) is not None:
            configs.append(auto_model.config)
        tokenizer = getattr(obj, "tokenizer", None)
        if tokenizer is not None:
            tokenizers.append(tokenizer)

    for config in configs:
        commit = getattr(config, "_commit_hash", None)
        if isinstance(commit, str) and _COMMIT_RE.fullmatch(commit):
            return commit
    for tokenizer in tokenizers:
        init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
        commit = init_kwargs.get("_commit_hash") if isinstance(init_kwargs, dict) else None
        if isinstance(commit, str) and _COMMIT_RE.fullmatch(commit):
            return commit
    return None


def _attach_metadata(function, *, model_id: str, requested_revision: str | None, model: Any):
    resolved_revision = _extract_resolved_revision(model)
    if resolved_revision is None and isinstance(requested_revision, str) and _COMMIT_RE.fullmatch(requested_revision):
        resolved_revision = requested_revision
    function.model_metadata = {
        "model_id": model_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "revision_status": "resolved_commit" if resolved_revision else "unknown_not_exposed_by_loader",
    }
    return function


def sentence_embedder(name="BAAI/bge-small-en-v1.5", revision=None):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name, **({"revision": revision} if revision else {}))

    def embed(texts):
        return model.encode(
            texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True
        )

    return _attach_metadata(
        embed, model_id=name, requested_revision=revision, model=model
    )


def cross_encoder_scorer(name="cross-encoder/ms-marco-MiniLM-L-6-v2", revision=None):
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(name, **({"revision": revision} if revision else {}))

    def score(pairs):
        return model.predict(pairs).tolist()

    return _attach_metadata(
        score, model_id=name, requested_revision=revision, model=model
    )
