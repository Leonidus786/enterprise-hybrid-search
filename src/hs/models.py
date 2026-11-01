"""Real model loaders with optional immutable Hugging Face revisions.

Model loading remains lazy so unit tests do not require torch or HF downloads.
"""
from __future__ import annotations
from pathlib import Path
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
        commit = init_kwargs.get("_commit_hash") if isinstance(
            init_kwargs, dict) else None
        if isinstance(commit, str) and _COMMIT_RE.fullmatch(commit):
            return commit
    return None


def _attach_metadata(
    function,
    *,
    model_id: str,
    requested_revision: str | None,
    model: Any,
):
    loaded_revision = _extract_resolved_revision(model)

    requested_pinned = (
        isinstance(requested_revision, str)
        and _COMMIT_RE.fullmatch(requested_revision) is not None
    )

    verified_from_local_snapshot = False

    if loaded_revision is not None:
        resolved_revision = loaded_revision

        if requested_pinned:
            verified_from_loaded_model = (
                loaded_revision.lower() == requested_revision.lower()
            )
            revision_status = (
                "verified_commit_match"
                if verified_from_loaded_model
                else "revision_mismatch"
            )
        else:
            verified_from_loaded_model = True
            revision_status = "loaded_model_revision_exposed"

    elif requested_pinned:
        # Fallback: verify the cached snapshot directory without network access.
        try:
            from huggingface_hub import snapshot_download

            snapshot_path = snapshot_download(
                repo_id=model_id,
                revision=requested_revision,
                local_files_only=True,
            )
            snapshot_revision = Path(snapshot_path).name
        except Exception:
            snapshot_revision = None

        if (
            isinstance(snapshot_revision, str)
            and _COMMIT_RE.fullmatch(snapshot_revision)
        ):
            resolved_revision = snapshot_revision
            verified_from_local_snapshot = (
                snapshot_revision.lower() == requested_revision.lower()
            )
            verified_from_loaded_model = False
            revision_status = (
                "verified_cached_snapshot_match"
                if verified_from_local_snapshot
                else "cached_snapshot_revision_mismatch"
            )
        else:
            resolved_revision = None
            verified_from_loaded_model = False
            revision_status = "requested_commit_unverified"

    else:
        resolved_revision = None
        verified_from_loaded_model = False
        revision_status = "unknown_not_exposed_by_loader"

    revision_verified = (
        verified_from_loaded_model or verified_from_local_snapshot
    )

    function.model_metadata = {
        "model_id": model_id,
        "requested_revision": requested_revision,
        "requested_pinned": requested_pinned,
        "resolved_revision": resolved_revision,
        "verified_from_loaded_model": verified_from_loaded_model,
        "verified_from_local_snapshot": verified_from_local_snapshot,
        "revision_verified": revision_verified,
        "revision_status": revision_status,
    }
    return function


def sentence_embedder(name="BAAI/bge-small-en-v1.5", revision=None):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        name, **({"revision": revision} if revision else {}))

    def embed(texts):
        return model.encode(
            texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True
        )

    return _attach_metadata(
        embed, model_id=name, requested_revision=revision, model=model
    )


def cross_encoder_scorer(
    name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    revision=None,
    *,
    batch_size: int = 8,
    device: str | None = None,
    max_length: int | None = None,
):
    """Create a configurable CrossEncoder scorer.

    ``batch_size=8`` is a provisional CPU-oriented default based on the
    project's exploratory 3-query profiling. Re-benchmark on the deployment
    hardware before treating it as optimal. ``device=None`` lets
    Sentence Transformers select the device; ``max_length=None`` preserves
    the model's own configured maximum sequence length.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if max_length is not None and (
        isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1
    ):
        raise ValueError("max_length must be a positive integer or None")
    if device is not None and (not isinstance(device, str) or not device.strip()):
        raise ValueError("device must be a non-empty string or None")

    from sentence_transformers import CrossEncoder

    model_kwargs = {}
    if revision:
        model_kwargs["revision"] = revision
    if device is not None:
        model_kwargs["device"] = device
    if max_length is not None:
        model_kwargs["max_length"] = max_length

    model = CrossEncoder(name, **model_kwargs)

    def score(pairs):
        predictions = model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)

    score.inference_config = {
        "batch_size": batch_size,
        "device": device,
        "max_length": max_length,
    }
    return _attach_metadata(
        score, model_id=name, requested_revision=revision, model=model
    )
