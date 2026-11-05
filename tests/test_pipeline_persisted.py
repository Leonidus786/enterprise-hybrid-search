from __future__ import annotations

import hashlib
import re
import sys
import types

import numpy as np
import pytest

from hs.cache import CachedEmbedder
from hs.chunking import chunk_text
from hs.documents import Document
from hs.index_store import build_index_snapshot, load_index_snapshot
from hs.models import sentence_embedder
from hs.pipeline import HybridPipeline


MODEL_ID = "test/deterministic-embedder"
MODEL_REVISION = "a" * 40
PREFIX = "query:"
EMBEDDING_CONFIG = {
    "adapter": "test-deterministic-embedder-v1",
    "normalize_embeddings": True,
    "dimension": 16,
}
SPARSE_CONFIG = {
    "algorithm": "bm25",
    "version": 1,
    "k1": 1.5,
    "b": 0.75,
}


def _vectors(texts):
    result = np.zeros((len(texts), 16), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in re.findall(r"\w+", text.lower()):
            index = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % 16
            result[row, index] += 1.0
    return result


def _query_embedder(
    *,
    model_id=MODEL_ID,
    revision=MODEL_REVISION,
    config=None,
    revision_verified=True,
    calls=None,
):
    def embed(texts):
        if calls is not None:
            calls.extend(texts)
        return _vectors(texts)

    embed.model_metadata = {
        "model_id": model_id,
        "resolved_revision": revision,
        "revision_verified": revision_verified,
    }
    embed.embedding_config = dict(EMBEDDING_CONFIG if config is None else config)
    return embed


def _build_snapshot(tmp_path, *, sparse_config=SPARSE_CONFIG):
    documents = [
        Document("doc-k8s", "kubernetes pods scheduling nodes"),
        Document("doc-bread", "sourdough bread baking starter"),
        Document("doc-ml", "gradient descent neural networks"),
    ]
    chunks = []
    for document in documents:
        chunks.extend(
            chunk_text(
                document.doc_id,
                document.text,
                size=None,
                overlap=0,
                source=document.source,
                title=document.title,
                metadata=document.metadata,
                document_revision_sha256=document.revision_sha256,
            )
        )

    document_calls = []

    def document_embedder(texts):
        document_calls.append(list(texts))
        return _vectors(texts)

    snapshot_dir = tmp_path / "snapshot"
    embedding_config = {**EMBEDDING_CONFIG, "query_prefix": PREFIX}
    build_index_snapshot(
        snapshot_dir,
        index_id="pipeline-persisted-test",
        documents=documents,
        chunks=chunks,
        embed_fn=document_embedder,
        embedding_model_id=MODEL_ID,
        embedding_model_revision=MODEL_REVISION,
        embedding_config=embedding_config,
        sparse_config=sparse_config,
        created_at_utc="2026-11-02T00:00:00+00:00",
    )
    assert len(document_calls) == 1
    return load_index_snapshot(snapshot_dir), chunks, document_calls


def test_fresh_and_persisted_pipeline_have_exact_ranking_parity(tmp_path):
    snapshot, chunks, document_calls = _build_snapshot(tmp_path)
    query_calls = []
    query_embedder = _query_embedder(calls=query_calls)
    fresh = HybridPipeline(
        chunks,
        _vectors,
        query_embed_fn=query_embedder,
        query_prefix=PREFIX,
    )

    query = "kubernetes pods"
    modes = ("bm25", "dense", "hybrid")
    fresh_rankings = {
        mode: fresh.search(query, mode=mode, k=len(chunks))
        for mode in modes
    }
    calls_before_load = list(query_calls)
    document_calls_before_load = list(document_calls)

    loaded = HybridPipeline.from_snapshot(
        snapshot,
        query_embed_fn=query_embedder,
        query_prefix=PREFIX,
    )

    # Constructing a persisted pipeline must not call either the query adapter
    # or the builder's document embedder.
    assert query_calls == calls_before_load
    assert document_calls == document_calls_before_load

    loaded_rankings = {
        mode: loaded.search(query, mode=mode, k=len(chunks))
        for mode in modes
    }
    assert loaded_rankings == fresh_rankings


def test_persisted_pipeline_embeds_queries_but_not_documents_during_load(tmp_path):
    snapshot, _, _ = _build_snapshot(tmp_path)
    query_calls = []
    query_embedder = _query_embedder(calls=query_calls)

    loaded = HybridPipeline.from_snapshot(
        snapshot,
        query_embed_fn=query_embedder,
        query_prefix=PREFIX,
    )
    assert query_calls == []

    loaded.search("kubernetes pods", mode="dense", k=2)
    assert query_calls == [PREFIX + "kubernetes pods"]


@pytest.mark.parametrize(
    "identity_kwargs",
    [
        {"model_id": "test/other-model"},
        {"revision": "b" * 40},
        {"config": {**EMBEDDING_CONFIG, "normalize_embeddings": False}},
    ],
    ids=["different-model", "different-revision", "different-config"],
)
def test_persisted_pipeline_rejects_incompatible_embedding_identity(
    tmp_path, identity_kwargs
):
    snapshot, _, _ = _build_snapshot(tmp_path)
    with pytest.raises(ValueError, match="does not match snapshot"):
        HybridPipeline.from_snapshot(
            snapshot,
            query_embed_fn=_query_embedder(**identity_kwargs),
            query_prefix=PREFIX,
        )


def test_persisted_pipeline_rejects_unverified_or_missing_identity(tmp_path):
    snapshot, _, _ = _build_snapshot(tmp_path)
    with pytest.raises(ValueError, match="unverified"):
        HybridPipeline.from_snapshot(
            snapshot,
            query_embed_fn=_query_embedder(revision_verified=False),
            query_prefix=PREFIX,
        )

    def unmarked_embedder(texts):
        return _vectors(texts)

    with pytest.raises(ValueError, match="missing verified model metadata"):
        HybridPipeline.from_snapshot(
            snapshot,
            query_embed_fn=unmarked_embedder,
            query_prefix=PREFIX,
        )

    unconfigured_embedder = _query_embedder()
    del unconfigured_embedder.embedding_config
    with pytest.raises(ValueError, match="embedding_config"):
        HybridPipeline.from_snapshot(
            snapshot,
            query_embed_fn=unconfigured_embedder,
            query_prefix=PREFIX,
        )


def test_persisted_pipeline_rejects_query_prefix_mismatch(tmp_path):
    snapshot, _, _ = _build_snapshot(tmp_path)
    with pytest.raises(ValueError, match="query_prefix"):
        HybridPipeline.from_snapshot(
            snapshot,
            query_embed_fn=_query_embedder(),
            query_prefix="",
        )


def test_persisted_pipeline_rejects_unsupported_sparse_configuration(tmp_path):
    snapshot, _, _ = _build_snapshot(
        tmp_path,
        sparse_config={
            "algorithm": "bm25",
            "version": 1,
            "k1": 1.5,
            "b": 0.75,
            "tokenizer": "unsupported",
        },
    )
    with pytest.raises(ValueError, match="unsupported BM25"):
        HybridPipeline.from_snapshot(
            snapshot,
            query_embed_fn=_query_embedder(),
            query_prefix=PREFIX,
        )


def test_persisted_pipeline_rejects_non_integer_sparse_version(tmp_path):
    snapshot, _, _ = _build_snapshot(
        tmp_path,
        sparse_config={
            "algorithm": "bm25",
            "version": 1.0,
            "k1": 1.5,
            "b": 0.75,
        },
    )
    with pytest.raises(ValueError, match="unsupported sparse index algorithm or version"):
        HybridPipeline.from_snapshot(
            snapshot,
            query_embed_fn=_query_embedder(),
            query_prefix=PREFIX,
        )


def test_persisted_pipeline_preserves_optional_reranker(tmp_path):
    snapshot, chunks, _ = _build_snapshot(tmp_path)

    def score_fn(pairs):
        return [
            float(len(set(query.lower().split()) & set(text.lower().split())))
            for query, text in pairs
        ]

    query_embedder = _query_embedder()
    fresh = HybridPipeline(
        chunks,
        _vectors,
        score_fn=score_fn,
        query_embed_fn=query_embedder,
        query_prefix=PREFIX,
    )
    loaded = HybridPipeline.from_snapshot(
        snapshot,
        query_embed_fn=query_embedder,
        query_prefix=PREFIX,
        score_fn=score_fn,
    )
    assert loaded.search("kubernetes pods", mode="hybrid_rerank", k=3) == fresh.search(
        "kubernetes pods", mode="hybrid_rerank", k=3
    )


def test_cached_embedder_identity_requires_wrapped_model_and_config(tmp_path):
    wrapped = _query_embedder()
    cached = CachedEmbedder(
        wrapped,
        tmp_path / "cache",
        MODEL_ID,
        model_revision=MODEL_REVISION,
        embedding_config=EMBEDDING_CONFIG,
    )
    assert cached.verified_embedding_identity == {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "config": EMBEDDING_CONFIG,
    }

    def metadata_only(texts):
        return _vectors(texts)

    metadata_only.model_metadata = dict(wrapped.model_metadata)
    untrusted = CachedEmbedder(
        metadata_only,
        tmp_path / "other-cache",
        MODEL_ID,
        model_revision=MODEL_REVISION,
        embedding_config=EMBEDDING_CONFIG,
    )
    assert untrusted.verified_embedding_identity is None

    mismatched = _query_embedder(config={**EMBEDDING_CONFIG, "dimension": 8})
    wrong_config = CachedEmbedder(
        mismatched,
        tmp_path / "third-cache",
        MODEL_ID,
        model_revision=MODEL_REVISION,
        embedding_config=EMBEDDING_CONFIG,
    )
    assert wrong_config.verified_embedding_identity is None

    wrong_model = _query_embedder(model_id="test/other-model")
    mismatched_model = CachedEmbedder(
        wrong_model,
        tmp_path / "fourth-cache",
        MODEL_ID,
        model_revision=MODEL_REVISION,
        embedding_config=EMBEDDING_CONFIG,
    )
    assert mismatched_model.verified_embedding_identity is None

    unverified = _query_embedder(revision_verified=False)
    unverified_cache = CachedEmbedder(
        unverified,
        tmp_path / "fifth-cache",
        MODEL_ID,
        model_revision=MODEL_REVISION,
        embedding_config=EMBEDDING_CONFIG,
    )
    assert unverified_cache.verified_embedding_identity is None


def test_sentence_embedder_declares_its_vector_configuration(monkeypatch):
    class FakeModel:
        config = types.SimpleNamespace(_commit_hash=MODEL_REVISION)

        def encode(self, texts, *, normalize_embeddings, batch_size, show_progress_bar):
            assert normalize_embeddings is True
            assert batch_size == 64
            assert show_progress_bar is True
            return _vectors(texts)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = lambda name, **kwargs: FakeModel()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    embed = sentence_embedder(name=MODEL_ID, revision=MODEL_REVISION)
    assert embed.embedding_config == {
        "adapter": "hs.models.sentence_embedder",
        "config_version": 1,
        "normalize_embeddings": True,
        "batch_size": 64,
    }
    assert embed.model_metadata["revision_verified"] is True
