
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import hs.index_store as store
from hs.chunking import chunk_text
from hs.documents import Document
from hs.index_store import build_index_snapshot


MODEL_ID = "test/fake-embedder"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

EMBEDDING_CONFIG = {
    "adapter": "test-deterministic-embedder-v1",
    "normalize_embeddings": True,
    "dimension": 3,
}

SPARSE_CONFIG = {
    "algorithm": "bm25",
    "version": 1,
    "k1": 1.5,
    "b": 0.75,
}

CREATED_AT = "2026-11-03T00:00:00+00:00"


def _corpus(index_id: str, text: str = "alpha beta gamma delta"):
    document = Document(
        index_id, text, source=f"{index_id}.txt", title=index_id)

    chunks = chunk_text(
        document.doc_id,
        document.text,
        size=2,
        overlap=0,
        source=document.source,
        title=document.title,
        metadata=document.metadata,
        document_revision_sha256=document.revision_sha256,
    )

    return [document], chunks


def _embedder(calls=None):
    def embed(texts):
        if calls is not None:
            calls.append(list(texts))

        return np.asarray(
            [[float(index + 1), 1.0, 0.0] for index, _ in enumerate(texts)],
            dtype=np.float32,
        )

    return embed


def _build_args(index_id: str, documents, chunks, embed_fn):
    return {
        "index_id": index_id,
        "documents": documents,
        "chunks": chunks,
        "embed_fn": embed_fn,
        "embedding_model_id": MODEL_ID,
        "embedding_model_revision": MODEL_REVISION,
        "embedding_config": EMBEDDING_CONFIG,
        "sparse_config": SPARSE_CONFIG,
        "created_at_utc": CREATED_AT,
    }


def _build_staged(
    root: Path,
    index_id: str,
    text: str = "alpha beta gamma delta",
):
    stage_parent = root / "staging" / f"manual-{index_id}"
    stage_parent.mkdir(parents=True, exist_ok=True)

    documents, chunks = _corpus(index_id, text)
    staged_snapshot = stage_parent / "snapshot"

    build_index_snapshot(
        staged_snapshot,
        **_build_args(index_id, documents, chunks, _embedder()),
    )

    return staged_snapshot


def _activate(
    root: Path,
    index_id: str,
    text: str = "alpha beta gamma delta",
):
    documents, chunks = _corpus(index_id, text)

    return store.build_and_activate_index_snapshot(
        root,
        **_build_args(index_id, documents, chunks, _embedder()),
    )


def test_build_and_activate_publishes_a_valid_snapshot_and_current_pointer(
    tmp_path,
):
    root = tmp_path / "indexes"
    calls = []
    documents, chunks = _corpus("idx-one")

    activated = store.build_and_activate_index_snapshot(
        root,
        **_build_args("idx-one", documents, chunks, _embedder(calls)),
    )

    assert activated.manifest["index_id"] == "idx-one"
    assert activated.path == root / "versions" / "idx-one"
    assert len(calls) == 1
    assert len(calls[0]) == len(chunks)
    assert (root / "CURRENT.json").is_file()
    assert not list((root / "staging").iterdir())

    pointer_bytes = (root / "CURRENT.json").read_bytes()

    assert pointer_bytes.endswith(b"\n")
    assert not pointer_bytes.endswith(b"\r\n")

    pointer = json.loads(pointer_bytes.decode("utf-8"))
    manifest_bytes = (activated.path / "manifest.json").read_bytes()

    assert pointer == {
        "format": "enterprise-hybrid-search-current",
        "schema_version": 1,
        "index_id": "idx-one",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "previous_index_id": None,
        "previous_manifest_sha256": None,
    }

    loaded = store.load_current_index_snapshot(root)
    assert loaded.manifest["index_id"] == "idx-one"


def test_loading_current_index_without_pointer_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="no index is active"):
        store.load_current_index_snapshot(tmp_path / "indexes")


def test_second_activation_records_previous_and_rollback_toggles_versions(
    tmp_path,
):
    root = tmp_path / "indexes"

    first = _activate(root, "idx-one", "alpha beta gamma")
    second = _activate(root, "idx-two", "delta epsilon zeta")

    assert first.path.is_dir()
    assert second.path.is_dir()

    assert (
        store.load_current_index_snapshot(root).manifest["index_id"]
        == "idx-two"
    )

    rolled_back = store.rollback_index_snapshot(root)

    assert rolled_back.manifest["index_id"] == "idx-one"
    assert (
        store.load_current_index_snapshot(root).manifest["index_id"]
        == "idx-one"
    )

    rolled_forward = store.rollback_index_snapshot(root)

    assert rolled_forward.manifest["index_id"] == "idx-two"
    assert (
        store.load_current_index_snapshot(root).manifest["index_id"]
        == "idx-two"
    )

    assert (root / "versions" / "idx-one").is_dir()
    assert (root / "versions" / "idx-two").is_dir()


def test_activation_rejects_snapshot_outside_staging(tmp_path):
    root = tmp_path / "indexes"
    outside = tmp_path / "outside" / "snapshot"

    documents, chunks = _corpus("idx-outside")

    build_index_snapshot(
        outside,
        **_build_args("idx-outside", documents, chunks, _embedder()),
    )

    with pytest.raises(ValueError, match="inside index_root/staging"):
        store.activate_index_snapshot(root, outside)

    assert not (root / "CURRENT.json").exists()
    assert outside.is_dir()


def test_invalid_staged_snapshot_does_not_replace_current_pointer(tmp_path):
    root = tmp_path / "indexes"

    _activate(root, "idx-one")
    pointer_before = (root / "CURRENT.json").read_bytes()

    staged = _build_staged(root, "idx-two")

    with (staged / "vectors.npy").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="(size|SHA-256) mismatch"):
        store.activate_index_snapshot(root, staged)

    assert (root / "CURRENT.json").read_bytes() == pointer_before
    assert not (root / "versions" / "idx-two").exists()

    assert (
        store.load_current_index_snapshot(root).manifest["index_id"]
        == "idx-one"
    )


def test_existing_immutable_version_is_never_overwritten(tmp_path):
    root = tmp_path / "indexes"

    first = _activate(root, "idx-one", "alpha beta gamma")
    first_manifest = (first.path / "manifest.json").read_bytes()

    staged = _build_staged(
        root,
        "idx-one",
        "different document content",
    )

    pointer_before = (root / "CURRENT.json").read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        store.activate_index_snapshot(root, staged)

    assert (
        (root / "versions" / "idx-one" / "manifest.json").read_bytes()
        == first_manifest
    )

    assert (root / "CURRENT.json").read_bytes() == pointer_before


def test_pointer_replace_failure_preserves_old_active_index_and_keeps_new_version(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "indexes"

    _activate(root, "idx-one")

    pointer_before = (root / "CURRENT.json").read_bytes()
    staged = _build_staged(root, "idx-two", "delta epsilon zeta")

    real_replace = store.os.replace

    def fail_current_replace(source, destination):
        if Path(destination).name == "CURRENT.json":
            raise OSError("simulated pointer replacement failure")

        return real_replace(source, destination)

    monkeypatch.setattr(store.os, "replace", fail_current_replace)

    with pytest.raises(OSError, match="simulated pointer replacement failure"):
        store.activate_index_snapshot(root, staged)

    assert (root / "CURRENT.json").read_bytes() == pointer_before

    assert (
        store.load_current_index_snapshot(root).manifest["index_id"]
        == "idx-one"
    )

    assert (root / "versions" / "idx-two").is_dir()

    monkeypatch.setattr(store.os, "replace", real_replace)

    recovered = store.rollback_index_snapshot(
        root,
        index_id="idx-two",
    )

    assert recovered.manifest["index_id"] == "idx-two"

    assert (
        store.load_current_index_snapshot(root).manifest["index_id"]
        == "idx-two"
    )


def test_rollback_without_previous_version_is_rejected(tmp_path):
    root = tmp_path / "indexes"

    _activate(root, "idx-one")

    with pytest.raises(ValueError, match="no previous index"):
        store.rollback_index_snapshot(root)


def test_explicit_rollback_rejects_missing_version(tmp_path):
    root = tmp_path / "indexes"

    _activate(root, "idx-one")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        store.rollback_index_snapshot(root, index_id="idx-missing")

    assert (
        store.load_current_index_snapshot(root).manifest["index_id"]
        == "idx-one"
    )


def test_current_pointer_manifest_checksum_mismatch_is_rejected(tmp_path):
    root = tmp_path / "indexes"

    _activate(root, "idx-one")

    manifest_path = root / "versions" / "idx-one" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["created_at_utc"] = "2026-11-03T00:01:00+00:00"

    manifest_path.write_text(
        store.canonical_json(manifest) + "\n",
        encoding="utf-8",
        newline="",
    )

    with pytest.raises(ValueError, match="manifest checksum"):
        store.load_current_index_snapshot(root)


def test_malformed_current_pointer_is_rejected_before_activation(tmp_path):
    root = tmp_path / "indexes"

    _activate(root, "idx-one")

    pointer_path = root / "CURRENT.json"
    malformed_pointer = b'{"index_id":"idx-one","index_id":"shadow"}\n'

    pointer_path.write_bytes(malformed_pointer)
    staged = _build_staged(root, "idx-two")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        store.activate_index_snapshot(root, staged)

    assert not (root / "versions" / "idx-two").exists()
    assert pointer_path.read_bytes() == malformed_pointer

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        store.load_current_index_snapshot(root)


def test_activation_with_embedding_mismatch_does_not_promote_snapshot(
    tmp_path,
):
    root = tmp_path / "indexes"
    staged = _build_staged(root, "idx-one")

    wrong_identity = {
        "model_id": MODEL_ID,
        "model_revision": "0" * 40,
        "config": EMBEDDING_CONFIG,
    }

    with pytest.raises(ValueError, match="incompatible"):
        store.activate_index_snapshot(
            root,
            staged,
            expected_embedding=wrong_identity,
        )

    assert not (root / "versions" / "idx-one").exists()
    assert not (root / "CURRENT.json").exists()
    assert staged.is_dir()


def test_build_failure_leaves_no_active_pointer_or_staging_directory(
    tmp_path,
):
    root = tmp_path / "indexes"
    documents, chunks = _corpus("idx-failed")

    def broken_embedder(texts):
        raise RuntimeError("model unavailable")

    with pytest.raises(RuntimeError, match="embedding generation failed"):
        store.build_and_activate_index_snapshot(
            root,
            **_build_args("idx-failed", documents, chunks, broken_embedder),
        )

    assert not (root / "CURRENT.json").exists()
    assert not list((root / "staging").iterdir())
    assert not list((root / "versions").iterdir())


def test_build_and_activate_rejects_existing_version_before_embedding(
    tmp_path,
):
    root = tmp_path / "indexes"

    _activate(root, "idx-one")

    calls = []
    documents, chunks = _corpus("idx-one", "changed content")

    with pytest.raises(FileExistsError, match="already exists"):
        store.build_and_activate_index_snapshot(
            root,
            **_build_args("idx-one", documents, chunks, _embedder(calls)),
        )

    assert calls == []

    assert (
        store.load_current_index_snapshot(root).manifest["index_id"]
        == "idx-one"
    )


def test_rollback_recovers_previous_snapshot_when_active_snapshot_is_corrupt(
    tmp_path,
):
    root = tmp_path / "indexes"

    _activate(root, "idx-one", "alpha beta gamma")
    _activate(root, "idx-two", "delta epsilon zeta")

    # Simulate corruption of the currently active snapshot after activation.
    active_vectors = root / "versions" / "idx-two" / "vectors.npy"

    with active_vectors.open("ab") as handle:
        handle.write(b"corrupt-active-artifact")

    # Recovery validates the rollback target, not the broken active snapshot.
    recovered = store.rollback_index_snapshot(root)

    assert recovered.manifest["index_id"] == "idx-one"

    assert (
        store.load_current_index_snapshot(root).manifest["index_id"]
        == "idx-one"
    )

    assert (root / "versions" / "idx-two").is_dir()


def test_index_id_rejects_trailing_period():
    with pytest.raises(ValueError, match="must not end with a period"):
        store._validate_index_id("idx.")


def test_active_snapshot_loads_in_fresh_python_process(tmp_path):
    root = tmp_path / "indexes"

    documents, chunks = _corpus("idx-fresh-process")

    store.build_and_activate_index_snapshot(
        root,
        **_build_args(
            "idx-fresh-process",
            documents,
            chunks,
            _embedder(),
        ),
    )

    child_script = r'''
import json
import sys

import numpy as np

from hs.index_store import load_current_index_snapshot
from hs.pipeline import HybridPipeline


MODEL_ID = "test/fake-embedder"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

EMBEDDING_CONFIG = {
    "adapter": "test-deterministic-embedder-v1",
    "normalize_embeddings": True,
    "dimension": 3,
}


def embed(texts):
    return np.asarray(
        [[float(index + 1), 1.0, 0.0] for index, _ in enumerate(texts)],
        dtype=np.float32,
    )


embed.model_metadata = {
    "model_id": MODEL_ID,
    "resolved_revision": MODEL_REVISION,
    "revision_verified": True,
}

embed.embedding_config = EMBEDDING_CONFIG

snapshot = load_current_index_snapshot(sys.argv[1])

pipeline = HybridPipeline.from_snapshot(
    snapshot,
    query_embed_fn=embed,
)

counts = {}

for mode in ("bm25", "dense", "hybrid"):
    results = pipeline.search("alpha beta", mode=mode, k=3)

    if not results:
        raise AssertionError(f"{mode} returned no results")

    counts[mode] = len(results)


print(
    json.dumps(
        {
            "index_id": snapshot.manifest["index_id"],
            "counts": counts,
        }
    )
)
'''

    project_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            child_script,
            str(root.resolve()),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )

    result = json.loads(completed.stdout)

    assert result["index_id"] == "idx-fresh-process"
    assert set(result["counts"]) == {"bm25", "dense", "hybrid"}
    assert all(count > 0 for count in result["counts"].values())
