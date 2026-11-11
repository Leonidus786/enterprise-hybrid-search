
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import hs.index_store as store
from hs.documents import Document


MODEL_ID = "test/lifecycle-embedder"
MODEL_REVISION = "a" * 40

EMBEDDING_CONFIG = {
    "adapter": "test-lifecycle-token-hash-v1",
    "normalize_embeddings": True,
    "dimension": 16,
}

SPARSE_CONFIG = {
    "algorithm": "bm25",
    "version": 1,
    "k1": 1.5,
    "b": 0.75,
}

CHUNK_SIZE = 2
CHUNK_OVERLAP = 0
CREATED_AT = "2026-11-11T00:00:00+00:00"


def _vectors(texts: list[str]) -> np.ndarray:
    """Create deterministic, offline token-hash vectors."""
    result = np.zeros(
        (len(texts), EMBEDDING_CONFIG["dimension"]),
        dtype=np.float32,
    )

    for row, text in enumerate(texts):
        for token in re.findall(r"\w+", text.lower()):
            index = (
                int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
                % EMBEDDING_CONFIG["dimension"]
            )
            result[row, index] += 1.0

    return result


def _embedder(calls: list[list[str]] | None = None):
    def embed(texts: list[str]) -> np.ndarray:
        if calls is not None:
            calls.append(list(texts))
        return _vectors(texts)

    embed.model_metadata = {
        "model_id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "revision_verified": True,
    }
    embed.embedding_config = dict(EMBEDDING_CONFIG)

    return embed


def _document(
    doc_id: str,
    text: str,
    *,
    revision: int,
) -> Document:
    return Document(
        doc_id=doc_id,
        text=text,
        source=f"{doc_id}-v{revision}.txt",
        title=f"{doc_id} revision {revision}",
        metadata={"revision": revision},
    )


def _assert_snapshot_integrity(
    snapshot,
    expected_document_ids: set[str],
) -> None:
    document_ids = [document.doc_id for document in snapshot.documents]
    chunk_ids = [chunk.chunk_id for chunk in snapshot.chunks]
    chunk_document_ids = {chunk.doc_id for chunk in snapshot.chunks}

    assert set(document_ids) == expected_document_ids
    assert len(document_ids) == len(set(document_ids))

    assert len(chunk_ids) == len(set(chunk_ids))
    assert chunk_document_ids <= expected_document_ids

    assert snapshot.vectors.ndim == 2
    assert snapshot.vectors.shape[0] == len(snapshot.chunks)
    assert snapshot.vectors.shape[1] == EMBEDDING_CONFIG["dimension"]


def _assert_vectors_preserved(
    before,
    after,
    *,
    include_document,
) -> None:
    after_rows = {
        chunk.chunk_id: row
        for row, chunk in enumerate(after.chunks)
    }

    for row, chunk in enumerate(before.chunks):
        if not include_document(chunk.doc_id):
            continue

        assert chunk.chunk_id in after_rows

        np.testing.assert_array_equal(
            before.vectors[row],
            after.vectors[after_rows[chunk.chunk_id]],
        )


def _snapshot_files(snapshot_path: str | Path) -> dict[str, bytes]:
    root = Path(snapshot_path)

    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_complete_rebuild_insert_update_delete_and_fresh_process_search(
    tmp_path,
):
    root = tmp_path / "indexes"

    # ---------------------------------------------------------
    # 1. Initial rebuild: documents A and B
    # ---------------------------------------------------------
    document_a_v1 = _document(
        "doc-a",
        "alpha beta gamma delta",
        revision=1,
    )
    document_b = _document(
        "doc-b",
        "bravo charlie delta echo",
        revision=1,
    )

    rebuild_calls: list[list[str]] = []

    initial = store.rebuild_and_activate_index_snapshot(
        root,
        index_id="idx-lifecycle-00-rebuild",
        documents=[document_a_v1, document_b],
        embed_fn=_embedder(rebuild_calls),
        embedding_model_id=MODEL_ID,
        embedding_model_revision=MODEL_REVISION,
        embedding_config=EMBEDDING_CONFIG,
        sparse_config=SPARSE_CONFIG,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        created_at_utc=CREATED_AT,
    )

    _assert_snapshot_integrity(
        initial,
        {"doc-a", "doc-b"},
    )

    assert initial.manifest["index_id"] == "idx-lifecycle-00-rebuild"
    assert rebuild_calls == [
        [chunk.text for chunk in initial.chunks]
    ]

    initial_a_chunks = {
        chunk.chunk_id: chunk
        for chunk in initial.chunks
        if chunk.doc_id == "doc-a"
    }

    assert initial_a_chunks

    # Preserve a byte-for-byte record of this immutable version.
    preserved_versions = [
        (initial.path, _snapshot_files(initial.path))
    ]

    # ---------------------------------------------------------
    # 2. Insert document C
    # ---------------------------------------------------------
    document_c = _document(
        "doc-c",
        "kubernetes pods scheduling nodes",
        revision=1,
    )

    insert_calls: list[list[str]] = []

    inserted = store.insert_document_and_activate_index_snapshot(
        root,
        index_id="idx-lifecycle-01-insert",
        document=document_c,
        embed_fn=_embedder(insert_calls),
    )

    _assert_snapshot_integrity(
        inserted,
        {"doc-a", "doc-b", "doc-c"},
    )

    inserted_c_texts = [
        chunk.text
        for chunk in inserted.chunks
        if chunk.doc_id == "doc-c"
    ]

    # Only the inserted document's chunks are embedded.
    assert insert_calls == [inserted_c_texts]

    # Existing A and B vectors remain exactly unchanged.
    _assert_vectors_preserved(
        initial,
        inserted,
        include_document=lambda doc_id: doc_id in {"doc-a", "doc-b"},
    )

    preserved_versions.append(
        (inserted.path, _snapshot_files(inserted.path))
    )

    # ---------------------------------------------------------
    # 3. Update document A
    # ---------------------------------------------------------
    document_a_v2 = _document(
        "doc-a",
        "alpha refreshed retrieval pipeline",
        revision=2,
    )

    assert document_a_v2.revision_sha256 != document_a_v1.revision_sha256

    update_calls: list[list[str]] = []

    updated = store.update_document_and_activate_index_snapshot(
        root,
        index_id="idx-lifecycle-02-update",
        document=document_a_v2,
        embed_fn=_embedder(update_calls),
    )

    _assert_snapshot_integrity(
        updated,
        {"doc-a", "doc-b", "doc-c"},
    )

    updated_document_a = next(
        document
        for document in updated.documents
        if document.doc_id == "doc-a"
    )

    assert updated_document_a.revision_sha256 == document_a_v2.revision_sha256

    updated_a_chunks = {
        chunk.chunk_id: chunk
        for chunk in updated.chunks
        if chunk.doc_id == "doc-a"
    }

    assert updated_a_chunks

    # Every updated chunk must reference the new document revision.
    assert {
        chunk.document_revision_sha256
        for chunk in updated_a_chunks.values()
    } == {document_a_v2.revision_sha256}

    # The old document revision must not survive in the updated chunks.
    assert document_a_v1.revision_sha256 not in {
        chunk.document_revision_sha256
        for chunk in updated_a_chunks.values()
    }

    old_a_texts = {
        chunk.text for chunk in initial_a_chunks.values()
    }

    updated_a_texts = [
        chunk.text
        for chunk in updated_a_chunks.values()
    ]

    # No old chunk content should remain.
    assert old_a_texts.isdisjoint(updated_a_texts)

    # Stable positional IDs may be reused, but chunk revisions must change.
    for chunk_id in set(initial_a_chunks) & set(updated_a_chunks):
        assert (
            initial_a_chunks[chunk_id].revision_sha256
            != updated_a_chunks[chunk_id].revision_sha256
        )

    # Only the replacement A chunks are embedded.
    assert update_calls == [updated_a_texts]

    # Unchanged B and C vectors remain exactly unchanged.
    _assert_vectors_preserved(
        inserted,
        updated,
        include_document=lambda doc_id: doc_id in {"doc-b", "doc-c"},
    )

    preserved_versions.append(
        (updated.path, _snapshot_files(updated.path))
    )

    # ---------------------------------------------------------
    # 4. Delete document B
    # ---------------------------------------------------------
    deleted_b_chunk_ids = {
        chunk.chunk_id
        for chunk in updated.chunks
        if chunk.doc_id == "doc-b"
    }

    assert deleted_b_chunk_ids

    deleted = store.delete_document_and_activate_index_snapshot(
        root,
        index_id="idx-lifecycle-03-delete",
        doc_id="doc-b",
    )

    _assert_snapshot_integrity(
        deleted,
        {"doc-a", "doc-c"},
    )

    final_chunk_ids = {
        chunk.chunk_id
        for chunk in deleted.chunks
    }

    assert deleted_b_chunk_ids.isdisjoint(final_chunk_ids)
    assert all(chunk.doc_id != "doc-b" for chunk in deleted.chunks)

    # Delete must reuse the exact vectors for A and C.
    _assert_vectors_preserved(
        updated,
        deleted,
        include_document=lambda doc_id: doc_id in {"doc-a", "doc-c"},
    )

    final_document_a = next(
        document
        for document in deleted.documents
        if document.doc_id == "doc-a"
    )
    assert final_document_a.revision_sha256 == document_a_v2.revision_sha256

    # ---------------------------------------------------------
    # 5. Verify all earlier immutable versions remain unchanged
    # ---------------------------------------------------------
    for version_path, original_files in preserved_versions:
        assert Path(version_path).is_dir()
        assert _snapshot_files(version_path) == original_files

    assert deleted.path.is_dir()

    current = store.load_current_index_snapshot(root)

    assert current.path == deleted.path
    assert current.manifest["index_id"] == "idx-lifecycle-03-delete"

    # ---------------------------------------------------------
    # 6. Fresh Python process: reload and search
    # ---------------------------------------------------------
    child_script = r'''
import hashlib
import json
import re
import sys

import numpy as np

from hs.index_store import load_current_index_snapshot
from hs.pipeline import HybridPipeline


MODEL_ID = "test/lifecycle-embedder"
MODEL_REVISION = "a" * 40

EMBEDDING_CONFIG = {
    "adapter": "test-lifecycle-token-hash-v1",
    "normalize_embeddings": True,
    "dimension": 16,
}


def vectors(texts):
    result = np.zeros((len(texts), 16), dtype=np.float32)

    for row, text in enumerate(texts):
        for token in re.findall(r"\w+", text.lower()):
            index = (
                int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
                % 16
            )
            result[row, index] += 1.0

    return result


def embed(texts):
    return vectors(texts)


embed.model_metadata = {
    "model_id": MODEL_ID,
    "resolved_revision": MODEL_REVISION,
    "revision_verified": True,
}
embed.embedding_config = dict(EMBEDDING_CONFIG)

snapshot = load_current_index_snapshot(sys.argv[1])

pipeline = HybridPipeline.from_snapshot(
    snapshot,
    query_embed_fn=embed,
)

valid_chunk_ids = {
    chunk.chunk_id
    for chunk in snapshot.chunks
}

results_by_mode = {}

for mode in ("bm25", "dense", "hybrid"):
    result_ids = pipeline.search(
        "alpha refreshed",
        mode=mode,
        k=10,
    )

    if not result_ids:
        raise AssertionError(f"{mode} returned no results")

    if not set(result_ids) <= valid_chunk_ids:
        raise AssertionError(f"{mode} returned an unknown chunk ID")

    result_doc_ids = sorted({
        pipeline.doc_of(chunk_id)
        for chunk_id in result_ids
    })

    if "doc-b" in result_doc_ids:
        raise AssertionError(f"{mode} returned deleted document B")

    results_by_mode[mode] = {
        "chunk_ids": result_ids,
        "doc_ids": result_doc_ids,
    }


print(json.dumps({
    "index_id": snapshot.manifest["index_id"],
    "doc_ids": sorted(document.doc_id for document in snapshot.documents),
    "chunk_ids": sorted(valid_chunk_ids),
    "results": results_by_mode,
}))
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

    output_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    assert output_lines, completed.stderr

    result = json.loads(output_lines[-1])

    assert result["index_id"] == "idx-lifecycle-03-delete"
    assert result["doc_ids"] == ["doc-a", "doc-c"]
    assert set(result["chunk_ids"]) == final_chunk_ids
    assert set(result["results"]) == {"bm25", "dense", "hybrid"}

    for mode_results in result["results"].values():
        assert mode_results["chunk_ids"]
        assert set(mode_results["doc_ids"]) <= {"doc-a", "doc-c"}
        assert "doc-b" not in mode_results["doc_ids"]
