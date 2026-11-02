import hashlib
import json

import pytest

from hs.chunking import Chunk, chunk_text
from hs.documents import Document
from hs.io import load_chunks, load_documents


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_document_content_hash_is_sha256_of_exact_utf8_text():
    text = "नमस्ते 搜索 🔎"
    document = Document("d1", text)
    assert document.content_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_same_document_id_with_changed_text_gets_new_content_and_revision_hashes():
    first = Document("d1", "original content")
    updated = Document("d1", "updated content")

    assert first.content_sha256 != updated.content_sha256
    assert first.revision_sha256 != updated.revision_sha256


def test_metadata_only_change_keeps_content_hash_but_changes_document_revision():
    first = Document("d1", "same content", metadata={"department": "sales"})
    updated = Document("d1", "same content", metadata={"department": "security"})

    assert first.content_sha256 == updated.content_sha256
    assert first.revision_sha256 != updated.revision_sha256


@pytest.mark.parametrize("field", ["source", "title"])
def test_source_or_title_change_updates_document_revision(field):
    first = Document("d1", "same", **{field: "first"})
    updated = Document("d1", "same", **{field: "second"})
    assert first.content_sha256 == updated.content_sha256
    assert first.revision_sha256 != updated.revision_sha256


def test_same_content_under_different_document_ids_is_preserved():
    # Duplicate content is not silently collapsed: document IDs remain distinct.
    first = Document("d1", "duplicate body")
    second = Document("d2", "duplicate body")
    assert first.content_sha256 == second.content_sha256
    assert first.revision_sha256 != second.revision_sha256


def test_metadata_change_is_reflected_in_document_revision_fingerprint():
    document = Document("d1", "text", metadata={"nested": {"tags": ["one"]}})
    revision = document.revision_sha256
    document.metadata["nested"]["tags"].append("two")

    assert document.revision_sha256 != revision


def test_document_revision_is_deterministic_for_equivalent_metadata_order():
    first = Document("d1", "text", metadata={"a": 1, "b": {"x": 2, "y": 3}})
    second = Document("d1", "text", metadata={"b": {"y": 3, "x": 2}, "a": 1})
    assert first.revision_sha256 == second.revision_sha256


def test_different_chunking_settings_change_chunk_revision_but_not_legacy_id():
    document = Document("d1", "one two three four")
    no_overlap = chunk_text(
        document.doc_id, document.text, size=3, overlap=0,
        document_revision_sha256=document.revision_sha256,
    )[0]
    overlap = chunk_text(
        document.doc_id, document.text, size=3, overlap=1,
        document_revision_sha256=document.revision_sha256,
    )[0]

    assert no_overlap.chunk_id == overlap.chunk_id == "d1#0"
    assert no_overlap.text == overlap.text == "one two three"
    assert no_overlap.content_sha256 == overlap.content_sha256
    assert no_overlap.revision_sha256 != overlap.revision_sha256


def test_document_revision_change_invalidates_chunk_revision_without_changing_chunk_id():
    first_doc = Document("d1", "one two three", metadata={"version": 1})
    updated_doc = Document("d1", "one two three", metadata={"version": 2})
    first = chunk_text(
        "d1", first_doc.text, size=3, overlap=0,
        document_revision_sha256=first_doc.revision_sha256,
    )[0]
    updated = chunk_text(
        "d1", updated_doc.text, size=3, overlap=0,
        document_revision_sha256=updated_doc.revision_sha256,
    )[0]

    assert first.chunk_id == updated.chunk_id == "d1#0"
    assert first.content_sha256 == updated.content_sha256
    assert first.revision_sha256 != updated.revision_sha256


def test_chunk_revision_is_stable_across_repeated_builds():
    document = Document("d1", "alpha beta gamma delta", metadata={"kind": "guide"})

    def build():
        return chunk_text(
            document.doc_id,
            document.text,
            size=2,
            overlap=1,
            source=document.source,
            title=document.title,
            metadata=document.metadata,
            document_revision_sha256=document.revision_sha256,
        )

    assert [c.revision_sha256 for c in build()] == [c.revision_sha256 for c in build()]


def test_chunk_content_hash_uses_exact_chunk_text():
    a = Chunk("d1#0", "d1", "alpha beta")
    b = Chunk("d1#0", "d1", "alpha  beta")
    assert a.content_sha256 != b.content_sha256


def test_loader_preserves_duplicate_content_for_distinct_document_ids(tmp_path):
    path = _write_jsonl(tmp_path / "corpus.jsonl", [
        {"doc_id": "d1", "text": "same body"},
        {"doc_id": "d2", "text": "same body"},
    ])
    docs = load_documents(path)
    chunks = load_chunks(path, size=None)

    assert [doc.doc_id for doc in docs] == ["d1", "d2"]
    assert docs[0].content_sha256 == docs[1].content_sha256
    assert docs[0].revision_sha256 != docs[1].revision_sha256
    assert len(chunks) == 2
    assert chunks[0].revision_sha256 != chunks[1].revision_sha256


def test_loader_propagates_document_revision_into_chunk_fingerprint(tmp_path):
    path = _write_jsonl(tmp_path / "corpus.jsonl", [
        {"doc_id": "d1", "text": "same body", "metadata": {"version": 1}},
    ])
    doc = load_documents(path)[0]
    chunk = load_chunks(path, size=None)[0]

    assert chunk.document_revision_sha256 == doc.revision_sha256


def test_legacy_direct_chunk_has_deterministic_identity_and_constructor():
    first = Chunk("d1#0", "d1", "legacy text")
    second = Chunk("d1#0", "d1", "legacy text")
    assert first.content_sha256 == second.content_sha256
    assert first.revision_sha256 == second.revision_sha256
    assert (first.chunk_id, first.doc_id, first.text) == ("d1#0", "d1", "legacy text")


def test_chunk_metadata_change_is_reflected_in_chunk_revision_fingerprint():
    chunk = Chunk("d1#0", "d1", "text", metadata={"tags": ["a"]})
    revision = chunk.revision_sha256
    chunk.metadata["tags"].append("b")
    assert chunk.revision_sha256 != revision


def test_direct_chunk_metadata_change_updates_chunk_revision():
    first = Chunk("d1#0", "d1", "same text", metadata={"classification": "public"})
    updated = Chunk("d1#0", "d1", "same text", metadata={"classification": "private"})
    assert first.content_sha256 == updated.content_sha256
    assert first.revision_sha256 != updated.revision_sha256


def test_direct_chunk_source_change_updates_chunk_revision():
    first = Chunk("d1#0", "d1", "same text", source="first.txt")
    updated = Chunk("d1#0", "d1", "same text", source="second.txt")
    assert first.content_sha256 == updated.content_sha256
    assert first.revision_sha256 != updated.revision_sha256
