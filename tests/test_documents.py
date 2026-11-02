import json
from pathlib import Path

import pytest

from hs.chunking import Chunk
from hs.documents import Document, normalize_metadata
from hs.io import load_chunks, load_documents


def write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_legacy_jsonl_schema_loads_as_documents(tmp_path):
    path = write_jsonl(tmp_path / "corpus.jsonl", [
        {"doc_id": "d1", "text": "legacy document text"},
    ])

    documents = load_documents(path)

    assert documents == [Document("d1", "legacy document text")]
    assert documents[0].source is None
    assert documents[0].title is None
    assert documents[0].metadata == {}


def test_metadata_fields_are_normalized_and_propagated_to_chunks(tmp_path):
    path = write_jsonl(tmp_path / "corpus.jsonl", [{
        "doc_id": "policy-1",
        "text": "alpha beta gamma delta epsilon",
        "source": "  handbook/policy.pdf  ",
        "title": "  Access Policy  ",
        "metadata": {
            "department": "security",
            "version": 3,
            "labels": ["access", "identity"],
            "published": True,
            "nested": {"reviewed": None},
        },
    }])

    docs = load_documents(path)
    chunks = load_chunks(path, size=3, overlap=1)

    assert docs[0].source == "handbook/policy.pdf"
    assert docs[0].title == "Access Policy"
    assert docs[0].metadata["department"] == "security"
    assert [chunk.chunk_id for chunk in chunks] == [
        "policy-1#0", "policy-1#1",
    ]
    assert all(chunk.source == docs[0].source for chunk in chunks)
    assert all(chunk.title == docs[0].title for chunk in chunks)
    assert all(chunk.metadata == docs[0].metadata for chunk in chunks)


def test_null_metadata_normalizes_to_empty_object(tmp_path):
    path = write_jsonl(tmp_path / "corpus.jsonl", [{
        "doc_id": "d1", "text": "text", "metadata": None,
    }])

    assert load_documents(path)[0].metadata == {}


def test_blank_optional_source_and_title_normalize_to_none(tmp_path):
    path = write_jsonl(tmp_path / "corpus.jsonl", [{
        "doc_id": "d1", "text": "text", "source": "   ", "title": "",
    }])

    document = load_documents(path)[0]

    assert document.source is None
    assert document.title is None


@pytest.mark.parametrize(
    "record, message",
    [
        ({"text": "missing id"}, "doc_id"),
        ({"doc_id": "  ", "text": "text"}, "doc_id"),
        ({"doc_id": "d1"}, "text"),
        ({"doc_id": "d1", "text": 123}, "text"),
        (["not", "an", "object"], "JSON object"),
    ],
)
def test_malformed_or_missing_required_fields_are_rejected(tmp_path, record, message):
    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_documents(path)


def test_duplicate_document_ids_are_rejected(tmp_path):
    path = write_jsonl(tmp_path / "corpus.jsonl", [
        {"doc_id": "d1", "text": "first"},
        {"doc_id": "d1", "text": "second"},
    ])

    with pytest.raises(ValueError, match="duplicate doc_id"):
        load_documents(path)


@pytest.mark.parametrize("field,value", [
    ("source", 42),
    ("title", ["not", "text"]),
    ("metadata", "not an object"),
])
def test_optional_fields_reject_invalid_types(tmp_path, field, value):
    row = {"doc_id": "d1", "text": "valid text", field: value}
    path = write_jsonl(tmp_path / "corpus.jsonl", [row])

    with pytest.raises(ValueError, match=field):
        load_documents(path)


@pytest.mark.parametrize(
    "metadata",
    [
        {"score": float("nan")},
        {"score": float("inf")},
        {"": "empty key"},
    ],
)
def test_metadata_rejects_non_json_values(tmp_path, metadata):
    path = write_jsonl(tmp_path / "corpus.jsonl", [{
        "doc_id": "d1", "text": "valid", "metadata": metadata,
    }])

    with pytest.raises(ValueError, match="metadata"):
        load_documents(path)



@pytest.mark.parametrize(
    "metadata",
    [
        {"items": [object()]},
        {1: "non-string key"},
        {"not_json": (1, 2)},
    ],
)
def test_metadata_normalizer_rejects_non_json_python_values(metadata):
    with pytest.raises(ValueError, match="metadata"):
        normalize_metadata(metadata)


def test_unknown_document_fields_are_rejected_instead_of_silently_dropped(tmp_path):
    path = write_jsonl(tmp_path / "corpus.jsonl", [{
        "doc_id": "d1", "text": "valid", "source_url": "https://example.test",
    }])

    with pytest.raises(ValueError, match="unsupported field"):
        load_documents(path)


def test_unicode_text_and_metadata_are_preserved(tmp_path):
    path = write_jsonl(tmp_path / "corpus.jsonl", [{
        "doc_id": "doc-हिन्दी",
        "text": "नमस्ते दुनिया — retrieval 搜索 🔎",
        "source": "स्रोत/文件.txt",
        "title": "भाषा दस्तावेज़",
        "metadata": {"language": "hi", "tags": ["हिन्दी", "中文", "🔎"]},
    }])

    document = load_documents(path)[0]
    chunk = load_chunks(path, size=None)[0]

    assert document.doc_id == "doc-हिन्दी"
    assert document.text == "नमस्ते दुनिया — retrieval 搜索 🔎"
    assert chunk.source == "स्रोत/文件.txt"
    assert chunk.metadata["tags"] == ["हिन्दी", "中文", "🔎"]


def test_opt_in_text_length_limit_rejects_oversized_document(tmp_path):
    path = write_jsonl(tmp_path / "corpus.jsonl", [
        {"doc_id": "d1", "text": "123456"},
    ])

    with pytest.raises(ValueError, match="max_text_chars=5"):
        load_documents(path, max_text_chars=5)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "10"])
def test_text_length_limit_must_be_positive_integer_or_none(tmp_path, limit):
    path = write_jsonl(tmp_path / "corpus.jsonl", [
        {"doc_id": "d1", "text": "ok"},
    ])

    with pytest.raises(ValueError, match="max_text_chars"):
        load_documents(path, max_text_chars=limit)


def test_text_length_limit_is_opt_in_and_default_remains_unlimited(tmp_path):
    path = write_jsonl(tmp_path / "corpus.jsonl", [
        {"doc_id": "d1", "text": "x" * 100_000},
    ])

    assert len(load_documents(path)[0].text) == 100_000


def test_empty_text_is_retained_as_document_but_not_searchable_chunk(tmp_path):
    path = write_jsonl(tmp_path / "corpus.jsonl", [
        {"doc_id": "empty", "text": "   \n"},
        {"doc_id": "valid", "text": "searchable text"},
    ])

    documents = load_documents(path)
    chunks = load_chunks(path, size=None)

    assert [doc.doc_id for doc in documents] == ["empty", "valid"]
    assert [chunk.doc_id for chunk in chunks] == ["valid"]


def test_all_empty_corpus_is_rejected_at_searchable_chunk_boundary(tmp_path):
    path = write_jsonl(tmp_path / "corpus.jsonl", [
        {"doc_id": "empty-1", "text": "   "},
        {"doc_id": "empty-2", "text": "\n\t"},
    ])

    assert len(load_documents(path)) == 2
    with pytest.raises(ValueError, match="no searchable text"):
        load_chunks(path)


def test_empty_jsonl_corpus_is_rejected(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n \n", encoding="utf-8")

    with pytest.raises(ValueError, match="corpus is empty"):
        load_documents(path)


def test_document_metadata_is_copied_from_caller_input():
    original = {"nested": {"tags": ["one"]}}
    document = Document("d1", "text", metadata=original)
    original["nested"]["tags"].append("two")

    assert document.metadata == {"nested": {"tags": ["one"]}}


def test_legacy_three_argument_chunk_construction_remains_compatible():
    chunk = Chunk("d1#0", "d1", "old positional API")

    assert chunk.source is None
    assert chunk.title is None
    assert chunk.metadata == {}
