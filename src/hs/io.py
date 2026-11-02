import json
from pathlib import Path

from .chunking import chunk_text
from .documents import Document


_DOCUMENT_FIELDS = {"doc_id", "text", "source", "title", "metadata"}


def _nonempty_str(value):
    return isinstance(value, str) and value.strip() != ""


def load_documents(path, *, max_text_chars=None):
    """Load validated JSONL documents.

    Required fields remain ``doc_id`` and ``text``. Optional fields are
    ``source``, ``title`` and an object-valued ``metadata`` field. Blank lines
    are ignored and duplicate document IDs are rejected.

    ``max_text_chars`` is an opt-in per-document text limit. The default is
    unlimited to preserve existing ingestion behavior.
    """
    if max_text_chars is not None and (
        isinstance(max_text_chars, bool)
        or not isinstance(max_text_chars, int)
        or max_text_chars < 1
    ):
        raise ValueError("max_text_chars must be a positive integer or None")

    documents = []
    seen = set()
    source_path = Path(path)

    for lineno, line in enumerate(
        source_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}: invalid JSON on line {lineno}: {exc.msg}"
            ) from exc

        if not isinstance(record, dict):
            raise ValueError(f"{path}: line {lineno} must contain a JSON object")

        unknown_fields = sorted(set(record) - _DOCUMENT_FIELDS)
        if unknown_fields:
            raise ValueError(
                f"{path}: line {lineno} has unsupported field(s): "
                + ", ".join(unknown_fields)
            )

        if not _nonempty_str(record.get("doc_id")):
            raise ValueError(
                f'{path}: line {lineno} needs a non-empty string "doc_id"'
            )
        if not isinstance(record.get("text"), str):
            raise ValueError(f'{path}: line {lineno} needs a string "text"')

        doc_id = record["doc_id"]
        if doc_id in seen:
            raise ValueError(
                f"{path}: duplicate doc_id {doc_id!r} on line {lineno}"
            )

        text = record["text"]
        if max_text_chars is not None and len(text) > max_text_chars:
            raise ValueError(
                f"{path}: line {lineno} text length {len(text)} exceeds "
                f"max_text_chars={max_text_chars}"
            )

        try:
            document = Document(
                doc_id=doc_id,
                text=text,
                source=record.get("source"),
                title=record.get("title"),
                metadata=record.get("metadata", {}),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: invalid document on line {lineno}: {exc}"
            ) from exc

        seen.add(doc_id)
        documents.append(document)

    if not documents:
        raise ValueError(f"{path}: corpus is empty")
    return documents


def load_chunks(path, size=200, overlap=40, *, max_text_chars=None):
    """Load JSONL documents and return searchable chunks.

    Legacy ``doc_id``/``text`` rows and the historical positional arguments
    remain supported. Source, title and custom metadata are copied to each
    generated chunk. An all-empty corpus is rejected explicitly.
    """
    documents = load_documents(path, max_text_chars=max_text_chars)
    chunks = []
    for document in documents:
        chunks.extend(
            chunk_text(
                document.doc_id,
                document.text,
                size,
                overlap,
                source=document.source,
                title=document.title,
                metadata=document.metadata,
                document_revision_sha256=document.revision_sha256,
            )
        )

    if not chunks:
        raise ValueError(
            f"{path}: corpus contains documents but no searchable text "
            "(all document text is empty or whitespace)"
        )
    return chunks


def load_queries(path):
    """JSON list of {"id": str, "query": str, "relevant": [non-empty list of str]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty JSON list of queries")
    ids = set()
    for i, q in enumerate(data):
        ok = (isinstance(q, dict) and _nonempty_str(q.get("id")) and isinstance(q.get("query"), str)
              and isinstance(q.get("relevant"), list) and q["relevant"]
              and all(_nonempty_str(r) for r in q["relevant"]))
        if not ok:
            raise ValueError(f'{path}: query #{i} needs "id", "query" (strings) and a non-empty "relevant" list of ids')
        if q["id"] in ids:
            raise ValueError(f"{path}: duplicate query id {q['id']!r}")
        ids.add(q["id"])
    return data
