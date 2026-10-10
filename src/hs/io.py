import json
from pathlib import Path
from .chunking import chunk_text

def _nonempty_str(v):
    return isinstance(v, str) and v.strip() != ""

def load_chunks(path, size=200, overlap=40):
    """JSONL, one {"doc_id": str, "text": str} per line -> list[Chunk]. Validates every row."""
    chunks, seen = [], set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: invalid JSON on line {lineno}: {e.msg}") from e
        if not isinstance(d, dict) or not _nonempty_str(d.get("doc_id")) or not isinstance(d.get("text"), str):
            raise ValueError(f'{path}: line {lineno} needs a non-empty string "doc_id" and a string "text"')
        if d["doc_id"] in seen:
            raise ValueError(f"{path}: duplicate doc_id {d['doc_id']!r} on line {lineno}")
        seen.add(d["doc_id"])
        chunks.extend(chunk_text(d["doc_id"], d["text"], size, overlap))
    if not seen:
        raise ValueError(f"{path}: corpus is empty")
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
