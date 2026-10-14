"""Convert a BEIR dataset split into the project's evaluation format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path, kind: str) -> dict[str, dict[str, Any]]:
    """Read JSONL records, preserving order and rejecting duplicate IDs."""
    if not path.is_file():
        raise FileNotFoundError(path)

    records: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path.name} at line {line_no}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path.name} at line {line_no}"
                )

            record_id = record.get("_id")
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(
                    f"Missing or invalid _id in {path.name} at line {line_no}"
                )

            if record_id in records:
                raise ValueError(
                    f"duplicate {kind} id {record_id!r} "
                    f"in {path.name} at line {line_no}"
                )

            text = record.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"Missing or invalid text in {path.name} at line {line_no}"
                )

            records[record_id] = record

    return records


def _read_qrels(
    path: Path,
    known_query_ids: set[str],
    known_doc_ids: set[str],
) -> tuple[dict[str, dict[str, int]], list[str], int]:
    """Read qrels and return positive grades, query order, and dropped labels."""
    if not path.is_file():
        raise FileNotFoundError(path)

    grades_by_query: dict[str, dict[str, int]] = {}
    query_order: list[str] = []
    seen_queries: set[str] = set()
    dropped_labels = 0
    first_data_row = True

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue

            fields = raw_line.rstrip("\r\n").split("\t")

            # BEIR qrels commonly contain a header, but it is optional.
            if first_data_row:
                first_data_row = False
                normalized = [field.strip().casefold() for field in fields]
                if normalized == ["query-id", "corpus-id", "score"]:
                    continue

            if len(fields) != 3:
                raise ValueError(
                    f"Malformed qrels at line {line_no}: "
                    "expected query-id, corpus-id and score"
                )

            query_id, doc_id, score_text = (
                field.strip() for field in fields
            )

            try:
                score = int(score_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid integer score in qrels at line {line_no}"
                ) from exc

            if query_id not in known_query_ids:
                raise ValueError(
                    f"unknown query {query_id!r} in qrels at line {line_no}"
                )

            if query_id not in seen_queries:
                seen_queries.add(query_id)
                query_order.append(query_id)
                grades_by_query[query_id] = {}

            # Zero and negative scores are not relevance labels.
            if score <= 0:
                continue

            if doc_id not in known_doc_ids:
                dropped_labels += 1
                continue

            # Dict insertion order preserves the qrels order of relevant docs.
            grades_by_query[query_id][doc_id] = score

    return grades_by_query, query_order, dropped_labels


def convert_beir(
    beir_dir: str | Path,
    out_dir: str | Path,
    split: str = "test",
) -> dict[str, int]:
    """Convert corpus.jsonl, queries.jsonl and qrels/<split>.tsv.

    Output:
        corpus.jsonl: doc_id, text, metadata
        queries.json: id, query, relevant, grades, metadata
    """
    source_dir = Path(beir_dir)

    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)

    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string")

    corpus = _read_jsonl(source_dir / "corpus.jsonl", "corpus")
    queries = _read_jsonl(source_dir / "queries.jsonl", "query")

    output_docs: list[dict[str, Any]] = []

    for doc_id, record in corpus.items():
        title = record.get("title", "")
        body = record["text"]

        if not isinstance(title, str):
            raise ValueError(f"Invalid title for document {doc_id!r}")

        title = title.strip()
        body = body.strip()

        if title and body:
            text = f"{title}. {body}"
        else:
            text = title or body

        # Empty documents are omitted so that no empty document can be
        # referenced accidentally during retrieval or evaluation.
        if not text.strip():
            continue

        output_docs.append(
            {
                "doc_id": doc_id,
                "text": text,
                "metadata": record.get("metadata") or {},
            }
        )

    valid_doc_ids = {doc["doc_id"] for doc in output_docs}

    qrels_path = source_dir / "qrels" / f"{split}.tsv"
    grades_by_query, qrels_query_order, dropped_labels = _read_qrels(
        qrels_path,
        known_query_ids=set(queries),
        known_doc_ids=valid_doc_ids,
    )

    output_queries: list[dict[str, Any]] = []
    dropped_queries = 0

    # Retain the original queries.jsonl order. A query is retained only if
    # at least one positive label still points to a valid output document.
    for query_id, record in queries.items():
        if query_id not in grades_by_query:
            continue

        grades = grades_by_query[query_id]
        if not grades:
            dropped_queries += 1
            continue

        output_queries.append(
            {
                "id": query_id,
                "query": record["text"].strip(),
                "relevant": list(grades),
                "grades": grades,
                "metadata": record.get("metadata") or {},
            }
        )

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    corpus_path = destination / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8", newline="\n") as handle:
        for doc in output_docs:
            handle.write(
                json.dumps(doc, ensure_ascii=False) + "\n"
            )

    queries_path = destination / "queries.json"
    queries_path.write_text(
        json.dumps(output_queries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "n_docs": len(output_docs),
        "n_queries": len(output_queries),
        "n_dropped_queries": dropped_queries,
        "n_dropped_labels": dropped_labels,
    }