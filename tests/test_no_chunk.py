"""Step 2b: benchmark path = one searchable item per document.

Contract:
  chunk_text(doc_id, text, size=None)   -> exactly one Chunk "<doc>#0" (whitespace normalised), [] if empty
  load_chunks(path, size=None)          -> one chunk per doc
  run_eval: --no-chunk | --chunk-size N (default 200) --chunk-overlap M (default 40); --no-chunk excludes --chunk-size
"""
import importlib.util
import json
from pathlib import Path

import pytest
from hs.chunking import chunk_text
from hs.io import load_chunks

ROOT = Path(__file__).resolve().parent.parent
LONG = " ".join(f"w{i}" for i in range(500))


def test_size_none_gives_one_chunk_with_normalised_whitespace():
    chunks = chunk_text("d", "alpha  beta\n gamma", size=None)
    assert [(c.chunk_id, c.doc_id, c.text) for c in chunks] == [("d#0", "d", "alpha beta gamma")]


def test_size_none_keeps_a_long_document_whole_and_ignores_overlap():
    chunks = chunk_text("d", LONG, size=None, overlap=999)
    assert len(chunks) == 1 and chunks[0].text == LONG


def test_size_none_on_empty_text_gives_nothing():
    assert chunk_text("d", "  \n ", size=None) == []


@pytest.mark.parametrize("size", [0, -1])
def test_non_positive_size_is_still_rejected(size):
    with pytest.raises(ValueError):
        chunk_text("d", "a b c", size=size)


def test_default_chunking_is_unchanged():
    assert len(chunk_text("d", LONG)) > 1


def _write(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"doc_id": "long", "text": LONG}) + "\n" + json.dumps({"doc_id": "short", "text": "tiny doc"}) + "\n",
        encoding="utf-8")
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps([{"id": "q1", "query": "tiny", "relevant": ["short"]}]), encoding="utf-8")
    return corpus, queries


def test_load_chunks_with_size_none_is_one_chunk_per_doc(tmp_path):
    corpus, _ = _write(tmp_path)
    assert sorted(c.chunk_id for c in load_chunks(corpus, size=None)) == ["long#0", "short#0"]


# ---------- CLI
def _script():
    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "scripts" / "run_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_chunk_flag_defaults():
    a = _script().build_parser().parse_args([])
    assert a.no_chunk is False and a.chunk_size == 200 and a.chunk_overlap == 40


def test_cli_no_chunk_excludes_chunk_size():
    with pytest.raises(SystemExit):
        _script().build_parser().parse_args(["--no-chunk", "--chunk-size", "50"])


def _chunks_seen_by_pipeline(tmp_path, *extra):
    corpus, queries = _write(tmp_path)
    mod = _script()
    seen = {}

    class Spy(mod.HybridPipeline):
        def __init__(self, chunks, *a, **kw):
            seen["chunks"] = list(chunks)
            super().__init__(chunks, *a, **kw)

    mod.HybridPipeline = Spy
    mod.main(["--fake", "--no-cache", "--k", "5", "--corpus", str(corpus), "--queries", str(queries), *extra])
    return seen["chunks"]


def test_main_default_splits_long_documents(tmp_path):
    assert len(_chunks_seen_by_pipeline(tmp_path)) > 2


def test_main_no_chunk_gives_exactly_one_item_per_document(tmp_path):
    chunks = _chunks_seen_by_pipeline(tmp_path, "--no-chunk")
    assert sorted(c.doc_id for c in chunks) == ["long", "short"]


def test_main_chunk_size_flag_is_honoured(tmp_path):
    big = len(_chunks_seen_by_pipeline(tmp_path, "--chunk-size", "50", "--chunk-overlap", "10"))
    default = len(_chunks_seen_by_pipeline(tmp_path))
    assert big > default
