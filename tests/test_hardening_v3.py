import importlib.util
import json
from pathlib import Path

import pytest
from hs.chunking import Chunk
from hs.fusion import rrf
from hs.io import load_chunks, load_queries
from hs.metrics import recall_at_k
from hs.pipeline import HybridPipeline
from hs.sparse import SparseIndex

ROOT = Path(__file__).resolve().parent.parent
CH = [Chunk("a#0", "a", "hello world"), Chunk("b#0", "b", "other text")]

# ---- A: BM25 params must be finite
@pytest.mark.parametrize("kw", [{"k1": float("nan")}, {"k1": float("inf")},
                                {"b": float("nan")}, {"b": float("inf")}])
def test_sparse_rejects_non_finite_params(kw):
    with pytest.raises(ValueError):
        SparseIndex(CH, **kw)

# ---- B: RRF constant must be finite
@pytest.mark.parametrize("k", [float("nan"), float("inf")])
def test_rrf_rejects_non_finite_constant(k):
    with pytest.raises(ValueError):
        rrf([["a"]], k=k)

# ---- C: count parameters must be real positive ints, rejected up front (ValueError, not TypeError later)
@pytest.mark.parametrize("bad", [1.5, True, "5"])
def test_pipeline_rejects_non_int_candidate_k(embed, bad):
    with pytest.raises(ValueError):
        HybridPipeline(CH, embed, candidate_k=bad)

@pytest.mark.parametrize("bad", [1.5, True, "5"])
def test_search_rejects_non_int_k(embed, bad):
    p = HybridPipeline(CH, embed)
    with pytest.raises(ValueError):
        p.search("hello", mode="bm25", k=bad)

def test_metric_rejects_non_int_k():
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, 1.5)

# ---- D: CLI default k must match the README table
def _load_script():
    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "scripts" / "run_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_cli_default_k_matches_readme_table():
    default_k = _load_script().build_parser().parse_args([]).k
    assert f"Recall@{default_k}" in (ROOT / "README.md").read_text(encoding="utf-8")

def test_cli_exposes_candidate_k():
    assert _load_script().build_parser().parse_args(["--candidate-k", "20"]).candidate_k == 20

# ---- F: io schema validation (docs claim boundaries are validated)
def _w(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p

def test_corpus_bad_json_names_the_line(tmp_path):
    p = _w(tmp_path, "c.jsonl", '{"doc_id": "a", "text": "ok"}\n{not json}\n')
    with pytest.raises(ValueError, match="line 2"):
        load_chunks(p)

@pytest.mark.parametrize("row", ['{"text": "x"}', '{"doc_id": "a"}', '{"doc_id": "", "text": "x"}',
                                 '{"doc_id": 1, "text": "x"}', '{"doc_id": "a", "text": 5}'])
def test_corpus_rows_need_nonempty_string_doc_id_and_string_text(tmp_path, row):
    with pytest.raises(ValueError):
        load_chunks(_w(tmp_path, "c.jsonl", row + "\n"))

def test_corpus_duplicate_doc_ids_rejected(tmp_path):
    row = '{"doc_id": "a", "text": "x"}\n'
    with pytest.raises(ValueError, match="duplicate"):
        load_chunks(_w(tmp_path, "c.jsonl", row * 2))

def test_empty_corpus_file_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_chunks(_w(tmp_path, "c.jsonl", "\n\n"))

@pytest.mark.parametrize("payload", [
    {"not": "a list"},
    [{"id": "q", "query": "x"}],
    [{"id": "q", "query": "x", "relevant": []}],
    [{"id": "q", "query": "x", "relevant": "doc"}],
    [{"id": "q", "query": 5, "relevant": ["a"]}],
    [{"id": "q", "query": "x", "relevant": ["a"]}, {"id": "q", "query": "y", "relevant": ["a"]}],
])
def test_queries_schema_is_validated(tmp_path, payload):
    with pytest.raises(ValueError):
        load_queries(_w(tmp_path, "q.json", json.dumps(payload)))
