from pathlib import Path
from hs.io import load_chunks, load_queries
from hs.pipeline import HybridPipeline
from hs.evaluate import evaluate

DATA = Path(__file__).resolve().parent.parent / "data" / "sample"

def test_load_chunks_reads_jsonl_and_chunks_each_doc():
    chunks = load_chunks(DATA / "corpus.jsonl", size=50, overlap=10)
    assert len(chunks) >= 12
    assert all(c.chunk_id.startswith(c.doc_id + "#") for c in chunks)

def test_load_queries_has_ids_text_and_relevant_docs():
    qs = load_queries(DATA / "queries.json")
    assert len(qs) >= 8
    assert all({"id", "query", "relevant"} <= set(q) for q in qs)

def test_every_labeled_doc_exists_in_corpus():
    chunks = load_chunks(DATA / "corpus.jsonl")
    docs = {c.doc_id for c in chunks}
    for q in load_queries(DATA / "queries.json"):
        assert set(q["relevant"]) <= docs, q["id"]

def test_end_to_end_eval_runs_all_modes(embed, score):
    chunks = load_chunks(DATA / "corpus.jsonl")
    p = HybridPipeline(chunks, embed, score_fn=score)
    res = evaluate(p, load_queries(DATA / "queries.json"),
                   ["bm25", "dense", "hybrid", "hybrid_rerank"], k=5)
    assert all(0.0 <= m["mrr"] <= 1.0 for m in res.values())
    assert res["hybrid_rerank"]["mrr"] > 0
