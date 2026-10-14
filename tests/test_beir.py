"""Step 1a: BEIR (SciFact-style) -> our corpus.jsonl / queries.json adapter.

Contract to implement:
    hs/beir.py     convert_beir(beir_dir, out_dir, split="test") -> dict
    scripts/prepare_beir.py    build_parser(), main(argv=None)
"""
import importlib.util
import json
from pathlib import Path

import pytest
from hs.beir import convert_beir
from hs.io import load_chunks, load_queries
from hs.pipeline import HybridPipeline

ROOT = Path(__file__).resolve().parent.parent

CORPUS = [
    {"_id": "d1", "title": "Alpha", "text": "first body", "metadata": {}},
    {"_id": "d2", "title": "", "text": "second body only", "metadata": {}},
    {"_id": "d3", "title": "Gamma", "text": "third body", "metadata": {}},
]
QUERIES = [
    {"_id": "q1", "text": "find first", "metadata": {}},
    {"_id": "q2", "text": "find second", "metadata": {}},
    {"_id": "q3", "text": "train-only query, not in test qrels", "metadata": {}},
]
QRELS = [("q1", "d1", 1), ("q1", "d3", 1), ("q2", "d2", 1)]


def write_beir(root, corpus=CORPUS, queries=QUERIES, qrels=QRELS, split="test", header=True):
    root.mkdir(parents=True, exist_ok=True)
    (root / "corpus.jsonl").write_text("\n".join(json.dumps(c) for c in corpus) + "\n", encoding="utf-8")
    (root / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries) + "\n", encoding="utf-8")
    (root / "qrels").mkdir(exist_ok=True)
    lines = (["query-id\tcorpus-id\tscore"] if header else []) + [f"{q}\t{d}\t{s}" for q, d, s in qrels]
    (root / "qrels" / f"{split}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def run(tmp_path, **kw):
    src = write_beir(tmp_path / "beir", **{k: v for k, v in kw.items() if k != "split_arg"})
    out = tmp_path / "out"
    summary = convert_beir(src, out, **({"split": kw["split_arg"]} if "split_arg" in kw else {}))
    return summary, out


def corpus_rows(out):
    return [json.loads(l) for l in (out / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------- happy path
def test_output_loads_with_our_own_loaders(tmp_path):
    _, out = run(tmp_path)
    assert {c.doc_id for c in load_chunks(out / "corpus.jsonl")} == {"d1", "d2", "d3"}
    assert [q["id"] for q in load_queries(out / "queries.json")] == ["q1", "q2"]


def test_only_queries_present_in_the_split_qrels_are_kept(tmp_path):
    _, out = run(tmp_path)
    ids = [q["id"] for q in load_queries(out / "queries.json")]
    assert "q3" not in ids


def test_title_is_joined_to_body_and_missing_title_is_not(tmp_path):
    _, out = run(tmp_path)
    text = {r["doc_id"]: r["text"] for r in corpus_rows(out)}
    assert text["d1"] == "Alpha. first body"
    assert text["d2"] == "second body only"


def test_relevant_ids_keep_qrels_order_and_grades_are_stored(tmp_path):
    _, out = run(tmp_path)
    q1 = {q["id"]: q for q in load_queries(out / "queries.json")}["q1"]
    assert q1["query"] == "find first"
    assert q1["relevant"] == ["d1", "d3"]
    assert q1["grades"] == {"d1": 1, "d3": 1}


def test_summary_counts(tmp_path):
    summary, _ = run(tmp_path)
    assert summary == {"n_docs": 3, "n_queries": 2, "n_dropped_queries": 0, "n_dropped_labels": 0}


def test_every_label_exists_in_the_pipeline_after_conversion(tmp_path, embed):
    _, out = run(tmp_path)
    p = HybridPipeline(load_chunks(out / "corpus.jsonl"), embed)
    for q in load_queries(out / "queries.json"):
        assert set(q["relevant"]) <= p.doc_ids


def test_unicode_survives_the_round_trip(tmp_path):
    corpus = CORPUS + [{"_id": "d9", "title": "naïve café", "text": "नमस्ते दुनिया", "metadata": {}}]
    _, out = run(tmp_path, corpus=corpus, qrels=QRELS + [("q1", "d9", 1)])
    text = {r["doc_id"]: r["text"] for r in corpus_rows(out)}
    assert "naïve café" in text["d9"] and "नमस्ते दुनिया" in text["d9"]


def test_nested_output_dir_is_created(tmp_path):
    src = write_beir(tmp_path / "beir")
    convert_beir(src, tmp_path / "a" / "b" / "c")
    assert (tmp_path / "a" / "b" / "c" / "corpus.jsonl").exists()


def test_split_argument_selects_the_matching_qrels_file(tmp_path):
    src = write_beir(tmp_path / "beir", qrels=[("q3", "d1", 1)], split="train")
    write_beir(tmp_path / "beir", split="test")  # adds test.tsv; train.tsv stays
    summary = convert_beir(src, tmp_path / "out", split="train")
    assert [q["id"] for q in load_queries(tmp_path / "out" / "queries.json")] == ["q3"]
    assert summary["n_queries"] == 1


def test_header_row_is_optional(tmp_path):
    _, out = run(tmp_path, header=False)
    assert len(load_queries(out / "queries.json")) == 2


# ---------- label hygiene (this is where benchmarks silently go wrong)
def test_zero_score_rows_are_not_relevant(tmp_path):
    _, out = run(tmp_path, qrels=[("q1", "d1", 1), ("q1", "d3", 0), ("q2", "d2", 1)])
    q1 = {q["id"]: q for q in load_queries(out / "queries.json")}["q1"]
    assert q1["relevant"] == ["d1"]
    assert "d3" not in q1["grades"]


def test_query_with_only_zero_scores_is_dropped_and_counted(tmp_path):
    summary, out = run(tmp_path, qrels=[("q1", "d1", 1), ("q2", "d2", 0)])
    assert [q["id"] for q in load_queries(out / "queries.json")] == ["q1"]
    assert summary["n_dropped_queries"] == 1


def test_label_pointing_at_missing_doc_is_dropped_and_counted(tmp_path):
    summary, out = run(tmp_path, qrels=QRELS + [("q1", "ghost", 1)])
    q1 = {q["id"]: q for q in load_queries(out / "queries.json")}["q1"]
    assert "ghost" not in q1["relevant"]
    assert summary["n_dropped_labels"] == 1


def test_query_left_without_labels_is_removed(tmp_path):
    summary, out = run(tmp_path, qrels=[("q1", "d1", 1), ("q2", "ghost", 1)])
    assert [q["id"] for q in load_queries(out / "queries.json")] == ["q1"]
    assert summary["n_dropped_queries"] == 1 and summary["n_dropped_labels"] == 1


def test_empty_docs_are_not_written_and_their_labels_are_dropped(tmp_path):
    corpus = CORPUS + [{"_id": "d4", "title": "", "text": "   ", "metadata": {}}]
    summary, out = run(tmp_path, corpus=corpus, qrels=QRELS + [("q2", "d4", 1)])
    assert "d4" not in {r["doc_id"] for r in corpus_rows(out)}
    assert summary["n_docs"] == 3 and summary["n_dropped_labels"] == 1
    # the whole point: loaders + pipeline must accept the result
    assert load_queries(out / "queries.json")


# ---------- hard failures
def test_duplicate_corpus_ids_raise(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        run(tmp_path, corpus=CORPUS + [CORPUS[0]])


def test_qrels_for_an_unknown_query_raise(tmp_path):
    with pytest.raises(ValueError, match="unknown query"):
        run(tmp_path, qrels=QRELS + [("q404", "d1", 1)])


def test_missing_split_file_raises_file_not_found(tmp_path):
    src = write_beir(tmp_path / "beir")
    with pytest.raises(FileNotFoundError):
        convert_beir(src, tmp_path / "out", split="dev")


def test_missing_beir_dir_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_beir(tmp_path / "nope", tmp_path / "out")


def test_malformed_qrels_row_names_the_line(tmp_path):
    src = write_beir(tmp_path / "beir")
    (src / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        convert_beir(src, tmp_path / "out")


def test_non_integer_score_raises(tmp_path):
    src = write_beir(tmp_path / "beir")
    (src / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\tabc\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        convert_beir(src, tmp_path / "out")


def test_bad_json_in_corpus_names_the_line(tmp_path):
    src = write_beir(tmp_path / "beir")
    (src / "corpus.jsonl").write_text('{"_id": "d1", "title": "", "text": "ok"}\n{broken\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        convert_beir(src, tmp_path / "out")


# ---------- CLI
def _script():
    spec = importlib.util.spec_from_file_location("prepare_beir", ROOT / "scripts" / "prepare_beir.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_defaults_to_the_test_split():
    a = _script().build_parser().parse_args(["--beir-dir", "x", "--out-dir", "y"])
    assert a.split == "test" and a.beir_dir == "x" and a.out_dir == "y"


def test_cli_requires_both_directories():
    with pytest.raises(SystemExit):
        _script().build_parser().parse_args(["--beir-dir", "x"])


def test_cli_main_converts_end_to_end(tmp_path, capsys):
    src = write_beir(tmp_path / "beir")
    _script().main(["--beir-dir", str(src), "--out-dir", str(tmp_path / "out")])
    assert (tmp_path / "out" / "queries.json").exists()
    assert "n_queries" in capsys.readouterr().out
