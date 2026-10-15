"""Step 3b: compare two result files with paired bootstrap + self-consistency check.

results JSON schema (what run_eval --output must write):
  {"dataset": {"searchable_items": int, "unique_documents": int, "total_queries": int, ...},
   "metrics":   {mode: {metric: aggregate}},
   "per_query": {mode: {qid: {metric: value}}}}      # same metric keys as aggregates

Contract (hs/compare.py):
  load_results(path) -> dict                         ValueError on bad JSON / missing 'per_query' / 'metrics'
  check_consistency(results, atol=1e-9) -> [str]     mean(per_query) must equal metrics (keys present in both); [] if fine
  compare(res_a, res_b, mode_a, mode_b, metric, **bootstrap_kw) -> stats dict from paired_bootstrap (delta = b - a)
scripts/compare_runs.py: --a FILE [--b FILE] --pair MODE_A MODE_B (repeatable) [--metrics m1 m2 ..] [--resamples N]
"""
import importlib.util
import json
from pathlib import Path

import pytest
from hs.compare import check_consistency, compare, load_results

ROOT = Path(__file__).resolve().parent.parent


def results(per_query, items=100):
    qids = next(iter(per_query.values())).keys()
    metrics = {}
    for mode, qs in per_query.items():
        keys = next(iter(qs.values())).keys()
        metrics[mode] = {k: sum(v[k] for v in qs.values()) / len(qs) for k in keys}
    return {"dataset": {"searchable_items": items, "unique_documents": items, "total_queries": len(qids)},
            "metrics": metrics, "per_query": per_query}


def pq(vals, metric="ndcg@10"):
    return {f"q{i}": {metric: v} for i, v in enumerate(vals)}


A = [0.2, 0.4, 0.6, 0.8]
BASE = results({"dense": pq(A), "hybrid": pq([v + 0.1 for v in A])})


def test_consistent_results_report_no_problems():
    assert check_consistency(BASE) == []


def test_mismatch_between_aggregate_and_per_query_is_reported():
    bad = json.loads(json.dumps(BASE))
    bad["metrics"]["dense"]["ndcg@10"] += 0.01
    msgs = check_consistency(bad)
    assert len(msgs) == 1 and "dense" in msgs[0] and "ndcg@10" in msgs[0]


def test_keys_only_in_aggregates_are_ignored():
    r = json.loads(json.dumps(BASE))
    r["metrics"]["dense"]["latency_p50_ms"] = 12.3
    r["metrics"]["dense"]["candidate_recall@50"] = 0.9
    assert check_consistency(r) == []


def test_compare_is_b_minus_a_and_uses_per_query_scores():
    r = compare(BASE, BASE, "dense", "hybrid", "ndcg@10", n_resamples=500)
    assert r["mean_delta"] == pytest.approx(0.1) and r["wins"] == 4 and r["significant"] is True


def test_compare_across_files_with_the_same_protocol():
    other = results({"bm25": pq([0.1, 0.2, 0.3, 0.4])})
    r = compare(other, BASE, "bm25", "hybrid", "ndcg@10", n_resamples=500)
    assert r["mean_delta"] == pytest.approx(((0.3 - 0.1) + (0.5 - 0.2) + (0.7 - 0.3) + (0.9 - 0.4)) / 4)


def test_different_protocols_cannot_be_compared():
    chunked = results({"dense": pq(A)}, items=160)
    with pytest.raises(ValueError, match="protocol"):
        compare(BASE, chunked, "dense", "dense", "ndcg@10")


def test_unknown_mode_or_metric_raises():
    with pytest.raises(ValueError, match="mode"):
        compare(BASE, BASE, "dense", "nope", "ndcg@10")
    with pytest.raises(ValueError, match="metric"):
        compare(BASE, BASE, "dense", "hybrid", "bogus@10")


def test_different_query_sets_raise():
    other = results({"dense": {f"z{i}": {"ndcg@10": 0.5} for i in range(4)}})
    with pytest.raises(ValueError, match="same query ids"):
        compare(BASE, other, "dense", "dense", "ndcg@10")


def test_load_results_round_trip_and_validation(tmp_path):
    good = tmp_path / "g.json"
    good.write_text(json.dumps(BASE), encoding="utf-8")
    assert load_results(good)["metrics"].keys() == BASE["metrics"].keys()
    for name, body in [("bad.json", "{nope"), ("noperq.json", json.dumps({"metrics": {}, "dataset": {}})),
                       ("nomet.json", json.dumps({"per_query": {}, "dataset": {}}))]:
        p = tmp_path / name
        p.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError):
            load_results(p)


# ---------- CLI
def _script():
    spec = importlib.util.spec_from_file_location("compare_runs", ROOT / "scripts" / "compare_runs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_prints_a_markdown_table_with_delta_ci_and_significance(tmp_path, capsys):
    f = tmp_path / "r.json"
    f.write_text(json.dumps(BASE), encoding="utf-8")
    _script().main(["--a", str(f), "--pair", "dense", "hybrid", "--metrics", "ndcg@10", "--resamples", "300"])
    out = capsys.readouterr().out
    assert "dense" in out and "hybrid" in out and "ndcg@10" in out
    assert "+0.100" in out and "[+0.100, +0.100]" in out and "4/0/0" in out and "yes" in out


def test_cli_accepts_two_files_and_several_pairs(tmp_path, capsys):
    fa, fb = tmp_path / "a.json", tmp_path / "b.json"
    fa.write_text(json.dumps(results({"bm25": pq([0.1, 0.2, 0.3, 0.4])})), encoding="utf-8")
    fb.write_text(json.dumps(BASE), encoding="utf-8")
    _script().main(["--a", str(fa), "--b", str(fb), "--pair", "bm25", "dense", "--pair", "bm25", "hybrid",
                    "--metrics", "ndcg@10", "--resamples", "300"])
    out = capsys.readouterr().out
    assert out.count("bm25") >= 2 and "dense" in out and "hybrid" in out


def test_cli_refuses_inconsistent_files(tmp_path):
    bad = json.loads(json.dumps(BASE))
    bad["metrics"]["dense"]["ndcg@10"] += 0.05
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(SystemExit):
        _script().main(["--a", str(f), "--pair", "dense", "hybrid", "--metrics", "ndcg@10"])


def test_cli_requires_a_pair():
    with pytest.raises(SystemExit):
        _script().build_parser().parse_args(["--a", "x.json"])
