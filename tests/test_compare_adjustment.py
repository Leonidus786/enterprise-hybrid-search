"""Tests for family-wise correction in the comparison CLI."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("compare_runs_adjustment", ROOT / "scripts" / "compare_runs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bonferroni_confidence_is_family_size_aware():
    mod = _script()
    assert mod.bonferroni_confidence(0.05, 1) == pytest.approx(0.95)
    assert mod.bonferroni_confidence(0.05, 9) == pytest.approx(1 - 0.05 / 9)


@pytest.mark.parametrize("alpha,count", [(0, 3), (1, 3), (-0.1, 3), (0.05, 0), (0.05, True)])
def test_bonferroni_confidence_rejects_invalid_arguments(alpha, count):
    with pytest.raises(ValueError):
        _script().bonferroni_confidence(alpha, count)


def test_cli_adjusts_for_all_pair_metric_hypotheses(tmp_path, capsys):
    mod = _script()
    qids = [f"q{i}" for i in range(6)]
    base_values = [0.10, 0.30, 0.40, 0.60, 0.70, 0.80]
    metrics_names = ["ndcg@10", "recall@100", "mrr"]
    per_query = {}
    for mode_idx, mode in enumerate(("a", "b", "c", "d")):
        per_query[mode] = {}
        for q_idx, qid in enumerate(qids):
            per_query[mode][qid] = {
                metric: min(1.0, max(0.0, base_values[q_idx] + mode_idx * (0.01 + metric_idx * 0.002)))
                for metric_idx, metric in enumerate(metrics_names)
            }
    metrics = {
        mode: {
            metric: sum(per_query[mode][q][metric] for q in qids) / len(qids)
            for metric in metrics_names
        }
        for mode in per_query
    }
    result = {
        "dataset": {
            "searchable_items": 10,
            "unique_documents": 10,
            "total_queries": 6,
            "evaluated_queries": 6,
            "query_fingerprint": "same",
        },
        "config": {
            "k": 10,
            "retrieval_k": 100,
            "candidate_k": 50,
            "level": "doc",
            "chunking": {"enabled": False},
            "fake_models": False,
        },
        "metrics": metrics,
        "per_query": per_query,
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    rows = mod.main([
        "--a", str(path),
        "--pair", "a", "b",
        "--pair", "b", "c",
        "--pair", "c", "d",
        "--metrics", *metrics_names,
        "--resamples", "1000",
    ])
    out = capsys.readouterr().out
    assert "9 hypotheses" in out
    assert "Bonferroni CI (99.444%)" in out
    assert "95% CI (nominal)" in out
    assert "Significant after adjustment?" in out
    assert len(rows) == 9
    assert all(row[3]["family_size"] == 9 for row in rows)
    assert all(row[3]["adjusted_confidence_level"] == pytest.approx(1 - 0.05 / 9) for row in rows)
