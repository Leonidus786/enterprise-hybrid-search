import copy

import pytest
from hs.compare import _protocol_mismatches, compare


def v4_result():
    return {
        "schema_version": 4,
        "dataset": {
            "searchable_items": 100,
            "unique_documents": 100,
            "total_queries": 2,
            "evaluated_queries": 2,
            "corpus_sha256": "c" * 64,
            "queries_file_sha256": "q" * 64,
            "query_fingerprint": "f" * 64,
        },
        "config": {
            "strategies": ["dense", "hybrid"],
            "k": 10,
            "retrieval_k": 100,
            "candidate_k": 50,
            "level": "doc",
            "chunking": {"enabled": False, "size": None, "overlap": 40},
            "fake_models": False,
            "embedding_model": "embed-model",
            "reranker_model": "reranker-model",
            "bge_prefix_enabled": True,
            "embedding_revision_requested": None,
            "reranker_revision_requested": None,
            "cache_enabled": False,
        },
        "provenance": {
            "source_code": {"sha256": "s" * 64},
            "models": {
                "embedding": {"model_id": "embed-model", "resolved_revision": "a" * 40, "revision_status": "resolved_commit"},
                "reranker": {"model_id": "reranker-model", "resolved_revision": "b" * 40, "revision_status": "resolved_commit"},
            },
            "runtime": {
                "python": "3.13.0",
                "platform": "linux",
                "machine": "x86_64",
                "processor": "cpu-x",
                "cpu_count_logical": 8,
                "packages": {"torch": "2.0"},
            },
        },
        "metrics": {
            "dense": {"ndcg@10": 0.50, "latency_p50_ms": 10.0},
            "hybrid": {"ndcg@10": 0.52, "latency_p50_ms": 12.0},
        },
        "per_query": {
            "dense": {"q1": {"ndcg@10": 0.4}, "q2": {"ndcg@10": 0.6}},
            "hybrid": {"q1": {"ndcg@10": 0.42}, "q2": {"ndcg@10": 0.62}},
        },
    }


def test_v4_same_run_mode_comparison_is_allowed():
    result = v4_result()
    assert _protocol_mismatches(result, result, "ndcg@10") == []
    stats = compare(result, result, "dense", "hybrid", "ndcg@10", n_resamples=200)
    assert stats["mean_delta"] == pytest.approx(0.02)


def test_v4_missing_cache_flag_is_rejected_for_latency_comparison():
    a = v4_result()
    b = copy.deepcopy(a)
    del b["config"]["cache_enabled"]
    mismatches = _protocol_mismatches(a, b, "latency_p50_ms")
    assert mismatches and "cache_enabled" in mismatches[0]


def test_v4_missing_model_revision_is_rejected_across_files():
    a = v4_result()
    b = copy.deepcopy(a)
    b["provenance"]["models"]["embedding"]["resolved_revision"] = None
    mismatches = _protocol_mismatches(a, b, "ndcg@10")
    assert mismatches and "resolved_revision" in mismatches[0]


def test_v4_different_corpus_identity_is_rejected():
    a = v4_result()
    b = copy.deepcopy(a)
    b["dataset"]["corpus_sha256"] = "d" * 64
    mismatches = _protocol_mismatches(a, b, "ndcg@10")
    assert mismatches and "corpus_sha256" in mismatches[0]


def test_v4_different_source_fingerprint_is_rejected():
    a = v4_result()
    b = copy.deepcopy(a)
    b["provenance"]["source_code"]["sha256"] = "t" * 64
    mismatches = _protocol_mismatches(a, b, "ndcg@10")
    assert mismatches and "source code differs" in mismatches[0]


def test_schema_v3_and_v4_comparison_is_rejected():
    a = v4_result()
    b = copy.deepcopy(a)
    b["schema_version"] = 3
    mismatches = _protocol_mismatches(a, b, "ndcg@10")
    assert mismatches and "schema_version" in mismatches[0]


def test_v4_missing_input_hash_is_rejected():
    a = v4_result()
    b = copy.deepcopy(a)
    b["dataset"]["corpus_sha256"] = None
    mismatches = _protocol_mismatches(a, b, "ndcg@10")
    assert mismatches and "corpus_sha256" in mismatches[0]
