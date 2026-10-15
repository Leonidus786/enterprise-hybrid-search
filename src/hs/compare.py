"""Validation and paired-bootstrap comparisons for saved retrieval runs."""
import json
import math
import random
import statistics
from collections.abc import Mapping
from pathlib import Path


def load_results(path):
    """Load a benchmark JSON file and validate the basic Step 3 schema."""
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load results JSON from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("results JSON must contain an object")
    if not isinstance(data.get("metrics"), dict):
        raise ValueError("results JSON must contain a 'metrics' object")
    if not isinstance(data.get("per_query"), dict):
        raise ValueError("results JSON must contain a 'per_query' object")
    if not isinstance(data.get("dataset", {}), dict):
        raise ValueError("results JSON 'dataset' must be an object")
    for mode, queries in data["per_query"].items():
        if not isinstance(mode, str) or not isinstance(queries, dict):
            raise ValueError("each per_query mode must map to a query-score object")
        for qid, scores in queries.items():
            if not isinstance(qid, str) or not isinstance(scores, dict):
                raise ValueError("each per-query score must map a query ID to a metric object")
    return data


def check_consistency(results, atol=1e-9):
    """Return errors when means of per-query scores disagree with aggregates."""
    if isinstance(atol, bool) or not isinstance(atol, (int, float)) or not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be a finite non-negative number")
    if not isinstance(results, dict) or not isinstance(results.get("metrics"), dict) or not isinstance(results.get("per_query"), dict):
        raise ValueError("results must contain 'metrics' and 'per_query' objects")

    problems = []
    for mode, aggregate in results["metrics"].items():
        if not isinstance(aggregate, dict) or mode not in results["per_query"]:
            continue
        per_query = results["per_query"][mode]
        if not per_query:
            continue
        if not isinstance(per_query, dict):
            problems.append(f"{mode}: per_query scores must be an object")
            continue
        for metric, aggregate_value in aggregate.items():
            available = [scores[metric] for scores in per_query.values() if isinstance(scores, dict) and metric in scores]
            if not available:
                continue  # aggregate-only metrics (e.g. latency) are expected
            if len(available) != len(per_query):
                problems.append(f"{mode} {metric}: missing per-query values for one or more query IDs")
                continue
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in available):
                problems.append(f"{mode} {metric}: per-query values must be finite numbers")
                continue
            if isinstance(aggregate_value, bool) or not isinstance(aggregate_value, (int, float)) or not math.isfinite(aggregate_value):
                problems.append(f"{mode} {metric}: aggregate value must be a finite number")
                continue
            mean_value = statistics.fmean(available)
            if not math.isclose(mean_value, aggregate_value, rel_tol=0.0, abs_tol=atol):
                problems.append(f"{mode} {metric}: aggregate={aggregate_value:.12g}, per-query mean={mean_value:.12g} (atol={atol:g})")
    return problems


def _protocol_mismatches(a, b, metric):
    """Compare only protocol fields that are explicitly present in both files."""
    mismatches = []
    da, db = a.get("dataset", {}), b.get("dataset", {})
    dataset_keys = ("searchable_items", "unique_documents", "total_queries", "evaluated_queries")
    for key in dataset_keys:
        if key in da and key in db and da[key] != db[key]:
            mismatches.append(f"dataset.{key}: {da[key]!r} != {db[key]!r}")
    # A query fingerprint is the strongest available check that query text and
    # relevance labels match. If only one run has it, do not silently compare.
    if ("query_fingerprint" in da) != ("query_fingerprint" in db):
        mismatches.append("dataset.query_fingerprint: missing from one run")
    elif "query_fingerprint" in da and da["query_fingerprint"] != db["query_fingerprint"]:
        mismatches.append(
            f"dataset.query_fingerprint: {da['query_fingerprint']!r} != {db['query_fingerprint']!r}"
        )

    ca, cb = a.get("config", {}), b.get("config", {})
    if isinstance(ca, dict) and isinstance(cb, dict):
        keys = ["k", "retrieval_k", "candidate_k", "level", "chunking", "fake_models", "embedding_model", "reranker_model", "bge_prefix_enabled"]
        if "latency" in metric.lower():
            keys.append("cache_enabled")
        for key in keys:
            if key in ca and key in cb and ca[key] != cb[key]:
                mismatches.append(f"config.{key}: {ca[key]!r} != {cb[key]!r}")
    if mismatches:
        return ["protocol mismatch: " + "; ".join(mismatches)]
    return []


def _percentile(values, percentile):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate percentile from empty values")
    position = (len(ordered) - 1) * percentile / 100.0
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def paired_bootstrap(deltas, n_resamples=10000, confidence=0.95, seed=42):
    """Bootstrap the mean of paired per-query deltas (B - A)."""
    if not deltas:
        raise ValueError("paired bootstrap requires at least one query")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in deltas):
        raise ValueError("paired deltas must be finite numbers")

    n = len(deltas)
    mean_delta = statistics.fmean(deltas)
    rng = random.Random(seed)
    boot_means = [statistics.fmean(rng.choices(deltas, k=n)) for _ in range(n_resamples)]
    alpha = (1.0 - confidence) / 2.0
    ci_low = _percentile(boot_means, alpha * 100.0)
    ci_high = _percentile(boot_means, (1.0 - alpha) * 100.0)
    wins = sum(delta > 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    ties = n - wins - losses
    return {
        "n": n,
        "mean_delta": mean_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence_level": confidence,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "significant": ci_low > 0.0 or ci_high < 0.0,
        "n_resamples": n_resamples,
        "seed": seed,
    }


def compare(res_a, res_b, mode_a, mode_b, metric, **bootstrap_kw):
    """Compare mode B against mode A on exactly the same query IDs."""
    for label, results in (("A", res_a), ("B", res_b)):
        if not isinstance(results, dict) or not isinstance(results.get("metrics"), dict) or not isinstance(results.get("per_query"), dict):
            raise ValueError(f"result {label} must contain metrics and per_query objects")
    if mode_a not in res_a["per_query"] or mode_a not in res_a["metrics"] or mode_b not in res_b["per_query"] or mode_b not in res_b["metrics"]:
        raise ValueError(f"unknown mode: {mode_a!r} or {mode_b!r}")
    if not isinstance(metric, str) or not metric:
        raise ValueError("metric must be a non-empty string")
    rows_a, rows_b = res_a["per_query"][mode_a], res_b["per_query"][mode_b]
    if not rows_a or not rows_b:
        raise ValueError("mode comparison requires non-empty per-query scores")
    if metric not in res_a["metrics"][mode_a] or metric not in res_b["metrics"][mode_b] or any(metric not in v for v in rows_a.values()) or any(metric not in v for v in rows_b.values()):
        raise ValueError(f"unknown metric or missing per-query metric: {metric!r}")

    protocol_errors = _protocol_mismatches(res_a, res_b, metric)
    if protocol_errors:
        raise ValueError("; ".join(protocol_errors))
    qids_a, qids_b = set(rows_a), set(rows_b)
    if qids_a != qids_b:
        raise ValueError("comparisons require the same query ids in both runs")
    qids = sorted(qids_a)
    deltas = []
    for qid in qids:
        a, b = rows_a[qid][metric], rows_b[qid][metric]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (a, b)):
            raise ValueError(f"non-finite or non-numeric score for query {qid!r}, metric {metric!r}")
        deltas.append(b - a)
    stats = paired_bootstrap(deltas, **bootstrap_kw)
    stats.update({"mode_a": mode_a, "mode_b": mode_b, "metric": metric})
    return stats
