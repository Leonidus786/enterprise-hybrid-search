#!/usr/bin/env python3
"""Validate ranking trace integrity for Enterprise Hybrid Search schema v4.

Read-only and privacy-safe: never prints query text, document text, prompts,
query hashes, or document IDs. It reports counts, field names, and outcomes.

Usage (Windows CMD):
  uv run python scripts\\validate_ranking_trace.py --input results\\scifact_doclevel_pinned_v2.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

TRACE_SCHEMA_VERSION = 1
METRIC_PREFIXES = ("precision@", "recall@", "candidate_recall@", "mrr", "ndcg@", "map")


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _id_list(trace: dict[str, Any], field: str, label: str, errors: list[str], strategy: str, query_id: str) -> list[str] | None:
    value = trace.get(field)
    if not isinstance(value, list):
        errors.append(f"{strategy}/{query_id}: _trace.{field} is missing or is not a list.")
        return None
    if any(isinstance(item, (dict, list, bool)) or item is None for item in value):
        errors.append(f"{strategy}/{query_id}: _trace.{field} contains a non-scalar ID.")
        return None
    normalized = [str(item) for item in value]
    if any(not item.strip() for item in normalized):
        errors.append(f"{strategy}/{query_id}: _trace.{field} contains an empty ID.")
    duplicates = len(normalized) - len(set(normalized))
    if duplicates:
        errors.append(f"{strategy}/{query_id}: _trace.{field} contains {duplicates} duplicate IDs.")
    return normalized


def validate(data: Any, expected_queries: int | None = None) -> tuple[list[str], list[str], dict[str, int]]:
    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, int] = {}

    if not isinstance(data, dict):
        return ["Root JSON value must be an object."], warnings, summary

    config = data.get("config")
    dataset = data.get("dataset")
    aggregate = data.get("metrics")
    per_query = data.get("per_query")
    root_trace = data.get("ranking_trace")

    if not isinstance(config, dict):
        errors.append("Missing or invalid config object.")
        config = {}
    strategies = config.get("strategies")
    if not isinstance(strategies, list) or not strategies or not all(isinstance(s, str) for s in strategies):
        errors.append("config.strategies must be a non-empty list of strings.")
        strategies = []

    artifact_count = dataset.get("evaluated_queries") if isinstance(dataset, dict) else None
    if not isinstance(artifact_count, int) or isinstance(artifact_count, bool) or artifact_count < 1:
        errors.append("dataset.evaluated_queries is missing or invalid.")
        artifact_count = None
    expected = expected_queries if expected_queries is not None else artifact_count
    if expected is not None:
        summary["expected_queries"] = expected

    if not isinstance(root_trace, dict):
        errors.append("Missing or invalid root ranking_trace object.")
    elif root_trace.get("schema_version") != TRACE_SCHEMA_VERSION:
        errors.append(
            f"Unexpected root ranking_trace.schema_version={root_trace.get('schema_version')!r}; "
            f"expected {TRACE_SCHEMA_VERSION}."
        )

    if not isinstance(aggregate, dict):
        errors.append("Missing or invalid metrics object.")
        aggregate = {}
    if not isinstance(per_query, dict):
        errors.append("Missing or invalid per_query object.")
        return errors, warnings, summary

    missing_agg = sorted(set(strategies) - set(aggregate))
    missing_pq = sorted(set(strategies) - set(per_query))
    if missing_agg:
        errors.append(f"Aggregate metrics missing strategies: {missing_agg}.")
    if missing_pq:
        errors.append(f"per_query missing strategies: {missing_pq}.")

    candidate_limit = config.get("candidate_k")
    retrieval_limit = config.get("retrieval_k")
    if not isinstance(candidate_limit, int) or isinstance(candidate_limit, bool):
        warnings.append("config.candidate_k is unavailable; candidate array length limit was not checked.")
        candidate_limit = None
    if not isinstance(retrieval_limit, int) or isinstance(retrieval_limit, bool):
        warnings.append("config.retrieval_k is unavailable; ranked array length limit was not checked.")
        retrieval_limit = None

    baseline_ids: set[str] | None = None
    total_traces = 0
    total_metrics = 0
    candidate_total = 0
    ranked_total = 0
    rerank_strategies_without_stage_trace: list[str] = []

    for strategy in strategies:
        rows = per_query.get(strategy)
        if not isinstance(rows, dict):
            errors.append(f"per_query[{strategy!r}] must be an object keyed by query ID.")
            continue
        query_ids = {str(qid) for qid in rows}
        if expected is not None and len(rows) != expected:
            errors.append(f"{strategy}: expected {expected} query records, found {len(rows)}.")
        if baseline_ids is None:
            baseline_ids = query_ids
        elif query_ids != baseline_ids:
            errors.append(f"{strategy}: query ID set differs from the first configured strategy.")

        metric_rows: dict[str, list[float]] = {}
        for query_id, row in rows.items():
            qid = str(query_id)
            if not isinstance(row, dict):
                errors.append(f"{strategy}/{qid}: per-query record is not an object.")
                continue
            trace = row.get("_trace")
            if not isinstance(trace, dict):
                errors.append(f"{strategy}/{qid}: missing _trace object.")
                continue
            total_traces += 1

            if trace.get("schema_version") != TRACE_SCHEMA_VERSION:
                errors.append(f"{strategy}/{qid}: unexpected _trace.schema_version.")
            if config.get("level") is not None and trace.get("level") != config.get("level"):
                errors.append(f"{strategy}/{qid}: _trace.level does not match config.level.")

            candidate_ids = _id_list(trace, "candidate_ids", "candidate", errors, strategy, qid)
            ranked_ids = _id_list(trace, "ranked_ids", "ranked", errors, strategy, qid)
            if candidate_ids is not None:
                candidate_total += 1
                if candidate_limit is not None and len(candidate_ids) > candidate_limit:
                    errors.append(
                        f"{strategy}/{qid}: candidate_ids has {len(candidate_ids)} IDs; candidate_k={candidate_limit}."
                    )
                reported = trace.get("candidate_count")
                if reported is not None and reported != len(candidate_ids):
                    errors.append(f"{strategy}/{qid}: candidate_count does not match candidate_ids length.")
            if ranked_ids is not None:
                ranked_total += 1
                if retrieval_limit is not None and len(ranked_ids) > retrieval_limit:
                    errors.append(
                        f"{strategy}/{qid}: ranked_ids has {len(ranked_ids)} IDs; retrieval_k={retrieval_limit}."
                    )
                reported = trace.get("returned_count")
                if reported is not None and reported != len(ranked_ids):
                    errors.append(f"{strategy}/{qid}: returned_count does not match ranked_ids length.")

            # IMPORTANT: candidate_ids is the bounded candidate pool (50 here),
            # while ranked_ids is the larger returned ranking (100 here).
            # The correct containment direction is candidate ⊆ ranked, not
            # ranked ⊆ candidate. Never compare against relevant_ids.
            if candidate_ids is not None and ranked_ids is not None:
                candidate_set = set(candidate_ids)
                ranked_set = set(ranked_ids)
                missing_candidates = candidate_set - ranked_set
                if missing_candidates:
                    errors.append(
                        f"{strategy}/{qid}: {len(missing_candidates)} candidate IDs are absent from ranked_ids."
                    )

            for key, value in row.items():
                if key == "_trace":
                    continue
                number = _finite_number(value)
                if number is not None and str(key).lower().startswith(METRIC_PREFIXES):
                    metric_rows.setdefault(str(key), []).append(number)

        # Reproduce macro-averaged aggregate ranking metrics.
        aggregate_row = aggregate.get(strategy)
        if not isinstance(aggregate_row, dict):
            continue
        for metric_name, values in metric_rows.items():
            aggregate_value = _finite_number(aggregate_row.get(metric_name))
            if aggregate_value is None or not values:
                continue
            mean_value = math.fsum(values) / len(values)
            total_metrics += 1
            if not math.isclose(mean_value, aggregate_value, rel_tol=1e-8, abs_tol=1e-8):
                errors.append(
                    f"{strategy}: mean per-query {metric_name}={mean_value:.12g} "
                    f"does not reproduce aggregate={aggregate_value:.12g}."
                )

        if "rerank" in strategy.lower():
            # The current v4 trace has candidate_ids and final ranked_ids only.
            # It does not persist an explicit pre/post-rerank pair, so it cannot
            # prove the reranker performed a permutation of precisely that pool.
            trace_sample = next((r.get("_trace") for r in rows.values() if isinstance(r, dict)), {})
            if isinstance(trace_sample, dict) and not any(
                key in trace_sample for key in ("rerank_input_ids", "reranked_ids", "pre_rerank_ids", "post_rerank_ids")
            ):
                rerank_strategies_without_stage_trace.append(strategy)

    if total_traces:
        summary["query_traces"] = total_traces
    if candidate_total:
        summary["candidate_arrays"] = candidate_total
    if ranked_total:
        summary["ranked_arrays"] = ranked_total
    if total_metrics:
        summary["aggregate_metrics_reproduced"] = total_metrics
    if rerank_strategies_without_stage_trace:
        warnings.append(
            "Reranker stage order cannot be independently verified from this artifact: "
            + ", ".join(rerank_strategies_without_stage_trace)
            + " has no separate pre/post-rerank ID arrays. Candidate coverage and final ranking integrity were checked."
        )
    if expected_queries is not None and artifact_count is not None and expected_queries != artifact_count:
        errors.append("CLI expected query count differs from dataset.evaluated_queries.")

    return errors, warnings, summary


def _self_test() -> int:
    """Exercise the trace contract, including the 50-candidate/100-ranked case."""
    base_trace = {
        "schema_version": 1,
        "level": "doc",
        "candidate_count": 50,
        "candidate_ids": list(range(50)),
        "returned_count": 100,
        "ranked_ids": list(range(100)),
        "relevant_ids": [7],
        "grades": {str(i): (1 if i == 7 else 0) for i in range(100)},
    }
    per_query = {
        strategy: {
            "q1": {
                "precision@10": 0.1,
                "recall@10": 1.0,
                "mrr": 1.0,
                "ndcg@10": 1.0,
                "candidate_recall@50": 1.0,
                "_trace": dict(base_trace),
            }
        }
        for strategy in ("bm25", "dense", "hybrid", "hybrid_rerank")
    }
    data = {
        "config": {"strategies": list(per_query), "level": "doc", "candidate_k": 50, "retrieval_k": 100},
        "dataset": {"evaluated_queries": 1},
        "metrics": {
            strategy: {
                "precision@10": 0.1,
                "recall@10": 1.0,
                "mrr": 1.0,
                "ndcg@10": 1.0,
                "candidate_recall@50": 1.0,
            }
            for strategy in per_query
        },
        "ranking_trace": {"schema_version": 1},
        "per_query": per_query,
    }
    errors, _, summary = validate(data)
    if errors:
        print("SELF-TEST FAIL:", errors)
        return 1
    if summary.get("query_traces") != 4 or summary.get("candidate_arrays") != 4 or summary.get("ranked_arrays") != 4:
        print("SELF-TEST FAIL: unexpected trace counts", summary)
        return 1
    print("SELF-TEST PASS: candidate_ids[50] ⊆ ranked_ids[100], four strategies, metrics reproduced.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Path to a result JSON artifact")
    parser.add_argument("--expected-queries", type=int, default=None)
    parser.add_argument("--self-test", action="store_true", help="Run synthetic validator self-test and exit")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()
    if not args.input:
        parser.error("--input is required unless --self-test is used")

    source = Path(args.input)
    if not source.is_file():
        print(f"ERROR: result file not found: {source}")
        return 2
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not read valid UTF-8 JSON ({type(exc).__name__}: {exc})")
        return 2

    errors, warnings, summary = validate(data, expected_queries=args.expected_queries)
    print("RANKING TRACE INTEGRITY VALIDATION")
    print(f"Artifact: {source.name}")
    print(f"Errors: {len(errors)} | Warnings: {len(warnings)}")
    for key, value in summary.items():
        print(f"PASS  {key}: {value}")
    for warning in warnings:
        print(f"WARN  {warning}")
    for error in errors:
        print(f"FAIL  {error}")
    print("RESULT: " + ("PASS" if not errors else "FAIL"))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
