#!/usr/bin/env python3
"""Generate a README benchmark section from an Enterprise Hybrid Search result JSON.

No query text, document text, prompts, or secrets are read or emitted.
The script only consumes aggregate metrics and provenance/configuration metadata.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

BEGIN = "<!-- BEGIN GENERATED HYBRID SEARCH BENCHMARK -->"
END = "<!-- END GENERATED HYBRID SEARCH BENCHMARK -->"
STRATEGY_LABELS = {
    "bm25": "BM25",
    "dense": "Dense",
    "hybrid": "Hybrid (BM25 + Dense, RRF)",
    "hybrid_rerank": "Hybrid + Cross-Encoder",
}
METRIC_COLUMNS = (
    ("precision@10", "Precision@10"),
    ("recall@10", "Recall@10"),
    ("recall@100", "Recall@100"),
    ("candidate_recall@50", "Candidate Recall@50"),
    ("mrr@10", "MRR@10"),
    ("ndcg@10", "nDCG@10"),
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def build_section(data: dict[str, Any], input_path: Path) -> str:
    config = data.get("config") or {}
    dataset = data.get("dataset") or {}
    metrics = data.get("metrics") or {}
    git = data.get("git") or {}
    provenance = data.get("provenance") or {}
    prov_git = provenance.get("git") or {}
    prov_models = provenance.get("models") or {}

    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("JSON does not contain a non-empty 'metrics' object")

    strategies = config.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        strategies = [name for name in STRATEGY_LABELS if name in metrics]
    strategies = [name for name in strategies if name in metrics]
    if not strategies:
        raise ValueError("No supported strategy metrics found in config/metrics")

    total_queries = _first(dataset, "total_queries")
    evaluated_queries = _first(dataset, "evaluated_queries")
    unique_docs = _first(dataset, "unique_documents")
    searchable_items = _first(dataset, "searchable_items")
    commit = _first(git, "commit") or _first(prov_git, "commit") or "not recorded"
    dirty = _first(git, "working_tree_dirty")
    if dirty is None:
        dirty = _first(prov_git, "working_tree_dirty")
    dirty_text = "not recorded" if dirty is None else ("dirty" if dirty else "clean")

    embedding = prov_models.get("embedding") or {}
    reranker = prov_models.get("reranker") or {}
    embedding_revision = _first(embedding, "resolved_revision") or _first(config, "embedding_revision_requested") or "not pinned"
    reranker_revision = _first(reranker, "resolved_revision") or _first(config, "reranker_revision_requested") or "not pinned"
    embedding_verified = _first(embedding, "revision_verified")
    reranker_verified = _first(reranker, "revision_verified")

    rows = [
        "| Strategy | " + " | ".join(label for _, label in METRIC_COLUMNS) + " |",
        "|---|" + "|".join(["---:"] * len(METRIC_COLUMNS)) + "|",
    ]
    for strategy in strategies:
        item = metrics.get(strategy) or {}
        values = []
        for key, _label in METRIC_COLUMNS:
            value = _first(item, key, "mrr" if key == "mrr@10" else key)
            values.append(_fmt(value))
        label = STRATEGY_LABELS.get(strategy, strategy)
        rows.append("| " + label + " | " + " | ".join(values) + " |")

    chunk_note = ""
    level = config.get("level", "not recorded")
    if config.get("no_chunk") is True:
        protocol_label = "Document-level (one searchable item per document)"
    elif config.get("chunk_size") is not None:
        protocol_label = f"Chunked (size={config.get('chunk_size')}, overlap={config.get('chunk_overlap', 'not recorded')})"
    else:
        protocol_label = f"{level}-level; chunking configuration recorded in JSON"

    cache_enabled = config.get("cache_enabled")
    cache_text = "not recorded" if cache_enabled is None else ("enabled" if cache_enabled else "disabled")

    verification_lines = [
        f"- Embedding model: `{config.get('embedding_model', 'not recorded')}`; revision `{embedding_revision}`; revision verified: `{embedding_verified}`.",
        f"- Reranker model: `{config.get('reranker_model', 'not recorded')}`; revision `{reranker_revision}`; revision verified: `{reranker_verified}`.",
        f"- Persistent embedding cache setting: `{cache_text}`.",
        f"- Git commit: `{commit}`; working-tree state recorded as `{dirty_text}`.",
        f"- Source artifact: `{input_path.as_posix()}`.",
    ]

    parts = [
        "## Benchmark results",
        "",
        f"**Dataset:** SciFact test | **Queries:** {evaluated_queries if evaluated_queries is not None else 'not recorded'} / {total_queries if total_queries is not None else 'not recorded'} | **Corpus documents:** {unique_docs if unique_docs is not None else 'not recorded'} | **Searchable items:** {searchable_items if searchable_items is not None else 'not recorded'}.",
        f"**Protocol:** {protocol_label} | **K:** {config.get('k', 'not recorded')} | **Retrieval K:** {config.get('retrieval_k', 'not recorded')} | **Candidate K:** {config.get('candidate_k', 'not recorded')}.",
        "",
        *rows,
        "",
        "### Reproducibility metadata",
        *verification_lines,
        "",
        "### Interpretation and limitations",
        "- These are observed results for this SciFact configuration, not evidence of state-of-the-art performance or enterprise-domain effectiveness.",
        "- Keep document-level and chunked experiments in separate tables; do not combine their metrics or confidence intervals.",
        "- The Cross-Encoder result is an experiment, not the default recommendation: on the recorded run it did not improve the displayed ranking metrics and added substantial CPU inference time. It should remain optional/off by default until it demonstrates a measured gain on the target corpus.",
        "- The latency values in an evaluation JSON are not a production latency claim. Publish P50/P95 only from the dedicated repeated, warm-model, interleaved latency protocol with uncached query embeddings and recorded hardware.",
        "- The benchmark uses English scientific text. It does not establish performance on private enterprise documents, other domains, multilingual/Hinglish queries, concurrent traffic, or million-document corpora.",
        "- This repository is a retrieval/evaluation toolkit prototype. Do not describe it as enterprise-ready or production-ready until authentication, tenant isolation, source ACL enforcement, deletion, monitoring, recovery, and security controls have been implemented and tested.",
        "",
        "> Numbers are generated from the result JSON. Do not hand-edit the table; rerun the generator after creating a new benchmark artifact.",
    ]
    return "\n".join(parts).rstrip() + "\n"


def write_csv(data: dict[str, Any], csv_path: Path) -> None:
    config = data.get("config") or {}
    dataset = data.get("dataset") or {}
    metrics = data.get("metrics") or {}
    strategies = config.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        strategies = [name for name in STRATEGY_LABELS if name in metrics]
    fields = [
        "strategy", "precision@10", "recall@10", "recall@100",
        "candidate_recall@50", "mrr@10", "ndcg@10",
        "latency_p50_ms_observed", "latency_p95_ms_observed",
        "evaluated_queries", "total_queries", "unique_documents",
        "level", "k", "retrieval_k", "candidate_k", "cache_enabled",
        "embedding_model", "reranker_model",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for strategy in strategies:
            item = metrics.get(strategy) or {}
            writer.writerow({
                "strategy": strategy,
                "precision@10": _first(item, "precision@10"),
                "recall@10": _first(item, "recall@10"),
                "recall@100": _first(item, "recall@100"),
                "candidate_recall@50": _first(item, "candidate_recall@50"),
                "mrr@10": _first(item, "mrr@10", "mrr"),
                "ndcg@10": _first(item, "ndcg@10"),
                "latency_p50_ms_observed": _first(item, "latency_p50_ms"),
                "latency_p95_ms_observed": _first(item, "latency_p95_ms"),
                "evaluated_queries": _first(dataset, "evaluated_queries"),
                "total_queries": _first(dataset, "total_queries"),
                "unique_documents": _first(dataset, "unique_documents"),
                "level": config.get("level"),
                "k": config.get("k"),
                "retrieval_k": config.get("retrieval_k"),
                "candidate_k": config.get("candidate_k"),
                "cache_enabled": config.get("cache_enabled"),
                "embedding_model": config.get("embedding_model"),
                "reranker_model": config.get("reranker_model"),
            })


def update_readme(readme_path: Path, section: str) -> None:
    text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# Enterprise Hybrid Search\n"
    has_begin = BEGIN in text
    has_end = END in text
    if has_begin != has_end:
        raise ValueError(f"README must contain both markers or neither: {BEGIN} / {END}")
    block = f"{BEGIN}\n{section.rstrip()}\n{END}"
    if has_begin:
        start = text.index(BEGIN)
        end = text.index(END, start) + len(END)
        updated = text[:start] + block + text[end:]
    else:
        updated = text.rstrip() + "\n\n" + block + "\n"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(updated, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate/update README benchmark results from a result JSON.")
    parser.add_argument("--input", required=True, type=Path, help="Result JSON, e.g. results/scifact_doclevel_pinned_v2.json")
    parser.add_argument("--readme", type=Path, help="Optional README.md to append/update between generated markers")
    parser.add_argument("--output", type=Path, help="Optional standalone Markdown output path")
    parser.add_argument("--csv-output", type=Path, help="Optional aggregate CSV output path")
    args = parser.parse_args()

    try:
        with args.input.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Result JSON root must be an object")
        section = build_section(data, args.input)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(section, encoding="utf-8", newline="\n")
            print(f"Wrote benchmark section: {args.output}")
        if args.csv_output:
            write_csv(data, args.csv_output)
            print(f"Wrote aggregate CSV: {args.csv_output}")
        if args.readme:
            update_readme(args.readme, section)
            print(f"Updated README block: {args.readme}")
        if not args.output and not args.readme and not args.csv_output:
            print(section)
        return 0
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
