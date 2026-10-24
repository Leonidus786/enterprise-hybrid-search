"""Repeatable latency protocol for hybrid-search strategies.

Query embeddings are deliberately uncached. Document embeddings may use the
persistent CachedEmbedder; index construction and warm-ups are outside measured
samples. This script reports timing evidence, not retrieval-quality evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hs.cache import CachedEmbedder  # noqa: E402
from hs.evaluate import percentile  # noqa: E402
from hs.io import load_chunks, load_queries  # noqa: E402
from hs.pipeline import MODES, HybridPipeline  # noqa: E402
from hs.provenance import build_run_provenance  # noqa: E402

BGE_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DEFAULT_RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")


def validate_trials(trials: int) -> None:
    """Require enough repeats for the declared latency protocol."""
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 5:
        raise ValueError("trials must be >= 5 for a valid latency protocol")


def require_verified_models(model_metadata):
    """Refuse a latency run unless every model revision was verified from the loaded model."""
    for name, metadata in model_metadata.items():
        if (
            not isinstance(metadata, dict)
            or metadata.get("verified_from_loaded_model") is not True
            or str(metadata.get("resolved_revision", "")).lower()
            != str(metadata.get("requested_revision", "")).lower()
        ):
            raise RuntimeError(
                f"{name} model revision not verified; refusing latency run: {metadata}"
            )


def rotated_order(strategies: Sequence[str], trial: int) -> list[str]:
    """Return a deterministic round-robin strategy order for one trial."""
    ordered = list(strategies)
    if not ordered:
        raise ValueError("at least one strategy is required")
    if isinstance(trial, bool) or not isinstance(trial, int) or trial < 0:
        raise ValueError("trial must be a non-negative integer")
    offset = trial % len(ordered)
    return ordered[offset:] + ordered[:offset]


def summarize_latencies(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize samples with the evaluator's linear-interpolation percentile."""
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("cannot summarize an empty latency sample")
    if any(not (value >= 0 and value < float("inf")) for value in samples):
        raise ValueError("latency samples must be finite and non-negative")
    return {
        "count": len(samples),
        "p50_ms": float(percentile(samples, 50)),
        "p95_ms": float(percentile(samples, 95)),
        "p99_ms": float(percentile(samples, 99)),
    }


def run_trials(
    pipeline: Any,
    queries: Sequence[dict[str, Any]],
    strategies: Sequence[str],
    *,
    trials: int = 5,
    warmup_queries: int = 10,
    level: str = "doc",
    k: int = 100,
) -> list[dict[str, Any]]:
    """Run warm-ups separately, then collect interleaved per-query timings.

    Warm-up calls are never included in the returned sample list. Strategies
    rotate by trial to reduce fixed-order bias; query order is stable.
    """
    validate_trials(trials)
    if isinstance(warmup_queries, bool) or not isinstance(warmup_queries, int) or warmup_queries < 0:
        raise ValueError("warmup_queries must be a non-negative integer")
    if level not in ("doc", "chunk"):
        raise ValueError('level must be "doc" or "chunk"')
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if not queries:
        raise ValueError("at least one query is required")
    strategy_list = list(strategies)
    if not strategy_list:
        raise ValueError("at least one strategy is required")
    unknown = set(strategy_list) - set(MODES)
    if unknown:
        raise ValueError(f"unsupported strategies: {sorted(unknown)}")

    search = pipeline.search_docs if level == "doc" else pipeline.search
    actual_warmups = min(warmup_queries, len(queries))

    # Warm up every selected strategy on the first N queries, outside samples.
    for query_item in queries[:actual_warmups]:
        for strategy in strategy_list:
            search(query_item["query"], mode=strategy, k=k)

    samples: list[dict[str, Any]] = []
    for trial in range(trials):
        order = rotated_order(strategy_list, trial)
        for query_index, query_item in enumerate(queries):
            query_id = query_item.get("id")
            if not isinstance(query_id, str) or not query_id.strip():
                query_id = f"query-{query_index + 1}"
            for strategy in order:
                start = time.perf_counter()
                search(query_item["query"], mode=strategy, k=k)
                total_ms = (time.perf_counter() - start) * 1000.0
                component_times = getattr(pipeline, "last_timing_ms", {}) or {}
                retrieval_ms = float(component_times.get("retrieval", 0.0))
                rerank_ms = float(component_times.get("rerank", 0.0))
                samples.append(
                    {
                        "trial": trial,
                        "strategy": strategy,
                        "query_id": query_id,
                        "total_ms": total_ms,
                        "retrieval_ms": retrieval_ms,
                        "rerank_ms": rerank_ms,
                    }
                )
    return samples


def _query_fingerprint(queries: Sequence[dict[str, Any]]) -> str:
    canonical = []
    for index, item in enumerate(queries):
        row = {
            "id": item.get("id", f"query-{index + 1}"),
            "query": item.get("query"),
            "relevant": sorted(item.get("relevant", [])),
        }
        if item.get("grades") is not None:
            row["grades"] = item["grades"]
        canonical.append(row)
    canonical.sort(key=lambda row: str(row["id"]))
    payload = json.dumps(
        canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=ROOT, capture_output=True, text=False, check=False,
    )
    if commit.returncode != 0 or status.returncode != 0:
        raise RuntimeError(
            "could not inspect Git state; run from the project repository")
    return {
        "commit": commit.stdout.strip(),
        "working_tree_dirty": bool(status.stdout),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in (
        "numpy", "torch", "sentence-transformers", "transformers", "huggingface-hub"
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _hardware_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {
        "processor": platform.processor() or None,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
    }
    try:
        import torch

        result["torch_num_threads"] = torch.get_num_threads()
        result["torch_num_interop_threads"] = torch.get_num_interop_threads()
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            result["cuda_device"] = torch.cuda.get_device_name(0)
        else:
            result["cuda_device"] = None
    except Exception as exc:  # Keep hardware capture best-effort and explicit.
        result["torch_runtime_metadata_error"] = type(exc).__name__
    return result


def _summarize_by_strategy(
    samples: Sequence[dict[str, Any]], strategies: Sequence[str]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for strategy in strategies:
        rows = [row for row in samples if row["strategy"] == strategy]
        summary[strategy] = {
            "sample_count": len(rows),
            "total": summarize_latencies([row["total_ms"] for row in rows]),
            "retrieval": summarize_latencies([row["retrieval_ms"] for row in rows]),
            "rerank": summarize_latencies([row["rerank_ms"] for row in rows]),
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repeatable latency measurement with uncached query embeddings."
    )
    parser.add_argument("--corpus", default="data/scifact_test/corpus.jsonl")
    parser.add_argument("--queries", default="data/scifact_test/queries.json")
    parser.add_argument("--level", choices=("doc", "chunk"), default="doc")
    parser.add_argument("--k", type=int, default=100,
                        help="Search depth for each timed request")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup-queries", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--strategies", nargs="+", choices=MODES, default=list(MODES)
    )
    chunk_group = parser.add_mutually_exclusive_group()
    chunk_group.add_argument("--no-chunk", action="store_true")
    chunk_group.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--chunk-overlap", type=int, default=40)
    parser.add_argument("--no-bge-prefix", action="store_true")
    parser.add_argument("--cache-dir", default=".cache/embeddings")
    parser.add_argument("--embedding-revision",
                        default=DEFAULT_EMBEDDING_REVISION)
    parser.add_argument("--reranker-revision",
                        default=DEFAULT_RERANKER_REVISION)
    parser.add_argument("--output", required=True,
                        help="New JSON output path; existing files are never overwritten")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    validate_trials(args.trials)
    if args.k < 1 or args.candidate_k < 1:
        raise ValueError("--k and --candidate-k must be positive")
    if args.warmup_queries < 0:
        raise ValueError("--warmup-queries must be >= 0")
    if args.max_queries is not None and args.max_queries < 1:
        raise ValueError("--max-queries must be >= 1")
    if args.chunk_size < 1 or args.chunk_overlap < 0:
        raise ValueError(
            "chunk size must be positive and overlap non-negative")
    if not _COMMIT_RE.fullmatch(args.embedding_revision or ""):
        raise ValueError(
            "--embedding-revision must be a full pinned commit SHA")
    if not _COMMIT_RE.fullmatch(args.reranker_revision or ""):
        raise ValueError(
            "--reranker-revision must be a full pinned commit SHA")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing latency artifact: {output_path}")

    git = _git_state()
    if git["working_tree_dirty"]:
        raise RuntimeError(
            "working tree is dirty. Commit the latency protocol first, then rerun from a clean tree."
        )

    # Load models without wrapping the query function in CachedEmbedder.
    from hs.models import cross_encoder_scorer, sentence_embedder

    embedding_model_id = "BAAI/bge-small-en-v1.5"
    reranker_model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    query_embedder = sentence_embedder(
        embedding_model_id, revision=args.embedding_revision
    )
    score_fn = cross_encoder_scorer(
        reranker_model_id, revision=args.reranker_revision
    )
    model_metadata = {
        "embedding": query_embedder.model_metadata,
        "reranker": score_fn.model_metadata,
    }
    require_verified_models(model_metadata)

    chunk_size = None if args.no_chunk else args.chunk_size
    chunks = load_chunks(
        args.corpus, size=chunk_size, overlap=args.chunk_overlap
    )
    all_queries = load_queries(args.queries)
    queries = all_queries[:args.max_queries] if args.max_queries else all_queries
    if not queries:
        raise ValueError("query dataset is empty")

    # Cache corpus/document vectors only; every query is embedded through the
    # raw query_embedder inside DenseIndex.search and is never disk-cached.
    document_embedder = CachedEmbedder(
        embed_fn=query_embedder,
        cache_dir=args.cache_dir,
        model_id=f"{embedding_model_id}@{args.embedding_revision}",
    )
    prefix = "" if args.no_bge_prefix else BGE_PREFIX
    pipeline = HybridPipeline(
        chunks,
        document_embedder,
        score_fn=score_fn,
        candidate_k=args.candidate_k,
        query_prefix=prefix,
        query_embed_fn=query_embedder,
    )

    samples = run_trials(
        pipeline,
        queries,
        args.strategies,
        trials=args.trials,
        warmup_queries=args.warmup_queries,
        level=args.level,
        k=args.k,
    )
    expected_per_strategy = args.trials * len(queries)
    summary = _summarize_by_strategy(samples, args.strategies)
    if any(summary[strategy]["sample_count"] != expected_per_strategy for strategy in args.strategies):
        raise RuntimeError(
            "sample count mismatch; refusing to save invalid protocol output")

    config = {
        "strategies": list(args.strategies),
        "level": args.level,
        "k": args.k,
        "candidate_k": args.candidate_k,
        "chunking": {
            "enabled": not args.no_chunk,
            "size": None if args.no_chunk else args.chunk_size,
            "overlap": args.chunk_overlap,
        },
        "embedding_model": embedding_model_id,
        "embedding_revision_requested": args.embedding_revision,
        "reranker_model": reranker_model_id,
        "reranker_revision_requested": args.reranker_revision,
        "document_embedding_cache_enabled": True,
        "query_embedding_cache_enabled": False,
        "document_embedding_cache_dir": str(Path(args.cache_dir)),
        "trials": args.trials,
        "warmup_queries_requested": args.warmup_queries,
        "warmup_queries_used": min(args.warmup_queries, len(queries)),
        "warmups_included_in_samples": False,
        "strategy_order": "rotate strategy order by trial; stable query order",
        "timed_stages": ["retrieval", "rerank", "total"],
    }
    provenance = build_run_provenance(
        repo_root=ROOT,
        corpus_path=args.corpus,
        queries_path=args.queries,
        run_config=config,
        query_fingerprint=_query_fingerprint(queries),
        model_metadata=model_metadata,
    )
    document = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "protocol_valid": args.trials >= 5,
            "minimum_required_trials": 5,
            "trials": args.trials,
            "warmups_excluded": True,
            "query_embedding_cache_enabled": False,
            "strategy_order_rotates_each_trial": True,
            "expected_samples_per_strategy": expected_per_strategy,
            "actual_samples_per_strategy": {
                strategy: summary[strategy]["sample_count"]
                for strategy in args.strategies
            },
        },
        "git": git,
        "environment": {
            "packages": _package_versions(),
            "hardware": _hardware_metadata(),
        },
        "config": config,
        "dataset": {
            "corpus_path": str(args.corpus),
            "queries_path": str(args.queries),
            "searchable_items": len(chunks),
            "unique_documents": len({chunk.doc_id for chunk in chunks}),
            "total_queries": len(all_queries),
            "evaluated_queries": len(queries),
            "query_fingerprint": _query_fingerprint(queries),
        },
        "model_metadata": model_metadata,
        "document_embedding_cache_stats": document_embedder.stats,
        "provenance": provenance,
        "summary": summary,
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, allow_nan=False)
    # Exclusive create ensures an earlier evidence artifact cannot be replaced.
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.write("\n")
    print(f"Latency artifact saved: {output_path}")
    print(f"Git commit: {git['commit']}")
    print(f"Protocol valid: {document['protocol']['protocol_valid']}")
    print(
        f"Warm-up queries excluded: {document['protocol']['warmups_excluded']}")
    print(
        f"Query embedding cache enabled: {config['query_embedding_cache_enabled']}")
    for strategy in args.strategies:
        print(json.dumps({strategy: summary[strategy]}, sort_keys=True))
    return document


if __name__ == "__main__":
    main()
