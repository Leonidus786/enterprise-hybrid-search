#!/usr/bin/env python
"""Diagnostic CrossEncoder batch-size profiler for enterprise-hybrid-search.

Measures only CrossEncoder.predict on real hybrid candidate pairs. Model loading,
corpus indexing, embedding and candidate retrieval are excluded from timed samples.
This is an exploratory tuning tool, not a publishable latency benchmark.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BGE_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DEFAULT_RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/scifact_test/corpus.jsonl")
    parser.add_argument("--queries", default="data/scifact_test/queries.json")
    parser.add_argument("--query-count", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", default=".cache/embeddings")
    parser.add_argument("--embedding-revision", default=DEFAULT_EMBEDDING_REVISION)
    parser.add_argument("--reranker-revision", default=DEFAULT_RERANKER_REVISION)
    parser.add_argument("--output", default="results/reranker_batch_profile_v1.json")
    args = parser.parse_args()
    if args.query_count < 1 or args.candidate_k < 1 or args.repeats < 1:
        parser.error("query-count, candidate-k and repeats must be positive")
    if not args.batch_sizes or any(x < 1 for x in args.batch_sizes):
        parser.error("batch-sizes must contain positive integers")
    return args


def _as_scores(value: Any, expected: int) -> np.ndarray:
    scores = np.asarray(value)
    if scores.ndim != 1 or scores.shape[0] != expected:
        raise ValueError(
            f"Expected one score per pair with shape ({expected},); got {scores.shape}"
        )
    scores = scores.astype(np.float64, copy=False)
    if not np.isfinite(scores).all():
        raise ValueError("CrossEncoder returned non-finite scores")
    return scores


def main() -> int:
    args = parse_args()

    from sentence_transformers import CrossEncoder
    from hs.cache import CachedEmbedder
    from hs.io import load_chunks, load_queries
    from hs.models import sentence_embedder
    from hs.pipeline import HybridPipeline

    chunks = load_chunks(args.corpus, size=None, overlap=40)
    queries = load_queries(args.queries)[: args.query_count]
    if not queries:
        raise RuntimeError("No queries loaded")

    embed = sentence_embedder(revision=args.embedding_revision)
    embed = CachedEmbedder(
        embed_fn=embed, cache_dir=args.cache_dir, model_id="BAAI/bge-small-en-v1.5"
    )
    pipeline = HybridPipeline(
        chunks,
        embed,
        candidate_k=args.candidate_k,
        query_prefix=BGE_PREFIX,
    )

    # Candidate retrieval is intentionally completed before inference timing.
    query_pairs: list[tuple[str, list[tuple[str, str]]]] = []
    for item in queries:
        pipeline.search(item["query"], mode="hybrid", k=100)
        candidate_ids = list(pipeline.last_candidate_ids)[: args.candidate_k]
        if not candidate_ids:
            raise RuntimeError(f"No hybrid candidates for query {item.get('id')}")
        pairs = [(item["query"], pipeline._by_id[cid].text) for cid in candidate_ids]
        query_pairs.append((str(item.get("id", "unknown")), pairs))

    model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        revision=args.reranker_revision,
        device=args.device,
    )

    results: dict[str, Any] = {}
    all_scores: dict[int, dict[str, np.ndarray]] = {}
    for batch_size in args.batch_sizes:
        scores_for_size: dict[str, np.ndarray] = {}
        durations_ms: list[float] = []
        score_deltas: list[float] = []
        pair_count = 0
        for query_id, pairs in query_pairs:
            # One unmeasured warm-up for every query/batch-size combination.
            model.predict(pairs[: min(4, len(pairs))], batch_size=batch_size,
                          show_progress_bar=False)
            for repeat in range(args.repeats):
                started = time.perf_counter()
                raw_scores = model.predict(
                    pairs,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                scores = _as_scores(raw_scores, len(pairs))
                durations_ms.append(elapsed_ms)
                pair_count += len(pairs)
                key = f"{query_id}:repeat-{repeat}"
                scores_for_size[key] = scores

        all_scores[batch_size] = scores_for_size
        results[str(batch_size)] = {
            "timed_calls": len(durations_ms),
            "pairs_scored_in_timed_calls": pair_count,
            "p50_ms": float(statistics.median(durations_ms)),
            "min_ms": float(min(durations_ms)),
            "max_ms": float(max(durations_ms)),
            "score_max_abs_delta_vs_batch_32": None,
            "durations_ms": durations_ms,
        }
    if 32 in all_scores:
        reference_scores = all_scores[32]
        for batch_size, rows in all_scores.items():
            if batch_size == 32:
                results[str(batch_size)]["score_max_abs_delta_vs_batch_32"] = 0.0
                continue
            deltas = [
                float(np.max(np.abs(scores - reference_scores[key])))
                for key, scores in rows.items() if key in reference_scores
            ]
            results[str(batch_size)]["score_max_abs_delta_vs_batch_32"] = (
                float(max(deltas)) if deltas else None
            )

    for batch_size in args.batch_sizes:
        print(
            f"batch_size={batch_size:<3} "
            f"P50={results[str(batch_size)]['p50_ms']:.2f} ms "
            f"min={results[str(batch_size)]['min_ms']:.2f} ms "
            f"max={results[str(batch_size)]['max_ms']:.2f} ms "
            f"max_score_delta_vs_32={results[str(batch_size)]['score_max_abs_delta_vs_batch_32']}"
        )

    try:
        import torch
        torch_info = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
        }
    except Exception as exc:  # pragma: no cover - diagnostic only
        torch_info = {"error": type(exc).__name__}

    document = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Exploratory predict-only CrossEncoder batch-size diagnostic; not final latency evidence",
        "config": {
            "query_count": len(query_pairs),
            "candidate_k": args.candidate_k,
            "repeats": args.repeats,
            "batch_sizes": args.batch_sizes,
            "device": args.device,
            "embedding_revision": args.embedding_revision,
            "reranker_revision": args.reranker_revision,
            "model_loading_timed": False,
            "candidate_retrieval_timed": False,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch_info,
        },
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved diagnostic: {output}")
    print("NOTE: Small exploratory sample only; do not publish as benchmark evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
