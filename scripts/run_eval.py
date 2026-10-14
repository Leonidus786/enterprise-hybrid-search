import json
import platform
import subprocess
from datetime import datetime, timezone

"""Evaluate BM25, dense, hybrid and reranked retrieval.

Real models are used by default.
--fake is an offline smoke test, NOT benchmark evidence.
"""

import argparse
import re
import sys
import zlib
from pathlib import Path

import numpy as np

# Support running this script directly from the repository.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from hs.pipeline import HybridPipeline  # noqa: E402
from hs.io import load_chunks, load_queries  # noqa: E402
from hs.evaluate import evaluate  # noqa: E402
from hs.cache import CachedEmbedder  # noqa: E402


BGE_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)

STRATEGIES = [
    "bm25",
    "dense",
    "hybrid",
    "hybrid_rerank",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate enterprise hybrid search strategies."
    )

    parser.add_argument(
        "--corpus",
        default="data/sample/corpus.jsonl",
    )
    parser.add_argument(
        "--queries",
        default="data/sample/queries.json",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Cutoff for ranking metrics",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=100,
        help="Number of results retrieved for recall evaluation",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="Candidates passed to the cross-encoder reranker",
    )
    parser.add_argument(
        "--level",
        choices=["doc", "chunk"],
        default="doc",
    )

    chunk_group = parser.add_mutually_exclusive_group()

    chunk_group.add_argument(
        "--no-chunk",
        action="store_true",
        help="Use exactly one searchable item per document",
    )
    chunk_group.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Words per chunk (default: 200)",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=40,
        help="Words of overlap between chunks (default: 40)",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Use fake models for an offline smoke test",
    )
    parser.add_argument(
        "--no-bge-prefix",
        action="store_true",
        help="Disable the BGE query instruction",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/embeddings",
        help="Directory for persistent embedding vectors",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable persistent embedding caching",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Evaluate only the first N queries",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        default=STRATEGIES,
        help="Strategies to evaluate",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save benchmark results as JSON",
    )
    return parser


def _fakes():
    def embed(texts):
        output = np.zeros(
            (len(texts), 128),
            dtype=np.float32,
        )

        for i, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                index = zlib.crc32(word.encode()) % 128
                output[i, index] += 1

        return output

    def score(pairs):
        return [
            len(
                set(re.findall(r"\w+", query.lower()))
                & set(re.findall(r"\w+", doc.lower()))
            )
            for query, doc in pairs
        ]

    return embed, score


def _git_info():
    repo_root = Path(__file__).resolve().parent.parent

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "working_tree_dirty": (
            bool(status.stdout.strip())
            if status.returncode == 0
            else None
        ),
    }


def save_results_json(path, results, args, chunks, total_queries, evaluated_queries):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_info(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "config": {
            "strategies": args.strategies,
            "k": args.k,
            "retrieval_k": args.retrieval_k,
            "candidate_k": args.candidate_k,
            "level": args.level,
            "chunking": {
                "enabled": not args.no_chunk,
                "size": None if args.no_chunk else args.chunk_size,
                "overlap": args.chunk_overlap,
            },
            "fake_models": args.fake,
            "embedding_model": (
                "fake-word-hash-128-v1"
                if args.fake
                else "BAAI/bge-small-en-v1.5"
            ),
            "cache_enabled": not args.no_cache,
            "bge_prefix_enabled": not (
                args.fake or args.no_bge_prefix
            ),
        },
        "dataset": {
            "corpus_path": str(Path(args.corpus)),
            "queries_path": str(Path(args.queries)),
            "searchable_items": len(chunks),
            "unique_documents": len({chunk.doc_id for chunk in chunks}),
            "total_queries": total_queries,
            "evaluated_queries": evaluated_queries,
        },
        "metrics": results,
    }

    output_path.write_text(
        json.dumps(document, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Results JSON saved: {output_path}")


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.k < 1:
        raise ValueError("--k must be >= 1")

    if args.retrieval_k < args.k:
        raise ValueError("--retrieval-k must be >= --k")

    if args.candidate_k < 1:
        raise ValueError("--candidate-k must be >= 1")

    if args.max_queries is not None and args.max_queries < 1:
        raise ValueError("--max-queries must be >= 1")

    if args.fake:
        embed, score = _fakes()
        embedding_model_id = "fake-word-hash-128-v1"
    else:
        from hs.models import (
            cross_encoder_scorer,
            sentence_embedder,
        )

        embed = sentence_embedder()
        score = cross_encoder_scorer()
        embedding_model_id = "BAAI/bge-small-en-v1.5"

    if not args.no_cache:
        embed = CachedEmbedder(
            embed_fn=embed,
            cache_dir=args.cache_dir,
            model_id=embedding_model_id,
        )

    prefix = (
        ""
        if args.fake or args.no_bge_prefix
        else BGE_PREFIX
    )

    # --no-chunk means one searchable item per document.
    # Otherwise, preserve the existing default chunking.
    chunk_size = None if args.no_chunk else args.chunk_size

    chunks = load_chunks(
        args.corpus,
        size=chunk_size,
        overlap=args.chunk_overlap,
    )

    pipeline = HybridPipeline(
        chunks,
        embed,
        score_fn=score,
        candidate_k=args.candidate_k,
        query_prefix=prefix,
    )

    queries = load_queries(args.queries)
    total_queries = len(queries)

    if args.max_queries is not None:
        queries = queries[:args.max_queries]

    print(
        f"Evaluating {len(queries)} of {total_queries} queries"
    )
    print(f"Strategies: {', '.join(args.strategies)}")
    print(f"Reranker candidate-k: {args.candidate_k}")

    if args.no_chunk:
        print("Chunking: disabled (one item per document)")
    else:
        print(
            f"Chunking: size={args.chunk_size}, "
            f"overlap={args.chunk_overlap}"
        )

    results = evaluate(
        pipeline,
        queries,
        args.strategies,
        k=args.k,
        level=args.level,
        retrieval_k=args.retrieval_k,
    )
    if args.output:
        save_results_json(
            args.output,
            results,
            args,
            chunks,
            total_queries,
            len(queries),
        )
    if not results:
        raise RuntimeError("Evaluation returned no results.")

    columns = list(next(iter(results.values())))
    width = max(
        20,
        max(len(column) for column in columns) + 2,
    )

    print(
        f"{'strategy':<18}"
        + "".join(
            f"{column:>{width}}"
            for column in columns
        )
    )

    for mode, metrics in results.items():
        print(
            f"{mode:<18}"
            + "".join(
                f"{metrics[column]:>{width}.3f}"
                for column in columns
            )
        )

    if (
        not args.no_cache
        and isinstance(embed, CachedEmbedder)
    ):
        print(f"\nEmbedding cache: {args.cache_dir}")
        print(f"Cache statistics: {embed.stats}")

    return results


if __name__ == "__main__":
    main()
