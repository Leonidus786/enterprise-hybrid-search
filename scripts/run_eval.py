"""Evaluate BM25, dense, hybrid and reranked retrieval.

Real models are used by default. ``--fake`` is an offline smoke test and is
not benchmark evidence. Saved Step 3 result files contain both aggregate and
per-query ranking metrics so that consistency and paired comparisons can be
checked without re-running the models.
"""
import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Support running this script directly from the repository.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hs.cache import CachedEmbedder  # noqa: E402
from hs.compare import check_consistency  # noqa: E402
from hs.evaluate import evaluate_with_details  # noqa: E402
from hs.io import load_chunks, load_queries  # noqa: E402
from hs.pipeline import HybridPipeline  # noqa: E402
from hs.provenance import build_run_provenance  # noqa: E402


BGE_PREFIX = "Represent this sentence for searching relevant passages: "
STRATEGIES = ["bm25", "dense", "hybrid", "hybrid_rerank"]


def _embedding_cache_identity(model_metadata, *, fake=False):
    """Return a safe persistent-cache identity, or None when revision is unknown."""
    metadata = (
        model_metadata.get("embedding")
        if isinstance(model_metadata, dict)
        else None
    )
    if not isinstance(metadata, dict):
        return None

    revision = metadata.get("resolved_revision")
    if not isinstance(revision, str) or not revision.strip():
        return None

    if fake:
        if metadata.get("revision_status") != "not_applicable_fake_model":
            return None
        config = {
            "adapter": "fake-word-hash-128-v1",
            "config_version": 1,
            "dimension": 128,
            "token_pattern": r"\w+",
            "lowercase": True,
            "hash": "crc32",
        }
    else:
        # A mutable model name/branch is not a durable cache identity. Require
        # the loader to have verified the actual immutable snapshot revision.
        if metadata.get("revision_verified") is not True:
            return None
        config = {
            "adapter": "hs.models.sentence_embedder",
            "config_version": 1,
            "normalize_embeddings": True,
            "batch_size": 64,
        }

    return revision.strip(), config


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate enterprise hybrid search strategies."
    )
    parser.add_argument("--corpus", default="data/sample/corpus.jsonl")
    parser.add_argument("--queries", default="data/sample/queries.json")
    parser.add_argument("--k", type=int, default=10, help="Cutoff for ranking metrics")
    parser.add_argument(
        "--retrieval-k", type=int, default=100,
        help="Number of results retrieved for recall evaluation",
    )
    parser.add_argument(
        "--candidate-k", type=int, default=50,
        help="Candidates passed to the cross-encoder reranker",
    )
    parser.add_argument(
        "--reranker-batch-size", type=int, default=8,
        help="CrossEncoder inference batch size (provisional CPU-oriented default: 8)",
    )
    parser.add_argument(
        "--reranker-device", default=None,
        help="Optional CrossEncoder device, e.g. cpu or cuda; default selects automatically",
    )
    parser.add_argument(
        "--reranker-max-length", type=int, default=None,
        help="Optional CrossEncoder maximum input length; default uses the model setting",
    )
    parser.add_argument("--level", choices=["doc", "chunk"], default="doc")
    chunk_group = parser.add_mutually_exclusive_group()
    chunk_group.add_argument(
        "--no-chunk", action="store_true",
        help="Use exactly one searchable item per document",
    )
    chunk_group.add_argument(
        "--chunk-size", type=int, default=200,
        help="Words per chunk (default: 200)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=40,
        help="Words of overlap between chunks (default: 40)",
    )
    parser.add_argument("--fake", action="store_true", help="Use fake models for an offline smoke test")
    parser.add_argument(
        "--embedding-revision", default=None,
        help="Optional Hugging Face revision/commit for the embedding model",
    )
    parser.add_argument(
        "--reranker-revision", default=None,
        help="Optional Hugging Face revision/commit for the cross-encoder model",
    )
    parser.add_argument("--no-bge-prefix", action="store_true", help="Disable the BGE query instruction")
    parser.add_argument(
        "--cache-dir", default=".cache/embeddings",
        help="Directory for persistent embedding vectors",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable persistent embedding caching")
    parser.add_argument(
        "--max-queries", type=int, default=None,
        help="Evaluate only the first N queries",
    )
    parser.add_argument(
        "--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES,
        help="Strategies to evaluate",
    )
    parser.add_argument("--output", default=None, help="Optional path to save benchmark results as JSON")
    parser.add_argument(
        "--include-query-text", action="store_true",
        help="Store raw query text in per-query traces (may expose sensitive data)",
    )
    return parser


def _fakes():
    def embed(texts):
        output = np.zeros((len(texts), 128), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                index = zlib.crc32(word.encode()) % 128
                output[i, index] += 1
        return output

    def score(pairs):
        return [
            len(set(re.findall(r"\w+", query.lower())) & set(re.findall(r"\w+", doc.lower())))
            for query, doc in pairs
        ]

    return embed, score


def _git_info():
    repo_root = Path(__file__).resolve().parent.parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root,
        capture_output=True, text=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root,
        capture_output=True, text=True, check=False,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "working_tree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _query_fingerprint(queries):
    """Fingerprint the exact evaluated query text and relevance labels."""
    canonical = []
    for item in queries:
        row = {
            "id": item.get("id"),
            "query": item.get("query"),
            "relevant": sorted(item.get("relevant", [])),
        }
        grades = item.get("grades")
        if grades is not None:
            row["grades"] = grades
        canonical.append(row)
    # Query order is not part of logical query/qrels identity. Sorting rows by
    # unique query ID makes the fingerprint stable when the same set is reordered.
    canonical.sort(key=lambda row: row["id"])
    payload = json.dumps(
        canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_results_json(
    path, results, args, chunks, total_queries, evaluated_queries,
    per_query=None, query_fingerprint=None, model_metadata=None,
):
    """Save metrics, ranking traces, and a verifiable provenance envelope."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep this helper safe if called directly, not only through main().
    has_per_query = per_query is not None
    if has_per_query:
        consistency_issues = check_consistency({"metrics": results, "per_query": per_query})
        if consistency_issues:
            raise ValueError("aggregate/per-query metrics are inconsistent: " + " | ".join(consistency_issues))

    config = {
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
        "embedding_model": "fake-word-hash-128-v1" if args.fake else "BAAI/bge-small-en-v1.5",
        "reranker_model": None if args.fake else "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "embedding_revision_requested": getattr(args, "embedding_revision", None),
        "reranker_revision_requested": getattr(args, "reranker_revision", None),
        "reranker_inference": {
            "batch_size": getattr(args,"reranker_batch_size",8),
            "device": getattr(args,"reranker_device",None),
            "max_length": getattr(args,"reranker_max_length",None),
        },
        "cache_enabled": bool(getattr(args, "cache_effective", not args.no_cache)),
        "cache_dir": (
            str(Path(getattr(args, "cache_dir", ".cache/embeddings")))
            if bool(getattr(args, "cache_effective", not args.no_cache))
            else None
        ),
        "cache_disabled_reason": getattr(args, "cache_disabled_reason", None),
        "bge_prefix_enabled": not (args.fake or args.no_bge_prefix),
        "query_text_included": bool(getattr(args, "include_query_text", False)),
    }
    dataset = {
        "corpus_path": str(Path(args.corpus)),
        "queries_path": str(Path(args.queries)),
        "searchable_items": len(chunks),
        "unique_documents": len({chunk.doc_id for chunk in chunks}),
        "total_queries": total_queries,
        "evaluated_queries": evaluated_queries,
        "query_fingerprint": query_fingerprint,
    }
    repo_root = Path(__file__).resolve().parent.parent
    provenance = build_run_provenance(
        repo_root=repo_root,
        corpus_path=args.corpus,
        queries_path=args.queries,
        run_config=config,
        query_fingerprint=query_fingerprint,
        model_metadata=model_metadata,
    )
    dataset["corpus_sha256"] = provenance["inputs"]["corpus"]["sha256"]
    dataset["queries_file_sha256"] = provenance["inputs"]["queries_and_relevance_labels"]["sha256"]

    document = {
        # Schema 4 adds data/model/source/runtime provenance to schema 3 traces.
        "schema_version": 4,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_info(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "config": config,
        "dataset": dataset,
        "provenance": provenance,
        "metrics": results,
    }
    if has_per_query:
        document["ranking_trace"] = {
            "schema_version": 1,
            "ranked_ids_key": "per_query.<strategy>.<query_id>._trace.ranked_ids",
            "candidate_ids_key": "per_query.<strategy>.<query_id>._trace.candidate_ids",
            "id_space": "document IDs when config.level=doc; chunk IDs when config.level=chunk",
            "candidate_semantics": "pre-reranking candidates mapped to evaluation ID space",
            "query_text_included": bool(getattr(args, "include_query_text", False)),
            "retrieval_scores_included": False,
        }
        document["per_query"] = per_query
    output_path.write_text(
        json.dumps(document, indent=2, allow_nan=False), encoding="utf-8"
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
    if args.reranker_batch_size < 1:
        raise ValueError("--reranker-batch-size must be >= 1")
    if args.reranker_max_length is not None and args.reranker_max_length < 1:
        raise ValueError("--reranker-max-length must be >= 1")
    if args.reranker_device is not None and not args.reranker_device.strip():
        raise ValueError("--reranker-device must be non-empty when provided")
    if args.max_queries is not None and args.max_queries < 1:
        raise ValueError("--max-queries must be >= 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    if args.chunk_overlap < 0:
        raise ValueError("--chunk-overlap must be >= 0")

    if args.fake:
        embed, score = _fakes()
        embedding_model_id = "fake-word-hash-128-v1"
        model_metadata = {
            "embedding": {
                "model_id": embedding_model_id,
                "requested_revision": None,
                "resolved_revision": "fake-word-hash-128-v1",
                "revision_status": "not_applicable_fake_model",
            },
            "reranker": {
                "model_id": "fake-token-overlap-scorer-v1",
                "requested_revision": None,
                "resolved_revision": "fake-token-overlap-scorer-v1",
                "revision_status": "not_applicable_fake_model",
            },
        }
    else:
        from hs.models import cross_encoder_scorer, sentence_embedder
        embed = sentence_embedder(revision=args.embedding_revision)
        score = cross_encoder_scorer(
            revision=args.reranker_revision,
            batch_size=args.reranker_batch_size,
            device=args.reranker_device,
            max_length=args.reranker_max_length,
        )
        embedding_model_id = "BAAI/bge-small-en-v1.5"
        model_metadata = {
            "embedding": getattr(embed, "model_metadata", {
                "model_id": embedding_model_id,
                "requested_revision": args.embedding_revision,
                "resolved_revision": None,
                "revision_status": "unknown_model_loader_metadata_unavailable",
            }),
            "reranker": getattr(score, "model_metadata", {
                "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "requested_revision": args.reranker_revision,
                "resolved_revision": None,
                "revision_status": "unknown_model_loader_metadata_unavailable",
            }),
        }

    cache_enabled = False
    cache_disabled_reason = None
    if not args.no_cache:
        cache_identity = _embedding_cache_identity(model_metadata, fake=args.fake)
        if cache_identity is None:
            cache_disabled_reason = "embedding_revision_unverified"
            print(
                "Embedding cache disabled: the loaded model's immutable revision "
                "could not be verified."
            )
        else:
            model_revision, embedding_config = cache_identity
            embed = CachedEmbedder(
                embed_fn=embed,
                cache_dir=args.cache_dir,
                model_id=embedding_model_id,
                model_revision=model_revision,
                embedding_config=embedding_config,
            )
            cache_enabled = True

    # Keep the saved run-provenance envelope honest when caching was requested
    # but deliberately disabled because the actual model revision was unknown.
    args.cache_effective = cache_enabled
    args.cache_disabled_reason = cache_disabled_reason

    prefix = "" if args.fake or args.no_bge_prefix else BGE_PREFIX
    chunk_size = None if args.no_chunk else args.chunk_size
    chunks = load_chunks(args.corpus, size=chunk_size, overlap=args.chunk_overlap)
    pipeline = HybridPipeline(
        chunks, embed, score_fn=score, candidate_k=args.candidate_k, query_prefix=prefix
    )

    queries = load_queries(args.queries)
    total_queries = len(queries)
    if args.max_queries is not None:
        queries = queries[:args.max_queries]

    print(f"Evaluating {len(queries)} of {total_queries} queries")
    print(f"Strategies: {', '.join(args.strategies)}")
    print(f"Reranker candidate-k: {args.candidate_k}")
    if args.no_chunk:
        print("Chunking: disabled (one item per document)")
    else:
        print(f"Chunking: size={args.chunk_size}, overlap={args.chunk_overlap}")

    evaluation = evaluate_with_details(
        pipeline, queries, args.strategies, k=args.k,
        level=args.level, retrieval_k=args.retrieval_k,
        include_query_text=args.include_query_text,
    )
    results = evaluation["metrics"]
    per_query = evaluation["per_query"]

    issues = check_consistency({"metrics": results, "per_query": per_query})
    if issues:
        raise RuntimeError("aggregate/per-query consistency check failed: " + " | ".join(issues))
    print(f"Per-query consistency: PASS ({sum(len(rows) for rows in per_query.values())} strategy-query scores)")

    if args.output:
        save_results_json(
            args.output, results, args, chunks, total_queries, len(queries),
            per_query=per_query, query_fingerprint=_query_fingerprint(queries),
            model_metadata=model_metadata,
        )
    if not results:
        raise RuntimeError("Evaluation returned no results.")

    columns = list(next(iter(results.values())))
    width = max(20, max(len(column) for column in columns) + 2)
    print(f"{'strategy':<18}" + "".join(f"{column:>{width}}" for column in columns))
    for mode, metrics in results.items():
        print(f"{mode:<18}" + "".join(f"{metrics[column]:>{width}.3f}" for column in columns))

    if cache_enabled and isinstance(embed, CachedEmbedder):
        print(f"\nEmbedding cache: {args.cache_dir}")
        print(f"Cache statistics: {embed.stats}")
    return results


if __name__ == "__main__":
    main()
