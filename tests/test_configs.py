import inspect
import json
from pathlib import Path

from hs.pipeline import HybridPipeline
from hs.sparse import SparseIndex
from scripts import run_eval, run_latency


ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    return json.loads(
        (ROOT / "configs" / name).read_text(encoding="utf-8")
    )


def test_default_config_matches_code():
    cfg = _load("default.json")
    args = run_eval.build_parser().parse_args([])
    defaults = cfg["parser_defaults"]

    assert (
        defaults["k"],
        defaults["retrieval_k"],
        defaults["candidate_k"],
    ) == (
        args.k,
        args.retrieval_k,
        args.candidate_k,
    )

    assert (
        defaults["level"],
        defaults["chunk_size"],
        defaults["chunk_overlap"],
    ) == (
        args.level,
        args.chunk_size,
        args.chunk_overlap,
    )

    assert cfg["strategies"] == run_eval.STRATEGIES

    assert (
        cfg["fusion"]["rrf_k"]
        == inspect.signature(HybridPipeline).parameters["rrf_k"].default
    )

    sparse_params = inspect.signature(SparseIndex).parameters

    assert cfg["sparse"]["bm25_k1"] == sparse_params["k1"].default
    assert cfg["sparse"]["bm25_b"] == sparse_params["b"].default

    assert (
        cfg["models"]["embedding"]["query_prefix"]
        == run_eval.BGE_PREFIX
    )


def test_latency_runner_constants_match_config_and_eval():
    cfg = _load("default.json")

    assert run_latency.BGE_PREFIX == run_eval.BGE_PREFIX

    assert (
        run_latency.DEFAULT_EMBEDDING_REVISION
        == cfg["models"]["embedding"]["revision"]
    )

    assert (
        run_latency.DEFAULT_RERANKER_REVISION
        == cfg["models"]["reranker"]["revision"]
    )


def test_family_size_matches_declared_hypotheses():
    family = _load("analysis_family.json")

    assert (
        len(family["metrics"]) * len(family["pairs"])
        == family["hypotheses"]
        == 6
    )