import inspect
import json
from pathlib import Path

from hs.models import cross_encoder_scorer, sentence_embedder
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
    assert cfg["default_strategies"] == run_eval.DEFAULT_STRATEGIES

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



def test_model_configuration_matches_adapters_and_runner_policies():
    cfg = _load("default.json")
    embedding = cfg["models"]["embedding"]
    reranker = cfg["models"]["reranker"]
    embedding_params = inspect.signature(sentence_embedder).parameters
    reranker_params = inspect.signature(cross_encoder_scorer).parameters

    assert embedding["id"] == embedding_params["name"].default
    assert embedding["normalize"] is True
    assert embedding["batch_size"] == 64
    assert embedding["revision"] == run_latency.DEFAULT_EMBEDDING_REVISION
    assert embedding_params["revision"].default is None

    assert reranker["id"] == reranker_params["name"].default
    assert reranker["revision"] == run_latency.DEFAULT_RERANKER_REVISION
    assert reranker["batch_size"] == reranker_params["batch_size"].default
    assert reranker["device"] == reranker_params["device"].default
    assert reranker["max_length"] == reranker_params["max_length"].default

    eval_args = run_eval.build_parser().parse_args([])
    assert eval_args.embedding_revision is None
    assert eval_args.reranker_revision is None
    assert eval_args.reranker_batch_size == reranker_params["batch_size"].default
    assert eval_args.reranker_device == reranker_params["device"].default
    assert eval_args.reranker_max_length == reranker_params["max_length"].default


def test_canonical_protocol_is_representable_by_evaluation_cli():
    protocol = _load("default.json")["canonical_protocol"]
    assert protocol["dataset"] == "BEIR SciFact test"

    argv = [
        "--corpus", protocol["corpus"],
        "--queries", protocol["queries"],
        "--level", protocol["level"],
    ]
    for key, flag in (
        ("no_chunk", "--no-chunk"),
        ("no_cache", "--no-cache"),
        ("no_bge_prefix", "--no-bge-prefix"),
    ):
        if protocol[key]:
            argv.append(flag)

    args = run_eval.build_parser().parse_args(argv)
    assert args.corpus == protocol["corpus"]
    assert args.queries == protocol["queries"]
    assert args.level == protocol["level"]
    assert args.no_chunk is protocol["no_chunk"]
    assert args.no_cache is protocol["no_cache"]
    assert args.no_bge_prefix is protocol["no_bge_prefix"]


def test_latency_parser_defaults_match_manifest():
    expected = _load("default.json")["latency_defaults"]
    actual = vars(
        run_latency.build_parser().parse_args(["--output", "unused.json"])
    )
    actual.pop("output")
    assert actual == expected

def test_family_size_matches_declared_hypotheses():
    family = _load("analysis_family.json")

    assert (
        len(family["metrics"]) * len(family["pairs"])
        == family["hypotheses"]
        == 6
    )