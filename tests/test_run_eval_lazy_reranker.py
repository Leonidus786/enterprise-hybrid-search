from types import SimpleNamespace

import hs.models as models
from scripts import run_eval


def _args(*, fake=False, strategies=None):
    return SimpleNamespace(
        fake=fake,
        strategies=(strategies or ["bm25", "dense", "hybrid"]),
        embedding_revision="embedding-revision",
        reranker_revision="reranker-revision",
        reranker_batch_size=4,
        reranker_device="cpu",
        reranker_max_length=256,
    )


def test_fake_runtime_omits_reranker_when_strategy_not_selected():
    embed, score, model_id, metadata = run_eval._load_eval_models(_args(fake=True))

    assert embed is not None
    assert score is None
    assert model_id == "fake-word-hash-128-v1"
    assert metadata["reranker"] == {
        "model_id": None,
        "requested_revision": None,
        "resolved_revision": None,
        "revision_status": "not_requested",
    }


def test_real_runtime_does_not_load_cross_encoder_without_hybrid_rerank(monkeypatch):
    embed = SimpleNamespace(model_metadata={"model_id": "test-embedder"})
    calls = []

    def sentence_embedder(*, revision):
        calls.append(("embedding", revision))
        return embed

    def cross_encoder_scorer(**kwargs):
        raise AssertionError("CrossEncoder must not load unless hybrid_rerank is selected")

    monkeypatch.setattr(models, "sentence_embedder", sentence_embedder)
    monkeypatch.setattr(models, "cross_encoder_scorer", cross_encoder_scorer)

    actual_embed, score, model_id, metadata = run_eval._load_eval_models(_args())

    assert actual_embed is embed
    assert score is None
    assert model_id == "BAAI/bge-small-en-v1.5"
    assert calls == [("embedding", "embedding-revision")]
    assert metadata["embedding"] == {"model_id": "test-embedder"}
    assert metadata["reranker"]["revision_status"] == "not_requested"
    assert metadata["reranker"]["model_id"] is None


def test_real_runtime_loads_cross_encoder_when_hybrid_rerank_is_selected(monkeypatch):
    embed = SimpleNamespace(model_metadata={"model_id": "test-embedder"})
    reranker_metadata = {"model_id": "test-reranker", "revision_status": "verified"}
    score = SimpleNamespace(model_metadata=reranker_metadata)
    calls = []

    def sentence_embedder(*, revision):
        calls.append(("embedding", revision))
        return embed

    def cross_encoder_scorer(**kwargs):
        calls.append(("reranker", kwargs))
        return score

    monkeypatch.setattr(models, "sentence_embedder", sentence_embedder)
    monkeypatch.setattr(models, "cross_encoder_scorer", cross_encoder_scorer)

    actual_embed, actual_score, _, metadata = run_eval._load_eval_models(
        _args(strategies=["bm25", "hybrid_rerank"])
    )

    assert actual_embed is embed
    assert actual_score is score
    assert calls == [
        ("embedding", "embedding-revision"),
        (
            "reranker",
            {
                "revision": "reranker-revision",
                "batch_size": 4,
                "device": "cpu",
                "max_length": 256,
            },
        ),
    ]
    assert metadata["reranker"] == reranker_metadata


def test_fake_runtime_keeps_fake_reranker_when_hybrid_rerank_is_selected():
    _, score, _, metadata = run_eval._load_eval_models(
        _args(fake=True, strategies=["hybrid_rerank"])
    )

    assert score is not None
    assert metadata["reranker"]["model_id"] == "fake-token-overlap-scorer-v1"
    assert metadata["reranker"]["revision_status"] == "not_applicable_fake_model"


def test_cli_defaults_to_reranker_off_but_keeps_explicit_opt_in():
    parser = run_eval.build_parser()

    defaults = parser.parse_args([])
    assert defaults.strategies == ["bm25", "dense", "hybrid"]
    assert defaults.strategies == run_eval.DEFAULT_STRATEGIES
    assert "hybrid_rerank" not in defaults.strategies
    assert "hybrid_rerank" in run_eval.STRATEGIES

    explicit = parser.parse_args(["--strategies", "hybrid_rerank"])
    assert explicit.strategies == ["hybrid_rerank"]
