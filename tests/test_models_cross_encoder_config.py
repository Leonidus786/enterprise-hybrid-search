from __future__ import annotations

import sys
import types

import pytest

from hs.models import cross_encoder_scorer


class _ArrayLike:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


def _install_dummy_cross_encoder(monkeypatch, captured):
    class DummyCrossEncoder:
        def __init__(self, model_id, **kwargs):
            captured["constructor"] = (model_id, kwargs)

        def predict(self, pairs, **kwargs):
            captured["predict"] = (pairs, kwargs)
            return _ArrayLike([0.25] * len(pairs))

    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = DummyCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_cross_encoder_scorer_passes_runtime_options(monkeypatch):
    captured = {}
    _install_dummy_cross_encoder(monkeypatch, captured)

    scorer = cross_encoder_scorer(
        "test/reranker",
        revision="release-tag",
        batch_size=8,
        device="cpu",
        max_length=256,
    )
    scores = scorer([("query", "document")])

    assert scores == [0.25]
    assert captured["constructor"] == (
        "test/reranker",
        {"revision": "release-tag", "device": "cpu", "max_length": 256},
    )
    assert captured["predict"] == (
        [("query", "document")],
        {"batch_size": 8, "show_progress_bar": False},
    )
    assert scorer.inference_config == {
        "batch_size": 8,
        "device": "cpu",
        "max_length": 256,
    }


def test_cross_encoder_scorer_preserves_revision_only_constructor(monkeypatch):
    captured = {}
    _install_dummy_cross_encoder(monkeypatch, captured)

    scorer = cross_encoder_scorer("test/reranker", revision="release-tag")
    assert scorer([("q", "d")]) == [0.25]
    assert captured["constructor"] == ("test/reranker", {"revision": "release-tag"})
    assert captured["predict"][1] == {
        "batch_size": 8,
        "show_progress_bar": False,
    }


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "8"])
def test_cross_encoder_scorer_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        cross_encoder_scorer(batch_size=batch_size)


@pytest.mark.parametrize("max_length", [0, -1, True, 1.5, "256"])
def test_cross_encoder_scorer_rejects_invalid_max_length(max_length):
    with pytest.raises(ValueError, match="max_length"):
        cross_encoder_scorer(max_length=max_length)


def test_cross_encoder_scorer_rejects_invalid_device():
    with pytest.raises(ValueError, match="device"):
        cross_encoder_scorer(device="   ")
