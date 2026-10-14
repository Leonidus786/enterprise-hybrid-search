"""Step 1b: on-disk embedding cache.

Contract to implement (hs/cache.py):
    CachedEmbedder(embed_fn, cache_dir, model_id)
      .__call__(texts) -> np.ndarray float32, shape (len(texts), dim), same row order as texts
      .stats -> {"hits": int, "misses": int}   (counted per UNIQUE text per call)
"""
import numpy as np
import pytest
from hs.cache import CachedEmbedder
from hs.chunking import Chunk
from hs.dense import DenseIndex


class Counting:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return self.fn(texts)


@pytest.fixture
def counting(embed):
    return Counting(embed)


def make(counting, tmp_path, model_id="m1"):
    return CachedEmbedder(counting, tmp_path / "cache", model_id)


def test_returns_exactly_what_the_wrapped_embedder_returns(counting, embed, tmp_path):
    out = make(counting, tmp_path)(["alpha beta", "gamma", "delta epsilon"])
    np.testing.assert_allclose(out, embed(["alpha beta", "gamma", "delta epsilon"]))
    assert out.dtype == np.float32


def test_second_call_with_same_texts_hits_the_cache(counting, tmp_path):
    e = make(counting, tmp_path)
    e(["a b", "c d"])
    e(["a b", "c d"])
    assert len(counting.calls) == 1


def test_only_missing_texts_are_sent_to_the_model(counting, embed, tmp_path):
    e = make(counting, tmp_path)
    e(["a", "b"])
    out = e(["b", "c"])
    assert counting.calls[-1] == ["c"]
    np.testing.assert_allclose(out, embed(["b", "c"]))      # order preserved across hit + miss


def test_duplicates_inside_one_call_are_embedded_once(counting, embed, tmp_path):
    out = make(counting, tmp_path)(["a", "a", "b"])
    assert sorted(counting.calls[0]) == ["a", "b"]
    assert out.shape[0] == 3
    np.testing.assert_allclose(out, embed(["a", "a", "b"]))


def test_cache_persists_across_instances(counting, tmp_path):
    make(counting, tmp_path)(["a", "b"])
    n = len(counting.calls)
    make(counting, tmp_path)(["a", "b"])                    # brand-new object, same directory
    assert len(counting.calls) == n
    assert any(p.is_file() for p in (tmp_path / "cache").rglob("*"))


def test_different_model_id_does_not_share_entries(counting, tmp_path):
    make(counting, tmp_path, "model-A")(["a"])
    make(counting, tmp_path, "model-B")(["a"])
    assert len(counting.calls) == 2


def test_texts_are_case_and_whitespace_sensitive_keys(counting, tmp_path):
    e = make(counting, tmp_path)
    e(["Python", "python", "python "])
    assert sorted(counting.calls[0]) == ["Python", "python", "python "]


def test_empty_input_returns_zero_rows_without_calling_the_model(counting, tmp_path):
    out = make(counting, tmp_path)([])
    assert out.shape[0] == 0 and counting.calls == []


def test_stats_count_unique_hits_and_misses(counting, tmp_path):
    e = make(counting, tmp_path)
    e(["a", "b"])
    e(["b", "c"])
    assert e.stats == {"hits": 1, "misses": 3}


def test_wrong_row_count_raises_and_does_not_poison_the_cache(embed, tmp_path):
    bad = Counting(lambda texts: np.ones((1, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        CachedEmbedder(bad, tmp_path / "cache", "m")(["a", "b", "c"])
    good = Counting(embed)
    CachedEmbedder(good, tmp_path / "cache", "m")(["a", "b", "c"])
    assert sorted(good.calls[0]) == ["a", "b", "c"]          # nothing was stored by the failed run


def test_non_finite_vectors_raise_and_are_not_cached(embed, tmp_path):
    bad = Counting(lambda texts: np.full((len(texts), 4), np.nan, dtype=np.float32))
    with pytest.raises(ValueError):
        CachedEmbedder(bad, tmp_path / "cache", "m")(["a"])
    good = Counting(embed)
    CachedEmbedder(good, tmp_path / "cache", "m")(["a"])
    assert good.calls == [["a"]]


def test_corrupted_cache_files_are_ignored_and_rebuilt(counting, embed, tmp_path):
    make(counting, tmp_path)(["a", "b"])
    files = [p for p in (tmp_path / "cache").rglob("*") if p.is_file()]
    assert files
    for p in files:
        p.write_bytes(b"garbage, not an npz")
    again = Counting(embed)
    out = CachedEmbedder(again, tmp_path / "cache", "m1")(["a", "b"])
    assert len(again.calls) == 1
    np.testing.assert_allclose(out, embed(["a", "b"]))


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_model_id_must_be_a_non_empty_string(counting, tmp_path, bad):
    with pytest.raises(ValueError):
        CachedEmbedder(counting, tmp_path / "cache", bad)


def test_works_as_the_embedder_of_a_dense_index(counting, embed, tmp_path):
    chunks = [Chunk("d1#0", "d1", "kubernetes pods scheduling"), Chunk("d2#0", "d2", "baking sourdough bread")]
    cached = make(counting, tmp_path)
    first = DenseIndex(chunks, cached).search("sourdough bread", k=1)
    docs_embedded = len(counting.calls)
    second = DenseIndex(chunks, cached).search("sourdough bread", k=1)   # rebuild: docs AND query cached
    assert first == second == DenseIndex(chunks, embed).search("sourdough bread", k=1)
    assert len(counting.calls) == docs_embedded
