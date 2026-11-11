"""Step 1c: run_eval.py gets --cache-dir / --no-cache."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _script():
    spec = importlib.util.spec_from_file_location(
        "run_eval", ROOT / "scripts" / "run_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _argv(*extra):
    return ["--fake", "--k", "5",
            "--corpus", str(ROOT / "data" / "sample" / "corpus.jsonl"),
            "--queries", str(ROOT / "data" / "sample" / "queries.json"), *extra]


def test_cache_flags_exist_with_sane_defaults():
    a = _script().build_parser().parse_args([])
    assert a.cache_dir == ".cache/embeddings"
    assert a.no_cache is False


def test_cache_identity_requires_verified_revision_for_real_models():
    mod = _script()
    unverified = {
        "embedding": {
            "resolved_revision": "mutable-or-unverified",
            "revision_verified": False,
            "revision_status": "unknown_not_exposed_by_loader",
        }
    }
    verified = {
        "embedding": {
            "resolved_revision": "a" * 40,
            "revision_verified": True,
            "revision_status": "loaded_model_revision_exposed",
        }
    }

    assert mod._embedding_cache_identity(unverified, fake=False) is None
    revision, config = mod._embedding_cache_identity(verified, fake=False)
    assert revision == "a" * 40
    assert config["normalize_embeddings"] is True


def test_fake_model_has_a_stable_cache_identity():
    mod = _script()
    metadata = {
        "embedding": {
            "resolved_revision": "fake-word-hash-128-v1",
            "revision_status": "not_applicable_fake_model",
        }
    }

    revision, config = mod._embedding_cache_identity(metadata, fake=True)

    assert revision == "fake-word-hash-128-v1"
    assert config["dimension"] == 128
    assert mod._embedding_cache_identity(metadata, fake=False) is None


def test_main_populates_the_cache_and_still_prints_the_table(tmp_path, capsys):
    cache = tmp_path / "c"
    _script().main(_argv("--cache-dir", str(cache)))
    out = capsys.readouterr().out
    assert "Strategies: bm25, dense, hybrid" in out
    assert "hybrid_rerank" not in out
    assert any(p.is_file() for p in cache.rglob("*"))


def test_second_run_with_a_warm_cache_gives_identical_metrics(tmp_path, capsys):
    cache = tmp_path / "c"

    _script().main(_argv("--cache-dir", str(cache)))
    first = capsys.readouterr().out

    _script().main(_argv("--cache-dir", str(cache)))
    second = capsys.readouterr().out

    def strip(output):
        return [
            " ".join(line.split()[:5])
            for line in output.splitlines()
            if line.strip()
            and not line.startswith("Cache statistics:")
        ]

    assert strip(first) == strip(second)


def test_no_cache_writes_nothing(tmp_path):
    cache = tmp_path / "c"
    _script().main(_argv("--cache-dir", str(cache), "--no-cache"))
    assert not cache.exists() or not any(p.is_file() for p in cache.rglob("*"))
