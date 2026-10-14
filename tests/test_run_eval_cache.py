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


def test_main_populates_the_cache_and_still_prints_the_table(tmp_path, capsys):
    cache = tmp_path / "c"
    _script().main(_argv("--cache-dir", str(cache)))
    out = capsys.readouterr().out
    assert "bm25" in out and "hybrid_rerank" in out
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
