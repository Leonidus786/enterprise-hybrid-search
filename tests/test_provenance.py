import hashlib
import json
from pathlib import Path

from hs.models import _extract_resolved_revision
from hs.provenance import build_run_provenance, capture_source_hashes, describe_file, sha256_file, sha256_json


class _Config:
    _commit_hash = "a" * 40


class _Model:
    config = _Config()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_sha256_file_matches_standard_digest(tmp_path):
    path = tmp_path / "data.json"
    path.write_bytes(b"hybrid search\n")
    assert sha256_file(path) == hashlib.sha256(b"hybrid search\n").hexdigest()


def test_describe_file_captures_relative_path_size_and_hash(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "data" / "corpus.jsonl"
    _write(target, '{"id":"d1"}\n')

    data = describe_file("data/corpus.jsonl", root)
    assert data["path"] == "data/corpus.jsonl"
    assert data["exists"] is True
    assert data["size_bytes"] == target.stat().st_size
    assert data["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert data["hash_status"] == "computed"


def test_missing_input_is_explicit_not_fabricated(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    data = describe_file("missing.jsonl", root)
    assert data["exists"] is False
    assert data["sha256"] is None
    assert data["hash_status"] == "unavailable_file_missing"


def test_provenance_captures_inputs_models_source_and_runtime(tmp_path):
    root = tmp_path / "repo"
    _write(root / "data/corpus.jsonl", '{"id":"d1"}\n')
    _write(root / "data/queries.json",
           json.dumps([{"id": "q1", "query": "test", "relevant": ["d1"]}]))
    _write(root / "src/hs/evaluate.py", "# evaluator v1\n")
    _write(root / "scripts/run_eval.py", "# runner v1\n")
    _write(root / "pyproject.toml", "[project]\nname='test'\n")
    query_fp = sha256_json([{"id": "q1", "query": "test", "relevant": ["d1"]}])
    models = {
        "embedding": {
            "model_id": "embed",
            "resolved_revision": "a" * 40,
            "verified_from_loaded_model": False,
            "verified_from_local_snapshot": True,
            "revision_verified": True,
        },
        "reranker": {
            "model_id": "rerank",
            "resolved_revision": "b" * 40,
            "verified_from_loaded_model": True,
            "verified_from_local_snapshot": False,
            "revision_verified": True,
        },
    }

    result = build_run_provenance(
        repo_root=root,
        corpus_path="data/corpus.jsonl",
        queries_path="data/queries.json",
        run_config={"k": 10, "chunking": {"enabled": False}},
        query_fingerprint=query_fp,
        model_metadata=models,
    )

    assert result["schema_version"] == 1
    assert len(result["inputs"]["corpus"]["sha256"]) == 64
    assert len(result["inputs"]
               ["queries_and_relevance_labels"]["sha256"]) == 64
    assert result["inputs"]["query_relevance_fingerprint"] == query_fp
    assert result["source_code"]["file_count"] >= 3
    assert len(result["config_fingerprint_sha256"]) == 64
    assert result["reproducibility_status"]["input_hashes_available"] is True
    assert result["reproducibility_status"]["all_model_revisions_resolved"] is True


def test_config_fingerprint_changes_when_config_changes(tmp_path):
    root = tmp_path / "repo"
    _write(root / "corpus.jsonl", '{"id":"d1"}\n')
    _write(root / "queries.json", "[]")
    _write(root / "src/hs/evaluate.py", "# evaluator\n")
    _write(root / "scripts/run_eval.py", "# runner\n")
    common = dict(
        repo_root=root,
        corpus_path="corpus.jsonl",
        queries_path="queries.json",
        query_fingerprint="qfp",
        model_metadata={"embedding": {"resolved_revision": "a" * 40}},
    )
    first = build_run_provenance(**common, run_config={"k": 10})
    second = build_run_provenance(**common, run_config={"k": 20})
    assert first["config_fingerprint_sha256"] != second["config_fingerprint_sha256"]


def test_source_fingerprint_changes_when_source_changes(tmp_path):
    root = tmp_path / "repo"
    _write(root / "src/hs/evaluate.py", "# v1\n")
    _write(root / "scripts/run_eval.py", "# runner\n")
    first = capture_source_hashes(root)["sha256"]
    _write(root / "src/hs/evaluate.py", "# v2\n")
    second = capture_source_hashes(root)["sha256"]
    assert first != second


def test_model_revision_is_detected_when_loader_exposes_commit():
    assert _extract_resolved_revision(_Model()) == "a" * 40


def test_json_canonical_hash_is_key_order_independent():
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})


def test_save_results_json_adds_schema_v4_provenance(tmp_path):
    import importlib.util
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location(
        "provenance_run_eval", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.json"
    _write(corpus, '{"id":"d1"}\n')
    _write(queries, '[{"id":"q1","query":"test","relevant":["d1"]}]')
    args = SimpleNamespace(
        strategies=["bm25"], k=1, retrieval_k=2, candidate_k=1, level="doc",
        no_chunk=True, chunk_size=200, chunk_overlap=40, fake=True,
        no_cache=True, no_bge_prefix=False, include_query_text=False,
        corpus=str(corpus), queries=str(queries), cache_dir=".cache/embeddings",
        embedding_revision=None, reranker_revision=None,
    )
    metrics = {"bm25": {
        "precision@1": 1.0, "recall@1": 1.0, "recall@2": 1.0,
        "candidate_recall@1": 1.0, "mrr": 1.0, "ndcg@1": 1.0,
        "latency_p50_ms": 1.0,
    }}
    per_query = {"bm25": {"q1": {
        "precision@1": 1.0, "recall@1": 1.0, "recall@2": 1.0,
        "candidate_recall@1": 1.0, "mrr": 1.0, "ndcg@1": 1.0,
    }}}
    output = tmp_path / "results.json"
    module.save_results_json(
        output, metrics, args, [SimpleNamespace(doc_id="d1")], 1, 1,
        per_query=per_query, query_fingerprint="f" * 64,
        model_metadata={
            "embedding": {"model_id": "fake-embed", "resolved_revision": "fake-1"},
            "reranker": {"model_id": "fake-rerank", "resolved_revision": "fake-1"},
        },
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 4
    assert saved["dataset"]["corpus_sha256"] == sha256_file(corpus)
    assert saved["dataset"]["queries_file_sha256"] == sha256_file(queries)
    assert len(saved["provenance"]["config_fingerprint_sha256"]) == 64
    assert saved["ranking_trace"]["schema_version"] == 1


def test_model_loaders_forward_requested_revision_and_record_resolved_commit(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    from hs.models import cross_encoder_scorer, sentence_embedder

    embedding_commit = "c" * 40
    reranker_commit = "d" * 40
    captured = {}

    class DummySentenceTransformer:
        def __init__(self, model_id, **kwargs):
            captured["embedding"] = (model_id, kwargs)
            self.config = SimpleNamespace(_commit_hash=embedding_commit)

        def encode(self, texts, **kwargs):
            return []

    class DummyCrossEncoder:
        def __init__(self, model_id, **kwargs):
            captured["reranker"] = (model_id, kwargs)
            self.model = SimpleNamespace(
                config=SimpleNamespace(_commit_hash=reranker_commit))

        def predict(self, pairs):
            return []

    stub = ModuleType("sentence_transformers")
    stub.SentenceTransformer = DummySentenceTransformer
    stub.CrossEncoder = DummyCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub)

    embed = sentence_embedder("test/embed", revision="main")
    rerank = cross_encoder_scorer("test/reranker", revision="release-tag")

    assert captured["embedding"] == ("test/embed", {"revision": "main"})
    assert captured["reranker"] == (
        "test/reranker", {"revision": "release-tag"})
    assert embed.model_metadata["requested_revision"] == "main"
    assert embed.model_metadata["resolved_revision"] == embedding_commit
    assert rerank.model_metadata["requested_revision"] == "release-tag"
    assert rerank.model_metadata["resolved_revision"] == reranker_commit


def test_fake_models_are_never_reported_as_resolved_model_revisions(tmp_path):
    root = tmp_path / "repo"
    _write(root / "corpus.jsonl", '{"id":"d1"}\n')
    _write(root / "queries.json",
           '[{"id":"q1","query":"test","relevant":["d1"]}]')
    _write(root / "src/hs/evaluate.py", "# evaluator\n")
    _write(root / "scripts/run_eval.py", "# runner\n")
    result = build_run_provenance(
        repo_root=root,
        corpus_path="corpus.jsonl",
        queries_path="queries.json",
        run_config={"fake_models": True, "k": 10},
        query_fingerprint="f" * 64,
        model_metadata={
            "embedding": {"model_id": "fake-embed", "resolved_revision": "fake-1", "revision_status": "not_applicable_fake_model"},
            "reranker": {"model_id": "fake-rerank", "resolved_revision": "fake-1", "revision_status": "not_applicable_fake_model"},
        },
    )
    assert result["reproducibility_status"]["all_model_revisions_resolved"] is False
    assert result["reproducibility_status"]["status"] == "fake_models_smoke_test_not_benchmark_evidence"


def test_query_fingerprint_is_independent_of_query_file_order():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location(
        "query_fingerprint_test_run_eval", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    queries = [
        {"id": "q2", "query": "second query", "relevant": ["d2", "d1"]},
        {"id": "q1", "query": "first query", "relevant": ["d3"]},
    ]
    assert module._query_fingerprint(
        queries) == module._query_fingerprint(list(reversed(queries)))
