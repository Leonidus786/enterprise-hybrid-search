"""Helpers for traceable, reproducible benchmark artifacts.

This module deliberately records unavailable hashes/revisions as unknown rather
than fabricating provenance. It performs no network requests.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

_HASH_BLOCK_SIZE = 1024 * 1024
_SOURCE_GLOBS = (
    "src/hs/*.py",
    "scripts/run_eval.py",
    "scripts/run_latency.py",
    "scripts/compare_runs.py",
    "pyproject.toml",
    "uv.lock",
    "requirements.txt",
)
_RUNTIME_PACKAGES = (
    "numpy",
    "sentence-transformers",
    "transformers",
    "huggingface-hub",
    "torch",
)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data using a deterministic canonical encoding."""
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _display_path(path: str | Path, root: Path) -> str:
    """Prefer repository-relative paths; avoid leaking unrelated absolute paths."""
    original = Path(path)
    resolved = original.resolve() if original.is_absolute() else (
        root.resolve() / original).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        # Keep the caller-supplied path (often relative) rather than a user's
        # absolute home path in the benchmark artifact.
        return str(path)


def describe_file(path: str | Path, root: str | Path) -> dict[str, Any]:
    """Describe an input file. Missing files are explicit and non-fatal."""
    root_path = Path(root).resolve()
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (
        root_path / candidate).resolve()
    exists = resolved.is_file()
    result: dict[str, Any] = {
        "path": _display_path(path, root_path),
        "exists": exists,
        "size_bytes": resolved.stat().st_size if exists else None,
        "sha256": sha256_file(resolved) if exists else None,
    }
    result["hash_status"] = "computed" if exists else "unavailable_file_missing"
    return result


def _git_call(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True,
            check=False, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.rstrip("\r\n")


def capture_git_state(repo_root: str | Path) -> dict[str, Any]:
    """Capture commit/branch/dirty files without changing the worktree."""
    root = Path(repo_root).resolve()
    commit = _git_call(root, "rev-parse", "HEAD")
    branch = _git_call(root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git_call(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )

    if status is None:
        dirty = None
        changed_files = None
    else:
        records = status.split("\0")
        if records and records[-1] == "":
            records.pop()

        dirty = bool(records)
        changed_files = []
        index = 0

        while index < len(records):
            record = records[index]
            index += 1

            # Porcelain v1 records contain two status characters,
            # one separator, and the path. Do not strip the path:
            # leading spaces can be part of a legitimate filename.
            if len(record) < 4 or record[2] != " ":
                dirty = None
                changed_files = None
                break

            status_code = record[:2]
            file_path = record[3:]
            changed_files.append(file_path)

            # With -z, rename/copy records contain the destination
            # first and the source in the following NUL-delimited field.
            if (
                "R" in status_code or "C" in status_code
            ) and index < len(records):
                index += 1

    return {
        "commit": commit,
        "branch": branch,
        "working_tree_dirty": dirty,
        "changed_files": changed_files,
        "capture_status": "captured" if commit is not None else "git_unavailable",
    }


def capture_source_hashes(repo_root: str | Path) -> dict[str, Any]:
    """Hash relevant Python source files; do not include data, caches or outputs."""
    root = Path(repo_root).resolve()
    candidates: set[Path] = set()
    for pattern in _SOURCE_GLOBS:
        candidates.update(path for path in root.glob(
            pattern) if path.is_file())
    hashes = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(candidates)
    }
    return {
        "files": hashes,
        "file_count": len(hashes),
        "sha256": sha256_json(hashes) if hashes else None,
        "status": "computed" if hashes else "no_source_files_found",
    }


def capture_runtime() -> dict[str, Any]:
    """Capture available runtime and coarse hardware information."""
    packages: dict[str, str | None] = {}
    for package in _RUNTIME_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine() or None,
        "processor": platform.processor() or None,
        "cpu_count_logical": os.cpu_count(),
        "packages": packages,
    }


def build_run_provenance(
    *,
    repo_root: str | Path,
    corpus_path: str | Path,
    queries_path: str | Path,
    run_config: dict[str, Any],
    query_fingerprint: str | None,
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance envelope for a saved benchmark result."""
    root = Path(repo_root).resolve()
    corpus = describe_file(corpus_path, root)
    query_file = describe_file(queries_path, root)
    git = capture_git_state(root)
    source = capture_source_hashes(root)
    runtime = capture_runtime()
    models = model_metadata or {}

    config_basis = {
        "run_config": run_config,
        "corpus_sha256": corpus["sha256"],
        "queries_file_sha256": query_file["sha256"],
        "query_relevance_fingerprint": query_fingerprint,
        "models": models,
        "source_fingerprint": source["sha256"],
        "runtime": runtime,
    }
    config_fingerprint = sha256_json(config_basis)

    model_entries = list(models.values())
    # Fake model labels are identifiers for deterministic smoke-test functions,
    # not loaded model commits. Never report them as resolved model revisions.
    fake_models = bool(run_config.get("fake_models"))
    known_model_revisions = (
        not fake_models
        and bool(model_entries)
        and all(
            isinstance(entry, dict)
            and bool(entry.get("resolved_revision"))
            and (
                entry.get("revision_verified") is True
                or (
                    "revision_verified" not in entry
                    and entry.get("verified_from_loaded_model") is True
                )
            )
            for entry in model_entries
        )
    )
    input_hashes_available = bool(
        corpus["sha256"] and query_file["sha256"] and query_fingerprint)

    core_provenance_available = bool(
        input_hashes_available
        and source["sha256"]
        and git["commit"]
        and known_model_revisions
    )
    if fake_models:
        provenance_status = "fake_models_smoke_test_not_benchmark_evidence"
    elif not core_provenance_available:
        provenance_status = "provenance_captured_with_gaps"
    elif git["working_tree_dirty"] is True:
        provenance_status = "provenance_captured_dirty_worktree"
    elif git["working_tree_dirty"] is None:
        provenance_status = "provenance_captured_git_state_unknown"
    else:
        provenance_status = "provenance_captured"

    return {
        "schema_version": 1,
        "inputs": {
            "corpus": corpus,
            "queries_and_relevance_labels": query_file,
            "query_relevance_fingerprint": query_fingerprint,
            "input_hash_status": "complete" if input_hashes_available else "partial_or_unavailable",
        },
        "models": models,
        "git": git,
        "source_code": source,
        "runtime": runtime,
        "config_fingerprint_sha256": config_fingerprint,
        "reproducibility_status": {
            "input_hashes_available": input_hashes_available,
            "source_fingerprint_available": source["sha256"] is not None,
            "git_commit_available": git["commit"] is not None,
            "all_model_revisions_resolved": known_model_revisions,
            "status": provenance_status,
        },
    }
