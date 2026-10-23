from types import SimpleNamespace

from hs.models import _attach_metadata


PINNED_SHA = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
OTHER_SHA = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


def _metadata(model, requested_revision):
    function = _attach_metadata(
        lambda _: None,
        model_id="BAAI/bge-small-en-v1.5",
        requested_revision=requested_revision,
        model=model,
    )
    return function.model_metadata


def test_requested_sha_alone_is_not_verified(monkeypatch):
    import huggingface_hub

    def missing_snapshot(repo_id, *, revision, local_files_only):
        raise FileNotFoundError("Snapshot is not cached")

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        missing_snapshot,
    )

    metadata = _metadata(SimpleNamespace(), PINNED_SHA)

    assert metadata["requested_pinned"] is True
    assert metadata["resolved_revision"] is None
    assert metadata["verified_from_loaded_model"] is False
    assert metadata["verified_from_local_snapshot"] is False
    assert metadata["revision_verified"] is False
    assert metadata["revision_status"] == "requested_commit_unverified"


def test_loaded_model_revision_matching_request_is_verified():
    model = SimpleNamespace(
        config=SimpleNamespace(_commit_hash=PINNED_SHA)
    )

    metadata = _metadata(model, PINNED_SHA)

    assert metadata["requested_pinned"] is True
    assert metadata["resolved_revision"] == PINNED_SHA
    assert metadata["verified_from_loaded_model"] is True
    assert metadata["revision_status"] == "verified_commit_match"


def test_loaded_model_revision_mismatch_is_not_verified():
    model = SimpleNamespace(
        config=SimpleNamespace(_commit_hash=OTHER_SHA)
    )

    metadata = _metadata(model, PINNED_SHA)

    assert metadata["requested_pinned"] is True
    assert metadata["resolved_revision"] == OTHER_SHA
    assert metadata["verified_from_loaded_model"] is False
    assert metadata["revision_status"] == "revision_mismatch"


def test_cached_snapshot_path_verifies_revision_when_loader_hides_it(monkeypatch, tmp_path):
    import huggingface_hub

    snapshot = (
        tmp_path
        / "models--BAAI--bge-small-en-v1.5"
        / "snapshots"
        / PINNED_SHA
    )
    snapshot.mkdir(parents=True)

    calls = []

    def fake_snapshot_download(repo_id, *, revision, local_files_only):
        calls.append((repo_id, revision, local_files_only))
        return str(snapshot)

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        fake_snapshot_download,
    )

    metadata = _metadata(SimpleNamespace(), PINNED_SHA)

    assert calls == [
        ("BAAI/bge-small-en-v1.5", PINNED_SHA, True)
    ]
    assert metadata["resolved_revision"] == PINNED_SHA
    assert metadata["verified_from_loaded_model"] is False
    assert metadata["verified_from_local_snapshot"] is True
    assert metadata["revision_verified"] is True
    assert metadata["revision_status"] == "verified_cached_snapshot_match"
