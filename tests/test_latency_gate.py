import pytest

from scripts.run_latency import require_verified_models

SHA = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def _meta(**overrides):
    base = {
        "requested_revision": SHA,
        "resolved_revision": SHA,
        "verified_from_loaded_model": False,
        "verified_from_local_snapshot": True,
        "revision_verified": True,
    }
    base.update(overrides)
    return base


def test_loaded_model_revision_verification_passes():
    require_verified_models(
        {
            "embedding": _meta(
                verified_from_loaded_model=True,
                verified_from_local_snapshot=False,
            ),
            "reranker": _meta(),
        }
    )


def test_verified_local_snapshot_passes():
    require_verified_models(
        {
            "embedding": _meta(),
            "reranker": _meta(),
        }
    )


def test_unverified_revision_is_rejected():
    with pytest.raises(RuntimeError, match="embedding"):
        require_verified_models(
            {
                "embedding": _meta(revision_verified=False),
                "reranker": _meta(),
            }
        )


def test_revision_mismatch_is_rejected():
    with pytest.raises(RuntimeError, match="reranker"):
        require_verified_models(
            {
                "embedding": _meta(),
                "reranker": _meta(resolved_revision="f" * 40),
            }
        )


def test_legacy_key_alone_is_not_enough():
    legacy = {
        "requested_revision": SHA,
        "resolved_revision": SHA,
        "verified_from_loaded_model": True,
    }

    with pytest.raises(RuntimeError):
        require_verified_models(
            {"embedding": legacy, "reranker": _meta()}
        )


def test_missing_metadata_is_rejected():
    with pytest.raises(RuntimeError):
        require_verified_models(
            {"embedding": None, "reranker": _meta()}
        )
