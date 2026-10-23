from types import SimpleNamespace

from hs import provenance


_COMMIT = "3fabc0e8bb36c7207c0b849a2c09fdda63a77bb0"


def _capture_with_status(monkeypatch, tmp_path, status_output):
    calls = []

    def fake_run(command, **kwargs):
        args = command[1:]
        calls.append(args)

        if args == ["rev-parse", "HEAD"]:
            stdout = _COMMIT + "\n"
        elif args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            stdout = "main\n"
        elif args[0] == "status":
            stdout = status_output
        else:
            raise AssertionError(f"Unexpected git command: {args}")

        return SimpleNamespace(returncode=0, stdout=stdout)

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)
    state = provenance.capture_git_state(tmp_path)
    return state, calls


def test_capture_git_state_uses_nul_delimited_porcelain(
    monkeypatch, tmp_path
):
    status_output = (
        " M PATCH_README.txt\0"
        "?? new file.txt\0"
        "R  renamed file.txt\0"
        "old file.txt\0"
    )

    state, calls = _capture_with_status(
        monkeypatch, tmp_path, status_output
    )

    assert state["working_tree_dirty"] is True
    assert state["changed_files"] == [
        "PATCH_README.txt",
        "new file.txt",
        "renamed file.txt",
    ]
    assert [
        args for args in calls if args[0] == "status"
    ] == [
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"]
    ]


def test_capture_git_state_preserves_leading_space_in_filename(
    monkeypatch, tmp_path
):
    state, _ = _capture_with_status(
        monkeypatch, tmp_path, " M  leading space.txt\0"
    )

    assert state["changed_files"] == [" leading space.txt"]


def test_capture_git_state_handles_clean_worktree(
    monkeypatch, tmp_path
):
    state, _ = _capture_with_status(monkeypatch, tmp_path, "")

    assert state["working_tree_dirty"] is False
    assert state["changed_files"] == []
