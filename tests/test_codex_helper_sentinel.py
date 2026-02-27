from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "harness_generator" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import codex_helper as ch


class _FakeProc:
    def __init__(self, *, stdout_text: str = "") -> None:
        self.stdout = io.StringIO(stdout_text)
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9


class _NoopThread:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def start(self) -> None:
        return None

    def join(self, timeout: float | None = None) -> None:
        return None


def _prepare_helper(tmp_path: Path) -> ch.CodexHelper:
    # Ensure repo has at least one tracked file for git baseline.
    (tmp_path / "README.md").write_text("seed\n", encoding="utf-8")
    return ch.CodexHelper(repo_path=tmp_path, copy_repo=False, codex_cli="opencode")


def _patch_common(monkeypatch: pytest.MonkeyPatch, helper: ch.CodexHelper) -> None:
    monkeypatch.setattr(ch.shutil, "which", lambda _: "/usr/bin/opencode")
    monkeypatch.setattr(ch, "_docker_opencode_image", lambda: "")
    monkeypatch.setattr(ch, "_resolve_opencode_model", lambda env: "")
    monkeypatch.setattr(ch, "_apply_opencode_exec_policy", lambda env: None)
    monkeypatch.setattr(ch, "_append_opencode_metadata", lambda repo_root, payload: None)
    monkeypatch.setattr(helper, "_maybe_prepare_gitnexus_context", lambda: None)


def test_run_codex_command_requires_done_even_when_diff_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    helper = _prepare_helper(tmp_path)
    _patch_common(monkeypatch, helper)
    monkeypatch.setattr(ch.subprocess, "Popen", lambda *args, **kwargs: _FakeProc(stdout_text="edited files\n"))

    diff_calls = {"n": 0}

    def _fake_git_diff_head() -> str:
        diff_calls["n"] += 1
        if diff_calls["n"] == 1:
            return ""
        return "M fuzz/targets.json"

    monkeypatch.setattr(helper, "_git_diff_head", _fake_git_diff_head)
    monkeypatch.setattr(helper, "_git_add_all", lambda: None)

    out = helper.run_codex_command(
        "produce fuzz plan",
        max_attempts=1,
        max_cli_retries=1,
        timeout=3,
    )

    assert out is None
    assert not (helper.working_dir / "done").exists()


def test_run_codex_command_succeeds_only_when_done_and_diff_exist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    helper = _prepare_helper(tmp_path)
    _patch_common(monkeypatch, helper)
    done_path = helper.working_dir / "done"

    def _fake_popen(*args, **kwargs):
        done_path.write_text("fuzz/PLAN.md\n", encoding="utf-8")
        return _FakeProc(stdout_text="ok\n")

    monkeypatch.setattr(ch.subprocess, "Popen", _fake_popen)

    diff_calls = {"n": 0}

    def _fake_git_diff_head() -> str:
        diff_calls["n"] += 1
        if diff_calls["n"] == 1:
            return ""
        return "M fuzz/PLAN.md"

    monkeypatch.setattr(helper, "_git_diff_head", _fake_git_diff_head)
    monkeypatch.setattr(helper, "_git_add_all", lambda: None)

    out = helper.run_codex_command(
        "produce fuzz plan",
        max_attempts=1,
        max_cli_retries=1,
        timeout=3,
    )

    assert out is not None
    assert done_path.is_file()


def test_run_codex_command_idle_timeout_retries_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    helper = _prepare_helper(tmp_path)
    _patch_common(monkeypatch, helper)
    done_path = helper.working_dir / "done"

    first = _FakeProc(stdout_text="")
    second = _FakeProc(stdout_text="")
    second.returncode = 0
    calls = {"n": 0}

    def _fake_popen(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return first
        done_path.write_text("fuzz/build.py\n", encoding="utf-8")
        return second

    monkeypatch.setattr(ch.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(ch.threading, "Thread", _NoopThread)
    monkeypatch.setattr(ch.time, "sleep", lambda _: None)
    monkeypatch.setenv("SHERPA_OPENCODE_IDLE_TIMEOUT_SEC", "1")

    diff_calls = {"n": 0}

    def _fake_git_diff_head() -> str:
        diff_calls["n"] += 1
        if diff_calls["n"] == 1:
            return ""
        return "M fuzz/build.py"

    monkeypatch.setattr(helper, "_git_diff_head", _fake_git_diff_head)
    monkeypatch.setattr(helper, "_git_add_all", lambda: None)

    out = helper.run_codex_command(
        "produce fuzz build script",
        max_attempts=1,
        max_cli_retries=2,
        timeout=30,
        initial_backoff=0,
    )

    assert calls["n"] == 2
    assert first.returncode is not None
    assert out is not None
    assert done_path.is_file()
