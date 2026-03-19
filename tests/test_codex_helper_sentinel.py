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


def test_run_codex_command_treats_file_progress_as_activity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    helper = _prepare_helper(tmp_path)
    _patch_common(monkeypatch, helper)
    done_path = helper.working_dir / "done"
    fuzz_dir = helper.working_dir / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    watched_file = fuzz_dir / "build.py"

    proc = _FakeProc(stdout_text="")
    monkeypatch.setattr(ch.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(ch.threading, "Thread", _NoopThread)
    monkeypatch.setattr(ch.time, "sleep", lambda _: None)

    ticks = {"n": 0}

    def _fake_time() -> float:
        ticks["n"] += 1
        now = ticks["n"] * 0.25
        if now >= 0.75 and not watched_file.exists():
            watched_file.write_text("# progress\n", encoding="utf-8")
        if now >= 1.25 and not done_path.exists():
            done_path.write_text("fuzz/build.py\n", encoding="utf-8")
        return now

    monkeypatch.setattr(ch.time, "time", _fake_time)

    def _fake_git_diff_head() -> str:
        return "M fuzz/build.py" if done_path.exists() else ""

    monkeypatch.setattr(helper, "_git_diff_head", _fake_git_diff_head)
    monkeypatch.setattr(helper, "_git_add_all", lambda: None)

    out = helper.run_codex_command(
        "produce fuzz build script",
        max_attempts=1,
        max_cli_retries=1,
        timeout=30,
        initial_backoff=0,
        idle_timeout_override=1,
        activity_probe_interval_sec=0.2,
        activity_watch_paths=["fuzz/build.py"],
    )

    assert out is not None
    assert done_path.is_file()
    assert proc.returncode is not None


def test_run_codex_command_prompt_uses_real_repo_root_in_native_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    helper = _prepare_helper(tmp_path)
    _patch_common(monkeypatch, helper)
    done_path = helper.working_dir / "done"
    captured: dict[str, object] = {}

    def _fake_popen(*args, **kwargs):
        cmd = args[0]
        captured["cmd"] = cmd
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
    cmd = captured.get("cmd")
    assert isinstance(cmd, list)
    prompt = str(cmd[-1])
    assert "do not assume /repo exists" in prompt
    assert f"The repository root is {helper.working_dir.resolve()}" in prompt
    assert "mounted at /repo" not in prompt


def test_run_codex_command_materializes_long_inputs_to_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    helper = _prepare_helper(tmp_path)
    _patch_common(monkeypatch, helper)
    done_path = helper.working_dir / "done"
    captured: dict[str, object] = {}

    def _fake_popen(*args, **kwargs):
        cmd = args[0]
        captured["cmd"] = cmd
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

    long_tasks = "\n".join(f"task line {i}" for i in range(220))
    long_ctx = "\n".join(f"context line {i}" for i in range(260))
    out = helper.run_codex_command(
        long_tasks,
        additional_context=long_ctx,
        max_attempts=1,
        max_cli_retries=1,
        timeout=3,
    )

    assert out is not None
    cmd = captured.get("cmd")
    assert isinstance(cmd, list)
    prompt = str(cmd[-1])
    assert ".git/sherpa-opencode/task.txt" in prompt
    assert ".git/sherpa-opencode/additional_context.txt" in prompt
    assert "task line 219" not in prompt
    assert "context line 259" not in prompt
    task_file = helper.working_dir / ".git" / "sherpa-opencode" / "task.txt"
    context_file = helper.working_dir / ".git" / "sherpa-opencode" / "additional_context.txt"
    assert task_file.is_file()
    assert context_file.is_file()
    assert len(task_file.read_text(encoding="utf-8").splitlines()) <= 50
    assert len(context_file.read_text(encoding="utf-8").splitlines()) <= 50


def test_run_codex_command_injects_global_policy_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    helper = _prepare_helper(tmp_path)
    _patch_common(monkeypatch, helper)
    done_path = helper.working_dir / "done"
    captured: dict[str, object] = {}

    def _fake_popen(*args, **kwargs):
        captured["cmd"] = args[0]
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
    cmd = captured.get("cmd")
    assert isinstance(cmd, list)
    prompt = str(cmd[-1])
    assert ".git/sherpa-opencode/opencode_policy.md" in prompt
    policy_file = helper.working_dir / ".git" / "sherpa-opencode" / "opencode_policy.md"
    assert policy_file.is_file()


def test_run_codex_command_policy_missing_falls_back_without_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    helper = _prepare_helper(tmp_path)
    _patch_common(monkeypatch, helper)
    done_path = helper.working_dir / "done"
    captured: dict[str, object] = {}
    monkeypatch.setenv("SHERPA_OPENCODE_POLICY_ENABLED", "1")
    monkeypatch.setenv("SHERPA_OPENCODE_POLICY_PATH", str(tmp_path / "missing-policy.md"))

    def _fake_popen(*args, **kwargs):
        captured["cmd"] = args[0]
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
    cmd = captured.get("cmd")
    assert isinstance(cmd, list)
    prompt = str(cmd[-1])
    assert ".git/sherpa-opencode/opencode_policy.md" not in prompt


def test_run_codex_command_can_disable_policy_injection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    helper = _prepare_helper(tmp_path)
    _patch_common(monkeypatch, helper)
    done_path = helper.working_dir / "done"
    captured: dict[str, object] = {}
    policy_path = tmp_path / "custom-policy.md"
    policy_path.write_text("policy text\n", encoding="utf-8")
    monkeypatch.setenv("SHERPA_OPENCODE_POLICY_PATH", str(policy_path))
    monkeypatch.setenv("SHERPA_OPENCODE_POLICY_ENABLED", "0")

    def _fake_popen(*args, **kwargs):
        captured["cmd"] = args[0]
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
    cmd = captured.get("cmd")
    assert isinstance(cmd, list)
    prompt = str(cmd[-1])
    assert ".git/sherpa-opencode/opencode_policy.md" not in prompt


def test_run_codex_command_injects_stage_skill_when_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    helper = _prepare_helper(tmp_path)
    _patch_common(monkeypatch, helper)
    done_path = helper.working_dir / "done"
    captured: dict[str, object] = {}

    def _fake_popen(*args, **kwargs):
        captured["cmd"] = args[0]
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
        stage_skill="plan",
        max_attempts=1,
        max_cli_retries=1,
        timeout=3,
    )

    assert out is not None
    cmd = captured.get("cmd")
    assert isinstance(cmd, list)
    prompt = str(cmd[-1])
    assert "Follow the stage contract for done content exactly." in prompt
    assert "Example: `echo fuzz/build.py > done`" not in prompt
    assert ".git/sherpa-opencode/stage_skill_plan.md" in prompt
    skill_file = helper.working_dir / ".git" / "sherpa-opencode" / "stage_skill_plan.md"
    assert skill_file.is_file()


def test_run_codex_command_stage_skill_missing_falls_back_when_not_strict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    helper = _prepare_helper(tmp_path)
    _patch_common(monkeypatch, helper)
    done_path = helper.working_dir / "done"
    captured: dict[str, object] = {}
    monkeypatch.setenv("SHERPA_OPENCODE_STAGE_SKILLS_ENABLED", "1")
    monkeypatch.setenv("SHERPA_OPENCODE_STAGE_SKILLS_STRICT", "0")
    monkeypatch.setenv("SHERPA_OPENCODE_STAGE_SKILLS_PATH", str(tmp_path / "missing-skills-root"))

    def _fake_popen(*args, **kwargs):
        captured["cmd"] = args[0]
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
        stage_skill="plan",
        max_attempts=1,
        max_cli_retries=1,
        timeout=3,
    )

    assert out is not None
    cmd = captured.get("cmd")
    assert isinstance(cmd, list)
    prompt = str(cmd[-1])
    assert ".git/sherpa-opencode/stage_skill_plan.md" not in prompt


def test_run_codex_command_stage_skill_missing_raises_when_strict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    helper = _prepare_helper(tmp_path)
    _patch_common(monkeypatch, helper)
    monkeypatch.setenv("SHERPA_OPENCODE_STAGE_SKILLS_ENABLED", "1")
    monkeypatch.setenv("SHERPA_OPENCODE_STAGE_SKILLS_STRICT", "1")
    monkeypatch.setenv("SHERPA_OPENCODE_STAGE_SKILLS_PATH", str(tmp_path / "missing-skills-root"))

    with pytest.raises(RuntimeError, match="stage skill missing"):
        helper.run_codex_command(
            "produce fuzz plan",
            stage_skill="plan",
            max_attempts=1,
            max_cli_retries=1,
            timeout=3,
        )
