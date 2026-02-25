from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
SRC_DIR = ROOT / "harness_generator" / "src"
for p in (APP_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import workflow_graph


class _FakeGenerator:
    def __init__(
        self,
        repo_root: Path,
        run_results: list[tuple[int, str, str]],
        bin_results: list[list[Path]],
        *,
        docker_image: str | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.docker_image = docker_image
        self._run_results = list(run_results)
        self._bin_results = list(bin_results)
        self.commands: list[list[str]] = []
        self.cwds: list[Path] = []

    def _python_runner(self) -> str:
        return "python"

    def _run_cmd(self, cmd, *, cwd, env, timeout):
        self.commands.append(list(cmd))
        self.cwds.append(Path(cwd))
        if not self._run_results:
            raise AssertionError("unexpected _run_cmd call")
        return self._run_results.pop(0)

    def _discover_fuzz_binaries(self):
        if not self._bin_results:
            raise AssertionError("unexpected _discover_fuzz_binaries call")
        return self._bin_results.pop(0)


@pytest.fixture
def _no_sleep(monkeypatch):
    monkeypatch.setattr(workflow_graph.time, "sleep", lambda _: None)


def test_build_retries_after_nonzero_exit(tmp_path: Path, monkeypatch, _no_sleep):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "build.py").write_text("print('build')\n", encoding="utf-8")
    (fuzz_dir / "out").mkdir(parents=True, exist_ok=True)
    fuzzer_bin = fuzz_dir / "out" / "demo_fuzz"
    fuzzer_bin.write_text("", encoding="utf-8")

    gen = _FakeGenerator(
        tmp_path,
        run_results=[(1, "compile failed", "error"), (0, "ok", "")],
        bin_results=[[fuzzer_bin]],
    )
    monkeypatch.setenv("SHERPA_WORKFLOW_BUILD_LOCAL_RETRIES", "2")
    monkeypatch.setenv("SHERPA_WORKFLOW_BUILD_RETRY_WITH_CLEAN", "0")

    out = workflow_graph._node_build({"generator": gen, "build_attempts": 0})

    assert out["last_error"] == ""
    assert out["message"].startswith("built (")
    assert out["build_rc"] == 0
    assert out["build_attempts"] == 2
    assert out["build_error_kind"] == ""
    assert out["build_error_code"] == ""
    assert len(gen.commands) == 2


def test_build_retries_with_clean_when_supported(tmp_path: Path, monkeypatch, _no_sleep):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "build.py").write_text("print('--clean')\n", encoding="utf-8")
    (fuzz_dir / "out").mkdir(parents=True, exist_ok=True)
    fuzzer_bin = fuzz_dir / "out" / "demo_fuzz"
    fuzzer_bin.write_text("", encoding="utf-8")

    gen = _FakeGenerator(
        tmp_path,
        run_results=[(1, "cmake failed", "error"), (0, "ok", "")],
        bin_results=[[fuzzer_bin]],
    )
    monkeypatch.setenv("SHERPA_WORKFLOW_BUILD_LOCAL_RETRIES", "1")
    monkeypatch.setenv("SHERPA_WORKFLOW_BUILD_RETRY_WITH_CLEAN", "1")

    out = workflow_graph._node_build({"generator": gen, "build_attempts": 0})

    assert out["last_error"] == ""
    assert out["build_rc"] == 0
    assert out["build_attempts"] == 2
    assert out["build_error_kind"] == ""
    assert out["build_error_code"] == ""
    assert len(gen.commands) == 2
    assert gen.commands[1][-1] == "--clean"


def test_build_failure_without_binaries_includes_artifact_diagnostics(tmp_path: Path, monkeypatch, _no_sleep):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "build.py").write_text("print('build')\n", encoding="utf-8")
    (fuzz_dir / "out").mkdir(parents=True, exist_ok=True)
    build_dir = tmp_path / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "libexample.a").write_text("", encoding="utf-8")

    gen = _FakeGenerator(
        tmp_path,
        run_results=[(0, "build ok", "")],
        bin_results=[[]],
    )
    monkeypatch.setenv("SHERPA_WORKFLOW_BUILD_LOCAL_RETRIES", "1")

    out = workflow_graph._node_build({"generator": gen, "build_attempts": 3})

    assert out["build_rc"] == 0
    assert "No fuzzer binaries found under fuzz/out/" in out["last_error"]
    assert out["build_error_kind"] == "source"
    assert out["build_error_code"] == "no_fuzzer_binaries"
    assert "build dir artifacts (static libs)" in out["build_stdout_tail"]
    assert out["build_attempts"] == 4


def test_build_sh_uses_sh_when_bash_missing(tmp_path: Path, monkeypatch, _no_sleep):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "build.sh").write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    (fuzz_dir / "out").mkdir(parents=True, exist_ok=True)
    fuzzer_bin = fuzz_dir / "out" / "demo_fuzz"
    fuzzer_bin.write_text("", encoding="utf-8")

    gen = _FakeGenerator(
        tmp_path,
        run_results=[(0, "ok", "")],
        bin_results=[[fuzzer_bin]],
    )
    monkeypatch.setenv("SHERPA_WORKFLOW_BUILD_LOCAL_RETRIES", "1")
    monkeypatch.setattr(
        workflow_graph.shutil,
        "which",
        lambda cmd: "/bin/sh" if cmd == "sh" else None,
    )

    out = workflow_graph._node_build({"generator": gen, "build_attempts": 0})

    assert out["last_error"] == ""
    assert out["build_error_kind"] == ""
    assert out["build_error_code"] == ""
    assert len(gen.commands) == 1
    assert gen.commands[0][0] == "sh"


def test_build_retries_in_repo_root_cwd_for_hardcoded_fuzz_paths(tmp_path: Path, monkeypatch, _no_sleep):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "build.py").write_text(
        "print('legacy build script')\n",
        encoding="utf-8",
    )
    (fuzz_dir / "out").mkdir(parents=True, exist_ok=True)
    fuzzer_bin = fuzz_dir / "out" / "demo_fuzz"
    fuzzer_bin.write_text("", encoding="utf-8")

    gen = _FakeGenerator(
        tmp_path,
        run_results=[
            (1, "", "FileNotFoundError: [Errno 2] No such file or directory: 'fuzz/targets.json'"),
            (0, "ok", ""),
        ],
        bin_results=[[fuzzer_bin]],
        docker_image="sherpa-fuzz-cpp:latest",
    )
    monkeypatch.setenv("SHERPA_WORKFLOW_BUILD_LOCAL_RETRIES", "1")

    out = workflow_graph._node_build({"generator": gen, "build_attempts": 0})

    assert out["last_error"] == ""
    assert out["build_rc"] == 0
    assert out["build_error_kind"] == ""
    assert out["build_error_code"] == ""
    assert len(gen.commands) == 2
    assert gen.commands[0] == ["python", "build.py"]
    assert gen.commands[1] == ["python", "fuzz/build.py"]
    assert gen.cwds[0] == fuzz_dir
    assert gen.cwds[1] == tmp_path


def test_build_failure_classifies_infra_docker_daemon(tmp_path: Path, monkeypatch, _no_sleep):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "build.py").write_text("print('build')\n", encoding="utf-8")

    gen = _FakeGenerator(
        tmp_path,
        run_results=[(1, "", "Cannot connect to the Docker daemon at unix:///var/run/docker.sock")],
        bin_results=[[]],
    )
    monkeypatch.setenv("SHERPA_WORKFLOW_BUILD_LOCAL_RETRIES", "1")
    monkeypatch.setenv("SHERPA_WORKFLOW_BUILD_RETRY_WITH_CLEAN", "0")

    out = workflow_graph._node_build({"generator": gen, "build_attempts": 0})

    assert out["build_rc"] == 1
    assert out["build_error_kind"] == "infra"
    assert out["build_error_code"] == "docker_daemon_unavailable"
    assert out["last_error"].startswith("build failed rc=1")


def test_route_after_build_stops_on_infra_error() -> None:
    route = workflow_graph._route_after_build_state(
        {
            "failed": False,
            "last_error": "build failed",
            "build_error_kind": "infra",
        }
    )
    assert route == "stop"


def test_route_after_build_sends_source_error_to_fix_build() -> None:
    route = workflow_graph._route_after_build_state(
        {
            "failed": False,
            "last_error": "build failed",
            "build_error_kind": "source",
        }
    )
    assert route == "fix_build"


def test_route_after_init_resumes_from_requested_step() -> None:
    route = workflow_graph._route_after_init_state(
        {
            "failed": False,
            "last_error": "",
            "resume_from_step": "run",
        }
    )
    assert route == "run"


def test_route_after_init_defaults_to_plan_for_invalid_resume_step() -> None:
    route = workflow_graph._route_after_init_state(
        {
            "failed": False,
            "last_error": "",
            "resume_from_step": "unknown-step",
        }
    )
    assert route == "plan"


def test_fix_build_hotfixes_libfuzzer_main_conflict(tmp_path: Path):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    build_py = fuzz_dir / "build.py"
    build_py.write_text(
        "\n".join(
            [
                "flags = [",
                "    '-std=c++11',",
                "    '-fsanitize=fuzzer,address,undefined',",
                "]",
                "cmd = [cxx] + flags + [source_path, harness_path, '-o', output_path]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    gen = SimpleNamespace(repo_root=tmp_path)
    state = {
        "generator": gen,
        "last_error": "ld: multiple definition of `main'",
        "build_stdout_tail": "",
        "build_stderr_tail": "",
    }

    out = workflow_graph._node_fix_build(state)

    assert out["last_error"] == ""
    assert "hotfix" in out["message"]
    assert "-DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION" in build_py.read_text(encoding="utf-8")
