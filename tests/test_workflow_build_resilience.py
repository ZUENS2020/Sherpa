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


def test_opencode_cli_retries_default_and_bounds(monkeypatch) -> None:
    monkeypatch.delenv("SHERPA_WORKFLOW_OPENCODE_CLI_RETRIES", raising=False)
    assert workflow_graph._opencode_cli_retries() == 2

    monkeypatch.setenv("SHERPA_WORKFLOW_OPENCODE_CLI_RETRIES", "0")
    assert workflow_graph._opencode_cli_retries() == 1

    monkeypatch.setenv("SHERPA_WORKFLOW_OPENCODE_CLI_RETRIES", "99")
    assert workflow_graph._opencode_cli_retries() == 8

    monkeypatch.setenv("SHERPA_WORKFLOW_OPENCODE_CLI_RETRIES", "bad")
    assert workflow_graph._opencode_cli_retries() == 2


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


def test_fix_build_allows_opencode_edits_under_fuzz_only(tmp_path: Path, monkeypatch):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    build_py = fuzz_dir / "build.py"
    build_py.write_text("print('v1')\n", encoding="utf-8")

    class _Patcher:
        def run_codex_command(self, *_args, **_kwargs):
            build_py.write_text("print('v2')\n", encoding="utf-8")
            (tmp_path / "done").write_text("fuzz/build.py\n", encoding="utf-8")

    monkeypatch.setattr(workflow_graph, "_llm_or_none", lambda: None)
    gen = SimpleNamespace(repo_root=tmp_path, patcher=_Patcher())
    state = {
        "generator": gen,
        "last_error": "error: missing header",
        "build_stdout_tail": "",
        "build_stderr_tail": "",
    }

    out = workflow_graph._node_fix_build(state)

    assert out["last_error"] == ""
    assert out["message"] == "opencode fixed build"
    assert "v2" in build_py.read_text(encoding="utf-8")


def test_fix_build_rejects_opencode_source_edits_outside_fuzz(tmp_path: Path, monkeypatch):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "build.py").write_text("print('ok')\n", encoding="utf-8")
    source_file = tmp_path / "upstream.c"
    source_file.write_text("int x = 1;\n", encoding="utf-8")

    class _Patcher:
        def run_codex_command(self, *_args, **_kwargs):
            source_file.write_text("int x = 2;\n", encoding="utf-8")
            (tmp_path / "done").write_text("upstream.c\n", encoding="utf-8")

    monkeypatch.setattr(workflow_graph, "_llm_or_none", lambda: None)
    gen = SimpleNamespace(repo_root=tmp_path, patcher=_Patcher())
    state = {
        "generator": gen,
        "last_error": "error: missing include",
        "build_stdout_tail": "",
        "build_stderr_tail": "",
    }

    out = workflow_graph._node_fix_build(state)

    assert out["message"] == "opencode fix_build touched disallowed files"
    assert "upstream.c" in out["last_error"]
    assert "Only `fuzz/` and `done` are allowed" in out["last_error"]


def test_build_failure_infra_error_includes_recovery_hint(tmp_path: Path, monkeypatch, _no_sleep):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "build.py").write_text("print('build')\n", encoding="utf-8")

    gen = _FakeGenerator(
        tmp_path,
        run_results=[(1, "", "temporary failure in name resolution")],
        bin_results=[[]],
    )
    monkeypatch.setenv("SHERPA_WORKFLOW_BUILD_LOCAL_RETRIES", "1")
    monkeypatch.setenv("SHERPA_WORKFLOW_BUILD_RETRY_WITH_CLEAN", "0")

    out = workflow_graph._node_build({"generator": gen, "build_attempts": 0})

    assert out["build_rc"] == 1
    assert out["build_error_kind"] == "infra"
    assert "recovery:" in out["last_error"]
    assert "DNS" in out["last_error"]


def test_fix_build_stops_after_noop_streak_threshold(tmp_path: Path, monkeypatch):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "build.py").write_text("print('same')\n", encoding="utf-8")

    class _Patcher:
        def run_codex_command(self, *_args, **_kwargs):
            (tmp_path / "done").write_text("fuzz/build.py\n", encoding="utf-8")

    monkeypatch.setattr(workflow_graph, "_llm_or_none", lambda: None)
    monkeypatch.setenv("SHERPA_FIX_BUILD_MAX_NOOP_STREAK", "3")
    gen = SimpleNamespace(repo_root=tmp_path, patcher=_Patcher())
    state = {
        "generator": gen,
        "last_error": "build failed",
        "build_stdout_tail": "",
        "build_stderr_tail": "",
        "fix_build_noop_streak": 2,
        "fix_build_attempt_history": [],
    }

    out = workflow_graph._node_fix_build(state)
    assert out["failed"] is True
    assert out["fix_build_terminal_reason"] == "fix_build_noop_streak_exceeded"
    assert "no-op streak exceeded" in out["last_error"]


def test_fix_build_noop_streak_resets_after_effective_change(tmp_path: Path, monkeypatch):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    build_py = fuzz_dir / "build.py"
    build_py.write_text("print('v1')\n", encoding="utf-8")

    class _Patcher:
        def run_codex_command(self, *_args, **_kwargs):
            build_py.write_text("print('v2')\n", encoding="utf-8")
            (tmp_path / "done").write_text("fuzz/build.py\n", encoding="utf-8")

    monkeypatch.setattr(workflow_graph, "_llm_or_none", lambda: None)
    gen = SimpleNamespace(repo_root=tmp_path, patcher=_Patcher())
    state = {
        "generator": gen,
        "last_error": "build failed",
        "build_stdout_tail": "",
        "build_stderr_tail": "",
        "fix_build_noop_streak": 2,
        "fix_build_attempt_history": [],
    }
    out = workflow_graph._node_fix_build(state)
    assert out["last_error"] == ""
    assert out["fix_build_noop_streak"] == 0
    assert out["message"] == "opencode fixed build"


def test_fix_build_rule_compiler_fuzzer_flag_mismatch(tmp_path: Path, monkeypatch):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    build_py = fuzz_dir / "build.py"
    build_py.write_text("cc = 'gcc'\ncmd = ['gcc', '-fsanitize=fuzzer']\n", encoding="utf-8")
    gen = SimpleNamespace(repo_root=tmp_path, patcher=SimpleNamespace(run_codex_command=lambda *_a, **_k: None))
    monkeypatch.setattr(workflow_graph, "_llm_or_none", lambda: None)
    out = workflow_graph._node_fix_build(
        {
            "generator": gen,
            "last_error": "gcc: error: unrecognized argument to '-fsanitize=' option: 'fuzzer'",
            "build_stdout_tail": "",
            "build_stderr_tail": "",
        }
    )
    assert out["last_error"] == ""
    assert "compiler_fuzzer_flag_mismatch" in (out.get("fix_build_rule_hits") or [])
    assert "clang" in build_py.read_text(encoding="utf-8")


def test_fix_build_rule_missing_llvmfuzzer_entrypoint(tmp_path: Path, monkeypatch):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    build_py = fuzz_dir / "build.py"
    build_py.write_text("flags = ['-O2']\ncmd = ['clang++', 'a.c']\n", encoding="utf-8")
    gen = SimpleNamespace(repo_root=tmp_path, patcher=SimpleNamespace(run_codex_command=lambda *_a, **_k: None))
    monkeypatch.setattr(workflow_graph, "_llm_or_none", lambda: None)
    out = workflow_graph._node_fix_build(
        {
            "generator": gen,
            "last_error": "undefined reference to `LLVMFuzzerTestOneInput'",
            "build_stdout_tail": "",
            "build_stderr_tail": "",
        }
    )
    assert out["last_error"] == ""
    assert "missing_llvmfuzzer_entrypoint" in (out.get("fix_build_rule_hits") or [])
    txt = build_py.read_text(encoding="utf-8")
    assert "clang++" not in txt


def test_fix_build_rule_fuzz_out_path_mismatch(tmp_path: Path, monkeypatch):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    build_py = fuzz_dir / "build.py"
    build_py.write_text(
        "def build_all(out_dir=\"fuzz/out\", cc=\"clang\"):\n"
        "    os.makedirs(out_dir, exist_ok=True)\n"
        "    compile_target(name, target_info, out_dir, cc)\n",
        encoding="utf-8",
    )
    gen = SimpleNamespace(repo_root=tmp_path, patcher=SimpleNamespace(run_codex_command=lambda *_a, **_k: None))
    monkeypatch.setattr(workflow_graph, "_llm_or_none", lambda: None)
    out = workflow_graph._node_fix_build(
        {
            "generator": gen,
            "last_error": "No fuzzer binaries found under fuzz/out/ after 1 command run(s)",
            "build_stdout_tail": "",
            "build_stderr_tail": "",
        }
    )
    assert out["last_error"] == ""
    assert "fuzz_out_path_mismatch" in (out.get("fix_build_rule_hits") or [])
    txt = build_py.read_text(encoding="utf-8")
    assert "out_dir=\"out\"" in txt
    assert "os.path.abspath(out_dir)" in txt


def test_fix_build_feedback_history_appended_and_trimmed(tmp_path: Path, monkeypatch):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "build.py").write_text("print('same')\n", encoding="utf-8")

    class _Patcher:
        def run_codex_command(self, *_args, **_kwargs):
            (tmp_path / "done").write_text("fuzz/build.py\n", encoding="utf-8")

    monkeypatch.setattr(workflow_graph, "_llm_or_none", lambda: None)
    monkeypatch.setenv("SHERPA_FIX_BUILD_FEEDBACK_HISTORY", "2")
    gen = SimpleNamespace(repo_root=tmp_path, patcher=_Patcher())
    state = {
        "generator": gen,
        "last_error": "build failed",
        "build_stdout_tail": "",
        "build_stderr_tail": "",
        "fix_build_attempt_history": [
            {"attempt_index": 1, "outcome": "noop"},
            {"attempt_index": 2, "outcome": "noop"},
        ],
    }
    out = workflow_graph._node_fix_build(state)
    hist = out.get("fix_build_attempt_history") or []
    assert len(hist) == 2
    assert hist[-1]["outcome"] == "noop"


def test_fix_build_rule_missing_zlib_link_flag_prefers_explicit_archive(tmp_path: Path, monkeypatch):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    build_py = fuzz_dir / "build.py"
    build_py.write_text(
        "import os\n"
        "build_dir = '/work/build'\n"
        "lib_path = ['-L' + build_dir]\n"
        "libs = ['-lz']\n",
        encoding="utf-8",
    )
    gen = SimpleNamespace(repo_root=tmp_path, patcher=SimpleNamespace(run_codex_command=lambda *_a, **_k: None))
    monkeypatch.setattr(workflow_graph, "_llm_or_none", lambda: None)
    out = workflow_graph._node_fix_build(
        {
            "generator": gen,
            "last_error": "/usr/bin/ld: cannot find -lz: No such file or directory",
            "build_stdout_tail": "",
            "build_stderr_tail": "",
        }
    )
    assert out["last_error"] == ""
    assert "missing_zlib_link_flag" in (out.get("fix_build_rule_hits") or [])
    txt = build_py.read_text(encoding="utf-8")
    assert "zlib_link_arg = '-lz'" in txt
    assert "libz.a" in txt
    assert "libs = [zlib_link_arg]" in txt


def test_fix_build_rule_collapsed_include_flags_split(tmp_path: Path, monkeypatch):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    build_py = fuzz_dir / "build.py"
    build_py.write_text(
        "cmd = ['clang++', '-I/work -I/work/build', 'harness.c', '-o', 'out/fz']\n",
        encoding="utf-8",
    )
    gen = SimpleNamespace(repo_root=tmp_path, patcher=SimpleNamespace(run_codex_command=lambda *_a, **_k: None))
    monkeypatch.setattr(workflow_graph, "_llm_or_none", lambda: None)
    out = workflow_graph._node_fix_build(
        {
            "generator": gen,
            "last_error": "fatal error: zlib.h: No such file or directory",
            "build_stdout_tail": "",
            "build_stderr_tail": "",
        }
    )
    assert out["last_error"] == ""
    assert "collapsed_include_flags" in (out.get("fix_build_rule_hits") or [])
    txt = build_py.read_text(encoding="utf-8")
    assert "'-I/work', '-I/work/build'" in txt


def test_fix_build_rule_cxx_for_c_source_mismatch(tmp_path: Path, monkeypatch):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    build_py = fuzz_dir / "build.py"
    build_py.write_text("cmd = ['clang++', 'harness.c', '-o', 'out/fz']\n", encoding="utf-8")
    gen = SimpleNamespace(repo_root=tmp_path, patcher=SimpleNamespace(run_codex_command=lambda *_a, **_k: None))
    monkeypatch.setattr(workflow_graph, "_llm_or_none", lambda: None)
    out = workflow_graph._node_fix_build(
        {
            "generator": gen,
            "last_error": "clang++: warning: treating 'c' input as 'c++' when in C++ mode",
            "build_stdout_tail": "",
            "build_stderr_tail": "",
        }
    )
    assert out["last_error"] == ""
    assert "cxx_for_c_source_mismatch" in (out.get("fix_build_rule_hits") or [])
    txt = build_py.read_text(encoding="utf-8")
    assert "clang++" not in txt
    assert "clang" in txt
