from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "harness_generator" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import fuzz_unharnessed_repo as fmod
from fuzz_unharnessed_repo import NonOssFuzzHarnessGenerator


def _fake_generator(repo_root: Path) -> NonOssFuzzHarnessGenerator:
    gen = NonOssFuzzHarnessGenerator.__new__(NonOssFuzzHarnessGenerator)
    gen.repo_root = repo_root
    gen.docker_image = "sherpa-fuzz-cpp:latest"
    return gen


def test_dockerize_translates_fuzzer_binary_and_artifact_prefix_before_exec(tmp_path: Path):
    repo_root = tmp_path / "repo"
    bin_path = repo_root / "fuzz" / "out" / "fuzzer"
    corpus_dir = repo_root / "fuzz" / "corpus" / "fuzzer"
    artifacts_dir = repo_root / "fuzz" / "out" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    gen = _fake_generator(repo_root)
    cmd = [
        str(bin_path),
        f"-artifact_prefix={artifacts_dir}/",
        "-print_final_stats=1",
        "-max_total_time=5",
        "--",
        str(corpus_dir),
    ]

    docker_cmd = gen._dockerize_cmd(cmd, cwd=repo_root, env={"ASAN_OPTIONS": "exitcode=76"})
    joined = " ".join(docker_cmd)

    assert str(bin_path) not in joined
    assert str(corpus_dir) not in joined
    assert str(artifacts_dir) not in joined

    assert "/work/fuzz/out/fuzzer" in joined
    assert "-artifact_prefix=/work/fuzz/out/artifacts/" in joined
    assert "/work/fuzz/corpus/fuzzer" in joined


def test_dockerize_autoinstall_triggers_for_build_py_from_fuzz_cwd(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    fuzz_dir = repo_root / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)

    gen = _fake_generator(repo_root)
    monkeypatch.setenv("SHERPA_AUTO_INSTALL_SYSTEM_DEPS", "1")

    docker_cmd = gen._dockerize_cmd(["python3", "build.py"], cwd=fuzz_dir, env={})
    joined = " ".join(docker_cmd)

    assert "-w /work/fuzz" in joined
    assert "dep_file=/work/fuzz/system_packages.txt" in joined
    assert "set -u" in joined
    assert "(docker/deps) installing from" in joined
    assert "continuing without auto-install" in joined
    assert "exec python3 build.py" in joined


def test_ensure_docker_image_switches_to_classic_builder_after_buildx_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")

    gen = _fake_generator(repo_root)

    calls: list[tuple[list[str], str | None]] = []
    scenarios = [
        (1, "ERROR: BuildKit is enabled but the buildx component is missing or broken.\n"),
        (0, "Successfully built image.\n"),
    ]

    class _FakeProc:
        def __init__(self, output: str, rc: int):
            self.stdout = io.StringIO(output)
            self.returncode: int | None = None
            self._rc = rc

        def wait(self):
            self.returncode = self._rc
            return self._rc

    def _fake_run(cmd, *args, **kwargs):
        c = list(cmd)
        if c[:2] == ["docker", "info"]:
            return SimpleNamespace(returncode=0)
        if c[:3] == ["docker", "image", "inspect"] and len(c) >= 4 and c[3] == "busybox:latest":
            return SimpleNamespace(returncode=1)
        if c[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    def _fake_popen(cmd, *args, **kwargs):
        env = kwargs.get("env") or {}
        calls.append((list(cmd), env.get("DOCKER_BUILDKIT")))
        if not scenarios:
            raise AssertionError("unexpected docker build invocation")
        rc, out = scenarios.pop(0)
        return _FakeProc(out, rc)

    monkeypatch.setattr(fmod.subprocess, "run", _fake_run)
    monkeypatch.setattr(fmod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(fmod.time, "sleep", lambda _: None)

    gen._ensure_docker_image("test-image:latest", dockerfile=dockerfile)

    assert len(calls) == 2
    assert any("--progress=plain" in arg for arg in calls[0][0])
    assert calls[0][1] == "1"
    assert not any("--progress=plain" in arg for arg in calls[1][0])
    assert calls[1][1] == "0"


def test_ensure_docker_image_retries_on_registry_eof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")

    gen = _fake_generator(repo_root)
    calls: list[tuple[list[str], str | None]] = []
    scenarios = [
        (1, 'Get "https://registry-1.docker.io/v2/": EOF\n'),
        (1, 'Get "https://registry-1.docker.io/v2/": EOF\n'),
        (0, "Successfully built image.\n"),
    ]

    class _FakeProc:
        def __init__(self, output: str, rc: int):
            self.stdout = io.StringIO(output)
            self.returncode: int | None = None
            self._rc = rc

        def wait(self):
            self.returncode = self._rc
            return self._rc

    def _fake_run(cmd, *args, **kwargs):
        c = list(cmd)
        if c[:2] == ["docker", "info"]:
            return SimpleNamespace(returncode=0)
        if c[:3] == ["docker", "image", "inspect"] and len(c) >= 4 and c[3] == "busybox:latest":
            return SimpleNamespace(returncode=1)
        if c[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    def _fake_popen(cmd, *args, **kwargs):
        env = kwargs.get("env") or {}
        calls.append((list(cmd), env.get("DOCKER_BUILDKIT")))
        if not scenarios:
            raise AssertionError("unexpected docker build invocation")
        rc, out = scenarios.pop(0)
        return _FakeProc(out, rc)

    monkeypatch.setattr(fmod.subprocess, "run", _fake_run)
    monkeypatch.setattr(fmod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(fmod.time, "sleep", lambda _: None)
    monkeypatch.setenv("SHERPA_DOCKER_BUILD_RETRIES", "3")
    monkeypatch.delenv("DOCKER_BUILDKIT", raising=False)

    gen._ensure_docker_image("test-image:latest", dockerfile=dockerfile)

    assert len(calls) == 3


def test_workflow_opencode_cli_retries_default_and_bounds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SHERPA_WORKFLOW_OPENCODE_CLI_RETRIES", raising=False)
    assert fmod._workflow_opencode_cli_retries() == 2

    monkeypatch.setenv("SHERPA_WORKFLOW_OPENCODE_CLI_RETRIES", "0")
    assert fmod._workflow_opencode_cli_retries() == 1

    monkeypatch.setenv("SHERPA_WORKFLOW_OPENCODE_CLI_RETRIES", "99")
    assert fmod._workflow_opencode_cli_retries() == 8

    monkeypatch.setenv("SHERPA_WORKFLOW_OPENCODE_CLI_RETRIES", "bad")
    assert fmod._workflow_opencode_cli_retries() == 2
