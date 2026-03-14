from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "harness_generator" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import codex_helper as ch


def _extract_build_arg(cmd: list[str], name: str) -> str:
    needle = f"{name}="
    for i, part in enumerate(cmd):
        if part == "--build-arg" and i + 1 < len(cmd) and str(cmd[i + 1]).startswith(needle):
            return str(cmd[i + 1])[len(needle) :]
    return ""


def test_ensure_opencode_image_falls_back_to_mirror_base_image(monkeypatch: pytest.MonkeyPatch):
    ch._ENSURED_OPENCODE_IMAGES.clear()

    build_calls: list[list[str]] = []

    def _fake_run(cmd, *args, **kwargs):
        c = [str(x) for x in cmd]
        if c[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def _fake_stream(cmd, *args, **kwargs):
        c = [str(x) for x in cmd]
        if c[:2] == ["docker", "build"]:
            build_calls.append(c)
            base = _extract_build_arg(c, "OPENCODE_BASE_IMAGE")
            if base == "node:20-slim":
                return (
                    1,
                    (
                        "failed to fetch anonymous token: "
                        'Get "https://auth.docker.io/token?...": net/http: TLS handshake timeout'
                    ),
                    "tls handshake timeout",
                )
            return (0, "ok", "ok")
        return (0, "", "")

    monkeypatch.setattr(ch.subprocess, "run", _fake_run)
    monkeypatch.setattr(ch, "_run_streaming_combined", _fake_stream)
    monkeypatch.setattr(ch.time, "sleep", lambda _: None)
    monkeypatch.setenv(
        "SHERPA_OPENCODE_BASE_IMAGES",
        "node:20-slim,m.daocloud.io/docker.io/library/node:20-slim",
    )
    monkeypatch.setenv("SHERPA_OPENCODE_BUILD_RETRIES", "1")

    ch._ensure_opencode_image("sherpa-opencode:test", env={})

    assert len(build_calls) >= 2
    assert _extract_build_arg(build_calls[0], "OPENCODE_BASE_IMAGE") == "node:20-slim"
    assert _extract_build_arg(build_calls[1], "OPENCODE_BASE_IMAGE") == "m.daocloud.io/docker.io/library/node:20-slim"


def test_normalize_model_for_opencode_prefixes_single_configured_provider(tmp_path: Path):
    config_path = tmp_path / "opencode.generated.json"
    config_path.write_text(
        json.dumps(
            {
                "provider": {
                    "zai": {
                        "models": {
                            "glm-4.7": {},
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    out = ch._normalize_model_for_opencode("glm-4.7", config_path=str(config_path))
    assert out == "zai/glm-4.7"


def test_resolve_opencode_home_dir_isolated_by_repo_name():
    shared_out = "/shared/output"
    a = ch._resolve_opencode_home_dir(shared_out, working_dir=Path("/shared/output/zlib-a1"))
    b = ch._resolve_opencode_home_dir(shared_out, working_dir=Path("/shared/output/zlib-b2"))

    assert a.startswith("/shared/output/.opencode-home/")
    assert b.startswith("/shared/output/.opencode-home/")
    assert a != b


def test_redact_cmd_for_log_masks_env_values():
    cmd = [
        "docker",
        "run",
        "-e",
        "OPENAI_API_KEY=sk-abc",
        "-e",
        "FOO=bar",
        "img",
    ]
    out = ch._redact_cmd_for_log(cmd, env={"OPENAI_API_KEY": "sk-abc"})
    assert "sk-abc" not in out
    assert "OPENAI_API_KEY=***" in out


def test_run_streaming_combined_redacts_output(monkeypatch: pytest.MonkeyPatch):
    class _FakeStdout:
        def __iter__(self):
            yield "OPENAI_API_KEY=sk-out-secret\n"
            yield "Authorization: Bearer sk-out-secret\n"
        def close(self):
            return None

    class _FakeProc:
        def __init__(self):
            self.stdout = _FakeStdout()
        def wait(self):
            return 0

    monkeypatch.setattr(
        ch.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProc(),
    )
    rc, scan, tail = ch._run_streaming_combined(
        ["echo", "x"],
        env={"OPENAI_API_KEY": "sk-out-secret"},
    )
    assert rc == 0
    assert "sk-out-secret" not in scan
    assert "sk-out-secret" not in tail
    assert "OPENAI_API_KEY=***" in scan


def test_opencode_k8s_job_forces_native_even_with_docker_image(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SHERPA_EXECUTOR_MODE", "k8s_job")
    monkeypatch.setenv("SHERPA_OPENCODE_DOCKER_IMAGE", "sherpa-opencode:latest")

    assert ch._opencode_container_mode_enabled() is False
    assert ch._build_opencode_cmd("opencode", ["run", "prompt"], Path("/tmp/repo"), {}) == [
        "opencode",
        "run",
        "prompt",
    ]


def test_opencode_non_k8s_keeps_container_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SHERPA_EXECUTOR_MODE", "docker")
    monkeypatch.setenv("SHERPA_OPENCODE_DOCKER_IMAGE", "sherpa-opencode:latest")

    env = {
        "SHERPA_OUTPUT_DIR": "/shared/output",
        "OPENCODE_CONFIG": "/app/config/opencode.generated.json",
    }
    cmd = ch._build_opencode_cmd("opencode", ["run", "prompt"], Path("/tmp/repo"), env)

    assert ch._opencode_container_mode_enabled() is True
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "sherpa-opencode:latest" in cmd


def test_build_blocklist_never_blocks_grep_family(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(
        "SHERPA_OPENCODE_BLOCKLIST",
        "grep,egrep,fgrep,rg,ripgrep,make,python",
    )

    blocked = ch._build_blocklist()

    assert "make" in blocked
    assert "python" in blocked
    assert "grep" not in blocked
    assert "egrep" not in blocked
    assert "fgrep" not in blocked
    assert "rg" not in blocked
    assert "ripgrep" not in blocked
