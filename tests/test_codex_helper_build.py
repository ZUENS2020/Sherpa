from __future__ import annotations

import json
import logging
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


def test_gitnexus_prepare_context_runs_natively(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "sample.txt").write_text("hello\n", encoding="utf-8")
    shared_out = tmp_path / "shared-output"
    shared_out.mkdir()

    monkeypatch.setattr(ch, "_ensure_git_repo", lambda path: object())
    helper = ch.CodexHelper(repo_path=repo_dir, copy_repo=False)

    run_calls: list[tuple[list[str], dict | None]] = []
    analyze_calls: list[tuple[list[str], dict | None]] = []

    def _fake_run(cmd, *args, **kwargs):
        run_calls.append(([str(x) for x in cmd], kwargs.get("env")))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def _fake_stream(cmd, *args, **kwargs):
        analyze_calls.append(([str(x) for x in cmd], kwargs.get("env")))
        return (0, "ok", "ok")

    monkeypatch.setenv("SHERPA_GITNEXUS_AUTO_ANALYZE", "1")
    monkeypatch.setenv("SHERPA_GITNEXUS_SKIP_EMBEDDINGS", "1")
    monkeypatch.setenv("SHERPA_OUTPUT_DIR", str(shared_out))
    monkeypatch.setattr(ch.shutil, "which", lambda name: "/usr/bin/gitnexus" if name == "gitnexus" else None)
    monkeypatch.setattr(ch.subprocess, "run", _fake_run)
    monkeypatch.setattr(ch, "_run_streaming_combined", _fake_stream)

    helper._maybe_prepare_gitnexus_context()

    clean_cmds = [cmd for cmd, _env in run_calls if cmd[:4] == ["gitnexus", "clean", "--all", "--force"]]
    assert clean_cmds, run_calls
    assert analyze_calls
    analyze_cmd, analyze_env = analyze_calls[0]
    assert analyze_cmd[:2] == ["gitnexus", "analyze"]
    assert "docker" not in analyze_cmd
    assert analyze_env is not None
    assert str(analyze_env.get("HOME", "")).startswith(str(shared_out / ".opencode-home"))


def test_gitnexus_prepare_context_skips_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    shared_out = tmp_path / "shared-output"
    shared_out.mkdir()

    monkeypatch.setattr(ch, "_ensure_git_repo", lambda path: object())
    helper = ch.CodexHelper(repo_path=repo_dir, copy_repo=False)

    called = {"stream": False}

    def _fake_stream(*args, **kwargs):
        called["stream"] = True
        return (0, "", "")

    monkeypatch.setenv("SHERPA_GITNEXUS_AUTO_ANALYZE", "1")
    monkeypatch.setenv("SHERPA_OUTPUT_DIR", str(shared_out))
    monkeypatch.setattr(ch.shutil, "which", lambda _name: None)
    monkeypatch.setattr(ch, "_run_streaming_combined", _fake_stream)

    with caplog.at_level(logging.WARNING):
        helper._maybe_prepare_gitnexus_context()

    assert called["stream"] is False
    assert "gitnexus binary not found" in caplog.text


def test_gitnexus_prepare_context_honors_embeddings_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    shared_out = tmp_path / "shared-output"
    shared_out.mkdir()

    monkeypatch.setattr(ch, "_ensure_git_repo", lambda path: object())
    helper = ch.CodexHelper(repo_path=repo_dir, copy_repo=False)

    analyze_cmds: list[list[str]] = []

    monkeypatch.setenv("SHERPA_GITNEXUS_AUTO_ANALYZE", "1")
    monkeypatch.setenv("SHERPA_GITNEXUS_SKIP_EMBEDDINGS", "0")
    monkeypatch.setenv("SHERPA_OUTPUT_DIR", str(shared_out))
    monkeypatch.setattr(ch.shutil, "which", lambda name: "/usr/bin/gitnexus" if name == "gitnexus" else None)
    monkeypatch.setattr(ch.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(
        ch,
        "_run_streaming_combined",
        lambda cmd, *args, **kwargs: (analyze_cmds.append([str(x) for x in cmd]) or (0, "", "")),
    )

    helper._maybe_prepare_gitnexus_context()

    assert analyze_cmds
    assert "--embeddings" in analyze_cmds[0]


def test_gitnexus_prepare_context_skips_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    monkeypatch.setattr(ch, "_ensure_git_repo", lambda path: object())
    helper = ch.CodexHelper(repo_path=repo_dir, copy_repo=False)

    called = {"run": False}

    def _fake_run(*args, **kwargs):
        called["run"] = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("SHERPA_GITNEXUS_AUTO_ANALYZE", "0")
    monkeypatch.setattr(ch.subprocess, "run", _fake_run)

    helper._maybe_prepare_gitnexus_context()

    assert called["run"] is False
