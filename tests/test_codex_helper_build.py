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
        if c[:2] == ["docker", "build"]:
            build_calls.append(c)
            base = _extract_build_arg(c, "OPENCODE_BASE_IMAGE")
            if base == "node:20-slim":
                return SimpleNamespace(
                    returncode=1,
                    stdout=(
                        "failed to fetch anonymous token: "
                        'Get "https://auth.docker.io/token?...": net/http: TLS handshake timeout'
                    ),
                    stderr="",
                )
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ch.subprocess, "run", _fake_run)
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
