from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import persistent_config as pc


def test_generated_runtime_paths_default_to_tmp(monkeypatch):
    monkeypatch.delenv("SHERPA_RUNTIME_CONFIG_DIR", raising=False)

    assert str(pc.runtime_generated_dir()) == "/tmp/sherpa-runtime"
    assert str(pc.opencode_env_path()) == "/tmp/sherpa-runtime/web_opencode.env"
    assert str(pc.opencode_runtime_config_path()) == "/tmp/sherpa-runtime/opencode.generated.json"


def test_generated_runtime_paths_honor_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SHERPA_RUNTIME_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("SHERPA_OPENCODE_CONFIG_PATH", raising=False)

    assert pc.runtime_generated_dir() == tmp_path
    assert pc.opencode_env_path() == tmp_path / "web_opencode.env"
    assert pc.opencode_runtime_config_path() == tmp_path / "opencode.generated.json"


def test_runtime_config_path_prefers_explicit_config_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SHERPA_RUNTIME_CONFIG_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("SHERPA_OPENCODE_CONFIG_PATH", str(tmp_path / "custom.json"))

    assert pc.opencode_runtime_config_path() == tmp_path / "custom.json"
