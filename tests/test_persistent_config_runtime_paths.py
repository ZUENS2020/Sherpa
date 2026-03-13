from __future__ import annotations

import errno
import sys
import tempfile
from pathlib import Path

import pytest


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


def test_save_config_uses_runtime_dir_for_tempfiles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    runtime_dir = tmp_path / "runtime"
    config_dir = tmp_path / "config"
    config_file = config_dir / "web_config.json"
    captured_dirs: list[str] = []
    real_mkstemp = tempfile.mkstemp

    monkeypatch.setattr(pc, "runtime_generated_dir", lambda: runtime_dir)
    monkeypatch.setattr(pc, "config_path", lambda: config_file)

    def _wrapped_mkstemp(*args, **kwargs):
        captured_dirs.append(str(kwargs.get("dir")))
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(pc.tempfile, "mkstemp", _wrapped_mkstemp)

    cfg = pc.WebPersistentConfig(openrouter_model="test-model")
    pc.save_config(cfg)

    assert config_file.is_file()
    assert captured_dirs == [str(runtime_dir)]


def test_save_config_falls_back_on_cross_device_replace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    runtime_dir = tmp_path / "runtime"
    config_dir = tmp_path / "config"
    config_file = config_dir / "web_config.json"

    monkeypatch.setattr(pc, "runtime_generated_dir", lambda: runtime_dir)
    monkeypatch.setattr(pc, "config_path", lambda: config_file)

    real_replace = Path.replace

    def _replace_once(self: Path, target: Path):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(pc.Path, "replace", _replace_once)

    cfg = pc.WebPersistentConfig(openrouter_model="cross-device-model")
    pc.save_config(cfg)

    assert config_file.is_file()
    assert "cross-device-model" in config_file.read_text(encoding="utf-8")

    monkeypatch.setattr(pc.Path, "replace", real_replace)


def test_write_opencode_env_uses_runtime_parent_and_cross_device_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runtime_dir = tmp_path / "runtime"
    env_file = runtime_dir / "web_opencode.env"
    captured_dirs: list[str] = []
    real_mkstemp = tempfile.mkstemp

    monkeypatch.setattr(pc, "opencode_env_path", lambda: env_file)

    def _wrapped_mkstemp(*args, **kwargs):
        captured_dirs.append(str(kwargs.get("dir")))
        return real_mkstemp(*args, **kwargs)

    def _replace_once(self: Path, target: Path):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(pc.tempfile, "mkstemp", _wrapped_mkstemp)
    monkeypatch.setattr(pc.Path, "replace", _replace_once)

    cfg = pc.WebPersistentConfig(
        openai_api_key="key",
        openai_base_url="https://example.invalid/v1",
        openai_model="model-x",
    )
    pc.write_opencode_env_file(cfg)

    assert captured_dirs == [str(runtime_dir)]
    assert env_file.is_file()
    content = env_file.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=key" in content
    assert "OPENAI_BASE_URL=https://example.invalid/v1" in content
    assert "OPENAI_MODEL=model-x" in content
