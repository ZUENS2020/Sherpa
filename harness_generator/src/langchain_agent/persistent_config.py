from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class WebPersistentConfig(BaseModel):
    # Chat (OpenRouter / OpenAI-compatible)
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "anthropic/claude-3.5-sonnet"

    # OpenCode / OpenAI
    openai_api_key: str | None = None
    # Optional: point OpenCode's OpenAI provider at an OpenAI-compatible proxy/router.
    # OPENAI_BASE_URL overrides the default endpoint.
    openai_base_url: str = ""
    openai_model: str = ""
    opencode_model: str = ""

    # Fuzz defaults
    fuzz_time_budget: int = 900
    fuzz_use_docker: bool = True
    fuzz_docker_image: str = "auto"

    # OSS-Fuzz (local checkout root)
    oss_fuzz_dir: str = ""

    # Git mirror / proxy
    sherpa_git_mirrors: str = ""
    sherpa_docker_http_proxy: str = ""
    sherpa_docker_https_proxy: str = ""
    sherpa_docker_no_proxy: str = ""
    sherpa_docker_proxy_host: str = "host.docker.internal"

    version: int = Field(default=1, description="Schema version")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    return _repo_root() / "config"


def config_path() -> Path:
    return config_dir() / "web_config.json"


def opencode_env_path() -> Path:
    # Used by fuzz pipeline (CodexHelper reads from a file path).
    return config_dir() / "web_opencode.env"


def load_config() -> WebPersistentConfig:
    path = config_path()
    if not path.is_file():
        cfg = WebPersistentConfig()
        cfg.fuzz_use_docker = True
        cfg.fuzz_docker_image = (cfg.fuzz_docker_image or "").strip() or "auto"
        default_oss_fuzz_dir = os.environ.get("SHERPA_DEFAULT_OSS_FUZZ_DIR", "").strip()
        if not cfg.oss_fuzz_dir.strip() and default_oss_fuzz_dir:
            cfg.oss_fuzz_dir = default_oss_fuzz_dir
        return cfg
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            cfg = WebPersistentConfig()
        else:
            cfg = WebPersistentConfig(**raw)

        cfg.fuzz_use_docker = True
        cfg.fuzz_docker_image = (cfg.fuzz_docker_image or "").strip() or "auto"
        default_oss_fuzz_dir = os.environ.get("SHERPA_DEFAULT_OSS_FUZZ_DIR", "").strip()
        if not cfg.oss_fuzz_dir.strip() and default_oss_fuzz_dir:
            cfg.oss_fuzz_dir = default_oss_fuzz_dir
        return cfg
    except Exception:
        cfg = WebPersistentConfig()
        cfg.fuzz_use_docker = True
        cfg.fuzz_docker_image = (cfg.fuzz_docker_image or "").strip() or "auto"
        default_oss_fuzz_dir = os.environ.get("SHERPA_DEFAULT_OSS_FUZZ_DIR", "").strip()
        if not cfg.oss_fuzz_dir.strip() and default_oss_fuzz_dir:
            cfg.oss_fuzz_dir = default_oss_fuzz_dir
        return cfg


def save_config(cfg: WebPersistentConfig) -> None:
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)

    path = config_path()
    payload = cfg.model_dump()

    tmp_fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(d))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        Path(tmp_name).replace(path)
    finally:
        try:
            if Path(tmp_name).exists() and str(Path(tmp_name)) != str(path):
                Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass


def _set_env_if_value(name: str, value: str | None) -> None:
    if value is None:
        return
    if isinstance(value, str) and value.strip() == "":
        os.environ.pop(name, None)
        return
    os.environ[name] = str(value)


def apply_config_to_env(cfg: WebPersistentConfig) -> None:
    # Chat / OpenRouter
    _set_env_if_value("OPENROUTER_API_KEY", cfg.openrouter_api_key)
    _set_env_if_value("OPENROUTER_BASE_URL", cfg.openrouter_base_url)
    _set_env_if_value("OPENROUTER_MODEL", cfg.openrouter_model)

    # OpenAI / OpenCode
    _set_env_if_value("OPENAI_API_KEY", cfg.openai_api_key)
    _set_env_if_value("OPENAI_BASE_URL", cfg.openai_base_url)
    _set_env_if_value("OPENAI_MODEL", cfg.openai_model)
    _set_env_if_value("OPENCODE_MODEL", cfg.opencode_model)

    # DeepSeek provider compatibility for OpenCode (when using DeepSeek base URL)
    if (cfg.openai_base_url or "").strip().startswith("https://api.deepseek.com"):
        _set_env_if_value("DEEPSEEK_API_KEY", cfg.openai_api_key)

    # Git mirror / proxy
    _set_env_if_value("SHERPA_GIT_MIRRORS", cfg.sherpa_git_mirrors)
    _set_env_if_value("SHERPA_DOCKER_HTTP_PROXY", cfg.sherpa_docker_http_proxy)
    _set_env_if_value("SHERPA_DOCKER_HTTPS_PROXY", cfg.sherpa_docker_https_proxy)
    _set_env_if_value("SHERPA_DOCKER_NO_PROXY", cfg.sherpa_docker_no_proxy)
    _set_env_if_value("SHERPA_DOCKER_PROXY_HOST", cfg.sherpa_docker_proxy_host)

    # Keep the OpenCode key file in sync for fuzz pipeline.
    write_opencode_env_file(cfg)


def write_opencode_env_file(cfg: WebPersistentConfig) -> None:
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = opencode_env_path()

    # Minimal env file used by CodexHelper(ai_key_path=...).
    # Prefer OPENAI_API_KEY (common, OpenAI-compatible).
    lines: list[str] = []
    if cfg.openai_api_key and cfg.openai_api_key.strip():
        key = cfg.openai_api_key.strip()
        lines.append(f"OPENAI_API_KEY={key}")

    if cfg.openai_base_url and cfg.openai_base_url.strip():
        lines.append(f"OPENAI_BASE_URL={cfg.openai_base_url.strip()}")

    if cfg.openai_model and cfg.openai_model.strip():
        lines.append(f"OPENAI_MODEL={cfg.openai_model.strip()}")

    content = "\n".join(lines) + ("\n" if lines else "")

    tmp_fd, tmp_name = tempfile.mkstemp(prefix=p.name + ".", dir=str(d))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(content)
        Path(tmp_name).replace(p)
    finally:
        try:
            if Path(tmp_name).exists() and str(Path(tmp_name)) != str(p):
                Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass


def as_public_dict(cfg: WebPersistentConfig) -> dict[str, Any]:
    data = cfg.model_dump()

    for key in ("openai_api_key", "openrouter_api_key"):
        raw = data.get(key)
        data[f"{key}_set"] = bool(isinstance(raw, str) and raw.strip())
        data[key] = ""

    return data
