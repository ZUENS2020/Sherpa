from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


_OPENCODE_SCHEMA_URL = "https://opencode.ai/config.json"
_OPENCODE_PROVIDER_MODEL_CANDIDATES: dict[str, list[str]] = {
    "openrouter": [
        "qwen/qwen3-coder-next",
        "anthropic/claude-3.7-sonnet",
        "deepseek/deepseek-reasoner",
    ],
    "deepseek": [
        "deepseek/deepseek-reasoner",
        "deepseek/deepseek-chat",
        "deepseek/deepseek-v3.2",
    ],
    "minimax": [
        "minimax/minimax-m2.1",
        "minimax/minimax-m2",
        "minimax/minimax-m2.1-lightning",
    ],
    "zai": [
        "zai/glm-4.7",
        "zai/glm-4.6",
        "zai/glm-4.5",
    ],
}
_OPENCODE_PROVIDER_ALIASES: dict[str, str] = {
    "glm": "zai",
    "zai": "zai",
    "zhipuai": "zai",
}


class OpencodeProviderConfig(BaseModel):
    name: str
    enabled: bool = True
    base_url: str = ""
    api_key: str | None = None
    clear_api_key: bool = False
    models: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


def _default_opencode_providers() -> list[OpencodeProviderConfig]:
    return [
        OpencodeProviderConfig(
            name="openrouter",
            enabled=True,
            base_url="https://openrouter.ai/api/v1",
            models=["qwen/qwen3-coder-next"],
        ),
        OpencodeProviderConfig(
            name="zai",
            enabled=False,
            base_url="https://open.bigmodel.cn/api/coding/paas/v4",
            models=["zai/glm-4.7"],
        ),
        OpencodeProviderConfig(
            name="deepseek",
            enabled=False,
            base_url="https://api.deepseek.com/v1",
            models=["deepseek/deepseek-reasoner"],
        ),
        OpencodeProviderConfig(
            name="minimax",
            enabled=False,
            base_url="https://api.minimax.io/v1",
            models=[],
        ),
    ]


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
    opencode_providers: list[OpencodeProviderConfig] = Field(default_factory=_default_opencode_providers)

    # Fuzz defaults
    fuzz_time_budget: int = 900
    # Per-round cap (seconds) when both total/run budgets are unlimited (0).
    # 0 means fully unlimited.
    sherpa_run_unlimited_round_budget_sec: int = 7200
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


def opencode_runtime_config_path() -> Path:
    raw = os.environ.get("SHERPA_OPENCODE_CONFIG_PATH", "").strip()
    if raw:
        return Path(raw).expanduser()
    return config_dir() / "opencode.generated.json"


def _normalize_provider_name(raw: str) -> str:
    name = (raw or "").strip().lower()
    if not name:
        return ""
    return _OPENCODE_PROVIDER_ALIASES.get(name, name)


def list_opencode_provider_models(provider: str) -> tuple[str, list[str]]:
    normalized = _normalize_provider_name(provider)
    if not normalized:
        return "", []
    items = list(_OPENCODE_PROVIDER_MODEL_CANDIDATES.get(normalized, []))
    return normalized, items


def _provider_config_by_name(cfg: "WebPersistentConfig", provider: str) -> OpencodeProviderConfig | None:
    name = _normalize_provider_name(provider)
    if not name:
        return None
    for item in normalize_opencode_providers(cfg.opencode_providers):
        if item.name == name:
            return item
    return None


def _default_provider_config(provider: str) -> OpencodeProviderConfig | None:
    name = _normalize_provider_name(provider)
    if not name:
        return None
    for item in _default_opencode_providers():
        if item.name == name:
            return item
    return None


def _best_provider_base_url(cfg: "WebPersistentConfig", provider: str) -> str:
    item = _provider_config_by_name(cfg, provider)
    if item and item.base_url.strip():
        return item.base_url.strip()
    default_item = _default_provider_config(provider)
    if default_item and default_item.base_url.strip():
        return default_item.base_url.strip()
    return ""


def _provider_from_base_url(raw_url: str) -> str:
    url = (raw_url or "").strip().lower()
    if not url:
        return ""
    if "openrouter.ai" in url:
        return "openrouter"
    if "deepseek.com" in url:
        return "deepseek"
    if "bigmodel.cn" in url or "zhipuai.cn" in url:
        return "zai"
    if "minimax.io" in url:
        return "minimax"
    return ""


def _best_provider_api_key(cfg: "WebPersistentConfig", provider: str) -> str:
    normalized = _normalize_provider_name(provider)
    item = _provider_config_by_name(cfg, normalized)
    if item and item.api_key and item.api_key.strip():
        return item.api_key.strip()
    if normalized == "openrouter" and cfg.openrouter_api_key and cfg.openrouter_api_key.strip():
        return cfg.openrouter_api_key.strip()
    # Legacy fallback for OPENAI_* fields: only when base_url clearly maps to the same provider.
    openai_provider = _provider_from_base_url(cfg.openai_base_url)
    if (
        normalized in {"deepseek", "zai", "minimax"}
        and normalized == openai_provider
        and cfg.openai_api_key
        and cfg.openai_api_key.strip()
    ):
        return cfg.openai_api_key.strip()
    return ""


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _extract_models_from_payload(payload: Any) -> list[str]:
    out: list[str] = []

    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    mid = item.get("id") or item.get("model") or item.get("model_name") or item.get("name")
                    if isinstance(mid, str):
                        out.append(mid)
                elif isinstance(item, str):
                    out.append(item)

        models = payload.get("models")
        if isinstance(models, list):
            for item in models:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    mid = item.get("id") or item.get("model") or item.get("model_name") or item.get("name")
                    if isinstance(mid, str):
                        out.append(mid)
        elif isinstance(models, dict):
            for key in models.keys():
                out.append(str(key))

        if isinstance(data, dict):
            model_list = data.get("model_list")
            if isinstance(model_list, list):
                for item in model_list:
                    if isinstance(item, dict):
                        mid = item.get("model") or item.get("model_name") or item.get("id") or item.get("name")
                        if isinstance(mid, str):
                            out.append(mid)
                    elif isinstance(item, str):
                        out.append(item)

    return _dedupe_keep_order(out)


def _http_get_json(url: str, *, api_key: str = "", timeout_s: float = 8.0) -> Any:
    headers = {
        "Accept": "application/json",
        "User-Agent": "sherpa-web/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8", errors="replace"))


def _fetch_models_openrouter(base_url: str, *, api_key: str = "") -> list[str]:
    endpoint = f"{base_url.rstrip('/')}/models"
    payload = _http_get_json(endpoint, api_key=api_key)
    return _extract_models_from_payload(payload)


def _fetch_models_openai_compatible(base_url: str, *, api_key: str = "") -> list[str]:
    endpoint = f"{base_url.rstrip('/')}/models"
    payload = _http_get_json(endpoint, api_key=api_key)
    return _extract_models_from_payload(payload)


def list_opencode_provider_models_resolved(
    provider: str,
    cfg: "WebPersistentConfig",
    *,
    api_key_override: str | None = None,
    base_url_override: str | None = None,
) -> tuple[str, list[str], str, str]:
    normalized = _normalize_provider_name(provider)
    if not normalized:
        return "", [], "none", "provider is required"

    fallback = list(_OPENCODE_PROVIDER_MODEL_CANDIDATES.get(normalized, []))
    if not fallback:
        return normalized, [], "none", f"unsupported provider: {provider}"

    base_url = (base_url_override or "").strip() or _best_provider_base_url(cfg, normalized)
    api_key = (api_key_override or "").strip() or _best_provider_api_key(cfg, normalized)
    if not base_url:
        return normalized, fallback, "builtin", "provider base_url not configured"
    if normalized in {"deepseek", "zai", "minimax"} and not api_key:
        return normalized, fallback, "builtin", "provider api_key not configured"

    try:
        if normalized == "openrouter":
            remote = _fetch_models_openrouter(base_url, api_key=api_key)
        else:
            remote = _fetch_models_openai_compatible(base_url, api_key=api_key)
        remote = _dedupe_keep_order(remote)
        if remote:
            return normalized, remote, "remote", ""
        return normalized, fallback, "builtin", "provider returned empty model list"
    except urllib.error.HTTPError as e:
        return normalized, fallback, "builtin", f"provider HTTP {e.code}"
    except urllib.error.URLError as e:
        return normalized, fallback, "builtin", f"provider unreachable: {e.reason}"
    except Exception as e:
        return normalized, fallback, "builtin", f"provider fetch failed: {e}"


def _normalize_provider_entry(entry: OpencodeProviderConfig) -> OpencodeProviderConfig | None:
    name = _normalize_provider_name(entry.name)
    if not name:
        return None

    base_url = (entry.base_url or "").strip()
    if name == "zai":
        legacy = "https://open.bigmodel.cn/api/paas/v4"
        coding = "https://open.bigmodel.cn/api/coding/paas/v4"
        if base_url.rstrip("/") == legacy:
            base_url = coding

    models: list[str] = []
    seen_models: set[str] = set()
    for model in entry.models:
        m = str(model or "").strip()
        if not m or m in seen_models:
            continue
        seen_models.add(m)
        models.append(m)

    headers: dict[str, str] = {}
    for k, v in (entry.headers or {}).items():
        kk = str(k or "").strip()
        vv = str(v or "").strip()
        if kk and vv:
            headers[kk] = vv

    options: dict[str, Any] = {}
    if isinstance(entry.options, dict):
        for k, v in entry.options.items():
            kk = str(k or "").strip()
            if not kk:
                continue
            options[kk] = v

    api_key = (entry.api_key or "").strip()
    return OpencodeProviderConfig(
        name=name,
        enabled=bool(entry.enabled),
        base_url=base_url,
        api_key=(api_key if api_key else None),
        clear_api_key=bool(entry.clear_api_key),
        models=models,
        headers=headers,
        options=options,
    )


def normalize_opencode_providers(entries: list[OpencodeProviderConfig] | None) -> list[OpencodeProviderConfig]:
    normalized: list[OpencodeProviderConfig] = []
    seen_names: set[str] = set()
    for raw in entries or []:
        item = _normalize_provider_entry(raw)
        if item is None:
            continue
        if item.name in seen_names:
            continue
        seen_names.add(item.name)
        normalized.append(item)
    return normalized


def _build_provider_node(entry: OpencodeProviderConfig) -> dict[str, Any]:
    node: dict[str, Any] = {}
    options: dict[str, Any] = {}

    if isinstance(entry.options, dict):
        options.update(entry.options)

    if entry.base_url:
        options["baseURL"] = entry.base_url

    if entry.api_key and entry.api_key.strip():
        options["apiKey"] = entry.api_key.strip()

    if entry.headers:
        existing_headers = options.get("headers")
        merged_headers: dict[str, Any] = {}
        if isinstance(existing_headers, dict):
            merged_headers.update(existing_headers)
        merged_headers.update(entry.headers)
        options["headers"] = merged_headers

    if options:
        node["options"] = options

    models: dict[str, dict[str, Any]] = {}
    for m in entry.models:
        model_name = str(m or "").strip()
        if model_name:
            models[model_name] = {}
    if models:
        node["models"] = models

    return node


def build_opencode_runtime_config(cfg: WebPersistentConfig) -> dict[str, Any]:
    providers: dict[str, Any] = {}
    for item in normalize_opencode_providers(cfg.opencode_providers):
        if not item.enabled:
            continue
        providers[item.name] = _build_provider_node(item)

    # Compatibility fallback for legacy fields when provider list is empty.
    if not providers:
        openrouter_node: dict[str, Any] = {}
        openrouter_options: dict[str, Any] = {}
        if cfg.openrouter_base_url.strip():
            openrouter_options["baseURL"] = cfg.openrouter_base_url.strip()
        if cfg.openrouter_api_key and cfg.openrouter_api_key.strip():
            openrouter_options["apiKey"] = cfg.openrouter_api_key.strip()
        if openrouter_options:
            openrouter_node["options"] = openrouter_options
        if cfg.openrouter_model.strip():
            openrouter_node["models"] = {cfg.openrouter_model.strip(): {}}
        if openrouter_node:
            providers["openrouter"] = openrouter_node

        openai_node: dict[str, Any] = {}
        openai_options: dict[str, Any] = {}
        if cfg.openai_base_url.strip():
            openai_options["baseURL"] = cfg.openai_base_url.strip()
        if cfg.openai_api_key and cfg.openai_api_key.strip():
            openai_options["apiKey"] = cfg.openai_api_key.strip()
        if openai_options:
            openai_node["options"] = openai_options
        model_name = (cfg.opencode_model or cfg.openai_model or "").strip()
        if model_name:
            openai_node["models"] = {model_name: {}}
        if openai_node:
            providers["openai"] = openai_node

    return {
        "$schema": _OPENCODE_SCHEMA_URL,
        "provider": providers,
        "mcp": {
            "gitnexus": {
                "type": "local",
                "command": ["gitnexus", "mcp"],
            }
        },
    }


def write_opencode_runtime_config_file(cfg: WebPersistentConfig) -> Path:
    p = opencode_runtime_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = build_opencode_runtime_config(cfg)

    tmp_fd, tmp_name = tempfile.mkstemp(prefix=p.name + ".", dir=str(p.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        Path(tmp_name).replace(p)
    finally:
        try:
            if Path(tmp_name).exists() and str(Path(tmp_name)) != str(p):
                Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass
    return p


def load_config() -> WebPersistentConfig:
    path = config_path()
    if not path.is_file():
        cfg = WebPersistentConfig()
        cfg.fuzz_use_docker = True
        cfg.fuzz_docker_image = (cfg.fuzz_docker_image or "").strip() or "auto"
        cfg.opencode_providers = normalize_opencode_providers(cfg.opencode_providers)
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
        cfg.opencode_providers = normalize_opencode_providers(cfg.opencode_providers)
        default_oss_fuzz_dir = os.environ.get("SHERPA_DEFAULT_OSS_FUZZ_DIR", "").strip()
        if not cfg.oss_fuzz_dir.strip() and default_oss_fuzz_dir:
            cfg.oss_fuzz_dir = default_oss_fuzz_dir
        return cfg
    except Exception:
        cfg = WebPersistentConfig()
        cfg.fuzz_use_docker = True
        cfg.fuzz_docker_image = (cfg.fuzz_docker_image or "").strip() or "auto"
        cfg.opencode_providers = normalize_opencode_providers(cfg.opencode_providers)
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
    _set_env_if_value(
        "SHERPA_RUN_UNLIMITED_ROUND_BUDGET_SEC",
        str(int(cfg.sherpa_run_unlimited_round_budget_sec)),
    )

    # Keep the OpenCode key file in sync for fuzz pipeline.
    write_opencode_env_file(cfg)
    cfg.opencode_providers = normalize_opencode_providers(cfg.opencode_providers)
    config_path = write_opencode_runtime_config_file(cfg)
    _set_env_if_value("OPENCODE_CONFIG", str(config_path))


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

    providers = data.get("opencode_providers")
    if isinstance(providers, list):
        for item in providers:
            if not isinstance(item, dict):
                continue
            raw = item.get("api_key")
            item["api_key_set"] = bool(isinstance(raw, str) and raw.strip())
            item["api_key"] = ""
            item["clear_api_key"] = False

    return data
