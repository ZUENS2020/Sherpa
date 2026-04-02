from __future__ import annotations

import json
import errno
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


_OPENCODE_SCHEMA_URL = "https://opencode.ai/config.json"

# ---------------------------------------------------------------------------
# Data-driven provider registry — add new providers here, no code changes needed.
# ---------------------------------------------------------------------------

class _ProviderDef:
    __slots__ = ("name", "base_url", "default_model", "models", "npm", "aliases")

    def __init__(
        self,
        name: str,
        base_url: str,
        default_model: str,
        models: list[str],
        npm: str,
        aliases: list[str] | None = None,
    ):
        self.name = name
        self.base_url = base_url
        self.default_model = default_model
        self.models = models
        self.npm = npm
        self.aliases = aliases or []


KNOWN_PROVIDERS: dict[str, _ProviderDef] = {}
_PROVIDER_ALIASES: dict[str, str] = {}

def _register(*defs: _ProviderDef) -> None:
    for d in defs:
        KNOWN_PROVIDERS[d.name] = d
        _PROVIDER_ALIASES[d.name] = d.name
        for alias in d.aliases:
            _PROVIDER_ALIASES[alias] = d.name

_register(
    _ProviderDef(
        name="minimax",
        base_url="https://api.minimaxi.com/anthropic/v1",
        default_model="MiniMax-M2.7-highspeed",
        models=["MiniMax-M2.7-highspeed", "MiniMax-Text-01"],
        npm="@ai-sdk/anthropic",
        aliases=["mini-max", "minimaxi"],
    ),
    _ProviderDef(
        name="jdcloud",
        base_url="https://modelservice.jdcloud.com/coding/openai/v1",
        default_model="GLM-5",
        models=["GLM-5", "glm-5"],
        npm="@ai-sdk/openai-compatible",
        aliases=["jdaip", "jd-openai", "jdcloud-opencode"],
    ),
    _ProviderDef(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-reasoner",
        models=["deepseek-reasoner", "deepseek-chat", "reasoner"],
        npm="@ai-sdk/openai-compatible",
        aliases=["deep-seek"],
    ),
    _ProviderDef(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="anthropic/claude-3.5-sonnet",
        models=["anthropic/claude-3.5-sonnet"],
        npm="@ai-sdk/openai-compatible",
    ),
)

# Backward-compatible constants (used elsewhere in the codebase)
_MINIMAX_PROVIDER = "minimax"
_DEEPSEEK_PROVIDER = "deepseek"
_DEEPSEEK_BASE_URL = KNOWN_PROVIDERS["deepseek"].base_url
_MINIMAX_BASE_URL = KNOWN_PROVIDERS["minimax"].base_url
_DEEPSEEK_DEFAULT_MODEL = KNOWN_PROVIDERS["deepseek"].default_model


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
            name=_DEEPSEEK_PROVIDER,
            enabled=True,
            base_url=_DEEPSEEK_BASE_URL,
            models=[_DEEPSEEK_DEFAULT_MODEL],
        ),
    ]


class WebPersistentConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    # Chat (OpenRouter / OpenAI-compatible)
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "anthropic/claude-3.5-sonnet"

    # OpenCode / OpenAI
    openai_api_key: str | None = None
    # Optional: point OpenCode's OpenAI provider at an OpenAI-compatible proxy/router.
    # OPENAI_BASE_URL overrides the default endpoint.
    openai_base_url: str = ""
    openai_model: str = _DEEPSEEK_DEFAULT_MODEL
    opencode_model: str = _DEEPSEEK_DEFAULT_MODEL
    opencode_providers: list[OpencodeProviderConfig] = Field(default_factory=_default_opencode_providers)

    # Fuzz defaults
    fuzz_time_budget: int = 900
    # Per-round cap (seconds) when both total/run budgets are unlimited (0).
    # 0 means fully unlimited.
    sherpa_run_unlimited_round_budget_sec: int = 7200
    sherpa_run_plateau_idle_growth_sec: int = 600
    fuzz_use_docker: bool = False
    fuzz_docker_image: str = ""

    # OSS-Fuzz (local checkout root)
    oss_fuzz_dir: str = ""

    # Git mirror / proxy
    sherpa_git_mirrors: str = ""
    sherpa_docker_http_proxy: str = ""
    sherpa_docker_https_proxy: str = ""
    sherpa_docker_no_proxy: str = ""
    sherpa_docker_proxy_host: str = "host.docker.internal"
    api_base_url: str = Field(default="", alias="apiBaseUrl")

    version: int = Field(default=1, description="Schema version")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    return _repo_root() / "config"


def runtime_generated_dir() -> Path:
    raw = os.environ.get("SHERPA_RUNTIME_CONFIG_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path("/tmp/sherpa-runtime")


def config_path() -> Path:
    return config_dir() / "web_config.json"


def opencode_env_path() -> Path:
    # Used by fuzz pipeline (CodexHelper reads from a file path).
    # Keep generated runtime files out of /app/config so non-root web pods
    # don't need write access to the config PVC.
    return runtime_generated_dir() / "web_opencode.env"


def opencode_runtime_config_path() -> Path:
    raw = os.environ.get("SHERPA_OPENCODE_CONFIG_PATH", "").strip()
    if raw:
        return Path(raw).expanduser()
    return runtime_generated_dir() / "opencode.generated.json"


def _replace_file(tmp_name: str, target: Path) -> None:
    tmp_path = Path(tmp_name)
    try:
        tmp_path.replace(target)
    except OSError as exc:
        if getattr(exc, "errno", None) != errno.EXDEV:
            raise
        shutil.copyfile(tmp_path, target)
        tmp_path.unlink(missing_ok=True)


def _write_json_file(path: Path, payload: dict[str, Any], *, temp_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(temp_dir))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        _replace_file(tmp_name, path)
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
    finally:
        try:
            if Path(tmp_name).exists() and str(Path(tmp_name)) != str(path):
                Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass


def _normalize_provider_name(raw: str) -> str:
    name = (raw or "").strip().lower()
    if not name:
        return ""
    return _PROVIDER_ALIASES.get(name, name)


def list_opencode_provider_models(provider: str) -> tuple[str, list[str]]:
    normalized = _normalize_provider_name(provider)
    if not normalized:
        return "", []
    pdef = KNOWN_PROVIDERS.get(normalized)
    items = list(pdef.models) if pdef else []
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
    for name, pdef in KNOWN_PROVIDERS.items():
        # Match when the known base_url domain appears in the given URL.
        from urllib.parse import urlparse as _up
        known_host = _up(pdef.base_url).hostname or ""
        if known_host and known_host in url:
            return name
    return ""


def _best_provider_api_key(cfg: "WebPersistentConfig", provider: str) -> str:
    normalized = _normalize_provider_name(provider)
    item = _provider_config_by_name(cfg, normalized)
    if item and item.api_key and item.api_key.strip():
        return item.api_key.strip()
    # Legacy fallback for OPENAI_* fields: only when base_url clearly maps to the same provider.
    openai_provider = _provider_from_base_url(cfg.openai_base_url)
    if normalized == openai_provider and cfg.openai_api_key and cfg.openai_api_key.strip():
        return cfg.openai_api_key.strip()
    return ""


def normalize_model_for_opencode(
    model: str | None,
    *,
    cfg: "WebPersistentConfig" | None = None,
    providers: list[OpencodeProviderConfig] | None = None,
) -> str:
    raw = str(model or "").strip()
    if not raw:
        return ""
    if "/" in raw:
        return raw

    normalized_entries = normalize_opencode_providers(
        providers if providers is not None else (cfg.opencode_providers if cfg is not None else _default_opencode_providers())
    )
    provider_models: dict[str, set[str]] = {}
    for item in normalized_entries:
        names: set[str] = set()
        for candidate in item.models:
            value = str(candidate or "").strip()
            if not value:
                continue
            names.add(value)
            if "/" in value:
                names.add(value.split("/", 1)[1])
        provider_models[item.name] = names

    matched: list[str] = []
    for provider, models in provider_models.items():
        if raw in models:
            matched.append(provider)
    if len(matched) == 1:
        return f"{matched[0]}/{raw}"

    # If configured providers are ambiguous/incomplete, fall back to global known
    # provider model catalogs so plain model names still normalize consistently.
    known_matched: list[str] = []
    for provider, pdef in KNOWN_PROVIDERS.items():
        known_names: set[str] = set()
        for candidate in pdef.models:
            value = str(candidate or "").strip()
            if not value:
                continue
            known_names.add(value)
            if "/" in value:
                known_names.add(value.split("/", 1)[1])
        if raw in known_names:
            known_matched.append(provider)
    if len(known_matched) == 1:
        return f"{known_matched[0]}/{raw}"

    if len(provider_models) == 1:
        only = next(iter(provider_models.keys()))
        return f"{only}/{raw}"

    if raw.lower().startswith("glm-"):
        return f"zai/{raw}"
    return raw


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

    pdef = KNOWN_PROVIDERS.get(normalized)
    fallback = list(pdef.models) if pdef else []
    if not fallback:
        return normalized, [], "none", f"unsupported provider: {provider}"

    base_url = (base_url_override or "").strip() or _best_provider_base_url(cfg, normalized)
    api_key = (api_key_override or "").strip() or _best_provider_api_key(cfg, normalized)
    if not base_url:
        return normalized, fallback, "builtin", "provider base_url not configured"
    if normalized in {_MINIMAX_PROVIDER, _DEEPSEEK_PROVIDER} and not api_key:
        return normalized, [], "none", f"unsupported provider credentials: {normalized} api_key not configured"

    try:
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
    # Look up known defaults; unknown providers pass through with user-supplied values.
    pdef = KNOWN_PROVIDERS.get(name)
    default_base_url = pdef.base_url if pdef else ""
    base_url = (entry.base_url or "").strip() or default_base_url

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
    pdef = KNOWN_PROVIDERS.get(entry.name)
    npm_pkg = pdef.npm if pdef else None
    if npm_pkg:
        node["npm"] = npm_pkg

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
            models[model_name] = {"name": model_name}
    if models:
        node["models"] = models

    return node


def _normalize_mcp_server_entry(name: str, entry: Any) -> dict[str, Any] | None:
    server_name = str(name or "").strip()
    if not server_name:
        return None
    if isinstance(entry, str):
        url = entry.strip()
        if not url:
            return None
        return {"type": "remote", "url": url, "enabled": True}
    if not isinstance(entry, dict):
        return None
    node = dict(entry)
    raw_type = str(node.get("type") or "").strip().lower()
    if not raw_type:
        raw_type = "remote" if str(node.get("url") or "").strip() else "local"
    if raw_type not in {"remote", "local"}:
        return None
    node["type"] = raw_type
    if "enabled" not in node:
        node["enabled"] = True
    if raw_type == "remote":
        url = str(node.get("url") or "").strip()
        if not url:
            return None
        node["url"] = url
    if raw_type == "local":
        cmd = node.get("command")
        if not isinstance(cmd, list) or not all(isinstance(x, str) and str(x).strip() for x in cmd):
            return None
    return node


def _opencode_mcp_servers_from_env() -> dict[str, Any]:
    raw = (os.environ.get("SHERPA_OPENCODE_MCP_SERVERS_JSON") or "").strip()
    if not raw:
        single_url = (os.environ.get("SHERPA_OPENCODE_MCP_URL") or "").strip()
        if not single_url:
            return {}
        return {"promefuzz": {"type": "remote", "url": single_url, "enabled": True}}

    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    out: dict[str, Any] = {}
    if isinstance(parsed, dict):
        for name, item in parsed.items():
            normalized = _normalize_mcp_server_entry(str(name), item)
            if normalized:
                out[str(name)] = normalized
    elif isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            payload = dict(item)
            payload.pop("name", None)
            normalized = _normalize_mcp_server_entry(name, payload)
            if normalized:
                out[name] = normalized
    return out


def build_opencode_runtime_config(cfg: WebPersistentConfig) -> dict[str, Any]:
    providers: dict[str, Any] = {}
    for item in normalize_opencode_providers(cfg.opencode_providers):
        if not item.enabled:
            continue
        providers[item.name] = _build_provider_node(item)

    out = {
        "$schema": _OPENCODE_SCHEMA_URL,
        "provider": providers,
    }
    mcp_servers = _opencode_mcp_servers_from_env()
    if mcp_servers:
        out["mcp"] = mcp_servers
    return out


def write_opencode_runtime_config_file(cfg: WebPersistentConfig) -> Path:
    p = opencode_runtime_config_path()
    payload = build_opencode_runtime_config(cfg)
    _write_json_file(p, payload, temp_dir=p.parent)
    return p


def _provider_from_model_or_url(model: str, base_url: str) -> str:
    provider_by_url = _provider_from_base_url(base_url)
    if provider_by_url:
        return provider_by_url
    low = str(model or "").strip().lower()
    for name, pdef in KNOWN_PROVIDERS.items():
        if any(low == m.lower() or name in low for m in pdef.models):
            return name
    return _DEEPSEEK_PROVIDER


def _sanitize_model_literal(raw: str | None) -> str:
    value = str(raw or "").strip()
    if value in {"-", "auto", "AUTO", "none", "None", "null", "NULL"}:
        return ""
    return value


def apply_llm_env_source(cfg: WebPersistentConfig) -> WebPersistentConfig:
    key = (
        os.environ.get("LLM_key", "").strip()
        or os.environ.get("DEEPSEEK_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("MINIMAX_API_KEY", "").strip()
    )
    base_url = (
        os.environ.get("DEEPSEEK_BASE_URL", "").strip()
        or os.environ.get("OPENAI_BASE_URL", "").strip()
        or os.environ.get("MINIMAX_BASE_URL", "").strip()
        or _DEEPSEEK_BASE_URL
    )
    model = (
        _sanitize_model_literal(os.environ.get("OPENCODE_MODEL", ""))
        or _sanitize_model_literal(os.environ.get("OPENAI_MODEL", ""))
        or _sanitize_model_literal(os.environ.get("DEEPSEEK_MODEL", ""))
        or _sanitize_model_literal(os.environ.get("MINIMAX_MODEL", ""))
    )
    if not model:
        provider_by_url = _provider_from_base_url(base_url)
        if provider_by_url and provider_by_url in KNOWN_PROVIDERS:
            model = KNOWN_PROVIDERS[provider_by_url].default_model
        else:
            model = _DEEPSEEK_DEFAULT_MODEL
    provider_name = _provider_from_model_or_url(model, base_url)
    cfg.openai_api_key = key or None
    cfg.openai_base_url = base_url
    cfg.openai_model = model
    cfg.opencode_model = model
    cfg.opencode_providers = [
        OpencodeProviderConfig(
            name=provider_name,
            enabled=True,
            base_url=base_url,
            api_key=(key or None),
            clear_api_key=False,
            models=[model],
            headers={},
            options={},
        )
    ]
    return cfg



def load_config() -> WebPersistentConfig:
    path = config_path()
    if not path.is_file():
        cfg = WebPersistentConfig()
        cfg.fuzz_use_docker = False
        cfg.fuzz_docker_image = ""
        cfg.opencode_providers = normalize_opencode_providers(cfg.opencode_providers)
        default_oss_fuzz_dir = os.environ.get("SHERPA_DEFAULT_OSS_FUZZ_DIR", "").strip()
        if not cfg.oss_fuzz_dir.strip() and default_oss_fuzz_dir:
            cfg.oss_fuzz_dir = default_oss_fuzz_dir
        return apply_llm_env_source(cfg)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            cfg = WebPersistentConfig()
        else:
            cfg = WebPersistentConfig(**raw)

        cfg.fuzz_use_docker = bool(cfg.fuzz_use_docker)
        cfg.fuzz_docker_image = (cfg.fuzz_docker_image or "").strip()
        cfg.opencode_providers = normalize_opencode_providers(cfg.opencode_providers)
        default_oss_fuzz_dir = os.environ.get("SHERPA_DEFAULT_OSS_FUZZ_DIR", "").strip()
        if not cfg.oss_fuzz_dir.strip() and default_oss_fuzz_dir:
            cfg.oss_fuzz_dir = default_oss_fuzz_dir
        return apply_llm_env_source(cfg)
    except Exception:
        cfg = WebPersistentConfig()
        cfg.fuzz_use_docker = False
        cfg.fuzz_docker_image = ""
        cfg.opencode_providers = normalize_opencode_providers(cfg.opencode_providers)
        default_oss_fuzz_dir = os.environ.get("SHERPA_DEFAULT_OSS_FUZZ_DIR", "").strip()
        if not cfg.oss_fuzz_dir.strip() and default_oss_fuzz_dir:
            cfg.oss_fuzz_dir = default_oss_fuzz_dir
        return apply_llm_env_source(cfg)


def save_config(cfg: WebPersistentConfig) -> None:
    path = config_path()
    payload = cfg.model_dump()
    _write_json_file(path, payload, temp_dir=runtime_generated_dir())


def _set_env_if_value(name: str, value: str | None) -> None:
    if value is None:
        return
    if isinstance(value, str) and value.strip() == "":
        os.environ.pop(name, None)
        return
    os.environ[name] = str(value)


def apply_config_to_env(cfg: WebPersistentConfig) -> None:
    apply_llm_env_source(cfg)
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
        _set_env_if_value("LLM_key", cfg.openai_api_key)

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
    _set_env_if_value(
        "SHERPA_RUN_PLATEAU_IDLE_GROWTH_SEC",
        str(int(cfg.sherpa_run_plateau_idle_growth_sec)),
    )

    # Keep the OpenCode key file in sync for fuzz pipeline.
    write_opencode_env_file(cfg)
    cfg.opencode_providers = normalize_opencode_providers(cfg.opencode_providers)
    config_path = write_opencode_runtime_config_file(cfg)
    _set_env_if_value("OPENCODE_CONFIG", str(config_path))


def write_opencode_env_file(cfg: WebPersistentConfig) -> None:
    p = opencode_env_path()
    d = p.parent
    d.mkdir(parents=True, exist_ok=True)

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
        _replace_file(tmp_name, p)
        try:
            os.chmod(p, 0o600)
        except Exception:
            pass
    finally:
        try:
            if Path(tmp_name).exists() and str(Path(tmp_name)) != str(p):
                Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass


def as_public_dict(cfg: WebPersistentConfig) -> dict[str, Any]:
    data = cfg.model_dump()
    # Native runtime baseline for k8s mode: keep Docker fields for backward
    # compatibility, but always expose them as disabled to avoid UI confusion.
    data["fuzz_use_docker"] = False
    data["fuzz_docker_image"] = ""

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
