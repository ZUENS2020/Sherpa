# main.py
from __future__ import annotations
from fastapi import FastAPI, Body, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import hashlib
import math
import os
import re
import shutil
import resource
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
import threading
import time
import queue
import uuid
from collections import deque
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from io import StringIO
from pathlib import Path
from urllib.parse import urlparse
import base64
import yaml
from fuzz_relative_functions import fuzz_logic
from job_store import JobStore, PostgresJobStore
from persistent_config import (
    WebPersistentConfig,
    apply_minimax_env_source,
    apply_config_to_env,
    as_public_dict,
    list_opencode_provider_models_resolved,
    normalize_model_for_opencode,
    opencode_env_path,
    opencode_runtime_config_path,
    load_config,
    save_config,
)

@asynccontextmanager
async def _lifespan(app: FastAPI):
    cfg = load_config()
    _cfg_set(cfg)
    apply_config_to_env(cfg)
    _init_job_store()
    yield


app = FastAPI(title="LangChain Agent API", version="1.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _http_metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = None
    status_code = 500
    try:
        response = await call_next(request)
        status_code = int(getattr(response, "status_code", 500))
        return response
    finally:
        elapsed_ms = max(0.0, (time.perf_counter() - start) * 1000.0)
        now_ts = time.time()
        with _HTTP_METRICS_LOCK:
            _HTTP_REQUEST_EVENTS.append((now_ts, elapsed_ms, status_code))
            cutoff = now_ts - 3600.0
            while _HTTP_REQUEST_EVENTS and _HTTP_REQUEST_EVENTS[0][0] < cutoff:
                _HTTP_REQUEST_EVENTS.popleft()

#创建线程池
_MAX_WORKERS = int(os.environ.get("SHERPA_WEB_MAX_WORKERS", "5"))
executor = ThreadPoolExecutor(max_workers=max(1, _MAX_WORKERS))


_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict] = {}
_APP_START = time.time()
_INIT_LOCK = threading.Lock()
_JOB_STORE: JobStore | None = None
_JOB_FUTURES_LOCK = threading.Lock()
_JOB_FUTURES: dict[str, Future] = {}
_K8S_METRICS_API_UNAVAILABLE_UNTIL = 0.0
_HTTP_METRICS_LOCK = threading.Lock()
_HTTP_REQUEST_EVENTS: deque[tuple[float, float, int]] = deque()

# In-memory API log retention limit (characters).
# 0 or negative means unlimited (no truncation).
_JOB_MEMORY_LOG_MAX_CHARS = int(os.environ.get("SHERPA_WEB_JOB_LOG_MAX_CHARS", "0"))
_JOB_RESTORE_LOG_MAX_CHARS = int(os.environ.get("SHERPA_WEB_RESTORE_LOG_MAX_CHARS", "200000"))

_SENSITIVE_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "MINIMAX_API_KEY",
    "ANTHROPIC_API_KEY",
    "DATABASE_URL",
    "POSTGRES_PASSWORD",
)

_SENSITIVE_KV_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|PASS))\s*=\s*([^\s,;]+)"
)
_AUTH_BEARER_RE = re.compile(r"(?i)\b(Authorization\s*:\s*Bearer\s+)([^\s]+)")

_ACTIVE_JOB_STDOUT_TEE: ContextVar[object | None] = ContextVar("ACTIVE_JOB_STDOUT_TEE", default=None)
_ACTIVE_JOB_STDERR_TEE: ContextVar[object | None] = ContextVar("ACTIVE_JOB_STDERR_TEE", default=None)


def _redact_sensitive_text(text: str) -> str:
    if not text:
        return text
    out = text
    for key in _SENSITIVE_ENV_KEYS:
        val = (os.environ.get(key) or "").strip()
        if val:
            out = out.replace(val, "***")
    out = _SENSITIVE_KV_RE.sub(lambda m: f"{m.group(1)}=***", out)
    out = _AUTH_BEARER_RE.sub(lambda m: f"{m.group(1)}***", out)
    return out


_REPO_ROOT = Path(__file__).resolve().parents[3]
_JOB_LOGS_DIR = Path(
    os.environ.get("SHERPA_WEB_JOB_LOG_DIR", "/app/job-logs/jobs")
).expanduser().resolve()


_CFG_LOCK = threading.Lock()
_CFG: WebPersistentConfig = WebPersistentConfig()


def _cfg_get() -> WebPersistentConfig:
    with _CFG_LOCK:
        return _CFG


def _cfg_set(cfg: WebPersistentConfig) -> None:
    global _CFG
    with _CFG_LOCK:
        _CFG = cfg


def _normalized_opencode_model_value(raw_model: object) -> str:
    value = str(raw_model or "").strip()
    if not value:
        return ""
    return normalize_model_for_opencode(value, cfg=_cfg_get())


def _track_job_future(job_id: str, future: Future) -> None:
    with _JOB_FUTURES_LOCK:
        _JOB_FUTURES[job_id] = future

    def _cleanup(_: Future) -> None:
        with _JOB_FUTURES_LOCK:
            if _JOB_FUTURES.get(job_id) is future:
                _JOB_FUTURES.pop(job_id, None)

    add_cb = getattr(future, "add_done_callback", None)
    if callable(add_cb):
        add_cb(_cleanup)
        return

    done_fn = getattr(future, "done", None)
    if callable(done_fn):
        try:
            if bool(done_fn()):
                _cleanup(future)
        except Exception:
            pass


def _cancel_job_future(job_id: str) -> bool:
    with _JOB_FUTURES_LOCK:
        fut = _JOB_FUTURES.get(job_id)
    if fut is None:
        return False
    cancel_fn = getattr(fut, "cancel", None)
    if not callable(cancel_fn):
        return False
    try:
        return bool(cancel_fn())
    except Exception:
        return False


def _read_text_if_exists(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _read_int_if_exists(path: str) -> int | None:
    raw = _read_text_if_exists(path)
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _cgroup_memory_status() -> dict[str, object]:
    current = _read_int_if_exists("/sys/fs/cgroup/memory.current")
    limit_raw = _read_text_if_exists("/sys/fs/cgroup/memory.max")
    oom_kill_count = None
    events_raw = _read_text_if_exists("/sys/fs/cgroup/memory.events")

    if current is None:
        current = _read_int_if_exists("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if not limit_raw:
        limit_raw = _read_text_if_exists("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if events_raw:
        for line in events_raw.splitlines():
            key, _, value = line.partition(" ")
            if key.strip() == "oom_kill":
                try:
                    oom_kill_count = int(value.strip())
                except Exception:
                    pass
                break

    limit_bytes = None
    if limit_raw and limit_raw != "max":
        try:
            parsed_limit = int(limit_raw)
            if parsed_limit < (1 << 60):
                limit_bytes = parsed_limit
        except Exception:
            pass

    usage_ratio = None
    if current is not None and limit_bytes and limit_bytes > 0:
        usage_ratio = current / limit_bytes

    return {
        "current_bytes": current,
        "limit_bytes": limit_bytes,
        "usage_ratio": usage_ratio,
        "oom_kill_count": oom_kill_count,
    }


def _process_rss_bytes() -> int | None:
    try:
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if raw > 0:
            return int(raw) * 1024
    except Exception:
        pass

    proc_status = _read_text_if_exists("/proc/self/status")
    for line in proc_status.splitlines():
        if not line.startswith("VmRSS:"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                return int(parts[1]) * 1024
            except Exception:
                return None
    return None


def _memory_status() -> dict[str, object]:
    cgroup = _cgroup_memory_status()
    usage_ratio = cgroup.get("usage_ratio")
    pressure = "unknown"
    if isinstance(usage_ratio, (int, float)):
        if usage_ratio >= 0.95:
            pressure = "critical"
        elif usage_ratio >= 0.85:
            pressure = "high"
        elif usage_ratio >= 0.7:
            pressure = "elevated"
        else:
            pressure = "normal"

    return {
        "process_rss_bytes": _process_rss_bytes(),
        "cgroup_current_bytes": cgroup.get("current_bytes"),
        "cgroup_limit_bytes": cgroup.get("limit_bytes"),
        "cgroup_usage_ratio": usage_ratio,
        "oom_kill_count": cgroup.get("oom_kill_count"),
        "pressure": pressure,
    }


def _k8s_worker_resources() -> dict[str, dict[str, str]] | None:
    request_cpu = (os.environ.get("SHERPA_K8S_JOB_CPU_REQUEST", "500m") or "").strip()
    limit_cpu = (os.environ.get("SHERPA_K8S_JOB_CPU_LIMIT", "2") or "").strip()
    request_memory = (os.environ.get("SHERPA_K8S_JOB_MEMORY_REQUEST", "4Gi") or "").strip()
    limit_memory = (os.environ.get("SHERPA_K8S_JOB_MEMORY_LIMIT", "64Gi") or "").strip()

    requests: dict[str, str] = {}
    limits: dict[str, str] = {}
    if request_cpu:
        requests["cpu"] = request_cpu
    if request_memory:
        requests["memory"] = request_memory
    if limit_cpu:
        limits["cpu"] = limit_cpu
    if limit_memory:
        limits["memory"] = limit_memory
    if not requests and not limits:
        return None

    resources: dict[str, dict[str, str]] = {}
    if requests:
        resources["requests"] = requests
    if limits:
        resources["limits"] = limits
    return resources


def _k8s_worker_memory_limit() -> str:
    return (os.environ.get("SHERPA_K8S_JOB_MEMORY_LIMIT", "64Gi") or "").strip()


def _docker_cli(args: list[str], *, timeout: int = 20) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["docker", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return int(proc.returncode), proc.stdout or "", proc.stderr or ""
    except Exception as e:
        return 1, "", str(e)


def _executor_mode() -> str:
    mode = (os.environ.get("SHERPA_EXECUTOR_MODE", "k8s_job") or "").strip().lower()
    if mode and mode != "k8s_job":
        raise RuntimeError(
            f"unsupported executor mode: {mode}. Only 'k8s_job' is supported."
        )
    return "k8s_job"


def _k8s_enabled() -> bool:
    return True


def _k8s_namespace() -> str:
    return (os.environ.get("SHERPA_K8S_NAMESPACE", "sherpa") or "").strip() or "sherpa"


def _k8s_worker_image() -> str:
    return (os.environ.get("SHERPA_K8S_WORKER_IMAGE", "sherpa-web:latest") or "").strip()


def _k8s_kubectl_bin() -> str:
    return (os.environ.get("SHERPA_KUBECTL_BIN", "kubectl") or "").strip() or "kubectl"


def _k8s_keep_finished_jobs() -> bool:
    raw = (os.environ.get("SHERPA_K8S_KEEP_FINISHED_JOBS", "0") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _k8s_job_ttl_seconds() -> int:
    try:
        return max(0, int(os.environ.get("SHERPA_K8S_JOB_TTL_SECONDS", "3600")))
    except Exception:
        return 3600


def _kubectl(args: list[str], *, input_text: str | None = None, timeout: int = 30) -> tuple[int, str, str]:
    cmd = [_k8s_kubectl_bin(), "-n", _k8s_namespace(), *args]
    try:
        proc = subprocess.run(
            cmd,
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return int(proc.returncode), proc.stdout or "", proc.stderr or ""
    except Exception as e:
        return 1, "", str(e)


def _k8s_job_name(job_id: str, *, resumed: bool, stage: str | None = None, seq: int | None = None) -> str:
    suffix = "resume" if resumed else "new"
    stage_part = (stage or "").strip().lower()
    if stage_part:
        stage_part = re.sub(r"[^a-z0-9-]+", "-", stage_part)[:16]
    idx_part = f"-{int(seq)}" if seq is not None else ""
    if stage_part:
        return f"sherpa-fuzz-{job_id[:10]}-{suffix}-{stage_part}{idx_part}"
    return f"sherpa-fuzz-{job_id[:16]}-{suffix}"


def _k8s_result_paths(job_id: str, *, stage: str | None = None, seq: int | None = None) -> tuple[Path, Path]:
    base = Path(os.environ.get("SHERPA_OUTPUT_DIR", "/shared/output")).expanduser().resolve()
    root = base / "_k8s_jobs" / job_id
    stage_part = (stage or "").strip().lower()
    if stage_part:
        stage_part = re.sub(r"[^a-z0-9-]+", "-", stage_part)[:16]
        prefix = f"stage-{int(seq or 0):02d}-{stage_part}"
    else:
        prefix = "result"
    return (root / f"{prefix}.json", root / f"{prefix}.error.txt")


def _k8s_analysis_companion_enabled() -> bool:
    raw = (os.environ.get("SHERPA_K8S_ANALYSIS_COMPANION_ENABLED", "1") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _k8s_analysis_companion_name(job_id: str) -> str:
    return f"sherpa-promefuzz-{job_id[:10]}"


def _k8s_analysis_companion_service_name(job_id: str) -> str:
    return _k8s_analysis_companion_name(job_id)


def _k8s_analysis_companion_image() -> str:
    return (os.environ.get("SHERPA_K8S_ANALYSIS_COMPANION_IMAGE") or _k8s_worker_image()).strip()


def _k8s_analysis_companion_port() -> int:
    raw = (os.environ.get("SHERPA_K8S_ANALYSIS_COMPANION_PORT") or "18080").strip()
    try:
        port = int(raw)
    except Exception:
        port = 18080
    return max(1, min(port, 65535))


def _k8s_analysis_companion_mcp_path() -> str:
    raw = (os.environ.get("SHERPA_K8S_ANALYSIS_COMPANION_MCP_PATH") or "/mcp").strip()
    if not raw:
        return "/mcp"
    if not raw.startswith("/"):
        return f"/{raw}"
    return raw


def _k8s_openrouter_embedding_secret_name() -> str:
    return (os.environ.get("SHERPA_K8S_OPENROUTER_EMBEDDING_SECRET_NAME", "sherpa-openrouter-embedding") or "").strip()


def _k8s_analysis_companion_url(job_id: str) -> str:
    svc = _k8s_analysis_companion_service_name(job_id)
    ns = _k8s_namespace()
    port = _k8s_analysis_companion_port()
    path = _k8s_analysis_companion_mcp_path()
    return f"http://{svc}.{ns}.svc.cluster.local:{port}{path}"


def _k8s_analysis_companion_manifest(*, pod_name: str, job_id: str) -> str:
    pvc_output = (os.environ.get("SHERPA_K8S_PVC_OUTPUT", "sherpa-shared-output") or "").strip()
    image = _k8s_analysis_companion_image()
    mcp_port = _k8s_analysis_companion_port()
    mcp_path = _k8s_analysis_companion_mcp_path()
    mcp_url = _k8s_analysis_companion_url(job_id)
    command = (
        os.environ.get("SHERPA_K8S_ANALYSIS_COMPANION_COMMAND")
        or (
            "set -eu\n"
            "export PYTHONUNBUFFERED=1\n"
            "/usr/local/bin/python /app/harness_generator/src/langchain_agent/promefuzz_companion.py &\n"
            "COMPANION_PID=$!\n"
            "exec /usr/local/bin/python -m promefuzz_mcp.server start "
            "--skip-build --transport streamable-http --host 0.0.0.0 "
            f"--port {mcp_port} --mcp-path {mcp_path}\n"
        )
    ).strip()
    companion_env_names = [
        "SHERPA_PROMEFUZZ_MCP_ROOT",
        "SHERPA_PROMEFUZZ_BUILD_BINARIES",
        "SHERPA_PROMEFUZZ_MAX_SOURCE_FILES",
        "SHERPA_PROMEFUZZ_POLL_SEC",
        "SHERPA_PROMEFUZZ_REFRESH_SEC",
        "SHERPA_PROMEFUZZ_RUN_ONCE",
        "SHERPA_PROMEFUZZ_REPO_ROOT_HINT",
    ]
    companion_env: list[dict[str, str]] = [
        {"name": "SHERPA_JOB_ID", "value": job_id},
        {"name": "SHERPA_OUTPUT_DIR", "value": (os.environ.get("SHERPA_OUTPUT_DIR") or "/shared/output").strip()},
        {"name": "SHERPA_PROMEFUZZ_MCP_PORT", "value": str(mcp_port)},
        {"name": "SHERPA_PROMEFUZZ_MCP_PATH", "value": mcp_path},
        {"name": "SHERPA_PROMEFUZZ_MCP_URL", "value": mcp_url},
        {"name": "PYTHONPATH", "value": "/app/promefuzz-mcp"},
    ]
    companion_env_from: list[dict[str, object]] = []
    embedding_secret = _k8s_openrouter_embedding_secret_name()
    if embedding_secret:
        companion_env_from.append({"secretRef": {"name": embedding_secret, "optional": True}})
    for name in companion_env_names:
        value = str(os.environ.get(name) or "").strip()
        if value:
            companion_env.append({"name": name, "value": value})
    if not any(str(item.get("name") or "") == "SHERPA_PROMEFUZZ_RUN_ONCE" for item in companion_env):
        companion_env.append({"name": "SHERPA_PROMEFUZZ_RUN_ONCE", "value": "0"})
    manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "labels": {
                "app.kubernetes.io/name": "sherpa",
                "sherpa/job-id": job_id,
                "sherpa/job-kind": "analysis-companion",
            },
        },
        "spec": {
            "restartPolicy": "Never",
            "serviceAccountName": (os.environ.get("SHERPA_K8S_JOB_SERVICE_ACCOUNT", "sherpa-web") or "sherpa-web"),
            "containers": [
                {
                    "name": "analysis-companion",
                    "image": image,
                    "imagePullPolicy": (os.environ.get("SHERPA_K8S_WORKER_IMAGE_PULL_POLICY", "IfNotPresent") or "IfNotPresent"),
                    "command": ["sh", "-lc", command],
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "runAsNonRoot": True,
                        "runAsUser": 10001,
                        "runAsGroup": 10001,
                        "capabilities": {"drop": ["ALL"]},
                    },
                    "env": companion_env,
                    "envFrom": companion_env_from,
                    "ports": [{"containerPort": mcp_port, "name": "mcp"}],
                    "volumeMounts": [
                        {"name": "shared-output", "mountPath": "/shared/output"},
                    ],
                }
            ],
            "volumes": [
                {"name": "shared-output", "persistentVolumeClaim": {"claimName": pvc_output}},
            ],
        },
    }
    return yaml.safe_dump(manifest, sort_keys=False)


def _k8s_analysis_companion_service_manifest(*, service_name: str, job_id: str) -> str:
    mcp_port = _k8s_analysis_companion_port()
    manifest = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": service_name,
            "labels": {
                "app.kubernetes.io/name": "sherpa",
                "sherpa/job-kind": "analysis-companion",
            },
        },
        "spec": {
            "selector": {
                "app.kubernetes.io/name": "sherpa",
                "sherpa/job-kind": "analysis-companion",
                "sherpa/job-id": job_id,
            },
            "ports": [
                {
                    "name": "mcp",
                    "port": mcp_port,
                    "targetPort": mcp_port,
                }
            ],
        },
    }
    return yaml.safe_dump(manifest, sort_keys=False)


def _k8s_start_analysis_companion(job_id: str) -> tuple[str, str, str]:
    if not _k8s_analysis_companion_enabled():
        return "", "", ""
    pod_name = _k8s_analysis_companion_name(job_id)
    service_name = _k8s_analysis_companion_service_name(job_id)
    mcp_url = _k8s_analysis_companion_url(job_id)
    pod_ok = False
    svc_ok = False
    pod_rc, pod_out, _ = _kubectl(["get", "pod", pod_name, "-o", "json"], timeout=15)
    if pod_rc == 0:
        try:
            pod_doc = json.loads(pod_out)
            phase = str(((pod_doc.get("status") or {}) if isinstance(pod_doc, dict) else {}).get("phase") or "").strip().lower()
            pod_ok = phase == "running"
        except Exception:
            pod_ok = False
    svc_rc, _, _ = _kubectl(["get", "service", service_name], timeout=15)
    svc_ok = svc_rc == 0
    if pod_ok and svc_ok:
        return pod_name, service_name, mcp_url
    _kubectl(["delete", "pod", pod_name, "--ignore-not-found=true"], timeout=20)
    _kubectl(["delete", "service", service_name, "--ignore-not-found=true"], timeout=20)
    service_manifest = _k8s_analysis_companion_service_manifest(service_name=service_name, job_id=job_id)
    svc_rc, svc_out, svc_err = _kubectl(["apply", "-f", "-"], input_text=service_manifest, timeout=30)
    if svc_rc != 0:
        raise RuntimeError(
            f"failed to create analysis companion service {service_name}: {(svc_out + svc_err).strip()}"
        )
    manifest = _k8s_analysis_companion_manifest(pod_name=pod_name, job_id=job_id)
    rc, out, err = _kubectl(["apply", "-f", "-"], input_text=manifest, timeout=30)
    if rc != 0:
        raise RuntimeError(f"failed to start analysis companion {pod_name}: {(out + err).strip()}")
    return pod_name, service_name, mcp_url


def _k8s_stop_analysis_companion(pod_name: str, service_name: str = "") -> None:
    if not pod_name and not service_name:
        return
    if pod_name:
        _kubectl(["delete", "pod", pod_name, "--ignore-not-found=true"], timeout=20)
    if service_name:
        _kubectl(["delete", "service", service_name, "--ignore-not-found=true"], timeout=20)


def _analysis_companion_status_for_job(job_id: str) -> dict[str, object]:
    jid = str(job_id or "").strip()
    if not jid:
        return {}
    base = Path(os.environ.get("SHERPA_OUTPUT_DIR", "/shared/output")).expanduser()
    status_path = base / "_k8s_jobs" / jid / "promefuzz" / "status.json"
    if not status_path.is_file():
        return {}
    try:
        raw = status_path.read_text(encoding="utf-8", errors="replace")
        parsed = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, object] = {}
    for key in (
        "state",
        "analysis_backend",
        "candidate_count",
        "updated_at",
        "repo_root",
        "error",
        "last_error",
        "preprocess_path",
        "coverage_hints_path",
        "rag_ok",
        "rag_knowledge_base_path",
        "rag_document_count",
        "rag_chunk_count",
        "embedding_provider",
        "embedding_model",
        "embedding_ok",
        "rag_degraded",
        "rag_degraded_reason",
        "semantic_query_count",
        "semantic_hit_count",
        "semantic_hit_rate",
        "cache_hit_rate",
        "mcp_url",
        "mcp_ready",
    ):
        if key in parsed:
            out[key] = parsed.get(key)
    if "mcp_url" not in out:
        out["mcp_url"] = _k8s_analysis_companion_url(jid)
    if "mcp_ready" not in out:
        out["mcp_ready"] = False
    if "last_error" not in out:
        out["last_error"] = out.get("error")
    return out


def _analysis_context_path_for_repo(repo_root: str | None) -> Path | None:
    raw = str(repo_root or "").strip()
    if not raw:
        return None
    try:
        root = Path(raw).expanduser()
    except Exception:
        return None
    return root / "fuzz" / "analysis_context.json"


def _has_reusable_analysis_context(repo_root: str | None) -> bool:
    analysis_path = _analysis_context_path_for_repo(repo_root)
    return bool(analysis_path and analysis_path.is_file())


def _k8s_analysis_companion_timeout_sec() -> int:
    raw = (os.environ.get("SHERPA_K8S_ANALYSIS_COMPANION_TIMEOUT_SEC") or "180").strip()
    try:
        return max(10, min(int(raw), 3600))
    except Exception:
        return 180


def _k8s_analysis_require_rag_ready() -> bool:
    raw = (os.environ.get("SHERPA_K8S_ANALYSIS_REQUIRE_RAG_READY", "1") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _k8s_analysis_rag_wait_timeout_sec() -> int:
    raw = (os.environ.get("SHERPA_K8S_ANALYSIS_RAG_WAIT_TIMEOUT_SEC", "120") or "").strip()
    try:
        return max(10, min(int(raw), 3600))
    except Exception:
        return 120


def _analysis_companion_is_ready(status_doc: dict[str, object], *, require_rag: bool) -> bool:
    if not isinstance(status_doc, dict) or not status_doc:
        return False
    state = str(status_doc.get("state") or "").strip().lower()
    if state in {"failed", "pod_failed", "pod_succeeded"}:
        return False
    mcp_ready = bool(status_doc.get("mcp_ready"))
    if not mcp_ready:
        return False
    if not require_rag:
        return True
    return bool(status_doc.get("rag_ok"))


def _k8s_wait_analysis_companion_result(
    job_id: str,
    pod_name: str,
    *,
    timeout_sec: int,
    require_rag: bool = False,
) -> dict[str, object]:
    start = time.time()
    poll_sec = max(1, min(10, int((os.environ.get("SHERPA_K8S_ANALYSIS_COMPANION_POLL_SEC") or "2").strip() or "2")))
    latest_status: dict[str, object] = {}
    while (time.time() - start) < timeout_sec:
        status_doc = _analysis_companion_status_for_job(job_id)
        if status_doc:
            latest_status = dict(status_doc)
            if _analysis_companion_is_ready(status_doc, require_rag=require_rag):
                return latest_status
            state = str(status_doc.get("state") or "").strip().lower()
            if state in {"degraded", "failed"}:
                return latest_status
        rc, out, _ = _kubectl(["get", "pod", pod_name, "-o", "json"], timeout=10)
        if rc == 0:
            try:
                doc = json.loads(out)
            except Exception:
                doc = {}
            phase = str(((doc.get("status") or {}) if isinstance(doc, dict) else {}).get("phase") or "").strip().lower()
            if phase in {"succeeded", "failed"}:
                status_doc = _analysis_companion_status_for_job(job_id)
                if status_doc:
                    return dict(status_doc)
                if not latest_status:
                    latest_status = {"state": f"pod_{phase}", "pod_phase": phase}
                return latest_status
        time.sleep(poll_sec)
    wait_mode = "rag_ready" if require_rag else "mcp_ready"
    raise TimeoutError(f"analysis companion timeout waiting for {wait_mode} after {timeout_sec}s")


def _k8s_proxy_env_from_items() -> list[dict[str, object]]:
    secret_name = (os.environ.get("SHERPA_K8S_PROXY_SECRET_NAME", "sherpa-runtime-proxy") or "").strip()
    if not secret_name:
        return []
    return [{"secretRef": {"name": secret_name, "optional": True}}]


def _k8s_git_env_items() -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for name in ("SHERPA_GIT_MIRRORS", "SHERPA_GITHUB_MIRROR"):
        value = (os.environ.get(name) or "").strip()
        if value:
            items.append({"name": name, "value": value})
    return items


def _k8s_build_manifest(job_name: str, payload: dict[str, object]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    payload_b64 = base64.b64encode(payload_json.encode("utf-8")).decode("ascii")
    raw_model = str(payload.get("model") or "").strip()
    normalized_model = _normalized_opencode_model_value(raw_model)
    ttl = _k8s_job_ttl_seconds()
    keep_finished = _k8s_keep_finished_jobs()

    config_name = (os.environ.get("SHERPA_K8S_CONFIGMAP_NAME", "sherpa-config") or "").strip()
    minimax_secret = (os.environ.get("SHERPA_K8S_MINIMAX_SECRET_NAME", "sherpa-minimax") or "").strip()
    pg_secret = (os.environ.get("SHERPA_K8S_POSTGRES_SECRET_NAME", "sherpa-postgres") or "").strip()

    pvc_tmp = (os.environ.get("SHERPA_K8S_PVC_TMP", "sherpa-shared-tmp") or "").strip()
    pvc_output = (os.environ.get("SHERPA_K8S_PVC_OUTPUT", "sherpa-shared-output") or "").strip()
    pvc_oss = (os.environ.get("SHERPA_K8S_PVC_OSS_FUZZ", "sherpa-oss-fuzz") or "").strip()
    pvc_cfg = (os.environ.get("SHERPA_K8S_PVC_CONFIG", "sherpa-config") or "").strip()
    pvc_logs = (os.environ.get("SHERPA_K8S_PVC_JOB_LOGS", "sherpa-job-logs") or "").strip()
    target_node_name = str(payload.get("target_node_name") or "").strip()
    worker_resources = _k8s_worker_resources()

    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "labels": {
                "app.kubernetes.io/name": "sherpa",
                "sherpa/job-id": str(payload.get("job_id") or ""),
                "sherpa/job-kind": "fuzz",
                "sherpa/stage": str(payload.get("stop_after_step") or payload.get("resume_from_step") or "full"),
            },
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": ttl,
            "template": {
                "metadata": {
                    "labels": {
                        "app.kubernetes.io/name": "sherpa",
                        "sherpa/job-id": str(payload.get("job_id") or ""),
                    },
                },
                "spec": {
                    "restartPolicy": "Never",
                    "serviceAccountName": (os.environ.get("SHERPA_K8S_JOB_SERVICE_ACCOUNT", "sherpa-web") or "sherpa-web"),
                    "securityContext": {
                        "seccompProfile": {"type": "RuntimeDefault"},
                        "fsGroup": 10001,
                        "fsGroupChangePolicy": "OnRootMismatch",
                    },
                    "initContainers": [
                        {
                            "name": "runtime-permissions",
                            "image": _k8s_worker_image(),
                            "imagePullPolicy": (os.environ.get("SHERPA_K8S_WORKER_IMAGE_PULL_POLICY", "IfNotPresent") or "IfNotPresent"),
                            "command": [
                                "sh",
                                "-lc",
                                (
                                    "set -eu\n"
                                    "for d in /app/config /app/job-logs /shared/tmp /shared/output /shared/oss-fuzz; do\n"
                                    '  mkdir -p "$d"\n'
                                    "  chown 10001:10001 \"$d\" || true\n"
                                    "  chmod 0777 \"$d\" || true\n"
                                    "  find \"$d\" -mindepth 1 -exec chown 10001:10001 {} + || true\n"
                                    "  find \"$d\" -mindepth 1 -exec chmod a+rwX {} + || true\n"
                                    "done\n"
                                ),
                            ],
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "runAsUser": 0,
                                "runAsGroup": 0,
                                "capabilities": {"drop": ["ALL"]},
                            },
                            "volumeMounts": [
                                {"name": "shared-tmp", "mountPath": "/shared/tmp"},
                                {"name": "shared-output", "mountPath": "/shared/output"},
                                {"name": "oss-fuzz", "mountPath": "/shared/oss-fuzz"},
                                {"name": "config", "mountPath": "/app/config"},
                                {"name": "job-logs", "mountPath": "/app/job-logs"},
                            ],
                        }
                    ],
                    "containers": [
                        {
                            "name": "worker",
                            "image": _k8s_worker_image(),
                            "imagePullPolicy": (os.environ.get("SHERPA_K8S_WORKER_IMAGE_PULL_POLICY", "IfNotPresent") or "IfNotPresent"),
                            "command": ["python", "/app/harness_generator/src/langchain_agent/k8s_job_worker.py"],
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "runAsNonRoot": True,
                                "runAsUser": 10001,
                                "runAsGroup": 10001,
                                "capabilities": {"drop": ["ALL"]},
                            },
                            "env": [
                                {"name": "SHERPA_K8S_WORKER_PAYLOAD_B64", "value": payload_b64},
                                {"name": "OPENCODE_MODEL", "value": normalized_model},
                                {"name": "OPENAI_MODEL", "value": raw_model},
                                {"name": "OPENCODE_CONFIG", "value": str(opencode_runtime_config_path())},
                                {"name": "SHERPA_K8S_JOB_MEMORY_LIMIT", "value": _k8s_worker_memory_limit()},
                                *_k8s_git_env_items(),
                            ],
                            "envFrom": [*_k8s_proxy_env_from_items()],
                            "volumeMounts": [
                                {"name": "shared-tmp", "mountPath": "/shared/tmp"},
                                {"name": "shared-output", "mountPath": "/shared/output"},
                                {"name": "oss-fuzz", "mountPath": "/shared/oss-fuzz"},
                                {"name": "config", "mountPath": "/app/config"},
                                {"name": "job-logs", "mountPath": "/app/job-logs"},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "shared-tmp", "persistentVolumeClaim": {"claimName": pvc_tmp}},
                        {"name": "shared-output", "persistentVolumeClaim": {"claimName": pvc_output}},
                        {"name": "oss-fuzz", "persistentVolumeClaim": {"claimName": pvc_oss}},
                        {"name": "config", "persistentVolumeClaim": {"claimName": pvc_cfg}},
                        {"name": "job-logs", "persistentVolumeClaim": {"claimName": pvc_logs}},
                    ],
                },
            },
        },
    }

    env_from = [*_k8s_proxy_env_from_items()]
    if config_name:
        env_from.append({"configMapRef": {"name": config_name}})
    if minimax_secret:
        env_from.append({"secretRef": {"name": minimax_secret}})
    if pg_secret:
        env_from.append({"secretRef": {"name": pg_secret}})
    manifest["spec"]["template"]["spec"]["containers"][0]["envFrom"] = env_from
    if worker_resources:
        manifest["spec"]["template"]["spec"]["containers"][0]["resources"] = worker_resources

    if target_node_name:
        manifest["spec"]["template"]["spec"]["nodeName"] = target_node_name

    if keep_finished:
        manifest["spec"]["ttlSecondsAfterFinished"] = None

    return yaml.safe_dump(manifest, sort_keys=False)


def _k8s_delete_job(job_name: str) -> None:
    if not job_name:
        return
    _kubectl(["delete", "job", job_name, "--ignore-not-found=true"], timeout=20)


def _k8s_collect_job_logs(job_name: str, *, tail: int = 200) -> str:
    if not job_name:
        return ""
    rc, out, err = _kubectl(["logs", f"job/{job_name}", f"--tail={tail}"], timeout=30)
    if rc == 0:
        return out
    return (out + "\n" + err).strip()


def _k8s_get_job_pod_details(job_name: str) -> dict[str, object]:
    if not job_name:
        return {}
    rc, out, _ = _kubectl(["get", "pod", "-l", f"job-name={job_name}", "-o", "json"], timeout=15)
    if rc != 0:
        return {}
    try:
        doc = json.loads(out)
    except Exception:
        return {}
    items = doc.get("items") if isinstance(doc, dict) else None
    if not isinstance(items, list) or not items or not isinstance(items[0], dict):
        return {}
    pod = items[0]
    meta = pod.get("metadata") if isinstance(pod.get("metadata"), dict) else {}
    status = pod.get("status") if isinstance(pod.get("status"), dict) else {}
    details: dict[str, object] = {
        "pod_name": str(meta.get("name") or "").strip(),
        "phase": str(status.get("phase") or "").strip(),
    }
    for key in ("reason", "message"):
        val = str(status.get(key) or "").strip()
        if val:
            details[key] = val
    cstats = status.get("containerStatuses")
    if isinstance(cstats, list):
        for cs in cstats:
            if not isinstance(cs, dict):
                continue
            details["container_name"] = str(cs.get("name") or "").strip()
            if cs.get("ready") is not None:
                details["container_ready"] = bool(cs.get("ready"))
            if cs.get("restartCount") is not None:
                details["restart_count"] = int(cs.get("restartCount") or 0)
            st = cs.get("state") if isinstance(cs.get("state"), dict) else {}
            waiting = st.get("waiting") if isinstance(st.get("waiting"), dict) else {}
            terminated = st.get("terminated") if isinstance(st.get("terminated"), dict) else {}
            if waiting:
                reason = str(waiting.get("reason") or "").strip()
                message = str(waiting.get("message") or "").strip()
                if reason:
                    details["container_reason"] = reason
                if message:
                    details["container_message"] = message
            if terminated:
                reason = str(terminated.get("reason") or "").strip()
                message = str(terminated.get("message") or "").strip()
                if reason:
                    details["terminated_reason"] = reason
                if message:
                    details["terminated_message"] = message
                if terminated.get("exitCode") is not None:
                    details["exit_code"] = int(terminated.get("exitCode") or 0)
                if terminated.get("signal") is not None:
                    details["signal"] = int(terminated.get("signal") or 0)
                if terminated.get("startedAt") is not None:
                    details["terminated_started_at"] = str(terminated.get("startedAt") or "")
                if terminated.get("finishedAt") is not None:
                    details["terminated_finished_at"] = str(terminated.get("finishedAt") or "")
            last_state = cs.get("lastState") if isinstance(cs.get("lastState"), dict) else {}
            last_terminated = last_state.get("terminated") if isinstance(last_state.get("terminated"), dict) else {}
            if last_terminated:
                last_reason = str(last_terminated.get("reason") or "").strip()
                last_message = str(last_terminated.get("message") or "").strip()
                if last_reason and "last_terminated_reason" not in details:
                    details["last_terminated_reason"] = last_reason
                if last_message and "last_terminated_message" not in details:
                    details["last_terminated_message"] = last_message
                if last_terminated.get("exitCode") is not None and "last_exit_code" not in details:
                    details["last_exit_code"] = int(last_terminated.get("exitCode") or 0)
            break
    return details


def _k8s_summarize_logs(logs: str, *, max_chars: int = 600) -> str:
    txt = str(logs or "").strip()
    if not txt:
        return ""
    if len(txt) <= max_chars:
        return txt
    return txt[-max_chars:]


def _classify_k8s_stage_failure(stage: str, pod_details: dict[str, object], logs: str, err_txt: str) -> dict[str, object]:
    stage_name = str(stage or "").strip().lower()
    detail_texts = [
        str(pod_details.get("reason") or "").strip(),
        str(pod_details.get("message") or "").strip(),
        str(pod_details.get("container_reason") or "").strip(),
        str(pod_details.get("container_message") or "").strip(),
        str(pod_details.get("terminated_reason") or "").strip(),
        str(pod_details.get("terminated_message") or "").strip(),
        str(pod_details.get("last_terminated_reason") or "").strip(),
        str(pod_details.get("last_terminated_message") or "").strip(),
        str(err_txt or "").strip(),
        _k8s_summarize_logs(logs),
    ]
    joined = "\n".join(part for part in detail_texts if part).lower()
    exit_code = pod_details.get("exit_code")
    if exit_code is None:
        exit_code = pod_details.get("last_exit_code")
    try:
        exit_code_int = int(exit_code) if exit_code is not None else None
    except Exception:
        exit_code_int = None

    error_kind = "unknown"
    error_code = "k8s_job_failed"
    run_error_kind = ""
    run_terminal_reason = ""

    if any(tok in joined for tok in ("oomkilled", "oom killed", "out of memory", "memory cgroup out of memory")) or exit_code_int == 137:
        error_kind = "resource"
        error_code = "oom_killed"
        if stage_name == "run":
            run_error_kind = "run_resource_exhaustion"
            run_terminal_reason = "run_resource_exhaustion"
    elif any(tok in joined for tok in ("deadlineexceeded", "deadline exceeded", "context deadline exceeded", "timed out", "timeout")) or exit_code_int == 143:
        error_kind = "timeout"
        error_code = "timeout"
        if stage_name == "run":
            run_error_kind = "run_timeout"
            run_terminal_reason = "run_timeout"
    elif stage_name == "run":
        error_kind = "runtime"
        error_code = "run_failed"
        run_error_kind = "run_exception"
        run_terminal_reason = "run_exception"

    result: dict[str, object] = {
        "stage": stage_name,
        "error_kind": error_kind,
        "error_code": error_code,
        "k8s_failure": {
            "pod_name": str(pod_details.get("pod_name") or "").strip(),
            "phase": str(pod_details.get("phase") or "").strip(),
            "reason": str(pod_details.get("reason") or "").strip(),
            "message": str(pod_details.get("message") or "").strip(),
            "container_reason": str(pod_details.get("container_reason") or "").strip(),
            "terminated_reason": str(pod_details.get("terminated_reason") or "").strip(),
            "exit_code": exit_code_int,
            "logs_tail": _k8s_summarize_logs(logs),
        },
    }
    if run_error_kind:
        result["run_error_kind"] = run_error_kind
    if run_terminal_reason:
        result["run_terminal_reason"] = run_terminal_reason
    return result


class _K8sJobFailure(RuntimeError):
    def __init__(self, message: str, *, result: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.result = dict(result or {})


def _k8s_get_job_pod_phase(job_name: str) -> tuple[str, str]:
    if not job_name:
        return "Unknown", ""
    rc, out, _ = _kubectl(["get", "pod", "-l", f"job-name={job_name}", "-o", "json"], timeout=15)
    if rc != 0:
        return "Unknown", ""
    try:
        doc = json.loads(out)
    except Exception:
        return "Unknown", ""
    items = doc.get("items") if isinstance(doc, dict) else None
    if not isinstance(items, list) or not items:
        return "Pending", "pod_not_created"
    pod = items[0] if isinstance(items[0], dict) else {}
    status = pod.get("status") if isinstance(pod, dict) else {}
    if not isinstance(status, dict):
        status = {}
    phase = str(status.get("phase") or "Unknown")
    detail = ""
    cstats = status.get("containerStatuses")
    if isinstance(cstats, list):
        for cs in cstats:
            if not isinstance(cs, dict):
                continue
            st = cs.get("state")
            if not isinstance(st, dict):
                continue
            waiting = st.get("waiting")
            if isinstance(waiting, dict):
                reason = str(waiting.get("reason") or "").strip()
                message = str(waiting.get("message") or "").strip()
                if reason:
                    detail = reason
                    if message:
                        detail = f"{reason}: {message}"
                    break
    return phase, detail


def _k8s_wait_job(job_name: str, *, timeout_sec: int, on_progress=None) -> tuple[str, str]:
    deadline = time.time() + max(1, timeout_sec)
    last_phase = "Unknown"
    last_progress = ""
    last_progress_emit = 0.0
    while time.time() < deadline:
        rc, out, _ = _kubectl(["get", "job", job_name, "-o", "json"], timeout=15)
        if rc != 0:
            if callable(on_progress):
                on_progress("Unknown", "get_job_failed")
            time.sleep(2)
            continue
        try:
            doc = json.loads(out)
        except Exception:
            if callable(on_progress):
                on_progress("Unknown", "bad_job_json")
            time.sleep(2)
            continue
        status = doc.get("status") or {}
        if int(status.get("succeeded") or 0) > 0:
            return "Succeeded", last_phase
        if int(status.get("failed") or 0) > 0:
            return "Failed", last_phase
        conditions = status.get("conditions") or []
        if isinstance(conditions, list):
            for c in conditions:
                if not isinstance(c, dict):
                    continue
                if c.get("status") == "True":
                    t = str(c.get("type") or "")
                    if t in {"Complete", "Failed"}:
                        return ("Succeeded" if t == "Complete" else "Failed"), last_phase
        active = int(status.get("active") or 0)
        if active > 0:
            last_phase = "Running"
        else:
            pod_phase, pod_detail = _k8s_get_job_pod_phase(job_name)
            if pod_phase in {"Running"}:
                last_phase = "Running"
            elif pod_phase in {"Pending", "Unknown"}:
                last_phase = "Pending"
            elif pod_phase in {"Succeeded"}:
                last_phase = "Running"
            else:
                last_phase = pod_phase or "Pending"
            if pod_detail:
                progress = f"{last_phase}:{pod_detail}"
            else:
                progress = last_phase
            now = time.time()
            if callable(on_progress) and (progress != last_progress or (now - last_progress_emit) >= 15):
                on_progress(last_phase, pod_detail)
                last_progress = progress
                last_progress_emit = now
            time.sleep(2)
            continue
        now = time.time()
        if callable(on_progress) and (last_phase != last_progress or (now - last_progress_emit) >= 15):
            on_progress(last_phase, "")
            last_progress = last_phase
            last_progress_emit = now
        time.sleep(2)
    return "Timeout", last_phase


def _k8s_get_job_node_name(job_name: str) -> str:
    if not job_name:
        return ""
    rc, out, _ = _kubectl(
        ["get", "pod", "-l", f"job-name={job_name}", "-o", "jsonpath={.items[0].spec.nodeName}"],
        timeout=15,
    )
    if rc != 0:
        return ""
    return str(out or "").strip()


def _k8s_node_can_run_job(node_name: str) -> tuple[bool, str]:
    global _K8S_METRICS_API_UNAVAILABLE_UNTIL
    if not node_name:
        return False, "empty_node"

    rc_get, out_get, err_get = _kubectl(["get", "node", node_name, "-o", "json"], timeout=15)
    if rc_get != 0:
        return False, f"get_node_failed:{(err_get or out_get).strip()}"
    try:
        node_doc = json.loads(out_get)
    except Exception as e:
        return False, f"bad_node_json:{e}"

    spec_doc = node_doc.get("spec") if isinstance(node_doc, dict) else {}
    if not isinstance(spec_doc, dict):
        spec_doc = {}
    if bool(spec_doc.get("unschedulable")):
        return False, "node_unschedulable"

    status_doc = node_doc.get("status") if isinstance(node_doc, dict) else {}
    if not isinstance(status_doc, dict):
        status_doc = {}
    conds = status_doc.get("conditions")
    ready = False
    if isinstance(conds, list):
        for c in conds:
            if not isinstance(c, dict):
                continue
            if str(c.get("type") or "") == "Ready" and str(c.get("status") or "") == "True":
                ready = True
                break
    if not ready:
        return False, "node_not_ready"

    max_cpu_pct = 95
    max_mem_pct = 95
    try:
        max_cpu_pct = max(1, min(100, int(os.environ.get("SHERPA_K8S_NODE_MAX_CPU_PCT", "95"))))
    except Exception:
        max_cpu_pct = 95
    try:
        max_mem_pct = max(1, min(100, int(os.environ.get("SHERPA_K8S_NODE_MAX_MEM_PCT", "95"))))
    except Exception:
        max_mem_pct = 95

    def _parse_cpu_to_millicores(raw: str) -> int | None:
        txt = str(raw or "").strip()
        if not txt:
            return None
        try:
            if txt.endswith("m"):
                return int(float(txt[:-1]))
            return int(float(txt) * 1000.0)
        except Exception:
            return None

    def _parse_memory_to_bytes(raw: str) -> int | None:
        txt = str(raw or "").strip()
        if not txt:
            return None
        units = {
            "Ki": 1024,
            "Mi": 1024 ** 2,
            "Gi": 1024 ** 3,
            "Ti": 1024 ** 4,
            "Pi": 1024 ** 5,
            "Ei": 1024 ** 6,
            "K": 1000,
            "M": 1000 ** 2,
            "G": 1000 ** 3,
            "T": 1000 ** 4,
            "P": 1000 ** 5,
            "E": 1000 ** 6,
        }
        for suffix, mul in units.items():
            if txt.endswith(suffix):
                try:
                    return int(float(txt[: -len(suffix)]) * float(mul))
                except Exception:
                    return None
        try:
            return int(float(txt))
        except Exception:
            return None

    # First preference: live usage from metrics-server.
    now_ts = time.time()
    rc_top, out_top, err_top = _kubectl(["top", "node", node_name, "--no-headers"], timeout=10)
    if rc_top == 0:
        line = ""
        for raw in (out_top or "").splitlines():
            txt = raw.strip()
            if txt:
                line = txt
                break
        if line:
            parts = line.split()
            if len(parts) >= 5:
                cpu_pct_txt = parts[2].rstrip("%")
                mem_pct_txt = parts[4].rstrip("%")
                try:
                    cpu_pct = int(cpu_pct_txt)
                    mem_pct = int(mem_pct_txt)
                    if cpu_pct >= max_cpu_pct:
                        return False, f"node_cpu_busy:{cpu_pct}%"
                    if mem_pct >= max_mem_pct:
                        return False, f"node_mem_busy:{mem_pct}%"
                    return True, f"node_ready_cpu={cpu_pct}%_mem={mem_pct}%"
                except Exception:
                    pass

    # Fallback: request-based capacity check (works without metrics-server).
    detail = (err_top or out_top).strip()
    if detail and re.search(r"metrics api not available", re.sub(r"\s+", " ", detail), re.IGNORECASE):
        try:
            backoff_sec = int(
                (os.environ.get("SHERPA_K8S_METRICS_API_UNAVAILABLE_BACKOFF_SEC") or "300").strip()
            )
        except Exception:
            backoff_sec = 300
        _K8S_METRICS_API_UNAVAILABLE_UNTIL = now_ts + float(max(30, backoff_sec))

    alloc = status_doc.get("allocatable") if isinstance(status_doc, dict) else {}
    if not isinstance(alloc, dict):
        alloc = {}
    alloc_cpu_m = _parse_cpu_to_millicores(str(alloc.get("cpu") or ""))
    alloc_mem_b = _parse_memory_to_bytes(str(alloc.get("memory") or ""))

    rc_pods, out_pods, err_pods = _kubectl(
        ["get", "pods", "-A", "--field-selector", f"spec.nodeName={node_name}", "-o", "json"],
        timeout=20,
    )
    if rc_pods != 0:
        return True, "node_ready_no_metrics_capacity_unknown"

    try:
        pods_doc = json.loads(out_pods)
    except Exception:
        return True, "node_ready_no_metrics_capacity_unknown"

    items = pods_doc.get("items") if isinstance(pods_doc, dict) else []
    if not isinstance(items, list):
        items = []

    req_cpu_m = 0
    req_mem_b = 0
    for pod in items:
        if not isinstance(pod, dict):
            continue
        pod_status = pod.get("status") if isinstance(pod.get("status"), dict) else {}
        phase = str(pod_status.get("phase") or "").strip()
        if phase in {"Succeeded", "Failed"}:
            continue
        pod_spec = pod.get("spec") if isinstance(pod.get("spec"), dict) else {}
        containers = pod_spec.get("containers")
        if not isinstance(containers, list):
            containers = []
        for c in containers:
            if not isinstance(c, dict):
                continue
            resources = c.get("resources") if isinstance(c.get("resources"), dict) else {}
            requests = resources.get("requests") if isinstance(resources.get("requests"), dict) else {}
            cpu_m = _parse_cpu_to_millicores(str(requests.get("cpu") or "0"))
            mem_b = _parse_memory_to_bytes(str(requests.get("memory") or "0"))
            if cpu_m is not None and cpu_m > 0:
                req_cpu_m += cpu_m
            if mem_b is not None and mem_b > 0:
                req_mem_b += mem_b

    cpu_req_pct: int | None = None
    mem_req_pct: int | None = None
    if alloc_cpu_m and alloc_cpu_m > 0:
        cpu_req_pct = int((float(req_cpu_m) / float(alloc_cpu_m)) * 100.0)
    if alloc_mem_b and alloc_mem_b > 0:
        mem_req_pct = int((float(req_mem_b) / float(alloc_mem_b)) * 100.0)

    if cpu_req_pct is not None and cpu_req_pct >= max_cpu_pct:
        return False, f"node_cpu_request_busy:{cpu_req_pct}%"
    if mem_req_pct is not None and mem_req_pct >= max_mem_pct:
        return False, f"node_mem_request_busy:{mem_req_pct}%"

    if cpu_req_pct is not None or mem_req_pct is not None:
        cpu_txt = f"{cpu_req_pct}%" if cpu_req_pct is not None else "n/a"
        mem_txt = f"{mem_req_pct}%" if mem_req_pct is not None else "n/a"
        return True, f"node_ready_req_cpu={cpu_txt}_req_mem={mem_txt}"

    return True, "node_ready_no_metrics_capacity_unknown"


def _execute_k8s_job(
    *,
    job_id: str,
    job_name: str,
    payload: dict[str, object],
    result_path: Path,
    error_path: Path,
    wait_timeout: int,
) -> tuple[object, str]:
    manifest = _k8s_build_manifest(job_name, payload)
    stage_name = str(payload.get("stop_after_step") or payload.get("resume_from_step") or "").strip().lower()
    rc_apply, _, err_apply = _kubectl(["apply", "-f", "-"], input_text=manifest, timeout=60)
    if rc_apply != 0:
        raise RuntimeError(f"k8s job submit failed: {err_apply.strip()}")
    _job_update(job_id, k8s_phase="Submitted")

    def _progress_cb(phase: str, detail: str) -> None:
        phase_txt = (phase or "Unknown").strip()
        detail_txt = (detail or "").strip()
        if detail_txt:
            _job_update(job_id, k8s_phase=f"{phase_txt}: {detail_txt[:220]}")
        else:
            _job_update(job_id, k8s_phase=phase_txt)

    status, last_phase = _k8s_wait_job(job_name, timeout_sec=wait_timeout, on_progress=_progress_cb)
    _job_update(job_id, k8s_phase=status)
    if status == "Timeout":
        _k8s_delete_job(job_name)
        raise RuntimeError("k8s_job_timeout")
    if status == "Failed":
        logs = _k8s_collect_job_logs(job_name)
        pod_details = _k8s_get_job_pod_details(job_name)
        err_txt = ""
        if error_path.is_file():
            err_txt = error_path.read_text(encoding="utf-8", errors="replace")
        failure_result = _classify_k8s_stage_failure(stage_name, pod_details, logs, err_txt)
        try:
            if not error_path.is_file():
                error_path.parent.mkdir(parents=True, exist_ok=True)
            error_lines = [f"k8s_job_failed phase={last_phase}"]
            pod_name = str(((failure_result.get("k8s_failure") or {}) if isinstance(failure_result, dict) else {}).get("pod_name") or "").strip()
            if pod_name:
                error_lines.append(f"pod={pod_name}")
            code = str(failure_result.get("error_code") or "").strip()
            if code:
                error_lines.append(f"error_code={code}")
            kind = str(failure_result.get("error_kind") or "").strip()
            if kind:
                error_lines.append(f"error_kind={kind}")
            log_tail = str(((failure_result.get("k8s_failure") or {}) if isinstance(failure_result, dict) else {}).get("logs_tail") or "").strip()
            if log_tail:
                error_lines.append("")
                error_lines.append(log_tail)
            error_path.write_text("\n".join(error_lines).strip() + "\n", encoding="utf-8")
        except Exception:
            pass
        try:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(
                    {
                        "ok": False,
                        "error": f"k8s_job_failed phase={last_phase}",
                        "result": failure_result,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass
        if not _k8s_keep_finished_jobs():
            _k8s_delete_job(job_name)
        msg = f"k8s_job_failed phase={last_phase}"
        if err_txt.strip():
            msg += f": {err_txt.strip()}"
        elif str(failure_result.get("error_code") or "").strip():
            msg += f": {failure_result.get('error_code')}"
        if logs.strip():
            print(f"[job {job_id}] k8s logs tail:\n{logs}")
        raise _K8sJobFailure(msg, result=failure_result)
    if not result_path.is_file():
        logs = _k8s_collect_job_logs(job_name)
        if logs.strip():
            print(f"[job {job_id}] k8s logs tail:\n{logs}")
        raise RuntimeError("k8s_job_missing_result")
    raw = result_path.read_text(encoding="utf-8", errors="replace")
    try:
        doc = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"k8s_job_bad_result_json: {e}")
    # Capture node before deleting the Job/Pod so stage pinning can persist
    # even when keep-finished-jobs is disabled.
    stage_node_name = _k8s_get_job_node_name(job_name)
    if not _k8s_keep_finished_jobs():
        _k8s_delete_job(job_name)
    if not bool(doc.get("ok")):
        raise RuntimeError(str(doc.get("error") or "k8s_worker_failed"))
    return doc.get("result"), stage_node_name


def _k8s_stage_wait_timeout_sec(
    *,
    stage: str,
    total_time_budget_sec: int,
    run_time_budget_sec: int,
    run_unlimited_round_budget_sec: int | None = None,
    run_fuzzer_count: int = 1,
    run_parallelism: int = 1,
) -> int:
    """Compute k8s stage wait timeout.

    For run stage this is multi-round aware:
    - finite run budget: use total run budget + grace.
    - unlimited run budget: estimate rounds from fuzzer_count/parallelism and
      multiply by per-round unlimited budget.
    """
    try:
        grace_run = int(os.environ.get("SHERPA_K8S_RUN_TIMEOUT_GRACE_SEC", "900"))
    except Exception:
        grace_run = 900
    try:
        grace_default = int(os.environ.get("SHERPA_K8S_STAGE_TIMEOUT_GRACE_SEC", "180"))
    except Exception:
        grace_default = 180
    try:
        inter_round_buffer_sec = int(
            os.environ.get("SHERPA_K8S_RUN_TIMEOUT_INTER_ROUND_BUFFER_SEC", "120")
        )
    except Exception:
        inter_round_buffer_sec = 120
    if run_unlimited_round_budget_sec is None:
        try:
            run_unlimited_round_budget = int(
                os.environ.get("SHERPA_RUN_UNLIMITED_ROUND_BUDGET_SEC", "7200")
            )
        except Exception:
            run_unlimited_round_budget = 7200
    else:
        run_unlimited_round_budget = int(run_unlimited_round_budget_sec)
    try:
        run_timeout_cap_sec = int(os.environ.get("SHERPA_K8S_RUN_TIMEOUT_MAX_SEC", "0"))
    except Exception:
        run_timeout_cap_sec = 0

    grace_run = max(60, grace_run)
    grace_default = max(30, grace_default)
    inter_round_buffer_sec = max(0, inter_round_buffer_sec)
    run_unlimited_round_budget = max(300, run_unlimited_round_budget)

    total_base = total_time_budget_sec if total_time_budget_sec > 0 else 7200
    if stage != "run":
        return max(300, total_base + grace_default)

    if run_time_budget_sec > 0:
        run_base = run_time_budget_sec
    else:
        safe_parallel = max(1, run_parallelism)
        safe_fuzzer_count = max(1, run_fuzzer_count)
        round_count = max(1, math.ceil(safe_fuzzer_count / safe_parallel))
        run_base = (round_count * run_unlimited_round_budget) + (
            max(0, round_count - 1) * inter_round_buffer_sec
        )
    try:
        seed_gen_retry_multiplier = int(os.environ.get("SHERPA_SEED_GEN_RETRY_MULTIPLIER", "3"))
    except Exception:
        seed_gen_retry_multiplier = 3
    seed_gen_retry_multiplier = max(1, seed_gen_retry_multiplier)
    run_base *= seed_gen_retry_multiplier
    wait_timeout = max(300, run_base + grace_run)
    if run_timeout_cap_sec > 0:
        wait_timeout = min(wait_timeout, run_timeout_cap_sec)
    return wait_timeout


def _estimate_run_fuzzer_count(repo_root: str) -> int:
    raw = str(repo_root or "").strip()
    if not raw:
        return 1
    root = Path(raw)
    if not root.exists():
        return 1

    fuzz_out = root / "fuzz" / "out"
    try:
        if fuzz_out.is_dir():
            count = 0
            for p in fuzz_out.iterdir():
                if not p.is_file():
                    continue
                if os.access(str(p), os.X_OK) or p.suffix.lower() == ".exe":
                    count += 1
            if count > 0:
                return count
    except Exception:
        pass

    execution_plan = root / "fuzz" / "execution_plan.json"
    try:
        if execution_plan.is_file():
            doc = json.loads(execution_plan.read_text(encoding="utf-8", errors="replace"))
            targets = doc.get("execution_targets")
            if isinstance(targets, list):
                count = len([t for t in targets if isinstance(t, dict)])
                if count > 0:
                    return count
    except Exception:
        pass
    return 1


def _estimate_run_parallelism(stage_ctx: dict[str, object]) -> int:
    raw = str(
        (stage_ctx or {}).get("run_parallel_fuzzers_override")
        or os.environ.get("SHERPA_PARALLEL_FUZZERS")
        or "3"
    ).strip()
    try:
        return max(1, min(int(raw), 64))
    except Exception:
        return 3


def _list_runtime_containers_for_repo(repo_root: str) -> list[str]:
    if _executor_mode() == "k8s_job":
        return []
    root = str(repo_root or "").strip()
    if not root:
        return []

    repo_sha1 = hashlib.sha1(root.encode("utf-8", errors="ignore")).hexdigest()
    found: set[str] = set()
    filters = [
        ["ps", "-q", "--filter", f"label=sherpa.repo_root_sha1={repo_sha1}"],
        ["ps", "-q", "--filter", f"volume={root}"],
    ]
    for cmd in filters:
        rc, out, _ = _docker_cli(cmd, timeout=15)
        if rc != 0:
            continue
        for line in out.splitlines():
            cid = line.strip()
            if cid:
                found.add(cid)
    return sorted(found)


def _stop_runtime_containers_for_repo(repo_root: str) -> list[str]:
    if _executor_mode() == "k8s_job":
        return []
    killed: list[str] = []
    for cid in _list_runtime_containers_for_repo(repo_root):
        rc, _, _ = _docker_cli(["rm", "-f", cid], timeout=20)
        if rc == 0:
            killed.append(cid)
    return killed


def _is_cancel_requested(job_id: str) -> bool:
    snap = _job_snapshot(job_id)
    return bool(snap and snap.get("cancel_requested"))


def _ensure_job_logs_dir() -> None:
    _JOB_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _job_log_path(job_id: str) -> Path:
    return _JOB_LOGS_DIR / f"{job_id}.log"


def _read_log_tail(path: Path, *, max_chars: int) -> str:
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if max_chars > 0:
        return text[-max_chars:]
    return text


def _hydrate_job_log_from_disk(job: dict) -> None:
    if (job.get("log") or "").strip():
        return
    raw = str(job.get("log_file") or "").strip()
    if not raw:
        return
    txt = _read_log_tail(Path(raw), max_chars=max(0, _JOB_RESTORE_LOG_MAX_CHARS))
    if txt:
        job["log"] = txt


def _job_store_database_url() -> str:
    return str(os.environ.get("DATABASE_URL", "") or "").strip()


def _persist_job_state(job: dict) -> None:
    store = _JOB_STORE
    if store is None:
        return
    try:
        store.upsert_job(job)
    except Exception:
        pass


def _restore_jobs_from_store() -> None:
    store = _JOB_STORE
    if store is None:
        return
    restored = store.load_jobs()
    if not restored:
        return

    now = time.time()
    changed_ids: list[str] = []
    for job_id, job in restored.items():
        status = str(job.get("status") or "").strip().lower()
        if status in {"queued", "running", "resuming"}:
            active_step = str(job.get("workflow_active_step") or "").strip().lower()
            last_step = str(job.get("workflow_last_step") or "").strip().lower()
            fallback = "plan" if status == "queued" else "build"
            resume_step_raw = active_step or last_step
            repo_root = str(job.get("workflow_repo_root") or "").strip()
            if not resume_step_raw or not repo_root:
                infer_step, infer_root = _infer_checkpoint_from_log_text(str(job.get("log") or ""))
                if infer_step and not resume_step_raw:
                    resume_step_raw = infer_step
                if infer_root and not repo_root:
                    repo_root = infer_root
            resume_step = _normalize_resume_step(resume_step_raw or fallback)
            job["status"] = "recoverable"
            job["recoverable"] = True
            job["last_interrupted_at"] = now
            job["last_resume_reason"] = "service_restart"
            job["error"] = str(job.get("error") or "").strip() or "job interrupted by service restart"
            if str(job.get("kind") or "") == "fuzz":
                job["resume_from_step"] = resume_step
                if repo_root:
                    job["resume_repo_root"] = repo_root
            job["updated_at"] = now
            changed_ids.append(job_id)
        _hydrate_job_log_from_disk(job)

    with _JOBS_LOCK:
        _JOBS.clear()
        _JOBS.update(restored)

    for job_id in changed_ids:
        job = restored.get(job_id)
        if job is not None:
            _persist_job_state(job)


def _init_job_store() -> None:
    global _JOB_STORE
    db_url = _job_store_database_url()
    if not db_url:
        raise RuntimeError("DATABASE_URL is required (Postgres-only job store)")
    store = PostgresJobStore(db_url)
    store.init_schema()
    _JOB_STORE = store
    _restore_jobs_from_store()


def _classify_log_level(line: str) -> str:
    txt = (line or "").lower()
    if any(k in txt for k in ["traceback", "exception", " fatal", "error", "failed", "cannot find"]):
        return "error"
    if any(k in txt for k in ["warn", "retry", "timeout", "deprecation"]):
        return "warn"
    return "info"


def _classify_log_category(line: str) -> str:
    txt = (line or "").lower()
    if "[wf" in txt:
        return "workflow"
    if "[opencodehelper]" in txt or "opencode" in txt:
        return "opencode"
    if "docker" in txt or "container" in txt:
        return "docker"
    if any(k in txt for k in ["cmake", "clang", "gcc", "linker", "ld:", "build"]):
        return "build"
    if "[task" in txt:
        return "task"
    if "[job" in txt:
        return "job"
    return "general"


_WF_STEP_ENTRY_RE = re.compile(r"\[wf[^\]]*\]\s*->\s*([a-z_]+)")
_WF_STEP_EXIT_RE = re.compile(r"\[wf[^\]]*\]\s*<-\s*([a-z_]+)")
_WF_REPO_ROOT_RE = re.compile(r"\brepo_root=(.+?)(?:\s+dt=|$)")
_WF_METRICS_RE = re.compile(r"\[wf-metrics\]\s*(\{.+)")


def _update_workflow_checkpoint_from_line(job_id: str, line: str) -> None:
    if not line:
        return
    entry = _WF_STEP_ENTRY_RE.search(line)
    if entry:
        step = entry.group(1).strip()
        _job_update(
            job_id,
            workflow_last_step=step,
            workflow_active_step=step,
            workflow_last_step_ts=time.time(),
        )

    exit_m = _WF_STEP_EXIT_RE.search(line)
    if exit_m:
        step = exit_m.group(1).strip()
        _job_update(
            job_id,
            workflow_last_step=step,
            workflow_last_completed_step=step,
            workflow_active_step="",
            workflow_last_step_ts=time.time(),
        )

    repo_m = _WF_REPO_ROOT_RE.search(line)
    if repo_m:
        repo_root = repo_m.group(1).strip()
        if repo_root:
            _job_update(job_id, workflow_repo_root=repo_root)

    # Parse structured per-fuzzer metrics emitted by workflow_graph.py
    metrics_m = _WF_METRICS_RE.search(line)
    if metrics_m:
        try:
            payload = json.loads(metrics_m.group(1))
            _job_update(
                job_id,
                fuzz_metrics=payload,
                fuzz_metrics_ts=payload.get("ts") or time.time(),
                fuzz_fuzzers=payload.get("fuzzers") or {},
                fuzz_max_cov=int(payload.get("max_cov") or 0),
                fuzz_max_ft=int(payload.get("max_ft") or 0),
                fuzz_total_execs_per_sec=int(payload.get("total_execs_per_sec") or 0),
                fuzz_crash_found=bool(payload.get("crash_found")),
                fuzz_coverage_history=payload.get("coverage_history") or [],
                fuzz_coverage_source_report=payload.get("coverage_source_report") or {},
                fuzz_coverage_loop_round=int(payload.get("coverage_loop_round") or 0),
                fuzz_coverage_loop_max_rounds=int(payload.get("coverage_loop_max_rounds") or 0),
                fuzz_coverage_plateau_streak=int(payload.get("coverage_plateau_streak") or 0),
                fuzz_coverage_seed_profile=str(payload.get("coverage_seed_profile") or ""),
                fuzz_coverage_quality_flags=payload.get("coverage_quality_flags") or [],
            )
        except Exception:
            pass


def _infer_checkpoint_from_log_text(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    active_step = ""
    last_step = ""
    repo_root = ""
    for line in text.splitlines():
        m1 = _WF_STEP_ENTRY_RE.search(line)
        if m1:
            step = m1.group(1).strip().lower()
            active_step = step
            last_step = step
        m2 = _WF_STEP_EXIT_RE.search(line)
        if m2:
            step = m2.group(1).strip().lower()
            last_step = step
            active_step = ""
        m3 = _WF_REPO_ROOT_RE.search(line)
        if m3:
            repo_root = m3.group(1).strip()
    step_out = active_step or last_step
    return step_out, repo_root


def _job_update(job_id: str, **fields: object) -> None:
    snapshot: dict | None = None
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        job.update(fields)
        job["updated_at"] = time.time()
        snapshot = dict(job)
    if snapshot is not None:
        _persist_job_state(snapshot)


def _job_append_log(job_id: str, chunk: str) -> None:
    if not chunk:
        return
    snapshot: dict | None = None
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        buf = (job.get("log", "") or "") + chunk
        if _JOB_MEMORY_LOG_MAX_CHARS > 0:
            job["log"] = buf[-_JOB_MEMORY_LOG_MAX_CHARS:]
        else:
            job["log"] = buf
        job["updated_at"] = time.time()
        snapshot = dict(job)
    if snapshot is not None:
        _persist_job_state(snapshot)


class _Tee(StringIO):
    def __init__(self, job_id: str, *, log_file: Path | None = None) -> None:
        super().__init__()
        self._job_id = job_id
        self._fh = None
        self._split_fhs: dict[str, object] = {}
        self._base_path: Path | None = None
        if log_file is not None:
            try:
                _ensure_job_logs_dir()
                self._fh = open(log_file, "a", encoding="utf-8")
                self._base_path = log_file.with_suffix("")
            except Exception:
                # Best-effort: if we cannot write to disk, keep in-memory logs working.
                self._fh = None
                self._base_path = None

    def _split_write(self, line: str) -> None:
        if not line or self._base_path is None:
            return
        level = _classify_log_level(line)
        category = _classify_log_category(line)
        targets = [
            f"{self._base_path.name}.level.{level}.log",
            f"{self._base_path.name}.cat.{category}.log",
        ]
        for filename in targets:
            handle = self._split_fhs.get(filename)
            if handle is None:
                try:
                    handle = open(self._base_path.parent / filename, "a", encoding="utf-8")
                    self._split_fhs[filename] = handle
                except Exception:
                    continue
            try:
                handle.write(line)
                handle.flush()
            except Exception:
                pass

    def write(self, s: str) -> int:
        safe = _redact_sensitive_text(s) if s else s
        if self._fh is not None and safe:
            try:
                self._fh.write(safe)
                self._fh.flush()
            except Exception:
                # Do not break the job if disk logging fails mid-run.
                pass
        if safe:
            for line in safe.splitlines(keepends=True):
                self._split_write(line)
                _update_workflow_checkpoint_from_line(self._job_id, line)
        _job_append_log(self._job_id, safe)
        return super().write(safe)

    def close(self) -> None:
        try:
            if self._fh is not None:
                self._fh.close()
            for handle in self._split_fhs.values():
                try:
                    handle.close()
                except Exception:
                    pass
        finally:
            super().close()


class _JobAwareStream:
    def __init__(self, original, *, stream_kind: str) -> None:
        self._original = original
        self._stream_kind = stream_kind

    def _active_tee(self):
        if self._stream_kind == "stdout":
            return _ACTIVE_JOB_STDOUT_TEE.get()
        return _ACTIVE_JOB_STDERR_TEE.get()

    def write(self, s: str) -> int:
        txt = str(s or "")
        tee = self._active_tee()
        if tee is not None and txt:
            try:
                tee.write(txt)
            except Exception:
                pass
        try:
            return int(self._original.write(txt))
        except Exception:
            return len(txt)

    def flush(self) -> None:
        tee = self._active_tee()
        if tee is not None:
            try:
                tee.flush()
            except Exception:
                pass
        try:
            self._original.flush()
        except Exception:
            pass

    def __getattr__(self, name: str):
        return getattr(self._original, name)


if not isinstance(sys.stdout, _JobAwareStream):
    sys.stdout = _JobAwareStream(sys.stdout, stream_kind="stdout")
if not isinstance(sys.stderr, _JobAwareStream):
    sys.stderr = _JobAwareStream(sys.stderr, stream_kind="stderr")


def _iso_time(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _status_for_counter(raw: str | None) -> str:
    s = str(raw or "").strip().lower()
    if s in {"queued"}:
        return "queued"
    if s in {"running", "resuming"}:
        return "running"
    if s in {"success", "resumed"}:
        return "success"
    if s in {"error", "recoverable", "resume_failed"}:
        return "error"
    return "error"


def _status_for_parent(raw: str | None) -> str:
    s = str(raw or "").strip().lower()
    if s in {"queued"}:
        return "queued"
    if s in {"running", "resuming", "recoverable"}:
        return "running"
    if s in {"success", "resumed"}:
        return "success"
    if s in {"error", "resume_failed"}:
        return "error"
    return "error"


def _is_status_terminal(raw: str | None) -> bool:
    s = str(raw or "").strip().lower()
    return s in {"success", "resumed", "error", "resume_failed"}


_RESUMABLE_WORKFLOW_STEPS = {
    "analysis",
    "plan",
    "synthesize",
    "build",
    "fix_build",
    "run",
    "crash-triage",
    "fix-harness",
    "coverage-analysis",
    "improve-harness",
    "re-build",
    "re-run",
    "crash-analysis",
    "repro_crash",
    "fix_crash",
}
_STAGED_WORKFLOW_STEPS = (
    "analysis",
    "plan",
    "synthesize",
    "build",
    "run",
    "crash-triage",
    "fix-harness",
    "coverage-analysis",
    "improve-harness",
    "re-build",
    "re-run",
    "crash-analysis",
)


def _normalize_resume_step(raw: str | None) -> str:
    s = str(raw or "").strip().lower()
    if s == "stop":
        return "stop"
    if s == "repro_crash":
        return "re-build"
    if s in {"crash_triage", "crash-triage"}:
        return "crash-triage"
    if s in {"fix_harness", "fix-harness"}:
        return "fix-harness"
    if s in {"crash_analysis", "crash-analysis"}:
        return "crash-analysis"
    if s in _RESUMABLE_WORKFLOW_STEPS:
        return s
    return "analysis"


def _staged_sequence_from(raw_start: str | None) -> list[str]:
    start = _normalize_resume_step(raw_start)
    if start in {"fix_build", "fix_crash"}:
        start = "build"
    try:
        idx = _STAGED_WORKFLOW_STEPS.index(start)
    except ValueError:
        idx = 0
    return list(_STAGED_WORKFLOW_STEPS[idx:])


def _legacy_error_code_for_job(job: dict | None) -> str:
    if not isinstance(job, dict):
        return ""
    direct = str(job.get("error_code") or "").strip()
    if direct:
        return direct
    resume = str(job.get("resume_error_code") or "").strip()
    if resume:
        return resume
    result = job.get("result")
    if isinstance(result, dict):
        for key in (
            "fix_build_terminal_reason",
            "run_terminal_reason",
            "build_error_code",
            "run_error_kind",
            "build_error_kind",
            "error_code",
        ):
            val = str(result.get(key) or "").strip()
            if val:
                return val
    status = str(job.get("status") or "").strip().lower()
    if status in {"error", "resume_failed", "recoverable"}:
        return "unknown_error"
    return ""


def _legacy_error_kind_for_job(job: dict | None) -> str:
    if not isinstance(job, dict):
        return ""
    result = job.get("result")
    if isinstance(result, dict):
        for key in ("run_error_kind", "build_error_kind", "error_kind"):
            val = str(result.get(key) or "").strip()
            if val:
                return val
    status = str(job.get("status") or "").strip().lower()
    if status in {"error", "resume_failed", "recoverable"}:
        return "unknown"
    return ""


def _legacy_error_signature_for_job(job: dict | None) -> str:
    if not isinstance(job, dict):
        return ""
    result = job.get("result")
    if isinstance(result, dict):
        for key in (
            "build_error_signature_short",
            "build_error_signature",
            "timeout_signature",
            "crash_signature",
            "error_signature",
        ):
            val = str(result.get(key) or "").strip()
            if val:
                return val
    return ""


def _coerce_error_object(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        return {}
    stage = str(raw.get("stage") or "").strip().lower()
    kind = str(raw.get("kind") or "").strip().lower()
    code = str(raw.get("code") or "").strip().lower()
    message = str(raw.get("message") or "").strip()
    detail = str(raw.get("detail") or "").strip()
    signature = str(raw.get("signature") or "").strip()
    retryable = bool(raw.get("retryable"))
    terminal = bool(raw.get("terminal"))
    at = int(_safe_float(raw.get("at")) or 0)
    if not (code or message or signature or terminal):
        return {}
    if at <= 0:
        at = int(time.time())
    return {
        "stage": stage,
        "kind": kind,
        "code": code,
        "message": message,
        "detail": detail,
        "signature": signature,
        "retryable": retryable,
        "terminal": terminal,
        "at": at,
    }


def _error_object_for_job(job: dict | None) -> dict[str, object]:
    if not isinstance(job, dict):
        return {}
    result = job.get("result")
    result_dict = result if isinstance(result, dict) else {}
    for source in (job.get("error"), result_dict.get("error")):
        normalized = _coerce_error_object(source)
        if normalized:
            return normalized

    code = _legacy_error_code_for_job(job)
    kind = _legacy_error_kind_for_job(job)
    signature = _legacy_error_signature_for_job(job)
    message = str(
        job.get("last_error")
        or result_dict.get("last_error")
        or job.get("error")
        or ""
    ).strip()
    stage = str(
        job.get("workflow_active_step")
        or job.get("workflow_last_step")
        or result_dict.get("last_step")
        or job.get("k8s_phase")
        or ""
    ).strip().lower()
    terminal = bool(result_dict.get("failed")) or str(job.get("status") or "").strip().lower() in {
        "error",
        "resume_failed",
        "recoverable",
    }
    retryable = bool(code) and not terminal
    if not (code or kind or signature or message or terminal):
        return {}
    if not kind and code:
        if code.startswith("run_"):
            kind = "run"
        elif code.startswith("build_") or "build" in code:
            kind = "build"
        elif "crash" in code:
            kind = "crash"
        elif "timeout" in code:
            kind = "timeout"
        else:
            kind = "generic_failure"
    return {
        "stage": stage,
        "kind": kind,
        "code": code,
        "message": message,
        "detail": message,
        "signature": signature,
        "retryable": retryable,
        "terminal": terminal,
        "at": int(_safe_float(job.get("updated_at")) or _safe_float(job.get("finished_at")) or time.time()),
    }


def _error_code_for_job(job: dict | None) -> str:
    return str(_error_object_for_job(job).get("code") or "")


def _error_kind_for_job(job: dict | None) -> str:
    return str(_error_object_for_job(job).get("kind") or "")


def _error_signature_for_job(job: dict | None) -> str:
    return str(_error_object_for_job(job).get("signature") or "")


def _runtime_mode_for_job(job: dict | None) -> str:
    if isinstance(job, dict):
        v = str(job.get("runtime_mode") or "").strip().lower()
        if v in {"native", "docker"}:
            return v
    return "native" if _executor_mode() == "k8s_job" else "docker"


def _phase_for_job(job: dict | None) -> str:
    if not isinstance(job, dict):
        return "unknown"
    for key in ("workflow_active_step", "workflow_last_step", "k8s_phase"):
        val = str(job.get(key) or "").strip()
        if val:
            return val
    status = str(job.get("status") or "").strip()
    return status or "unknown"


def _status_upper(status: str) -> str:
    lowered = str(status or "").strip().lower()
    mapping = {
        "queued": "QUEUED",
        "running": "RUNNING",
        "resuming": "RUNNING",
        "recoverable": "FAILED",
        "resume_failed": "FAILED",
        "success": "SUCCESS",
        "resumed": "COMPLETED",
        "error": "ERROR",
    }
    return mapping.get(lowered, (str(status or "").strip().upper() or "UNKNOWN"))


def _task_display_repo(job: dict | None) -> str | None:
    if not isinstance(job, dict):
        return None
    repo = str(job.get("repo") or "").strip()
    if repo and repo.lower() != "batch":
        return repo
    request = job.get("request") if isinstance(job.get("request"), dict) else {}
    jobs = request.get("jobs") if isinstance(request, dict) else []
    if isinstance(jobs, list):
        repos: list[str] = []
        for item in jobs:
            if not isinstance(item, dict):
                continue
            code_url = str(item.get("code_url") or "").strip()
            if code_url:
                parsed = urlparse(code_url)
                path = str(parsed.path or "").rstrip("/")
                slug = path.rsplit("/", 1)[-1] if path else ""
                if slug.endswith(".git"):
                    slug = slug[:-4]
                repos.append(slug or code_url)
        if repos:
            if len(set(repos)) == 1:
                return repos[0]
            return f"{repos[0]} (+{len(repos) - 1} more)"
    return repo or None


def _task_progress_from_children(derived_status: str, children_status: dict[str, int]) -> int:
    total = int(children_status.get("total") or 0)
    if total <= 0:
        return 100 if derived_status in {"success"} else 0
    success = int(children_status.get("success") or 0)
    error = int(children_status.get("error") or 0)
    done = max(0, min(total, success + error))
    pct = int(round((float(done) / float(total)) * 100.0))
    if derived_status in {"running", "queued"}:
        return max(0, min(99, pct))
    return max(0, min(100, pct if pct > 0 else (100 if derived_status == "success" else 0)))


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.glob("**/*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except Exception:
                continue
    return total


def _safe_float(raw: object) -> float | None:
    try:
        if raw is None:
            return None
        v = float(raw)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def _status_bucket(raw: str | None) -> str:
    return _status_for_counter(raw)


def _job_duration_seconds(job: dict) -> float | None:
    start = _safe_float(job.get("started_at"))
    end = _safe_float(job.get("finished_at"))
    if start is None or end is None:
        return None
    dur = end - start
    if dur < 0:
        return None
    return dur


def _format_duration_human(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    sec = max(0, int(round(seconds)))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _format_percent(value: float | None, digits: int = 1) -> str | None:
    if value is None:
        return None
    return f"{value:.{digits}f}"


def _format_trend(current: float | None, previous: float | None, *, unit_suffix: str = "%") -> str | None:
    if current is None or previous is None:
        return None
    delta = current - previous
    arrow = "▲" if delta >= 0 else "▼"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}{unit_suffix} {arrow}"


def _extract_numeric_values_by_keys(obj: object, keys: set[str], *, max_count: int = 128) -> list[float]:
    out: list[float] = []
    stack: list[object] = [obj]
    while stack and len(out) < max_count:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if str(k) in keys:
                    fv = _safe_float(v)
                    if fv is not None:
                        out.append(fv)
                        if len(out) >= max_count:
                            break
                if isinstance(v, (dict, list, tuple)):
                    stack.append(v)
        elif isinstance(cur, (list, tuple)):
            for v in cur:
                if isinstance(v, (dict, list, tuple)):
                    stack.append(v)
    return out


def _count_jobs_in_window(jobs: list[dict], *, field: str, window_start: float, window_end: float) -> int:
    n = 0
    for job in jobs:
        ts = _safe_float(job.get(field))
        if ts is None:
            continue
        if window_start <= ts < window_end:
            n += 1
    return n


def _performance_series_from_jobs(now: float, fuzz_jobs: list[dict]) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    # 6 x 4h windows over last 24h + current point.
    windows = [24, 20, 16, 12, 8, 4, 0]
    for h in windows:
        end_ts = now - float(h * 3600)
        start_ts = end_ts - float(4 * 3600)
        started = _count_jobs_in_window(
            fuzz_jobs, field="started_at", window_start=start_ts, window_end=end_ts
        )
        finished = [
            job for job in fuzz_jobs
            if (ts := _safe_float(job.get("finished_at"))) is not None and start_ts <= ts < end_ts
        ]
        durations = [d for d in (_job_duration_seconds(job) for job in finished) if d is not None]
        avg_latency = (sum(durations) / float(len(durations))) if durations else None
        points.append(
            {
                "time": datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime("%H:%M"),
                "throughput": started,
                "latency": round(avg_latency, 2) if avg_latency is not None else None,
            }
        )
    return points


_RUN_EXEC_RATE_RE = re.compile(
    r"(?:stat::(?:average_)?execs?_per_sec|exec/s)\s*[:=]\s*(?P<value>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _extract_execs_per_sec_from_text(text: str) -> float | None:
    if not text:
        return None
    latest: float | None = None
    for line in text.splitlines():
        m = _RUN_EXEC_RATE_RE.search(line)
        if not m:
            continue
        try:
            latest = float(m.group("value"))
        except Exception:
            continue
    if latest is None or latest <= 0:
        return None
    return latest


def _job_execs_per_sec(job: dict) -> float | None:
    result = job.get("result")
    if isinstance(result, dict):
        values = _extract_numeric_values_by_keys(
            result,
            {
                "final_execs_per_sec",
                "execs_per_sec",
                "average_exec_per_sec",
            },
            max_count=64,
        )
        positive_values = [float(v) for v in values if float(v) > 0]
        if positive_values:
            return sum(positive_values)
    text = str(job.get("log") or "")
    if not text:
        log_file = str(job.get("log_file") or "").strip()
        if log_file:
            text = _read_log_tail(Path(log_file), max_chars=65536)
    return _extract_execs_per_sec_from_text(text)

def _collect_exec_rates_for_system(fuzz_jobs: list[dict], now: float) -> list[float]:
    rates: list[float] = []

    # Always prefer currently running fuzz jobs when real run metrics are present.
    for j in fuzz_jobs:
        if _status_bucket(str(j.get("status") or "")) != "running":
            continue
        rate = _job_execs_per_sec(j)
        if rate is not None and rate > 0:
            rates.append(rate)
    if rates:
        return rates

    # If no running metrics are available, fall back to recent successful jobs
    # with progressively wider windows. Keep this real-data-only (no estimation).
    for window_sec in (300.0, 3600.0, 21600.0, 86400.0):
        cutoff = now - window_sec
        window_rates: list[float] = []
        for j in fuzz_jobs:
            if _status_bucket(str(j.get("status") or "")) != "success":
                continue
            finished_at = _safe_float(j.get("finished_at"))
            if finished_at is None or finished_at < cutoff:
                continue
            rate = _job_execs_per_sec(j)
            if rate is not None and rate > 0:
                window_rates.append(rate)
        if window_rates:
            return window_rates

    return rates
def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    seq = sorted(values)
    if len(seq) == 1:
        return seq[0]
    q_clamped = max(0.0, min(1.0, q))
    pos = q_clamped * float(len(seq) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return seq[lo]
    frac = pos - float(lo)
    return seq[lo] + (seq[hi] - seq[lo]) * frac


def _http_metrics_snapshot(*, now: float, window_sec: float = 300.0) -> dict[str, float | int | None]:
    cutoff = now - float(max(30.0, window_sec * 2.0))
    with _HTTP_METRICS_LOCK:
        while _HTTP_REQUEST_EVENTS and _HTTP_REQUEST_EVENTS[0][0] < cutoff:
            _HTTP_REQUEST_EVENTS.popleft()
        events = list(_HTTP_REQUEST_EVENTS)

    current_start = now - window_sec
    prev_start = now - (window_sec * 2.0)
    current = [e for e in events if current_start <= e[0] < now]
    previous = [e for e in events if prev_start <= e[0] < current_start]
    current_lat = [float(e[1]) for e in current]
    current_errors = sum(1 for e in current if int(e[2]) >= 500)
    previous_errors = sum(1 for e in previous if int(e[2]) >= 500)

    current_qps = (float(len(current)) / window_sec) if window_sec > 0 else None
    previous_qps = (float(len(previous)) / window_sec) if window_sec > 0 else None
    current_err_ratio = (float(current_errors) / float(len(current)) * 100.0) if current else 0.0
    previous_err_ratio = (float(previous_errors) / float(len(previous)) * 100.0) if previous else 0.0
    return {
        "qps": current_qps,
        "qps_prev": previous_qps,
        "lat_p95_ms": _percentile(current_lat, 0.95),
        "error_ratio_pct": current_err_ratio,
        "error_ratio_prev_pct": previous_err_ratio,
        "request_count": len(current),
    }


def _format_tokens_per_hour(tokens_per_hour: float | None) -> str | None:
    if tokens_per_hour is None:
        return None
    v = max(0.0, tokens_per_hour)
    if v >= 1_000_000.0:
        return f"{v / 1_000_000.0:.2f}M / hr"
    if v >= 1_000.0:
        return f"{v / 1_000.0:.1f}K / hr"
    return f"{int(round(v))} / hr"


def _job_token_estimate(job: dict) -> float | None:
    result = job.get("result")
    totals = _extract_numeric_values_by_keys(result, {"total_tokens"}, max_count=16)
    if totals:
        return max(0.0, max(totals))
    prompts = _extract_numeric_values_by_keys(result, {"prompt_tokens", "input_tokens"}, max_count=16)
    completions = _extract_numeric_values_by_keys(result, {"completion_tokens", "output_tokens"}, max_count=16)
    if prompts or completions:
        return max(0.0, (max(prompts) if prompts else 0.0) + (max(completions) if completions else 0.0))
    return None


def _extract_coverage_values(obj: object, *, max_count: int = 256) -> list[float]:
    keys = {
        "final_cov",
        "max_cov",
        "coverage",
        "cov",
        "coverage_percent",
        "line_coverage",
        "function_coverage",
        "branch_coverage",
    }
    raw_vals = _extract_numeric_values_by_keys(obj, keys, max_count=max_count)
    out: list[float] = []
    for value in raw_vals:
        if value < 0:
            continue
        if value <= 1.0:
            out.append(value * 100.0)
            continue
        if value <= 100.0:
            out.append(value)
    return out


def _system_status() -> dict:
    now = time.time()
    with _JOBS_LOCK:
        jobs = list(_JOBS.values())
    bucketed = [_status_for_counter(str(j.get("status") or "")) for j in jobs]
    counts = {
        "total": len(jobs),
        "queued": sum(1 for b in bucketed if b == "queued"),
        "running": sum(1 for b in bucketed if b == "running"),
        "success": sum(1 for b in bucketed if b == "success"),
        "error": sum(1 for b in bucketed if b == "error"),
    }
    counts_by_kind: dict[str, int] = {}
    for j in jobs:
        k = str(j.get("kind") or "unknown")
        counts_by_kind[k] = counts_by_kind.get(k, 0) + 1
    active = [
        {
            "job_id": j.get("job_id"),
            "status": j.get("status"),
            "repo": j.get("repo"),
            "updated_at": j.get("updated_at"),
            "kind": j.get("kind"),
        }
        for j in jobs
        if _status_for_counter(str(j.get("status") or "")) in {"queued", "running"}
    ]
    cfg = _cfg_get()
    log_dir = _JOB_LOGS_DIR
    memory = _memory_status()
    fuzz_jobs = [j for j in jobs if str(j.get("kind") or "") == "fuzz"]
    task_jobs = [j for j in jobs if str(j.get("kind") or "") == "task"]

    def _status_counts(rows: list[dict]) -> dict[str, int]:
        return {
            "total": len(rows),
            "queued": sum(1 for j in rows if _status_bucket(str(j.get("status") or "")) == "queued"),
            "running": sum(1 for j in rows if _status_bucket(str(j.get("status") or "")) == "running"),
            "success": sum(1 for j in rows if _status_bucket(str(j.get("status") or "")) == "success"),
            "error": sum(1 for j in rows if _status_bucket(str(j.get("status") or "")) == "error"),
        }

    task_counts = _status_counts(task_jobs)
    fuzz_counts = _status_counts(fuzz_jobs)

    finished_fuzz = [j for j in fuzz_jobs if _status_bucket(str(j.get("status") or "")) in {"success", "error"}]
    success_fuzz = [j for j in finished_fuzz if _status_bucket(str(j.get("status") or "")) == "success"]
    error_fuzz = [j for j in finished_fuzz if _status_bucket(str(j.get("status") or "")) == "error"]
    finished_total = len(finished_fuzz)
    success_rate = (float(len(success_fuzz)) / float(finished_total) * 100.0) if finished_total > 0 else None
    failure_rate = (float(len(error_fuzz)) / float(finished_total) * 100.0) if finished_total > 0 else None

    durations = [d for d in (_job_duration_seconds(j) for j in finished_fuzz) if d is not None]
    avg_fuzz_seconds = (sum(durations) / float(len(durations))) if durations else None

    window_sec = 3600.0
    curr_start = now - window_sec
    prev_start = now - (2.0 * window_sec)
    current_finished = [
        j for j in finished_fuzz
        if (ts := _safe_float(j.get("finished_at"))) is not None and curr_start <= ts < now
    ]
    previous_finished = [
        j for j in finished_fuzz
        if (ts := _safe_float(j.get("finished_at"))) is not None and prev_start <= ts < curr_start
    ]
    current_failure_rate = (
        float(sum(1 for j in current_finished if _status_bucket(str(j.get("status") or "")) == "error"))
        / float(len(current_finished))
        * 100.0
    ) if current_finished else None
    previous_failure_rate = (
        float(sum(1 for j in previous_finished if _status_bucket(str(j.get("status") or "")) == "error"))
        / float(len(previous_finished))
        * 100.0
    ) if previous_finished else None
    current_health = (100.0 - current_failure_rate) if current_failure_rate is not None else None
    previous_health = (100.0 - previous_failure_rate) if previous_failure_rate is not None else None

    current_errors = sum(
        1
        for j in error_fuzz
        if (ts := _safe_float(j.get("finished_at"))) is not None and curr_start <= ts < now
    )
    previous_errors = sum(
        1
        for j in error_fuzz
        if (ts := _safe_float(j.get("finished_at"))) is not None and prev_start <= ts < curr_start
    )

    now_24h = now - 86400.0
    success_24h = sum(
        1
        for j in success_fuzz
        if (ts := _safe_float(j.get("finished_at"))) is not None and ts >= now_24h
    )
    previous_success_24h = sum(
        1
        for j in success_fuzz
        if (ts := _safe_float(j.get("finished_at"))) is not None and (now_24h - 86400.0) <= ts < now_24h
    )

    coverage_values: list[float] = []
    for j in fuzz_jobs:
        coverage_values.extend(_extract_coverage_values(j.get("result"), max_count=24))
    avg_coverage = (sum(coverage_values) / float(len(coverage_values))) if coverage_values else None

    running_fuzz = sum(1 for j in fuzz_jobs if _status_bucket(str(j.get("status") or "")) == "running")

    cgroup_ratio = _safe_float(memory.get("cgroup_usage_ratio"))
    cluster_load_pct = (max(0.0, min(100.0, cgroup_ratio * 100.0)) if cgroup_ratio is not None else None)
    http_metrics = _http_metrics_snapshot(now=now, window_sec=300.0)
    http_error_ratio = _safe_float(http_metrics.get("error_ratio_pct"))
    http_error_ratio_prev = _safe_float(http_metrics.get("error_ratio_prev_pct"))
    http_qps = _safe_float(http_metrics.get("qps"))
    http_p95 = _safe_float(http_metrics.get("lat_p95_ms"))

    performance_series = _performance_series_from_jobs(now, fuzz_jobs)
    recent_exec_rates = _collect_exec_rates_for_system(fuzz_jobs, now)
    recent_jobs_per_sec = (sum(recent_exec_rates) / 1000.0) if recent_exec_rates else None

    token_window_sec = 3600.0
    token_cutoff = now - token_window_sec
    token_sum = 0.0
    token_count = 0
    for j in fuzz_jobs:
        ts = _safe_float(j.get("updated_at")) or _safe_float(j.get("finished_at")) or _safe_float(j.get("created_at"))
        if ts is None or ts < token_cutoff:
            continue
        est = _job_token_estimate(j)
        if est is None:
            continue
        token_sum += est
        token_count += 1
    tokens_per_hour = (token_sum / (token_window_sec / 3600.0)) if token_count > 0 else None

    agent_health_matrix: list[int] = []
    latest_fuzz = sorted(
        fuzz_jobs,
        key=lambda j: float(_safe_float(j.get("updated_at")) or 0.0),
        reverse=True,
    )[:32]
    for j in latest_fuzz:
        agent_health_matrix.append(0 if _status_bucket(str(j.get("status") or "")) == "error" else 1)

    signal_points: list[float] = []
    if current_health is not None:
        signal_points.append(current_health)
    if cluster_load_pct is not None:
        signal_points.append(max(0.0, 100.0 - cluster_load_pct))
    if http_error_ratio is not None:
        signal_points.append(max(0.0, 100.0 - http_error_ratio))
    composite_health = (sum(signal_points) / float(len(signal_points))) if signal_points else None
    previous_signal_points: list[float] = []
    if previous_health is not None:
        previous_signal_points.append(previous_health)
    if cluster_load_pct is not None:
        previous_signal_points.append(max(0.0, 100.0 - cluster_load_pct))
    if http_error_ratio_prev is not None:
        previous_signal_points.append(max(0.0, 100.0 - http_error_ratio_prev))
    composite_health_prev = (
        (sum(previous_signal_points) / float(len(previous_signal_points))) if previous_signal_points else None
    )

    gateway_sli = None
    if http_error_ratio is not None:
        gateway_sli = max(0.0, min(100.0, 100.0 - http_error_ratio))
    fastapi_status = "UP"
    if http_error_ratio is not None and http_error_ratio >= 5.0:
        fastapi_status = "DEGRADED"
    if http_error_ratio is not None and http_error_ratio >= 20.0:
        fastapi_status = "ERROR"

    overview = {
        "avg_fuzz_time": _format_duration_human(avg_fuzz_seconds),
        "active_agents": str(running_fuzz),
        "cluster_health": _format_percent(composite_health, 1),
        "cluster_health_trend": _format_trend(composite_health, composite_health_prev, unit_suffix="%"),
        "crash_triage_rate": str(current_errors),
        "crash_triage_rate_trend": _format_trend(float(current_errors), float(previous_errors), unit_suffix=""),
        "harnesses_synthesized": str(success_24h),
        "harnesses_synthesized_trend": _format_trend(float(success_24h), float(previous_success_24h), unit_suffix=""),
        "avg_coverage": _format_percent(avg_coverage, 2),
        "avg_coverage_trend": None,
        "main_tasks_running": str(task_counts["running"]),
        "main_tasks_queued": str(task_counts["queued"]),
        "child_jobs_running": str(fuzz_counts["running"]),
        "child_jobs_queued": str(fuzz_counts["queued"]),
    }
    telemetry = {
        "llm_token_usage": _format_tokens_per_hour(tokens_per_hour),
        "llm_token_status": ("Active" if token_count > 0 else "--"),
        "k8s_pod_capacity": (f"{cluster_load_pct:.0f}% CAP" if cluster_load_pct is not None else None),
        "k8s_pod_status": ("Normal" if cluster_load_pct is not None and cluster_load_pct < 90.0 else ("Expansion Req" if cluster_load_pct is not None else None)),
        "fastapi_gateway": (f"{gateway_sli:.2f}% SLI" if gateway_sli is not None else None),
        "fastapi_status": fastapi_status,
        "agent_health_matrix": agent_health_matrix,
        "performance_series": performance_series,
    }
    execution_summary = {
        "failure_rate": (f"{failure_rate:.2f}%" if failure_rate is not None else None),
        "fuzzing_jobs_24h": str(
            sum(
                1
                for j in fuzz_jobs
                if (ts := _safe_float(j.get("created_at"))) is not None and ts >= now_24h
            )
        ),
        "cluster_load_peak": (f"{cluster_load_pct:.0f}%" if cluster_load_pct is not None else None),
        "repos_queued": str(task_counts["queued"]),
        "avg_triage_time_ms": None,
        "success_ratio": (f"{success_rate:.2f}" if success_rate is not None else None),
        "main_tasks_running": str(task_counts["running"]),
        "main_tasks_queued": str(task_counts["queued"]),
        "child_jobs_running": str(fuzz_counts["running"]),
        "child_jobs_queued": str(fuzz_counts["queued"]),
    }
    task_finished = [
        j for j in task_jobs
        if _status_bucket(str(j.get("status") or "")) in {"success", "error"}
    ]
    task_success = sum(1 for j in task_finished if _status_bucket(str(j.get("status") or "")) == "success")
    task_success_rate = (
        float(task_success) / float(len(task_finished)) * 100.0 if task_finished else None
    )
    task_failed = sum(1 for j in task_jobs if _status_bucket(str(j.get("status") or "")) == "error")
    tasks_tab_metrics = {
        "total_jobs": str(len(task_jobs)),
        "execs_per_sec": (f"{recent_jobs_per_sec:.1f}" if recent_jobs_per_sec is not None else None),
        "success_rate": (f"{task_success_rate:.1f}" if task_success_rate is not None else None),
        "failed_tasks": str(task_failed),
    }
    return {
        "ok": True,
        "server_time": now,
        "server_time_iso": _iso_time(now),
        "uptime_sec": now - _APP_START,
        "jobs": counts,
        "jobs_by_kind": counts_by_kind,
        "workers": {
            "max": _MAX_WORKERS,
        },
        "active_jobs": active[:8],
        "logs": {
            "dir": str(log_dir),
            "exists": log_dir.exists(),
            "size_bytes": _dir_size(log_dir),
        },
        "memory": memory,
        "config": {
            "runtime_mode": "native",
            "oss_fuzz_dir": cfg.oss_fuzz_dir,
            "openai_base_url": cfg.openai_base_url,
            "openai_api_key_set": bool(cfg.openai_api_key),
            "openrouter_model": cfg.openrouter_model,
        },
        "overview": overview,
        "telemetry": telemetry,
        "execution": {"summary": execution_summary},
        "tasks_tab_metrics": tasks_tab_metrics,
    }


def _metrics_payload() -> str:
    now = time.time()
    window_sec = max(60, int(os.environ.get("SHERPA_METRICS_FAILURE_WINDOW_SEC", "3600")))
    cutoff = now - float(window_sec)
    with _JOBS_LOCK:
        jobs = [dict(j) for j in _JOBS.values()]

    bucketed = [_status_for_counter(str(j.get("status") or "")) for j in jobs]
    status_counts = {
        "queued": sum(1 for b in bucketed if b == "queued"),
        "running": sum(1 for b in bucketed if b == "running"),
        "success": sum(1 for b in bucketed if b == "success"),
        "error": sum(1 for b in bucketed if b == "error"),
    }
    recoverable = sum(1 for j in jobs if str(j.get("status") or "").strip().lower() == "recoverable")

    finished_in_window = []
    for j in jobs:
        ts_raw = j.get("finished_at")
        try:
            ts = float(ts_raw)
        except Exception:
            continue
        if ts >= cutoff:
            finished_in_window.append(j)

    failed_in_window = [
        j
        for j in finished_in_window
        if _status_for_counter(str(j.get("status") or "")) == "error"
    ]
    finished_total = len(finished_in_window)
    failed_total = len(failed_in_window)
    failure_rate = (failed_total / finished_total) if finished_total > 0 else 0.0
    memory = _memory_status()

    lines = [
        "# HELP sherpa_jobs_total Total jobs in memory.",
        "# TYPE sherpa_jobs_total gauge",
        f"sherpa_jobs_total {len(jobs)}",
        "# HELP sherpa_jobs_status Jobs by status bucket.",
        "# TYPE sherpa_jobs_status gauge",
        f'sherpa_jobs_status{{status="queued"}} {status_counts["queued"]}',
        f'sherpa_jobs_status{{status="running"}} {status_counts["running"]}',
        f'sherpa_jobs_status{{status="success"}} {status_counts["success"]}',
        f'sherpa_jobs_status{{status="error"}} {status_counts["error"]}',
        "# HELP sherpa_jobs_recoverable_total Recoverable jobs.",
        "# TYPE sherpa_jobs_recoverable_total gauge",
        f"sherpa_jobs_recoverable_total {recoverable}",
        f"# HELP sherpa_jobs_finished_window_total Jobs finished in last {window_sec} seconds.",
        "# TYPE sherpa_jobs_finished_window_total gauge",
        f"sherpa_jobs_finished_window_total {finished_total}",
        f"# HELP sherpa_jobs_failed_window_total Failed jobs finished in last {window_sec} seconds.",
        "# TYPE sherpa_jobs_failed_window_total gauge",
        f"sherpa_jobs_failed_window_total {failed_total}",
        f"# HELP sherpa_jobs_failure_rate_window Failure rate in last {window_sec} seconds.",
        "# TYPE sherpa_jobs_failure_rate_window gauge",
        f"sherpa_jobs_failure_rate_window {failure_rate:.6f}",
    ]
    rss_bytes = memory.get("process_rss_bytes")
    if isinstance(rss_bytes, int):
        lines.extend(
            [
                "# HELP sherpa_process_resident_memory_bytes Resident memory size of the web process.",
                "# TYPE sherpa_process_resident_memory_bytes gauge",
                f"sherpa_process_resident_memory_bytes {rss_bytes}",
            ]
        )
    cgroup_current = memory.get("cgroup_current_bytes")
    if isinstance(cgroup_current, int):
        lines.extend(
            [
                "# HELP sherpa_cgroup_memory_current_bytes Current cgroup memory usage.",
                "# TYPE sherpa_cgroup_memory_current_bytes gauge",
                f"sherpa_cgroup_memory_current_bytes {cgroup_current}",
            ]
        )
    cgroup_limit = memory.get("cgroup_limit_bytes")
    if isinstance(cgroup_limit, int):
        lines.extend(
            [
                "# HELP sherpa_cgroup_memory_limit_bytes Effective cgroup memory limit.",
                "# TYPE sherpa_cgroup_memory_limit_bytes gauge",
                f"sherpa_cgroup_memory_limit_bytes {cgroup_limit}",
            ]
        )
    usage_ratio = memory.get("cgroup_usage_ratio")
    if isinstance(usage_ratio, (int, float)):
        lines.extend(
            [
                "# HELP sherpa_cgroup_memory_usage_ratio Current cgroup memory usage divided by limit.",
                "# TYPE sherpa_cgroup_memory_usage_ratio gauge",
                f"sherpa_cgroup_memory_usage_ratio {usage_ratio:.6f}",
            ]
        )
    oom_kill_count = memory.get("oom_kill_count")
    if isinstance(oom_kill_count, int):
        lines.extend(
            [
                "# HELP sherpa_cgroup_memory_oom_kill_total Total OOM kills reported by the current cgroup.",
                "# TYPE sherpa_cgroup_memory_oom_kill_total gauge",
                f"sherpa_cgroup_memory_oom_kill_total {oom_kill_count}",
            ]
        )
    return "\n".join(lines) + "\n"

class fuzz_model(BaseModel):
    code_url: str
    email: str | None = None
    model: str | None = None
    temperature: float = 0.5
    timeout: int = 10
    max_tokens: int = 0
    time_budget: int | None = None
    total_time_budget: int | None = None
    run_time_budget: int | None = None
    # Frontend-local compatibility fields
    total_duration: int | None = None
    single_duration: int | None = None
    unlimited_round_limit: int | None = None
    docker: bool | None = None
    docker_image: str | None = None


class task_model(BaseModel):
    jobs: list[fuzz_model]
    auto_init: bool = True
    build_images: bool = True
    images: list[str] | None = None
    force_build: bool = False
    oss_fuzz_repo_url: str | None = None
    force_clone: bool = False


class provider_models_request(BaseModel):
    api_key: str | None = None
    base_url: str | None = None


def _resolve_job_docker_policy(request: fuzz_model, cfg: WebPersistentConfig) -> tuple[bool, str]:
    if _executor_mode() == "k8s_job":
        # Native runtime baseline: keep API fields for compatibility, but
        # runtime no longer depends on docker flags/image in k8s mode.
        return False, ""
    docker_enabled = request.docker if request.docker is not None else cfg.fuzz_use_docker
    docker_image_value = (
        (request.docker_image or "").strip()
        or (cfg.fuzz_docker_image or "").strip()
        or "auto"
    )
    return bool(docker_enabled), docker_image_value


def _normalize_budget_value(raw: int | None, *, field_name: str) -> int:
    if raw is None:
        raise RuntimeError(f"{field_name} is required")
    value = int(raw)
    if value == -1:
        return 0
    if value < 0:
        raise RuntimeError(f"{field_name} must be >= 0 or -1 for unlimited")
    return value


def _normalize_round_limit_value(raw: int | None, *, fallback: int) -> int:
    if raw is None:
        value = int(fallback)
    else:
        value = int(raw)
    if value == -1:
        return 0
    if value < 0:
        raise RuntimeError("unlimited_round_limit must be >= 0 or -1 for unlimited")
    return value


def _enforce_docker_only(jobs: list[fuzz_model], cfg: WebPersistentConfig) -> None:
    # Native runtime baseline: keep for compatibility hook, no hard enforcement.
    return None


@app.get("/api/config")
def get_config():
    cfg = _cfg_get()
    return as_public_dict(cfg)


@app.get("/api/opencode/providers/{provider}/models")
def get_opencode_provider_models(provider: str):
    cfg = _cfg_get()
    normalized, models, source, warning = list_opencode_provider_models_resolved(provider, cfg)
    if not normalized:
        raise HTTPException(status_code=400, detail="provider is required")
    if not models:
        raise HTTPException(status_code=404, detail=f"unsupported provider: {provider}")
    payload: dict[str, object] = {
        "provider": normalized,
        "models": models,
        "source": source,
    }
    if warning:
        payload["warning"] = warning
    return payload


@app.post("/api/opencode/providers/{provider}/models")
def post_opencode_provider_models(provider: str, request: provider_models_request = Body(...)):
    cfg = _cfg_get()
    normalized, models, source, warning = list_opencode_provider_models_resolved(
        provider,
        cfg,
        api_key_override=request.api_key,
        base_url_override=request.base_url,
    )
    if not normalized:
        raise HTTPException(status_code=400, detail="provider is required")
    if not models:
        raise HTTPException(status_code=404, detail=f"unsupported provider: {provider}")
    payload: dict[str, object] = {
        "provider": normalized,
        "models": models,
        "source": source,
    }
    if warning:
        payload["warning"] = warning
    return payload


@app.put("/api/config")
def put_config(request: dict = Body(...)):
    if not isinstance(request, dict):
        raise HTTPException(status_code=400, detail="config payload must be a JSON object")

    current = _cfg_get()
    payload = current.model_dump()
    lightweight_only_keys = {
        "apiBaseUrl",
        "api_base_url",
        "sherpa_run_plateau_idle_growth_sec",
    }
    request_keys = set(request.keys())
    is_lightweight_update = bool(request_keys) and request_keys.issubset(lightweight_only_keys)

    if is_lightweight_update:
        api_base_url = str(request.get("apiBaseUrl") or request.get("api_base_url") or "").strip()
        payload["api_base_url"] = api_base_url
        if "sherpa_run_plateau_idle_growth_sec" in request:
            payload["sherpa_run_plateau_idle_growth_sec"] = request.get("sherpa_run_plateau_idle_growth_sec")
    else:
        merged = dict(payload)
        for key, value in request.items():
            if key == "apiBaseUrl":
                merged["api_base_url"] = value
            else:
                merged[key] = value
        try:
            validated = WebPersistentConfig(**merged)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid config payload: {exc}") from exc
        payload = validated.model_dump()

    candidate = WebPersistentConfig(**payload)
    if int(candidate.fuzz_time_budget) < 0:
        raise HTTPException(
            status_code=400,
            detail="fuzz_time_budget must be >= 0 (0 means unlimited).",
        )
    if int(candidate.sherpa_run_unlimited_round_budget_sec) < 0:
        raise HTTPException(
            status_code=400,
            detail="sherpa_run_unlimited_round_budget_sec must be >= 0 (0 means fully unlimited).",
        )
    plateau_idle = int(candidate.sherpa_run_plateau_idle_growth_sec)
    if plateau_idle < 30 or plateau_idle > 86_400:
        raise HTTPException(
            status_code=400,
            detail="sherpa_run_plateau_idle_growth_sec must be in [30, 86400].",
        )

    # Frontend no longer controls provider/API fields.
    controlled_fields = (
        "openai_api_key",
        "openai_base_url",
        "openai_model",
        "opencode_model",
        "opencode_providers",
        "openrouter_api_key",
        "openrouter_base_url",
        "openrouter_model",
    )
    for key in controlled_fields:
        payload[key] = getattr(current, key)

    # Native runtime baseline in k8s mode; keep fields for compatibility only.
    payload["fuzz_use_docker"] = False
    payload["fuzz_docker_image"] = ""
    runtime_cfg = apply_minimax_env_source(WebPersistentConfig(**payload))

    # Keep persisted config free of API secrets; runtime values come from environment.
    persisted_cfg = WebPersistentConfig(**runtime_cfg.model_dump())
    persisted_cfg.openai_api_key = None
    persisted_cfg.openrouter_api_key = None
    for item in persisted_cfg.opencode_providers:
        item.api_key = None
        item.clear_api_key = False

    try:
        save_config(persisted_cfg)
        _cfg_set(runtime_cfg)
        apply_config_to_env(runtime_cfg)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"保存配置失败: {exc}") from exc
    return {"ok": True}


@app.get("/api/system")
def get_system_status():
    return _system_status()


@app.get("/api/metrics")
def get_metrics():
    return Response(content=_metrics_payload(), media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/api/health")
def health_check():
    return {"ok": True}


def _create_job(kind: str, repo: str | None = None) -> str:
    job_id = uuid.uuid4().hex
    now = time.time()
    job_payload: dict | None = None
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "kind": kind,
            "status": "queued",
            "repo": repo,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "result": None,
            "log": "",
            "log_file": None,
            "cancel_requested": False,
            "last_cancel_requested_at": None,
            "runtime_mode": _runtime_mode_for_job(None),
        }
        job_payload = dict(_JOBS[job_id])
    if job_payload is not None:
        _persist_job_state(job_payload)
    return job_id


def _ensure_oss_fuzz_checkout(*, repo_url: str, target_dir: Path, force: bool) -> None:
    if target_dir.is_dir() and (target_dir / "infra" / "helper.py").is_file():
        print("[init] oss-fuzz already present")
        return

    auto_repair = (os.environ.get("SHERPA_OSS_FUZZ_AUTO_REPAIR", "1") or "").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    should_repair = bool(force or auto_repair)

    if target_dir.exists():
        if not should_repair:
            raise RuntimeError(
                f"oss-fuzz dir exists but invalid (missing infra/helper.py): {target_dir}"
            )
        print(
            "[init] oss-fuzz directory exists but invalid; "
            f"auto-repair enabled, resetting: {target_dir}"
        )
        for child in target_dir.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink(missing_ok=True)
            except Exception:
                pass
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", repo_url, str(target_dir)]
    print("[init] " + " ".join(cmd))
    subprocess.check_call(cmd)
    if not (target_dir / "infra" / "helper.py").is_file():
        raise RuntimeError("oss-fuzz clone completed but infra/helper.py not found")


def _ensure_docker_image(image: str, dockerfile: Path, *, force: bool) -> None:
    if not force:
        try:
            probe = subprocess.run(
                ["docker", "image", "inspect", image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                text=True,
            )
            if probe.returncode == 0:
                return
        except FileNotFoundError:
            raise RuntimeError("docker not found in PATH")
        except Exception:
            pass

    if not dockerfile.is_file():
        raise RuntimeError(f"Dockerfile not found: {dockerfile}")

    def _wait_for_docker_daemon() -> None:
        max_wait_s = 45
        deadline = time.time() + max_wait_s
        while True:
            try:
                probe = subprocess.run(
                    ["docker", "info"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    text=True,
                )
                if probe.returncode == 0:
                    return
            except Exception:
                pass
            if time.time() >= deadline:
                raise RuntimeError("docker daemon not ready")
            time.sleep(2)

    def _docker_daemon_unreachable(output: str) -> bool:
        needles = [
            "Cannot connect to the Docker daemon",
            "dial tcp",
            "no such host",
            "Error response from daemon: dial tcp",
        ]
        return any(n in output for n in needles)

    def _buildkit_unavailable(output: str) -> bool:
        needles = [
            "BuildKit is enabled but the buildx component is missing",
            "buildx component is missing or broken",
        ]
        return any(n in output for n in needles)

    def _run_build(cmd: list[str], *, buildkit: str | None = None) -> tuple[int, str]:
        print("[init] " + " ".join(cmd))
        env = os.environ.copy()
        if buildkit is not None:
            env["DOCKER_BUILDKIT"] = buildkit
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )

        output_chunks: list[str] = []
        line_q: queue.Queue[str | None] = queue.Queue()

        def _reader() -> None:
            try:
                if proc.stdout is None:
                    return
                for line in proc.stdout:
                    line_q.put(line)
            finally:
                line_q.put(None)

        th = threading.Thread(target=_reader, daemon=True)
        th.start()

        last_heartbeat = time.monotonic()
        while True:
            try:
                item = line_q.get(timeout=0.2)
            except queue.Empty:
                item = ""

            if item is None:
                break

            if item:
                output_chunks.append(item)
                print(item, end="")

            now = time.monotonic()
            if now - last_heartbeat >= 10:
                print("[init] docker build still running...")
                last_heartbeat = now

            if proc.poll() is not None and line_q.empty():
                break

        try:
            th.join(timeout=1)
        except Exception:
            pass

        rc = proc.wait() if proc.poll() is None else int(proc.returncode or 0)
        return rc, "".join(output_chunks)

    build_cmds = [
        ["docker", "build", "--progress=plain", "-t", image, "-f", str(dockerfile), str(_REPO_ROOT)],
        ["docker", "build", "-t", image, "-f", str(dockerfile), str(_REPO_ROOT)],
    ]

    max_attempts = 5
    backoff = 3.0
    last_output = ""
    last_rc = 1

    for attempt in range(1, max_attempts + 1):
        try:
            _wait_for_docker_daemon()
        except Exception:
            if attempt == max_attempts:
                raise
        retry_outer = False
        for cmd in build_cmds:
            rc, output = _run_build(cmd)
            last_rc = rc
            last_output = output
            if rc == 0:
                return
            if "unknown flag: --progress" in output:
                # Try without --progress on older Docker.
                continue
            if _buildkit_unavailable(output):
                print("[init] buildx unavailable; retrying docker build with classic builder (DOCKER_BUILDKIT=0)")
                legacy_cmd = [arg for arg in cmd if not arg.startswith("--progress=")]
                rc2, output2 = _run_build(legacy_cmd, buildkit="0")
                last_rc = rc2
                last_output = output2
                if rc2 == 0:
                    return
                if _docker_daemon_unreachable(output2) and attempt < max_attempts:
                    print(f"[init] docker daemon not ready; retrying in {backoff:.0f}s (attempt {attempt}/{max_attempts})")
                    time.sleep(backoff)
                    backoff *= 2
                    retry_outer = True
                    break
                # Keep trying other build command variants before failing.
                continue
            if _docker_daemon_unreachable(output) and attempt < max_attempts:
                print(f"[init] docker daemon not ready; retrying in {backoff:.0f}s (attempt {attempt}/{max_attempts})")
                time.sleep(backoff)
                backoff *= 2
                retry_outer = True
                break
        if retry_outer:
            continue
        raise RuntimeError(f"docker build failed (rc={last_rc}) for {image}")

    raise RuntimeError(f"docker build failed after retries for {image}. Last output:\n{last_output}")


def _job_snapshot(job_id: str) -> dict | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        view = dict(job)
    _hydrate_job_log_from_disk(view)
    return view


def _enrich_job_view(view: dict) -> None:
    """Add workflow/resume/cancel tracking fields and fuzz metrics to a job API view."""
    # -- workflow & resume tracking --
    view.setdefault("k8s_phase", None)
    view.setdefault("cancel_requested", False)
    view.setdefault("last_cancel_requested_at", None)
    view.setdefault("workflow_active_step", None)
    view.setdefault("workflow_last_step", None)
    view.setdefault("workflow_last_step_ts", None)
    view.setdefault("parent_id", None)
    view.setdefault("recoverable", None)
    view.setdefault("resume_attempts", 0)
    view.setdefault("resume_error_code", None)
    view.setdefault("resume_from_step", None)
    view.setdefault("last_resume_reason", None)
    view.setdefault("last_resume_requested_at", None)
    view.setdefault("last_resume_started_at", None)
    view.setdefault("last_resume_finished_at", None)
    view.setdefault("last_interrupted_at", None)
    view.setdefault("request", None)
    view.setdefault("analysis_companion_pod", None)
    view.setdefault("analysis_companion_service", None)
    view.setdefault("analysis_companion_url", "")
    view.setdefault("analysis_companion_ready", False)
    view.setdefault("analysis_companion_active", False)
    view.setdefault("analysis_companion_error", None)
    view.setdefault("analysis_companion_last_error", None)
    view.setdefault("analysis_companion_stopped_at", None)
    view.setdefault("analysis_companion_state", "")
    view.setdefault("analysis_companion_backend", "")
    view.setdefault("analysis_companion_candidate_count", 0)
    view.setdefault("analysis_companion_updated_at", "")
    view.setdefault("analysis_companion_repo_root", "")
    view.setdefault("analysis_companion_status_error", "")
    view.setdefault("analysis_companion_preprocess_path", "")
    view.setdefault("analysis_companion_coverage_hints_path", "")
    view.setdefault("analysis_companion_rag_ok", False)
    view.setdefault("analysis_companion_rag_knowledge_base_path", "")
    view.setdefault("analysis_companion_rag_document_count", 0)
    view.setdefault("analysis_companion_rag_chunk_count", 0)
    view.setdefault("analysis_companion_embedding_provider", "openrouter")
    view.setdefault("analysis_companion_embedding_model", "")
    view.setdefault("analysis_companion_embedding_ok", False)
    view.setdefault("analysis_companion_rag_degraded", False)
    view.setdefault("analysis_companion_rag_degraded_reason", "")
    view.setdefault("analysis_companion_semantic_query_count", 0)
    view.setdefault("analysis_companion_semantic_hit_count", 0)
    view.setdefault("analysis_companion_semantic_hit_rate", 0.0)
    view.setdefault("analysis_companion_cache_hit_rate", 0.0)
    # -- per-fuzzer performance metrics --
    view.setdefault("fuzz_metrics", None)
    view.setdefault("fuzz_metrics_ts", None)
    view.setdefault("fuzz_fuzzers", {})
    view.setdefault("fuzz_max_cov", 0)
    view.setdefault("fuzz_max_ft", 0)
    view.setdefault("fuzz_total_execs_per_sec", 0)
    view.setdefault("fuzz_crash_found", False)
    view.setdefault("fuzz_coverage_history", [])
    view.setdefault("fuzz_coverage_source_report", {})
    view.setdefault("fuzz_coverage_loop_round", 0)
    view.setdefault("fuzz_coverage_loop_max_rounds", 0)
    view.setdefault("fuzz_coverage_plateau_streak", 0)
    view.setdefault("fuzz_coverage_seed_profile", "")
    view.setdefault("fuzz_coverage_quality_flags", [])

    companion_status = _analysis_companion_status_for_job(str(view.get("id") or ""))
    if companion_status:
        view["analysis_companion_state"] = str(companion_status.get("state") or "")
        view["analysis_companion_backend"] = str(companion_status.get("analysis_backend") or "")
        view["analysis_companion_url"] = str(companion_status.get("mcp_url") or view.get("analysis_companion_url") or "")
        view["analysis_companion_ready"] = bool(companion_status.get("mcp_ready"))
        try:
            view["analysis_companion_candidate_count"] = int(companion_status.get("candidate_count") or 0)
        except Exception:
            view["analysis_companion_candidate_count"] = 0
        view["analysis_companion_updated_at"] = str(companion_status.get("updated_at") or "")
        view["analysis_companion_repo_root"] = str(companion_status.get("repo_root") or "")
        view["analysis_companion_status_error"] = str(companion_status.get("error") or "")
        view["analysis_companion_last_error"] = str(
            companion_status.get("last_error")
            or companion_status.get("error")
            or ""
        )
        view["analysis_companion_preprocess_path"] = str(companion_status.get("preprocess_path") or "")
        view["analysis_companion_coverage_hints_path"] = str(companion_status.get("coverage_hints_path") or "")
        view["analysis_companion_rag_ok"] = bool(companion_status.get("rag_ok"))
        view["analysis_companion_rag_knowledge_base_path"] = str(companion_status.get("rag_knowledge_base_path") or "")
        try:
            view["analysis_companion_rag_document_count"] = int(companion_status.get("rag_document_count") or 0)
        except Exception:
            view["analysis_companion_rag_document_count"] = 0
        try:
            view["analysis_companion_rag_chunk_count"] = int(companion_status.get("rag_chunk_count") or 0)
        except Exception:
            view["analysis_companion_rag_chunk_count"] = 0
        view["analysis_companion_embedding_provider"] = str(
            companion_status.get("embedding_provider") or "openrouter"
        )
        view["analysis_companion_embedding_model"] = str(companion_status.get("embedding_model") or "")
        view["analysis_companion_embedding_ok"] = bool(companion_status.get("embedding_ok"))
        view["analysis_companion_rag_degraded"] = bool(companion_status.get("rag_degraded"))
        view["analysis_companion_rag_degraded_reason"] = str(
            companion_status.get("rag_degraded_reason") or ""
        )
        try:
            view["analysis_companion_semantic_query_count"] = int(companion_status.get("semantic_query_count") or 0)
        except Exception:
            view["analysis_companion_semantic_query_count"] = 0
        try:
            view["analysis_companion_semantic_hit_count"] = int(companion_status.get("semantic_hit_count") or 0)
        except Exception:
            view["analysis_companion_semantic_hit_count"] = 0
        try:
            view["analysis_companion_semantic_hit_rate"] = float(companion_status.get("semantic_hit_rate") or 0.0)
        except Exception:
            view["analysis_companion_semantic_hit_rate"] = 0.0
        try:
            view["analysis_companion_cache_hit_rate"] = float(companion_status.get("cache_hit_rate") or 0.0)
        except Exception:
            view["analysis_companion_cache_hit_rate"] = 0.0


def _derive_task_status(job: dict) -> dict:
    children = list(job.get("children") or [])
    if not children:
        view = dict(job)
        err = _error_object_for_job(view)
        view["error"] = err
        view["error_code"] = str(err.get("code") or "")
        view["error_kind"] = str(err.get("kind") or "")
        view["error_signature"] = str(err.get("signature") or "")
        view["phase"] = _phase_for_job(view)
        view["runtime_mode"] = _runtime_mode_for_job(view)
        _enrich_job_view(view)
        return view
    child_jobs = []
    with _JOBS_LOCK:
        for cid in children:
            cjob = _JOBS.get(cid)
            if cjob:
                child_view = dict(cjob)
                _hydrate_job_log_from_disk(child_view)
                child_jobs.append(child_view)
    total = len(child_jobs)
    buckets = [_status_for_parent(str(j.get("status") or "")) for j in child_jobs]
    queued = sum(1 for b in buckets if b == "queued")
    running = sum(1 for b in buckets if b == "running")
    success = sum(1 for b in buckets if b == "success")
    error = sum(1 for b in buckets if b == "error")
    if total == 0:
        derived = str(job.get("status") or "queued")
    elif queued or running:
        derived = "running"
    elif error:
        derived = "error"
    else:
        derived = "success"
    view = dict(job)
    view["status"] = derived
    view["children_status"] = {
        "total": total,
        "queued": queued,
        "running": running,
        "success": success,
        "error": error,
    }
    for c in child_jobs:
        cerr = _error_object_for_job(c)
        c["error"] = cerr
        c["error_code"] = str(cerr.get("code") or "")
        c["error_kind"] = str(cerr.get("kind") or "")
        c["error_signature"] = str(cerr.get("signature") or "")
        c["phase"] = _phase_for_job(c)
        c["runtime_mode"] = _runtime_mode_for_job(c)
        _enrich_job_view(c)
    view["children"] = child_jobs
    err = _error_object_for_job(view)
    view["error"] = err
    view["error_code"] = str(err.get("code") or "")
    view["error_kind"] = str(err.get("kind") or "")
    view["error_signature"] = str(err.get("signature") or "")
    view["phase"] = _phase_for_job(view)
    view["runtime_mode"] = _runtime_mode_for_job(view)
    _enrich_job_view(view)
    if derived in {"success", "error"} and not job.get("finished_at"):
        done_ts = time.time()
        _job_update(job.get("job_id"), finished_at=done_ts, status=derived)
        view["finished_at"] = done_ts
        view["status"] = derived
    return view


def _derive_task_status_from_snapshot(job: dict, jobs_snapshot: dict[str, dict]) -> tuple[str, dict, list[dict]]:
    child_ids = list(job.get("children") or [])
    child_jobs = [dict(jobs_snapshot[cid]) for cid in child_ids if cid in jobs_snapshot]
    total = len(child_jobs)
    buckets = [_status_for_parent(str(j.get("status") or "")) for j in child_jobs]
    queued = sum(1 for b in buckets if b == "queued")
    running = sum(1 for b in buckets if b == "running")
    success = sum(1 for b in buckets if b == "success")
    error = sum(1 for b in buckets if b == "error")
    if total == 0:
        derived = str(job.get("status") or "queued")
    elif queued or running:
        derived = "running"
    elif error:
        derived = "error"
    else:
        derived = "success"
    return (
        derived,
        {
            "total": total,
            "queued": queued,
            "running": running,
            "success": success,
            "error": error,
        },
        child_jobs,
    )


def _list_tasks(limit: int = 50) -> list[dict]:
    capped_limit = max(1, min(int(limit), 200))
    with _JOBS_LOCK:
        jobs_snapshot = {job_id: dict(job) for job_id, job in _JOBS.items()}

    tasks: list[dict] = []
    for job in jobs_snapshot.values():
        if job.get("kind") != "task":
            continue
        derived_status, children_status, child_jobs = _derive_task_status_from_snapshot(job, jobs_snapshot)
        active_child = next(
            (c for c in child_jobs if _status_for_parent(str(c.get("status") or "")) in {"running", "queued"}),
            child_jobs[0] if child_jobs else None,
        )
        stage_value = _phase_for_job(active_child) if active_child else _phase_for_job(job)
        progress_value = _task_progress_from_children(derived_status, children_status)
        status_upper = _status_upper(derived_status)
        tasks.append(
            {
                "job_id": job.get("job_id"),
                "id": job.get("job_id"),
                "status": status_upper,
                "status_raw": derived_status,
                "stage": str(stage_value or "").upper() or "UNKNOWN",
                "repo": _task_display_repo(job),
                "repo_raw": job.get("repo"),
                "created_at": job.get("created_at"),
                "created_at_iso": _iso_time(job.get("created_at")),
                "updated_at": job.get("updated_at"),
                "updated_at_iso": _iso_time(job.get("updated_at")),
                "started_at": job.get("started_at"),
                "started_at_iso": _iso_time(job.get("started_at")),
                "finished_at": job.get("finished_at"),
                "finished_at_iso": _iso_time(job.get("finished_at")),
                "error": _error_object_for_job(job),
                "error_code": _error_code_for_job(job),
                "error_kind": _error_kind_for_job(job),
                "error_signature": _error_signature_for_job(job),
                "phase": _phase_for_job(job),
                "runtime_mode": _runtime_mode_for_job(job),
                "result": job.get("result"),
                "children_status": children_status,
                "child_count": children_status.get("total", 0),
                "progress": progress_value,
                "active_child_id": active_child.get("job_id") if active_child else None,
                "active_child_status": _status_upper(str(active_child.get("status") or "")) if active_child else None,
                "active_child_phase": _phase_for_job(active_child) if active_child else None,
                "cancel_requested": job.get("cancel_requested", False),
                "last_cancel_requested_at": job.get("last_cancel_requested_at"),
                "workflow_active_step": job.get("workflow_active_step"),
                "workflow_last_step": job.get("workflow_last_step"),
                "workflow_last_step_ts": job.get("workflow_last_step_ts"),
                "recoverable": job.get("recoverable"),
                "resume_attempts": job.get("resume_attempts", 0),
                "resume_error_code": job.get("resume_error_code"),
                "last_resume_reason": job.get("last_resume_reason"),
                "last_interrupted_at": job.get("last_interrupted_at"),
                "request": job.get("request"),
                # Per-fuzzer metrics from active child (or self for fuzz jobs)
                "fuzz_fuzzers": (active_child or job).get("fuzz_fuzzers", {}),
                "fuzz_max_cov": (active_child or job).get("fuzz_max_cov", 0),
                "fuzz_max_ft": (active_child or job).get("fuzz_max_ft", 0),
                "fuzz_total_execs_per_sec": (active_child or job).get("fuzz_total_execs_per_sec", 0),
                "fuzz_crash_found": (active_child or job).get("fuzz_crash_found", False),
                "fuzz_coverage_loop_round": (active_child or job).get("fuzz_coverage_loop_round", 0),
                "fuzz_coverage_loop_max_rounds": (active_child or job).get("fuzz_coverage_loop_max_rounds", 0),
                "fuzz_coverage_plateau_streak": (active_child or job).get("fuzz_coverage_plateau_streak", 0),
                "fuzz_coverage_seed_profile": (active_child or job).get("fuzz_coverage_seed_profile", ""),
                "fuzz_coverage_quality_flags": (active_child or job).get("fuzz_coverage_quality_flags", []),
            }
        )
    tasks.sort(key=lambda item: float(item.get("created_at") or 0.0), reverse=True)
    return tasks[:capped_limit]


def _mark_resume_failed(job_id: str, *, code: str, message: str) -> None:
    _job_update(
        job_id,
        status="resume_failed",
        recoverable=False,
        resume_error_code=code,
        error=message,
        finished_at=time.time(),
    )


def _run_fuzz_job(
    job_id: str,
    request: fuzz_model,
    cfg: WebPersistentConfig,
    *,
    resumed: bool,
    trigger: str,
    resume_from_step: str | None = None,
    resume_repo_root: str | None = None,
) -> None:
    cancel_error = "cancelled by user"
    if _is_cancel_requested(job_id):
        _job_update(
            job_id,
            status="error",
            error=cancel_error,
            recoverable=False,
            finished_at=time.time(),
        )
        return
    start_ts = time.time()
    next_status = "resuming" if resumed else "running"
    _job_update(
        job_id,
        status=next_status,
        started_at=start_ts,
        finished_at=None,
        recoverable=False,
        error=None,
        request=request.model_dump(),
        resume_from_step=(_normalize_resume_step(resume_from_step) if resumed else ""),
        resume_repo_root=(resume_repo_root or ""),
        last_resume_reason=trigger if resumed else None,
        last_resume_started_at=start_ts if resumed else None,
    )
    log_file = _job_log_path(job_id)
    _job_update(job_id, log_file=str(log_file))
    tee = _Tee(job_id, log_file=log_file)
    out_token = _ACTIVE_JOB_STDOUT_TEE.set(tee)
    err_token = _ACTIVE_JOB_STDERR_TEE.set(tee)
    companion_pod = ""
    companion_service = ""
    companion_url = ""
    try:
        print(f"[job {job_id}] start repo={request.code_url} resumed={int(resumed)} trigger={trigger}")
        if _is_cancel_requested(job_id):
            raise RuntimeError(cancel_error)
        print(f"[job {job_id}] about to dispatch k8s worker...")
        docker_enabled, docker_image_value = _resolve_job_docker_policy(request, cfg)
        total_budget_src = (
            request.total_time_budget
            if request.total_time_budget is not None
            else (
                request.total_duration
                if request.total_duration is not None
                else (request.time_budget if request.time_budget is not None else cfg.fuzz_time_budget)
            )
        )
        run_budget_src = (
            request.run_time_budget
            if request.run_time_budget is not None
            else (
                request.single_duration
                if request.single_duration is not None
                else total_budget_src
            )
        )
        total_time_budget_value = _normalize_budget_value(total_budget_src, field_name="total_time_budget")
        run_time_budget_value = _normalize_budget_value(run_budget_src, field_name="run_time_budget")
        unlimited_round_limit_value = _normalize_round_limit_value(
            request.unlimited_round_limit,
            fallback=int(cfg.sherpa_run_unlimited_round_budget_sec),
        )
        coverage_loop_max_rounds = 0
        max_fix_rounds = 0
        same_error_max_retries = 0
        total_budget_log = "unlimited" if total_time_budget_value == 0 else f"{total_time_budget_value}s"
        run_budget_log = "unlimited" if run_time_budget_value == 0 else f"{run_time_budget_value}s"
        openai_key = (
            os.environ.get("OPENAI_API_KEY")
            or cfg.openai_api_key
            or ""
        ).strip()
        opencode_model_env = (os.environ.get("OPENCODE_MODEL") or "").strip()
        openai_model = (
            os.environ.get("OPENAI_MODEL")
            or opencode_model_env
            or "deepseek-reasoner"
        ).strip()
        if openai_key:
            model_value = request.model or opencode_model_env or openai_model
        else:
            model_value = request.model or cfg.openrouter_model
        runtime_mode = "native" if _executor_mode() == "k8s_job" else "docker"
        print(
            f"[job {job_id}] params runtime={runtime_mode} "
            f"time_budget={total_budget_log} run_time_budget={run_budget_log} "
            f"unlimited_round_limit={unlimited_round_limit_value if unlimited_round_limit_value > 0 else 'unlimited'} "
            f"max_tokens={request.max_tokens} model={model_value} "
            f"coverage_loop_max_rounds={coverage_loop_max_rounds} "
            f"max_fix_rounds={max_fix_rounds} "
            f"same_error_max_retries={same_error_max_retries}"
        )
        print(f"[job {job_id}] log_file={log_file}")
        mode = _executor_mode()
        docker_image = None if mode == "k8s_job" else docker_image_value
        print(
            f"[job {job_id}] executor_mode={mode} "
            f"docker_image={docker_image if docker_image else '(native)'}"
        )
        if _is_cancel_requested(job_id):
            raise RuntimeError(cancel_error)
        try:
                start_step = _normalize_resume_step(resume_from_step) if resumed else "analysis"
                stage_results: list[dict[str, object]] = []
                stage_job_names: list[str] = []
                current_repo_root = str(resume_repo_root or "").strip()
                current_node_name = ""
                last_result: object = {}
                stage_ctx: dict[str, str] = {
                    "last_fuzzer": "",
                    "last_crash_artifact": "",
                    "re_workspace_root": "",
                    "restart_to_plan_reason": "",
                    "restart_to_plan_stage": "",
                    "restart_to_plan_error_text": "",
                    "restart_to_plan_report_path": "",
                    "run_oom_retry_count": "",
                    "run_rss_limit_mb_override": "",
                    "run_parallel_fuzzers_override": "",
                }
                current_stage = start_step
                if current_stage in {"fix_build", "fix_crash"}:
                    current_stage = "build"
                if current_stage not in _STAGED_WORKFLOW_STEPS:
                    current_stage = "analysis"
                try:
                    max_stage_dispatches = int((os.environ.get("SHERPA_STAGE_DISPATCH_MAX") or "0").strip())
                except Exception:
                    max_stage_dispatches = 0
                dispatch_count = 0
                companion_mcp_ready = False
                if mode == "k8s_job" and _k8s_analysis_companion_enabled():
                    try:
                        companion_pod, companion_service, companion_url = _k8s_start_analysis_companion(job_id)
                        if companion_pod:
                            print(
                                f"[job {job_id}] analysis companion started pod={companion_pod} "
                                f"service={companion_service or '-'}"
                            )
                            _job_update(
                                job_id,
                                analysis_companion_pod=companion_pod,
                                analysis_companion_service=companion_service,
                                analysis_companion_url=companion_url,
                                analysis_companion_active=True,
                                analysis_companion_error=None,
                            )
                            status_doc = _k8s_wait_analysis_companion_result(
                                job_id,
                                companion_pod,
                                timeout_sec=_k8s_analysis_companion_timeout_sec(),
                                require_rag=False,
                            )
                            state_txt = str(status_doc.get("state") or "").strip()
                            backend_txt = str(status_doc.get("analysis_backend") or "").strip()
                            err_txt = str(status_doc.get("error") or "").strip()
                            mcp_url_txt = str(status_doc.get("mcp_url") or companion_url).strip()
                            mcp_ready = _analysis_companion_is_ready(status_doc, require_rag=False)
                            print(
                                f"[job {job_id}] analysis companion ready "
                                f"state={state_txt or '-'} backend={backend_txt or '-'} "
                                f"mcp_ready={int(mcp_ready)}"
                            )
                            _job_update(
                                job_id,
                                analysis_companion_url=mcp_url_txt or companion_url,
                                analysis_companion_ready=mcp_ready,
                                analysis_companion_error=(err_txt or None),
                            )
                            companion_mcp_ready = bool(mcp_ready)
                    except Exception as e:
                        print(f"[job {job_id}] analysis companion failed (continuing): {e}")
                        _job_update(
                            job_id,
                            analysis_companion_error=str(e),
                            analysis_companion_active=False,
                            analysis_companion_ready=False,
                        )
                        companion_mcp_ready = False

                while current_stage:
                    stage = current_stage
                    dispatch_count += 1
                    if max_stage_dispatches > 0 and dispatch_count > max_stage_dispatches:
                        raise RuntimeError("staged_workflow_dispatch_limit_exceeded")
                    idx = dispatch_count
                    if _is_cancel_requested(job_id):
                        raise RuntimeError(cancel_error)
                    if (
                        stage == "plan"
                        and companion_pod
                        and _k8s_analysis_require_rag_ready()
                    ):
                        try:
                            rag_status = _k8s_wait_analysis_companion_result(
                                job_id,
                                companion_pod,
                                timeout_sec=_k8s_analysis_rag_wait_timeout_sec(),
                                require_rag=True,
                            )
                            rag_ready = _analysis_companion_is_ready(rag_status, require_rag=True)
                            mcp_url_txt = str(rag_status.get("mcp_url") or companion_url).strip()
                            err_txt = str(rag_status.get("error") or rag_status.get("last_error") or "").strip()
                            companion_mcp_ready = rag_ready
                            _job_update(
                                job_id,
                                analysis_companion_url=mcp_url_txt or companion_url,
                                analysis_companion_ready=rag_ready,
                                analysis_companion_error=(None if rag_ready else (err_txt or "rag_not_ready")),
                            )
                            if not rag_ready:
                                print(
                                    f"[job {job_id}] analysis companion not rag-ready before plan "
                                    f"(state={str(rag_status.get('state') or '-')} rag_ok={int(bool(rag_status.get('rag_ok')))}); "
                                    "degraded continue without MCP injection"
                                )
                        except Exception as e:
                            companion_mcp_ready = False
                            _job_update(
                                job_id,
                                analysis_companion_ready=False,
                                analysis_companion_error=f"rag_wait_timeout:{e}",
                            )
                            print(
                                f"[job {job_id}] analysis companion rag wait failed (continuing): {e}"
                            )
                    if stage == "analysis" and _has_reusable_analysis_context(current_repo_root):
                        analysis_context_path = _analysis_context_path_for_repo(current_repo_root)
                        reusable_path = str(analysis_context_path or "")
                        stage_result = {
                            "message": "analysis skipped: reuse existing analysis context",
                            "repo_root": current_repo_root,
                            "workflow_last_step": "analysis",
                            "workflow_recommended_next": "plan",
                            "restart_to_plan": False,
                            "analysis_done": True,
                            "analysis_degraded": False,
                            "analysis_context_path": reusable_path,
                            "analysis_report_path": reusable_path,
                            "analysis_reused": True,
                        }
                        stage_results.append(
                            {
                                "stage": stage,
                                "job_name": "",
                                "ok": True,
                                "repo_root": current_repo_root,
                                "stage_ctx": dict(stage_ctx),
                                "result": stage_result,
                            }
                        )
                        _job_update(
                            job_id,
                            workflow_last_step=stage,
                            workflow_active_step="",
                            k8s_phase="analysis:Reused",
                        )
                        print(
                            f"[job {job_id}] stage {stage} reused existing analysis context: "
                            f"{reusable_path or '(unknown path)'}"
                        )
                        last_result = stage_result
                        current_stage = "plan"
                        continue
                    job_name = _k8s_job_name(job_id, resumed=resumed, stage=stage, seq=idx)
                    result_path, error_path = _k8s_result_paths(job_id, stage=stage, seq=idx)
                    stage_job_names.append(job_name)
                    _job_update(
                        job_id,
                        k8s_job_name=job_name,
                        k8s_job_names=stage_job_names,
                        k8s_result_path=str(result_path),
                        k8s_error_path=str(error_path),
                        k8s_phase=f"{stage}:Submitting",
                        workflow_active_step=stage,
                    )
                    can_pin_node = False
                    if current_node_name:
                        can_pin_node, node_check_reason = _k8s_node_can_run_job(current_node_name)
                        if not can_pin_node:
                            print(
                                f"[job {job_id}] stage {stage} skip node pinning ({current_node_name}): {node_check_reason}"
                            )
                        else:
                            if node_check_reason in {"node_ready", "node_ready_no_metrics"}:
                                print(f"[job {job_id}] stage {stage} node pinning on {current_node_name}")
                            else:
                                print(
                                    f"[job {job_id}] stage {stage} node pinning on {current_node_name}: {node_check_reason}"
                                )

                    payload = {
                        "job_id": job_id,
                        "repo_url": request.code_url,
                        "max_len": int(request.max_tokens),
                        "time_budget": int(total_time_budget_value),
                        "run_time_budget": int(run_time_budget_value),
                        "coverage_loop_max_rounds": int(coverage_loop_max_rounds),
                        "max_fix_rounds": int(max_fix_rounds),
                        "same_error_max_retries": int(same_error_max_retries),
                        "email": request.email,
                        "docker_image": docker_image,
                        "ai_key_path": str(opencode_env_path()),
                        "oss_fuzz_dir": cfg.oss_fuzz_dir,
                        "model": model_value,
                        "resume_from_step": stage,
                        "resume_repo_root": (current_repo_root or None),
                        "stop_after_step": stage,
                        "last_fuzzer": (stage_ctx.get("last_fuzzer") or None),
                        "last_crash_artifact": (stage_ctx.get("last_crash_artifact") or None),
                        "re_workspace_root": (stage_ctx.get("re_workspace_root") or None),
                        "restart_to_plan_reason": (stage_ctx.get("restart_to_plan_reason") or None),
                        "restart_to_plan_stage": (stage_ctx.get("restart_to_plan_stage") or None),
                        "restart_to_plan_error_text": (stage_ctx.get("restart_to_plan_error_text") or None),
                        "restart_to_plan_report_path": (stage_ctx.get("restart_to_plan_report_path") or None),
                        "run_oom_retry_count": (stage_ctx.get("run_oom_retry_count") or None),
                        "run_rss_limit_mb_override": (stage_ctx.get("run_rss_limit_mb_override") or None),
                        "run_parallel_fuzzers_override": (stage_ctx.get("run_parallel_fuzzers_override") or None),
                        "run_unlimited_round_budget_sec": int(
                            stage_ctx.get("run_timeout_budget_sec_override") or unlimited_round_limit_value
                        ),
                        "analysis_companion_url": ((companion_url or None) if companion_mcp_ready else None),
                        "analysis_companion_ready": bool(companion_mcp_ready),
                        "result_path": str(result_path),
                        "error_path": str(error_path),
                        "target_node_name": (current_node_name if can_pin_node else None),
                    }
                    run_fuzzer_count = 1
                    run_parallelism = 1
                    if stage == "run":
                        run_fuzzer_count = _estimate_run_fuzzer_count(current_repo_root or "")
                        run_parallelism = _estimate_run_parallelism(stage_ctx)
                    effective_round_budget = int(
                        stage_ctx.get("run_timeout_budget_sec_override") or unlimited_round_limit_value
                    )
                    wait_timeout = _k8s_stage_wait_timeout_sec(
                        stage=stage,
                        total_time_budget_sec=total_time_budget_value,
                        run_time_budget_sec=run_time_budget_value,
                        run_unlimited_round_budget_sec=effective_round_budget,
                        run_fuzzer_count=run_fuzzer_count,
                        run_parallelism=run_parallelism,
                    )
                    wait_override_key = f"{stage}_timeout_wait_sec_override"
                    try:
                        wait_override_sec = int(stage_ctx.get(wait_override_key) or 0)
                    except Exception:
                        wait_override_sec = 0
                    if wait_override_sec > 0:
                        wait_timeout = max(wait_timeout, wait_override_sec)
                    stage_result: object
                    stage_node_name: str = ""
                    stage_failed = False
                    stage_fail_error = ""
                    stage_fail_reason = ""
                    try:
                        stage_result, stage_node_name = _execute_k8s_job(
                            job_id=job_id,
                            job_name=job_name,
                            payload=payload,
                            result_path=result_path,
                            error_path=error_path,
                            wait_timeout=wait_timeout,
                        )
                    except _K8sJobFailure as e:
                        stage_failed = True
                        stage_fail_error = _redact_sensitive_text(str(e))
                        failure_doc = dict(e.result or {})
                        stage_fail_reason = str(failure_doc.get("error_code") or "").strip() or "k8s_job_failed"
                        oom_retry_count = int(stage_ctx.get("run_oom_retry_count") or 0)
                        if stage == "run" and stage_fail_reason == "oom_killed" and oom_retry_count < 1:
                            rss_raw = (os.environ.get("SHERPA_RUN_RSS_LIMIT_MB") or "").strip()
                            try:
                                base_rss = int(rss_raw) if rss_raw else 131072
                            except Exception:
                                base_rss = 131072
                            retry_rss = max(2048, int(base_rss * 0.75))
                            stage_ctx["run_oom_retry_count"] = str(oom_retry_count + 1)
                            stage_ctx["run_rss_limit_mb_override"] = str(retry_rss)
                            stage_ctx["run_parallel_fuzzers_override"] = "1"
                            stage_result = {
                                "message": "run stage oom_killed; retrying run once with reduced rss/parallel",
                                "repo_root": current_repo_root,
                                "workflow_last_step": stage,
                                "workflow_recommended_next": "run",
                                "restart_to_plan": False,
                                "run_oom_retry_count": stage_ctx["run_oom_retry_count"],
                                "run_rss_limit_mb_override": stage_ctx["run_rss_limit_mb_override"],
                                "run_parallel_fuzzers_override": stage_ctx["run_parallel_fuzzers_override"],
                            }
                            stage_failed = False
                            stage_fail_reason = ""
                            stage_fail_error = ""
                        else:
                            stage_result = {
                                "message": f"stage {stage} failed in k8s worker; restarting from plan",
                                "repo_root": current_repo_root,
                                "workflow_last_step": stage,
                                "workflow_recommended_next": "plan",
                                "restart_to_plan": True,
                                "restart_to_plan_reason": stage_fail_reason,
                                "restart_to_plan_stage": stage,
                                "restart_to_plan_error_text": stage_fail_error,
                                "restart_to_plan_report_path": "",
                                "error": stage_fail_error,
                            }
                    except Exception as e:
                        is_k8s_timeout = "k8s_job_timeout" in str(e)
                        timeout_retry_key = f"{stage}_timeout_retry_count"
                        timeout_retry_count = int(stage_ctx.get(timeout_retry_key) or 0)
                        try:
                            max_timeout_retries = int(os.environ.get("SHERPA_K8S_TIMEOUT_MAX_RETRIES", "0"))
                            if max_timeout_retries <= 0:
                                max_timeout_retries = int(os.environ.get("SHERPA_RUN_TIMEOUT_MAX_RETRIES", "3"))
                        except Exception:
                            max_timeout_retries = 3
                        if stage in ("run", "build") and is_k8s_timeout and timeout_retry_count < max_timeout_retries:
                            current_wait = max(300, wait_timeout)
                            try:
                                current_wait = max(
                                    current_wait, int(stage_ctx.get(wait_override_key) or current_wait)
                                )
                            except Exception:
                                pass
                            extended_wait = int(current_wait * 1.5)
                            stage_ctx[timeout_retry_key] = str(timeout_retry_count + 1)
                            stage_ctx[wait_override_key] = str(extended_wait)
                            if stage == "run":
                                stage_ctx["run_timeout_budget_sec_override"] = str(
                                    int(
                                        int(stage_ctx.get("run_timeout_budget_sec_override") or unlimited_round_limit_value or 7200)
                                        * 1.5
                                    )
                                )
                            print(
                                f"[job {job_id}] {stage} stage k8s_job_timeout; "
                                f"retrying with extended timeout "
                                f"(retry {timeout_retry_count + 1}, "
                                f"wait {current_wait}s -> {extended_wait}s)"
                            )
                            stage_result = {
                                "message": (
                                    f"{stage} stage k8s_job_timeout; retrying with extended timeout "
                                    f"(retry {timeout_retry_count + 1}, "
                                    f"wait {current_wait}s -> {extended_wait}s)"
                                ),
                                "repo_root": current_repo_root,
                                "workflow_last_step": stage,
                                "workflow_recommended_next": stage,
                                "restart_to_plan": False,
                                timeout_retry_key: stage_ctx[timeout_retry_key],
                                wait_override_key: stage_ctx[wait_override_key],
                            }
                            if stage == "run":
                                stage_result["run_timeout_budget_sec_override"] = stage_ctx.get(
                                    "run_timeout_budget_sec_override", ""
                                )
                            stage_failed = False
                            stage_fail_reason = ""
                            stage_fail_error = ""
                        else:
                            stage_failed = True
                            stage_fail_error = _redact_sensitive_text(str(e))
                            if is_k8s_timeout:
                                stage_fail_reason = "k8s_job_timeout"
                                print(
                                    f"[job {job_id}] {stage} stage k8s_job_timeout; "
                                    f"max retries exhausted ({timeout_retry_count}/{max_timeout_retries}) "
                                    f"-> fallback to plan"
                                )
                            else:
                                stage_fail_reason = "stage_dispatch_exception"
                            stage_result = {
                                "message": f"stage {stage} dispatch failed; restarting from plan",
                                "repo_root": current_repo_root,
                                "workflow_last_step": stage,
                                "workflow_recommended_next": "plan",
                                "restart_to_plan": True,
                                "restart_to_plan_reason": stage_fail_reason,
                                "restart_to_plan_stage": stage,
                                "restart_to_plan_error_text": stage_fail_error,
                                "restart_to_plan_report_path": "",
                                "error": stage_fail_error,
                            }
                    if stage_node_name:
                        if current_node_name and stage_node_name != current_node_name:
                            print(
                                f"[job {job_id}] stage {stage} node drift {current_node_name} -> {stage_node_name}, updating pin"
                            )
                        elif not current_node_name:
                            print(f"[job {job_id}] stage {stage} node selected: {stage_node_name}")
                        current_node_name = stage_node_name
                    else:
                        print(f"[job {job_id}] stage {stage} node unknown, continue without updating pin")

                    if isinstance(stage_result, dict):
                        current_repo_root = str(stage_result.get("repo_root") or current_repo_root).strip()
                        for key in (
                            "last_fuzzer",
                            "last_crash_artifact",
                            "re_workspace_root",
                            "restart_to_plan_reason",
                            "restart_to_plan_stage",
                            "restart_to_plan_error_text",
                            "restart_to_plan_report_path",
                            "run_oom_retry_count",
                            "run_rss_limit_mb_override",
                            "run_parallel_fuzzers_override",
                            "run_timeout_retry_count",
                            "run_timeout_budget_sec_override",
                        ):
                            v = str(stage_result.get(key) or "").strip()
                            if v:
                                stage_ctx[key] = v
                        if not bool(stage_result.get("restart_to_plan")):
                            stage_ctx["restart_to_plan_reason"] = ""
                            stage_ctx["restart_to_plan_stage"] = ""
                            stage_ctx["restart_to_plan_error_text"] = ""
                            stage_ctx["restart_to_plan_report_path"] = ""
                        stage_results.append(
                            {
                                "stage": stage,
                                "job_name": job_name,
                                "ok": (not stage_failed),
                                "repo_root": current_repo_root,
                                "stage_ctx": dict(stage_ctx),
                                "result": stage_result,
                            }
                        )
                        if current_repo_root:
                            _job_update(job_id, workflow_repo_root=current_repo_root, resume_repo_root=current_repo_root)
                    else:
                        stage_results.append(
                            {
                                "stage": stage,
                                "job_name": job_name,
                                "ok": True,
                                "repo_root": current_repo_root,
                            }
                        )
                    _job_update(
                        job_id,
                        workflow_last_step=stage,
                        workflow_active_step="",
                        k8s_phase=(f"{stage}:Succeeded" if not stage_failed else f"{stage}:Failed->Plan"),
                    )
                    last_result = stage_result
                    if stage_failed:
                        print(
                            f"[job {job_id}] stage {stage} failed ({stage_fail_reason}): "
                            f"{stage_fail_error} -> fallback to plan"
                        )
                    else:
                        print(f"[job {job_id}] stage {stage} completed via job {job_name}")
                    next_stage = ""
                    if isinstance(stage_result, dict):
                        terminal_reason = str(stage_result.get("fix_build_terminal_reason") or "").strip()
                        if stage == "build" and terminal_reason == "requires_env_rebuild":
                            next_stage = "build"
                            print(
                                f"[job {job_id}] stage {stage} requested env rebuild; dispatching fresh build job"
                            )
                        else:
                            next_stage = _normalize_resume_step(stage_result.get("workflow_recommended_next"))
                    if next_stage in {"", "stop"}:
                        break
                    current_stage = next_stage

                res = dict(last_result) if isinstance(last_result, dict) else {"message": str(last_result or "")}
                res["stage_results"] = stage_results
                res["stage_job_names"] = stage_job_names
                print(f"[job {job_id}] staged k8s workflow finished ({len(stage_results)} stages)")
        except Exception as fuzz_err:
            print(f"[job {job_id}] k8s worker failed: {fuzz_err}")
            import traceback
            traceback.print_exc()
            raise
        if _is_cancel_requested(job_id):
            _job_update(
                job_id,
                status="error",
                error=cancel_error,
                result=None,
                recoverable=False,
                last_resume_finished_at=time.time() if resumed else None,
            )
            return
        res_failed = bool(isinstance(res, dict) and res.get("failed"))
        run_terminal_reason = str((res.get("run_terminal_reason") if isinstance(res, dict) else "") or "").strip()
        final_status = ("resumed" if resumed else "success")
        final_error = None
        if res_failed:
            final_status = "error"
            final_error = str(
                (
                    res.get("last_error")
                    if isinstance(res, dict)
                    else ""
                )
                or run_terminal_reason
                or "workflow_failed"
            ).strip()
        _job_update(
            job_id,
            status=final_status,
            error=final_error,
            result=res,
            recoverable=False,
            resume_error_code=None,
            last_resume_finished_at=time.time() if resumed else None,
        )
    except Exception as e:
        if _is_cancel_requested(job_id):
            fail_status = "error"
            err_text = cancel_error
        else:
            fail_status = "resume_failed" if resumed else "error"
            err_text = _redact_sensitive_text(str(e))
        fail_result = None
        if isinstance(e, _K8sJobFailure):
            fail_result = dict(e.result or {})
        if isinstance(err_text, str) and ":" in err_text:
            reason = err_text.split(":", 1)[0].strip()
            if fail_result is None and reason.startswith("fix_build_"):
                fail_result = {"fix_build_terminal_reason": reason}
            elif fail_result is None and reason.startswith("run_"):
                fail_result = {"run_terminal_reason": reason}
        _job_update(
            job_id,
            status=fail_status,
            error=err_text,
            result=fail_result,
            recoverable=False,
            last_resume_finished_at=time.time() if resumed else None,
        )
    finally:
        try:
            if companion_pod or companion_service:
                _k8s_stop_analysis_companion(companion_pod, companion_service)
                print(
                    f"[job {job_id}] analysis companion stopped pod={companion_pod or '-'} "
                    f"service={companion_service or '-'}"
                )
        except Exception as e:
            print(f"[job {job_id}] analysis companion stop failed: {e}")
            _job_update(job_id, analysis_companion_error=str(e))
        finally:
            _job_update(
                job_id,
                analysis_companion_active=False,
                analysis_companion_ready=False,
                analysis_companion_stopped_at=time.time(),
            )
        _ACTIVE_JOB_STDOUT_TEE.reset(out_token)
        _ACTIVE_JOB_STDERR_TEE.reset(err_token)
        try:
            tee.close()
        except Exception:
            pass
        _job_update(job_id, finished_at=time.time())


def _submit_fuzz_job(request: fuzz_model, cfg: WebPersistentConfig) -> str:
    job_id = _create_job("fuzz", request.code_url)
    _job_update(
        job_id,
        request=request.model_dump(),
        recoverable=True,
        resume_attempts=0,
        resume_error_code=None,
    )
    fut = executor.submit(_run_fuzz_job, job_id, request, cfg, resumed=False, trigger="new")
    _track_job_future(job_id, fut)
    return job_id


def _resume_fuzz_job(job_id: str, cfg: WebPersistentConfig, *, trigger: str) -> dict[str, object]:
    job = _job_snapshot(job_id)
    if not job:
        return {"accepted": False, "reason": "job_not_found"}
    if str(job.get("kind") or "") != "fuzz":
        return {"accepted": False, "reason": "job_not_fuzz"}

    status = str(job.get("status") or "").strip().lower()
    if status in {"queued", "running", "resuming"}:
        return {"accepted": False, "reason": "already_in_progress"}
    if status in {"success", "resumed"}:
        return {"accepted": False, "reason": "already_completed"}

    raw_request = job.get("request")
    if not isinstance(raw_request, dict):
        _mark_resume_failed(job_id, code="missing_resume_context", message="missing request payload for resume")
        return {"accepted": False, "reason": "missing_resume_context"}

    try:
        req = fuzz_model.model_validate(raw_request)
    except Exception as e:
        _mark_resume_failed(job_id, code="invalid_resume_context", message=f"invalid request payload for resume: {e}")
        return {"accepted": False, "reason": "invalid_resume_context"}

    attempts = int(job.get("resume_attempts") or 0) + 1
    resume_step = _normalize_resume_step(
        str(job.get("resume_from_step") or "")
        or str(job.get("workflow_active_step") or "")
        or str(job.get("workflow_last_step") or "")
        or "build"
    )
    resume_repo_root = str(job.get("resume_repo_root") or job.get("workflow_repo_root") or "").strip()
    if resume_step not in {"analysis", "plan"} and not resume_repo_root:
        _mark_resume_failed(
            job_id,
            code="missing_resume_workspace",
            message=f"cannot resume from step `{resume_step}` without saved repo_root",
        )
        return {"accepted": False, "reason": "missing_resume_workspace"}
    _job_update(
        job_id,
        status="resuming",
        recoverable=False,
        error=None,
        result=None,
        finished_at=None,
        resume_attempts=attempts,
        resume_from_step=resume_step,
        resume_repo_root=resume_repo_root,
        last_resume_reason=trigger,
        last_resume_requested_at=time.time(),
    )
    fut = executor.submit(
        _run_fuzz_job,
        job_id,
        req,
        cfg,
        resumed=True,
        trigger=trigger,
        resume_from_step=resume_step,
        resume_repo_root=resume_repo_root,
    )
    _track_job_future(job_id, fut)
    return {"accepted": True, "reason": "resuming", "resume_attempts": attempts}


def _resume_task_job(job_id: str, cfg: WebPersistentConfig, *, trigger: str) -> dict[str, object]:
    job = _job_snapshot(job_id)
    if not job:
        return {"accepted": False, "reason": "job_not_found"}
    if str(job.get("kind") or "") != "task":
        return {"accepted": False, "reason": "job_not_task"}

    status = str(job.get("status") or "").strip().lower()
    if status in {"queued", "running", "resuming"}:
        return {"accepted": False, "reason": "already_in_progress"}
    if status in {"success", "resumed"}:
        return {"accepted": False, "reason": "already_completed"}

    child_ids = [str(x) for x in (job.get("children") or []) if str(x).strip()]
    if not child_ids:
        _mark_resume_failed(job_id, code="missing_resume_children", message="missing child jobs for task resume")
        return {"accepted": False, "reason": "missing_resume_children"}

    resumed_any = False
    for cid in child_ids:
        child = _job_snapshot(cid)
        if not child:
            continue
        child_status = str(child.get("status") or "").strip().lower()
        if child_status in {"success", "resumed"}:
            continue
        if child_status in {"queued", "running", "resuming"}:
            resumed_any = True
            continue
        out = _resume_fuzz_job(cid, cfg, trigger=f"{trigger}:task:{job_id}")
        if bool(out.get("accepted")):
            resumed_any = True

    if resumed_any:
        attempts = int(job.get("resume_attempts") or 0) + 1
        _job_update(
            job_id,
            status="resuming",
            recoverable=False,
            error=None,
            finished_at=None,
            resume_attempts=attempts,
            last_resume_reason=trigger,
            last_resume_requested_at=time.time(),
        )
        return {"accepted": True, "reason": "resuming", "resume_attempts": attempts}

    refreshed = _job_snapshot(job_id) or job
    derived = _derive_task_status(refreshed)
    final_status = str(derived.get("status") or "").strip().lower()
    if final_status in {"success", "error"} and not _is_status_terminal(job.get("status")):
        _job_update(job_id, status=final_status, finished_at=float(derived.get("finished_at") or time.time()))
    return {"accepted": False, "reason": "no_resumable_children"}


def _stop_fuzz_job(job_id: str, *, reason: str, trigger: str) -> dict[str, object]:
    snap = _job_snapshot(job_id)
    if not snap:
        return {"accepted": False, "reason": "job_not_found"}
    if str(snap.get("kind") or "") != "fuzz":
        return {"accepted": False, "reason": "job_not_fuzz"}

    status = str(snap.get("status") or "").strip().lower()
    now = time.time()
    _job_update(
        job_id,
        cancel_requested=True,
        last_cancel_requested_at=now,
        last_cancel_reason=trigger,
        recoverable=False,
    )

    future_cancelled = _cancel_job_future(job_id)
    repo_root = str(snap.get("workflow_repo_root") or snap.get("resume_repo_root") or "").strip()
    killed_containers = _stop_runtime_containers_for_repo(repo_root) if repo_root else []
    k8s_job_name = str(snap.get("k8s_job_name") or "").strip()
    raw_names = snap.get("k8s_job_names") or []
    k8s_job_names = [str(x).strip() for x in raw_names if str(x).strip()] if isinstance(raw_names, list) else []
    if k8s_job_name and k8s_job_name not in k8s_job_names:
        k8s_job_names.append(k8s_job_name)
    for name in k8s_job_names:
        _k8s_delete_job(name)

    if status not in {"success", "resumed", "error", "resume_failed"}:
        _job_update(
            job_id,
            status="error",
            error=reason,
            result=None,
            recoverable=False,
            finished_at=now,
        )

    current = _job_snapshot(job_id) or snap
    return {
        "accepted": True,
        "reason": "stopped",
        "status": str(current.get("status") or ""),
        "future_cancelled": bool(future_cancelled),
        "killed_containers": killed_containers,
        "k8s_job_name": (k8s_job_name or None),
        "k8s_job_names": k8s_job_names,
        "repo_root": repo_root or None,
    }


def _stop_task_job(job_id: str, *, reason: str, trigger: str) -> dict[str, object]:
    snap = _job_snapshot(job_id)
    if not snap:
        return {"accepted": False, "reason": "job_not_found"}
    if str(snap.get("kind") or "") != "task":
        return {"accepted": False, "reason": "job_not_task"}

    now = time.time()
    _job_update(
        job_id,
        cancel_requested=True,
        last_cancel_requested_at=now,
        last_cancel_reason=trigger,
        recoverable=False,
    )
    parent_future_cancelled = _cancel_job_future(job_id)

    child_ids = [str(x) for x in (snap.get("children") or []) if str(x).strip()]
    child_results: list[dict[str, object]] = []
    for cid in child_ids:
        child_results.append(_stop_fuzz_job(cid, reason=reason, trigger=f"{trigger}:task:{job_id}"))

    _job_update(
        job_id,
        status="error",
        error=reason,
        recoverable=False,
        finished_at=now,
    )
    refreshed = _job_snapshot(job_id) or snap
    derived = _derive_task_status(refreshed)
    _job_update(
        job_id,
        status="error",
        error=reason,
        recoverable=False,
        finished_at=time.time(),
    )

    return {
        "accepted": True,
        "reason": "stopped",
        "status": "error",
        "children_status": derived.get("children_status"),
        "stopped_children": child_results,
        "parent_future_cancelled": bool(parent_future_cancelled),
    }


def _auto_resume_enabled() -> bool:
    raw = (os.environ.get("SHERPA_WEB_AUTO_RESUME_ON_START", "0") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _auto_resume_recoverable_jobs(cfg: WebPersistentConfig) -> None:
    if not _auto_resume_enabled():
        return
    with _JOBS_LOCK:
        snapshot = {job_id: dict(job) for job_id, job in _JOBS.items()}

    task_ids = [
        job_id
        for job_id, job in snapshot.items()
        if str(job.get("kind") or "") == "task"
        and str(job.get("status") or "").strip().lower() == "recoverable"
    ]
    for tid in task_ids:
        _resume_task_job(tid, cfg, trigger="auto_startup")

    with _JOBS_LOCK:
        snapshot2 = {job_id: dict(job) for job_id, job in _JOBS.items()}
    for job_id, job in snapshot2.items():
        if str(job.get("kind") or "") != "fuzz":
            continue
        if str(job.get("status") or "").strip().lower() != "recoverable":
            continue
        parent_id = str(job.get("parent_id") or "").strip()
        if parent_id and parent_id in snapshot2:
            continue
        _resume_fuzz_job(job_id, cfg, trigger="auto_startup")


@app.post("/api/task")
async def task_api(request: task_model = Body(...)):
    cfg = _cfg_get()
    _enforce_docker_only(request.jobs, cfg)
    job_id = _create_job("task", "batch")
    _job_update(
        job_id,
        request=request.model_dump(),
        recoverable=True,
        resume_attempts=0,
        resume_error_code=None,
    )

    def _runner() -> None:
        cancel_error = "cancelled by user"
        if _is_cancel_requested(job_id):
            _job_update(
                job_id,
                status="error",
                error=cancel_error,
                recoverable=False,
                finished_at=time.time(),
            )
            return
        _job_update(job_id, status="running", started_at=time.time())
        log_file = _job_log_path(job_id)
        _job_update(job_id, log_file=str(log_file))
        tee = _Tee(job_id, log_file=log_file)
        out_token = _ACTIVE_JOB_STDOUT_TEE.set(tee)
        err_token = _ACTIVE_JOB_STDERR_TEE.set(tee)
        had_error = False
        child_ids: list[str] = []
        try:
            print(f"[task {job_id}] start (jobs={len(request.jobs)})")
            if _is_cancel_requested(job_id):
                raise RuntimeError(cancel_error)
            if request.auto_init:
                with _INIT_LOCK:
                    if _is_cancel_requested(job_id):
                        raise RuntimeError(cancel_error)
                    auto_init_oss_fuzz = (
                        (os.environ.get("SHERPA_AUTO_INIT_OSS_FUZZ", "0") or "")
                        .strip()
                        .lower()
                        in {"1", "true", "yes", "on"}
                    )
                    # Native k8s staged runtime does not require oss-fuzz checkout.
                    # Force-disable auto-init here to avoid unrelated git clone failures.
                    if _executor_mode() == "k8s_job" and auto_init_oss_fuzz:
                        auto_init_oss_fuzz = False
                        print("[task] skip oss-fuzz auto-init in k8s native runtime")
                    if auto_init_oss_fuzz:
                        repo_url = (
                            (request.oss_fuzz_repo_url or "").strip()
                            or os.environ.get("SHERPA_OSS_FUZZ_REPO_URL", "").strip()
                            or "https://github.com/google/oss-fuzz.git"
                        )
                        target_dir = Path(
                            (cfg.oss_fuzz_dir or "").strip()
                            or os.environ.get("SHERPA_DEFAULT_OSS_FUZZ_DIR", "").strip()
                            or str(_REPO_ROOT / "oss-fuzz")
                        ).expanduser().resolve()
                        print(f"[task] ensure oss-fuzz at {target_dir} from {repo_url}")
                        _ensure_oss_fuzz_checkout(
                            repo_url=repo_url,
                            target_dir=target_dir,
                            force=request.force_clone,
                        )
                    else:
                        print("[task] skip oss-fuzz auto-init (SHERPA_AUTO_INIT_OSS_FUZZ=0)")

                    should_build_images = request.build_images
                    if should_build_images:
                        if _executor_mode() == "k8s_job":
                            print("[task] skip prebuild images in k8s native runtime mode")
                            should_build_images = False
                    if should_build_images:
                        # Only build if any job uses Docker (explicit or default config).
                        use_docker_jobs = any(
                            (j.docker if j.docker is not None else cfg.fuzz_use_docker)
                            for j in request.jobs
                        )
                        if use_docker_jobs:
                            from fuzz_unharnessed_repo import DOCKERFILE_FUZZ_CPP, DOCKERFILE_FUZZ_JAVA
                            images = request.images
                            if not images:
                                # Only prebuild explicitly requested non-auto images.
                                # 'auto' images are built lazily by the workflow once language is known.
                                inferred: set[str] = set()
                                for j in request.jobs:
                                    img = (j.docker_image or "").strip().lower()
                                    if img in {"cpp", "c", "cxx"}:
                                        inferred.add("cpp")
                                    elif img in {"java", "jazzer"}:
                                        inferred.add("java")
                                images = sorted(inferred)
                            if not images:
                                print("[task] skip prebuild images (no explicit image hints); lazy-build on demand")
                            for img in images:
                                name = (img or "").strip().lower()
                                if name in {"cpp", "c", "cxx"}:
                                    tag = os.environ.get("SHERPA_DOCKER_IMAGE_CPP", "sherpa-fuzz-cpp:latest")
                                    print(f"[task] ensure image {tag}")
                                    _ensure_docker_image(tag, DOCKERFILE_FUZZ_CPP, force=request.force_build)
                                elif name in {"java", "jazzer"}:
                                    tag = os.environ.get("SHERPA_DOCKER_IMAGE_JAVA", "sherpa-fuzz-java:latest")
                                    print(f"[task] ensure image {tag}")
                                    _ensure_docker_image(tag, DOCKERFILE_FUZZ_JAVA, force=request.force_build)
                                else:
                                    print(f"[task] skip unknown image hint: {img}")
            # Submit child jobs after parent setup logs are written.
            # Each job now uses ContextVar-based log routing, so concurrent jobs
            # remain isolated without process-global stdout/stderr switching.
            for job in request.jobs:
                if _is_cancel_requested(job_id):
                    break
                child_id = _submit_fuzz_job(job, cfg)
                child_ids.append(child_id)
                _job_update(child_id, parent_id=job_id)
            if _is_cancel_requested(job_id):
                _job_update(
                    job_id,
                    status="error",
                    error=cancel_error,
                    recoverable=False,
                    finished_at=time.time(),
                )
            elif child_ids:
                _job_update(job_id, result="submitted", children=child_ids)
            else:
                _job_update(job_id, status="success", result="submitted (0 jobs)", finished_at=time.time())
            tee.write(f"[task {job_id}] submitted {len(child_ids)} fuzz jobs\n")
        except Exception as e:
            had_error = True
            _job_update(job_id, status="error", error=(cancel_error if _is_cancel_requested(job_id) else str(e)))
        finally:
            _ACTIVE_JOB_STDOUT_TEE.reset(out_token)
            _ACTIVE_JOB_STDERR_TEE.reset(err_token)
            try:
                tee.close()
            except Exception:
                pass
            if had_error:
                _job_update(job_id, finished_at=time.time())

    fut = executor.submit(_runner)
    _track_job_future(job_id, fut)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/task/{job_id}")
def get_task(job_id: str):
    job = _job_snapshot(job_id)
    if not job:
        return {"error": "job_not_found"}
    if job.get("kind") != "task":
        return {"error": "job_not_task"}
    return _derive_task_status(job)


@app.post("/api/task/{job_id}/resume")
def resume_task(job_id: str):
    cfg = _cfg_get()
    snap0 = _job_snapshot(job_id) or {}
    kind = str(snap0.get("kind") or "")
    if kind == "fuzz":
        out = _resume_fuzz_job(job_id, cfg, trigger="manual_api")
    else:
        out = _resume_task_job(job_id, cfg, trigger="manual_api")
    snap = _job_snapshot(job_id)
    return {
        "job_id": job_id,
        "kind": kind or str((snap or {}).get("kind") or ""),
        "accepted": bool(out.get("accepted")),
        "reason": str(out.get("reason") or ""),
        "resume_attempts": int(out.get("resume_attempts") or (snap or {}).get("resume_attempts") or 0),
        "status": str((snap or {}).get("status") or ""),
    }


@app.post("/api/task/{job_id}/stop")
def stop_task(job_id: str):
    snap0 = _job_snapshot(job_id) or {}
    kind = str(snap0.get("kind") or "")
    reason = "cancelled by user"
    trigger = "manual_api"

    if kind == "fuzz":
        out = _stop_fuzz_job(job_id, reason=reason, trigger=trigger)
    else:
        out = _stop_task_job(job_id, reason=reason, trigger=trigger)

    snap = _job_snapshot(job_id)
    return {
        "job_id": job_id,
        "kind": kind or str((snap or {}).get("kind") or ""),
        "accepted": bool(out.get("accepted")),
        "reason": str(out.get("reason") or ""),
        "status": str((snap or {}).get("status") or ""),
        "details": out,
    }


@app.get("/api/tasks")
def list_tasks(limit: int = 50):
    return {
        "items": _list_tasks(limit=limit),
    }


@app.get("/")
def service_root():
    return {
        "service": "sherpa-web",
        "role": "api-backend-only",
        "entrypoint": "Use Ingress at / for UI and /api/* for API",
    }


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
