# main.py
from __future__ import annotations
from fastapi import FastAPI, Body, HTTPException, Response
from pydantic import BaseModel
import json
import hashlib
import os
import re
import shutil
import resource
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
import threading
import time
import queue
import uuid
from datetime import datetime, timezone
from contextlib import redirect_stdout, redirect_stderr, asynccontextmanager
from io import StringIO
from pathlib import Path
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

    rc_top, out_top, err_top = _kubectl(["top", "node", node_name, "--no-headers"], timeout=10)
    if rc_top != 0:
        detail = (err_top or out_top).strip()
        if detail:
            detail_norm = re.sub(r"\s+", " ", detail).strip()
            # Canonicalize common metrics-server absence into a stable token.
            if re.search(r"metrics api not available", detail_norm, re.IGNORECASE):
                return True, "node_ready_no_metrics_warn:metrics_api_not_available"
            # Avoid confusing "error:" prefixes in warning-only status messages.
            detail_norm = re.sub(r"^\s*error:\s*", "", detail_norm, flags=re.IGNORECASE)
            detail_token = re.sub(r"\s+", "_", detail_norm)[:160]
            return True, f"node_ready_no_metrics_warn:{detail_token}"
        return True, "node_ready_no_metrics_warn"
    line = ""
    for raw in (out_top or "").splitlines():
        txt = raw.strip()
        if txt:
            line = txt
            break
    if not line:
        return True, "node_ready_empty_metrics"
    parts = line.split()
    if len(parts) < 5:
        return True, "node_ready_bad_metrics"
    cpu_pct_txt = parts[2].rstrip("%")
    mem_pct_txt = parts[4].rstrip("%")
    try:
        cpu_pct = int(cpu_pct_txt)
        mem_pct = int(mem_pct_txt)
    except Exception:
        return True, "node_ready_unparsed_metrics"
    if cpu_pct >= max_cpu_pct:
        return False, f"node_cpu_busy:{cpu_pct}%"
    if mem_pct >= max_mem_pct:
        return False, f"node_mem_busy:{mem_pct}%"
    return True, f"node_ready_cpu={cpu_pct}%_mem={mem_pct}%"


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
) -> int:
    """Compute outer k8s-job timeout with explicit grace to avoid false timeouts."""
    try:
        grace_run = int(os.environ.get("SHERPA_K8S_RUN_TIMEOUT_GRACE_SEC", "300"))
    except Exception:
        grace_run = 300
    try:
        grace_default = int(os.environ.get("SHERPA_K8S_STAGE_TIMEOUT_GRACE_SEC", "180"))
    except Exception:
        grace_default = 180
    grace_run = max(60, grace_run)
    grace_default = max(30, grace_default)

    total_base = total_time_budget_sec if total_time_budget_sec > 0 else 7200
    run_base = run_time_budget_sec if run_time_budget_sec > 0 else total_base
    base = run_base if stage == "run" else total_base
    grace = grace_run if stage == "run" else grace_default
    return max(300, base + grace)


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
    "plan",
    "synthesize",
    "build",
    "fix_build",
    "run",
    "coverage-analysis",
    "improve-harness",
    "re-build",
    "re-run",
    "repro_crash",
    "fix_crash",
}
_STAGED_WORKFLOW_STEPS = (
    "plan",
    "synthesize",
    "build",
    "run",
    "coverage-analysis",
    "improve-harness",
    "re-build",
    "re-run",
)


def _normalize_resume_step(raw: str | None) -> str:
    s = str(raw or "").strip().lower()
    if s == "repro_crash":
        return "re-build"
    if s in _RESUMABLE_WORKFLOW_STEPS:
        return s
    return "plan"


def _staged_sequence_from(raw_start: str | None) -> list[str]:
    start = _normalize_resume_step(raw_start)
    if start in {"fix_build", "fix_crash"}:
        start = "build"
    try:
        idx = _STAGED_WORKFLOW_STEPS.index(start)
    except ValueError:
        idx = 0
    return list(_STAGED_WORKFLOW_STEPS[idx:])


def _error_code_for_job(job: dict | None) -> str:
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


def _error_kind_for_job(job: dict | None) -> str:
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


def _error_signature_for_job(job: dict | None) -> str:
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
def put_config(request: WebPersistentConfig = Body(...)):
    if int(request.fuzz_time_budget) < 0:
        raise HTTPException(
            status_code=400,
            detail="fuzz_time_budget must be >= 0 (0 means unlimited).",
        )
    if int(request.sherpa_run_unlimited_round_budget_sec) < 0:
        raise HTTPException(
            status_code=400,
            detail="sherpa_run_unlimited_round_budget_sec must be >= 0 (0 means fully unlimited).",
        )

    current = _cfg_get()
    payload = request.model_dump()

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


def _derive_task_status(job: dict) -> dict:
    children = list(job.get("children") or [])
    if not children:
        view = dict(job)
        view["error_code"] = _error_code_for_job(view)
        view["error_kind"] = _error_kind_for_job(view)
        view["error_signature"] = _error_signature_for_job(view)
        view["phase"] = _phase_for_job(view)
        view["runtime_mode"] = _runtime_mode_for_job(view)
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
        c["error_code"] = _error_code_for_job(c)
        c["error_kind"] = _error_kind_for_job(c)
        c["error_signature"] = _error_signature_for_job(c)
        c["phase"] = _phase_for_job(c)
        c["runtime_mode"] = _runtime_mode_for_job(c)
    view["children"] = child_jobs
    view["error_code"] = _error_code_for_job(view)
    view["error_kind"] = _error_kind_for_job(view)
    view["error_signature"] = _error_signature_for_job(view)
    view["phase"] = _phase_for_job(view)
    view["runtime_mode"] = _runtime_mode_for_job(view)
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
        tasks.append(
            {
                "job_id": job.get("job_id"),
                "status": derived_status,
                "repo": job.get("repo"),
                "created_at": job.get("created_at"),
                "created_at_iso": _iso_time(job.get("created_at")),
                "updated_at": job.get("updated_at"),
                "updated_at_iso": _iso_time(job.get("updated_at")),
                "started_at": job.get("started_at"),
                "started_at_iso": _iso_time(job.get("started_at")),
                "finished_at": job.get("finished_at"),
                "finished_at_iso": _iso_time(job.get("finished_at")),
                "error": job.get("error"),
                "error_code": _error_code_for_job(job),
                "error_kind": _error_kind_for_job(job),
                "error_signature": _error_signature_for_job(job),
                "phase": _phase_for_job(job),
                "runtime_mode": _runtime_mode_for_job(job),
                "result": job.get("result"),
                "children_status": children_status,
                "child_count": children_status.get("total", 0),
                "active_child_id": active_child.get("job_id") if active_child else None,
                "active_child_status": active_child.get("status") if active_child else None,
                "active_child_phase": _phase_for_job(active_child) if active_child else None,
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
    try:
        with redirect_stdout(tee), redirect_stderr(tee):
            print(f"[job {job_id}] start repo={request.code_url} resumed={int(resumed)} trigger={trigger}")
            if _is_cancel_requested(job_id):
                raise RuntimeError(cancel_error)
            print(f"[job {job_id}] about to dispatch k8s worker...")
            docker_enabled, docker_image_value = _resolve_job_docker_policy(request, cfg)
            total_time_budget_value = (
                request.total_time_budget
                if request.total_time_budget is not None
                else (request.time_budget if request.time_budget is not None else cfg.fuzz_time_budget)
            )
            run_time_budget_value = (
                request.run_time_budget
                if request.run_time_budget is not None
                else total_time_budget_value
            )
            total_time_budget_value = int(total_time_budget_value)
            run_time_budget_value = int(run_time_budget_value)
            if total_time_budget_value < 0:
                raise RuntimeError("total_time_budget must be >= 0")
            if run_time_budget_value < 0:
                raise RuntimeError("run_time_budget must be >= 0")
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
                start_step = _normalize_resume_step(resume_from_step) if resumed else "plan"
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
                    current_stage = "plan"
                try:
                    max_stage_dispatches = int((os.environ.get("SHERPA_STAGE_DISPATCH_MAX") or "0").strip())
                except Exception:
                    max_stage_dispatches = 0
                dispatch_count = 0
                while current_stage:
                    stage = current_stage
                    dispatch_count += 1
                    if max_stage_dispatches > 0 and dispatch_count > max_stage_dispatches:
                        raise RuntimeError("staged_workflow_dispatch_limit_exceeded")
                    idx = dispatch_count
                    if _is_cancel_requested(job_id):
                        raise RuntimeError(cancel_error)
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
                            print(f"[job {job_id}] stage {stage} node pinning on {current_node_name}: {node_check_reason}")

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
                        "result_path": str(result_path),
                        "error_path": str(error_path),
                        "target_node_name": (current_node_name if can_pin_node else None),
                    }
                    wait_timeout = _k8s_stage_wait_timeout_sec(
                        stage=stage,
                        total_time_budget_sec=total_time_budget_value,
                        run_time_budget_sec=run_time_budget_value,
                    )
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
                        stage_failed = True
                        stage_fail_error = _redact_sensitive_text(str(e))
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
            _job_update(
                job_id,
                status=("resumed" if resumed else "success"),
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
    if resume_step != "plan" and not resume_repo_root:
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
        had_error = False
        child_ids: list[str] = []
        try:
            with redirect_stdout(tee), redirect_stderr(tee):
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
            # Submit child jobs outside parent stdout/stderr redirection.
            # redirect_stdout is process-global; keeping it active here can clobber
            # child job log redirection and break frontend progress tracking.
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
