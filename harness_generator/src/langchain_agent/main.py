# main.py
from __future__ import annotations
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
import hashlib
import os
import re
import shutil
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
from fuzz_relative_functions import fuzz_logic
from job_store import SQLiteJobStore
from persistent_config import (
    OpencodeProviderConfig,
    WebPersistentConfig,
    apply_config_to_env,
    as_public_dict,
    list_opencode_provider_models_resolved,
    normalize_opencode_providers,
    opencode_env_path,
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
_JOB_STORE: SQLiteJobStore | None = None
_JOB_FUTURES_LOCK = threading.Lock()
_JOB_FUTURES: dict[str, Future] = {}

# In-memory API log retention limit (characters).
# 0 or negative means unlimited (no truncation).
_JOB_MEMORY_LOG_MAX_CHARS = int(os.environ.get("SHERPA_WEB_JOB_LOG_MAX_CHARS", "0"))
_JOB_RESTORE_LOG_MAX_CHARS = int(os.environ.get("SHERPA_WEB_RESTORE_LOG_MAX_CHARS", "200000"))


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


def _list_runtime_containers_for_repo(repo_root: str) -> list[str]:
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


def _job_store_mode() -> str:
    return (os.environ.get("SHERPA_WEB_JOB_STORE_MODE", "sqlite") or "").strip().lower()


def _job_store_db_path() -> Path:
    return Path(
        os.environ.get("SHERPA_WEB_JOB_DB_PATH", "/app/job-store/jobs.sqlite3")
    ).expanduser().resolve()


def _job_store_enabled() -> bool:
    return _job_store_mode() not in {"", "memory", "inmemory", "none", "off", "0", "false", "no"}


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
    if not _job_store_enabled():
        _JOB_STORE = None
        return
    try:
        store = SQLiteJobStore(_job_store_db_path())
        store.init_schema()
        _JOB_STORE = store
        _restore_jobs_from_store()
    except Exception as e:
        print(f"[warn] job store disabled due to init error: {e}")
        _JOB_STORE = None


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
        if self._fh is not None and s:
            try:
                self._fh.write(s)
                self._fh.flush()
            except Exception:
                # Do not break the job if disk logging fails mid-run.
                pass
        if s:
            for line in s.splitlines(keepends=True):
                self._split_write(line)
                _update_workflow_checkpoint_from_line(self._job_id, line)
        _job_append_log(self._job_id, s)
        return super().write(s)

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


_RESUMABLE_WORKFLOW_STEPS = {"plan", "synthesize", "build", "fix_build", "run", "fix_crash"}


def _normalize_resume_step(raw: str | None) -> str:
    s = str(raw or "").strip().lower()
    if s in _RESUMABLE_WORKFLOW_STEPS:
        return s
    return "plan"


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
        "config": {
            "oss_fuzz_dir": cfg.oss_fuzz_dir,
            "openai_base_url": cfg.openai_base_url,
            "openai_api_key_set": bool(cfg.openai_api_key),
            "openrouter_model": cfg.openrouter_model,
        },
    }

class fuzz_model(BaseModel):
    code_url: str
    email: str | None = None
    model: str | None = None
    temperature: float = 0.5
    timeout: int = 10
    max_tokens: int = 1000
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
    docker_enabled = request.docker if request.docker is not None else cfg.fuzz_use_docker
    docker_image_value = (
        (request.docker_image or "").strip()
        or (cfg.fuzz_docker_image or "").strip()
        or "auto"
    )
    return bool(docker_enabled), docker_image_value


def _enforce_docker_only(jobs: list[fuzz_model], cfg: WebPersistentConfig) -> None:
    for idx, job in enumerate(jobs):
        docker_enabled, docker_image_value = _resolve_job_docker_policy(job, cfg)
        if not docker_enabled:
            raise HTTPException(
                status_code=400,
                detail=f"Docker-only policy: jobs[{idx}] must set docker=true (or enable default docker in config).",
            )
        if not docker_image_value.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Docker-only policy: jobs[{idx}] must provide docker_image (or configure fuzz_docker_image).",
            )


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


def _merge_opencode_provider_secrets(
    current: WebPersistentConfig,
    payload: dict,
) -> None:
    current_by_name: dict[str, OpencodeProviderConfig] = {
        str(p.name or "").strip().lower(): p for p in (current.opencode_providers or [])
    }

    incoming_entries = payload.get("opencode_providers")
    if not isinstance(incoming_entries, list):
        payload["opencode_providers"] = []
        return

    merged_entries: list[OpencodeProviderConfig] = []
    for raw in incoming_entries:
        if not isinstance(raw, dict):
            continue

        name = str(raw.get("name") or "").strip().lower()
        if not name:
            continue

        existing = current_by_name.get(name)
        clear_flag = bool(raw.get("clear_api_key"))
        raw_key = raw.get("api_key")
        next_key = ""
        if isinstance(raw_key, str):
            next_key = raw_key.strip()

        if clear_flag:
            resolved_key: str | None = ""
        elif next_key:
            resolved_key = next_key
        elif existing and isinstance(existing.api_key, str) and existing.api_key.strip():
            resolved_key = existing.api_key
        else:
            resolved_key = ""

        models_raw = raw.get("models")
        models = list(models_raw) if isinstance(models_raw, list) else []
        headers_raw = raw.get("headers")
        headers = dict(headers_raw) if isinstance(headers_raw, dict) else {}
        options_raw = raw.get("options")
        options = dict(options_raw) if isinstance(options_raw, dict) else {}

        merged_entries.append(
            OpencodeProviderConfig(
                name=name,
                enabled=bool(raw.get("enabled", True)),
                base_url=str(raw.get("base_url") or "").strip(),
                api_key=(resolved_key if resolved_key else None),
                clear_api_key=False,
                models=models,
                headers=headers,
                options=options,
            )
        )

    payload["opencode_providers"] = [
        item.model_dump() for item in normalize_opencode_providers(merged_entries)
    ]


@app.put("/api/config")
def put_config(request: WebPersistentConfig = Body(...)):
    if request.fuzz_use_docker is False:
        raise HTTPException(
            status_code=400,
            detail="Docker-only policy: fuzz_use_docker must remain enabled.",
        )
    if int(request.fuzz_time_budget) < 0:
        raise HTTPException(
            status_code=400,
            detail="fuzz_time_budget must be >= 0 (0 means unlimited).",
        )

    current = _cfg_get()
    payload = request.model_dump()
    # Preserve existing secrets when frontend submits null/omits key fields.
    if request.openai_api_key is None:
        payload["openai_api_key"] = current.openai_api_key
    if request.openrouter_api_key is None:
        payload["openrouter_api_key"] = current.openrouter_api_key
    _merge_opencode_provider_secrets(current, payload)
    payload["fuzz_use_docker"] = True
    payload["fuzz_docker_image"] = (payload.get("fuzz_docker_image") or "").strip() or "auto"
    cfg = WebPersistentConfig(**payload)
    save_config(cfg)
    _cfg_set(cfg)
    apply_config_to_env(cfg)
    return {"ok": True}


@app.get("/api/system")
def get_system_status():
    return _system_status()


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
        }
        job_payload = dict(_JOBS[job_id])
    if job_payload is not None:
        _persist_job_state(job_payload)
    return job_id


def _ensure_oss_fuzz_checkout(*, repo_url: str, target_dir: Path, force: bool) -> None:
    if target_dir.is_dir() and (target_dir / "infra" / "helper.py").is_file():
        print("[init] oss-fuzz already present")
        return
    if target_dir.exists():
        if not force:
            raise RuntimeError(
                f"oss-fuzz dir exists but invalid (missing infra/helper.py): {target_dir}"
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
        return dict(job)
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
    view["children"] = child_jobs
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
                "result": job.get("result"),
                "children_status": children_status,
                "child_count": children_status.get("total", 0),
                "active_child_id": active_child.get("job_id") if active_child else None,
                "active_child_status": active_child.get("status") if active_child else None,
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
            print(f"[job {job_id}] about to call fuzz_logic...")
            docker_enabled, docker_image_value = _resolve_job_docker_policy(request, cfg)
            if not docker_enabled:
                raise RuntimeError("Docker-only policy violation: non-Docker fuzz execution is disabled.")
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
            print(
                f"[job {job_id}] params docker={docker_enabled} docker_image={docker_image_value} "
                f"time_budget={total_budget_log} run_time_budget={run_budget_log} "
                f"max_tokens={request.max_tokens} model={model_value}"
            )
            print(f"[job {job_id}] log_file={log_file}")
            docker_image = docker_image_value
            print(f"[job {job_id}] calling fuzz_logic with docker_image={docker_image}...")
            if _is_cancel_requested(job_id):
                raise RuntimeError(cancel_error)
            try:
                res = fuzz_logic(
                    request.code_url,
                    max_len=request.max_tokens,
                    time_budget=total_time_budget_value,
                    run_time_budget=run_time_budget_value,
                    email=request.email,
                    docker_image=docker_image,
                    ai_key_path=opencode_env_path(),
                    oss_fuzz_dir=cfg.oss_fuzz_dir,
                    model=model_value,
                    resume_from_step=(_normalize_resume_step(resume_from_step) if resumed else None),
                    resume_repo_root=(resume_repo_root if resumed else None),
                )
                print(f"[job {job_id}] fuzz_logic returned successfully")
            except Exception as fuzz_err:
                print(f"[job {job_id}] fuzz_logic failed: {fuzz_err}")
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
            err_text = str(e)
        _job_update(
            job_id,
            status=fail_status,
            error=err_text,
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

                        should_build_images = request.build_images
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
        "entrypoint": "Use sherpa-gateway at / for UI and /api/* for API",
    }


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
