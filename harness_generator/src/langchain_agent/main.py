# main.py
from __future__ import annotations
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import asyncio
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import queue
import uuid
from datetime import datetime, timezone
from contextlib import redirect_stdout, redirect_stderr, asynccontextmanager
from io import StringIO
from pathlib import Path
from fuzz_relative_functions import fuzz_logic
from persistent_config import (
    WebPersistentConfig,
    apply_config_to_env,
    as_public_dict,
    opencode_env_path,
    load_config,
    save_config,
)

@asynccontextmanager
async def _lifespan(app: FastAPI):
    cfg = load_config()
    _cfg_set(cfg)
    apply_config_to_env(cfg)
    os.environ["SHERPA_ACCEPT_DIFF_WITHOUT_DONE"] = "0"
    yield


app = FastAPI(title="LangChain Agent API", version="1.0", lifespan=_lifespan)

# 设置静态文件目录
current_dir = os.path.dirname(os.path.abspath(__file__))#获取当前绝对路径
static_dir = os.path.join(current_dir, "static")
print(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

#创建线程池
_MAX_WORKERS = int(os.environ.get("SHERPA_WEB_MAX_WORKERS", "5"))
executor = ThreadPoolExecutor(max_workers=max(1, _MAX_WORKERS))


_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict] = {}
_APP_START = time.time()
_INIT_LOCK = threading.Lock()

# In-memory API log retention limit (characters).
# 0 or negative means unlimited (no truncation).
_JOB_MEMORY_LOG_MAX_CHARS = int(os.environ.get("SHERPA_WEB_JOB_LOG_MAX_CHARS", "0"))


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


def _ensure_job_logs_dir() -> None:
    _JOB_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _job_log_path(job_id: str) -> Path:
    return _JOB_LOGS_DIR / f"{job_id}.log"


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


def _job_update(job_id: str, **fields: object) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        job.update(fields)
        job["updated_at"] = time.time()


def _job_append_log(job_id: str, chunk: str) -> None:
    if not chunk:
        return
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
    counts = {
        "total": len(jobs),
        "queued": sum(1 for j in jobs if j.get("status") == "queued"),
        "running": sum(1 for j in jobs if j.get("status") == "running"),
        "success": sum(1 for j in jobs if j.get("status") == "success"),
        "error": sum(1 for j in jobs if j.get("status") == "error"),
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
        if j.get("status") in {"queued", "running"}
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


@app.put("/api/config")
def put_config(request: WebPersistentConfig = Body(...)):
    if request.fuzz_use_docker is False:
        raise HTTPException(
            status_code=400,
            detail="Docker-only policy: fuzz_use_docker must remain enabled.",
        )

    current = _cfg_get()
    payload = request.model_dump()
    # Preserve existing secrets when frontend submits null/omits key fields.
    if request.openai_api_key is None:
        payload["openai_api_key"] = current.openai_api_key
    if request.openrouter_api_key is None:
        payload["openrouter_api_key"] = current.openrouter_api_key
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
        }
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

    for attempt in range(1, max_attempts + 1):
        try:
            _wait_for_docker_daemon()
        except Exception:
            if attempt == max_attempts:
                raise
        for cmd in build_cmds:
            rc, output = _run_build(cmd)
            last_output = output
            if rc == 0:
                return
            if "unknown flag: --progress" in output:
                # Try without --progress on older Docker.
                continue
            if _buildkit_unavailable(output):
                print("[init] buildx unavailable; retrying docker build with DOCKER_BUILDKIT=0")
                rc2, output2 = _run_build(cmd, buildkit="0")
                last_output = output2
                if rc2 == 0:
                    return
            if _docker_daemon_unreachable(output) and attempt < max_attempts:
                print(f"[init] docker daemon not ready; retrying in {backoff:.0f}s (attempt {attempt}/{max_attempts})")
                time.sleep(backoff)
                backoff *= 2
                break
            raise RuntimeError(f"docker build failed (rc={rc}) for {image}")

    raise RuntimeError(f"docker build failed after retries for {image}. Last output:\n{last_output}")


def _job_snapshot(job_id: str) -> dict | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        return dict(job)


def _derive_task_status(job: dict) -> dict:
    children = list(job.get("children") or [])
    if not children:
        return dict(job)
    child_jobs = []
    with _JOBS_LOCK:
        for cid in children:
            cjob = _JOBS.get(cid)
            if cjob:
                child_jobs.append(dict(cjob))
    total = len(child_jobs)
    queued = sum(1 for j in child_jobs if j.get("status") == "queued")
    running = sum(1 for j in child_jobs if j.get("status") == "running")
    success = sum(1 for j in child_jobs if j.get("status") == "success")
    error = sum(1 for j in child_jobs if j.get("status") == "error")
    if total == 0:
        derived = job.get("status")
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
    queued = sum(1 for j in child_jobs if j.get("status") == "queued")
    running = sum(1 for j in child_jobs if j.get("status") == "running")
    success = sum(1 for j in child_jobs if j.get("status") == "success")
    error = sum(1 for j in child_jobs if j.get("status") == "error")
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
            (c for c in child_jobs if c.get("status") in {"running", "queued"}),
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


def _submit_fuzz_job(request: fuzz_model, cfg: WebPersistentConfig) -> str:
    job_id = _create_job("fuzz", request.code_url)

    def _runner() -> None:
        _job_update(job_id, status="running", started_at=time.time())
        log_file = _job_log_path(job_id)
        _job_update(job_id, log_file=str(log_file))
        tee = _Tee(job_id, log_file=log_file)
        had_error = False
        try:
            with redirect_stdout(tee), redirect_stderr(tee):
                print(f"[job {job_id}] start repo={request.code_url}")
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
                if int(total_time_budget_value) <= 0:
                    raise RuntimeError("total_time_budget must be > 0")
                if int(run_time_budget_value) <= 0:
                    raise RuntimeError("run_time_budget must be > 0")
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
                    f"time_budget={total_time_budget_value}s run_time_budget={run_time_budget_value}s "
                    f"max_tokens={request.max_tokens} model={model_value}"
                )
                print(f"[job {job_id}] log_file={log_file}")
                docker_image = docker_image_value
                print(f"[job {job_id}] calling fuzz_logic with docker_image={docker_image}...")
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
                    )
                    print(f"[job {job_id}] fuzz_logic returned successfully")
                except Exception as fuzz_err:
                    print(f"[job {job_id}] fuzz_logic failed: {fuzz_err}")
                    import traceback
                    traceback.print_exc()
                    raise
                _job_update(job_id, status="success", result=res)
        except Exception as e:
            _job_update(job_id, status="error", error=str(e))
        finally:
            try:
                tee.close()
            except Exception:
                pass
            _job_update(job_id, finished_at=time.time())

    executor.submit(_runner)
    return job_id


@app.post("/api/task")
async def task_api(request: task_model = Body(...)):
    cfg = _cfg_get()
    _enforce_docker_only(request.jobs, cfg)
    job_id = _create_job("task", "batch")

    def _runner() -> None:
        _job_update(job_id, status="running", started_at=time.time())
        log_file = _job_log_path(job_id)
        _job_update(job_id, log_file=str(log_file))
        tee = _Tee(job_id, log_file=log_file)
        had_error = False
        child_ids: list[str] = []
        try:
            with redirect_stdout(tee), redirect_stderr(tee):
                print(f"[task {job_id}] start (jobs={len(request.jobs)})")
                if request.auto_init:
                    with _INIT_LOCK:
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
                child_id = _submit_fuzz_job(job, cfg)
                child_ids.append(child_id)
            if child_ids:
                _job_update(job_id, result="submitted", children=child_ids)
            else:
                _job_update(job_id, status="success", result="submitted (0 jobs)", finished_at=time.time())
            tee.write(f"[task {job_id}] submitted {len(child_ids)} fuzz jobs\n")
        except Exception as e:
            had_error = True
            _job_update(job_id, status="error", error=str(e))
        finally:
            try:
                tee.close()
            except Exception:
                pass
            if had_error:
                _job_update(job_id, finished_at=time.time())

    executor.submit(_runner)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/task/{job_id}")
def get_task(job_id: str):
    job = _job_snapshot(job_id)
    if not job:
        return {"error": "job_not_found"}
    if job.get("kind") != "task":
        return {"error": "job_not_task"}
    return _derive_task_status(job)


@app.get("/api/tasks")
def list_tasks(limit: int = 50):
    return {
        "items": _list_tasks(limit=limit),
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    path = os.path.join(static_dir, "index.html")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
    # finla = create_agent_outside("what is the weather outside?")
    # print(finla)
