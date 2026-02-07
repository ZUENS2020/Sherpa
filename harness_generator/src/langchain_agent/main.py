# main.py
from __future__ import annotations
from fastapi import FastAPI, Body
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
import uuid
from datetime import datetime, timezone
from contextlib import redirect_stdout, redirect_stderr
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

app = FastAPI(title="LangChain Agent API", version="1.0")

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


_REPO_ROOT = Path(__file__).resolve().parents[3]
_JOB_LOGS_DIR = _REPO_ROOT / "config" / "logs" / "jobs"


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


@app.on_event("startup")
def _load_persistent_config() -> None:
    cfg = load_config()
    _cfg_set(cfg)
    apply_config_to_env(cfg)


def _job_log_path(job_id: str) -> Path:
    return _JOB_LOGS_DIR / f"{job_id}.log"


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
        # Keep last 50k characters to avoid unbounded memory.
        job["log"] = buf[-50000:]
        job["updated_at"] = time.time()


class _Tee(StringIO):
    def __init__(self, job_id: str, *, log_file: Path | None = None) -> None:
        super().__init__()
        self._job_id = job_id
        self._fh = None
        if log_file is not None:
            try:
                _ensure_job_logs_dir()
                self._fh = open(log_file, "a", encoding="utf-8")
            except Exception:
                # Best-effort: if we cannot write to disk, keep in-memory logs working.
                self._fh = None

    def write(self, s: str) -> int:
        if self._fh is not None and s:
            try:
                self._fh.write(s)
                self._fh.flush()
            except Exception:
                # Do not break the job if disk logging fails mid-run.
                pass
        _job_append_log(self._job_id, s)
        return super().write(s)

    def close(self) -> None:
        try:
            if self._fh is not None:
                self._fh.close()
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


@app.get("/api/config")
def get_config():
    cfg = _cfg_get()
    return as_public_dict(cfg)


@app.put("/api/config")
def put_config(request: WebPersistentConfig = Body(...)):
    cfg = request
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

    cmd = ["docker", "build", "--progress=plain", "-t", image, "-f", str(dockerfile), str(_REPO_ROOT)]
    print("[init] " + " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        # Retry without --progress for older Docker.
        cmd = ["docker", "build", "-t", image, "-f", str(dockerfile), str(_REPO_ROOT)]
        print("[init] " + " ".join(cmd))
        rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"docker build failed (rc={rc}) for {image}")


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
        _job_update(job.get("job_id"), finished_at=time.time(), status=derived)
    return view

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
                docker_enabled = request.docker if request.docker is not None else cfg.fuzz_use_docker
                docker_image_value = (
                    (request.docker_image or "").strip()
                    or cfg.fuzz_docker_image
                )
                time_budget_value = request.time_budget if request.time_budget is not None else cfg.fuzz_time_budget
                print(
                    f"[job {job_id}] params docker={docker_enabled} docker_image={docker_image_value} "
                    f"time_budget={time_budget_value}s max_tokens={request.max_tokens} model={request.model or cfg.openrouter_model}"
                )
                print(f"[job {job_id}] log_file={log_file}")
                docker_image = docker_image_value if docker_enabled else None
                res = fuzz_logic(
                    request.code_url,
                    max_len=request.max_tokens,
                    time_budget=time_budget_value,
                    email=request.email,
                    docker_image=docker_image,
                    ai_key_path=opencode_env_path(),
                    oss_fuzz_dir=cfg.oss_fuzz_dir,
                )
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
    job_id = _create_job("task", "batch")

    def _runner() -> None:
        _job_update(job_id, status="running", started_at=time.time())
        log_file = _job_log_path(job_id)
        _job_update(job_id, log_file=str(log_file))
        tee = _Tee(job_id, log_file=log_file)
        try:
            with redirect_stdout(tee), redirect_stderr(tee):
                print(f"[task {job_id}] start (jobs={len(request.jobs)})")
                if request.auto_init:
                    with _INIT_LOCK:
                        repo_url = (
                            (request.oss_fuzz_repo_url or "").strip()
                            or os.environ.get("SHERPA_OSS_FUZZ_REPO_URL", "").strip()
                            or "https://github.com/google/oss-fuzz"
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
                                images = request.images or ["cpp", "java"]
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

                child_ids: list[str] = []
                for job in request.jobs:
                    child_id = _submit_fuzz_job(job, cfg)
                    child_ids.append(child_id)
                _job_update(job_id, result="submitted", children=child_ids)
                print(f"[task {job_id}] submitted {len(child_ids)} fuzz jobs")
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
