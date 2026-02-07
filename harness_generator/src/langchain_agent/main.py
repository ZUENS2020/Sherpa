# main.py
from __future__ import annotations
from fastapi import FastAPI, Body
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import uuid
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from fuzz_relative_functions import fuzz_logic
from persistent_config import (
    WebPersistentConfig,
    apply_config_to_env,
    as_public_dict,
    codex_env_path,
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
executor = ThreadPoolExecutor(max_workers=5)


_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict] = {}


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

@app.post("/fuzz_code")#对代码仓库进行模糊测试(后续添加发送邮件功能)
async def fuzz_code(request: fuzz_model = Body(...)):
    """对代码仓库进行模糊测试（异步提交任务，前端轮询查看状态/日志）"""
    cfg = _cfg_get()
    job_id = uuid.uuid4().hex
    now = time.time()
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "repo": request.code_url,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "result": None,
            "log": "",
            "log_file": None,
        }

    def _runner() -> None:
        _job_update(job_id, status="running", started_at=time.time())
        log_file = _job_log_path(job_id)
        _job_update(job_id, log_file=str(log_file))
        tee = _Tee(job_id, log_file=log_file)
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
                    ai_key_path=codex_env_path(),
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
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/fuzz/{job_id}")
def fuzz_job(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return {"error": "job_not_found"}
        # shallow copy
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "repo": job["repo"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
            "error": job["error"],
            "result": job["result"],
            "log": job["log"],
            "log_file": job.get("log_file"),
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