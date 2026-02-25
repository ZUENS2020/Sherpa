from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import main as web_main
from job_store import SQLiteJobStore


@pytest.fixture(autouse=True)
def _isolate_job_store_state(monkeypatch):
    with web_main._JOBS_LOCK:
        web_main._JOBS.clear()
    monkeypatch.setattr(web_main, "_JOB_STORE", None)
    monkeypatch.setenv("SHERPA_WEB_JOB_STORE_MODE", "memory")
    monkeypatch.setenv("SHERPA_WEB_AUTO_RESUME_ON_START", "0")
    yield
    with web_main._JOBS_LOCK:
        web_main._JOBS.clear()
    web_main._JOB_STORE = None


def test_sqlite_store_restores_parent_child_relationship(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    store.init_schema()
    web_main._JOB_STORE = store

    task_id = web_main._create_job("task", "batch")
    child_id = web_main._create_job("fuzz", "https://github.com/example/repo.git")
    web_main._job_update(child_id, status="success", result="ok", finished_at=time.time())
    web_main._job_update(task_id, status="running", children=[child_id], result="submitted")

    with web_main._JOBS_LOCK:
        web_main._JOBS.clear()

    web_main._restore_jobs_from_store()

    task = web_main._job_snapshot(task_id)
    assert task is not None
    assert task["kind"] == "task"
    assert task["children"] == [child_id]

    child = web_main._job_snapshot(child_id)
    assert child is not None
    assert child["status"] == "success"
    assert child["result"] == "ok"


def test_sqlite_restore_marks_inflight_jobs_as_recoverable(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    store.init_schema()
    web_main._JOB_STORE = store

    job_id = web_main._create_job("fuzz", "https://github.com/example/repo.git")
    web_main._job_update(job_id, status="running", started_at=time.time())

    with web_main._JOBS_LOCK:
        web_main._JOBS.clear()

    web_main._restore_jobs_from_store()
    restored = web_main._job_snapshot(job_id)

    assert restored is not None
    assert restored["status"] == "recoverable"
    assert restored.get("recoverable") is True
    assert restored.get("last_resume_reason") == "service_restart"
    assert "service restart" in str(restored.get("error") or "")
    assert restored.get("finished_at") is None


def test_sqlite_restore_hydrates_log_from_disk_when_memory_log_empty(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    store.init_schema()
    web_main._JOB_STORE = store

    job_id = web_main._create_job("fuzz", "https://github.com/example/repo.git")
    log_file = tmp_path / f"{job_id}.log"
    log_file.write_text("line-1\nline-2\nline-3\n", encoding="utf-8")
    web_main._job_update(job_id, status="success", log_file=str(log_file), log="")

    with web_main._JOBS_LOCK:
        web_main._JOBS.clear()

    web_main._restore_jobs_from_store()
    restored = web_main._job_snapshot(job_id)

    assert restored is not None
    assert "line-3" in (restored.get("log") or "")


def test_auto_resume_recoverable_task_resumes_child(monkeypatch, tmp_path: Path):
    class _ImmediateExecutor:
        def submit(self, fn, *args, **kwargs):
            fn(*args, **kwargs)

            class _Done:
                def result(self, timeout=None):
                    return None

            return _Done()

    store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    store.init_schema()
    web_main._JOB_STORE = store
    web_main._cfg_set(web_main.WebPersistentConfig())
    monkeypatch.setenv("SHERPA_WEB_AUTO_RESUME_ON_START", "1")
    monkeypatch.setattr(web_main, "executor", _ImmediateExecutor())
    monkeypatch.setattr(web_main, "fuzz_logic", lambda *args, **kwargs: "ok")

    task_id = web_main._create_job("task", "batch")
    child_id = web_main._create_job("fuzz", "https://github.com/example/repo.git")
    web_main._job_update(
        child_id,
        status="running",
        request={"code_url": "https://github.com/example/repo.git"},
        workflow_last_step="run",
        workflow_repo_root=str(tmp_path),
        parent_id=task_id,
    )
    web_main._job_update(task_id, status="running", children=[child_id])

    with web_main._JOBS_LOCK:
        web_main._JOBS.clear()
    web_main._restore_jobs_from_store()

    web_main._auto_resume_recoverable_jobs(web_main._cfg_get())

    child = web_main._job_snapshot(child_id)
    task = web_main._job_snapshot(task_id)
    assert child is not None
    assert child["status"] == "resumed"
    assert task is not None
    assert task["status"] in {"resuming", "running", "success", "resumed"}
