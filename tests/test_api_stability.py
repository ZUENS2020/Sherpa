from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import main as web_main
from persistent_config import WebPersistentConfig


class _ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)

        class _Done:
            def result(self, timeout: float | None = None):
                return None

        return _Done()


@pytest.fixture(autouse=True)
def _isolate_runtime_state(monkeypatch, tmp_path: Path):
    with web_main._JOBS_LOCK:
        web_main._JOBS.clear()

    monkeypatch.setattr(web_main, "executor", _ImmediateExecutor())
    monkeypatch.setattr(web_main, "save_config", lambda cfg: None)
    monkeypatch.setattr(web_main, "apply_config_to_env", lambda cfg: None)
    monkeypatch.setattr(web_main, "_job_log_path", lambda job_id: tmp_path / f"{job_id}.log")

    yield

    with web_main._JOBS_LOCK:
        web_main._JOBS.clear()


def test_get_config_masks_secret_values():
    with TestClient(web_main.app) as client:
        web_main._cfg_set(
            WebPersistentConfig(
                openai_api_key="openai-secret",
                openrouter_api_key="openrouter-secret",
            )
        )
        response = client.get("/api/config")

    assert response.status_code == 200
    data = response.json()
    assert data["openai_api_key"] == ""
    assert data["openrouter_api_key"] == ""
    assert data["openai_api_key_set"] is True
    assert data["openrouter_api_key_set"] is True


def test_put_config_preserves_existing_secrets_when_payload_is_null():
    with TestClient(web_main.app) as client:
        web_main._cfg_set(
            WebPersistentConfig(
                openai_api_key="keep-openai",
                openrouter_api_key="keep-openrouter",
            )
        )
        response = client.put(
            "/api/config",
            json={
                "openai_api_key": None,
                "openrouter_api_key": None,
                "fuzz_time_budget": 1200,
            },
        )

    assert response.status_code == 200
    cfg = web_main._cfg_get()
    assert cfg.openai_api_key == "keep-openai"
    assert cfg.openrouter_api_key == "keep-openrouter"
    assert cfg.fuzz_time_budget == 1200


def test_task_submit_no_auto_init_is_successful_and_does_not_crash(monkeypatch):
    def _fake_submit(job, _cfg):
        child_id = web_main._create_job("fuzz", job.code_url)
        web_main._job_update(child_id, status="success", result="ok", finished_at=time.time())
        return child_id

    monkeypatch.setattr(web_main, "_submit_fuzz_job", _fake_submit)

    with TestClient(web_main.app) as client:
        response = client.post(
            "/api/task",
            json={
                "jobs": [{"code_url": "https://github.com/example/repo.git"}],
                "auto_init": False,
            },
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        status = client.get(f"/api/task/{job_id}").json()

    assert status["status"] == "success"
    assert status["finished_at"] is not None


def test_task_submit_marks_error_and_finished_at_when_init_fails(monkeypatch):
    def _raise_init_error(*args, **kwargs):
        raise RuntimeError("clone failed")

    monkeypatch.setattr(web_main, "_ensure_oss_fuzz_checkout", _raise_init_error)

    with TestClient(web_main.app) as client:
        response = client.post(
            "/api/task",
            json={"jobs": [{"code_url": "https://github.com/example/repo.git"}]},
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        status = client.get(f"/api/task/{job_id}").json()

    assert status["status"] == "error"
    assert "clone failed" in (status.get("error") or "")
    assert status["finished_at"] is not None
