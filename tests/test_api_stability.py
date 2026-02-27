from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import main as web_main
from persistent_config import (
    OpencodeProviderConfig,
    WebPersistentConfig,
    build_opencode_runtime_config,
    list_opencode_provider_models_resolved,
)


class _ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)

        class _Done:
            def result(self, timeout: float | None = None):
                return None

        return _Done()


@pytest.fixture(autouse=True)
def _isolate_runtime_state(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SHERPA_WEB_JOB_STORE_MODE", "memory")
    monkeypatch.setenv("SHERPA_WEB_AUTO_RESUME_ON_START", "0")
    with web_main._JOBS_LOCK:
        web_main._JOBS.clear()
    monkeypatch.setattr(web_main, "_JOB_STORE", None)
    web_main._cfg_set(WebPersistentConfig())

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


def test_get_config_masks_opencode_provider_secret_values():
    with TestClient(web_main.app) as client:
        web_main._cfg_set(
            WebPersistentConfig(
                opencode_providers=[
                    OpencodeProviderConfig(
                        name="minimax",
                        enabled=True,
                        api_key="minimax-secret",
                        base_url="https://api.minimax.io/v1",
                        models=["minimax-text-01"],
                    )
                ]
            )
        )
        response = client.get("/api/config")

    assert response.status_code == 200
    data = response.json()
    providers = data.get("opencode_providers") or []
    assert providers
    assert providers[0]["name"] == "minimax"
    assert providers[0]["api_key"] == ""
    assert providers[0]["api_key_set"] is True


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


def test_put_config_preserves_and_clears_opencode_provider_secret():
    with TestClient(web_main.app) as client:
        web_main._cfg_set(
            WebPersistentConfig(
                opencode_providers=[
                    OpencodeProviderConfig(
                        name="minimax",
                        enabled=True,
                        api_key="keep-provider-key",
                        base_url="https://api.minimax.io/v1",
                    )
                ]
            )
        )

        # Preserve: empty api_key with clear_api_key=false keeps existing key.
        response_keep = client.put(
            "/api/config",
            json={
                "fuzz_time_budget": 1000,
                "opencode_providers": [
                    {
                        "name": "minimax",
                        "enabled": True,
                        "base_url": "https://api.minimax.io/v1",
                        "api_key": "",
                        "clear_api_key": False,
                        "models": ["minimax-text-01"],
                        "headers": {},
                        "options": {},
                    }
                ],
            },
        )
        assert response_keep.status_code == 200
        cfg_keep = web_main._cfg_get()
        assert cfg_keep.opencode_providers[0].api_key == "keep-provider-key"

        # Clear: explicit clear_api_key removes the saved key.
        response_clear = client.put(
            "/api/config",
            json={
                "fuzz_time_budget": 1000,
                "opencode_providers": [
                    {
                        "name": "minimax",
                        "enabled": True,
                        "base_url": "https://api.minimax.io/v1",
                        "api_key": "",
                        "clear_api_key": True,
                        "models": ["minimax-text-01"],
                        "headers": {},
                        "options": {},
                    }
                ],
            },
        )
        assert response_clear.status_code == 200
        cfg_clear = web_main._cfg_get()
        assert cfg_clear.opencode_providers[0].api_key in {None, ""}


def test_put_config_rejects_disabling_docker():
    with TestClient(web_main.app) as client:
        response = client.put(
            "/api/config",
            json={
                "fuzz_use_docker": False,
                "fuzz_time_budget": 900,
                "fuzz_docker_image": "auto",
            },
        )

    assert response.status_code == 400
    assert "Docker-only policy" in response.json().get("detail", "")


def test_put_config_accepts_unlimited_budget_zero():
    with TestClient(web_main.app) as client:
        response = client.put(
            "/api/config",
            json={
                "fuzz_use_docker": True,
                "fuzz_time_budget": 0,
                "fuzz_docker_image": "auto",
            },
        )

    assert response.status_code == 200
    cfg = web_main._cfg_get()
    assert cfg.fuzz_time_budget == 0


def test_put_config_rejects_negative_budget():
    with TestClient(web_main.app) as client:
        response = client.put(
            "/api/config",
            json={
                "fuzz_use_docker": True,
                "fuzz_time_budget": -1,
                "fuzz_docker_image": "auto",
            },
        )

    assert response.status_code == 400
    assert "fuzz_time_budget must be >= 0" in response.json().get("detail", "")


def test_get_opencode_provider_models_supports_glm_alias():
    with TestClient(web_main.app) as client:
        response = client.get("/api/opencode/providers/glm/models")

    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "zai"
    assert isinstance(data["models"], list)
    assert any(str(m).startswith("zai/glm-") for m in data["models"])


def test_get_opencode_provider_models_rejects_unknown_provider():
    with TestClient(web_main.app) as client:
        response = client.get("/api/opencode/providers/unknown/models")

    assert response.status_code == 404


def test_post_opencode_provider_models_uses_request_overrides(monkeypatch):
    captured: dict[str, object] = {}

    def _fake(provider, cfg, *, api_key_override=None, base_url_override=None):
        captured["provider"] = provider
        captured["api_key_override"] = api_key_override
        captured["base_url_override"] = base_url_override
        return "zai", ["zai/glm-4.7"], "remote", ""

    monkeypatch.setattr(web_main, "list_opencode_provider_models_resolved", _fake)

    with TestClient(web_main.app) as client:
        response = client.post(
            "/api/opencode/providers/glm/models",
            json={
                "api_key": "glm-test-key",
                "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "zai"
    assert data["models"] == ["zai/glm-4.7"]
    assert data["source"] == "remote"
    assert captured["provider"] == "glm"
    assert captured["api_key_override"] == "glm-test-key"
    assert captured["base_url_override"] == "https://open.bigmodel.cn/api/coding/paas/v4"


def test_model_listing_does_not_cross_use_openai_key_for_other_provider():
    cfg = WebPersistentConfig(
        openai_api_key="openrouter-key",
        openai_base_url="https://openrouter.ai/api/v1",
        opencode_providers=[
            OpencodeProviderConfig(
                name="deepseek",
                enabled=True,
                api_key=None,
                base_url="https://api.deepseek.com/v1",
            )
        ],
    )

    normalized, models, source, warning = list_opencode_provider_models_resolved("deepseek", cfg)
    assert normalized == "deepseek"
    assert source == "builtin"
    assert models
    assert "api_key not configured" in warning


def test_build_opencode_runtime_config_uses_local_mcp_command_array():
    cfg = WebPersistentConfig(
        opencode_providers=[
            OpencodeProviderConfig(
                name="deepseek",
                enabled=True,
                base_url="https://api.deepseek.com/v1",
                api_key="dummy",
                models=["deepseek/deepseek-reasoner"],
            )
        ]
    )
    payload = build_opencode_runtime_config(cfg)
    mcp = payload.get("mcp", {})
    gitnexus = mcp.get("gitnexus", {})
    assert gitnexus.get("type") == "local"
    assert gitnexus.get("command") == ["gitnexus", "mcp"]


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


def test_task_submit_child_spawn_happens_outside_parent_stdout_redirect(monkeypatch):
    observed_parent_redirect_flags: list[bool] = []

    def _fake_submit(job, _cfg):
        observed_parent_redirect_flags.append(isinstance(sys.stdout, web_main._Tee))
        child_id = web_main._create_job("fuzz", job.code_url)
        web_main._job_update(child_id, status="running")
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

    assert observed_parent_redirect_flags == [False]
    assert status["status"] == "running"


def test_task_submit_rejects_non_docker_job():
    with TestClient(web_main.app) as client:
        response = client.post(
            "/api/task",
            json={
                "jobs": [
                    {
                        "code_url": "https://github.com/example/repo.git",
                        "docker": False,
                    }
                ],
                "auto_init": False,
            },
        )

    assert response.status_code == 400
    assert "Docker-only policy" in response.json().get("detail", "")


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


def test_task_submit_accepts_unlimited_total_and_run_budget(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_fuzz_logic(*args, **kwargs):
        captured["time_budget"] = kwargs.get("time_budget")
        captured["run_time_budget"] = kwargs.get("run_time_budget")
        return "ok"

    monkeypatch.setattr(web_main, "fuzz_logic", _fake_fuzz_logic)

    with TestClient(web_main.app) as client:
        response = client.post(
            "/api/task",
            json={
                "jobs": [
                    {
                        "code_url": "https://github.com/example/repo.git",
                        "total_time_budget": 0,
                        "run_time_budget": 0,
                    }
                ],
                "auto_init": False,
            },
        )
        assert response.status_code == 200
        task_id = response.json()["job_id"]
        status = client.get(f"/api/task/{task_id}").json()

    assert status["status"] == "success"
    assert captured["time_budget"] == 0
    assert captured["run_time_budget"] == 0


def test_list_tasks_returns_recent_tasks_with_child_summary():
    task_old = web_main._create_job("task", "batch")
    time.sleep(0.001)
    task_new = web_main._create_job("task", "batch")
    child = web_main._create_job("fuzz", "https://github.com/example/repo.git")
    web_main._job_update(task_new, children=[child], status="running")
    web_main._job_update(child, status="running")

    with TestClient(web_main.app) as client:
        response = client.get("/api/tasks?limit=10")

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 2
    assert items[0]["job_id"] == task_new
    assert items[0]["status"] == "running"
    assert items[0]["child_count"] == 1
    assert items[0]["children_status"]["running"] == 1
    assert items[1]["job_id"] == task_old


def test_list_tasks_applies_limit_and_filters_non_task_jobs():
    web_main._create_job("fuzz", "https://github.com/example/repo.git")
    web_main._create_job("task", "batch")
    time.sleep(0.001)
    task_latest = web_main._create_job("task", "batch")

    with TestClient(web_main.app) as client:
        response = client.get("/api/tasks?limit=1")

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["job_id"] == task_latest


def test_resume_task_resumes_recoverable_child_job(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(web_main, "fuzz_logic", lambda *args, **kwargs: {"ok": True})

    task_id = web_main._create_job("task", "batch")
    child_id = web_main._create_job("fuzz", "https://github.com/example/repo.git")
    web_main._job_update(
        child_id,
        status="recoverable",
        request={"code_url": "https://github.com/example/repo.git"},
        resume_from_step="run",
        workflow_repo_root=str(tmp_path),
        resume_repo_root=str(tmp_path),
        parent_id=task_id,
    )
    web_main._job_update(task_id, status="recoverable", children=[child_id])

    with TestClient(web_main.app) as client:
        response = client.post(f"/api/task/{task_id}/resume")
        assert response.status_code == 200
        data = response.json()
        assert data["accepted"] is True
        task = client.get(f"/api/task/{task_id}").json()

    child = web_main._job_snapshot(child_id)
    assert child is not None
    assert child["status"] == "resumed"
    assert child["result"] == {"ok": True}
    assert task["status"] == "success"


def test_resume_task_missing_child_request_marks_resume_failed(monkeypatch):
    monkeypatch.setattr(web_main, "fuzz_logic", lambda *args, **kwargs: {"ok": True})

    task_id = web_main._create_job("task", "batch")
    child_id = web_main._create_job("fuzz", "https://github.com/example/repo.git")
    web_main._job_update(child_id, status="recoverable", parent_id=task_id)
    web_main._job_update(task_id, status="recoverable", children=[child_id])

    with TestClient(web_main.app) as client:
        response = client.post(f"/api/task/{task_id}/resume")
        assert response.status_code == 200
        data = response.json()
        assert data["accepted"] is False
        assert data["reason"] == "no_resumable_children"

    child = web_main._job_snapshot(child_id)
    assert child is not None
    assert child["status"] == "resume_failed"
    assert child.get("resume_error_code") == "missing_resume_context"


def test_resume_task_request_is_idempotent(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(web_main, "fuzz_logic", lambda *args, **kwargs: "ok")

    task_id = web_main._create_job("task", "batch")
    child_id = web_main._create_job("fuzz", "https://github.com/example/repo.git")
    web_main._job_update(
        child_id,
        status="recoverable",
        request={"code_url": "https://github.com/example/repo.git"},
        resume_from_step="run",
        workflow_repo_root=str(tmp_path),
        resume_repo_root=str(tmp_path),
        parent_id=task_id,
    )
    web_main._job_update(task_id, status="recoverable", children=[child_id])

    with TestClient(web_main.app) as client:
        first = client.post(f"/api/task/{task_id}/resume").json()
        second = client.post(f"/api/task/{task_id}/resume").json()

    assert first["accepted"] is True
    assert second["accepted"] is False
    assert second["reason"] in {"already_completed", "no_resumable_children", "already_in_progress"}


def test_resume_fuzz_job_uses_saved_resume_step_and_repo(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def _fake_fuzz_logic(*args, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(web_main, "fuzz_logic", _fake_fuzz_logic)

    child_id = web_main._create_job("fuzz", "https://github.com/example/repo.git")
    web_main._job_update(
        child_id,
        status="recoverable",
        request={"code_url": "https://github.com/example/repo.git"},
        resume_from_step="run",
        workflow_repo_root=str(tmp_path),
        resume_repo_root=str(tmp_path),
    )

    with TestClient(web_main.app) as client:
        response = client.post(f"/api/task/{child_id}/resume")
        assert response.status_code == 200
        body = response.json()
        assert body["accepted"] is True
        assert body["kind"] == "fuzz"

    assert captured.get("resume_from_step") == "run"
    assert str(captured.get("resume_repo_root")) == str(tmp_path)


def test_stop_fuzz_job_marks_error_and_cancel_requested(monkeypatch):
    job_id = web_main._create_job("fuzz", "https://github.com/example/repo.git")
    web_main._job_update(job_id, status="running", workflow_repo_root="/tmp/repo")

    monkeypatch.setattr(web_main, "_cancel_job_future", lambda _: True)
    monkeypatch.setattr(web_main, "_stop_runtime_containers_for_repo", lambda _: ["cid-1"])

    with TestClient(web_main.app) as client:
        response = client.post(f"/api/task/{job_id}/stop")
        assert response.status_code == 200
        body = response.json()

    snap = web_main._job_snapshot(job_id)
    assert snap is not None
    assert body["accepted"] is True
    assert body["kind"] == "fuzz"
    assert body["reason"] == "stopped"
    assert body["details"]["future_cancelled"] is True
    assert body["details"]["killed_containers"] == ["cid-1"]
    assert snap["status"] == "error"
    assert snap["cancel_requested"] is True
    assert snap["error"] == "cancelled by user"


def test_stop_task_job_stops_children(monkeypatch):
    task_id = web_main._create_job("task", "batch")
    child_id = web_main._create_job("fuzz", "https://github.com/example/repo.git")
    web_main._job_update(task_id, status="running", children=[child_id])
    web_main._job_update(child_id, status="running", parent_id=task_id, workflow_repo_root="/tmp/repo")

    monkeypatch.setattr(web_main, "_cancel_job_future", lambda _: True)
    monkeypatch.setattr(web_main, "_stop_runtime_containers_for_repo", lambda _: ["cid-child"])

    with TestClient(web_main.app) as client:
        response = client.post(f"/api/task/{task_id}/stop")
        assert response.status_code == 200
        body = response.json()

    task_snap = web_main._job_snapshot(task_id)
    child_snap = web_main._job_snapshot(child_id)
    assert task_snap is not None
    assert child_snap is not None
    assert body["accepted"] is True
    assert body["kind"] == "task"
    assert body["reason"] == "stopped"
    assert body["status"] == "error"
    assert body["details"]["parent_future_cancelled"] is True
    assert len(body["details"]["stopped_children"]) == 1
    assert body["details"]["stopped_children"][0]["accepted"] is True
    assert task_snap["status"] == "error"
    assert task_snap["cancel_requested"] is True
    assert child_snap["status"] == "error"
    assert child_snap["cancel_requested"] is True
    assert child_snap["error"] == "cancelled by user"


def test_stop_task_job_not_found_returns_rejected():
    with TestClient(web_main.app) as client:
        response = client.post("/api/task/not-found/stop")

    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is False
    assert body["reason"] == "job_not_found"


def test_ensure_docker_image_buildkit_fallback_uses_classic_builder_without_progress(monkeypatch, tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")

    calls: list[tuple[list[str], str | None]] = []
    scenarios = [
        (1, "ERROR: BuildKit is enabled but the buildx component is missing or broken.\n"),
        (0, "Successfully built image.\n"),
    ]

    class _FakeProc:
        def __init__(self, output: str, rc: int):
            self.stdout = io.StringIO(output)
            self.returncode: int | None = None
            self._rc = rc

        def poll(self):
            return self.returncode

        def wait(self, timeout: float | None = None):
            self.returncode = self._rc
            return self._rc

    def _fake_run(cmd, *args, **kwargs):
        if list(cmd[:2]) == ["docker", "info"]:
            return SimpleNamespace(returncode=0)
        if list(cmd[:3]) == ["docker", "image", "inspect"]:
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    def _fake_popen(cmd, *args, **kwargs):
        env = kwargs.get("env") or {}
        calls.append((list(cmd), env.get("DOCKER_BUILDKIT")))
        if not scenarios:
            raise AssertionError("unexpected docker build invocation")
        rc, out = scenarios.pop(0)
        return _FakeProc(out, rc)

    monkeypatch.setattr(web_main.subprocess, "run", _fake_run)
    monkeypatch.setattr(web_main.subprocess, "Popen", _fake_popen)

    web_main._ensure_docker_image("test:image", dockerfile, force=True)

    assert len(calls) >= 2
    assert any(
        buildkit == "0" and not any(arg.startswith("--progress=") for arg in cmd)
        for cmd, buildkit in calls
    )
