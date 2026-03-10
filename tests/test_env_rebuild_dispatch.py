from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import main as web_main
from persistent_config import WebPersistentConfig


def test_run_fuzz_job_redispatches_build_after_env_rebuild_request(monkeypatch, tmp_path: Path):
    captured_stages: list[str] = []

    def _fake_execute_k8s_job(*, job_id, job_name, payload, result_path, error_path, wait_timeout):
        stage = str(payload.get("resume_from_step") or "")
        captured_stages.append(stage)
        if stage == "plan":
            return ({"repo_root": str(tmp_path)}, "node-test")
        if stage == "synthesize":
            return ({"repo_root": str(tmp_path)}, "node-test")
        if stage == "build" and captured_stages.count("build") == 1:
            return (
                {
                    "repo_root": str(tmp_path),
                    "fix_build_terminal_reason": "requires_env_rebuild",
                    "message": "fix_build completed (requires env rebuild)",
                },
                "node-test",
            )
        return ({"repo_root": str(tmp_path), "message": f"{stage} ok"}, "node-test")

    monkeypatch.setattr(web_main, "_execute_k8s_job", _fake_execute_k8s_job)
    monkeypatch.setattr(web_main, "_JOB_STORE", None)
    with web_main._JOBS_LOCK:
        web_main._JOBS.clear()

    request = web_main.fuzz_model(code_url="https://github.com/example/repo.git")
    job_id = web_main._create_job("fuzz", request.code_url)
    web_main._run_fuzz_job(job_id, request, WebPersistentConfig(), resumed=False, trigger="new")

    assert captured_stages[:4] == ["plan", "synthesize", "build", "build"]
    assert captured_stages.count("build") == 2
    job = web_main._job_snapshot(job_id)
    assert job is not None
    assert job["status"] == "success"
