from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import main as web_main


def test_execute_k8s_job_captures_stage_node_before_job_delete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    result_path = tmp_path / "result.json"
    error_path = tmp_path / "error.txt"
    result_path.write_text('{"ok": true, "result": {"message": "ok"}}', encoding="utf-8")
    error_path.write_text("", encoding="utf-8")

    monkeypatch.setenv("SHERPA_K8S_KEEP_FINISHED_JOBS", "0")
    monkeypatch.setattr(web_main, "_k8s_build_manifest", lambda job_name, payload: "kind: Job\n")
    monkeypatch.setattr(web_main, "_kubectl", lambda args, input_text=None, timeout=30: (0, "", ""))
    monkeypatch.setattr(web_main, "_k8s_wait_job", lambda job_name, timeout_sec: ("Succeeded", "Running"))
    monkeypatch.setattr(web_main, "_job_update", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_main, "_k8s_get_job_node_name", lambda job_name: "node-a")

    deleted: list[str] = []
    monkeypatch.setattr(web_main, "_k8s_delete_job", lambda name: deleted.append(name))

    result, node_name = web_main._execute_k8s_job(
        job_id="job-1",
        job_name="job-name-1",
        payload={"job_id": "job-1"},
        result_path=result_path,
        error_path=error_path,
        wait_timeout=30,
    )

    assert node_name == "node-a"
    assert isinstance(result, dict)
    assert result.get("message") == "ok"
    assert deleted == ["job-name-1"]

