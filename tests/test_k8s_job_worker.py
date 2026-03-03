from __future__ import annotations

import base64
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import k8s_job_worker


def _payload_b64(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def test_worker_forces_local_mode_ignores_payload_docker_image(tmp_path: Path, monkeypatch):
    captured: dict = {}
    result_path = tmp_path / "result.json"
    error_path = tmp_path / "error.txt"

    payload = {
        "job_id": "job-local-only",
        "repo_url": "https://github.com/madler/zlib.git",
        "max_len": 1234,
        "time_budget": 900,
        "run_time_budget": 600,
        "docker_image": "sherpa-fuzz-cpp:latest",
        "model": "minimax/MiniMax-M2.5",
        "result_path": str(result_path),
        "error_path": str(error_path),
    }

    def _fake_fuzz_logic(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "message": "fake"}

    monkeypatch.setenv("SHERPA_K8S_WORKER_PAYLOAD_B64", _payload_b64(payload))
    monkeypatch.setattr(k8s_job_worker, "fuzz_logic", _fake_fuzz_logic)

    rc = k8s_job_worker.main()
    assert rc == 0
    assert captured["docker_image"] is None
    assert result_path.is_file()
    out = json.loads(result_path.read_text(encoding="utf-8"))
    assert out["ok"] is True
    assert out["job_id"] == "job-local-only"
    assert out["result"]["message"] == "fake"
