from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import k8s_job_worker


def _payload_b64(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def test_worker_forces_native_mode_ignores_payload_docker_image(tmp_path: Path, monkeypatch):
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


def test_worker_forces_native_mode_when_payload_docker_image_missing(tmp_path: Path, monkeypatch):
    captured: dict = {}
    result_path = tmp_path / "result.json"
    error_path = tmp_path / "error.txt"

    payload = {
        "job_id": "job-auto-default",
        "repo_url": "https://github.com/madler/zlib.git",
        "max_len": 1000,
        "time_budget": 900,
        "run_time_budget": 900,
        "model": "minimax/MiniMax-M2.5",
        "result_path": str(result_path),
        "error_path": str(error_path),
    }

    def _fake_fuzz_logic(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setenv("SHERPA_K8S_WORKER_PAYLOAD_B64", _payload_b64(payload))
    monkeypatch.setattr(k8s_job_worker, "fuzz_logic", _fake_fuzz_logic)

    rc = k8s_job_worker.main()
    assert rc == 0
    assert captured["docker_image"] is None


def test_worker_preserves_zero_budgets_for_unlimited_mode(tmp_path: Path, monkeypatch):
    captured: dict = {}
    result_path = tmp_path / "result.json"
    error_path = tmp_path / "error.txt"

    payload = {
        "job_id": "job-unlimited",
        "repo_url": "https://github.com/madler/zlib.git",
        "max_len": 1000,
        "time_budget": 0,
        "run_time_budget": 0,
        "coverage_loop_max_rounds": 5,
        "max_fix_rounds": 4,
        "same_error_max_retries": 2,
        "result_path": str(result_path),
        "error_path": str(error_path),
    }

    def _fake_fuzz_logic(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setenv("SHERPA_K8S_WORKER_PAYLOAD_B64", _payload_b64(payload))
    monkeypatch.setattr(k8s_job_worker, "fuzz_logic", _fake_fuzz_logic)

    rc = k8s_job_worker.main()
    assert rc == 0
    assert captured["time_budget"] == 0
    assert captured["run_time_budget"] == 0
    # Round/retry knobs are intentionally hard-disabled in worker runtime.
    assert captured["coverage_loop_max_rounds"] == 0
    assert captured["max_fix_rounds"] == 0
    assert captured["same_error_max_retries"] == 0


def test_worker_bootstraps_runtime_opencode_config_before_fuzz_logic(tmp_path: Path, monkeypatch):
    captured: dict = {}
    result_path = tmp_path / "result.json"
    error_path = tmp_path / "error.txt"
    runtime_dir = tmp_path / "runtime"

    payload = {
        "job_id": "job-bootstrap-opencode",
        "repo_url": "https://github.com/fmtlib/fmt.git",
        "max_len": 1000,
        "time_budget": 900,
        "run_time_budget": 900,
        "model": "MiniMax-M2.5",
        "result_path": str(result_path),
        "error_path": str(error_path),
    }

    monkeypatch.setenv("SHERPA_K8S_WORKER_PAYLOAD_B64", _payload_b64(payload))
    monkeypatch.setenv("SHERPA_RUNTIME_CONFIG_DIR", str(runtime_dir))
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_CONFIG", raising=False)

    def _fake_fuzz_logic(**kwargs):
        captured.update(kwargs)
        cfg_path = Path(str(os.environ.get("OPENCODE_CONFIG") or ""))
        assert cfg_path == runtime_dir / "opencode.generated.json"
        assert cfg_path.is_file()
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        provider = data["provider"]["minimax"]
        assert provider["options"]["baseURL"] == "https://api.minimaxi.com/anthropic/v1"
        assert provider["options"]["apiKey"] == "test-minimax-key"
        return {"ok": True}

    monkeypatch.setattr(k8s_job_worker, "fuzz_logic", _fake_fuzz_logic)

    rc = k8s_job_worker.main()
    assert rc == 0
    assert captured["docker_image"] is None


def test_worker_configures_git_safe_directory_entries(tmp_path: Path, monkeypatch):
    calls: list[list[str]] = []
    result_path = tmp_path / "result.json"
    error_path = tmp_path / "error.txt"

    payload = {
        "job_id": "job-git-safe-dir",
        "repo_url": "https://github.com/madler/zlib.git",
        "max_len": 1000,
        "time_budget": 900,
        "run_time_budget": 900,
        "resume_repo_root": "/shared/output/libarchive-17d00079",
        "result_path": str(result_path),
        "error_path": str(error_path),
    }

    def _fake_fuzz_logic(**kwargs):
        return {"ok": True}

    def _fake_run(cmd, **kwargs):
        calls.append([str(x) for x in cmd])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("SHERPA_K8S_WORKER_PAYLOAD_B64", _payload_b64(payload))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "0")
    for i in range(0, 16):
        monkeypatch.delenv(f"GIT_CONFIG_KEY_{i}", raising=False)
        monkeypatch.delenv(f"GIT_CONFIG_VALUE_{i}", raising=False)
    monkeypatch.setattr(k8s_job_worker, "fuzz_logic", _fake_fuzz_logic)
    monkeypatch.setattr(k8s_job_worker.subprocess, "run", _fake_run)
    monkeypatch.setattr(k8s_job_worker.os, "geteuid", lambda: 10001)

    rc = k8s_job_worker.main()
    assert rc == 0
    assert os.environ.get("GIT_CONFIG_COUNT") == "3"
    assert os.environ.get("GIT_CONFIG_KEY_0") == "safe.directory"
    assert os.environ.get("GIT_CONFIG_VALUE_0") == "*"
    assert os.environ.get("GIT_CONFIG_VALUE_2") == "/shared/output/libarchive-17d00079"
    assert any(
        cmd[:5] == ["git", "config", "--global", "--add", "safe.directory"] and cmd[-1] == "*"
        for cmd in calls
    )


def test_repair_shared_permissions_runs_when_root(monkeypatch):
    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        calls.append([str(x) for x in cmd])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(k8s_job_worker.os, "geteuid", lambda: 0)
    monkeypatch.setattr(k8s_job_worker.subprocess, "run", _fake_run)

    k8s_job_worker._repair_shared_permissions()
    assert calls
    assert calls[0][0:2] == ["sh", "-lc"]
    assert "chown -R 10001:10001 \"/shared/output/_k8s_jobs\"" in calls[0][2]


def test_write_error_resilient_after_permission_error(tmp_path: Path, monkeypatch):
    target = tmp_path / "stage-03-build.error.txt"
    original = Path.write_text
    state = {"raised": False}

    def _flaky_write(self: Path, *args, **kwargs):
        if self == target and not state["raised"]:
            state["raised"] = True
            raise PermissionError("simulated")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _flaky_write)
    k8s_job_worker._write_error(target, "boom")
    assert target.read_text(encoding="utf-8") == "boom\n"
