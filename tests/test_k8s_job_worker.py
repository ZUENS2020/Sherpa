from __future__ import annotations

import base64
import json
import os
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


def _write_context_docs(context_dir: Path, *, control: dict | None = None, workflow: dict | None = None) -> None:
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "control_context.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "updated_at": 1,
                "job_id": "job-test",
                **(control or {}),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (context_dir / "workflow_context.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "updated_at": 1,
                "job_id": "job-test",
                **(workflow or {}),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


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
        "model": "minimax/MiniMax-M2.7-highspeed",
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
        "model": "minimax/MiniMax-M2.7-highspeed",
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
        "analysis_companion_url": "http://sherpa-promefuzz-job-bootstrap.sherpa-dev.svc.cluster.local:18080/mcp",
        "model": "deepseek-reasoner",
        "result_path": str(result_path),
        "error_path": str(error_path),
    }

    monkeypatch.setenv("SHERPA_K8S_WORKER_PAYLOAD_B64", _payload_b64(payload))
    monkeypatch.setenv("SHERPA_RUNTIME_CONFIG_DIR", str(runtime_dir))
    monkeypatch.setenv("LLM_key", "test-deepseek-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_CONFIG", raising=False)

    def _fake_fuzz_logic(**kwargs):
        captured.update(kwargs)
        cfg_path = Path(str(os.environ.get("OPENCODE_CONFIG") or ""))
        assert cfg_path == runtime_dir / "opencode.generated.json"
        assert cfg_path.is_file()
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        provider = data["provider"]["deepseek"]
        assert provider["options"]["baseURL"] == "https://api.deepseek.com/v1"
        assert provider["options"]["apiKey"] == "test-deepseek-key"
        assert data.get("mcp", {}).get("promefuzz", {}).get("url") == payload["analysis_companion_url"]
        assert os.environ.get("SHERPA_OPENCODE_MCP_URL") == payload["analysis_companion_url"]
        return {"ok": True}

    monkeypatch.setattr(k8s_job_worker, "fuzz_logic", _fake_fuzz_logic)

    rc = k8s_job_worker.main()
    assert rc == 0
    assert captured["docker_image"] is None


def test_worker_applies_run_oom_retry_overrides_to_env(tmp_path: Path, monkeypatch):
    captured: dict = {}
    result_path = tmp_path / "result.json"
    error_path = tmp_path / "error.txt"
    context_dir = tmp_path / "fuzz" / "context"
    _write_context_docs(
        context_dir,
        control={
            "run_rss_limit_mb_override": "98304",
            "run_parallel_fuzzers_override": "1",
        },
    )
    payload = {
        "job_id": "job-oom-retry-override",
        "repo_url": "https://github.com/fmtlib/fmt.git",
        "max_len": 1000,
        "time_budget": 900,
        "run_time_budget": 900,
        "context_dir": str(context_dir),
        "result_path": str(result_path),
        "error_path": str(error_path),
    }

    monkeypatch.setenv("SHERPA_K8S_WORKER_PAYLOAD_B64", _payload_b64(payload))
    monkeypatch.delenv("SHERPA_RUN_RSS_LIMIT_MB", raising=False)
    monkeypatch.delenv("SHERPA_PARALLEL_FUZZERS", raising=False)

    def _fake_fuzz_logic(**kwargs):
        captured.update(kwargs)
        assert os.environ.get("SHERPA_RUN_RSS_LIMIT_MB") == "98304"
        assert os.environ.get("SHERPA_PARALLEL_FUZZERS") == "1"
        return {"ok": True}

    monkeypatch.setattr(k8s_job_worker, "fuzz_logic", _fake_fuzz_logic)
    rc = k8s_job_worker.main()
    assert rc == 0
    assert captured["docker_image"] is None


def test_worker_skips_mcp_injection_when_companion_not_ready(tmp_path: Path, monkeypatch):
    captured: dict = {}
    result_path = tmp_path / "result.json"
    error_path = tmp_path / "error.txt"
    runtime_dir = tmp_path / "runtime"
    payload = {
        "job_id": "job-companion-not-ready",
        "repo_url": "https://github.com/fmtlib/fmt.git",
        "max_len": 1000,
        "time_budget": 900,
        "run_time_budget": 900,
        "analysis_companion_ready": False,
        "analysis_companion_url": "http://sherpa-promefuzz-job.svc.cluster.local:18080/mcp",
        "result_path": str(result_path),
        "error_path": str(error_path),
    }
    monkeypatch.setenv("SHERPA_K8S_WORKER_PAYLOAD_B64", _payload_b64(payload))
    monkeypatch.setenv("SHERPA_RUNTIME_CONFIG_DIR", str(runtime_dir))
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    monkeypatch.delenv("SHERPA_OPENCODE_MCP_URL", raising=False)
    monkeypatch.delenv("SHERPA_OPENCODE_MCP_SERVERS_JSON", raising=False)

    def _fake_fuzz_logic(**kwargs):
        captured.update(kwargs)
        cfg_path = Path(str(os.environ.get("OPENCODE_CONFIG") or ""))
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert "promefuzz" not in (data.get("mcp") or {})
        assert os.environ.get("SHERPA_OPENCODE_MCP_URL") in {None, ""}
        return {"ok": True}

    monkeypatch.setattr(k8s_job_worker, "fuzz_logic", _fake_fuzz_logic)
    rc = k8s_job_worker.main()
    assert rc == 0
    assert captured["docker_image"] is None


def test_worker_forwards_repair_context_fields(tmp_path: Path, monkeypatch):
    captured: dict = {}
    result_path = tmp_path / "result.json"
    error_path = tmp_path / "error.txt"
    context_dir = tmp_path / "fuzz" / "context"
    _write_context_docs(
        context_dir,
        workflow={
            "crash_triage_label": "harness_bug",
            "crash_triage_confidence": 1.0,
            "crash_triage_reason": "bad harness contract",
            "crash_triage_done": True,
            "repair_mode": True,
            "repair_origin_stage": "fix-harness",
            "repair_error_kind": "harness_bug",
            "repair_error_code": "crash_triage_harness_bug",
            "repair_signature": "sig-123",
            "repair_recent_attempts": [{"origin": "fix-harness", "error_kind": "harness_bug"}],
            "repair_error_digest": {"error_code": "crash_triage_harness_bug"},
        },
    )
    payload = {
        "job_id": "job-repair-forward",
        "repo_url": "https://github.com/madler/zlib.git",
        "max_len": 1000,
        "time_budget": 900,
        "run_time_budget": 900,
        "context_dir": str(context_dir),
        "result_path": str(result_path),
        "error_path": str(error_path),
    }

    monkeypatch.setenv("SHERPA_K8S_WORKER_PAYLOAD_B64", _payload_b64(payload))

    def _fake_fuzz_logic(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(k8s_job_worker, "fuzz_logic", _fake_fuzz_logic)
    rc = k8s_job_worker.main()
    assert rc == 0
    assert captured["context_dir"] == str(context_dir)


def test_worker_forwards_restart_and_decision_trace_fields(tmp_path: Path, monkeypatch):
    captured: dict = {}
    result_path = tmp_path / "result.json"
    error_path = tmp_path / "error.txt"
    context_dir = tmp_path / "fuzz" / "context"
    _write_context_docs(
        context_dir,
        workflow={
            "restart_to_plan_reason": "compile_error",
            "restart_to_plan_stage": "build",
            "restart_to_plan_error_text": "ld: undefined reference",
            "restart_to_plan_report_path": "/tmp/reports/restart.md",
            "decision_trace_count": 7,
            "latest_decision_snapshot": {"kind": "choose_repair", "choice": "plan_repair_build"},
            "crash_signature_dedup_hit": True,
            "target_score_breakdown_available": True,
        },
    )
    payload = {
        "job_id": "job-forward-restart-and-trace",
        "repo_url": "https://github.com/madler/zlib.git",
        "max_len": 1000,
        "time_budget": 900,
        "run_time_budget": 900,
        "context_dir": str(context_dir),
        "result_path": str(result_path),
        "error_path": str(error_path),
    }
    monkeypatch.setenv("SHERPA_K8S_WORKER_PAYLOAD_B64", _payload_b64(payload))

    def _fake_fuzz_logic(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(k8s_job_worker, "fuzz_logic", _fake_fuzz_logic)
    rc = k8s_job_worker.main()
    assert rc == 0
    assert captured["context_dir"] == str(context_dir)


def test_worker_applies_unlimited_round_budget_override_to_env(tmp_path: Path, monkeypatch):
    captured: dict = {}
    result_path = tmp_path / "result.json"
    error_path = tmp_path / "error.txt"
    payload = {
        "job_id": "job-unlimited-round-budget-override",
        "repo_url": "https://github.com/fmtlib/fmt.git",
        "max_len": 1000,
        "time_budget": 900,
        "run_time_budget": 900,
        "run_unlimited_round_budget_sec": "1800",
        "result_path": str(result_path),
        "error_path": str(error_path),
    }
    monkeypatch.setenv("SHERPA_K8S_WORKER_PAYLOAD_B64", _payload_b64(payload))
    monkeypatch.delenv("SHERPA_RUN_UNLIMITED_ROUND_BUDGET_SEC", raising=False)

    def _fake_fuzz_logic(**kwargs):
        captured.update(kwargs)
        assert os.environ.get("SHERPA_RUN_UNLIMITED_ROUND_BUDGET_SEC") == "1800"
        return {"ok": True}

    monkeypatch.setattr(k8s_job_worker, "fuzz_logic", _fake_fuzz_logic)
    rc = k8s_job_worker.main()
    assert rc == 0
    assert captured["docker_image"] is None


def test_worker_fails_fast_when_opencode_defunct_exceeds_threshold(tmp_path: Path, monkeypatch):
    result_path = tmp_path / "result.json"
    error_path = tmp_path / "error.txt"
    payload = {
        "job_id": "job-defunct-guard",
        "repo_url": "https://github.com/fmtlib/fmt.git",
        "max_len": 1000,
        "time_budget": 900,
        "run_time_budget": 900,
        "result_path": str(result_path),
        "error_path": str(error_path),
    }
    monkeypatch.setenv("SHERPA_K8S_WORKER_PAYLOAD_B64", _payload_b64(payload))
    monkeypatch.setenv("SHERPA_OPENCODE_DEFUNCT_THRESHOLD", "3")
    counts = iter([5, 5])
    monkeypatch.setattr(k8s_job_worker, "_count_opencode_defunct_processes", lambda: next(counts))
    monkeypatch.setattr(k8s_job_worker, "_reap_any_dead_children", lambda: 0)

    rc = k8s_job_worker.main()
    assert rc == 1
    assert result_path.is_file()
    assert error_path.is_file()
    txt = error_path.read_text(encoding="utf-8", errors="replace")
    assert "opencode defunct process count exceeded threshold" in txt


def test_worker_reap_allows_continue_when_after_count_is_below_threshold(tmp_path: Path, monkeypatch):
    result_path = tmp_path / "result.json"
    error_path = tmp_path / "error.txt"
    payload = {
        "job_id": "job-defunct-reap-ok",
        "repo_url": "https://github.com/fmtlib/fmt.git",
        "max_len": 1000,
        "time_budget": 900,
        "run_time_budget": 900,
        "result_path": str(result_path),
        "error_path": str(error_path),
    }

    monkeypatch.setenv("SHERPA_K8S_WORKER_PAYLOAD_B64", _payload_b64(payload))
    monkeypatch.setenv("SHERPA_OPENCODE_DEFUNCT_THRESHOLD", "3")
    counts = iter([5, 2])
    monkeypatch.setattr(k8s_job_worker, "_count_opencode_defunct_processes", lambda: next(counts))
    monkeypatch.setattr(k8s_job_worker, "_reap_any_dead_children", lambda: 3)
    monkeypatch.setattr(k8s_job_worker, "fuzz_logic", lambda **kwargs: {"ok": True})

    rc = k8s_job_worker.main()
    assert rc == 0
    assert result_path.is_file()
