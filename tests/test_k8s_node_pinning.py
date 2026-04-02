from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


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
    monkeypatch.setattr(
        web_main,
        "_k8s_wait_job",
        lambda job_name, timeout_sec, on_progress=None: ("Succeeded", "Running"),
    )
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


def test_execute_k8s_job_failed_run_classifies_pod_and_persists_failure_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    result_path = tmp_path / "result.json"
    error_path = tmp_path / "error.txt"

    monkeypatch.setenv("SHERPA_K8S_KEEP_FINISHED_JOBS", "0")
    monkeypatch.setattr(web_main, "_k8s_build_manifest", lambda job_name, payload: "kind: Job\n")
    monkeypatch.setattr(web_main, "_kubectl", lambda args, input_text=None, timeout=30: (0, "", ""))
    monkeypatch.setattr(
        web_main,
        "_k8s_wait_job",
        lambda job_name, timeout_sec, on_progress=None: ("Failed", "Running"),
    )
    monkeypatch.setattr(web_main, "_job_update", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        web_main,
        "_k8s_get_job_pod_details",
        lambda job_name: {
            "pod_name": "run-pod-1",
            "phase": "Failed",
            "terminated_reason": "OOMKilled",
            "exit_code": 137,
        },
    )
    monkeypatch.setattr(web_main, "_k8s_collect_job_logs", lambda job_name: "tail line 1\ntail line 2")

    deleted: list[str] = []
    monkeypatch.setattr(web_main, "_k8s_delete_job", lambda name: deleted.append(name))

    with pytest.raises(web_main._K8sJobFailure) as exc:
        web_main._execute_k8s_job(
            job_id="job-2",
            job_name="job-name-2",
            payload={"job_id": "job-2", "stop_after_step": "run"},
            result_path=result_path,
            error_path=error_path,
            wait_timeout=30,
        )

    err = exc.value
    assert err.result["error_code"] == "oom_killed"
    assert err.result["error_kind"] == "resource"
    assert err.result["run_error_kind"] == "run_resource_exhaustion"
    assert err.result["run_terminal_reason"] == "run_resource_exhaustion"
    assert "tail line 2" in err.result["k8s_failure"]["logs_tail"]
    assert "oom_killed" in error_path.read_text(encoding="utf-8")
    assert result_path.is_file()
    assert deleted == ["job-name-2"]


def test_k8s_stage_wait_timeout_run_unlimited_is_round_aware(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SHERPA_K8S_RUN_TIMEOUT_GRACE_SEC", "600")
    monkeypatch.setenv("SHERPA_RUN_UNLIMITED_ROUND_BUDGET_SEC", "7200")
    monkeypatch.setenv("SHERPA_K8S_RUN_TIMEOUT_INTER_ROUND_BUFFER_SEC", "120")
    monkeypatch.setenv("SHERPA_SEED_GEN_RETRY_MULTIPLIER", "1")

    timeout = web_main._k8s_stage_wait_timeout_sec(
        stage="run",
        total_time_budget_sec=0,
        run_time_budget_sec=0,
        run_fuzzer_count=5,
        run_parallelism=2,
    )

    # ceil(5/2)=3 rounds => 3*7200 + 2*120 + 600
    assert timeout == 22440


def test_k8s_stage_wait_timeout_run_finite_budget_not_multiplied(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SHERPA_K8S_RUN_TIMEOUT_GRACE_SEC", "600")
    monkeypatch.setenv("SHERPA_SEED_GEN_RETRY_MULTIPLIER", "1")

    timeout = web_main._k8s_stage_wait_timeout_sec(
        stage="run",
        total_time_budget_sec=0,
        run_time_budget_sec=1800,
        run_fuzzer_count=8,
        run_parallelism=2,
    )

    assert timeout == 2400


def test_k8s_stage_wait_timeout_run_applies_seed_retry_multiplier(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SHERPA_K8S_RUN_TIMEOUT_GRACE_SEC", "600")
    monkeypatch.setenv("SHERPA_SEED_GEN_RETRY_MULTIPLIER", "3")

    timeout = web_main._k8s_stage_wait_timeout_sec(
        stage="run",
        total_time_budget_sec=0,
        run_time_budget_sec=1800,
        run_fuzzer_count=1,
        run_parallelism=1,
    )

    assert timeout == 6000


def test_normalize_resume_step_preserves_stop_signal():
    assert web_main._normalize_resume_step("stop") == "stop"
    assert web_main._normalize_resume_step("STOP") == "stop"
    assert web_main._normalize_resume_step("repro_crash") == "re-build"
    assert web_main._normalize_resume_step(None) == "analysis"


def test_estimate_run_fuzzer_count_prefers_fuzz_out_executables(tmp_path: Path):
    repo_root = tmp_path / "repo"
    out_dir = repo_root / "fuzz" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    f1 = out_dir / "a_fuzz"
    f2 = out_dir / "b_fuzz"
    f1.write_text("x", encoding="utf-8")
    f2.write_text("x", encoding="utf-8")
    f1.chmod(0o755)
    f2.chmod(0o755)

    assert web_main._estimate_run_fuzzer_count(str(repo_root)) == 2


def test_estimate_run_fuzzer_count_falls_back_to_execution_plan(tmp_path: Path):
    repo_root = tmp_path / "repo"
    fuzz_dir = repo_root / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "execution_plan.json").write_text(
        '{"execution_targets":[{"target_name":"a"},{"target_name":"b"},{"target_name":"c"}]}',
        encoding="utf-8",
    )

    assert web_main._estimate_run_fuzzer_count(str(repo_root)) == 3


def test_analysis_companion_manifest_uses_promefuzz_runner(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SHERPA_K8S_ANALYSIS_COMPANION_COMMAND", raising=False)
    monkeypatch.setenv("SHERPA_OUTPUT_DIR", "/shared/output")
    manifest = web_main._k8s_analysis_companion_manifest(
        pod_name="sherpa-promefuzz-job1",
        job_id="job-123",
    )
    doc = yaml.safe_load(manifest)
    assert isinstance(doc, dict)
    assert doc["spec"]["restartPolicy"] == "Never"
    cmd = doc["spec"]["containers"][0]["command"][-1]
    assert "promefuzz_companion.py" in cmd
    env = doc["spec"]["containers"][0]["env"]
    env_names = {str(item.get("name")) for item in env}
    assert "SHERPA_JOB_ID" in env_names
    assert "SHERPA_OUTPUT_DIR" in env_names
    assert "SHERPA_PROMEFUZZ_RUN_ONCE" in env_names
    env_from = doc["spec"]["containers"][0].get("envFrom") or []
    assert isinstance(env_from, list)
    assert any(
        str(((x.get("secretRef") or {}) if isinstance(x, dict) else {}).get("name") or "")
        == "sherpa-openrouter-embedding"
        for x in env_from
    )


def test_enrich_job_view_reads_analysis_companion_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    job_id = "job-analysis-status-1"
    status_path = tmp_path / "_k8s_jobs" / job_id / "promefuzz" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        '{"state":"ready","analysis_backend":"promefuzz-mcp","candidate_count":7,"updated_at":"2026-03-29T00:00:00Z","rag_ok":true,"rag_document_count":12,"rag_chunk_count":51,"rag_knowledge_base_path":"/shared/output/_k8s_jobs/job-analysis-status-1/promefuzz/work/knowledge","embedding_provider":"openrouter","embedding_model":"text-embedding-3-small","embedding_ok":true,"rag_degraded":false,"rag_degraded_reason":"","semantic_query_count":8,"semantic_hit_count":6,"semantic_hit_rate":0.75,"cache_hit_rate":1.0}',
        encoding="utf-8",
    )
    monkeypatch.setenv("SHERPA_OUTPUT_DIR", str(tmp_path))
    view = {"id": job_id}
    web_main._enrich_job_view(view)

    assert view["analysis_companion_state"] == "ready"
    assert view["analysis_companion_backend"] == "promefuzz-mcp"
    assert view["analysis_companion_candidate_count"] == 7
    assert view["analysis_companion_rag_ok"] is True
    assert view["analysis_companion_rag_document_count"] == 12
    assert view["analysis_companion_rag_chunk_count"] == 51
    assert view["analysis_companion_embedding_provider"] == "openrouter"
    assert view["analysis_companion_embedding_model"] == "text-embedding-3-small"
    assert view["analysis_companion_embedding_ok"] is True
    assert view["analysis_companion_rag_degraded"] is False
    assert view["analysis_companion_semantic_query_count"] == 8
    assert view["analysis_companion_semantic_hit_count"] == 6
    assert view["analysis_companion_semantic_hit_rate"] == pytest.approx(0.75)
    assert view["analysis_companion_cache_hit_rate"] == pytest.approx(1.0)


def test_wait_analysis_companion_result_returns_first_available_state(monkeypatch: pytest.MonkeyPatch):
    calls = {"n": 0}

    def _fake_status(_job_id: str):
        calls["n"] += 1
        if calls["n"] < 2:
            return {"state": "running", "mcp_ready": False}
        return {"state": "ready", "analysis_backend": "fallback-heuristic", "mcp_ready": True}

    monkeypatch.setattr(web_main, "_analysis_companion_status_for_job", _fake_status)
    monkeypatch.setattr(web_main, "_kubectl", lambda *a, **k: (0, '{"status":{"phase":"Running"}}', ""))
    monkeypatch.setattr(web_main.time, "sleep", lambda _s: None)

    out = web_main._k8s_wait_analysis_companion_result("job-1", "pod-1", timeout_sec=10)
    assert out["state"] == "ready"
    assert out["mcp_ready"] is True


def test_wait_analysis_companion_result_require_rag(monkeypatch: pytest.MonkeyPatch):
    calls = {"n": 0}

    def _fake_status(_job_id: str):
        calls["n"] += 1
        if calls["n"] < 3:
            return {"state": "ready", "mcp_ready": True, "rag_ok": False}
        return {"state": "ready", "mcp_ready": True, "rag_ok": True}

    monkeypatch.setattr(web_main, "_analysis_companion_status_for_job", _fake_status)
    monkeypatch.setattr(web_main, "_kubectl", lambda *a, **k: (0, '{"status":{"phase":"Running"}}', ""))
    monkeypatch.setattr(web_main.time, "sleep", lambda _s: None)

    out = web_main._k8s_wait_analysis_companion_result(
        "job-1",
        "pod-1",
        timeout_sec=10,
        require_rag=True,
    )
    assert out["state"] == "ready"
    assert out["mcp_ready"] is True
    assert out["rag_ok"] is True


def test_wait_analysis_companion_result_times_out(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(web_main, "_analysis_companion_status_for_job", lambda _job_id: {})
    monkeypatch.setattr(web_main, "_kubectl", lambda *a, **k: (0, '{"status":{"phase":"Running"}}', ""))
    now = {"t": 0.0}

    def _fake_time():
        now["t"] += 1.5
        return now["t"]

    monkeypatch.setattr(web_main.time, "time", _fake_time)
    monkeypatch.setattr(web_main.time, "sleep", lambda _s: None)

    with pytest.raises(TimeoutError):
        web_main._k8s_wait_analysis_companion_result("job-1", "pod-1", timeout_sec=2)


def test_k8s_start_analysis_companion_reuses_running_resources(monkeypatch: pytest.MonkeyPatch):
    job_id = "job-reuse-123"
    expected_name = web_main._k8s_analysis_companion_name(job_id)
    calls: list[list[str]] = []

    def _fake_kubectl(args, **_kwargs):
        calls.append(list(args))
        if args[:3] == ["get", "pod", expected_name]:
            return (
                0,
                '{"status":{"phase":"Running"}}',
                "",
            )
        if args[:3] == ["get", "service", expected_name]:
            return (0, "ok", "")
        raise AssertionError(f"unexpected kubectl call: {args}")

    monkeypatch.setattr(web_main, "_kubectl", _fake_kubectl)
    monkeypatch.setattr(web_main, "_k8s_analysis_companion_enabled", lambda: True)

    pod, svc, url = web_main._k8s_start_analysis_companion(job_id)
    assert pod == expected_name
    assert svc == expected_name
    assert expected_name in url
    # In reuse path there should be no delete/apply calls.
    assert all(cmd[0] == "get" for cmd in calls)


def test_run_fuzz_job_reuses_analysis_context_on_reentry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo_root = tmp_path / "repo"
    (repo_root / "fuzz").mkdir(parents=True, exist_ok=True)
    analysis_context = repo_root / "fuzz" / "analysis_context.json"

    request = web_main.fuzz_model(
        code_url="https://github.com/example/repo.git",
        max_tokens=0,
        total_duration=-1,
        single_duration=-1,
    )
    cfg = web_main.WebPersistentConfig()

    latest_job: dict[str, object] = {}

    def _fake_job_update(_job_id: str, **kwargs):
        latest_job.update(kwargs)

    monkeypatch.setattr(web_main, "_job_update", _fake_job_update)
    monkeypatch.setattr(web_main, "_job_log_path", lambda _job_id: tmp_path / "job.log")
    monkeypatch.setattr(web_main, "_is_cancel_requested", lambda _job_id: False)
    monkeypatch.setattr(web_main, "_resolve_job_docker_policy", lambda _request, _cfg: (False, ""))
    monkeypatch.setattr(web_main, "_executor_mode", lambda: "k8s_job")
    monkeypatch.setattr(web_main, "_k8s_stage_wait_timeout_sec", lambda **_kwargs: 30)
    monkeypatch.setattr(web_main, "_k8s_node_can_run_job", lambda _node: (True, "node_ready"))
    monkeypatch.setattr(web_main, "_k8s_analysis_companion_enabled", lambda: False)

    dispatched_stages: list[str] = []
    plan_count = {"n": 0}

    def _fake_execute_k8s_job(*, payload, **_kwargs):
        stage = str(payload.get("stop_after_step") or "")
        dispatched_stages.append(stage)
        if stage == "analysis":
            analysis_context.write_text('{"generated_at": 1}\n', encoding="utf-8")
            return (
                {
                    "repo_root": str(repo_root),
                    "workflow_recommended_next": "plan",
                    "analysis_done": True,
                },
                "node-a",
            )
        if stage == "plan":
            plan_count["n"] += 1
            next_step = "analysis" if plan_count["n"] == 1 else "stop"
            return (
                {
                    "repo_root": str(repo_root),
                    "workflow_recommended_next": next_step,
                },
                "node-a",
            )
        raise AssertionError(f"unexpected stage dispatched: {stage}")

    monkeypatch.setattr(web_main, "_execute_k8s_job", _fake_execute_k8s_job)

    web_main._run_fuzz_job("job-analysis-reuse-1", request, cfg, resumed=False, trigger="new")

    assert dispatched_stages == ["analysis", "plan", "plan"]
    result = latest_job.get("result")
    assert isinstance(result, dict)
    stage_results = result.get("stage_results")
    assert isinstance(stage_results, list)
    reused_entries = [
        row
        for row in stage_results
        if isinstance(row, dict)
        and str(row.get("stage") or "") == "analysis"
        and bool((row.get("result") or {}).get("analysis_reused"))
    ]
    assert len(reused_entries) == 1
    assert str(latest_job.get("status") or "") == "success"
