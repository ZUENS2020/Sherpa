from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
SRC_DIR = ROOT / "harness_generator" / "src"
for p in (APP_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import main as web_main


def test_build_stage_payload_populates_context_and_paths(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    error_path = tmp_path / "error.log"
    payload = web_main._build_stage_payload(
        job_id="j1",
        repo_url="https://example.com/repo.git",
        max_tokens=4096,
        total_time_budget_value=3600,
        run_time_budget_value=1800,
        coverage_loop_max_rounds=10,
        max_fix_rounds=3,
        same_error_max_retries=2,
        email="u@example.com",
        docker_image="native",
        model_value="GLM-5",
        stage="plan",
        current_repo_root=str(tmp_path / "repo"),
        context_dir=str(tmp_path / "repo/fuzz/context"),
        control_ctx={"run_timeout_budget_sec_override": "1200"},
        unlimited_round_limit_value=7200,
        companion_url="http://companion:8080",
        companion_mcp_ready=True,
        result_path=result_path,
        error_path=error_path,
        current_node_name="node-a",
        can_pin_node=True,
    )

    assert payload["job_id"] == "j1"
    assert payload["resume_from_step"] == "plan"
    assert payload["context_dir"] == str(tmp_path / "repo/fuzz/context")
    assert payload["result_path"] == str(result_path)
    assert payload["error_path"] == str(error_path)
    assert payload["target_node_name"] == "node-a"
    assert int(payload["run_unlimited_round_budget_sec"]) == 1200


def test_handle_k8s_job_failure_oom_retries_once(monkeypatch) -> None:
    monkeypatch.setenv("SHERPA_RUN_RSS_LIMIT_MB", "8192")
    control_ctx: dict[str, object] = {}
    failure = web_main._K8sJobFailure("boom", result={"error_code": "oom_killed"})
    stage_result, stage_failed, stage_fail_error, stage_fail_reason = web_main._handle_k8s_job_failure(
        stage="run",
        failure=failure,
        control_ctx=control_ctx,
        current_repo_root="/tmp/repo",
        job_id="j1",
    )

    assert stage_failed is False
    assert stage_fail_error == ""
    assert stage_fail_reason == ""
    assert stage_result["workflow_recommended_next"] == "run"
    assert stage_result["restart_to_plan"] is False
    assert control_ctx["run_oom_retry_count"] == "1"
    assert control_ctx["run_parallel_fuzzers_override"] == "1"
    assert int(str(control_ctx["run_rss_limit_mb_override"])) > 0


def test_handle_stage_dispatch_exception_timeout_retries(monkeypatch) -> None:
    monkeypatch.setenv("SHERPA_K8S_TIMEOUT_MAX_RETRIES", "2")
    control_ctx: dict[str, object] = {}
    stage_result, stage_failed, stage_fail_error, stage_fail_reason = web_main._handle_stage_dispatch_exception(
        stage="run",
        exc=RuntimeError("k8s_job_timeout: wait exceeded"),
        control_ctx=control_ctx,
        wait_timeout=600,
        wait_override_key="run_timeout_wait_sec_override",
        unlimited_round_limit_value=7200,
        current_repo_root="/tmp/repo",
        job_id="j1",
    )

    assert stage_failed is False
    assert stage_fail_error == ""
    assert stage_fail_reason == ""
    assert stage_result["workflow_recommended_next"] == "run"
    assert stage_result["restart_to_plan"] is False
    assert control_ctx["run_timeout_retry_count"] == "1"
    assert int(str(control_ctx["run_timeout_wait_sec_override"])) > 600
    assert int(str(control_ctx["run_timeout_budget_sec_override"])) > 7200


def test_finalize_stage_result_merges_context_and_clears_restart_fields(monkeypatch) -> None:
    monkeypatch.setattr(web_main, "context_dir_for_repo_root", lambda _: "")
    control_ctx: dict[str, object] = {}
    workflow_ctx: dict[str, object] = {
        "restart_to_plan_reason": "old",
        "restart_to_plan_stage": "run",
        "restart_to_plan_error_text": "err",
        "restart_to_plan_report_path": "/tmp/r.txt",
    }
    stage_result = {
        "repo_root": "/tmp/repo",
        "workflow_recommended_next": "run",
        "restart_to_plan": False,
        "coverage_should_improve": True,
    }

    repo_root, context_dir, next_control, next_workflow, stage_record = web_main._finalize_stage_result(
        stage_result=stage_result,
        stage="coverage-analysis",
        job_name="job-x",
        stage_failed=False,
        current_repo_root="/tmp/repo",
        context_dir="",
        control_ctx=control_ctx,
        workflow_ctx=workflow_ctx,
        job_id="j1",
    )

    assert repo_root == "/tmp/repo"
    assert context_dir == ""
    assert next_workflow["restart_to_plan_reason"] == ""
    assert next_workflow["restart_to_plan_stage"] == ""
    assert next_workflow["restart_to_plan_error_text"] == ""
    assert next_workflow["restart_to_plan_report_path"] == ""
    assert stage_record["ok"] is True
    assert stage_record["stage"] == "coverage-analysis"
    assert stage_record["result"] == stage_result


def test_finalize_stage_result_non_dict_keeps_minimal_record() -> None:
    repo_root, context_dir, next_control, next_workflow, stage_record = web_main._finalize_stage_result(
        stage_result="ok",
        stage="plan",
        job_name="job-x",
        stage_failed=False,
        current_repo_root="/tmp/repo",
        context_dir="",
        control_ctx={},
        workflow_ctx={},
        job_id="j1",
    )

    assert repo_root == "/tmp/repo"
    assert context_dir == ""
    assert next_control == {}
    assert next_workflow == {}
    assert stage_record == {
        "stage": "plan",
        "job_name": "job-x",
        "ok": True,
        "repo_root": "/tmp/repo",
    }


def test_next_stage_from_result_normalizes_and_handles_non_dict() -> None:
    assert web_main._next_stage_from_result({"workflow_recommended_next": "per-input-replay"}) == "per-input-replay"
    assert web_main._next_stage_from_result({"workflow_recommended_next": "coverage-analysis"}) == "coverage-analysis"
    assert web_main._next_stage_from_result({"workflow_recommended_next": "coverage_analysis"}) == "analysis"
    assert web_main._next_stage_from_result({"workflow_recommended_next": ""}) == ""
    assert web_main._next_stage_from_result("x") == ""


def test_update_stage_node_pin_returns_expected_value() -> None:
    assert (
        web_main._update_stage_node_pin(
            stage="plan",
            stage_node_name="node-a",
            current_node_name="",
            job_id="j1",
        )
        == "node-a"
    )
    assert (
        web_main._update_stage_node_pin(
            stage="plan",
            stage_node_name="",
            current_node_name="node-a",
            job_id="j1",
        )
        == "node-a"
    )
