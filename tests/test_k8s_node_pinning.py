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

    timeout = web_main._k8s_stage_wait_timeout_sec(
        stage="run",
        total_time_budget_sec=0,
        run_time_budget_sec=1800,
        run_fuzzer_count=8,
        run_parallelism=2,
    )

    assert timeout == 2400


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
