from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import workflow_context_store as store


def test_merge_result_into_contexts_persists_run_and_coverage_fields() -> None:
    control = {"run_parallel_fuzzers_override": "1"}
    workflow = {}
    result = {
        "run_details": [{"fuzzer": "demo_fuzz", "final_cov": 12}],
        "coverage_seed_quality": 0.42,
        "coverage_quality_flags": ["low_early_yield"],
        "coverage_seed_generation_degraded": True,
        "coverage_should_improve": True,
        "coverage_improve_mode": "seed_replan",
        "coverage_replan_required": True,
        "cold_start_seed_replan_triggered": True,
        "degraded_seed_replan_triggered": False,
        "run_rss_limit_mb_override": "98304",
    }
    merged_control, merged_workflow = store.merge_result_into_contexts(
        result, control=control, workflow=workflow
    )

    assert merged_control["run_rss_limit_mb_override"] == "98304"
    assert merged_workflow["run_details"][0]["fuzzer"] == "demo_fuzz"
    assert merged_workflow["coverage_seed_quality"] == 0.42
    assert merged_workflow["coverage_quality_flags"] == ["low_early_yield"]
    assert merged_workflow["coverage_seed_generation_degraded"] is True
    assert merged_workflow["coverage_should_improve"] is True
    assert merged_workflow["coverage_improve_mode"] == "seed_replan"
    assert merged_workflow["coverage_replan_required"] is True
    assert merged_workflow["cold_start_seed_replan_triggered"] is True
    assert merged_workflow["degraded_seed_replan_triggered"] is False


def test_write_read_context_docs_keep_control_workflow_boundary(tmp_path: Path) -> None:
    context_dir = tmp_path / "fuzz" / "context"
    control = {
        "run_parallel_fuzzers_override": "1",
        "run_timeout_budget_sec_override": "900",
    }
    workflow = {
        "coverage_quality_flags": ["missing_required_families"],
        "run_details": [{"fuzzer": "demo_fuzz"}],
        "decision_trace_count": 5,
    }
    store.write_context_docs(
        context_dir,
        control=control,
        workflow=workflow,
        job_id="job-ctx",
    )
    read_control, read_workflow = store.read_context_docs(context_dir, job_id="job-ctx")
    read_control = store.strip_meta(read_control)
    read_workflow = store.strip_meta(read_workflow)

    assert read_control["run_parallel_fuzzers_override"] == "1"
    assert read_control["run_timeout_budget_sec_override"] == "900"
    assert "coverage_quality_flags" not in read_control
    assert read_workflow["coverage_quality_flags"] == ["missing_required_families"]
    assert read_workflow["run_details"][0]["fuzzer"] == "demo_fuzz"
    assert read_workflow["decision_trace_count"] == 5
    assert "run_parallel_fuzzers_override" not in read_workflow


def test_context_store_rejects_generator_key() -> None:
    merged_control, merged_workflow = store.merge_result_into_contexts(
        {
            "generator": object(),
            "coverage_should_improve": True,
        },
        control={},
        workflow={},
    )
    assert "generator" not in merged_control
    assert "generator" not in merged_workflow
    assert merged_workflow["coverage_should_improve"] is True
