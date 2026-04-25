from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
SRC_DIR = ROOT / "harness_generator" / "src"
for p in (APP_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import workflow_coverage_decision as d


def _base_kwargs() -> dict:
    return {
        "run_error_kind": "",
        "crash_found": False,
        "failed": False,
        "recoverable_run_error_kinds": {"oom_killed", "k8s_job_timeout"},
        "plateau_detected": True,
        "current_cov": 10,
        "prev_cov": 10,
        "current_ft": 20,
        "prev_ft": 20,
        "prev_plateau_streak": 1,
        "current_seed_profile": "parser-token",
        "quality_flags": [],
        "seed_families_missing": [],
        "cold_start_failure": False,
        "seed_generation_degraded": False,
        "quality_score": 0.8,
        "cold_start_quality_threshold": 0.55,
        "early_new_units_30s": 5,
        "cold_start_early_units_threshold": 0,
        "merge_retained_low": False,
        "configured_parallel_units": 2,
        "parallel_cpu_budget": 4,
        "total_execs_per_sec": 2000,
        "underutilized_execs_threshold": 1500,
        "current_depth_class": "medium",
        "current_round": 2,
        "max_rounds": 10,
        "unlimited_rounds": False,
    }


def test_cold_start_triggers_seed_replan() -> None:
    kwargs = _base_kwargs()
    kwargs.update(
        {
            "cold_start_failure": True,
            "quality_score": 0.2,
            "early_new_units_30s": 0,
        }
    )
    out = d.evaluate_coverage_decision(**kwargs)
    assert out["cold_start_seed_replan_triggered"] is True
    assert out["should_improve"] is True
    assert out["replan_required"] is True
    assert out["improve_mode"] == "seed_replan"


def test_seed_quality_issue_prefers_in_place() -> None:
    kwargs = _base_kwargs()
    kwargs.update(
        {
            "plateau_detected": False,
            "quality_flags": ["low_early_yield"],
        }
    )
    out = d.evaluate_coverage_decision(**kwargs)
    assert out["seed_quality_issue"] is True
    assert out["should_improve"] is True
    assert out["improve_mode"] == "in_place"
    assert out["replan_required"] is False


def test_budget_exhausted_sets_stop_reason() -> None:
    kwargs = _base_kwargs()
    kwargs.update(
        {
            "current_round": 10,
            "max_rounds": 10,
            "quality_flags": ["low_early_yield"],
        }
    )
    out = d.evaluate_coverage_decision(**kwargs)
    assert out["should_improve"] is False
    assert out["round_budget_exhausted"] is True
    assert out["stop_reason"] == "coverage_loop_budget_exhausted"
