"""Carved from workflow_graph.py - core workflow state types and low-level runtime helpers. Leaf module: stdlib + sibling helpers only."""

from __future__ import annotations
from loguru import logger
import hashlib
import importlib
import json
import os
import re
import subprocess
import tempfile
import textwrap
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypedDict, cast
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from persistent_config import load_config
import workflow_common as _wf_common
import workflow_observability as _wf_obs
import workflow_coverage_decision as _wf_coverage_decision
import workflow_normalization as _wf_norm
import workflow_target_scoring as _wf_target_scoring
import workflow_target_selection as _wf_target_selection
import workflow_summary as _wf_summary
import procedural_memory as _proc_mem
from coverage_replay import collect_per_input_frontier
from seed_families import seed_families_for_target as _seed_families_for_target
from workflow_context_store import (
    context_dir_for_repo_root,
    merge_result_into_contexts,
    read_context_docs,
    strip_meta,
    write_context_docs,
)
from fuzz_unharnessed_repo import (
    FuzzerRunResult,
    HarnessGeneratorError,
    NonOssFuzzHarnessGenerator,
    RepoSpec,
    extract_crash_stack_signature,
    snapshot_repo_text,
    write_patch_from_snapshot,
)


_RECOVERABLE_RUN_ERROR_KINDS = {
    "run_no_progress",
    "run_seed_rejected",
    "run_idle_timeout",
    "run_timeout",
    "run_finalize_timeout",
    "run_resource_exhaustion",
    "dict_parse_error",
    "run_stalled_after_coverage",
}


_FATAL_RUN_ERROR_KINDS = {
    "run_exception",
    "nonzero_exit_without_crash",
    "workflow_time_budget_exceeded",
}


class FuzzWorkflowState(TypedDict, total=False):
    repo_url: str
    model: str
    email: Optional[str]
    time_budget: int
    run_time_budget: int
    max_len: int
    docker_image: Optional[str]
    ai_key_path: str
    workflow_started_at: float
    resume_from_step: str
    resume_repo_root: str
    stop_after_step: str
    coverage_loop_max_rounds: int
    coverage_loop_round: int
    coverage_should_improve: bool
    coverage_improve_reason: str
    coverage_history: list[dict[str, Any]]
    coverage_target_name: str
    coverage_target_api: str
    coverage_seed_profile: str
    coverage_target_depth_score: int
    coverage_target_depth_class: str
    coverage_selection_bias_reason: str
    coverage_target_score_breakdown: dict[str, Any]
    coverage_plateau_streak: int
    coverage_last_max_cov: int
    coverage_last_ft: int
    coverage_replan_required: bool
    coverage_replan_effective: bool
    coverage_replan_reason: str
    coverage_improve_mode: str
    coverage_round_budget_exhausted: bool
    coverage_stop_reason: str
    coverage_corpus_sources: list[str]
    coverage_seed_counts: dict[str, int]
    coverage_seed_counts_raw: dict[str, int]
    coverage_seed_counts_filtered: dict[str, int]
    coverage_seed_noise_rejected_count: int
    coverage_seed_generation_failed_fuzzers: list[str]
    coverage_seed_generation_error_by_fuzzer: dict[str, str]
    coverage_seed_generation_failed_count: int
    coverage_seed_generation_degraded: bool
    coverage_missing_execution_targets: list[str]
    coverage_seed_family_coverage: dict[str, Any]
    coverage_seed_feedback: dict[str, Any]
    coverage_harness_feedback: dict[str, Any]
    coverage_quality_oracle: str
    coverage_bottleneck_kind: str
    coverage_bottleneck_reason: str
    coverage_parallel_diagnosis_code: str
    coverage_parallel_diagnosis: str
    coverage_parallel_engine: str
    coverage_parallel_outer: int
    coverage_parallel_inner: int
    coverage_parallel_cpu_budget: int
    coverage_parallel_utilization_ratio: float
    coverage_total_execs_per_sec: int
    coverage_underutilized_execs_threshold: int
    coverage_run_error_kind_effective: str
    coverage_repo_examples_filtered: bool
    coverage_repo_examples_rejected_count: int
    coverage_repo_examples_accepted_count: int
    coverage_per_input_manifest_path: str
    coverage_frontier_path: str
    coverage_frontier_summary: dict[str, Any]
    coverage_replay_runtime_sec: float
    coverage_replay_binary_hash: str
    coverage_replay_binary_dir: str
    coverage_replay_binary_count: int
    coverage_replay_stage_success: bool
    coverage_replay_error: str
    coverage_replay_manifest_fresh_for_current_binary: bool
    coverage_replay_queue_drained: bool
    coverage_replay_pending_inputs: int
    coverage_replay_failed_inputs: int
    coverage_replay_processed_inputs: int
    coverage_replay_total_inputs: int
    coverage_source_report: dict[str, Any]
    coverage_uncovered_functions: list[str]
    coverage_run_feedback_path: str
    coverage_run_feedback_summary: dict[str, Any]
    coverage_exhausted_targets: list[str]
    coverage_attempted_targets: list[str]
    coverage_feedback_for_plan: str
    cold_start_seed_replan_triggered: bool
    cold_start_trigger_snapshot: dict[str, Any]
    auto_stop_policy: str
    auto_stop_blocked_reason: str
    continuous_loop_count: int
    target_score_breakdown_available: bool
    crash_stack_signature: str
    crash_stack_type: str
    crash_stack_top_frames: str
    dry_run_result: dict[str, Any]
    seed_pre_check_result: dict[str, Any]
    antlr_context_path: str
    antlr_context_summary: str
    target_analysis_path: str
    target_analysis_summary: str
    analysis_context_path: str
    analysis_done: bool
    analysis_degraded: bool
    analysis_error: str
    analysis_report_path: str
    analysis_evidence_count: int
    selected_targets_path: str
    execution_plan_path: str
    harness_index_path: str
    repo_understanding_path: str
    build_strategy_path: str
    build_mode: str
    build_target_source: str
    selected_target_api: str
    selected_target_runtime_viability: str
    coverage_seed_quality: dict[str, Any]
    coverage_seed_families_suggested: list[str]
    coverage_seed_families_covered: list[str]
    coverage_seed_families_missing: list[str]
    coverage_quality_flags: list[str]
    degraded_seed_replan_triggered: bool
    plan_retry_reason: str
    plan_targets_schema_valid_before_retry: bool
    plan_targets_schema_valid_after_retry: bool
    plan_used_fallback_targets: bool
    replan_effective: bool
    replan_stop_reason: str
    vuln_hunting_enabled: bool
    vuln_hunt_iteration: int
    vuln_hunt_highest_priority: float
    vuln_hunt_enabled: bool
    vuln_hunt_active_candidate_id: str
    vuln_hunt_candidate_count: int
    vuln_hunt_degraded: bool
    vuln_hunt_last_reason: str
    vuln_hunt_summary_path: str
    vuln_hunt_events_path: str
    vuln_hunt_rerun_requested: bool
    _vuln_hunt_entry_source: str
    vuln_focus_profile: str
    target_surface_policy: str
    security_evidence_count: int
    vuln_candidate_count: int
    security_priority_mode: bool
    latest_vuln_decision_snapshot: dict[str, Any]

    step_count: int
    max_steps: int
    last_step: str
    last_error: str
    build_rc: int
    build_stdout_tail: str
    build_stderr_tail: str
    build_full_log_path: str
    build_template_cache_path: str
    build_error_signature: str
    build_error_signature_before: str
    build_error_signature_after: str
    same_build_error_repeats: int
    same_error_max_retries: int
    build_error_kind: str
    build_error_code: str
    build_error_signature_short: str
    build_attempts: int
    fix_build_attempts: int
    max_fix_rounds: int
    fix_build_noop_streak: int
    fix_build_attempt_history: list[dict[str, Any]]
    fix_build_rule_hits: list[str]
    fix_build_terminal_reason: str
    fix_build_last_diff_paths: list[str]
    fix_action_type: str
    fix_effect: str
    codex_hint: str
    failed: bool
    repo_root: str
    run_rc: int
    crash_evidence: str
    run_error_kind: str
    run_terminal_reason: str
    run_idle_seconds: int
    synthesize_selected_target_name: str
    synthesize_selected_target_api: str
    synthesize_observed_target_api: str
    synthesize_observed_harness: str
    synthesize_target_drifted: bool
    synthesize_target_drift_reason: str
    synthesize_target_relation: str
    synthesize_target_runtime_viability: str
    run_children_exit_count: int
    run_cancel_requested_count: int
    run_cancel_effective_count: int
    run_parallel_engine: str
    run_parallel_outer: int
    run_parallel_inner: int
    run_parallel_cpu_budget: int
    run_details: list[dict[str, Any]]
    run_batch_plan: list[dict[str, Any]]
    first_crash_fuzzer: str
    early_stop_reason: str
    early_stopped_fuzzers: list[str]
    last_crash_artifact: str
    last_fuzzer: str
    crash_signature: str
    same_crash_repeats: int
    timeout_signature: str
    same_timeout_repeats: int
    crash_fix_attempts: int
    crash_repro_done: bool
    crash_repro_ok: bool
    crash_repro_rc: int
    crash_repro_report_path: str
    crash_repro_json_path: str
    crash_triage_done: bool
    crash_triage_label: str
    crash_triage_confidence: float
    crash_triage_reason: str
    crash_triage_signal_lines: list[str]
    crash_triage_report_path: str
    crash_triage_json_path: str
    crash_analysis_done: bool
    crash_analysis_verdict: str
    crash_analysis_reason: str
    crash_analysis_report_path: str
    crash_analysis_json_path: str
    vuln_candidates_path: str
    crash_vuln_report_path: str
    latest_crash_vuln_candidate: dict[str, Any]
    crash_vuln_candidate_count: int
    re_build_done: bool
    re_build_ok: bool
    re_build_rc: int
    re_build_report_path: str
    re_build_json_path: str
    re_run_done: bool
    re_run_ok: bool
    re_run_rc: int
    re_run_report_path: str
    re_run_json_path: str
    re_workspace_root: str
    restart_to_plan: bool
    restart_to_plan_reason: str
    restart_to_plan_stage: str
    restart_to_plan_error_text: str
    restart_to_plan_report_path: str
    restart_to_plan_count: int
    fix_harness_attempts: int
    next: str
    fix_patch_path: str
    fix_patch_files: list[str]
    fix_patch_bytes: int
    summary_path: str
    summary_json_path: str
    plan_fix_on_crash: bool
    plan_max_fix_rounds: int
    repair_mode: bool
    repair_origin_stage: str
    repair_error_kind: str
    repair_error_code: str
    repair_signature: str
    repair_stdout_tail: str
    repair_stderr_tail: str
    repair_recent_attempts: list[dict[str, Any]]
    repair_error_digest: dict[str, Any]
    repair_attempt_index: int
    repair_strategy_force_change: bool
    target_scoring_enabled: bool
    target_score_breakdown_available: bool
    constraint_memory_count: int
    constraint_memory_path: str
    decision_traces: list[dict[str, Any]]
    decision_trace_count: int
    latest_decision_snapshot: dict[str, Any]
    crash_signature_dedup_hit: bool
    error: dict[str, Any]


class FuzzWorkflowRuntimeState(FuzzWorkflowState, total=False):
    generator: NonOssFuzzHarnessGenerator
    crash_found: bool
    message: str


def _has_error_payload(err: dict[str, Any] | None) -> bool:
    if not isinstance(err, dict):
        return False
    return bool(
        str(err.get("code") or "").strip()
        or str(err.get("message") or "").strip()
        or bool(err.get("terminal"))
    )


def _coerce_error_payload(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    err = {
        "stage": str(raw.get("stage") or "").strip().lower(),
        "kind": str(raw.get("kind") or "").strip().lower(),
        "code": str(raw.get("code") or "").strip().lower(),
        "message": str(raw.get("message") or "").strip(),
        "detail": str(raw.get("detail") or "").strip(),
        "signature": str(raw.get("signature") or "").strip(),
        "retryable": bool(raw.get("retryable")),
        "terminal": bool(raw.get("terminal")),
        "at": int(raw.get("at") or 0),
    }
    if err["at"] <= 0:
        err["at"] = int(time.time())
    return err


def _derive_error_from_legacy(state: dict[str, Any]) -> dict[str, Any]:
    stage = str(state.get("last_step") or "").strip().lower()
    code = str(
        state.get("build_error_code")
        or state.get("run_error_kind")
        or state.get("restart_to_plan_reason")
        or state.get("error_code")
        or ""
    ).strip().lower()
    kind = str(
        state.get("build_error_kind")
        or state.get("run_error_kind")
        or state.get("repair_error_kind")
        or state.get("error_kind")
        or ""
    ).strip().lower()
    message = str(state.get("last_error") or "").strip()
    if not message and bool(state.get("failed")):
        message = str(state.get("message") or "").strip()
    signature = str(
        state.get("build_error_signature_short")
        or state.get("build_error_signature")
        or state.get("timeout_signature")
        or state.get("crash_signature")
        or state.get("error_signature")
        or ""
    ).strip()
    terminal = bool(state.get("failed"))
    if not code and (message or terminal):
        code = "unknown_error"
    if not kind and code:
        if code.startswith("run_"):
            kind = "run"
        elif code.startswith("build_") or "build" in code:
            kind = "build"
        elif "crash" in code:
            kind = "crash"
        elif "timeout" in code:
            kind = "timeout"
        else:
            kind = "generic_failure"
    retryable = bool(code) and not terminal
    return {
        "stage": stage,
        "kind": kind,
        "code": code,
        "message": message,
        "detail": message,
        "signature": signature,
        "retryable": retryable,
        "terminal": terminal,
        "at": int(time.time()),
    }


def _project_error_legacy_fields(state: dict[str, Any], err: dict[str, Any]) -> dict[str, Any]:
    out = dict(state)
    if not _has_error_payload(err):
        return out
    code = str(err.get("code") or "").strip().lower()
    kind = str(err.get("kind") or "").strip().lower()
    message = str(err.get("message") or "").strip()
    signature = str(err.get("signature") or "").strip()
    stage = str(err.get("stage") or "").strip().lower()
    if message:
        out["last_error"] = message
    out["error_code"] = code
    out["error_kind"] = kind
    if signature:
        out["error_signature"] = signature
    if not str(out.get("repair_error_code") or "").strip() and code:
        out["repair_error_code"] = code
    if not str(out.get("repair_error_kind") or "").strip() and kind:
        out["repair_error_kind"] = kind
    if bool(out.get("restart_to_plan")) and not str(out.get("restart_to_plan_error_text") or "").strip() and message:
        out["restart_to_plan_error_text"] = message
    if stage == "run":
        if code and not str(out.get("run_error_kind") or "").strip():
            out["run_error_kind"] = code
        if code and not str(out.get("run_terminal_reason") or "").strip():
            out["run_terminal_reason"] = code
    if stage == "build" or kind == "build":
        if code and not str(out.get("build_error_code") or "").strip():
            out["build_error_code"] = code
        if kind and not str(out.get("build_error_kind") or "").strip():
            out["build_error_kind"] = kind
        if signature and not str(out.get("build_error_signature_short") or "").strip():
            out["build_error_signature_short"] = signature[:12]
    return out


def _normalize_error_state(state: dict[str, Any]) -> dict[str, Any]:
    out = dict(state)
    existing = _coerce_error_payload(out.get("error"))
    derived = _derive_error_from_legacy(out)
    if _has_error_payload(existing):
        err = {**derived, **existing}
    else:
        err = derived
    if _has_error_payload(err):
        out["error"] = err
        out = _project_error_legacy_fields(out, err)
    else:
        out["error"] = {}
    return out


def _wf_log(state: dict[str, Any] | None, msg: str) -> None:
    _wf_common.wf_log(state, msg)


def _fmt_dt(seconds: float) -> str:
    return _wf_common.fmt_dt(seconds)


def _sha256_text(text: str) -> str:
    return _wf_common.sha256_text(text)


def _bounded_float(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except Exception:
        return float(default)


def _opencode_defunct_threshold() -> int:
    raw = (os.environ.get("SHERPA_OPENCODE_DEFUNCT_THRESHOLD") or "3").strip()
    try:
        return max(0, min(int(raw), 200))
    except Exception:
        return 3


def _count_opencode_defunct_processes() -> int:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "stat=,args="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return 0
    if int(proc.returncode or 0) != 0:
        return 0
    count = 0
    for raw_line in str(proc.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        stat = parts[0] if parts else ""
        cmd = parts[1] if len(parts) > 1 else ""
        cmd_l = cmd.lower()
        if "opencode" not in cmd_l:
            continue
        if "<defunct>" in cmd_l or stat.startswith("Z"):
            count += 1
    return count


def _enter_step(state: FuzzWorkflowRuntimeState, step_name: str) -> tuple[FuzzWorkflowRuntimeState, bool]:
    normalized_in = _normalize_error_state(cast(dict[str, Any], state))
    out, stop = _wf_common.enter_step(normalized_in, step_name)
    next_state = cast(FuzzWorkflowRuntimeState, _normalize_error_state(out))
    if stop:
        return next_state, stop
    defunct_count = _count_opencode_defunct_processes()
    next_state = cast(FuzzWorkflowRuntimeState, {**next_state, "opencode_defunct_count": defunct_count})
    threshold = _opencode_defunct_threshold()
    if threshold > 0 and defunct_count > threshold:
        msg = (
            f"opencode defunct process count exceeded threshold: "
            f"{defunct_count}>{threshold}; fail-fast to avoid stage hang"
        )
        guarded = cast(
            FuzzWorkflowRuntimeState,
            _normalize_error_state({
                **next_state,
                "last_step": step_name,
                "failed": True,
                "last_error": msg,
                "message": "workflow stopped (opencode defunct safeguard)",
                "error": {
                    "stage": step_name,
                    "kind": "infra",
                    "code": "opencode_defunct_safeguard",
                    "message": msg,
                    "detail": msg,
                    "signature": "",
                    "retryable": False,
                    "terminal": True,
                    "at": int(time.time()),
                },
            }),
        )
        _wf_log(cast(dict[str, Any], guarded), f"<- {step_name} stop=opencode-defunct count={defunct_count} threshold={threshold}")
        return guarded, True
    if defunct_count > 0:
        _wf_log(cast(dict[str, Any], next_state), f"{step_name}: opencode_defunct_count={defunct_count}")
    return next_state, False
