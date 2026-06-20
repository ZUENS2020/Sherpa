"""Carved from workflow_graph.py - '_node_coverage_analysis' LangGraph node."""

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

from wf_state import (
    FuzzWorkflowRuntimeState,
    _RECOVERABLE_RUN_ERROR_KINDS,
    _enter_step,
    _fmt_dt,
    _wf_log,
)
from workflow_helpers import (
    _append_vuln_hunt_event,
    _auto_stop_policy,
    _build_harness_feedback,
    _build_seed_feedback,
    _cold_start_seed_replan_early_units_30s_threshold,
    _cold_start_seed_replan_quality_threshold,
    _coverage_frontier_feedback_lines,
    _coverage_underutilized_execs_threshold,
    _effective_run_error_kind,
    _emit_fuzz_metrics,
    _execution_plan_targets,
    _execution_target_identity,
    _load_selected_targets_doc,
    _max_cov_from_run_details,
    _preferred_execution_target,
    _quality_flags_from_seed_quality,
    _resolve_current_coverage_binary,
    _seed_quality_from_run_details_for_target,
    _target_type_from_run_details,
    _update_vuln_candidate_feedback,
    _vuln_hunt_event_from_state,
    _write_run_feedback_artifact,
)


def _node_coverage_analysis(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    state, stop_now = _enter_step(state, "coverage-analysis")
    if stop_now:
        return state
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> coverage-analysis")
    try:
        analysis_repo_root = (
            Path(str(state.get("repo_root") or "").strip())
            if str(state.get("repo_root") or "").strip()
            else getattr(gen, "repo_root", Path.cwd())
        )
        max_rounds = max(
            0,
            int(
                state.get("coverage_loop_max_rounds")
                if state.get("coverage_loop_max_rounds") is not None
                else 0
            ),
        )
        current_round = max(0, int(state.get("coverage_loop_round") or 0))
        unlimited_rounds = max_rounds == 0
        run_details = list(state.get("run_details") or [])
        history = list(state.get("coverage_history") or [])
        current_cov = _max_cov_from_run_details(run_details)
        current_ft = 0
        current_target_name = str(state.get("coverage_target_name") or "")
        current_target_api = str(state.get("coverage_target_api") or "")
        current_target_type = str(state.get("coverage_target_type") or "")
        execution_targets = _execution_plan_targets(analysis_repo_root)
        preferred_target = _preferred_execution_target(
            execution_targets,
            cast(dict[str, Any], state),
            run_details=run_details,
        )
        preferred_identity = _execution_target_identity(preferred_target) if preferred_target else {}
        if preferred_identity:
            current_target_name = str(preferred_identity.get("target_name") or current_target_name)
            current_target_api = str(preferred_identity.get("target_api") or current_target_api)
            current_target_type = str(preferred_identity.get("target_type") or current_target_type)
        if not current_target_type:
            current_target_type = _target_type_from_run_details(
                run_details,
                target_name=current_target_name,
                target_api=current_target_api,
            )
        selected_target_score_breakdown: dict[str, Any] = {}
        try:
            for item in _load_selected_targets_doc(analysis_repo_root):
                item_api = str(item.get("api") or "").strip()
                item_name = str(item.get("target_name") or item.get("name") or "").strip()
                if current_target_api and item_api and item_api == current_target_api:
                    selected_target_score_breakdown = dict(
                        item.get("score_breakdown")
                        or item.get("target_score_breakdown")
                        or {}
                    )
                    break
                if current_target_name and item_name and item_name == current_target_name:
                    selected_target_score_breakdown = dict(
                        item.get("score_breakdown")
                        or item.get("target_score_breakdown")
                        or {}
                    )
                    break
        except Exception:
            selected_target_score_breakdown = {}
        if run_details:
            try:
                current_ft = max(int(detail.get("final_ft") or 0) for detail in run_details)
            except Exception:
                current_ft = 0
            if not current_target_name:
                current_target_name = str(run_details[0].get("fuzzer") or "")
        plateau_detected = any(bool(detail.get("plateau_detected")) for detail in run_details)
        plateau_idle_seconds = max(int(detail.get("plateau_idle_seconds") or 0) for detail in run_details) if run_details else 0
        prev_cov = max(0, int(state.get("coverage_last_max_cov") or 0))
        prev_ft = max(0, int(state.get("coverage_last_ft") or 0))
        best_cov = max(prev_cov, current_cov)
        best_ft = max(prev_ft, current_ft)
        prev_plateau_streak = max(0, int(state.get("coverage_plateau_streak") or 0))
        current_seed_profile = str(state.get("coverage_seed_profile") or "")
        current_depth_score = int(state.get("coverage_target_depth_score") or 0)
        current_depth_class = str(state.get("coverage_target_depth_class") or "")
        current_selection_bias_reason = str(state.get("coverage_selection_bias_reason") or "")
        seed_quality = _seed_quality_from_run_details_for_target(
            run_details,
            dict(state.get("coverage_seed_quality") or {}),
            target_name=current_target_name,
            target_api=current_target_api,
            fuzzer_name=str(preferred_identity.get("expected_fuzzer_name") or current_target_name),
        )
        seed_feedback_state = dict(state)
        seed_feedback_state["coverage_seed_quality"] = seed_quality
        seed_feedback_state["coverage_quality_flags"] = _quality_flags_from_seed_quality(seed_quality)
        seed_feedback = _build_seed_feedback(cast(dict[str, Any], seed_feedback_state))
        frontier_summary = dict(state.get("coverage_frontier_summary") or {})
        coverage_frontier_input_count = int(frontier_summary.get("top_input_count") or 0)
        coverage_replay_stage_success = bool(state.get("coverage_replay_stage_success") or False)
        coverage_replay_manifest_fresh = bool(
            state.get("coverage_replay_manifest_fresh_for_current_binary") or False
        )
        coverage_replay_queue_drained = bool(state.get("coverage_replay_queue_drained") or False)
        coverage_replay_pending_inputs = int(state.get("coverage_replay_pending_inputs") or 0)
        coverage_replay_failed_inputs = int(state.get("coverage_replay_failed_inputs") or 0)
        coverage_replay_error = str(state.get("coverage_replay_error") or "")
        quality_flags = _quality_flags_from_seed_quality(seed_quality)
        if not quality_flags and not seed_quality:
            quality_flags = list(state.get("coverage_quality_flags") or [])
        seed_families_suggested = list(state.get("coverage_seed_families_suggested") or [])
        seed_families_covered = list(state.get("coverage_seed_families_covered") or [])
        seed_families_missing = list(state.get("coverage_seed_families_missing") or [])
        if not current_seed_profile:
            for detail in run_details:
                profile = str(detail.get("seed_profile") or "")
                if profile:
                    current_seed_profile = profile
                    break
        total_execs_per_sec = 0
        try:
            total_execs_per_sec = max(0, sum(int(detail.get("final_execs_per_sec") or 0) for detail in run_details))
        except Exception:
            total_execs_per_sec = 0
        parallel_outer = max(1, int(state.get("run_parallel_outer") or 1))
        parallel_inner = max(1, int(state.get("run_parallel_inner") or 1))
        parallel_cpu_budget = max(1, int(state.get("run_parallel_cpu_budget") or 1))
        parallel_engine = str(state.get("run_parallel_engine") or "single")
        configured_parallel_units = max(1, parallel_outer * parallel_inner)
        parallel_utilization_ratio = float(configured_parallel_units) / float(max(1, parallel_cpu_budget))
        if parallel_utilization_ratio < 0.0:
            parallel_utilization_ratio = 0.0
        if parallel_utilization_ratio > 1.0:
            parallel_utilization_ratio = 1.0
        underutilized_execs_threshold = _coverage_underutilized_execs_threshold()
        cold_start_quality_threshold = _cold_start_seed_replan_quality_threshold()
        cold_start_early_units_threshold = _cold_start_seed_replan_early_units_30s_threshold()
        replan_reason = ""
        improve_mode = ""
        round_budget_exhausted = False
        stop_reason = ""
        run_error_kind_raw = str(state.get("run_error_kind") or "").strip().lower()
        run_error_kind = _effective_run_error_kind(cast(dict[str, Any], state)) or run_error_kind_raw
        cold_start_failure = bool(seed_feedback.get("cold_start_failure") or False)
        seed_generation_degraded = bool(state.get("coverage_seed_generation_degraded") or False)
        quality_score = float(seed_feedback.get("seed_score") or seed_quality.get("seed_score") or 0.0)
        attack_hint_missing_values = list(
            seed_feedback.get("attack_hint_missing_values")
            if seed_feedback.get("attack_hint_missing_values") is not None
            else seed_quality.get("attack_hint_missing_values")
            or []
        )
        attack_hint_coverage_ratio = float(
            seed_feedback.get("attack_hint_coverage_ratio")
            if seed_feedback.get("attack_hint_coverage_ratio") is not None
            else seed_quality.get("attack_hint_coverage_ratio")
            or 1.0
        )
        early_new_units_30s = int(
            seed_feedback.get("early_new_units_30s")
            if seed_feedback.get("early_new_units_30s") is not None
            else seed_quality.get("early_new_units_30s") or 0
        )
        merge_retained_ratio = float(seed_feedback.get("merge_retained_ratio_files") or 1.0)
        merge_retained_low = bool(merge_retained_ratio > 0.0 and merge_retained_ratio < 0.35)
        cold_start_seed_replan_triggered = bool(
            cold_start_failure
            and quality_score < cold_start_quality_threshold
            and early_new_units_30s <= cold_start_early_units_threshold
        )
        # Family-based flags (missing_suggested_families, seed_family_undercovered,
        # repo_examples_missing) are advisory and should NOT trigger replan.
        # Only actual runtime performance signals should drive replan decisions.
        degraded_seed_replan_triggered = bool(
            seed_generation_degraded
            and (
                quality_score < cold_start_quality_threshold
                or early_new_units_30s <= cold_start_early_units_threshold
                or any(
                    flag in quality_flags
                    for flag in {
                        "low_early_yield",
                        "missing_execution_targets",
                    }
                )
            )
        )
        decision = _wf_coverage_decision.evaluate_coverage_decision(
            run_error_kind=run_error_kind,
            crash_found=bool(state.get("crash_found")),
            failed=bool(state.get("failed")),
            recoverable_run_error_kinds=set(_RECOVERABLE_RUN_ERROR_KINDS),
            plateau_detected=plateau_detected,
            current_cov=current_cov,
            prev_cov=prev_cov,
            current_ft=current_ft,
            prev_ft=prev_ft,
            prev_plateau_streak=prev_plateau_streak,
            current_seed_profile=current_seed_profile,
            quality_flags=quality_flags,
            seed_families_missing=seed_families_missing,
            cold_start_failure=cold_start_failure,
            seed_generation_degraded=seed_generation_degraded,
            quality_score=quality_score,
            cold_start_quality_threshold=cold_start_quality_threshold,
            early_new_units_30s=early_new_units_30s,
            cold_start_early_units_threshold=cold_start_early_units_threshold,
            merge_retained_low=merge_retained_low,
            configured_parallel_units=configured_parallel_units,
            parallel_cpu_budget=parallel_cpu_budget,
            total_execs_per_sec=total_execs_per_sec,
            underutilized_execs_threshold=underutilized_execs_threshold,
            current_depth_class=current_depth_class,
            coverage_replay_stage_success=coverage_replay_stage_success,
            coverage_replay_manifest_fresh=coverage_replay_manifest_fresh,
            coverage_replay_queue_drained=coverage_replay_queue_drained,
            coverage_frontier_input_count=coverage_frontier_input_count,
            current_round=current_round,
            max_rounds=max_rounds,
            unlimited_rounds=unlimited_rounds,
        )
        plateau_no_gain = bool(decision.get("plateau_no_gain") or False)
        plateau_streak = int(decision.get("plateau_streak") or 0)
        requested_replan = bool(decision.get("requested_replan") or False)
        cold_start_seed_replan_triggered = bool(
            decision.get("cold_start_seed_replan_triggered") or False
        )
        degraded_seed_replan_triggered = bool(
            decision.get("degraded_seed_replan_triggered") or False
        )
        seed_quality_issue = bool(decision.get("seed_quality_issue") or False)
        parallel_diagnosis_code = str(decision.get("parallel_diagnosis_code") or "balanced")
        parallel_diagnosis = str(
            decision.get("parallel_diagnosis") or "parallelism looks balanced for current coverage signal"
        )
        quality_degraded = bool(
            decision.get("quality_degraded")
            or list(state.get("coverage_missing_execution_targets") or [])
        )
        quality_oracle = "quality_degraded" if quality_degraded else "ok"
        coverage_bottleneck_kind = str(decision.get("coverage_bottleneck_kind") or "none")
        coverage_bottleneck_reason = str(decision.get("coverage_bottleneck_reason") or "")
        auto_stop_policy = _auto_stop_policy()
        should_improve = bool(decision.get("should_improve") or False)
        replan_required = bool(decision.get("replan_required") or False)
        improve_mode = str(decision.get("improve_mode") or "")
        replan_reason = str(decision.get("replan_reason") or "")
        round_budget_exhausted = bool(decision.get("round_budget_exhausted") or False)
        stop_reason = str(decision.get("stop_reason") or "")

        next_round = current_round + (1 if should_improve else 0)
        reason = "skip coverage loop"
        if should_improve:
            if plateau_detected:
                round_budget_text = "unlimited" if unlimited_rounds else str(max_rounds)
                reason = (
                    f"coverage plateau detected; mode={improve_mode}; round={next_round}/{round_budget_text}, "
                    f"max_cov={current_cov}, prev_cov={prev_cov}, max_ft={current_ft}, prev_ft={prev_ft}, "
                    f"plateau_streak={plateau_streak}, idle_no_growth={plateau_idle_seconds}s"
                )
            else:
                round_budget_text = "unlimited" if unlimited_rounds else str(max_rounds)
                reason = (
                    f"mode={improve_mode or 'in_place'}; round={next_round}/{round_budget_text}, max_cov={current_cov}, prev_cov={prev_cov}, "
                    f"max_ft={current_ft}, prev_ft={prev_ft}"
                )
            if seed_quality_issue:
                reason += f"; seed_quality_flags={','.join(quality_flags) or 'none'}"
            if cold_start_failure:
                reason += "; cold_start_failure=1"
            if seed_generation_degraded:
                reason += "; seed_generation_degraded=1"
        if merge_retained_low:
            reason += f"; merge_retained_ratio_files={merge_retained_ratio:.2f}"
        if attack_hint_missing_values:
            reason += f"; attack_hint_missing_values={','.join(attack_hint_missing_values[:4])}"
        if coverage_replay_error:
            reason += f"; per_input_replay={coverage_replay_error}"
        elif coverage_replay_pending_inputs > 0:
            reason += f"; per_input_replay_pending={coverage_replay_pending_inputs}"
        if parallel_diagnosis_code != "balanced":
            reason += f"; parallel_diagnosis={parallel_diagnosis_code}"
            if coverage_bottleneck_kind != "none":
                reason += f"; bottleneck={coverage_bottleneck_kind}:{coverage_bottleneck_reason}"
        elif round_budget_exhausted:
            if requested_replan:
                reason = (
                    f"coverage plateau detected but replan budget exhausted; "
                    f"round={current_round}/{max_rounds if not unlimited_rounds else 'unlimited'}, max_cov={current_cov}, max_ft={current_ft}, "
                    f"plateau_streak={plateau_streak}"
                )
            else:
                reason = (
                    f"coverage loop budget exhausted; round={current_round}/{max_rounds if not unlimited_rounds else 'unlimited'}, "
                    f"max_cov={current_cov}, max_ft={current_ft}"
                )
        history.append(
            {
                "index": len(history) + 1,
                "round": next_round if should_improve else current_round,
                "max_rounds": max_rounds,
                "max_cov": current_cov,
                "max_ft": current_ft,
                "prev_cov": prev_cov,
                "prev_ft": prev_ft,
                "plateau_detected": plateau_detected,
                "plateau_idle_seconds": plateau_idle_seconds,
                "plateau_streak": plateau_streak,
                "seed_profile": current_seed_profile,
                "target_name": current_target_name,
                "target_api": current_target_api or current_target_name,
                "target_depth_score": current_depth_score,
                "target_depth_class": current_depth_class,
                "selection_bias_reason": current_selection_bias_reason,
                "replan_required": replan_required,
                "replan_effective": bool(state.get("coverage_replan_effective") or False),
                "replan_reason": replan_reason or str(state.get("coverage_replan_reason") or ""),
                "improve_mode": improve_mode,
                "round_budget_exhausted": round_budget_exhausted,
                "stop_reason": stop_reason,
                "corpus_sources": list(state.get("coverage_corpus_sources") or []),
                "seed_counts": dict(state.get("coverage_seed_counts") or {}),
                "seed_quality": seed_quality,
                "seed_families_suggested": seed_families_suggested,
                "seed_families_covered": seed_families_covered,
                "seed_families_missing": seed_families_missing,
                "attack_hint_coverage_ratio": attack_hint_coverage_ratio,
                "attack_hint_missing_values": attack_hint_missing_values,
                "quality_flags": quality_flags,
                "quality_oracle": quality_oracle,
                "coverage_frontier_summary": frontier_summary,
                "coverage_replay_stage_success": coverage_replay_stage_success,
                "coverage_replay_manifest_fresh_for_current_binary": coverage_replay_manifest_fresh,
                "coverage_replay_queue_drained": coverage_replay_queue_drained,
                "coverage_replay_pending_inputs": coverage_replay_pending_inputs,
                "coverage_replay_failed_inputs": coverage_replay_failed_inputs,
                "coverage_replay_error": coverage_replay_error,
                "coverage_bottleneck_kind": coverage_bottleneck_kind,
                "coverage_bottleneck_reason": coverage_bottleneck_reason,
                "parallel_diagnosis_code": parallel_diagnosis_code,
                "parallel_diagnosis": parallel_diagnosis,
                "parallel_engine": parallel_engine,
                "parallel_outer": parallel_outer,
                "parallel_inner": parallel_inner,
                "parallel_cpu_budget": parallel_cpu_budget,
                "parallel_utilization_ratio": parallel_utilization_ratio,
                "total_execs_per_sec": total_execs_per_sec,
                "underutilized_execs_threshold": underutilized_execs_threshold,
                "repo_examples_filtered": bool(state.get("coverage_repo_examples_filtered") or False),
                "repo_examples_rejected_count": int(state.get("coverage_repo_examples_rejected_count") or 0),
                "repo_examples_accepted_count": int(state.get("coverage_repo_examples_accepted_count") or 0),
                "crash_found": bool(state.get("crash_found")),
                "run_error_kind": str(state.get("run_error_kind") or ""),
                "run_error_kind_effective": run_error_kind,
                "cold_start_seed_replan_triggered": cold_start_seed_replan_triggered,
                "degraded_seed_replan_triggered": degraded_seed_replan_triggered,
                "cold_start_trigger_snapshot": {
                    "quality_score": round(quality_score, 6),
                    "quality_threshold": round(cold_start_quality_threshold, 6),
                    "early_new_units_30s": int(early_new_units_30s),
                    "early_units_30s_threshold": int(cold_start_early_units_threshold),
                    "cold_start_failure": bool(cold_start_failure),
                    "seed_generation_degraded": bool(seed_generation_degraded),
                },
                "should_improve": should_improve,
                "ts": int(time.time()),
            }
        )
        out = {
            **state,
            "last_step": "coverage-analysis",
            "last_error": "",
            "coverage_loop_max_rounds": max_rounds,
            "coverage_loop_round": next_round if should_improve else current_round,
            "coverage_should_improve": should_improve,
            "coverage_improve_reason": reason,
            "coverage_history": history,
            "coverage_target_name": current_target_name or str(state.get("coverage_target_name") or ""),
            "coverage_target_api": current_target_api or str(state.get("coverage_target_api") or ""),
            "coverage_target_type": current_target_type,
            "coverage_seed_profile": current_seed_profile,
            "coverage_seed_quality": seed_quality,
            "coverage_seed_families_suggested": seed_families_suggested,
            "coverage_seed_families_covered": seed_families_covered,
            "coverage_seed_families_missing": seed_families_missing,
            "coverage_quality_flags": quality_flags,
            "coverage_quality_oracle": quality_oracle,
            "coverage_bottleneck_kind": coverage_bottleneck_kind,
            "coverage_bottleneck_reason": coverage_bottleneck_reason,
            "coverage_parallel_diagnosis_code": parallel_diagnosis_code,
            "coverage_parallel_diagnosis": parallel_diagnosis,
            "coverage_parallel_engine": parallel_engine,
            "coverage_parallel_outer": parallel_outer,
            "coverage_parallel_inner": parallel_inner,
            "coverage_parallel_cpu_budget": parallel_cpu_budget,
            "coverage_parallel_utilization_ratio": parallel_utilization_ratio,
            "coverage_total_execs_per_sec": total_execs_per_sec,
            "coverage_underutilized_execs_threshold": underutilized_execs_threshold,
            "coverage_target_depth_score": current_depth_score,
            "coverage_target_depth_class": current_depth_class,
            "coverage_selection_bias_reason": current_selection_bias_reason,
            "coverage_target_score_breakdown": selected_target_score_breakdown,
            "coverage_plateau_streak": plateau_streak,
            "coverage_last_max_cov": best_cov,
            "coverage_last_ft": best_ft,
            "coverage_replan_required": replan_required,
            "coverage_replan_reason": replan_reason or str(state.get("coverage_replan_reason") or ""),
            "coverage_improve_mode": improve_mode,
            "coverage_round_budget_exhausted": round_budget_exhausted,
            "coverage_stop_reason": stop_reason,
            "coverage_repo_examples_filtered": bool(state.get("coverage_repo_examples_filtered") or False),
            "coverage_repo_examples_rejected_count": int(state.get("coverage_repo_examples_rejected_count") or 0),
            "coverage_repo_examples_accepted_count": int(state.get("coverage_repo_examples_accepted_count") or 0),
            "coverage_per_input_manifest_path": str(state.get("coverage_per_input_manifest_path") or ""),
            "coverage_frontier_path": str(state.get("coverage_frontier_path") or ""),
            "coverage_frontier_summary": frontier_summary,
            "coverage_replay_runtime_sec": float(state.get("coverage_replay_runtime_sec") or 0.0),
            "coverage_replay_binary_hash": str(state.get("coverage_replay_binary_hash") or ""),
            "coverage_replay_stage_success": coverage_replay_stage_success,
            "coverage_replay_error": coverage_replay_error,
            "coverage_replay_manifest_fresh_for_current_binary": coverage_replay_manifest_fresh,
            "coverage_replay_queue_drained": coverage_replay_queue_drained,
            "coverage_replay_pending_inputs": coverage_replay_pending_inputs,
            "coverage_replay_failed_inputs": coverage_replay_failed_inputs,
            "coverage_replay_processed_inputs": int(state.get("coverage_replay_processed_inputs") or 0),
            "coverage_replay_total_inputs": int(state.get("coverage_replay_total_inputs") or 0),
            "coverage_run_error_kind_effective": run_error_kind,
            "cold_start_seed_replan_triggered": cold_start_seed_replan_triggered,
            "degraded_seed_replan_triggered": degraded_seed_replan_triggered,
            "cold_start_trigger_snapshot": {
                "quality_score": round(quality_score, 6),
                "quality_threshold": round(cold_start_quality_threshold, 6),
                "early_new_units_30s": int(early_new_units_30s),
                "early_units_30s_threshold": int(cold_start_early_units_threshold),
                "cold_start_failure": bool(cold_start_failure),
                "seed_generation_degraded": bool(seed_generation_degraded),
            },
            "message": "coverage analysis done",
            "auto_stop_policy": auto_stop_policy,
            "auto_stop_blocked_reason": str(state.get("auto_stop_blocked_reason") or ""),
            "continuous_loop_count": int(state.get("continuous_loop_count") or 0),
            "target_score_breakdown_available": bool(
                selected_target_score_breakdown or state.get("target_score_breakdown_available")
            ),
        }
        if should_improve or current_cov > prev_cov:
            # Reset the circuit-breaker counter whenever the coverage loop
            # makes genuine progress so that a brief stall followed by new
            # coverage does not accumulate towards a spurious replan.
            out["continuous_loop_count"] = 0
        elif auto_stop_policy == "hard_fail_only" and (not bool(out.get("failed"))) and not str(out.get("last_error") or "").strip() and not bool(should_improve):
            out["auto_stop_blocked_reason"] = "coverage_no_improve"
            out["continuous_loop_count"] = int(out.get("continuous_loop_count") or 0) + 1
        out["coverage_seed_feedback"] = _build_seed_feedback(cast(dict[str, Any], out))
        out["coverage_harness_feedback"] = _build_harness_feedback(cast(dict[str, Any], out))
        hunt_event = _vuln_hunt_event_from_state(cast(dict[str, Any], out))
        if hunt_event:
            out["vuln_hunt_events_path"] = _append_vuln_hunt_event(analysis_repo_root, hunt_event)
            out.update(_update_vuln_candidate_feedback(analysis_repo_root, cast(dict[str, Any], out), hunt_event))

        # Collect source coverage report (llvm-cov) for improve feedback
        source_report: dict[str, Any] | None = None
        try:
            if gen is not None:
                current_bin = _resolve_current_coverage_binary(analysis_repo_root, cast(FuzzWorkflowRuntimeState, state))
                if current_bin is not None:
                    source_report = gen.collect_source_coverage(current_bin)
        except Exception:
            pass
        if source_report:
            out["coverage_source_report"] = source_report
            out["coverage_uncovered_functions"] = list(source_report.get("uncovered_functions") or [])
        run_feedback_artifact = _write_run_feedback_artifact(
            repo_root=analysis_repo_root,
            source_report=dict(source_report or {}),
            frontier_summary=frontier_summary,
        )
        out["coverage_run_feedback_path"] = str(run_feedback_artifact.get("path") or "")
        out["coverage_run_feedback_summary"] = dict(run_feedback_artifact.get("summary") or {})

        # Track exhausted targets for coverage-guided replan.
        # Each entry is either a plain target name (legacy) or a dict
        # {"name": ..., "round": ...}.  Entries older than
        # SHERPA_EXHAUSTED_TARGET_TTL rounds (default 5) are pruned so that
        # a temporarily-failed target can be retried later.
        _exhausted_ttl = int(os.environ.get("SHERPA_EXHAUSTED_TARGET_TTL", "5"))
        _raw_exhausted = list(state.get("coverage_exhausted_targets") or [])
        # Normalise legacy plain-string entries
        exhausted_entries: list[dict[str, Any]] = []
        for _e in _raw_exhausted:
            if isinstance(_e, dict):
                exhausted_entries.append(_e)
            elif isinstance(_e, str) and _e:
                exhausted_entries.append({"name": _e, "round": max(0, current_round - 1)})
        # Add current target if plateau detected or fatal run error
        _exhausted_names = {e["name"] for e in exhausted_entries}
        _run_err = str(state.get("coverage_run_error_kind_effective") or "")
        _exhaust_now = (
            (plateau_streak >= 2)
            or (_run_err == "dict_parse_error")  # dict errors mean target is fundamentally broken
        )
        if _exhaust_now and current_target_api and current_target_api not in _exhausted_names:
            exhausted_entries.append({"name": current_target_api, "round": current_round})
        # Prune expired entries
        exhausted_entries = [
            e for e in exhausted_entries
            if current_round - int(e.get("round") or 0) < _exhausted_ttl
        ]
        out["coverage_exhausted_targets"] = exhausted_entries
        # Convenience flat list for downstream consumers
        exhausted = [str(e.get("name") or e) for e in exhausted_entries]

        # Build coverage feedback for plan stage (replan context)
        if replan_required and history:
            feedback_lines = [f"Previously exhausted targets (avoid re-selecting):"]
            for t in exhausted:
                feedback_lines.append(f"  - {t}")
            for h_entry in history[-3:]:
                feedback_lines.append(
                    f"  Round {h_entry.get('round')}: target={h_entry.get('target_api')}, "
                    f"cov={h_entry.get('max_cov')}, ft={h_entry.get('max_ft')}, "
                    f"plateau={h_entry.get('plateau_detected')}"
                )
            feedback_lines.extend(_coverage_frontier_feedback_lines(frontier_summary))
            feedback_summary = dict(out.get("coverage_run_feedback_summary") or {})
            top_function_gaps = [
                dict(item)
                for item in list(feedback_summary.get("top_function_gaps") or [])
                if isinstance(item, dict)
            ]
            if top_function_gaps:
                feedback_lines.append("- top_function_gaps:")
                for item in top_function_gaps[:5]:
                    feedback_lines.append(
                        "  - "
                        f"{str(item.get('name') or '').strip()} "
                        f"({str(item.get('file') or '').strip()}:{int(item.get('line') or 0)}) "
                        f"kind={str(item.get('kind') or 'unknown')}"
                    )
            out["coverage_feedback_for_plan"] = "\n".join(feedback_lines)

        _wf_log(
            cast(dict[str, Any], out),
            f"<- coverage-analysis improve={int(should_improve)} {reason} dt={_fmt_dt(time.perf_counter()-t0)}",
        )
        _emit_fuzz_metrics(cast(dict[str, Any], out))
        return out
    except Exception as e:
        out = {**state, "last_step": "coverage-analysis", "last_error": str(e), "message": "coverage analysis failed"}
        _wf_log(cast(dict[str, Any], out), f"<- coverage-analysis err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
