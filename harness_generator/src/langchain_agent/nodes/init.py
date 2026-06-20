"""Carved from workflow_graph.py - '_node_init' LangGraph node."""

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
    FuzzWorkflowState,
    _fmt_dt,
    _normalize_error_state,
    _wf_log,
)
from workflow_helpers import (
    _alloc_output_workdir,
    _analysis_companion_enabled,
    _check_promefuzz_runtime_deps,
    _default_run_rss_limit_mb,
    _promefuzz_mcp_root_exists,
    _read_repro_context,
)
from workflow_vuln_scoring import (  # noqa: E402
    _VULN_REPLAN_PRIORITY_THRESHOLD,
    _vuln_hunting_enabled,
    _vuln_score_mode,
    _vuln_internal_api_min_score,
    _vuln_non_public_reachability_cap,
    _vuln_min_evidence_confidence,
    _vuln_topk,
    _vuln_max_iterations_per_candidate,
    _vuln_replan_priority_threshold,
    _vuln_score_weights,
    _security_signal_ids,
    _security_signal_patterns,
    _empty_security_scores,
    _compute_security_signal_scores,
    _derive_security_priority,
    _extract_security_scores,
    _top_security_signals,
    _signal_slug,
    _signal_vuln_category,
    _signal_sanitizer_hint,
    _attack_boundary_values,
    _candidate_attack_hint,
    _normalize_attack_hint,
    _candidate_priority,
)


def _node_init(state: FuzzWorkflowState) -> FuzzWorkflowRuntimeState:
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> init")
    repo_url = (state.get("repo_url") or "").strip()
    if not repo_url:
        raise ValueError("repo_url is required")

    ai_key_path = Path(state.get("ai_key_path") or "").expanduser().resolve()
    if not ai_key_path:
        raise ValueError("ai_key_path is required")

    time_budget = _wf_common.parse_budget_value(state.get("time_budget"), default=900)
    run_time_budget_raw = state.get("run_time_budget")
    if run_time_budget_raw is None:
        run_time_budget = time_budget
    else:
        run_time_budget = _wf_common.parse_budget_value(run_time_budget_raw, default=time_budget)
    if time_budget < 0:
        raise ValueError("time_budget must be >= 0")
    if run_time_budget < 0:
        raise ValueError("run_time_budget must be >= 0")
    max_len_raw = state.get("max_len")
    max_len = int(max_len_raw) if max_len_raw is not None else 0
    docker_image = (state.get("docker_image") or "").strip() or None
    codex_cli = (os.environ.get("SHERPA_CODEX_CLI") or os.environ.get("CODEX_CLI") or "opencode").strip()

    if _analysis_companion_enabled() and _promefuzz_mcp_root_exists():
        dep_ok, dep_err = _check_promefuzz_runtime_deps()
        if not dep_ok:
            raise RuntimeError(f"init prerequisite failed: {dep_err}")

    raw_resume_repo_root = (state.get("resume_repo_root") or "").strip()
    workdir: Path | None = None
    if raw_resume_repo_root:
        candidate = Path(raw_resume_repo_root).expanduser().resolve()
        if candidate.exists() and candidate.is_dir():
            workdir = candidate
    if workdir is None:
        workdir = _alloc_output_workdir(repo_url)
    generator = NonOssFuzzHarnessGenerator(
        repo_spec=RepoSpec(url=repo_url, workdir=workdir),
        ai_key_path=ai_key_path,
        max_len=max_len,
        time_budget_per_target=run_time_budget,
        rss_limit_mb=_default_run_rss_limit_mb(),
        docker_image=docker_image,
        codex_cli=codex_cli,
    )

    resume_step = (state.get("resume_from_step") or "").strip().lower()

    out = cast(
        FuzzWorkflowRuntimeState,
        {
            **state,
            "generator": generator,
            "crash_found": False,
            "message": "initialized",
            "plan_retry_reason": str(state.get("plan_retry_reason") or ""),
            "plan_targets_schema_valid_before_retry": bool(state.get("plan_targets_schema_valid_before_retry") or False),
            "plan_targets_schema_valid_after_retry": bool(state.get("plan_targets_schema_valid_after_retry") or False),
            "plan_used_fallback_targets": bool(state.get("plan_used_fallback_targets") or False),
            "step_count": int(state.get("step_count") or 0),
            "max_steps": int(state.get("max_steps")) if state.get("max_steps") is not None else 0,
            "last_step": "init",
            "last_error": "",
            "build_rc": 0,
            "build_stdout_tail": "",
            "build_stderr_tail": "",
            "build_full_log_path": "",
            "build_error_signature": "",
            "build_error_signature_before": "",
            "build_error_signature_after": "",
            "same_build_error_repeats": 0,
            "same_error_max_retries": max(
                0,
                int(
                    state.get("same_error_max_retries")
                    if state.get("same_error_max_retries") is not None
                    else 0
                ),
            ),
            "build_error_kind": "",
            "build_error_code": "",
            "build_error_signature_short": "",
            "build_attempts": int(state.get("build_attempts") or 0),
            "fix_build_attempts": int(state.get("fix_build_attempts") or 0),
            "max_fix_rounds": max(
                0,
                int(state.get("max_fix_rounds") if state.get("max_fix_rounds") is not None else 0),
            ),
            "fix_build_noop_streak": int(state.get("fix_build_noop_streak") or 0),
            "fix_build_attempt_history": list(state.get("fix_build_attempt_history") or []),
            "fix_build_rule_hits": list(state.get("fix_build_rule_hits") or []),
            "fix_build_terminal_reason": str(state.get("fix_build_terminal_reason") or ""),
            "fix_build_last_diff_paths": list(state.get("fix_build_last_diff_paths") or []),
            "fix_action_type": "",
            "fix_effect": "",
            "codex_hint": "",
            "failed": False,
            "repo_root": str(generator.repo_root),
            "run_rc": 0,
            "crash_evidence": "none",
            "run_error_kind": "",
            "run_terminal_reason": "",
            "run_idle_seconds": 0,
            "run_children_exit_count": 0,
            "last_crash_artifact": str(state.get("last_crash_artifact") or ""),
            "last_fuzzer": str(state.get("last_fuzzer") or ""),
            "crash_signature": "",
            "same_crash_repeats": 0,
            "crash_fix_attempts": int(state.get("crash_fix_attempts") or 0),
            "crash_repro_done": bool(state.get("crash_repro_done") or False),
            "crash_repro_ok": bool(state.get("crash_repro_ok") or False),
            "crash_repro_rc": int(state.get("crash_repro_rc") or 0),
            "crash_repro_report_path": str(state.get("crash_repro_report_path") or ""),
            "crash_repro_json_path": str(state.get("crash_repro_json_path") or ""),
            "crash_triage_done": bool(state.get("crash_triage_done") or False),
            "crash_triage_label": str(state.get("crash_triage_label") or ""),
            "crash_triage_confidence": float(state.get("crash_triage_confidence") or 0.0),
            "crash_triage_reason": str(state.get("crash_triage_reason") or ""),
            "crash_triage_signal_lines": list(state.get("crash_triage_signal_lines") or []),
            "crash_triage_report_path": str(state.get("crash_triage_report_path") or ""),
            "crash_triage_json_path": str(state.get("crash_triage_json_path") or ""),
            "re_build_done": bool(state.get("re_build_done") or False),
            "re_build_ok": bool(state.get("re_build_ok") or False),
            "re_build_rc": int(state.get("re_build_rc") or 0),
            "re_build_report_path": str(state.get("re_build_report_path") or ""),
            "re_build_json_path": str(state.get("re_build_json_path") or ""),
            "re_run_done": bool(state.get("re_run_done") or False),
            "re_run_ok": bool(state.get("re_run_ok") or False),
            "re_run_rc": int(state.get("re_run_rc") or 0),
            "re_run_report_path": str(state.get("re_run_report_path") or ""),
            "re_run_json_path": str(state.get("re_run_json_path") or ""),
            "re_workspace_root": str(state.get("re_workspace_root") or ""),
            "restart_to_plan": bool(state.get("restart_to_plan") or False),
            "restart_to_plan_reason": str(state.get("restart_to_plan_reason") or ""),
            "restart_to_plan_stage": str(state.get("restart_to_plan_stage") or ""),
            "restart_to_plan_error_text": str(state.get("restart_to_plan_error_text") or ""),
            "restart_to_plan_report_path": str(state.get("restart_to_plan_report_path") or ""),
            "restart_to_plan_count": int(state.get("restart_to_plan_count") or 0),
            "fix_harness_attempts": int(state.get("fix_harness_attempts") or 0),
            "plan_fix_on_crash": True,
            "plan_max_fix_rounds": 0,
            "repair_mode": bool(state.get("repair_mode") or False),
            "repair_origin_stage": str(state.get("repair_origin_stage") or ""),
            "repair_error_kind": str(state.get("repair_error_kind") or ""),
            "repair_error_code": str(state.get("repair_error_code") or ""),
            "repair_signature": str(state.get("repair_signature") or ""),
            "repair_stdout_tail": str(state.get("repair_stdout_tail") or ""),
            "repair_stderr_tail": str(state.get("repair_stderr_tail") or ""),
            "repair_recent_attempts": list(state.get("repair_recent_attempts") or []),
            "coverage_loop_max_rounds": max(
                0,
                int(
                    state.get("coverage_loop_max_rounds")
                    if state.get("coverage_loop_max_rounds") is not None
                    else 0
                ),
            ),
            "coverage_loop_round": int(state.get("coverage_loop_round") or 0),
            "coverage_should_improve": bool(state.get("coverage_should_improve") or False),
            "coverage_improve_reason": str(state.get("coverage_improve_reason") or ""),
            "coverage_bottleneck_kind": str(state.get("coverage_bottleneck_kind") or ""),
            "coverage_bottleneck_reason": str(state.get("coverage_bottleneck_reason") or ""),
            "coverage_history": list(state.get("coverage_history") or []),
            "coverage_target_name": str(state.get("coverage_target_name") or ""),
            "coverage_seed_profile": str(state.get("coverage_seed_profile") or ""),
            "coverage_plateau_streak": int(state.get("coverage_plateau_streak") or 0),
            "coverage_last_max_cov": int(state.get("coverage_last_max_cov") or 0),
            "coverage_last_ft": int(state.get("coverage_last_ft") or 0),
            "coverage_replan_required": bool(state.get("coverage_replan_required") or False),
            "coverage_improve_mode": str(state.get("coverage_improve_mode") or ""),
            "coverage_round_budget_exhausted": bool(state.get("coverage_round_budget_exhausted") or False),
            "coverage_stop_reason": str(state.get("coverage_stop_reason") or ""),
            "coverage_corpus_sources": list(state.get("coverage_corpus_sources") or []),
            "coverage_seed_counts": dict(state.get("coverage_seed_counts") or {}),
            "coverage_target_score_breakdown": dict(state.get("coverage_target_score_breakdown") or {}),
            "coverage_per_input_manifest_path": str(state.get("coverage_per_input_manifest_path") or ""),
            "coverage_frontier_path": str(state.get("coverage_frontier_path") or ""),
            "coverage_frontier_summary": dict(state.get("coverage_frontier_summary") or {}),
            "coverage_replay_runtime_sec": float(state.get("coverage_replay_runtime_sec") or 0.0),
            "coverage_replay_binary_hash": str(state.get("coverage_replay_binary_hash") or ""),
            "coverage_replay_binary_dir": str(state.get("coverage_replay_binary_dir") or ""),
            "coverage_replay_binary_count": int(state.get("coverage_replay_binary_count") or 0),
            "coverage_replay_stage_success": bool(state.get("coverage_replay_stage_success") or False),
            "coverage_replay_error": str(state.get("coverage_replay_error") or ""),
            "coverage_replay_manifest_fresh_for_current_binary": bool(
                state.get("coverage_replay_manifest_fresh_for_current_binary") or False
            ),
            "coverage_replay_queue_drained": bool(state.get("coverage_replay_queue_drained") or False),
            "coverage_replay_pending_inputs": int(state.get("coverage_replay_pending_inputs") or 0),
            "coverage_replay_failed_inputs": int(state.get("coverage_replay_failed_inputs") or 0),
            "coverage_replay_processed_inputs": int(state.get("coverage_replay_processed_inputs") or 0),
            "coverage_replay_total_inputs": int(state.get("coverage_replay_total_inputs") or 0),
            "analysis_done": bool(state.get("analysis_done") or False),
            "analysis_degraded": bool(state.get("analysis_degraded") or False),
            "analysis_error": str(state.get("analysis_error") or ""),
            "analysis_report_path": str(state.get("analysis_report_path") or ""),
            "analysis_context_path": str(state.get("analysis_context_path") or ""),
            "analysis_evidence_count": int(state.get("analysis_evidence_count") or 0),
            "security_evidence_count": int(state.get("security_evidence_count") or 0),
            "vuln_candidate_count": int(state.get("vuln_candidate_count") or 0),
            "vuln_hunting_enabled": bool(state.get("vuln_hunting_enabled") or _vuln_hunting_enabled()),
            "vuln_focus_profile": str(state.get("vuln_focus_profile") or "broad_high_risk"),
            "target_surface_policy": str(state.get("target_surface_policy") or "risk_first"),
            "security_priority_mode": bool(
                state.get("security_priority_mode")
                if state.get("security_priority_mode") is not None
                else (_vuln_hunting_enabled() and _vuln_score_mode() == "risk_first_v1")
            ),
            "latest_vuln_decision_snapshot": dict(state.get("latest_vuln_decision_snapshot") or {}),
            "antlr_context_path": str(state.get("antlr_context_path") or ""),
            "antlr_context_summary": str(state.get("antlr_context_summary") or ""),
            "target_analysis_path": str(state.get("target_analysis_path") or ""),
            "target_analysis_summary": str(state.get("target_analysis_summary") or ""),
            "target_scoring_enabled": bool(state.get("target_scoring_enabled") or False),
            "target_score_breakdown_available": bool(state.get("target_score_breakdown_available") or False),
            "constraint_memory_count": int(state.get("constraint_memory_count") or 0),
            "constraint_memory_path": str(state.get("constraint_memory_path") or ""),
            "decision_traces": list(state.get("decision_traces") or []),
            "decision_trace_count": int(state.get("decision_trace_count") or 0),
            "latest_decision_snapshot": dict(state.get("latest_decision_snapshot") or {}),
            "crash_signature_dedup_hit": bool(state.get("crash_signature_dedup_hit") or False),
        },
    )

    # Restore crash context from previous run stage when crash recovery is resumed
    # as a separate k8s stage job. Without this, init resets crash state and
    # re-build/re-run would be incorrectly skipped.
    if resume_step in {
        "analysis",
        "plan",
        "synthesize",
        "build",
        "run",
        "per-input-replay",
        "crash-triage",
        "fix-harness",
        "coverage-analysis",
        "improve-harness",
        "re-build",
        "re-run",
    }:
        try:
            repro_doc = _read_repro_context(generator.repo_root)
            if isinstance(repro_doc, dict):
                if not str(out.get("last_fuzzer") or "").strip():
                    out["last_fuzzer"] = str(repro_doc.get("last_fuzzer") or "")
                if not str(out.get("last_crash_artifact") or "").strip():
                    out["last_crash_artifact"] = str(repro_doc.get("last_crash_artifact") or "")
                if not str(out.get("re_workspace_root") or "").strip():
                    out["re_workspace_root"] = str(repro_doc.get("re_workspace_root") or "")
            summary_json = generator.repo_root / "run_summary.json"
            if summary_json.is_file():
                doc = json.loads(summary_json.read_text(encoding="utf-8", errors="replace"))
                if isinstance(doc, dict):
                    out["crash_found"] = bool(doc.get("crash_found") or False)
                    out["run_error_kind"] = str(doc.get("run_error_kind") or "")
                    out["run_details"] = list(doc.get("run_details") or [])
                    if not str(out.get("last_fuzzer") or "").strip():
                        out["last_fuzzer"] = str(doc.get("last_fuzzer") or "")
                    if not str(out.get("last_crash_artifact") or "").strip():
                        out["last_crash_artifact"] = str(doc.get("last_crash_artifact") or "")
                    out["crash_evidence"] = str(doc.get("crash_evidence") or "none")
                    out["run_rc"] = int(doc.get("run_rc") or 0)
                    coverage_loop = doc.get("coverage_loop")
                    if isinstance(coverage_loop, dict):
                        out["coverage_loop_max_rounds"] = max(
                            0,
                            int(
                                coverage_loop.get("max_rounds")
                                if coverage_loop.get("max_rounds") is not None
                                else (
                                    out.get("coverage_loop_max_rounds")
                                    if out.get("coverage_loop_max_rounds") is not None
                                    else 0
                                )
                            ),
                        )
                        out["coverage_loop_round"] = int(coverage_loop.get("round") or out.get("coverage_loop_round") or 0)
                        out["coverage_should_improve"] = bool(
                            coverage_loop.get("should_improve") or out.get("coverage_should_improve") or False
                        )
                        out["coverage_improve_reason"] = str(
                            coverage_loop.get("reason") or out.get("coverage_improve_reason") or ""
                        )
                        out["coverage_history"] = list(
                            coverage_loop.get("history") or out.get("coverage_history") or []
                        )
                        out["coverage_target_name"] = str(coverage_loop.get("target_name") or out.get("coverage_target_name") or "")
                        out["coverage_seed_profile"] = str(coverage_loop.get("seed_profile") or out.get("coverage_seed_profile") or "")
                        out["coverage_target_depth_score"] = int(
                            coverage_loop.get("target_depth_score") or out.get("coverage_target_depth_score") or 0
                        )
                        out["coverage_target_depth_class"] = str(
                            coverage_loop.get("target_depth_class") or out.get("coverage_target_depth_class") or ""
                        )
                        out["coverage_selection_bias_reason"] = str(
                            coverage_loop.get("selection_bias_reason") or out.get("coverage_selection_bias_reason") or ""
                        )
                        out["coverage_plateau_streak"] = int(coverage_loop.get("plateau_streak") or out.get("coverage_plateau_streak") or 0)
                        out["coverage_last_max_cov"] = int(coverage_loop.get("last_max_cov") or out.get("coverage_last_max_cov") or 0)
                        out["coverage_last_ft"] = int(coverage_loop.get("last_ft") or out.get("coverage_last_ft") or 0)
                        out["coverage_replan_required"] = bool(coverage_loop.get("replan_required") or out.get("coverage_replan_required") or False)
                        out["coverage_replan_effective"] = bool(
                            coverage_loop.get("replan_effective") if "replan_effective" in coverage_loop else out.get("coverage_replan_effective") or False
                        )
                        out["coverage_replan_reason"] = str(
                            coverage_loop.get("replan_reason") or out.get("coverage_replan_reason") or ""
                        )
                        out["coverage_improve_mode"] = str(coverage_loop.get("improve_mode") or out.get("coverage_improve_mode") or "")
                        out["coverage_round_budget_exhausted"] = bool(
                            coverage_loop.get("round_budget_exhausted") or out.get("coverage_round_budget_exhausted") or False
                        )
                        out["coverage_stop_reason"] = str(
                            coverage_loop.get("stop_reason") or out.get("coverage_stop_reason") or ""
                        )
                        out["coverage_corpus_sources"] = list(coverage_loop.get("corpus_sources") or out.get("coverage_corpus_sources") or [])
                        out["coverage_seed_counts"] = dict(coverage_loop.get("seed_counts") or out.get("coverage_seed_counts") or {})
                        out["coverage_repo_examples_filtered"] = bool(
                            coverage_loop.get("repo_examples_filtered")
                            if "repo_examples_filtered" in coverage_loop
                            else out.get("coverage_repo_examples_filtered") or False
                        )
                        out["coverage_repo_examples_rejected_count"] = int(
                            coverage_loop.get("repo_examples_rejected_count")
                            or out.get("coverage_repo_examples_rejected_count")
                            or 0
                        )
                        out["coverage_repo_examples_accepted_count"] = int(
                            coverage_loop.get("repo_examples_accepted_count")
                            or out.get("coverage_repo_examples_accepted_count")
                            or 0
                        )
                    plan_policy = doc.get("plan_policy")
                    if isinstance(plan_policy, dict):
                        out["plan_fix_on_crash"] = bool(plan_policy.get("fix_on_crash", out["plan_fix_on_crash"]))
                    out["plan_max_fix_rounds"] = 0
                    build_fix_policy = doc.get("build_fix_policy")
                    if isinstance(build_fix_policy, dict):
                        _ = build_fix_policy
                    out["max_fix_rounds"] = 0
                    out["same_error_max_retries"] = 0
                    re_stage = doc.get("re_stage")
                    if isinstance(re_stage, dict):
                        if not str(out.get("re_workspace_root") or "").strip():
                            out["re_workspace_root"] = str(re_stage.get("workspace_root") or "")
                        out["re_build_done"] = bool(re_stage.get("re_build_done") or False)
                        out["re_build_ok"] = bool(re_stage.get("re_build_ok") or False)
                        out["re_build_rc"] = int(re_stage.get("re_build_rc") or 0)
                        out["re_build_report_path"] = str(re_stage.get("re_build_report_path") or "")
                        out["re_build_json_path"] = str(re_stage.get("re_build_json_path") or "")
                        out["re_run_done"] = bool(re_stage.get("re_run_done") or False)
                        out["re_run_ok"] = bool(re_stage.get("re_run_ok") or False)
                        out["re_run_rc"] = int(re_stage.get("re_run_rc") or 0)
                        out["re_run_report_path"] = str(re_stage.get("re_run_report_path") or "")
                        out["re_run_json_path"] = str(re_stage.get("re_run_json_path") or "")
                    restart_ctx = doc.get("restart_to_plan")
                    if isinstance(restart_ctx, dict):
                        out["restart_to_plan"] = bool(restart_ctx.get("active") or False)
                        out["restart_to_plan_reason"] = str(restart_ctx.get("reason") or "")
                        out["restart_to_plan_stage"] = str(restart_ctx.get("stage") or "")
                        out["restart_to_plan_error_text"] = str(restart_ctx.get("error_text") or "")
                        out["restart_to_plan_report_path"] = str(restart_ctx.get("report_path") or "")
                        out["restart_to_plan_count"] = int(restart_ctx.get("count") or 0)
        except Exception:
            pass

    out = cast(FuzzWorkflowRuntimeState, _normalize_error_state(cast(dict[str, Any], out)))
    _wf_log(cast(dict[str, Any], out), f"<- init ok repo_root={out.get('repo_root')} dt={_fmt_dt(time.perf_counter()-t0)}")
    return out
