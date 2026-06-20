""""""

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

from workflow_public_api import (  # noqa: E402
    _NON_PUBLIC_API_NAME_PATTERNS,
    _PUBLIC_API_SYMBOL_CACHE,
    _CALLGRAPH_REVERSE_CACHE,
    _is_internal_api_symbol,
    _vuln_public_api_enforce,
    _companion_work_dir,
    _load_public_api_symbols,
    _load_api_loc_index,
    _api_surface_class,
    _is_non_public_api,
    _is_unlinkable_binding_api,
    _load_callgraph_reverse,
    _load_callgraph_forward,
    _nearest_public_entry,
    coverage_potential,
)

from wf_state import (  # re-export: keep workflow_graph._<name> valid
    _RECOVERABLE_RUN_ERROR_KINDS,
    _FATAL_RUN_ERROR_KINDS,
    FuzzWorkflowState,
    FuzzWorkflowRuntimeState,
    _has_error_payload,
    _coerce_error_payload,
    _derive_error_from_legacy,
    _project_error_legacy_fields,
    _normalize_error_state,
    _wf_log,
    _fmt_dt,
    _sha256_text,
    _bounded_float,
    _opencode_defunct_threshold,
    _count_opencode_defunct_processes,
    _enter_step,
)
from workflow_helpers import (  # re-export: keep workflow_graph._<name> valid
    _effective_run_error_kind,
    _clear_error_markers_on_success,
    _record_decision_trace,
    _emit_fuzz_metrics,
    _grace_wait_for_file,
    _calc_parallel_batch_budget,
    _llm_or_none,
    _repro_context_path,
    _read_repro_context,
    _write_repro_context,
    _extract_json_object,
    _validate_targets_json,
    _infer_target_type,
    _opencode_done_path,
    _opencode_feedback_dir,
    _feedback_group_for_stage,
    _feedback_file_for_stage,
    _feedback_text_limits,
    _trim_feedback_text,
    _build_fix_harness_crash_context,
    _write_stage_feedback,
    _collect_feedback_for_group,
    _try_hotfix_missing_decl,
    _install_coverage_cc_wrapper,
    _apply_coverage_cc_wrapper_env,
    _inject_coverage_instrumentation,
    _clear_opencode_done_sentinel,
    _infer_repair_origin_stage,
    _repair_mode_active,
    _constraint_memory_path,
    _constraint_repeat_threshold,
    _load_constraint_memory,
    _write_constraint_memory,
    _constraint_fix_hint,
    _record_constraint_memory_observation,
    _constraint_memory_snapshot_from_state,
    _procedural_memory_library_class,
    _procedural_memory_system_packages_nonempty,
    _record_procedural_memory,
    _build_repair_snapshot,
    _infer_target_lang_from_repo,
    _infer_seed_profile,
    _normalize_seed_profile,
    _score_target_depth,
    _runtime_viability_details,
    _is_test_or_demo_helper_target,
    _load_targets_doc,
    _enrich_targets_depth,
    _select_primary_target,
    _selected_targets_path,
    _execution_plan_path,
    _harness_index_path,
    _observed_target_path,
    _execution_targets_max,
    _execution_targets_min_required,
    _runtime_viability_rank,
    _target_scoring_weights,
    _clamp_score,
    _target_component_coverage_gap,
    _target_component_complexity,
    _target_component_api_relevance,
    _target_component_consumer_order_support,
    _target_score_breakdown,
    _load_seed_feedback_by_fuzzer,
    _load_target_runtime_cooldown_index,
    _target_runtime_penalty,
    _selection_target_key,
    _ENTRYPOINT_HINTS_SUFFIX,
    _ENTRYPOINT_HINTS_EXACT,
    _ENTRYPOINT_LEAF_HINTS,
    _vuln_entrypoint_bias_weight,
    _is_library_entrypoint,
    _NON_HARNESSABLE_EXACT,
    _NON_HARNESSABLE_TOKENS,
    _is_non_harnessable_target,
    _library_entrypoint_bias,
    _selection_mode,
    _coverage_potential_enabled,
    _coverage_potential_weight,
    _entrypoint_risk_bias,
    _execution_depth_bias,
    _target_surface_penalty,
    _apply_selected_target_filters,
    _target_analysis_lookup_keys,
    _targets_material_signature,
    _load_target_analysis_security_index,
    _load_security_evidence_list,
    _load_vuln_candidate_inventory,
    _vuln_candidates_path,
    _load_vuln_candidates_doc,
    _active_vuln_candidates,
    _write_vuln_candidates_doc,
    _vuln_candidate_matches_feedback,
    _feedback_status_for_vuln_candidate,
    _update_vuln_candidate_feedback,
    _vuln_candidate_id,
    _normalize_analysis_vuln_candidate,
    _write_analysis_vuln_candidates,
    _vuln_hunt_summary_path,
    _vuln_hunt_events_path,
    _vuln_hunt_event_from_state,
    _append_vuln_hunt_event,
    _write_vuln_hunt_summary,
    _run_vuln_hunt_subphase,
    _lookup_target_security_candidate,
    _build_selected_target_row,
    _build_selected_targets_doc,
    _write_selected_targets_doc,
    _load_selected_targets_doc,
    _build_execution_plan_doc,
    _write_execution_plan_doc,
    _sync_execution_plan_doc_from_selected_targets,
    _selected_target_row_for_execution_target,
    _workflow_target_state_from_execution_plan,
    _load_execution_plan_doc,
    _execution_plan_targets,
    _execution_target_sort_key,
    _primary_execution_target,
    _execution_target_fuzzer_name,
    _execution_target_fuzzer_aliases,
    _execution_target_identity,
    _execution_target_matches_token,
    _find_execution_target_for_tokens,
    _preferred_execution_target,
    _target_type_from_run_details,
    _matching_run_details_for_target,
    _order_fuzzer_bins_by_execution_plan,
    _filter_fuzzer_bins_by_execution_plan,
    _discover_harness_sources,
    _normalize_exec_target_token,
    _token_overlap_ratio,
    _build_harness_index_doc,
    _write_harness_index_doc,
    _load_harness_index_doc,
    _write_observed_target_doc,
    _load_observed_target_doc,
    _infer_harness_primary_api,
    _readme_drift_status,
    _analyze_harness_target_alignment,
    _build_fallback_targets_doc,
    _write_fallback_targets_json,
    _summarize_build_error,
    _splice_sources_list,
    _extract_actionable_build_locations,
    _build_file_targeted_fix_lines,
    _repair_strategy_repeat_threshold,
    _extract_repair_symbols,
    _extract_repair_top_trace,
    _build_repair_error_digest,
    _validate_execution_plan_harness_consistency,
    _validate_build_repair_contract,
    _validate_harness_source_contract,
    _classify_build_failure,
    _build_failure_recovery_advice,
    _collect_key_artifact_hashes,
    _has_codex_key,
    _build_seed_feedback,
    _coverage_attack_hint_feedback_lines,
    _coverage_frontier_feedback_lines,
    _build_run_feedback_summary,
    _write_run_feedback_artifact,
    _resolve_current_coverage_binary,
    _replay_out_dir,
    _binary_looks_profile_instrumented,
    _materialize_replay_binaries,
    _resolve_per_input_replay_binary,
    _aggregate_seed_quality_from_run_details,
    _seed_quality_from_run_details_for_target,
    _quality_flags_from_seed_quality,
    _build_harness_feedback,
    _slug_from_repo_url,
    _alloc_output_workdir,
    _remaining_time_budget_sec,
    _opencode_cli_retries,
    _analysis_opencode_advisory_enabled,
    _analysis_opencode_timeout_sec,
    _analysis_opencode_idle_timeout_sec,
    _vuln_hunt_idle_timeout_sec,
    _plan_idle_timeout_sec,
    _fix_build_max_noop_streak,
    _fix_build_max_attempts,
    _effective_max_fix_rounds,
    _effective_same_error_retry_limit,
    _fix_build_feedback_history_limit,
    _fix_build_context_max_chars,
    _fix_build_stdout_max_chars,
    _fix_build_stderr_max_chars,
    _fix_build_keep_recent_errors,
    _fix_build_context_history_limit,
    _fix_build_ruleset,
    _run_idle_timeout_sec,
    _synthesize_opencode_idle_timeout_sec,
    _synthesize_opencode_attempts,
    _fix_build_same_signature_plan_threshold,
    _contains_cjk_text,
    _synthesize_activity_watch_paths,
    _build_scaffold_path,
    _build_template_cache_path,
    _find_static_lib,
    _load_build_template_cache_doc,
    _write_build_template_cache_doc,
    _cache_successful_build_template,
    _restore_cached_build_template_if_missing,
    _build_runtime_facts_path,
    _repo_understanding_path,
    _load_repo_understanding_doc,
    _repo_understanding_is_complete,
    _load_build_strategy_doc,
    _load_build_runtime_facts_doc,
    _contains_forbidden_repo_fuzz_target_usage,
    _extract_repo_fuzz_target_usages,
    _allowed_repo_fuzz_targets,
    _infer_fuzzer_entry_strategy,
    _write_build_strategy_doc,
    _build_scaffold_precheck,
    _run_finalize_timeout_sec,
    _run_unlimited_round_budget_sec,
    _verify_stage_no_ai,
    _max_same_timeout_repeats,
    _run_stop_on_first_crash,
    _run_parallel_early_stop_enabled,
    _run_cpu_budget,
    _run_outer_parallelism_max,
    _run_inner_workers_min,
    _run_inner_workers_target,
    _run_parallel_engine,
    _run_ignore_non_fatal_enabled,
    _auto_stop_policy,
    _coverage_underutilized_execs_threshold,
    _cold_start_seed_replan_quality_threshold,
    _cold_start_seed_replan_early_units_30s_threshold,
    _solve_parallelism,
    _time_budget_exceeded_state,
    _make_plan_hint,
    _derive_plan_policy,
    _load_opencode_prompt_templates,
    _render_opencode_prompt,
    _procedural_stage_for_prompt,
    _inject_procedural_lessons,
    _render_opencode_prompt_safe,
    _attach_prompt_render_status,
    _default_run_rss_limit_mb,
    _antlr_assist_enabled,
    _antlr_assist_max_files,
    _collect_antlr_assist_context,
    _prepare_antlr_assist_context,
    _collect_target_analysis_context,
    _prepare_target_analysis_context,
    _collect_analysis_companion_context,
    _read_json_doc,
    _materialize_analysis_context_from_companion,
    _build_analysis_evidence_index,
    _analysis_companion_enabled,
    _promefuzz_mcp_root_exists,
    _check_promefuzz_runtime_deps,
    _max_cov_from_run_details,
    _normalize_crash_triage_label,
    _normalize_crash_analysis_verdict,
    _crash_vuln_status,
    _crash_vuln_confidence,
    _crash_vuln_sanitizer_signal,
    _crash_vuln_selected_target,
    _write_crash_vuln_candidate,
    _re_restart_limit,
    _detect_harness_error,
    _bytes_human,
    _tree_file_stats,
    _collect_fuzz_inventory,
    _write_run_summary,
)
from nodes.init import _node_init
from nodes.analysis import _node_analysis
from nodes.vuln_hunt import _node_vuln_hunt
from nodes.plan import _node_plan
from nodes.synthesize import _node_synthesize
from nodes.build import _node_build
from nodes.run import _node_run
from nodes.per_input_replay import _node_per_input_replay
from nodes.coverage_analysis import _node_coverage_analysis
from nodes.improve_harness import _node_improve_harness
from nodes.re_build import _node_re_build
from nodes.re_run import _node_re_run
from nodes.crash_triage import _node_crash_triage
from nodes.crash_analysis import _node_crash_analysis
from nodes.fix_build import _node_fix_build
from nodes.fix_crash import _node_fix_crash
from nodes.fix_harness_after_run import _node_fix_harness_after_run


@dataclass(frozen=True)
class FuzzWorkflowInput:
    repo_url: str
    email: Optional[str]
    time_budget: int
    run_time_budget: int
    max_len: int
    docker_image: Optional[str]
    ai_key_path: Path
    model: Optional[str] = None
    context_dir: Optional[str] = None
    resume_from_step: Optional[str] = None
    resume_repo_root: Optional[Path] = None
    stop_after_step: Optional[str] = None
    coverage_loop_max_rounds: int = 0
    max_fix_rounds: int = 0
    same_error_max_retries: int = 0


def _route_after_build_state(state: FuzzWorkflowRuntimeState) -> str:
    state = cast(FuzzWorkflowRuntimeState, _normalize_error_state(cast(dict[str, Any], state)))
    if bool(state.get("restart_to_plan")):
        return "plan"
    err = dict(state.get("error") or {})
    if not _has_error_payload(err):
        return "run"
    return "plan"


def _route_after_run_state(state: FuzzWorkflowRuntimeState) -> str:
    state = cast(FuzzWorkflowRuntimeState, _normalize_error_state(cast(dict[str, Any], state)))
    if bool(state.get("restart_to_plan")):
        return "plan"
    if bool(state.get("crash_found")):
        return "crash-triage"
    err = dict(state.get("error") or {})
    terminal_reason = str(state.get("run_terminal_reason") or err.get("code") or "").strip().lower()
    # Coverage plateau is a coverage signal, not a hard run failure.
    # Replay the corpus first so coverage-analysis can classify it.
    if terminal_reason == "coverage_plateau":
        return "per-input-replay"
    run_error_kind = _effective_run_error_kind(cast(dict[str, Any], state)) or str(
        state.get("run_error_kind") or err.get("code") or ""
    ).strip().lower()
    if run_error_kind in _RECOVERABLE_RUN_ERROR_KINDS:
        return "per-input-replay"
    if run_error_kind in _FATAL_RUN_ERROR_KINDS:
        return "plan"
    if run_error_kind:
        return "plan"
    return "per-input-replay"


def _route_after_per_input_replay_state(state: FuzzWorkflowRuntimeState) -> str:
    state = cast(FuzzWorkflowRuntimeState, _normalize_error_state(cast(dict[str, Any], state)))
    err = dict(state.get("error") or {})
    if bool(state.get("failed")) or bool(err.get("terminal")):
        return "stop"
    if str(state.get("last_error") or err.get("message") or "").strip():
        return "stop"
    return "coverage-analysis"


def _route_after_coverage_analysis_state(state: FuzzWorkflowRuntimeState) -> str:
    # Set source on original dict before _normalize_error_state makes a copy.
    if bool(state.get("coverage_should_improve")) and _vuln_hunting_enabled():
        state["_vuln_hunt_entry_source"] = "coverage-analysis"
    state = cast(FuzzWorkflowRuntimeState, _normalize_error_state(cast(dict[str, Any], state)))
    err = dict(state.get("error") or {})
    if bool(state.get("failed")) or bool(err.get("terminal")):
        return "stop"
    if str(state.get("last_error") or err.get("message") or "").strip():
        return "stop"
    if bool(state.get("coverage_should_improve")):
        if _vuln_hunting_enabled():
            return "vuln-hunt"
        return "improve-harness"
    # Circuit breaker: force a full replan after repeated no-improvement
    # loops instead of blindly re-running the same failing configuration.
    max_continuous = int(os.environ.get("SHERPA_MAX_CONTINUOUS_LOOP", "3"))
    loop_count = int(state.get("continuous_loop_count") or 0)
    if loop_count >= max_continuous:
        if _vuln_hunting_enabled():
            return "vuln-hunt"
        return "plan"
    if _auto_stop_policy() == "hard_fail_only":
        return "run"
    return "stop"


def _route_after_improve_harness_state(state: FuzzWorkflowRuntimeState) -> str:
    # Set source on original dict before _normalize_error_state makes a copy.
    if (
        bool(state.get("coverage_should_improve"))
        and _vuln_hunting_enabled()
        and str(state.get("coverage_improve_mode") or "").strip() == "in_place"
    ):
        state["_vuln_hunt_entry_source"] = "improve-harness"
    state = cast(FuzzWorkflowRuntimeState, _normalize_error_state(cast(dict[str, Any], state)))
    err = dict(state.get("error") or {})
    if bool(state.get("failed")) or bool(err.get("terminal")):
        return "stop"
    if str(state.get("last_error") or err.get("message") or "").strip():
        return "stop"
    # Circuit breaker: force replan after repeated no-improvement loops.
    max_continuous = int(os.environ.get("SHERPA_MAX_CONTINUOUS_LOOP", "3"))
    loop_count = int(state.get("continuous_loop_count") or 0)
    if loop_count >= max_continuous:
        return "plan"
    if str(state.get("coverage_improve_mode") or "").strip() == "replan" and not bool(
        state.get("coverage_replan_effective", True)
    ):
        if _auto_stop_policy() == "hard_fail_only":
            return "plan"
        return "stop"
    if bool(state.get("coverage_round_budget_exhausted")):
        if _auto_stop_policy() == "hard_fail_only":
            return "plan"
        return "stop"
    if bool(state.get("coverage_should_improve")):
        mode = str(state.get("coverage_improve_mode") or "").strip()
        if _vuln_hunting_enabled() and mode == "in_place":
            return "vuln-hunt"
        if mode == "in_place":
            return "build"
        return "plan"
    return "stop"


def _route_after_analysis_state(state: FuzzWorkflowRuntimeState) -> str:
    # Set source on original dict before _normalize_error_state makes a copy.
    if _vuln_hunting_enabled():
        state["_vuln_hunt_entry_source"] = "analysis"
    state = cast(FuzzWorkflowRuntimeState, _normalize_error_state(cast(dict[str, Any], state)))
    err = dict(state.get("error") or {})
    if bool(state.get("failed")) or bool(err.get("terminal")):
        return "stop"
    if str(state.get("last_error") or err.get("message") or "").strip() and not bool(state.get("analysis_degraded")):
        return "stop"
    if _vuln_hunting_enabled():
        return "vuln-hunt"
    return "plan"


def _route_after_plan_state(state: FuzzWorkflowRuntimeState) -> str:
    state = cast(FuzzWorkflowRuntimeState, _normalize_error_state(cast(dict[str, Any], state)))
    err = dict(state.get("error") or {})
    if bool(state.get("failed")) or bool(err.get("terminal")) or str(state.get("last_error") or err.get("message") or "").strip():
        return "stop"
    return "synthesize"


def _route_after_synthesize_state(state: FuzzWorkflowRuntimeState) -> str:
    state = cast(FuzzWorkflowRuntimeState, _normalize_error_state(cast(dict[str, Any], state)))
    err = dict(state.get("error") or {})
    if bool(state.get("failed")) or bool(err.get("terminal")) or str(state.get("last_error") or err.get("message") or "").strip():
        return "stop"
    return "build"


def _route_after_fix_build_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("restart_to_plan")):
        return "plan"
    if int(state.get("same_build_error_repeats") or 0) >= _fix_build_same_signature_plan_threshold():
        return "plan"
    terminal_reason = (state.get("fix_build_terminal_reason") or "").strip()
    if terminal_reason == "requires_env_rebuild":
        return "build"
    if terminal_reason:
        return "fix_build"
    if (state.get("last_error") or "").strip():
        return "fix_build"
    return "build"


def _route_after_fix_crash_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")):
        return "stop"
    if (state.get("last_error") or "").strip():
        return "stop"
    return "build"


def _route_after_crash_triage_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")):
        return "stop"
    if bool(state.get("restart_to_plan")):
        return "plan"
    label = _normalize_crash_triage_label(str(state.get("crash_triage_label") or ""))
    if label == "harness_bug":
        return "plan"
    if label == "upstream_bug":
        return "re-build"
    return "plan"


def _route_after_fix_harness_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")):
        return "stop"
    if bool(state.get("restart_to_plan")):
        return "plan"
    if (state.get("last_error") or "").strip():
        return "plan"
    return "build"


def _route_after_re_build_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")):
        return "stop"
    if not bool(state.get("crash_found")):
        return "stop"
    if bool(state.get("restart_to_plan")):
        if int(state.get("restart_to_plan_count") or 0) > _re_restart_limit():
            return "stop"
        return "plan"
    if bool(state.get("re_build_done")) and bool(state.get("re_build_ok")):
        return "re-run"
    return "stop"


def _route_after_re_run_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")):
        return "stop"
    if not bool(state.get("crash_found")):
        return "stop"
    if bool(state.get("restart_to_plan")):
        if int(state.get("restart_to_plan_count") or 0) > _re_restart_limit():
            return "stop"
        return "plan"
    if bool(state.get("crash_repro_done")) and not bool(state.get("crash_repro_ok")):
        return "plan"
    if bool(state.get("crash_repro_done")) and bool(state.get("crash_repro_ok")):
        return "crash-analysis"
    return "stop"


def _route_after_crash_analysis_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")):
        return "stop"
    if bool(state.get("restart_to_plan")):
        if int(state.get("restart_to_plan_count") or 0) > _re_restart_limit():
            return "stop"
        return "plan"
    verdict = _normalize_crash_analysis_verdict(str(state.get("crash_analysis_verdict") or ""))
    if verdict == "false_positive":
        return "plan"
    return "stop"


def _recommended_next_step(state: FuzzWorkflowRuntimeState) -> str:
    state = cast(FuzzWorkflowRuntimeState, _normalize_error_state(cast(dict[str, Any], state)))
    last_step = str(state.get("last_step") or "").strip().lower()
    if not last_step:
        return "stop"
    if last_step == "init":
        return _route_after_init_state(state)
    if last_step == "analysis":
        return _route_after_analysis_state(state)
    if last_step == "vuln-hunt":
        return _route_after_vuln_hunt_state(state)
    if last_step == "plan":
        return _route_after_plan_state(state)
    if last_step == "synthesize":
        return _route_after_synthesize_state(state)
    if last_step == "build":
        return _route_after_build_state(state)
    if last_step == "fix_build":
        return _route_after_fix_build_state(state)
    if last_step == "run":
        return _route_after_run_state(state)
    if last_step == "per-input-replay":
        return _route_after_per_input_replay_state(state)
    if last_step == "fix_crash":
        return _route_after_fix_crash_state(state)
    if last_step == "crash-triage":
        return _route_after_crash_triage_state(state)
    if last_step == "fix-harness":
        return _route_after_fix_harness_state(state)
    if last_step == "coverage-analysis":
        return _route_after_coverage_analysis_state(state)
    if last_step == "improve-harness":
        return _route_after_improve_harness_state(state)
    if last_step == "re-build":
        return _route_after_re_build_state(state)
    if last_step == "re-run":
        return _route_after_re_run_state(state)
    if last_step == "crash-analysis":
        return _route_after_crash_analysis_state(state)
    return "stop"


def _route_after_init_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")) or (state.get("last_error") or "").strip():
        return "stop"
    raw = (state.get("resume_from_step") or "").strip().lower()
    if raw in {"fix-harness", "fix_harness"}:
        raw = "plan"
    if raw in {"fix_build", "fix_crash"}:
        raw = "build"
    if raw == "vuln_hunt":
        raw = "vuln-hunt"
    allowed = {
        "analysis",
        "vuln-hunt",
        "plan",
        "synthesize",
        "build",
        "run",
        "per-input-replay",
        "crash-triage",
        "coverage-analysis",
        "improve-harness",
        "re-build",
        "re-run",
        "crash-analysis",
    }
    if raw == "repro_crash":
        raw = "re-build"
    if raw in allowed:
        return raw
    return "analysis"


def _route_after_vuln_hunt_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")):
        return "stop"
    if str(state.get("last_error") or "").strip():
        return "stop"

    source = str(state.get("_vuln_hunt_entry_source") or "").strip()

    if source == "improve-harness":
        return "build"

    if source == "coverage-analysis":
        active_priority = float(state.get("vuln_hunt_highest_priority") or 0.0)
        if active_priority >= _vuln_replan_priority_threshold():
            return "plan"
        return "improve-harness"

    # "analysis" or unknown sources — initial flow, unchanged
    return "plan"


def _should_stage_stop(state: FuzzWorkflowRuntimeState, step_name: str) -> bool:
    target = (state.get("stop_after_step") or "").strip().lower()
    return bool(target) and target == step_name


def _apply_stage_stop_guard(state: FuzzWorkflowRuntimeState, step_name: str, next_step: str) -> str:
    if _should_stage_stop(state, step_name):
        return "stop"
    return next_step


def build_fuzz_workflow() -> StateGraph:
    graph: StateGraph = StateGraph(FuzzWorkflowRuntimeState)

    graph.add_node("init", _node_init)
    graph.add_node("analysis", _node_analysis)
    graph.add_node("vuln-hunt", _node_vuln_hunt)
    graph.add_node("plan", _node_plan)
    graph.add_node("synthesize", _node_synthesize)
    graph.add_node("build", _node_build)
    graph.add_node("per-input-replay", _node_per_input_replay)
    graph.add_node("coverage-analysis", _node_coverage_analysis)
    graph.add_node("improve-harness", _node_improve_harness)
    graph.add_node("re-build", _node_re_build)
    graph.add_node("re-run", _node_re_run)
    graph.add_node("crash-analysis", _node_crash_analysis)
    graph.add_node("run", _node_run)
    graph.add_node("crash-triage", _node_crash_triage)

    graph.set_entry_point("init")

    def _route_after_plan(state: FuzzWorkflowRuntimeState) -> str:
        if (state.get("last_error") or "").strip():
            return "stop"
        if _should_stage_stop(state, "plan"):
            return "stop"
        return "synthesize"

    def _route_after_analysis(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_analysis_state(state)
        return _apply_stage_stop_guard(state, "analysis", nxt)

    def _route_after_synthesize(state: FuzzWorkflowRuntimeState) -> str:
        if (state.get("last_error") or "").strip():
            return "stop"
        if _should_stage_stop(state, "synthesize"):
            return "stop"
        return "build"

    def _route_after_build(state: FuzzWorkflowRuntimeState) -> str:
        if not (state.get("last_error") or "").strip():
            if _should_stage_stop(state, "build"):
                return "stop"
        return _route_after_build_state(state)

    def _route_after_run(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_run_state(state)
        return _apply_stage_stop_guard(state, "run", nxt)

    def _route_after_per_input_replay(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_per_input_replay_state(state)
        return _apply_stage_stop_guard(state, "per-input-replay", nxt)

    def _route_after_crash_triage(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_crash_triage_state(state)
        return _apply_stage_stop_guard(state, "crash-triage", nxt)

    def _route_after_coverage_analysis(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_coverage_analysis_state(state)
        return _apply_stage_stop_guard(state, "coverage-analysis", nxt)

    def _route_after_improve_harness(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_improve_harness_state(state)
        return _apply_stage_stop_guard(state, "improve-harness", nxt)

    def _route_after_re_build(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_re_build_state(state)
        return _apply_stage_stop_guard(state, "re-build", nxt)

    def _route_after_re_run(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_re_run_state(state)
        return _apply_stage_stop_guard(state, "re-run", nxt)

    def _route_after_crash_analysis(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_crash_analysis_state(state)
        return _apply_stage_stop_guard(state, "crash-analysis", nxt)

    graph.add_conditional_edges(
        "init",
        _route_after_init_state,
        {
            "analysis": "analysis",
            "vuln-hunt": "vuln-hunt",
            "plan": "plan",
            "synthesize": "synthesize",
            "build": "build",
            "run": "run",
            "per-input-replay": "per-input-replay",
            "crash-triage": "crash-triage",
            "coverage-analysis": "coverage-analysis",
            "improve-harness": "improve-harness",
            "re-build": "re-build",
            "re-run": "re-run",
            "crash-analysis": "crash-analysis",
            "stop": END,
        },
    )
    graph.add_conditional_edges("analysis", _route_after_analysis, {"vuln-hunt": "vuln-hunt", "plan": "plan", "stop": END})
    graph.add_conditional_edges(
        "vuln-hunt",
        lambda state: _apply_stage_stop_guard(state, "vuln-hunt", _route_after_vuln_hunt_state(state)),
        {"plan": "plan", "improve-harness": "improve-harness", "build": "build", "stop": END},
    )
    graph.add_conditional_edges("plan", _route_after_plan, {"synthesize": "synthesize", "stop": END})
    graph.add_conditional_edges("synthesize", _route_after_synthesize, {"build": "build", "stop": END})
    graph.add_conditional_edges(
        "build",
        _route_after_build,
        {"run": "run", "plan": "plan", "stop": END},
    )
    graph.add_conditional_edges(
        "run",
        _route_after_run,
        {
            "per-input-replay": "per-input-replay",
            "crash-triage": "crash-triage",
            "plan": "plan",
            "stop": END,
        },
    )
    graph.add_conditional_edges(
        "per-input-replay",
        _route_after_per_input_replay,
        {"coverage-analysis": "coverage-analysis", "stop": END},
    )
    graph.add_conditional_edges(
        "crash-triage",
        _route_after_crash_triage,
        {"re-build": "re-build", "plan": "plan", "stop": END},
    )
    graph.add_conditional_edges(
        "coverage-analysis",
        _route_after_coverage_analysis,
        {"vuln-hunt": "vuln-hunt", "improve-harness": "improve-harness", "stop": END},
    )
    graph.add_conditional_edges(
        "improve-harness",
        _route_after_improve_harness,
        {"build": "build", "plan": "plan", "vuln-hunt": "vuln-hunt", "stop": END},
    )
    graph.add_conditional_edges(
        "re-build",
        _route_after_re_build,
        {"re-run": "re-run", "plan": "plan", "stop": END},
    )
    graph.add_conditional_edges(
        "re-run",
        _route_after_re_run,
        {"crash-analysis": "crash-analysis", "plan": "plan", "stop": END},
    )
    graph.add_conditional_edges(
        "crash-analysis",
        _route_after_crash_analysis,
        {"plan": "plan", "stop": END},
    )
    return graph


def run_fuzz_workflow(inp: FuzzWorkflowInput) -> dict[str, Any]:
    total_budget_log = "unlimited" if int(inp.time_budget) == 0 else f"{int(inp.time_budget)}s"
    run_budget_log = "unlimited" if int(inp.run_time_budget) == 0 else f"{int(inp.run_time_budget)}s"
    resume_step = (inp.resume_from_step or "").strip().lower()
    if resume_step == "repro_crash":
        resume_step = "re-build"
    resume_root = str(inp.resume_repo_root or "").strip()
    stop_after_step = (inp.stop_after_step or "").strip().lower()
    job_id = str(
        os.environ.get("SHERPA_CURRENT_JOB_ID")
        or os.environ.get("SHERPA_JOB_ID")
        or ""
    ).strip()
    resolved_context_dir = str(inp.context_dir or "").strip()
    if not resolved_context_dir:
        guessed = context_dir_for_repo_root(inp.resume_repo_root)
        resolved_context_dir = str(guessed or "").strip()
    control_doc, workflow_doc = read_context_docs(
        resolved_context_dir or None,
        job_id=job_id,
    )
    control_state = strip_meta(control_doc)
    workflow_state = strip_meta(workflow_doc)
    _wf_log(
        None,
        "workflow start "
        f"repo={inp.repo_url} docker_image={inp.docker_image or '(native)'} "
        f"time_budget={total_budget_log} run_time_budget={run_budget_log} "
        f"resume_step={resume_step or '-'} resume_root={resume_root or '-'} "
        f"stop_after_step={stop_after_step or '-'}",
    )
    t0 = time.perf_counter()
    try:
        max_steps_env = int(os.environ.get("SHERPA_WORKFLOW_MAX_STEPS", "0"))
    except Exception:
        max_steps_env = 0
    # max_steps <= 0 means unlimited workflow steps.
    max_steps = 0 if max_steps_env <= 0 else max(3, max_steps_env)
    wf = build_fuzz_workflow().compile()
    # Keep persisted contexts as defaults, but ensure current stage dispatch
    # parameters from k8s payload always take precedence.
    invoke_payload: dict[str, Any] = {
        **control_state,
        **workflow_state,
        "repo_url": inp.repo_url,
        "model": str(inp.model or ""),
        "email": inp.email,
        "time_budget": inp.time_budget,
        "run_time_budget": inp.run_time_budget,
        "workflow_started_at": time.time(),
        "max_len": inp.max_len,
        "docker_image": inp.docker_image,
        "ai_key_path": str(inp.ai_key_path),
        "resume_from_step": resume_step,
        "resume_repo_root": str(inp.resume_repo_root or ""),
        "stop_after_step": stop_after_step,
        "context_dir": resolved_context_dir,
        "coverage_loop_max_rounds": max(
            0,
            int(inp.coverage_loop_max_rounds if inp.coverage_loop_max_rounds is not None else 0),
        ),
        "max_fix_rounds": max(
            0,
            int(inp.max_fix_rounds if inp.max_fix_rounds is not None else 0),
        ),
        "same_error_max_retries": max(
            0,
            int(inp.same_error_max_retries if inp.same_error_max_retries is not None else 0),
        ),
        "max_steps": max_steps,
    }
    raw: Any = wf.invoke(invoke_payload)
    out = _normalize_error_state(cast(dict[str, Any], raw) if isinstance(raw, dict) else {})
    final_context_dir = str(context_dir_for_repo_root(out.get("repo_root")) or resolved_context_dir).strip()
    if final_context_dir:
        current_control_doc, current_workflow_doc = read_context_docs(
            final_context_dir,
            job_id=job_id,
        )
        merged_control_doc, merged_workflow_doc = merge_result_into_contexts(
            out,
            control=current_control_doc,
            workflow=current_workflow_doc,
        )
        try:
            write_context_docs(
                final_context_dir,
                control=merged_control_doc,
                workflow=merged_workflow_doc,
                job_id=job_id,
            )
        except Exception:
            pass
    try:
        _write_run_summary(out)
    except Exception:
        pass
    msg = str(out.get("message") or "Fuzzing completed.").strip()
    recommended_next = _recommended_next_step(cast(FuzzWorkflowRuntimeState, out))
    err = dict(out.get("error") or {})
    if bool(out.get("failed")) or bool(err.get("terminal")):
        _wf_log(out, f"workflow end status=failed dt={_fmt_dt(time.perf_counter()-t0)}")
        terminal_reason = str(out.get("run_terminal_reason") or err.get("code") or "").strip() or str(
            out.get("fix_build_terminal_reason") or ""
        ).strip()
        if terminal_reason:
            msg = f"{terminal_reason}: {msg}"
        raise RuntimeError(msg or "workflow failed")
    # If we stopped due to an error but didn't mark failed, still surface it.
    # crash_found alone is not enough to suppress — if we're still in an
    # unrecovered repair (e.g. crash_triage→harness_bug→plan_repair failed),
    # the error must surface as a workflow failure.
    last_error = str(out.get("last_error") or err.get("message") or "").strip()
    repair_unrecovered = bool(out.get("repair_mode")) and bool(last_error)
    should_surface = bool(last_error) and (not bool(out.get("crash_found")) or repair_unrecovered)
    if should_surface:
        if stop_after_step and recommended_next != "stop":
            _wf_log(
                out,
                (
                    "workflow end status=stage_recoverable "
                    f"next={recommended_next} dt={_fmt_dt(time.perf_counter()-t0)}"
                ),
            )
        else:
            _wf_log(out, f"workflow end status=error dt={_fmt_dt(time.perf_counter()-t0)}")
            raise RuntimeError(last_error)

    if not (should_surface and stop_after_step and recommended_next != "stop"):
        _wf_log(out, f"workflow end status=ok dt={_fmt_dt(time.perf_counter()-t0)}")
    return {
        "message": msg,
        "error": dict(out.get("error") or {}),
        "repo_root": str(out.get("repo_root") or ""),
        "workflow_last_step": str(out.get("last_step") or ""),
        "workflow_active_step": str(out.get("next") or ""),
        "workflow_recommended_next": str(recommended_next or ""),
        "stop_after_step": stop_after_step,
        "fix_build_terminal_reason": str(out.get("fix_build_terminal_reason") or ""),
        "fix_build_attempts": int(out.get("fix_build_attempts") or 0),
        "fix_build_noop_streak": int(out.get("fix_build_noop_streak") or 0),
        "fix_build_rule_hits": list(out.get("fix_build_rule_hits") or []),
        "run_error_kind": str(out.get("run_error_kind") or ""),
        "run_terminal_reason": str(out.get("run_terminal_reason") or ""),
        "run_idle_seconds": int(out.get("run_idle_seconds") or 0),
        "run_children_exit_count": int(out.get("run_children_exit_count") or 0),
        "coverage_loop_max_rounds": int(
            out.get("coverage_loop_max_rounds")
            if out.get("coverage_loop_max_rounds") is not None
            else 0
        ),
        "coverage_loop_round": int(out.get("coverage_loop_round") or 0),
        "coverage_should_improve": bool(out.get("coverage_should_improve") or False),
        "coverage_improve_reason": str(out.get("coverage_improve_reason") or ""),
        "coverage_bottleneck_kind": str(out.get("coverage_bottleneck_kind") or ""),
        "coverage_bottleneck_reason": str(out.get("coverage_bottleneck_reason") or ""),
        "coverage_target_name": str(out.get("coverage_target_name") or ""),
        "coverage_target_api": str(out.get("coverage_target_api") or ""),
        "coverage_target_type": str(out.get("coverage_target_type") or ""),
        "coverage_seed_profile": str(out.get("coverage_seed_profile") or ""),
        "coverage_seed_quality": dict(out.get("coverage_seed_quality") or {}),
        "coverage_seed_families_suggested": list(out.get("coverage_seed_families_suggested") or []),
        "coverage_seed_families_covered": list(out.get("coverage_seed_families_covered") or []),
        "coverage_seed_families_missing": list(out.get("coverage_seed_families_missing") or []),
        "coverage_quality_flags": list(out.get("coverage_quality_flags") or []),
        "coverage_quality_oracle": str(out.get("coverage_quality_oracle") or ""),
        "coverage_target_depth_score": int(out.get("coverage_target_depth_score") or 0),
        "coverage_target_depth_class": str(out.get("coverage_target_depth_class") or ""),
        "coverage_selection_bias_reason": str(out.get("coverage_selection_bias_reason") or ""),
        "coverage_target_score_breakdown": dict(out.get("coverage_target_score_breakdown") or {}),
        "coverage_plateau_streak": int(out.get("coverage_plateau_streak") or 0),
        "coverage_last_max_cov": int(out.get("coverage_last_max_cov") or 0),
        "coverage_last_ft": int(out.get("coverage_last_ft") or 0),
        "coverage_replan_required": bool(out.get("coverage_replan_required") or False),
        "coverage_replan_reason": str(out.get("coverage_replan_reason") or ""),
        "coverage_replan_effective": bool(out.get("coverage_replan_effective") or False),
        "coverage_improve_mode": str(out.get("coverage_improve_mode") or ""),
        "coverage_round_budget_exhausted": bool(out.get("coverage_round_budget_exhausted") or False),
        "coverage_stop_reason": str(out.get("coverage_stop_reason") or ""),
        "coverage_corpus_sources": list(out.get("coverage_corpus_sources") or []),
        "coverage_seed_counts": dict(out.get("coverage_seed_counts") or {}),
        "coverage_parallel_diagnosis_code": str(out.get("coverage_parallel_diagnosis_code") or ""),
        "coverage_parallel_diagnosis": str(out.get("coverage_parallel_diagnosis") or ""),
        "coverage_per_input_manifest_path": str(out.get("coverage_per_input_manifest_path") or ""),
        "coverage_frontier_path": str(out.get("coverage_frontier_path") or ""),
        "coverage_frontier_summary": dict(out.get("coverage_frontier_summary") or {}),
        "coverage_replay_runtime_sec": float(out.get("coverage_replay_runtime_sec") or 0.0),
        "coverage_replay_binary_hash": str(out.get("coverage_replay_binary_hash") or ""),
        "coverage_replay_binary_dir": str(out.get("coverage_replay_binary_dir") or ""),
        "coverage_replay_binary_count": int(out.get("coverage_replay_binary_count") or 0),
        "coverage_replay_stage_success": bool(out.get("coverage_replay_stage_success") or False),
        "coverage_replay_error": str(out.get("coverage_replay_error") or ""),
        "coverage_replay_manifest_fresh_for_current_binary": bool(
            out.get("coverage_replay_manifest_fresh_for_current_binary") or False
        ),
        "coverage_replay_queue_drained": bool(out.get("coverage_replay_queue_drained") or False),
        "coverage_replay_pending_inputs": int(out.get("coverage_replay_pending_inputs") or 0),
        "coverage_replay_failed_inputs": int(out.get("coverage_replay_failed_inputs") or 0),
        "coverage_replay_processed_inputs": int(out.get("coverage_replay_processed_inputs") or 0),
        "coverage_replay_total_inputs": int(out.get("coverage_replay_total_inputs") or 0),
        "cold_start_seed_replan_triggered": bool(out.get("cold_start_seed_replan_triggered") or False),
        "degraded_seed_replan_triggered": bool(out.get("degraded_seed_replan_triggered") or False),
        "cold_start_trigger_snapshot": dict(out.get("cold_start_trigger_snapshot") or {}),
        "coverage_history": list(out.get("coverage_history") or []),
        "analysis_evidence_count": int(out.get("analysis_evidence_count") or 0),
        "security_evidence_count": int(out.get("security_evidence_count") or 0),
        "vuln_candidate_count": int(out.get("vuln_candidate_count") or 0),
        "vuln_hunting_enabled": bool(out.get("vuln_hunting_enabled") or False),
        "vuln_hunt_enabled": bool(out.get("vuln_hunt_enabled") or False),
        "vuln_hunt_iteration": int(out.get("vuln_hunt_iteration") or 0),
        "vuln_hunt_active_candidate_id": str(out.get("vuln_hunt_active_candidate_id") or ""),
        "vuln_hunt_highest_priority": float(out.get("vuln_hunt_highest_priority") or 0.0),
        "vuln_hunt_candidate_count": int(out.get("vuln_hunt_candidate_count") or 0),
        "vuln_hunt_degraded": bool(out.get("vuln_hunt_degraded") or False),
        "vuln_hunt_last_reason": str(out.get("vuln_hunt_last_reason") or ""),
        "vuln_hunt_summary_path": str(out.get("vuln_hunt_summary_path") or ""),
        "vuln_hunt_events_path": str(out.get("vuln_hunt_events_path") or ""),
        "vuln_hunt_rerun_requested": bool(out.get("vuln_hunt_rerun_requested") or False),
        "vuln_focus_profile": str(out.get("vuln_focus_profile") or ""),
        "target_surface_policy": str(out.get("target_surface_policy") or ""),
        "security_priority_mode": bool(out.get("security_priority_mode") or False),
        "latest_vuln_decision_snapshot": dict(out.get("latest_vuln_decision_snapshot") or {}),
        "vuln_candidates_path": str(out.get("vuln_candidates_path") or ""),
        "crash_vuln_report_path": str(out.get("crash_vuln_report_path") or ""),
        "latest_crash_vuln_candidate": dict(out.get("latest_crash_vuln_candidate") or {}),
        "crash_vuln_candidate_count": int(out.get("crash_vuln_candidate_count") or 0),
        "target_scoring_enabled": bool(out.get("target_scoring_enabled") or False),
        "target_score_breakdown_available": bool(out.get("target_score_breakdown_available") or False),
        "constraint_memory_count": int(out.get("constraint_memory_count") or 0),
        "constraint_memory_path": str(out.get("constraint_memory_path") or ""),
        "decision_trace_count": int(out.get("decision_trace_count") or 0),
        "latest_decision_snapshot": dict(out.get("latest_decision_snapshot") or {}),
        "crash_signature_dedup_hit": bool(out.get("crash_signature_dedup_hit") or False),
        "plan_retry_reason": str(out.get("plan_retry_reason") or ""),
        "plan_targets_schema_valid_before_retry": bool(out.get("plan_targets_schema_valid_before_retry") or False),
        "plan_targets_schema_valid_after_retry": bool(out.get("plan_targets_schema_valid_after_retry") or False),
        "plan_used_fallback_targets": bool(out.get("plan_used_fallback_targets") or False),
        "max_fix_rounds": int(out.get("max_fix_rounds") if out.get("max_fix_rounds") is not None else 0),
        "same_error_max_retries": int(
            out.get("same_error_max_retries")
            if out.get("same_error_max_retries") is not None
            else 0
        ),
        "fix_action_type": str(out.get("fix_action_type") or ""),
        "fix_effect": str(out.get("fix_effect") or ""),
        "build_error_signature_before": str(out.get("build_error_signature_before") or ""),
        "build_error_signature_after": str(out.get("build_error_signature_after") or ""),
        "crash_repro_done": bool(out.get("crash_repro_done") or False),
        "crash_repro_ok": bool(out.get("crash_repro_ok") or False),
        "crash_repro_rc": int(out.get("crash_repro_rc") or 0),
        "crash_repro_report_path": str(out.get("crash_repro_report_path") or ""),
        "crash_repro_json_path": str(out.get("crash_repro_json_path") or ""),
        "crash_triage_done": bool(out.get("crash_triage_done") or False),
        "crash_triage_label": str(out.get("crash_triage_label") or ""),
        "crash_triage_confidence": float(out.get("crash_triage_confidence") or 0.0),
        "crash_triage_reason": str(out.get("crash_triage_reason") or ""),
        "crash_triage_report_path": str(out.get("crash_triage_report_path") or ""),
        "crash_triage_json_path": str(out.get("crash_triage_json_path") or ""),
        "repair_mode": bool(out.get("repair_mode") or False),
        "repair_origin_stage": str(out.get("repair_origin_stage") or ""),
        "repair_error_kind": str(out.get("repair_error_kind") or ""),
        "repair_error_code": str(out.get("repair_error_code") or ""),
        "repair_signature": str(out.get("repair_signature") or ""),
        "repair_recent_attempts": list(out.get("repair_recent_attempts") or []),
        "repair_error_digest": dict(out.get("repair_error_digest") or {}),
        "re_build_done": bool(out.get("re_build_done") or False),
        "re_build_ok": bool(out.get("re_build_ok") or False),
        "re_build_rc": int(out.get("re_build_rc") or 0),
        "re_build_report_path": str(out.get("re_build_report_path") or ""),
        "re_build_json_path": str(out.get("re_build_json_path") or ""),
        "re_run_done": bool(out.get("re_run_done") or False),
        "re_run_ok": bool(out.get("re_run_ok") or False),
        "re_run_rc": int(out.get("re_run_rc") or 0),
        "re_run_report_path": str(out.get("re_run_report_path") or ""),
        "re_run_json_path": str(out.get("re_run_json_path") or ""),
        "crash_analysis_done": bool(out.get("crash_analysis_done") or False),
        "crash_analysis_verdict": str(out.get("crash_analysis_verdict") or ""),
        "crash_analysis_reason": str(out.get("crash_analysis_reason") or ""),
        "crash_analysis_report_path": str(out.get("crash_analysis_report_path") or ""),
        "crash_analysis_json_path": str(out.get("crash_analysis_json_path") or ""),
        "re_workspace_root": str(out.get("re_workspace_root") or ""),
        "last_fuzzer": str(out.get("last_fuzzer") or ""),
        "last_crash_artifact": str(out.get("last_crash_artifact") or ""),
        "restart_to_plan": bool(out.get("restart_to_plan") or False),
        "restart_to_plan_reason": str(out.get("restart_to_plan_reason") or ""),
        "restart_to_plan_stage": str(out.get("restart_to_plan_stage") or ""),
        "restart_to_plan_error_text": str(out.get("restart_to_plan_error_text") or ""),
        "restart_to_plan_report_path": str(out.get("restart_to_plan_report_path") or ""),
        "restart_to_plan_count": int(out.get("restart_to_plan_count") or 0),
        "build_error_kind": str(out.get("build_error_kind") or ""),
        "build_error_code": str(out.get("build_error_code") or ""),
    }
