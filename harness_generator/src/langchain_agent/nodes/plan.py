"""Carved from workflow_graph.py - '_node_plan' LangGraph node."""

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
    _enter_step,
    _fmt_dt,
    _wf_log,
)
from workflow_helpers import (
    _attach_prompt_render_status,
    _build_fix_harness_crash_context,
    _build_repair_snapshot,
    _clear_error_markers_on_success,
    _clear_opencode_done_sentinel,
    _collect_feedback_for_group,
    _coverage_attack_hint_feedback_lines,
    _derive_plan_policy,
    _enrich_targets_depth,
    _has_codex_key,
    _make_plan_hint,
    _materialize_analysis_context_from_companion,
    _normalize_seed_profile,
    _opencode_cli_retries,
    _plan_idle_timeout_sec,
    _prepare_antlr_assist_context,
    _prepare_target_analysis_context,
    _record_decision_trace,
    _remaining_time_budget_sec,
    _render_opencode_prompt_safe,
    _run_vuln_hunt_subphase,
    _select_primary_target,
    _targets_material_signature,
    _validate_targets_json,
    _write_execution_plan_doc,
    _write_fallback_targets_json,
    _write_selected_targets_doc,
    _write_stage_feedback,
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


def _node_plan(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "plan")
    if stop_now:
        return state
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> plan")
    hint = (state.get("codex_hint") or "").strip()
    prompt_render_issue = ""
    restart_to_plan = bool(state.get("restart_to_plan") or False)
    restart_reason = str(state.get("restart_to_plan_reason") or "").strip()
    restart_stage = str(state.get("restart_to_plan_stage") or "").strip()
    restart_error_text = str(state.get("restart_to_plan_error_text") or "").strip()
    restart_report_path = str(state.get("restart_to_plan_report_path") or "").strip()
    repair_snapshot = _build_repair_snapshot(cast(dict[str, Any], state))
    repair_mode = bool(repair_snapshot.get("repair_mode"))
    repair_origin_stage = str(repair_snapshot.get("repair_origin_stage") or "build")
    repair_recent_attempts = list(repair_snapshot.get("repair_recent_attempts") or [])
    repair_attempt_index = int(repair_snapshot.get("repair_attempt_index") or 0)
    repair_force_strategy_change = bool(repair_snapshot.get("repair_strategy_force_change") or False)
    repair_error_digest = dict(repair_snapshot.get("repair_error_digest") or {})
    constraint_memory_entry = dict(repair_snapshot.get("constraint_memory_entry") or {})
    constraint_memory_count = int(repair_snapshot.get("constraint_memory_count") or 0)
    constraint_memory_path = str(repair_snapshot.get("constraint_memory_path") or "")
    antlr_context_path = str(state.get("antlr_context_path") or "").strip()
    antlr_context_summary = str(state.get("antlr_context_summary") or "").strip()
    target_analysis_path = str(state.get("target_analysis_path") or "").strip()
    target_analysis_summary = str(state.get("target_analysis_summary") or "").strip()
    analysis_context_path = str(state.get("analysis_context_path") or "").strip()
    analysis_evidence_count = int(state.get("analysis_evidence_count") or 0)
    security_evidence_count = int(state.get("security_evidence_count") or 0)
    vuln_candidate_count = int(state.get("vuln_candidate_count") or 0)
    if not antlr_context_path and not antlr_context_summary:
        antlr_context_path, antlr_context_summary = _prepare_antlr_assist_context(gen.repo_root)
    if not target_analysis_path and not target_analysis_summary:
        target_analysis_path, target_analysis_summary = _prepare_target_analysis_context(gen.repo_root)
    if not analysis_context_path:
        try:
            (
                analysis_context_path,
                analysis_evidence_count,
                security_evidence_count,
                vuln_candidate_count,
                companion_summary,
            ) = _materialize_analysis_context_from_companion(
                repo_root=gen.repo_root,
                antlr_context_path=antlr_context_path,
                antlr_context_summary=antlr_context_summary,
                target_analysis_path=target_analysis_path,
                target_analysis_summary=target_analysis_summary,
            )
            if analysis_context_path:
                _wf_log(
                    cast(dict[str, Any], state),
                    "plan: hydrated analysis_context.json from companion artifacts"
                    + (f" ({companion_summary})" if companion_summary else ""),
                )
        except Exception as exc:
            _wf_log(cast(dict[str, Any], state), f"plan: companion analysis hydration skipped: {exc}")
    if _vuln_hunting_enabled() and (
        int(state.get("vuln_hunt_iteration") or 0) <= 0
        or bool(state.get("vuln_hunt_rerun_requested") or False)
    ):
        try:
            hunt_state = _run_vuln_hunt_subphase({**state, "analysis_context_path": analysis_context_path})
            state = cast(FuzzWorkflowRuntimeState, hunt_state)
            state["vuln_hunt_rerun_requested"] = False
            vuln_candidate_count = int(state.get("vuln_candidate_count") or vuln_candidate_count)
            hunt_note = (
                "Vulnerability hunt candidate worklist is available at `fuzz/vuln_candidates.json`; "
                f"candidate_count={vuln_candidate_count}, "
                f"active_candidate={state.get('vuln_hunt_active_candidate_id') or 'none'}."
            )
            hint = (hint + "\n\n" + hunt_note).strip() if hint else hunt_note
            if state.get("vuln_hunt_degraded"):
                prompt_render_issue = "; ".join(
                    x
                    for x in [
                        prompt_render_issue,
                        str(state.get("vuln_hunt_last_reason") or ""),
                    ]
                    if str(x).strip()
                )
        except Exception as exc:
            degraded = f"vuln_hunt_failed:{exc}"
            prompt_render_issue = "; ".join(x for x in [prompt_render_issue, degraded] if str(x).strip())
            state = cast(FuzzWorkflowRuntimeState, _attach_prompt_render_status(dict(state), issue=degraded))
            _wf_log(cast(dict[str, Any], state), f"plan: vuln hunt degraded -> {exc}")
    if antlr_context_summary:
        antlr_note = (
            "ANTLR-assisted static context is available. Prefer this structure-grounded context when selecting targets.\n"
            f"{antlr_context_summary}"
        )
        hint = (hint + "\n\n" + antlr_note).strip() if hint else antlr_note
    if target_analysis_summary:
        target_note = (
            "Tool-assisted target analysis is available. Use `fuzz/target_analysis.json` when selecting targets and seed profiles.\n"
            f"{target_analysis_summary}"
        )
        hint = (hint + "\n\n" + target_note).strip() if hint else target_note
    if analysis_context_path:
        analysis_note = (
            "Unified analysis context is available at `fuzz/analysis_context.json`; "
            f"evidence_count={analysis_evidence_count}. "
            "Prefer evidence-backed target choices and cite evidence ids in PLAN rationale."
        )
        hint = (hint + "\n\n" + analysis_note).strip() if hint else analysis_note
    injected_ctx = ""
    prev_plan_text = ""
    prev_targets_text = ""
    fuzz_dir = gen.repo_root / "fuzz"
    plan_md_path = fuzz_dir / "PLAN.md"
    targets_json_path = fuzz_dir / "targets.json"
    try:
        if plan_md_path.is_file():
            prev_plan_text = plan_md_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        prev_plan_text = ""
    try:
        if targets_json_path.is_file():
            prev_targets_text = targets_json_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        prev_targets_text = ""
    prev_target_name = str(state.get("coverage_target_name") or "")
    prev_target_depth_score = int(state.get("coverage_target_depth_score") or 0)
    prev_target_depth_class = str(state.get("coverage_target_depth_class") or "")
    if restart_to_plan:
        report_tail = ""
        if restart_report_path:
            try:
                rp = Path(restart_report_path)
                if rp.is_file():
                    report_tail = "\n".join(
                        rp.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
                    )
            except Exception:
                report_tail = ""
        injected_ctx = (
            "Previous cycle failed and this planning step is now in repair mode.\n"
            f"- restart stage: {restart_stage or 'unknown'}\n"
            f"- restart reason: {restart_reason or 'unknown'}\n"
            f"- restart error: {(restart_error_text or 'n/a')[:4096]}\n"
        )
        if report_tail:
            injected_ctx += "\n=== re failure report tail ===\n" + report_tail + "\n"
        hint = (hint + "\n\n" + injected_ctx).strip() if hint else injected_ctx
    if repair_mode:
        repair_error_text = str(repair_snapshot.get("repair_error_text") or "")
        repair_stderr_tail = str(repair_snapshot.get("repair_stderr_tail") or "")
        repair_stdout_tail = str(repair_snapshot.get("repair_stdout_tail") or "")
        repair_signature = str(repair_snapshot.get("repair_signature") or "")
        repair_kind = str(repair_snapshot.get("repair_error_kind") or "generic_failure")
        repair_code = str(repair_snapshot.get("repair_error_code") or "")
        repair_blocks: list[str] = [
            "Repair context for this planning round:",
            f"- repair_origin_stage: {repair_origin_stage}",
            f"- repair_error_kind: {repair_kind}",
            f"- repair_error_code: {repair_code or 'n/a'}",
            f"- repair_signature: {repair_signature or 'n/a'}",
            f"- repair_attempt_index: {repair_attempt_index}",
        ]
        if repair_force_strategy_change:
            repair_blocks.append(
                "Strategy gate: repeated failure signature detected. This round must change strategy materially "
                "(target combination, harness API path, or build/link approach)."
            )
        if repair_error_digest:
            repair_blocks.append(
                "=== repair error digest ===\n"
                + json.dumps(repair_error_digest, ensure_ascii=False, indent=2)
            )
        if repair_error_text:
            repair_blocks.append("=== repair error ===\n" + repair_error_text[:4096])
        if repair_stderr_tail:
            repair_blocks.append("=== repair stderr tail ===\n" + "\n".join(repair_stderr_tail.splitlines()[-200:]))
        if repair_stdout_tail:
            repair_blocks.append("=== repair stdout tail ===\n" + "\n".join(repair_stdout_tail.splitlines()[-120:]))
        if repair_recent_attempts:
            repair_blocks.append(
                "=== recent repair attempts ===\n"
                + json.dumps(repair_recent_attempts[-5:], ensure_ascii=False, indent=2)
            )
        if constraint_memory_entry:
            repair_blocks.append(
                "=== constraint memory (repeated crash guidance) ===\n"
                + json.dumps(
                    {
                        "path": constraint_memory_path,
                        "entry_count": constraint_memory_count,
                        "active_entry": constraint_memory_entry,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        if repair_origin_stage == "fix-harness":
            crash_ctx, crash_known_issues = _build_fix_harness_crash_context(
                gen.repo_root,
                include_contents=True,
            )
            if crash_ctx:
                repair_blocks.append(crash_ctx)
            if crash_known_issues:
                prompt_render_issue = "; ".join(
                    x for x in [prompt_render_issue, *crash_known_issues] if str(x).strip()
                )
        repair_hint = "\n\n".join(part for part in repair_blocks if part.strip())
        hint = (hint + "\n\n" + repair_hint).strip() if hint else repair_hint
    seed_feedback = dict(state.get("coverage_seed_feedback") or {})
    harness_feedback = dict(state.get("coverage_harness_feedback") or {})
    quality_oracle = str(state.get("coverage_quality_oracle") or "").strip()
    if seed_feedback or harness_feedback or quality_oracle:
        feedback_lines: list[str] = ["Coverage feedback signals for planning:"]
        if quality_oracle:
            feedback_lines.append(f"- quality_oracle: {quality_oracle}")
        feedback_lines.extend(_coverage_attack_hint_feedback_lines(seed_feedback))
        if seed_feedback:
            feedback_lines.append("=== SeedFeedback ===\n" + json.dumps(seed_feedback, ensure_ascii=False, indent=2))
        if harness_feedback:
            feedback_lines.append("=== HarnessFeedback ===\n" + json.dumps(harness_feedback, ensure_ascii=False, indent=2))
        feedback_hint = "\n\n".join(feedback_lines)
        hint = (hint + "\n\n" + feedback_hint).strip() if hint else feedback_hint
    planning_feedback = _collect_feedback_for_group(gen.repo_root, "planning_synth", limit=3)
    if planning_feedback:
        feedback_hint = "Recent planning/synthesis failures (use these to avoid repeating the same mistakes):\n" + planning_feedback
        hint = (hint + "\n\n" + feedback_hint).strip() if hint else feedback_hint
    if not _has_codex_key():
        out = {
            **state,
            "last_step": "plan",
            "last_error": "Missing OPENAI_API_KEY for planning",
            "message": "plan failed",
        }
        out = _attach_prompt_render_status(out)
        _wf_log(cast(dict[str, Any], out), f"<- plan err=missing-key dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    try:
        plan_template_name = "plan_with_hint"
        plan_stage_skill = "plan"
        if repair_mode:
            if repair_origin_stage == "crash":
                plan_template_name = "plan_repair_crash_with_hint"
                plan_stage_skill = "plan_repair_crash"
            elif repair_origin_stage == "fix-harness":
                plan_template_name = "plan_repair_fix_harness_with_hint"
                plan_stage_skill = "plan_repair_fix_harness"
            elif repair_origin_stage == "coverage":
                plan_template_name = "plan_repair_coverage_with_hint"
                plan_stage_skill = "plan_repair_coverage"
            else:
                plan_template_name = "plan_repair_build_with_hint"
                plan_stage_skill = "plan_repair_build"
        render_known_issues: list[str] = []
        if repair_mode:
            digest = dict(repair_error_digest or {})
            if not str(digest.get("error_code") or "").strip():
                render_known_issues.append("missing repair_error_digest.error_code")
            if not str(digest.get("signature") or "").strip():
                render_known_issues.append("missing repair_error_digest.signature")
            if not str(digest.get("error_kind") or "").strip():
                render_known_issues.append("missing repair_error_digest.error_kind")
            if repair_origin_stage == "fix-harness":
                _, crash_known_issues = _build_fix_harness_crash_context(
                    gen.repo_root,
                    include_contents=False,
                )
                render_known_issues.extend(crash_known_issues)
        if hint:
            prompt, render_issue = _render_opencode_prompt_safe(
                plan_template_name,
                fallback_name="plan_repair_build_with_hint" if repair_mode else "plan_with_hint",
                hint=hint,
                fallback_hint=hint,
                known_issues=render_known_issues,
            )
            if render_issue:
                prompt_render_issue = str(render_issue)
                hint = (hint + "\n\nKnown Issues:\n- " + render_issue).strip()
                _wf_log(cast(dict[str, Any], state), f"plan: prompt render degraded -> {render_issue}")
            gen.patcher.run_codex_command(
                prompt,
                stage_skill=plan_stage_skill,
                timeout=_remaining_time_budget_sec(state),
                max_attempts=1,
                max_cli_retries=_opencode_cli_retries(),
                idle_timeout_override=_plan_idle_timeout_sec(),
            )
        else:
            gen._pass_plan_targets(timeout=_remaining_time_budget_sec(state))

        strict_targets = (os.environ.get("SHERPA_PLAN_STRICT_TARGETS_SCHEMA", "1").strip().lower() in {"1", "true", "yes", "on"})
        plan_retry_reason = ""
        plan_targets_schema_valid_before_retry = True
        plan_targets_schema_valid_after_retry = True
        plan_used_fallback_targets = False
        ok_targets, targets_err = _validate_targets_json(gen.repo_root)
        if strict_targets and not ok_targets:
            plan_retry_reason = "targets-schema"
            plan_targets_schema_valid_before_retry = False
            _wf_log(cast(dict[str, Any], state), f"plan: targets.json schema invalid -> {targets_err}; retrying once")
            cleared_done = _clear_opencode_done_sentinel(gen.repo_root)
            if cleared_done:
                _wf_log(cast(dict[str, Any], state), "plan: cleared stale done sentinel before schema-fix retry")
            prompt, schema_render_issue = _render_opencode_prompt_safe(
                "plan_fix_targets_schema",
                fallback_name="plan_with_hint",
                schema_error=targets_err,
                fallback_hint=f"Known Issues:\n- targets schema invalid: {targets_err}",
            )
            if schema_render_issue:
                prompt_render_issue = str(schema_render_issue)
                _wf_log(cast(dict[str, Any], state), f"plan: schema-fix prompt render degraded -> {schema_render_issue}")
            gen.patcher.run_codex_command(
                prompt,
                stage_skill="plan_fix_targets_schema",
                timeout=_remaining_time_budget_sec(state),
                max_attempts=1,
                max_cli_retries=_opencode_cli_retries(),
                idle_timeout_override=_plan_idle_timeout_sec(),
            )
            ok_targets, targets_err = _validate_targets_json(gen.repo_root)
            plan_targets_schema_valid_after_retry = bool(ok_targets)
            if not ok_targets:
                _wf_log(cast(dict[str, Any], state), f"plan: schema retry still invalid -> {targets_err}; applying deterministic fallback")
                plan_used_fallback_targets = _write_fallback_targets_json(
                    gen.repo_root,
                    antlr_context_path=antlr_context_path,
                    target_analysis_path=target_analysis_path,
                )
                ok_targets, targets_err = _validate_targets_json(gen.repo_root)
                if ok_targets:
                    plan_targets_schema_valid_after_retry = True
                    _wf_log(cast(dict[str, Any], state), "plan: deterministic fallback produced schema-valid targets.json")
                else:
                    plan_targets_schema_valid_after_retry = False
                out = {
                    **state,
                    "last_step": "plan",
                    "plan_retry_reason": plan_retry_reason,
                    "plan_targets_schema_valid_before_retry": plan_targets_schema_valid_before_retry,
                    "plan_targets_schema_valid_after_retry": plan_targets_schema_valid_after_retry,
                    "plan_used_fallback_targets": plan_used_fallback_targets,
                    "last_error": f"targets schema validation failed: {targets_err}",
                    "message": "plan failed",
                }
                out = _attach_prompt_render_status(out, issue=prompt_render_issue or targets_err)
                if not ok_targets:
                    _wf_log(cast(dict[str, Any], out), f"<- plan err=targets-schema dt={_fmt_dt(time.perf_counter()-t0)}")
                    return out

        # Back-fill depth_score/depth_class when OpenCode omits them so that
        # _select_primary_target can differentiate targets on replan.
        _enrich_targets_depth(gen.repo_root)

        fix_on_crash, _ = _derive_plan_policy(gen.repo_root)
        plan_hint = _make_plan_hint(gen.repo_root)
        if antlr_context_summary:
            plan_hint = (
                (plan_hint.strip() + "\n\n") if plan_hint.strip() else ""
            ) + (
                "Use `fuzz/antlr_plan_context.json` as grammar-aware grounding for API/entrypoint selection.\n"
                f"{antlr_context_summary}"
            )
        # Always skip already-attempted targets when re-entering plan, and
        # prefer deeper targets when an explicit replan was requested.
        _attempted = list(state.get("coverage_attempted_targets") or [])
        _is_replan = str(state.get("coverage_improve_mode") or "") == "replan" or bool(
            state.get("coverage_replan_required") or False
        )
        primary_target = _select_primary_target(
            gen.repo_root,
            exclude_names=_attempted if _attempted else None,
            prefer_deeper=_is_replan,
        )
        selected_targets_path = ""
        execution_plan_path = ""
        try:
            selected_targets_path, selected_targets_doc = _write_selected_targets_doc(
                gen.repo_root,
                exclude_names=_attempted if _attempted else None,
                prefer_deeper=_is_replan,
            )
            execution_plan_path, _ = _write_execution_plan_doc(gen.repo_root, selected_targets_doc)
        except Exception:
            selected_targets_doc = []
        selected_primary = selected_targets_doc[0] if selected_targets_doc else dict(primary_target)
        new_target_name = str(
            selected_primary.get("target_name")
            or selected_primary.get("target")
            or selected_primary.get("name")
            or primary_target.get("name")
            or ""
        )
        new_target_api = str(
            selected_primary.get("api")
            or primary_target.get("api")
            or new_target_name
        )
        new_target_type = str(
            selected_primary.get("target_type")
            or primary_target.get("target_type")
            or "generic"
        ).strip().lower()
        new_seed_profile = _normalize_seed_profile(
            str(
                selected_primary.get("seed_profile")
                or primary_target.get("seed_profile")
                or ""
            ),
            target_type=new_target_type,
            name=new_target_name,
            context=new_target_api,
        )
        new_depth_score = int(
            selected_primary.get("depth_score")
            or primary_target.get("depth_score")
            or 0
        )
        new_depth_class = str(
            selected_primary.get("depth_class")
            or primary_target.get("depth_class")
            or ""
        )
        new_selection_bias_reason = str(
            selected_primary.get("selection_bias_reason")
            or primary_target.get("selection_bias_reason")
            or ""
        )
        runner_up = selected_targets_doc[1] if len(selected_targets_doc) > 1 else {}
        seed_families_suggested = list(selected_primary.get("seed_families_suggested") or [])
        seed_families_optional = list(selected_primary.get("seed_families_optional") or [])
        selected_runtime_viability = str(selected_primary.get("runtime_viability") or "").strip().lower()
        target_scoring_enabled = bool(
            selected_primary.get("target_scoring_enabled")
            or any(bool(item.get("target_score_breakdown")) for item in selected_targets_doc)
        )
        target_score_breakdown_available = bool(
            selected_primary.get("target_score_breakdown_available")
            or selected_primary.get("score_breakdown")
            or any(bool(item.get("score_breakdown")) for item in selected_targets_doc)
        )
        security_priority_mode = bool(
            selected_primary.get("security_priority_mode")
            if selected_primary.get("security_priority_mode") is not None
            else (_vuln_hunting_enabled() and _vuln_score_mode() == "risk_first_v1")
        )
        replan_mode = str(state.get("coverage_improve_mode") or "") == "replan" or bool(state.get("coverage_replan_required") or False)
        replan_effective = bool(state.get("coverage_replan_effective") or False)
        replan_stop_reason = ""
        coverage_should_improve = bool(state.get("coverage_should_improve") or False)
        coverage_round_budget_exhausted = bool(state.get("coverage_round_budget_exhausted") or False)
        coverage_stop_reason = str(state.get("coverage_stop_reason") or "")
        coverage_replan_effective = bool(state.get("coverage_replan_effective") or False)
        coverage_replan_reason = str(state.get("coverage_replan_reason") or "")
        if replan_mode:
            new_plan_text = ""
            new_targets_text = ""
            try:
                if plan_md_path.is_file():
                    new_plan_text = plan_md_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                new_plan_text = ""
            try:
                if targets_json_path.is_file():
                    new_targets_text = targets_json_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                new_targets_text = ""
            depth_rank = {"shallow": 0, "medium": 1, "deep": 2}
            plan_changed = new_plan_text != prev_plan_text
            prev_targets_sig = _targets_material_signature(prev_targets_text)
            new_targets_sig = _targets_material_signature(new_targets_text)
            if prev_targets_sig is not None and new_targets_sig is not None:
                targets_changed = new_targets_sig != prev_targets_sig
            else:
                targets_changed = new_targets_text != prev_targets_text
            target_changed = new_target_name != prev_target_name
            # Treat depth changes as material only when replan actually moves to
            # a different target. This avoids false "effective replan" positives
            # when the same target gets minor heuristic score drift.
            depth_improved = bool(
                target_changed
                and (
                    new_depth_score > prev_target_depth_score
                    or depth_rank.get(new_depth_class, -1) > depth_rank.get(prev_target_depth_class, -1)
                )
            )
            replan_effective = any((plan_changed, targets_changed, target_changed, depth_improved))
            coverage_replan_effective = replan_effective
            if replan_effective:
                replan_stop_reason = ""
                coverage_replan_reason = (
                    "depth_improved"
                    if depth_improved and not target_changed
                    else "target_changed"
                    if target_changed
                    else "plan_changed"
                )
            else:
                replan_stop_reason = "no_material_change"
                coverage_should_improve = False
                coverage_round_budget_exhausted = True
                coverage_stop_reason = "no_material_change"
                coverage_replan_reason = "no_material_change"
                repair_force_strategy_change = True

        # ── Corpus carry-over on target change ──────────────────────────
        # When a replan selects a different target, copy the old fuzzer's
        # corpus into the new fuzzer's corpus dir so coverage progress
        # isn't lost.  Only carry over if seed profiles match (otherwise
        # the input format may be incompatible).
        _corpus_carryover_count = 0
        if replan_mode and target_changed and new_target_name:
            _prev_seed_prof = str(state.get("coverage_seed_profile") or "")
            if _prev_seed_prof == new_seed_profile or not _prev_seed_prof:
                _corpus_root = gen.repo_root / "fuzz" / "corpus"
                if _corpus_root.is_dir():
                    # Collect all corpus files from previous fuzzers
                    _old_corpus_files: list[Path] = []
                    for _sub in _corpus_root.iterdir():
                        if _sub.is_dir() and _sub.name != new_target_name:
                            for _cf in _sub.iterdir():
                                if _cf.is_file():
                                    _old_corpus_files.append(_cf)
                    if _old_corpus_files:
                        _new_corpus_dir = _corpus_root / new_target_name
                        _new_corpus_dir.mkdir(parents=True, exist_ok=True)
                        for _cf in _old_corpus_files:
                            _dst = _new_corpus_dir / _cf.name
                            if not _dst.exists():
                                try:
                                    import shutil
                                    shutil.copy2(str(_cf), str(_dst))
                                    _corpus_carryover_count += 1
                                except Exception:
                                    pass
                        if _corpus_carryover_count > 0:
                            print(f"[*] Corpus carry-over: copied {_corpus_carryover_count} files to {new_target_name}")

        out = {
            **state,
            "last_step": "plan",
            "last_error": "",
            "failed": False,
            "codex_hint": plan_hint,
            "plan_fix_on_crash": fix_on_crash,
            "plan_max_fix_rounds": 0,
            "plan_retry_reason": plan_retry_reason,
            "plan_targets_schema_valid_before_retry": plan_targets_schema_valid_before_retry,
            "plan_targets_schema_valid_after_retry": plan_targets_schema_valid_after_retry,
            "plan_used_fallback_targets": plan_used_fallback_targets,
            "antlr_context_path": antlr_context_path,
            "antlr_context_summary": antlr_context_summary,
            "target_analysis_path": target_analysis_path,
            "target_analysis_summary": target_analysis_summary,
            "analysis_context_path": analysis_context_path or str(state.get("analysis_context_path") or ""),
            "analysis_evidence_count": analysis_evidence_count,
            "security_evidence_count": security_evidence_count,
            "vuln_candidate_count": vuln_candidate_count,
            "vuln_hunting_enabled": bool(state.get("vuln_hunting_enabled") or _vuln_hunting_enabled()),
            "vuln_focus_profile": str(state.get("vuln_focus_profile") or "broad_high_risk"),
            "target_surface_policy": str(state.get("target_surface_policy") or "risk_first"),
            "security_priority_mode": bool(security_priority_mode),
            "selected_targets_path": selected_targets_path,
            "execution_plan_path": execution_plan_path,
            "coverage_attempted_targets": list(
                dict.fromkeys(
                    _attempted + [new_target_name or prev_target_name]
                )
            ),
            # Reset continuous loop counter on replan so the new strategy
            # gets a fresh set of attempts.
            "continuous_loop_count": 0,
            "coverage_target_name": new_target_name or prev_target_name,
            "coverage_target_api": new_target_api,
            "coverage_target_type": new_target_type,
            "selected_target_api": new_target_api or str(state.get("selected_target_api") or ""),
            "selected_target_runtime_viability": selected_runtime_viability or str(state.get("selected_target_runtime_viability") or ""),
            "coverage_seed_profile": new_seed_profile,
            "coverage_seed_families_suggested": list(seed_families_suggested),
            "coverage_seed_families_covered": [],
            "coverage_seed_families_missing": list(seed_families_suggested),
            "coverage_seed_quality": dict(state.get("coverage_seed_quality") or {}),
            "coverage_quality_flags": list(state.get("coverage_quality_flags") or []),
            "coverage_target_depth_score": new_depth_score,
            "coverage_target_depth_class": new_depth_class,
            "coverage_selection_bias_reason": new_selection_bias_reason,
            "coverage_target_score_breakdown": dict(
                selected_primary.get("score_breakdown")
                or selected_primary.get("target_score_breakdown")
                or {}
            ),
            "coverage_should_improve": coverage_should_improve,
            "coverage_round_budget_exhausted": coverage_round_budget_exhausted,
            "coverage_stop_reason": coverage_stop_reason,
            "coverage_replan_effective": coverage_replan_effective,
            "coverage_replan_reason": coverage_replan_reason,
            "replan_effective": replan_effective,
            "replan_stop_reason": replan_stop_reason,
            "restart_to_plan": False,
            "restart_to_plan_reason": "",
            "restart_to_plan_stage": "",
            "restart_to_plan_error_text": "",
            "restart_to_plan_report_path": "",
            "repair_mode": repair_mode,
            "repair_origin_stage": repair_origin_stage,
            "repair_error_kind": str(repair_snapshot.get("repair_error_kind") or ""),
            "repair_error_code": str(repair_snapshot.get("repair_error_code") or ""),
            "repair_signature": str(repair_snapshot.get("repair_signature") or ""),
            "repair_stdout_tail": str(repair_snapshot.get("repair_stdout_tail") or ""),
            "repair_stderr_tail": str(repair_snapshot.get("repair_stderr_tail") or ""),
            "repair_strategy_force_change": bool(repair_force_strategy_change),
            "repair_recent_attempts": (
                (repair_recent_attempts + [{
                    "step": str(state.get("last_step") or ""),
                    "origin": repair_origin_stage,
                    "error_kind": str(repair_snapshot.get("repair_error_kind") or ""),
                    "error_code": str(repair_snapshot.get("repair_error_code") or ""),
                    "signature": str(repair_snapshot.get("repair_signature") or ""),
                    "message": str(repair_snapshot.get("repair_error_text") or "")[:512],
                }])[-5:]
                if repair_mode
                else []
            ),
            "constraint_memory_count": constraint_memory_count,
            "constraint_memory_path": constraint_memory_path,
            "crash_signature_dedup_hit": bool(
                repair_snapshot.get("crash_signature_dedup_hit")
                or state.get("crash_signature_dedup_hit")
                or False
            ),
            "target_scoring_enabled": target_scoring_enabled,
            "target_score_breakdown_available": target_score_breakdown_available,
            "message": "planned",
        }
        out = _attach_prompt_render_status(out, issue=prompt_render_issue)
        out = _clear_error_markers_on_success(out)
        security_breakdown = dict(selected_primary.get("security_score_breakdown") or {})
        api_surface_exception = dict(selected_primary.get("api_surface_exception") or {})
        runner_up = selected_targets_doc[1] if len(selected_targets_doc) > 1 else {}
        selection_delta_vs_runner_up = (
            {
                "score_total": round(
                    float(selected_primary.get("score_total") or selected_primary.get("target_score") or 0.0)
                    - float(runner_up.get("score_total") or runner_up.get("target_score") or 0.0),
                    4,
                ),
                "execution_depth_bias": round(
                    float(selected_primary.get("execution_depth_bias") or 0.0)
                    - float(runner_up.get("execution_depth_bias") or 0.0),
                    4,
                ),
                "callback_penalty": round(
                    float(selected_primary.get("callback_penalty") or 0.0)
                    - float(runner_up.get("callback_penalty") or 0.0),
                    4,
                ),
                "priority": round(
                    float(selected_primary.get("effective_priority") or selected_primary.get("priority") or 0.0)
                    - float(runner_up.get("effective_priority") or runner_up.get("priority") or 0.0),
                    4,
                ),
            }
            if runner_up
            else {}
        )
        tie_break_reason_parts: list[str] = []
        if str(selected_primary.get("penalty_reason") or ""):
            tie_break_reason_parts.append(
                f"penalty={selected_primary.get('penalty_reason')}"
            )
        if float(selected_primary.get("effective_priority") or 0.0) != float(
            selected_primary.get("priority") or 0.0
        ):
            tie_break_reason_parts.append("effective_priority_adjusted")
        if float(selected_primary.get("execution_depth_bias") or 0.0) > 0.0:
            tie_break_reason_parts.append("deeper_path_bias")
        if float(selected_primary.get("callback_penalty") or 0.0) > 0.0:
            tie_break_reason_parts.append("shallow_callback_penalty")
        if runner_up:
            if float(selected_primary.get("execution_depth_bias") or 0.0) > float(
                runner_up.get("execution_depth_bias") or 0.0
            ):
                tie_break_reason = "higher_execution_depth_bias"
            elif float(selected_primary.get("callback_penalty") or 0.0) < float(
                runner_up.get("callback_penalty") or 0.0
            ):
                tie_break_reason = "lower_callback_penalty"
            elif tie_break_reason_parts:
                tie_break_reason = ",".join(tie_break_reason_parts)
            else:
                tie_break_reason = "higher_security_priority"
        else:
            tie_break_reason = (
                ",".join(tie_break_reason_parts)
                if tie_break_reason_parts
                else "risk_first_default"
            )
        choose_target_snapshot = {
            "kind": "choose_target",
            "selected_target": str(selected_primary.get("target") or new_target_name or ""),
            "selected_api": str(selected_primary.get("api") or new_target_api or ""),
            "score_total": float(selected_primary.get("score_total") or selected_primary.get("target_score") or 0.0),
            "score_breakdown": dict(
                selected_primary.get("score_breakdown")
                or selected_primary.get("target_score_breakdown")
                or {}
            ),
            "penalty_reason": str(
                selected_primary.get("penalty_reason")
                or selected_primary.get("target_score_penalty_reason")
                or ""
            ),
            "selected_targets_path": selected_targets_path,
            "degraded_reason": "" if selected_targets_doc else "selected_targets_missing_or_empty",
            "security_priority_mode": bool(security_priority_mode),
            "top_vuln_candidate": str(selected_primary.get("target") or new_target_name or ""),
            "security_score_breakdown": security_breakdown,
            "api_surface_exception_used": bool(api_surface_exception.get("used") or False),
            "tie_break_reason": tie_break_reason,
            "selection_delta_vs_runner_up": selection_delta_vs_runner_up,
        }
        out["latest_vuln_decision_snapshot"] = {
            "kind": "choose_target",
            "selected_target": str(choose_target_snapshot.get("selected_target") or ""),
            "selected_api": str(choose_target_snapshot.get("selected_api") or ""),
            "security_priority_mode": bool(security_priority_mode),
            "top_vuln_candidate": str(choose_target_snapshot.get("top_vuln_candidate") or ""),
            "security_score_breakdown": security_breakdown,
            "api_surface_exception_used": bool(choose_target_snapshot.get("api_surface_exception_used") or False),
            "tie_break_reason": str(choose_target_snapshot.get("tie_break_reason") or ""),
            "selection_delta_vs_runner_up": dict(choose_target_snapshot.get("selection_delta_vs_runner_up") or {}),
        }
        out = _record_decision_trace(
            out,
            stage="plan",
            tool="opencode",
            model=str(state.get("model") or ""),
            latency_ms=int(max(0.0, (time.perf_counter() - t0) * 1000.0)),
            error_kind="",
            error_code="",
            retry_count=1 if plan_retry_reason else 0,
            decision_snapshot=choose_target_snapshot,
        )
        _wf_log(cast(dict[str, Any], out), f"<- plan ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    except Exception as e:
        _write_stage_feedback(
            gen.repo_root,
            stage="plan",
            error_text=str(e),
            state=cast(dict[str, Any], state),
        )
        out = {**state, "last_step": "plan", "last_error": str(e), "message": "plan failed", "failed": True}
        out = _attach_prompt_render_status(out, issue=prompt_render_issue or str(e))
        _wf_log(cast(dict[str, Any], out), f"<- plan err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
