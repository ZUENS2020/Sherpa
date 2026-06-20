"""Carved from workflow_graph.py - '_node_improve_harness' LangGraph node."""

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
    _auto_stop_policy,
    _build_harness_feedback,
    _build_seed_feedback,
    _coverage_frontier_feedback_lines,
    _has_codex_key,
    _opencode_cli_retries,
    _remaining_time_budget_sec,
    _render_opencode_prompt_safe,
    _synthesize_activity_watch_paths,
    _synthesize_opencode_idle_timeout_sec,
)


def _node_improve_harness(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "improve-harness")
    if stop_now:
        return state
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> improve-harness")
    prompt_render_issue = ""
    try:
        def _target_names(values: Any) -> list[str]:
            names: list[str] = []
            for item in list(values or []):
                if isinstance(item, dict):
                    value = item.get("name") or item.get("target_name") or item.get("api") or item.get("target")
                else:
                    value = item
                name = str(value or "").strip()
                if name:
                    names.append(name)
            return names

        if not bool(state.get("coverage_should_improve")):
            out = {
                **state,
                "last_step": "improve-harness",
                "last_error": "",
                "message": "improve-harness skipped",
            }
            _wf_log(cast(dict[str, Any], out), f"<- improve-harness skip dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        cov_reason = str(state.get("coverage_improve_reason") or "").strip()
        target_name = str(state.get("coverage_target_name") or "").strip()
        target_api = str(state.get("coverage_target_api") or "").strip()
        seed_profile = str(state.get("coverage_seed_profile") or "").strip()
        selected_target_api = str(state.get("selected_target_api") or "").strip()
        depth_class = str(state.get("coverage_target_depth_class") or "").strip()
        depth_score = int(state.get("coverage_target_depth_score") or 0)
        selection_bias_reason = str(state.get("coverage_selection_bias_reason") or "").strip()
        replan_reason = str(state.get("coverage_replan_reason") or "").strip()
        replan_required = bool(state.get("coverage_replan_required"))
        seed_quality = dict(state.get("coverage_seed_quality") or {})
        quality_flags = list(state.get("coverage_quality_flags") or [])
        seed_families_suggested = list(state.get("coverage_seed_families_suggested") or [])
        seed_families_covered = list(state.get("coverage_seed_families_covered") or [])
        seed_families_missing = list(state.get("coverage_seed_families_missing") or [])
        seed_counts_raw = dict(state.get("coverage_seed_counts_raw") or {})
        seed_counts_filtered = dict(state.get("coverage_seed_counts_filtered") or {})
        seed_noise_rejected_count = int(state.get("coverage_seed_noise_rejected_count") or 0)
        seed_feedback = dict(state.get("coverage_seed_feedback") or _build_seed_feedback(cast(dict[str, Any], state)))
        harness_feedback = dict(state.get("coverage_harness_feedback") or _build_harness_feedback(cast(dict[str, Any], state)))
        quality_oracle = str(state.get("coverage_quality_oracle") or "ok")
        coverage_bottleneck_kind = str(state.get("coverage_bottleneck_kind") or "").strip() or "none"
        coverage_bottleneck_reason = str(state.get("coverage_bottleneck_reason") or "").strip()
        seed_first_repair = bool(
            coverage_bottleneck_kind == "seed_limited"
            or (
            seed_feedback.get("cold_start_failure")
            or float(seed_feedback.get("merge_retained_ratio_files") or 1.0) < 0.35
            or bool(seed_families_missing)
            )
        )
        improve_mode = str(state.get("coverage_improve_mode") or "").strip() or ("replan" if replan_required else "in_place")
        if replan_required:
            hint = (
                "Coverage-loop improvement task (replan target):\n"
                "- The current target has plateaued across consecutive rounds and coverage/features are no longer improving.\n"
                "- Re-evaluate `fuzz/targets.json` and select targets more likely to improve coverage or reveal bugs.\n"
                "- Use `fuzz/target_analysis.json` and `fuzz/antlr_plan_context.json` as primary evidence.\n"
                f"- Current target: {target_name or 'unknown'}\n"
                f"- Current target API: {target_api or selected_target_api or 'unknown'}\n"
                f"- Current seed_profile: {seed_profile or 'generic'}\n"
                f"- Current depth: {depth_class or 'unknown'} (score={depth_score})\n"
                f"- Current selection reason: {selection_bias_reason or 'n/a'}\n"
                f"- Replan reason: {replan_reason or cov_reason or 'coverage plateau'}\n"
                f"- Quality oracle: {quality_oracle}\n"
                f"- Coverage bottleneck: {coverage_bottleneck_kind} ({coverage_bottleneck_reason or 'n/a'})\n"
                f"- SeedFeedback: {json.dumps(seed_feedback, ensure_ascii=False)}\n"
                f"- HarnessFeedback: {json.dumps(harness_feedback, ensure_ascii=False)}\n"
                "- If the current target is shallow, prioritize medium/deep candidates and avoid helper/checksum/copy-style shallow sinks.\n"
                "- Prefer deeper entrypoints such as decode/inflate/deflate/parse/read/load/scan/archive/stream.\n"
            )
            # Append coverage-guided target selection context
            coverage_feedback_for_plan = str(state.get("coverage_feedback_for_plan") or "")
            exhausted_targets = _target_names(state.get("coverage_exhausted_targets"))
            if exhausted_targets:
                hint += f"- AVOID re-selecting these exhausted targets: {', '.join(exhausted_targets)}\n"
            if coverage_feedback_for_plan:
                hint += f"- Coverage history context:\n{coverage_feedback_for_plan}\n"
        else:
            hint = (
                "Coverage-loop improvement task (in-place target repair):\n"
                "- Modify only fuzzer-related files under `fuzz/`; do not edit upstream product source.\n"
                "- Do not modify `fuzz/targets.json` and do not add a second target in this mode.\n"
                "- Prioritize seed generation, dictionary, input modeling, call ordering, boundary cases, and corpus bootstrap.\n"
                "- Keep the scaffold buildable and runnable for the next build/run cycle.\n"
                f"- Current target: {target_name or 'unknown'}\n"
                f"- Current target API: {target_api or selected_target_api or 'unknown'}\n"
                f"- Current seed_profile: {seed_profile or 'generic'}\n"
                f"- Seed quality flags: {', '.join(quality_flags) if quality_flags else 'none'}\n"
                f"- Suggested seed families: {', '.join(seed_families_suggested) if seed_families_suggested else 'none'}\n"
                f"- Covered seed families: {', '.join(seed_families_covered) if seed_families_covered else 'none'}\n"
                f"- Missing seed families: {', '.join(seed_families_missing) if seed_families_missing else 'none'}\n"
                f"- Seed raw counts: {seed_counts_raw or {}}\n"
                f"- Seed filtered counts: {seed_counts_filtered or {}}\n"
                f"- Seed noise rejected: {seed_noise_rejected_count}\n"
                f"- Seed quality summary: {json.dumps(seed_quality, ensure_ascii=False) if seed_quality else '{}'}\n"
                f"- Quality oracle: {quality_oracle}\n"
                f"- Coverage bottleneck: {coverage_bottleneck_kind} ({coverage_bottleneck_reason or 'n/a'})\n"
                f"- Repair focus decision: {'seed-first' if seed_first_repair else 'harness-first'}\n"
                f"- SeedFeedback: {json.dumps(seed_feedback, ensure_ascii=False)}\n"
                f"- HarnessFeedback: {json.dumps(harness_feedback, ensure_ascii=False)}\n"
                f"- Current depth: {depth_class or 'unknown'} (score={depth_score})\n"
                f"- Current selection reason: {selection_bias_reason or 'n/a'}\n"
                f"- Diagnostic summary: {cov_reason}"
            )

        # Append fine-grained source coverage feedback if available
        frontier_summary = dict(state.get("coverage_frontier_summary") or {})
        frontier_lines = _coverage_frontier_feedback_lines(frontier_summary)
        if frontier_lines:
            hint += "\n--- Per-Input Frontier ---\n"
            hint += "\n".join(frontier_lines) + "\n"
            frontier_path = str(state.get("coverage_frontier_path") or "").strip()
            if frontier_path:
                hint += f"- Full frontier summary: {frontier_path}\n"

        source_report = dict(state.get("coverage_source_report") or {})
        uncovered_fns = list(state.get("coverage_uncovered_functions") or [])
        if source_report or uncovered_fns:
            hint += "\n--- Source Coverage Report ---\n"
            if source_report:
                hint += (
                    f"- Function coverage: {source_report.get('covered_functions', '?')}"
                    f"/{source_report.get('total_functions', '?')} "
                    f"({source_report.get('coverage_pct', '?')}%)\n"
                )
            if uncovered_fns:
                hint += f"- Top uncovered functions (focus improvement here):\n"
                for fn in uncovered_fns[:10]:
                    hint += f"  * {fn}\n"
            hint += "- Consider adding API calls or input patterns that exercise uncovered functions.\n"
            # Also point to the full report file
            report_path = source_report.get("report_path")
            if report_path:
                hint += f"- Full coverage report: {report_path}\n"
        run_feedback_summary = dict(state.get("coverage_run_feedback_summary") or {})
        if run_feedback_summary:
            hint += "\n--- Run Feedback Summary ---\n"
            for item in list(run_feedback_summary.get("top_function_gaps") or [])[:5]:
                if not isinstance(item, dict):
                    continue
                hint += (
                    f"- function_gap: {str(item.get('name') or '').strip()} "
                    f"({str(item.get('file') or '').strip()}:{int(item.get('line') or 0)}) "
                    f"kind={str(item.get('kind') or 'unknown')}, "
                    f"ratio={float(item.get('region_coverage_ratio') or 0.0):.2f}\n"
                )
            for item in list(run_feedback_summary.get("top_path_frontiers") or [])[:3]:
                if not isinstance(item, dict):
                    continue
                hint += (
                    f"- path_frontier: {str(item.get('input_relpath') or '').strip()} "
                    f"score={float(item.get('frontier_score') or 0.0):.2f}, "
                    f"functions={int(item.get('covered_function_count') or 0)}, "
                    f"regions={int(item.get('covered_region_count') or 0)}\n"
                )
            run_feedback_path = str(state.get("coverage_run_feedback_path") or "").strip()
            if run_feedback_path:
                hint += f"- Full run feedback: {run_feedback_path}\n"

        opencode_applied = False
        if improve_mode == "in_place":
            if not _has_codex_key():
                out = {
                    **state,
                    "last_step": "improve-harness",
                    "last_error": "Missing OPENAI_API_KEY for improve-harness in-place repair",
                    "message": "improve-harness failed",
                }
                out = _attach_prompt_render_status(out)
                _wf_log(cast(dict[str, Any], out), f"<- improve-harness err=missing-key dt={_fmt_dt(time.perf_counter()-t0)}")
                return out
            prompt, render_issue = _render_opencode_prompt_safe(
                "improve_harness_in_place_with_hint",
                fallback_name="plan_repair_coverage_with_hint",
                hint=hint,
                fallback_hint=hint,
            )
            if render_issue:
                prompt_render_issue = str(render_issue)
                _wf_log(cast(dict[str, Any], state), f"improve-harness: prompt render degraded -> {render_issue}")
            ctx_parts: list[str] = []
            try:
                exec_plan = gen.repo_root / "fuzz" / "execution_plan.json"
                harness_index = gen.repo_root / "fuzz" / "harness_index.json"
                coverage_report = gen.repo_root / "fuzz" / "coverage_report.txt"
                if exec_plan.is_file():
                    ctx_parts.append("=== fuzz/execution_plan.json ===\n" + exec_plan.read_text(encoding="utf-8", errors="replace"))
                if harness_index.is_file():
                    ctx_parts.append("=== fuzz/harness_index.json ===\n" + harness_index.read_text(encoding="utf-8", errors="replace"))
                if coverage_report.is_file():
                    ctx_parts.append("=== fuzz/coverage_report.txt ===\n" + coverage_report.read_text(encoding="utf-8", errors="replace")[:4000])
            except Exception:
                pass
            gen.patcher.run_codex_command(
                prompt,
                additional_context=("\n\n".join(ctx_parts) if ctx_parts else None),
                stage_skill="improve_harness_in_place",
                timeout=_remaining_time_budget_sec(state),
                max_attempts=1,
                max_cli_retries=_opencode_cli_retries(),
                idle_timeout_override=_synthesize_opencode_idle_timeout_sec(),
                activity_watch_paths=_synthesize_activity_watch_paths(),
            )
            opencode_applied = True
        out = {
            **state,
            "last_step": "improve-harness",
            "last_error": "",
            "codex_hint": hint,
            "coverage_improve_mode": improve_mode,
            "repair_mode": bool(replan_required),
            "repair_origin_stage": "coverage" if replan_required else str(state.get("repair_origin_stage") or ""),
            "repair_error_kind": "coverage_plateau" if replan_required else str(state.get("repair_error_kind") or ""),
            "repair_error_code": "coverage_replan_required" if replan_required else str(state.get("repair_error_code") or ""),
            "repair_signature": (
                f"coverage:{target_name or target_api or 'unknown'}:{int(state.get('coverage_plateau_streak') or 0)}"
                if replan_required
                else str(state.get("repair_signature") or "")
            ),
            "repair_error_digest": (
                {
                    "origin": "coverage",
                    "improve_mode": improve_mode,
                    "replan_required": True,
                    "target_name": target_name or "",
                    "target_api": target_api or selected_target_api or "",
                    "seed_profile": seed_profile or "",
                    "depth_class": depth_class or "",
                    "depth_score": depth_score,
                    "selection_bias_reason": selection_bias_reason or "",
                    "replan_reason": replan_reason or cov_reason or "",
                    "quality_flags": quality_flags,
                    "seed_families_suggested": seed_families_suggested,
                    "seed_families_covered": seed_families_covered,
                    "seed_families_missing": seed_families_missing,
                    "seed_counts_raw": seed_counts_raw,
                    "seed_counts_filtered": seed_counts_filtered,
                    "seed_noise_rejected_count": seed_noise_rejected_count,
                    "seed_quality": seed_quality,
                    "quality_oracle": quality_oracle,
                    "coverage_bottleneck_kind": coverage_bottleneck_kind,
                    "coverage_bottleneck_reason": coverage_bottleneck_reason,
                    "seed_feedback": seed_feedback,
                    "harness_feedback": harness_feedback,
                }
                if replan_required
                else dict(state.get("repair_error_digest") or {})
            ),
            "message": "improve-harness in-place edits applied" if opencode_applied else "improve-harness prepared plan hint",
            "auto_stop_policy": _auto_stop_policy(),
            "auto_stop_blocked_reason": str(state.get("auto_stop_blocked_reason") or ""),
            "continuous_loop_count": int(state.get("continuous_loop_count") or 0),
        }
        out = _attach_prompt_render_status(out, issue=prompt_render_issue)
        if out["auto_stop_policy"] == "hard_fail_only":
            replan_ineffective = str(out.get("coverage_improve_mode") or "").strip() == "replan" and not bool(
                out.get("coverage_replan_effective", True)
            )
            budget_exhausted = bool(out.get("coverage_round_budget_exhausted"))
            if replan_ineffective:
                out["auto_stop_blocked_reason"] = "coverage_replan_ineffective"
                out["continuous_loop_count"] = int(out.get("continuous_loop_count") or 0) + 1
            elif budget_exhausted:
                out["auto_stop_blocked_reason"] = "coverage_round_budget_exhausted"
                out["continuous_loop_count"] = int(out.get("continuous_loop_count") or 0) + 1
        _wf_log(cast(dict[str, Any], out), f"<- improve-harness ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    except Exception as e:
        out = {**state, "last_step": "improve-harness", "last_error": str(e), "message": "improve-harness failed"}
        out = _attach_prompt_render_status(out, issue=prompt_render_issue or str(e))
        _wf_log(cast(dict[str, Any], out), f"<- improve-harness err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
