"""Carved from workflow_graph.py - '_node_synthesize' LangGraph node."""

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
    _analyze_harness_target_alignment,
    _attach_prompt_render_status,
    _build_fix_harness_crash_context,
    _build_repair_snapshot,
    _build_template_cache_path,
    _clear_error_markers_on_success,
    _collect_feedback_for_group,
    _coverage_attack_hint_feedback_lines,
    _find_static_lib,
    _has_codex_key,
    _infer_repair_origin_stage,
    _load_repo_understanding_doc,
    _load_security_evidence_list,
    _load_selected_targets_doc,
    _load_targets_doc,
    _load_vuln_candidate_inventory,
    _opencode_cli_retries,
    _readme_drift_status,
    _remaining_time_budget_sec,
    _render_opencode_prompt_safe,
    _repo_understanding_is_complete,
    _repo_understanding_path,
    _restore_cached_build_template_if_missing,
    _sync_execution_plan_doc_from_selected_targets,
    _synthesize_activity_watch_paths,
    _synthesize_opencode_attempts,
    _synthesize_opencode_idle_timeout_sec,
    _time_budget_exceeded_state,
    _validate_build_repair_contract,
    _validate_execution_plan_harness_consistency,
    _validate_harness_source_contract,
    _workflow_target_state_from_execution_plan,
    _write_build_strategy_doc,
    _write_harness_index_doc,
    _write_observed_target_doc,
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


def _node_synthesize(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "synthesize")
    if stop_now:
        return state
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> synthesize")
    hint = (state.get("codex_hint") or "").strip()
    prompt_render_issue = ""

    def _synthesize_allowed_edit_paths() -> tuple[str, ...]:
        return ("fuzz/**", "done")

    repair_mode = bool(state.get("repair_mode") or False)
    repair_origin_stage = str(state.get("repair_origin_stage") or "").strip().lower()
    if repair_origin_stage not in {"build", "crash", "coverage", "fix-harness"}:
        repair_origin_stage = _infer_repair_origin_stage(cast(dict[str, Any], state))
    antlr_context_path = str(state.get("antlr_context_path") or "").strip()
    antlr_context_summary = str(state.get("antlr_context_summary") or "").strip()
    target_analysis_path = str(state.get("target_analysis_path") or "").strip()
    target_analysis_summary = str(state.get("target_analysis_summary") or "").strip()
    analysis_context_path = str(state.get("analysis_context_path") or "").strip()
    analysis_evidence_count = int(state.get("analysis_evidence_count") or 0)
    selected_targets_path = str(state.get("selected_targets_path") or "").strip()
    selected_target_api = str(state.get("selected_target_api") or "").strip()
    selected_target_runtime_viability = str(state.get("selected_target_runtime_viability") or "").strip().lower()
    selected_target_doc = _load_selected_targets_doc(gen.repo_root)
    # Overwrite targets.json with only selected targets so the synthesize
    # agent generates harnesses for the correct execution targets.
    if selected_target_doc:
        if _vuln_hunting_enabled():
            _must_run = [t for t in selected_target_doc if bool(t.get("must_run") or False)]
            _filtered = _must_run if _must_run else selected_target_doc
        else:
            _filtered = selected_target_doc
        _filtered_targets_path = gen.repo_root / "fuzz" / "targets.json"
        _filtered_targets_path.parent.mkdir(parents=True, exist_ok=True)
        _filtered_targets_path.write_text(
            json.dumps(_filtered, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    # Sync execution_plan.json from selected_targets.json before AI runs
    # so the agent sees a consistent contract. Also rebuild selected_targets
    # from targets.json if the file is missing.
    if selected_target_doc:
        _sync_execution_plan_doc_from_selected_targets(gen.repo_root)
    else:
        _targets_doc = _load_targets_doc(gen.repo_root)
        if _targets_doc:
            selected_target_doc = _targets_doc
            _write_selected_targets_doc(gen.repo_root)
            _sync_execution_plan_doc_from_selected_targets(gen.repo_root)
    selected_target_name = ""
    if selected_target_doc:
        selected_primary = selected_target_doc[0]
        selected_target_name = str(selected_primary.get("target_name") or selected_primary.get("name") or "").strip()
    if antlr_context_summary and "antlr_plan_context.json" not in hint:
        hint = (
            (hint.strip() + "\n\n") if hint.strip() else ""
        ) + (
            "Use grammar-aware context from `fuzz/antlr_plan_context.json` while generating harness/build glue.\n"
            f"{antlr_context_summary}"
        )
    if target_analysis_summary and "target_analysis.json" not in hint:
        hint = (
            (hint.strip() + "\n\n") if hint.strip() else ""
        ) + (
            "Use `fuzz/target_analysis.json` to preserve the selected target's seed_profile and risk signals while generating harness/build glue.\n"
            f"{target_analysis_summary}"
        )
    if analysis_context_path and "analysis_context.json" not in hint:
        hint = (
            (hint.strip() + "\n\n") if hint.strip() else ""
        ) + (
            "Use `fuzz/analysis_context.json` as the canonical evidence index for API/callgraph/consumer pattern decisions.\n"
            f"Available evidence_count={analysis_evidence_count}."
        )
    # Inject vulnerability-directed harness guidance from security evidence
    if _vuln_hunting_enabled() and analysis_context_path:
        security_evidence, security_issue = _load_security_evidence_list(
            gen.repo_root,
            analysis_context_path,
        )
        vuln_candidates, vuln_candidates_issue = _load_vuln_candidate_inventory(
            gen.repo_root,
            analysis_context_path,
        )
        if security_issue:
            issue_text = str(security_issue or "").strip()
            if issue_text:
                if not prompt_render_issue:
                    prompt_render_issue = issue_text
                elif issue_text not in prompt_render_issue:
                    prompt_render_issue = f"{prompt_render_issue}; {issue_text}"
            _wf_log(cast(dict[str, Any], state), f"synthesize: security evidence degraded -> {security_issue}")
        if vuln_candidates_issue:
            issue_text = str(vuln_candidates_issue or "").strip()
            if issue_text:
                if not prompt_render_issue:
                    prompt_render_issue = issue_text
                elif issue_text not in prompt_render_issue:
                    prompt_render_issue = f"{prompt_render_issue}; {issue_text}"
            _wf_log(cast(dict[str, Any], state), f"synthesize: vuln candidates degraded -> {vuln_candidates_issue}")
        high_conf: list[dict[str, Any]] = []
        for entry in security_evidence:
            try:
                confidence = float(entry.get("confidence") or 0.0)
            except Exception:
                confidence = 0.0
            if confidence >= 0.5:
                high_conf.append(entry)
        high_priority_candidates: list[dict[str, Any]] = []
        for candidate in vuln_candidates:
            try:
                priority = float(candidate.get("priority") or 0.0)
            except Exception:
                priority = 0.0
            if priority <= 0.0:
                continue
            high_priority_candidates.append(dict(candidate))
        high_priority_candidates.sort(
            key=lambda item: (
                -float(item.get("priority") or 0.0),
                -float(item.get("vuln_likelihood") or 0.0),
                str(item.get("candidate_id") or ""),
            )
        )
        if high_conf or high_priority_candidates:
            vuln_hint_lines = [
                "\n## Vulnerability-Directed Harness Guidance",
                "Prioritize exercising these high-risk code paths and candidate triggers:",
            ]
            for candidate in high_priority_candidates[:5]:
                attack_hint = dict(candidate.get("attack_hint") or {})
                candidate_id = str(candidate.get("candidate_id") or "candidate").strip() or "candidate"
                target_api = str(candidate.get("target_api") or candidate.get("api") or "unknown_api").strip() or "unknown_api"
                signal_type = str(candidate.get("signal_type") or "unknown_signal").strip() or "unknown_signal"
                priority = float(candidate.get("priority") or 0.0)
                key_code_path = [str(x).strip() for x in list(attack_hint.get("key_code_path") or []) if str(x).strip()]
                boundary_values = [str(x).strip() for x in list(attack_hint.get("boundary_values") or []) if str(x).strip()]
                trigger_condition = str(attack_hint.get("trigger_condition") or "").strip()
                vuln_category = str(attack_hint.get("vuln_category") or signal_type).strip() or signal_type
                sanitizer_hint = str(attack_hint.get("sanitizer_hint") or "address").strip() or "address"
                evidence_refs = [str(item.get("evidence_id") or "").strip() for item in list(candidate.get("evidence") or []) if str(item.get("evidence_id") or "").strip()]
                location = str(candidate.get("target_file") or candidate.get("file") or "").strip()
                vuln_hint_lines.append(
                    f"- {candidate_id}: api={target_api}, signal={signal_type}, category={vuln_category}, priority={priority:.2f}"
                    + (f" [{location}]" if location else "")
                )
                if trigger_condition:
                    vuln_hint_lines.append(f"  trigger_condition: {trigger_condition}")
                if key_code_path:
                    vuln_hint_lines.append(f"  key_code_path: {' -> '.join(key_code_path[:6])}")
                if boundary_values:
                    vuln_hint_lines.append(f"  boundary_values: {', '.join(boundary_values[:6])}")
                vuln_hint_lines.append(f"  sanitizer_hint: {sanitizer_hint}")
                if evidence_refs:
                    vuln_hint_lines.append(f"  evidence_refs: {', '.join(evidence_refs[:6])}")
            for entry in high_conf[:8]:
                signal_id = str(entry.get("signal_id") or "unknown_signal").strip() or "unknown_signal"
                summary = str(entry.get("summary") or "n/a").strip() or "n/a"
                source_path = str(entry.get("source_path") or "").strip()
                source_line = int(entry.get("line") or 0) if str(entry.get("line") or "").strip() else 0
                location = source_path
                if source_line > 0:
                    location = f"{source_path}:{source_line}" if source_path else f"line:{source_line}"
                suffix = f" [{location}]" if location else ""
                vuln_hint_lines.append(f"- {signal_id}: {summary}{suffix}")
            vuln_hint_lines.extend(
                [
                    "Design the harness to:",
                    "- Feed attacker-controlled data through the listed key_code_path sequences",
                    "- Materialize the listed boundary_values in seed inputs and parser state",
                    "- Prefer target-specific sanitizer guidance when choosing crash-oriented execution assumptions",
                    "- Test error handling paths (corrupt headers, truncated input, invalid checksums)",
                ]
            )
            hint = (hint + "\n" + "\n".join(vuln_hint_lines)).strip()
    if selected_targets_path:
        selected_target_soft_hint = (
            "Use `fuzz/selected_targets.json` as a preferred target plan, not a hard stop.\n"
            f"Prefer the selected target `{selected_target_api or selected_target_name or 'unknown'}` if it is runtime-executable.\n"
            "If the selected target is compile-time-only, detail-only, constexpr-only, or otherwise not a viable runtime fuzz entrypoint,\n"
            "you may choose a nearby runtime-executable replacement target.\n"
            "When you do that, you MUST record in `fuzz/README.md`:\n"
            "- Selected target: <original target>\n"
            "- Final target: <observed runtime target>\n"
            "- Technical reason: <why the original target is not the best runtime entrypoint>\n"
            "- Relation: <how the final target relates to the original target>\n"
            "Prefer public/runtime parser APIs over generic wrappers when a direct runtime target exists."
        )
        if "selected_targets.json" not in hint:
            hint = ((hint.strip() + "\n\n") if hint.strip() else "") + selected_target_soft_hint
    planning_feedback = _collect_feedback_for_group(gen.repo_root, "planning_synth", limit=3)
    if planning_feedback:
        feedback_hint = (
            "Recent planning/synthesis failures from previous attempts "
            "(use these to avoid repeating the same mistakes):\n"
            + planning_feedback
        )
        hint = (hint + "\n\n" + feedback_hint).strip() if hint else feedback_hint
    if repair_mode:
        repair_snapshot = _build_repair_snapshot(cast(dict[str, Any], state))
        repair_error_kind = str(repair_snapshot.get("repair_error_kind") or "generic_failure").strip()
        repair_error_code = str(repair_snapshot.get("repair_error_code") or "").strip()
        repair_signature = str(repair_snapshot.get("repair_signature") or "").strip()
        repair_stderr_tail = str(repair_snapshot.get("repair_stderr_tail") or "").strip()
        repair_stdout_tail = str(repair_snapshot.get("repair_stdout_tail") or "").strip()
        repair_recent_attempts = list(repair_snapshot.get("repair_recent_attempts") or [])
        repair_attempt_index = int(repair_snapshot.get("repair_attempt_index") or 0)
        repair_force_strategy_change = bool(repair_snapshot.get("repair_strategy_force_change") or False)
        repair_error_digest = dict(repair_snapshot.get("repair_error_digest") or {})
        constraint_memory_entry = dict(repair_snapshot.get("constraint_memory_entry") or {})
        constraint_memory_count = int(repair_snapshot.get("constraint_memory_count") or 0)
        constraint_memory_path = str(repair_snapshot.get("constraint_memory_path") or "")
        repair_lines: list[str] = [
            "Repair mode context (consume this before editing):",
            f"- repair_origin_stage: {repair_origin_stage}",
            f"- repair_error_kind: {repair_error_kind or 'generic_failure'}",
            f"- repair_error_code: {repair_error_code or 'n/a'}",
            f"- repair_signature: {repair_signature or 'n/a'}",
            f"- repair_attempt_index: {repair_attempt_index}",
        ]
        if repair_force_strategy_change:
            repair_lines.append(
                "Strategy gate: repeated failure signature detected. This round must produce a materially different "
                "repair strategy (target selection, harness API path, or build/link design)."
            )
        if repair_error_digest:
            repair_lines.append(
                "=== repair error digest ===\n"
                + json.dumps(repair_error_digest, ensure_ascii=False, indent=2)
            )
        if repair_stderr_tail:
            repair_lines.append("=== repair stderr tail ===\n" + "\n".join(repair_stderr_tail.splitlines()[-200:]))
        if repair_stdout_tail:
            repair_lines.append("=== repair stdout tail ===\n" + "\n".join(repair_stdout_tail.splitlines()[-120:]))
        if repair_error_code == "non_public_api_usage":
            repair_lines.extend(
                [
                    "Repair priority: resolve non-public API usage in harness first.",
                    "Replace internal/private symbols (for example `detail::`, `internal::`, `impl::`) with public/stable APIs.",
                    "When no public alternative exists, add `api_surface_exception` in `fuzz/repo_understanding.json` with non-empty `reason` and `evidence`.",
                ]
            )
        if repair_recent_attempts:
            repair_lines.append(
                "=== repair recent attempts ===\n"
                + json.dumps(repair_recent_attempts[-5:], ensure_ascii=False, indent=2)
            )
        if constraint_memory_entry:
            repair_lines.append(
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
        repair_hint = "\n\n".join(part for part in repair_lines if part.strip())
        hint = (hint + "\n\n" + repair_hint).strip() if hint else repair_hint
    seed_feedback = dict(state.get("coverage_seed_feedback") or {})
    harness_feedback = dict(state.get("coverage_harness_feedback") or {})
    quality_oracle = str(state.get("coverage_quality_oracle") or "").strip()
    if seed_feedback or harness_feedback or quality_oracle:
        feedback_lines: list[str] = [
            "Coverage feedback signals for scaffold synthesis:",
            "- Consume SeedFeedback/HarnessFeedback first, then decide whether to change seed modeling, harness logic, or target mapping.",
        ]
        if quality_oracle:
            feedback_lines.append(f"- quality_oracle: {quality_oracle}")
        feedback_lines.extend(_coverage_attack_hint_feedback_lines(seed_feedback))
        if seed_feedback:
            feedback_lines.append("=== SeedFeedback ===\n" + json.dumps(seed_feedback, ensure_ascii=False, indent=2))
        if harness_feedback:
            feedback_lines.append("=== HarnessFeedback ===\n" + json.dumps(harness_feedback, ensure_ascii=False, indent=2))
        feedback_hint = "\n\n".join(feedback_lines)
        hint = (hint + "\n\n" + feedback_hint).strip() if hint else feedback_hint
    restored_from_cache = False
    try:
        restored_from_cache = _restore_cached_build_template_if_missing(gen.repo_root)
    except Exception:
        restored_from_cache = False
    if restored_from_cache:
        _wf_log(cast(dict[str, Any], state), "synthesize: restored cached build.py/build_strategy template")

    def _remember_prompt_render_issue(issue: str) -> None:
        nonlocal prompt_render_issue
        issue_text = str(issue or "").strip()
        if not issue_text:
            return
        if not prompt_render_issue:
            prompt_render_issue = issue_text
            return
        if issue_text not in prompt_render_issue:
            prompt_render_issue = f"{prompt_render_issue}; {issue_text}"

    def _synthesis_output_status() -> dict[str, Any]:
        fuzz_dir = gen.repo_root / "fuzz"
        harnesses: list[str] = []
        has_build_script = False
        has_readme = False
        has_repo_understanding = False
        has_build_strategy = False
        scan_errors: list[str] = []
        try:
            candidates = list(fuzz_dir.rglob("*"))
        except Exception as e:
            candidates = []
            scan_errors.append(f"rglob_failed:{e}")
        for p in candidates:
            try:
                if not p.is_file():
                    continue
                rel = p.relative_to(fuzz_dir)
                rel_posix = rel.as_posix()
                if rel_posix.startswith("out/") or rel_posix.startswith("corpus/"):
                    continue
                if p.suffix.lower() in {".c", ".cc", ".cpp", ".cxx", ".java"}:
                    harnesses.append(rel_posix)
                if rel_posix in {"build.py", "build.sh"}:
                    has_build_script = True
                if rel_posix == "README.md":
                    has_readme = True
                if rel_posix == "repo_understanding.json":
                    has_repo_understanding = True
                if rel_posix == "build_strategy.json":
                    has_build_strategy = True
            except Exception as e:
                scan_errors.append(f"scan_item_failed:{p}:{e}")
        return {
            "harnesses": harnesses,
            "has_harness": bool(harnesses),
            "has_build_script": has_build_script,
            "has_readme": has_readme,
            "has_repo_understanding": has_repo_understanding,
            "has_build_strategy": has_build_strategy,
            # build_strategy.json is generated deterministically later by _write_build_strategy_doc.
            "has_required": bool(harnesses) and has_build_script and has_readme and has_repo_understanding,
            "has_partial": bool(harnesses) or has_build_script or has_readme or has_repo_understanding or has_build_strategy,
            "scan_errors": scan_errors[:8],
            "scan_error_count": len(scan_errors),
        }

    def _has_min_synthesis_outputs() -> bool:
        return bool(_synthesis_output_status().get("has_harness"))

    def _has_required_synthesis_outputs() -> bool:
        return bool(_synthesis_output_status().get("has_required"))

    def _missing_synthesis_items() -> list[str]:
        status = _synthesis_output_status()
        missing: list[str] = []
        if not status.get("has_harness"):
            missing.append("one harness source file under fuzz/ (`*_fuzz.cc`, `*.c`, `*.cpp`, or `*.java`)")
        if not status.get("has_build_script"):
            missing.append("`fuzz/build.py` or `fuzz/build.sh`")
        if not status.get("has_readme"):
            missing.append("`fuzz/README.md`")
        if not status.get("has_repo_understanding"):
            missing.append("`fuzz/repo_understanding.json`")
        return missing

    def _synthesis_grace_wait(max_sec: int) -> bool:
        if max_sec <= 0:
            return _has_min_synthesis_outputs()
        deadline = time.time() + max_sec
        while time.time() < deadline:
            if _has_min_synthesis_outputs():
                return True
            time.sleep(1)
        return _has_min_synthesis_outputs()

    def _required_synthesis_grace_wait(max_sec: int) -> bool:
        if max_sec <= 0:
            return _has_required_synthesis_outputs()
        deadline = time.time() + max_sec
        while time.time() < deadline:
            if _has_required_synthesis_outputs():
                return True
            time.sleep(1)
        return _has_required_synthesis_outputs()

    def _completion_context() -> str:
        plan = gen.repo_root / "fuzz" / "PLAN.md"
        targets = gen.repo_root / "fuzz" / "targets.json"
        parts: list[str] = []
        try:
            if plan.is_file():
                parts.append("=== fuzz/PLAN.md ===\n" + plan.read_text(encoding="utf-8", errors="replace"))
            if targets.is_file():
                parts.append("=== fuzz/targets.json ===\n" + targets.read_text(encoding="utf-8", errors="replace"))
            if antlr_context_path:
                antlr_path_obj = Path(antlr_context_path)
                if not antlr_path_obj.is_absolute():
                    antlr_path_obj = gen.repo_root / antlr_path_obj
                if antlr_path_obj.is_file():
                    parts.append(
                        "=== fuzz/antlr_plan_context.json ===\n"
                        + antlr_path_obj.read_text(encoding="utf-8", errors="replace")
                    )
            if target_analysis_path:
                analysis_path_obj = Path(target_analysis_path)
                if not analysis_path_obj.is_absolute():
                    analysis_path_obj = gen.repo_root / analysis_path_obj
                if analysis_path_obj.is_file():
                    parts.append(
                        "=== fuzz/target_analysis.json ===\n"
                        + analysis_path_obj.read_text(encoding="utf-8", errors="replace")
                    )
            if selected_targets_path:
                selected_path_obj = Path(selected_targets_path)
                if not selected_path_obj.is_absolute():
                    selected_path_obj = gen.repo_root / selected_path_obj
                if selected_path_obj.is_file():
                    parts.append(
                        "=== fuzz/selected_targets.json ===\n"
                        + selected_path_obj.read_text(encoding="utf-8", errors="replace")
                    )
            status = _synthesis_output_status()
            if status.get("harnesses"):
                parts.append("=== existing harness files ===\n" + "\n".join(str(x) for x in status.get("harnesses") or []))
            build_py = gen.repo_root / "fuzz" / "build.py"
            if build_py.is_file():
                parts.append("=== existing fuzz/build.py ===\n" + build_py.read_text(encoding="utf-8", errors="replace"))
            build_strategy = gen.repo_root / "fuzz" / "build_strategy.json"
            if build_strategy.is_file():
                parts.append("=== existing fuzz/build_strategy.json ===\n" + build_strategy.read_text(encoding="utf-8", errors="replace"))
            build_runtime_facts = gen.repo_root / "fuzz" / "build_runtime_facts.json"
            if build_runtime_facts.is_file():
                parts.append("=== existing fuzz/build_runtime_facts.json ===\n" + build_runtime_facts.read_text(encoding="utf-8", errors="replace"))
            build_cache = _build_template_cache_path(gen.repo_root)
            if build_cache.is_file():
                parts.append("=== existing fuzz/build_template_cache.json ===\n" + build_cache.read_text(encoding="utf-8", errors="replace"))
            build_sh = gen.repo_root / "fuzz" / "build.sh"
            if build_sh.is_file():
                parts.append("=== existing fuzz/build.sh ===\n" + build_sh.read_text(encoding="utf-8", errors="replace"))
            readme = gen.repo_root / "fuzz" / "README.md"
            if readme.is_file():
                parts.append("=== existing fuzz/README.md ===\n" + readme.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass
        return "\n\n".join(parts)

    def _run_post_synthesize_build_validation() -> None:
        raw_enabled = (os.environ.get("SHERPA_SYNTH_BUILD_VALIDATE") or "1").strip().lower()
        if raw_enabled in {"0", "false", "no", "off"}:
            return
        fuzz_dir = gen.repo_root / "fuzz"
        build_py = fuzz_dir / "build.py"
        if not build_py.is_file() or not hasattr(gen, "_run_cmd"):
            return
        remaining = _remaining_time_budget_sec(state, min_timeout=0)
        if remaining <= 0:
            return
        raw_timeout = (os.environ.get("SHERPA_SYNTH_BUILD_VALIDATE_TIMEOUT_SEC") or "1800").strip()
        try:
            cfg_timeout = max(10, min(int(raw_timeout), 3600))
        except Exception:
            cfg_timeout = 1800
        timeout = min(remaining, cfg_timeout)
        cmd = [gen._python_runner(), "build.py"] if hasattr(gen, "_python_runner") else [shutil.which("python3") or "python", "build.py"]
        build_env = os.environ.copy()
        include_root = str(gen.repo_root)
        for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
            prev = build_env.get(key, "").strip()
            build_env[key] = f"{include_root}:{prev}" if prev else include_root
        rc, out, err = gen._run_cmd(list(cmd), cwd=fuzz_dir, env=build_env, timeout=timeout)
        bins = gen._discover_fuzz_binaries() if rc == 0 else []
        if rc == 0 and bins:
            _wf_log(cast(dict[str, Any], state), "synthesize: build scaffold validation passed")
            return
        diag = ((out or "") + "\n" + (err or "")).lower()
        path_error_signals = (
            "could not find",
            "cannot find -l",
            "no such file or directory",
            "undefined reference",
        )
        if not any(sig in diag for sig in path_error_signals):
            return
        static_probe = _find_static_lib(gen.repo_root, "libarchive*.a")
        static_probe_txt = str(static_probe.relative_to(gen.repo_root)) if isinstance(static_probe, Path) else "(none found)"
        prompt = textwrap.dedent(
            """
            Repair `fuzz/build.py` for static library artifact discovery.
            The scaffold validation build failed, and this looks like a path/link discovery issue.

            Required edits:
            1. Add a reusable helper:
               def find_static_lib(repo_root, lib_name_pattern):
                   ...
            2. Include candidate constants in build.py:
               STATIC_LIB_NAMES and SEARCH_PATHS.
            3. Resolve library artifacts via multiple candidates + recursive glob fallback.
            4. Verify the selected artifact path exists before final link command.
            5. Keep non-root compatibility (no install-to-system-dir flow).

            Do not run commands. Only edit `fuzz/build.py` and `fuzz/build_strategy.json` if needed.
            Write the path string `fuzz/build.py` as the sole text of `./done` (run `echo 'fuzz/build.py' > ./done`; do **not** copy the file's contents).
            """
        ).strip()
        context = (
            "=== synth-validate build stdout (tail) ===\n"
            + "\n".join((out or "").splitlines()[-120:])
            + "\n\n=== synth-validate build stderr (tail) ===\n"
            + "\n".join((err or "").splitlines()[-120:])
            + "\n\n=== static-lib-probe ===\n"
            + static_probe_txt
            + "\n\n=== existing fuzz/build.py ===\n"
            + build_py.read_text(encoding="utf-8", errors="replace")
        )
        gen.patcher.run_codex_command(
            prompt,
            additional_context=context,
            stage_skill="synthesize_complete_scaffold",
            timeout=min(remaining, 300),
            max_attempts=_synthesize_opencode_attempts(),
            max_cli_retries=_opencode_cli_retries(),
            allowed_edit_paths=_synthesize_allowed_edit_paths(),
        )
        _wf_log(cast(dict[str, Any], state), "synthesize: applied post-validation build.py repair for path/link issue")

    def _run_synthesize_completion(timeout: int) -> None:
        missing_items = "\n".join(f"- {item}" for item in _missing_synthesis_items()) or "- no missing items detected"
        completion_hint = f"Complete required fuzz scaffold artifacts.\nMissing items:\n{missing_items}"
        prompt, render_issue = _render_opencode_prompt_safe(
            "synthesize_complete_scaffold",
            fallback_name="synthesize_with_hint",
            missing_items=missing_items,
            hint=completion_hint,
            fallback_hint=completion_hint,
        )
        if render_issue:
            _remember_prompt_render_issue(render_issue)
            _wf_log(cast(dict[str, Any], state), f"synthesize completion: prompt render degraded -> {render_issue}")
        gen.patcher.run_codex_command(
            prompt,
            additional_context=_completion_context() or None,
            stage_skill="synthesize_complete_scaffold",
            timeout=timeout,
            max_attempts=_synthesize_opencode_attempts(),
            max_cli_retries=_opencode_cli_retries(),
            idle_timeout_override=_synthesize_opencode_idle_timeout_sec(),
            activity_watch_paths=_synthesize_activity_watch_paths(),
            allowed_edit_paths=_synthesize_allowed_edit_paths(),
        )

    def _run_required_scaffold_repair(timeout: int) -> None:
        missing_items = _missing_synthesis_items()
        if not missing_items:
            return
        missing_txt = "\n".join(f"- {item}" for item in missing_items)
        prompt = textwrap.dedent(
            f"""
            Complete required scaffold files only. Do not rewrite unrelated files.

            Missing required items:
            {missing_txt}

            Rules:
            - Keep existing harness/build files unchanged unless needed to satisfy required items.
            - If README is missing, create `fuzz/README.md` with required fields:
              Selected target, Final target, Technical reason, Relation, Harness file.
            - If strategy is missing, create a valid `fuzz/build_strategy.json` matching the current harness/build path.
            - Do NOT run commands.
            - Write the path string `fuzz/out/` as the sole text of `./done` before exit (run `echo 'fuzz/out/' > ./done`; do **not** copy file contents).
            """
        ).strip()
        gen.patcher.run_codex_command(
            prompt,
            additional_context=_completion_context() or None,
            stage_skill="synthesize_complete_scaffold",
            timeout=timeout,
            max_attempts=_synthesize_opencode_attempts(),
            max_cli_retries=_opencode_cli_retries(),
            idle_timeout_override=_synthesize_opencode_idle_timeout_sec(),
            activity_watch_paths=_synthesize_activity_watch_paths(),
            allowed_edit_paths=_synthesize_allowed_edit_paths(),
        )

    def _ensure_min_readme_fallback() -> bool:
        readme = gen.repo_root / "fuzz" / "README.md"
        if readme.is_file():
            return False
        status = _synthesis_output_status()
        harnesses = list(status.get("harnesses") or [])
        if not harnesses:
            return False
        selected_label = selected_target_api or selected_target_name or "unknown"
        harness_label = harnesses[0]
        body = (
            "# Fuzz Harness Notes\n\n"
            f"- Selected target: {selected_label}\n"
            "- Final target: unknown\n"
            "- Technical reason: scaffold fallback README generated locally\n"
            "- Relation: to be updated after target alignment analysis\n"
            f"- Harness file: {harness_label}\n"
        )
        try:
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(body, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), "synthesize: generated fallback fuzz/README.md")
            return True
        except Exception:
            return False

    def _run_readme_alignment_completion(timeout: int, alignment: dict[str, Any]) -> None:
        selected_label = str(alignment.get("expected_api") or alignment.get("expected_target_name") or "").strip() or "unknown"
        observed_label = str(alignment.get("observed_api") or "").strip() or "unknown"
        observed_harness = str(alignment.get("observed_harness") or "").strip() or "unknown"
        prompt = textwrap.dedent(
            f"""
            Update `fuzz/README.md` only. Do not rewrite the harness.

            The generated harness drifted from the originally selected target.
            Make `fuzz/README.md` consistent with the actual harness and include these exact fields:
            - Selected target: {selected_label}
            - Final target: {observed_label}
            - Technical reason: <brief technical explanation>
            - Relation: <how the final target relates to the selected target>
            - Harness file: {observed_harness}

            Requirements:
            - The README must describe the actual observed target, not the original one.
            - Keep the README concise.
            - Do not edit any source/build files.
            - Write the path string `fuzz/README.md` as the sole text of `./done` before finishing (run `echo 'fuzz/README.md' > ./done`; do **not** copy the file's contents).
            """
        ).strip()
        gen.patcher.run_codex_command(
            prompt,
            additional_context=_completion_context() or None,
            stage_skill="synthesize_complete_scaffold",
            timeout=timeout,
            max_attempts=_synthesize_opencode_attempts(),
            max_cli_retries=_opencode_cli_retries(),
            idle_timeout_override=_synthesize_opencode_idle_timeout_sec(),
            activity_watch_paths=_synthesize_activity_watch_paths(),
            allowed_edit_paths=_synthesize_allowed_edit_paths(),
        )

    if not _has_codex_key():
        out = {
            **state,
            "last_step": "synthesize",
            "last_error": "Missing OPENAI_API_KEY for synthesis",
            "message": "synthesize failed",
        }
        out = _attach_prompt_render_status(out)
        _wf_log(cast(dict[str, Any], out), f"<- synthesize err=missing-key dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    try:
        remaining_before = _remaining_time_budget_sec(state, min_timeout=0)
        if remaining_before <= 0:
            return _time_budget_exceeded_state(state, step_name="synthesize")

        synth_template_name = "synthesize_with_hint"
        synth_stage_skill = "synthesize"
        if repair_mode:
            if repair_origin_stage == "crash":
                synth_template_name = "synthesize_repair_crash_with_hint"
                synth_stage_skill = "synthesize_repair_crash"
            elif repair_origin_stage == "fix-harness":
                synth_template_name = "synthesize_repair_fix_harness_with_hint"
                synth_stage_skill = "synthesize_repair_fix_harness"
            elif repair_origin_stage == "coverage":
                synth_template_name = "synthesize_repair_coverage_with_hint"
                synth_stage_skill = "synthesize_repair_coverage"
            else:
                synth_template_name = "synthesize_repair_build_with_hint"
                synth_stage_skill = "synthesize_repair_build"
        if hint:
            prompt, render_issue = _render_opencode_prompt_safe(
                synth_template_name,
                fallback_name="synthesize_with_hint",
                hint=hint,
                fallback_hint=hint,
            )
            if render_issue:
                _remember_prompt_render_issue(render_issue)
                _wf_log(cast(dict[str, Any], state), f"synthesize: prompt render degraded -> {render_issue}")
            # Provide context from plan/targets if present.
            plan = (gen.repo_root / "fuzz" / "PLAN.md")
            targets = (gen.repo_root / "fuzz" / "targets.json")
            ctx = ""
            try:
                if repair_mode and repair_origin_stage == "fix-harness":
                    crash_ctx, crash_known_issues = _build_fix_harness_crash_context(
                        gen.repo_root,
                        include_contents=True,
                    )
                    if crash_ctx:
                        ctx += crash_ctx + "\n\n"
                    for issue in crash_known_issues:
                        _remember_prompt_render_issue(issue)
                if plan.is_file():
                    ctx += "=== fuzz/PLAN.md ===\n" + plan.read_text(encoding="utf-8", errors="replace") + "\n\n"
                if targets.is_file():
                    ctx += "=== fuzz/targets.json ===\n" + targets.read_text(encoding="utf-8", errors="replace") + "\n"
                if antlr_context_path:
                    antlr_path_obj = Path(antlr_context_path)
                    if not antlr_path_obj.is_absolute():
                        antlr_path_obj = gen.repo_root / antlr_path_obj
                    if antlr_path_obj.is_file():
                        ctx += "\n=== fuzz/antlr_plan_context.json ===\n" + antlr_path_obj.read_text(
                            encoding="utf-8", errors="replace"
                        )
                if target_analysis_path:
                    analysis_path_obj = Path(target_analysis_path)
                    if not analysis_path_obj.is_absolute():
                        analysis_path_obj = gen.repo_root / analysis_path_obj
                    if analysis_path_obj.is_file():
                        ctx += "\n=== fuzz/target_analysis.json ===\n" + analysis_path_obj.read_text(
                            encoding="utf-8", errors="replace"
                        )
                if analysis_context_path:
                    analysis_ctx_obj = Path(analysis_context_path)
                    if not analysis_ctx_obj.is_absolute():
                        analysis_ctx_obj = gen.repo_root / analysis_ctx_obj
                    if analysis_ctx_obj.is_file():
                        ctx += "\n=== fuzz/analysis_context.json ===\n" + analysis_ctx_obj.read_text(
                            encoding="utf-8", errors="replace"
                        )
                if selected_targets_path:
                    selected_path_obj = Path(selected_targets_path)
                    if not selected_path_obj.is_absolute():
                        selected_path_obj = gen.repo_root / selected_path_obj
                    if selected_path_obj.is_file():
                        ctx += "\n=== fuzz/selected_targets.json ===\n" + selected_path_obj.read_text(
                            encoding="utf-8", errors="replace"
                        )
            except Exception:
                pass
            gen.patcher.run_codex_command(
                prompt,
                additional_context=ctx or None,
                stage_skill=synth_stage_skill,
                timeout=_remaining_time_budget_sec(state),
                max_attempts=_synthesize_opencode_attempts(),
                max_cli_retries=_opencode_cli_retries(),
                idle_timeout_override=_synthesize_opencode_idle_timeout_sec(),
                activity_watch_paths=_synthesize_activity_watch_paths(),
                allowed_edit_paths=_synthesize_allowed_edit_paths(),
            )
            grace_raw = os.environ.get("SHERPA_SYNTHESIZE_GRACE_SEC", "15").strip()
            try:
                grace_sec = max(0, min(int(grace_raw), 60))
            except Exception:
                grace_sec = 15
            if not _has_min_synthesis_outputs() and not _synthesis_grace_wait(grace_sec):
                remaining_after_hint = _remaining_time_budget_sec(state, min_timeout=0)
                if remaining_after_hint <= 0:
                    raise HarnessGeneratorError(
                        "synthesize incomplete after hint-mode and no remaining workflow time budget"
                    )
                _wf_log(
                    cast(dict[str, Any], state),
                    "synthesize: missing harness after hint-mode; retrying full synthesize",
                )
                gen._pass_synthesize_harness(timeout=remaining_after_hint)
            elif not _has_required_synthesis_outputs():
                remaining_after_hint = _remaining_time_budget_sec(state, min_timeout=0)
                if remaining_after_hint <= 0:
                    raise HarnessGeneratorError(
                        "synthesize incomplete after hint-mode and no remaining workflow time budget"
                    )
                _wf_log(
                    cast(dict[str, Any], state),
                    "synthesize: partial scaffold detected after hint-mode; completing missing build scaffold",
                )
                _run_synthesize_completion(remaining_after_hint)
        else:
            remaining_direct = _remaining_time_budget_sec(state, min_timeout=0)
            if remaining_direct <= 0:
                return _time_budget_exceeded_state(state, step_name="synthesize")
            gen._pass_synthesize_harness(timeout=_remaining_time_budget_sec(state))
            if _has_min_synthesis_outputs() and not _has_required_synthesis_outputs():
                remaining_after_direct = _remaining_time_budget_sec(state, min_timeout=0)
                if remaining_after_direct <= 0:
                    raise HarnessGeneratorError("synthesize incomplete after direct synthesize and no remaining workflow time budget")
                _wf_log(
                    cast(dict[str, Any], state),
                    "synthesize: partial scaffold detected; completing missing build scaffold",
                )
                _run_synthesize_completion(remaining_after_direct)

        if not _has_min_synthesis_outputs() and not _synthesis_grace_wait(10):
            remaining_for_harness_repair = _remaining_time_budget_sec(state, min_timeout=0)
            if remaining_for_harness_repair > 0:
                _wf_log(
                    cast(dict[str, Any], state),
                    "synthesize: harness missing after grace wait; running forced harness repair",
                )
                _run_required_scaffold_repair(remaining_for_harness_repair)
            if not _has_min_synthesis_outputs() and not _synthesis_grace_wait(3):
                raise HarnessGeneratorError("synthesize incomplete: missing harness source under fuzz/")
        if not _has_required_synthesis_outputs():
            try:
                required_grace_sec = max(0, min(int((os.environ.get("SHERPA_SYNTHESIZE_REQUIRED_GRACE_SEC") or "8").strip()), 60))
            except Exception:
                required_grace_sec = 8
            if required_grace_sec > 0 and _required_synthesis_grace_wait(required_grace_sec):
                _wf_log(
                    cast(dict[str, Any], state),
                    f"synthesize: required scaffold became complete during grace wait ({required_grace_sec}s)",
                )
            else:
                required_status_before = _synthesis_output_status()
                if required_status_before.get("scan_error_count"):
                    _wf_log(
                        cast(dict[str, Any], state),
                        "synthesize: required scaffold check saw scan errors: "
                        + ", ".join(str(x) for x in (required_status_before.get("scan_errors") or [])[:3]),
                    )
            remaining_for_required = _remaining_time_budget_sec(state, min_timeout=0)
            if remaining_for_required > 0 and not _has_required_synthesis_outputs():
                _wf_log(
                    cast(dict[str, Any], state),
                    "synthesize: required scaffold still missing; running forced required-scaffold repair",
                )
                _run_required_scaffold_repair(remaining_for_required)
            if not _has_required_synthesis_outputs():
                _ensure_min_readme_fallback()
            if not _has_required_synthesis_outputs():
                missing = ", ".join(_missing_synthesis_items()) or "unknown required files"
                diag = _synthesis_output_status()
                diag_bits: list[str] = []
                if int(diag.get("scan_error_count") or 0) > 0:
                    diag_bits.append(f"scan_errors={int(diag.get('scan_error_count') or 0)}")
                harness_count = len(list(diag.get("harnesses") or []))
                diag_bits.append(f"harnesses={harness_count}")
                diag_tail = f" [diagnostics: {', '.join(diag_bits)}]" if diag_bits else ""
                raise HarnessGeneratorError(f"synthesize incomplete: missing required scaffold items: {missing}{diag_tail}")
        _run_post_synthesize_build_validation()
        _, execution_plan_doc = _sync_execution_plan_doc_from_selected_targets(gen.repo_root)
        boundary_target_state = _workflow_target_state_from_execution_plan(
            gen.repo_root,
            execution_plan_doc,
        )
        harness_index_path = ""
        harness_index_doc: dict[str, Any] = {}
        try:
            harness_ok, harness_reason, harness_index_doc = _validate_execution_plan_harness_consistency(
                gen.repo_root,
                execution_plan_doc=execution_plan_doc,
            )
            harness_index_path, harness_index_doc = _write_harness_index_doc(
                gen.repo_root,
                execution_plan_doc=execution_plan_doc,
            )
            if not harness_ok:
                raise HarnessGeneratorError(f"synthesize incomplete: {harness_reason}")
            repair_ok, repair_reason = _validate_build_repair_contract(
                gen.repo_root,
                state,
                harness_index_doc,
            )
            if not repair_ok:
                raise HarnessGeneratorError(f"synthesize incomplete: {repair_reason}")
            harness_contract_ok, harness_contract_reason = _validate_harness_source_contract(
                gen.repo_root,
                harness_index_doc,
            )
            if not harness_contract_ok:
                raise HarnessGeneratorError(f"synthesize incomplete: {harness_contract_reason}")
        except HarnessGeneratorError:
            raise
        except Exception as e:
            raise HarnessGeneratorError(f"synthesize incomplete: unable to build harness index: {e}")
        target_alignment = _analyze_harness_target_alignment(gen.repo_root)
        readme_alignment = {
            "complete": True,
            "missing": [],
            "relation": "",
            "reason": "",
        }
        if target_alignment.get("drifted"):
            _wf_log(
                cast(dict[str, Any], state),
                "synthesize: soft target drift accepted: "
                + str(target_alignment.get("reason") or "selected target drift detected"),
            )
            readme_alignment = _readme_drift_status(gen.repo_root, target_alignment)
            if not bool(readme_alignment.get("complete")):
                remaining_for_readme = _remaining_time_budget_sec(state, min_timeout=0)
                if remaining_for_readme > 0:
                    _wf_log(
                        cast(dict[str, Any], state),
                        "synthesize: README drift record incomplete; repairing README metadata",
                    )
                    _run_readme_alignment_completion(remaining_for_readme, target_alignment)
                    readme_alignment = _readme_drift_status(gen.repo_root, target_alignment)
        observed_target_path = ""
        repo_understanding_path = ""
        build_strategy_path = ""
        build_strategy_doc: dict[str, Any] = {}
        try:
            observed_target_path, _ = _write_observed_target_doc(
                gen.repo_root,
                expected_target_name=str(target_alignment.get("expected_target_name") or selected_target_name),
                expected_api=str(target_alignment.get("expected_api") or selected_target_api),
                observed_api=str(target_alignment.get("observed_api") or ""),
                observed_harness=str(target_alignment.get("observed_harness") or ""),
                drifted=bool(target_alignment.get("drifted") or False),
                drift_reason=str(readme_alignment.get("reason") or target_alignment.get("reason") or ""),
                relation=str(readme_alignment.get("relation") or ""),
                runtime_viability=selected_target_runtime_viability,
            )
        except Exception:
            observed_target_path = ""
        repo_understanding = _load_repo_understanding_doc(gen.repo_root)
        repo_understanding_ok, repo_understanding_reason = _repo_understanding_is_complete(repo_understanding)
        if not repo_understanding_ok:
            raise HarnessGeneratorError(f"synthesize incomplete: {repo_understanding_reason}")
        repo_understanding_path = str(_repo_understanding_path(gen.repo_root))
        try:
            build_strategy_path, build_strategy_doc = _write_build_strategy_doc(gen.repo_root)
        except Exception:
            build_strategy_path = ""
            build_strategy_doc = {}
        out = {
            **state,
            "last_step": "synthesize",
            "codex_hint": "",
            "restart_to_plan": False,
            "restart_to_plan_reason": "",
            "restart_to_plan_stage": "",
            "restart_to_plan_error_text": "",
            "restart_to_plan_report_path": "",
            "repo_understanding_path": repo_understanding_path,
            "observed_target_path": observed_target_path,
            "build_strategy_path": build_strategy_path,
            "harness_index_path": harness_index_path,
            "build_mode": str(build_strategy_doc.get("build_mode") or ""),
            "build_target_source": "external_scaffold",
            "synthesize_selected_target_name": str(target_alignment.get("expected_target_name") or selected_target_name),
            "synthesize_selected_target_api": str(target_alignment.get("expected_api") or selected_target_api),
            "synthesize_observed_target_api": str(target_alignment.get("observed_api") or ""),
            "synthesize_observed_harness": str(target_alignment.get("observed_harness") or ""),
            "synthesize_target_drifted": bool(target_alignment.get("drifted") or False),
            "synthesize_target_drift_reason": str(readme_alignment.get("reason") or target_alignment.get("reason") or ""),
            "synthesize_target_relation": str(readme_alignment.get("relation") or ""),
            "synthesize_target_runtime_viability": selected_target_runtime_viability,
            "coverage_target_api": str(target_alignment.get("observed_api") or selected_target_api or ""),
            "coverage_target_name": str(target_alignment.get("expected_target_name") or selected_target_name or state.get("coverage_target_name") or ""),
            **boundary_target_state,
            "analysis_context_path": analysis_context_path or str(state.get("analysis_context_path") or ""),
            "analysis_evidence_count": analysis_evidence_count,
            "target_scoring_enabled": bool(state.get("target_scoring_enabled") or False),
            "target_score_breakdown_available": bool(state.get("target_score_breakdown_available") or False),
            "constraint_memory_count": int(state.get("constraint_memory_count") or 0),
            "crash_signature_dedup_hit": bool(state.get("crash_signature_dedup_hit") or False),
            "message": "synthesized",
        }
        out = _attach_prompt_render_status(out, issue=prompt_render_issue)
        out = _clear_error_markers_on_success(out)
        _wf_log(cast(dict[str, Any], out), f"<- synthesize ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    except Exception as e:
        _write_stage_feedback(
            gen.repo_root,
            stage="synthesize",
            error_text=str(e),
            state=cast(dict[str, Any], state),
        )
        out = {**state, "last_step": "synthesize", "last_error": str(e), "message": "synthesize failed", "failed": True}
        out = _attach_prompt_render_status(out, issue=prompt_render_issue or str(e))
        _wf_log(cast(dict[str, Any], out), f"<- synthesize err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
