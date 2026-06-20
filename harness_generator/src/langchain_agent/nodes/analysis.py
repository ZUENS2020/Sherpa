"""Carved from workflow_graph.py - '_node_analysis' LangGraph node."""

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
    _analysis_opencode_advisory_enabled,
    _analysis_opencode_idle_timeout_sec,
    _analysis_opencode_timeout_sec,
    _attach_prompt_render_status,
    _build_analysis_evidence_index,
    _clear_error_markers_on_success,
    _collect_analysis_companion_context,
    _has_codex_key,
    _prepare_antlr_assist_context,
    _prepare_target_analysis_context,
    _read_json_doc,
    _render_opencode_prompt_safe,
    _write_analysis_vuln_candidates,
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


def _node_analysis(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "analysis")
    if stop_now:
        return state
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> analysis")
    hint = (state.get("codex_hint") or "").strip()
    attempts = 2
    last_err = ""
    antlr_context_path = str(state.get("antlr_context_path") or "")
    antlr_context_summary = str(state.get("antlr_context_summary") or "")
    target_analysis_path = str(state.get("target_analysis_path") or "")
    target_analysis_summary = str(state.get("target_analysis_summary") or "")
    analysis_context_path = str(state.get("analysis_context_path") or "")
    analysis_report_path = ""
    companion_doc: dict[str, Any] = {}
    companion_summary = ""
    analysis_evidence_count = int(state.get("analysis_evidence_count") or 0)
    prompt_render_issue = ""
    analysis_advisory_degraded = False
    analysis_advisory_error = ""

    for attempt in range(1, attempts + 1):
        try:
            antlr_context_path, antlr_context_summary = _prepare_antlr_assist_context(gen.repo_root)
            target_analysis_path, target_analysis_summary = _prepare_target_analysis_context(gen.repo_root)
            companion_doc, companion_summary = _collect_analysis_companion_context()
            antlr_doc = _read_json_doc(antlr_context_path)
            target_doc = _read_json_doc(target_analysis_path)
            evidence_doc = _build_analysis_evidence_index(
                repo_root=gen.repo_root,
                antlr_doc=antlr_doc,
                target_doc=target_doc,
                companion_doc=companion_doc,
            )
            analysis_evidence_count = int((evidence_doc.get("summary") or {}).get("evidence_count") or 0)
            fuzz_dir = gen.repo_root / "fuzz"
            fuzz_dir.mkdir(parents=True, exist_ok=True)
            analysis_doc = {
                "mode": "pre-plan-analysis",
                "generated_at": int(time.time()),
                "repo_root": str(gen.repo_root),
                "antlr_context_path": antlr_context_path,
                "antlr_context_summary": antlr_context_summary,
                "target_analysis_path": target_analysis_path,
                "target_analysis_summary": target_analysis_summary,
                "vuln_hunting_enabled": bool(_vuln_hunting_enabled()),
                "vuln_focus_profile": "broad_high_risk",
                "target_surface_policy": "risk_first",
                "companion": companion_doc,
                "analysis_evidence": evidence_doc,
            }
            analysis_path = fuzz_dir / "analysis_context.json"
            analysis_path.write_text(json.dumps(analysis_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            analysis_context_path = str(analysis_path)
            analysis_report_path = str(analysis_path)
            vuln_candidates_doc = _write_analysis_vuln_candidates(gen.repo_root, analysis_context_path)

            if _has_codex_key() and _analysis_opencode_advisory_enabled():
                analysis_lines: list[str] = [
                    "Generate analysis artifacts for downstream planning.",
                    "Do not rewrite the full `fuzz/analysis_context.json`; it is already system-generated.",
                    "Do not rewrite `fuzz/vuln_candidates.json`; it is already system-generated.",
                    "Write concise AI advisory findings to `fuzz/vuln_hypotheses.md` and keep them evidence-linked.",
                    "Bounded analysis mode: after required files, use at most 6 additional MCP/tool reads in the first pass.",
                    "Prefer existing `analysis_evidence.security_evidence[]` and `fuzz/vuln_candidates.json`; treat them as sufficient unless empty or corrupt.",
                    "Do not call semantic/comprehension MCP tools unless this coordinator hint explicitly asks for semantic enrichment.",
                    "After one bounded evidence pass, write `fuzz/vuln_hypotheses.md` and `./done`; do not continue open-ended exploration.",
                ]
                if antlr_context_summary:
                    analysis_lines.append(f"ANTLR context: {antlr_context_summary}")
                if target_analysis_summary:
                    analysis_lines.append(f"Target analysis: {target_analysis_summary}")
                if companion_summary:
                    analysis_lines.append(f"Companion signals: {companion_summary}")
                analysis_lines.append(f"Evidence index count: {analysis_evidence_count}")
                analysis_hint = "\n".join(analysis_lines)
                if hint:
                    analysis_hint = f"{analysis_hint}\n\nCoordinator hint:\n{hint}"
                prompt, render_issue = _render_opencode_prompt_safe(
                    "analysis_with_hint",
                    fallback_name="plan_with_hint",
                    hint=analysis_hint,
                    fallback_hint=analysis_hint,
                )
                if render_issue:
                    prompt_render_issue = str(render_issue)
                    _wf_log(cast(dict[str, Any], state), f"analysis: prompt render degraded -> {render_issue}")
                try:
                    gen.patcher.run_codex_command(
                        prompt,
                        stage_skill="analysis",
                        timeout=_analysis_opencode_timeout_sec(state),
                        max_attempts=1,
                        max_cli_retries=1,
                        idle_timeout_override=_analysis_opencode_idle_timeout_sec(),
                        activity_watch_paths=("fuzz/vuln_hypotheses.md", "done"),
                    )
                except Exception as e:
                    analysis_advisory_degraded = True
                    analysis_advisory_error = str(e)[:4096]
                    prompt_render_issue = prompt_render_issue or f"analysis_advisory_failed: {analysis_advisory_error}"
                    _wf_log(
                        cast(dict[str, Any], state),
                        f"analysis: OpenCode advisory degraded; continuing with system evidence: {analysis_advisory_error}",
                    )
                if not analysis_path.is_file():
                    analysis_path.write_text(json.dumps(analysis_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

                # ── Refresh target_analysis_summary from potentially updated file ──
                _ta_path = gen.repo_root / "fuzz" / "target_analysis.json"
                if _ta_path.is_file():
                    try:
                        _refreshed_doc = json.loads(_ta_path.read_text(encoding="utf-8", errors="replace"))
                        if isinstance(_refreshed_doc, dict):
                            _rec = _refreshed_doc.get("recommended_targets") or []
                            target_analysis_summary = (
                                f"target_analysis_file=fuzz/target_analysis.json; "
                                f"candidates={len(_refreshed_doc.get('candidate_functions') or [])}; "
                                + "recommended="
                                + ", ".join(
                                    f"{r.get('name', '?')}:{r.get('seed_profile', '?')}"
                                    for r in _rec[:5]
                                )
                            )
                    except Exception:
                        pass

            out = {
                **state,
                "last_step": "analysis",
                "last_error": "",
                "failed": False,
                "analysis_done": True,
                "analysis_degraded": False,
                "analysis_error": "",
                "analysis_advisory_degraded": analysis_advisory_degraded,
                "analysis_advisory_error": analysis_advisory_error,
                "analysis_report_path": analysis_report_path,
                "analysis_context_path": analysis_context_path,
                "analysis_evidence_count": analysis_evidence_count,
                "security_evidence_count": int((evidence_doc.get("summary") or {}).get("security_evidence_count") or 0),
                "vuln_candidate_count": max(
                    int((evidence_doc.get("summary") or {}).get("vuln_candidate_count") or 0),
                    int(vuln_candidates_doc.get("candidate_count") or 0),
                ),
                "vuln_candidates_path": str(vuln_candidates_doc.get("path") or ""),
                "vuln_hunting_enabled": bool(_vuln_hunting_enabled()),
                "vuln_focus_profile": "broad_high_risk",
                "target_surface_policy": "risk_first",
                "security_priority_mode": bool(_vuln_hunting_enabled() and _vuln_score_mode() == "risk_first_v1"),
                "antlr_context_path": antlr_context_path,
                "antlr_context_summary": antlr_context_summary,
                "target_analysis_path": target_analysis_path,
                "target_analysis_summary": target_analysis_summary,
                "message": "analysis completed",
            }
            out = _attach_prompt_render_status(out, issue=prompt_render_issue)
            out = _clear_error_markers_on_success(out)
            _wf_log(cast(dict[str, Any], out), f"<- analysis ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out
        except Exception as e:
            last_err = str(e)
            if attempt < attempts:
                _wf_log(cast(dict[str, Any], state), f"analysis attempt {attempt} failed; retrying once: {last_err}")
                continue
            break

    fallback_error = last_err or "analysis_failed"
    out = {
        **state,
        "last_step": "analysis",
        "last_error": "",
        "failed": False,
        "analysis_done": False,
        "analysis_degraded": True,
        "analysis_error": fallback_error[:4096],
        "analysis_report_path": analysis_report_path,
        "analysis_context_path": analysis_context_path,
        "analysis_evidence_count": analysis_evidence_count,
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
        "antlr_context_path": antlr_context_path,
        "antlr_context_summary": antlr_context_summary,
        "target_analysis_path": target_analysis_path,
        "target_analysis_summary": target_analysis_summary,
        "message": "analysis degraded",
    }
    out = _attach_prompt_render_status(out, issue=prompt_render_issue or fallback_error)
    out = _clear_error_markers_on_success(out)
    _wf_log(cast(dict[str, Any], out), f"<- analysis degraded err={fallback_error} dt={_fmt_dt(time.perf_counter()-t0)}")
    return out
