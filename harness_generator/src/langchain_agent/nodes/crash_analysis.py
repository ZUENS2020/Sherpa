"""Carved from workflow_graph.py - '_node_crash_analysis' LangGraph node."""

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
    _constraint_memory_path,
    _grace_wait_for_file,
    _normalize_crash_analysis_verdict,
    _opencode_cli_retries,
    _record_constraint_memory_observation,
    _record_decision_trace,
    _remaining_time_budget_sec,
    _render_opencode_prompt_safe,
    _trim_feedback_text,
    _write_crash_vuln_candidate,
)


def _node_crash_analysis(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "crash-analysis")
    if stop_now:
        return state

    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> crash-analysis")

    repo_root = gen.repo_root
    crash_info = repo_root / "crash_info.md"
    re_run_report = repo_root / "re_run_report.md"
    triage_json_path = repo_root / "crash_triage.json"
    analysis_json_path = repo_root / "crash_analysis.json"
    analysis_md_path = repo_root / "crash_analysis.md"

    info_text = crash_info.read_text(encoding="utf-8", errors="replace") if crash_info.is_file() else ""
    re_run_text = re_run_report.read_text(encoding="utf-8", errors="replace") if re_run_report.is_file() else ""
    triage_doc: dict[str, Any] = {}
    if triage_json_path.is_file():
        try:
            parsed = json.loads(triage_json_path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(parsed, dict):
                triage_doc = parsed
        except Exception:
            triage_doc = {}

    prompt_render_issue = ""
    prompt, render_issue = _render_opencode_prompt_safe(
        "crash_analysis_with_hint",
        fallback_name="analysis_with_hint",
        fallback_hint=str(state.get("codex_hint") or ""),
        known_issues=["crash-analysis prompt render degraded"],
        hint=str(state.get("codex_hint") or ""),
    )
    if render_issue:
        prompt_render_issue = str(render_issue)
        _wf_log(cast(dict[str, Any], state), f"crash-analysis prompt degraded: {prompt_render_issue}")
    context_parts = [
        f"last_fuzzer: {str(state.get('last_fuzzer') or '').strip()}",
        f"last_crash_artifact: {str(state.get('last_crash_artifact') or '').strip()}",
        f"crash_signature: {str(state.get('crash_signature') or '').strip()}",
    ]
    if info_text:
        context_parts.append("=== crash_info.md ===\n" + info_text)
    if re_run_text:
        context_parts.append("=== re_run_report.md ===\n" + _trim_feedback_text(re_run_text))
    if triage_doc:
        context_parts.append("=== crash_triage.json ===\n" + json.dumps(triage_doc, ensure_ascii=False, indent=2))
    try:
        import contract_analysis  # local import; best-effort
        crash_text = "\n".join(filter(None, [info_text, re_run_text]))
        contract_block = contract_analysis.build_contract_triage_context(
            repo_root, crash_text,
            fuzzer_name=str(state.get("last_fuzzer") or "").strip(),
        )
        if contract_block:
            context_parts.append(contract_block)
            _wf_log(cast(dict[str, Any], state), "crash-analysis: injected documented API preconditions (contract-aware)")
    except Exception as _contract_exc:  # never block analysis on contract extraction
        _wf_log(cast(dict[str, Any], state), f"crash-analysis: contract extraction skipped: {_contract_exc}")
    context = "\n\n".join(context_parts)

    verdict = "unknown"
    reason = "model output invalid/incomplete"
    evidence: list[str] = []
    recommended_action = "stop_report"
    model_output_valid = False
    try:
        gen.patcher.run_codex_command(
            prompt,
            additional_context=context or None,
            stage_skill="crash_analysis",
            timeout=_remaining_time_budget_sec(state),
            max_attempts=1,
            max_cli_retries=_opencode_cli_retries(),
            activity_watch_paths=("crash_analysis.json", "done"),
        )
        _grace_wait_for_file(analysis_json_path, max_sec=5)
        parsed_doc: dict[str, Any] = {}
        if analysis_json_path.is_file():
            try:
                raw = analysis_json_path.read_text(encoding="utf-8", errors="replace").strip()
                if raw:
                    loaded = json.loads(raw)
                    if isinstance(loaded, dict):
                        parsed_doc = loaded
            except Exception:
                parsed_doc = {}
        if parsed_doc:
            verdict = _normalize_crash_analysis_verdict(str(parsed_doc.get("verdict") or ""))
            reason = str(parsed_doc.get("reason") or "").strip()
            ev = parsed_doc.get("evidence")
            if not ev:
                ev = parsed_doc.get("signals")
            evidence = [str(x).strip() for x in list(ev or []) if str(x).strip()]
            recommended_action = str(parsed_doc.get("recommended_action") or "").strip().lower() or (
                "repair_harness" if verdict == "false_positive" else "stop_report"
            )
            model_output_valid = bool(reason and evidence)
    except Exception as e:
        reason = f"model output invalid/incomplete: {e}"
        model_output_valid = False

    if not model_output_valid:
        verdict = "unknown"
        recommended_action = "stop_report"
        if not reason.startswith("model output invalid/incomplete"):
            reason = "model output invalid/incomplete"
        evidence = ["model output invalid/incomplete"]

    if not evidence:
        evidence = ["no concrete crash-analysis evidence captured"]
    if verdict == "false_positive":
        recommended_action = "repair_harness"
    elif recommended_action not in {"repair_harness", "stop_report"}:
        recommended_action = "stop_report"

    analysis_doc = {
        "verdict": verdict,
        "reason": reason,
        "evidence": evidence,
        "recommended_action": recommended_action,
        "last_fuzzer": str(state.get("last_fuzzer") or ""),
        "last_crash_artifact": str(state.get("last_crash_artifact") or ""),
        "crash_signature": str(state.get("crash_signature") or ""),
    }
    analysis_json_path.write_text(json.dumps(analysis_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_lines = [
        "# Crash Analysis",
        "",
        f"- verdict: {verdict}",
        f"- recommended_action: {recommended_action}",
        f"- reason: {reason}",
        "",
        "## Evidence",
        "",
    ]
    for line in evidence:
        md_lines.append(f"- {line}")
    analysis_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    constraint_count = int(state.get("constraint_memory_count") or 0)
    constraint_path = str(_constraint_memory_path(repo_root))
    if str(state.get("crash_signature") or "").strip():
        try:
            analysis_confidence = 0.8 if verdict in {"false_positive", "real_bug"} else 0.45
            constraint_count, constraint_path, _ = _record_constraint_memory_observation(
                repo_root=repo_root,
                signature=str(state.get("crash_signature") or ""),
                stage="crash-analysis",
                classification=verdict,
                reason=reason,
                evidence=evidence,
                confidence=analysis_confidence,
                repeats=int(state.get("same_crash_repeats") or 0) + 1,
            )
        except Exception as exc:
            _wf_log(cast(dict[str, Any], state), f"crash-analysis: constraint memory update skipped: {exc}")

    false_positive = verdict == "false_positive"
    restart_reason = "crash_false_positive" if false_positive else ""
    restart_error = reason[:4096] if false_positive else ""
    now_ts = int(time.time())
    crash_vuln = _write_crash_vuln_candidate(
        repo_root,
        cast(dict[str, Any], state),
        stage="crash-analysis",
        classification=verdict,
        confidence=(0.8 if verdict in {"false_positive", "real_bug"} else 0.45),
        reason=reason,
        evidence=evidence,
        triage_label=str(triage_doc.get("label") or state.get("crash_triage_label") or ""),
        analysis_verdict=verdict,
        info_text=info_text,
        runtime_text=re_run_text,
    )
    out = {
        **state,
        "last_step": "crash-analysis",
        "last_error": "",
        "crash_analysis_done": True,
        "crash_analysis_verdict": verdict,
        "crash_analysis_reason": reason,
        "crash_analysis_report_path": str(analysis_md_path),
        "crash_analysis_json_path": str(analysis_json_path),
        "restart_to_plan": false_positive,
        "restart_to_plan_reason": restart_reason,
        "restart_to_plan_stage": "crash-analysis" if false_positive else "",
        "restart_to_plan_error_text": restart_error,
        "restart_to_plan_report_path": str(analysis_md_path) if false_positive else "",
        "repair_mode": false_positive,
        "repair_origin_stage": "crash" if false_positive else "",
        "repair_error_kind": "false_positive_crash" if false_positive else "",
        "repair_error_code": restart_reason if false_positive else "",
        "repair_signature": str(state.get("crash_signature") or "")[:12] if false_positive else "",
        "repair_stdout_tail": "",
        "repair_stderr_tail": "",
        "repair_attempt_index": (int(state.get("repair_attempt_index") or 0) + 1) if false_positive else 0,
        "repair_strategy_force_change": bool(false_positive),
        "repair_error_digest": (
            {
                "error_code": restart_reason,
                "error_kind": "false_positive_crash",
                "signature": str(state.get("crash_signature") or "")[:12],
                "failing_files": [],
                "symbols": [],
                "first_seen": now_ts,
                "latest_seen": now_ts,
                "top_trace": evidence[0] if evidence else reason[:256],
            }
            if false_positive
            else {}
        ),
        "repair_recent_attempts": (
            (list(state.get("repair_recent_attempts") or []) + [{
                "step": "crash-analysis",
                "origin": "crash",
                "error_kind": "false_positive_crash",
                "error_code": restart_reason,
                "signature": str(state.get("crash_signature") or "")[:12],
                "attempt_index": int(state.get("repair_attempt_index") or 0) + 1,
                "message": reason[:512],
            }])[-5:]
            if false_positive
            else []
        ),
        "constraint_memory_count": constraint_count,
        "constraint_memory_path": constraint_path,
        "crash_signature_dedup_hit": bool(int(state.get("same_crash_repeats") or 0) > 0),
        "vuln_candidates_path": str(crash_vuln.get("path") or ""),
        "crash_vuln_report_path": str(crash_vuln.get("report_path") or ""),
        "latest_crash_vuln_candidate": dict(crash_vuln.get("candidate") or {}),
        "crash_vuln_candidate_count": int(crash_vuln.get("candidate_count") or 0),
        "message": "crash-analysis false_positive" if false_positive else "crash-analysis stop",
    }
    choose_repair_snapshot = {
        "kind": "choose_repair",
        "classification_stage": "crash-analysis",
        "classification": verdict,
        "repair_mode": bool(false_positive),
        "repair_origin_stage": str(out.get("repair_origin_stage") or ""),
        "repair_signature": str(out.get("repair_signature") or ""),
        "constraint_memory_count": int(constraint_count),
        "vuln_candidate_status": str(dict(crash_vuln.get("candidate") or {}).get("validation_status") or ""),
        "degraded_reason": "" if model_output_valid else "model_output_invalid_or_incomplete",
    }
    out = _attach_prompt_render_status(out, issue=prompt_render_issue)
    out = _record_decision_trace(
        out,
        stage="crash-analysis",
        tool="opencode",
        model=str(state.get("model") or ""),
        latency_ms=int(max(0.0, (time.perf_counter() - t0) * 1000.0)),
        error_kind="" if model_output_valid else "model_output_invalid",
        error_code="" if model_output_valid else "model_output_invalid",
        retry_count=0,
        decision_snapshot=choose_repair_snapshot,
    )
    _wf_log(
        cast(dict[str, Any], out),
        f"<- crash-analysis verdict={verdict} action={recommended_action} dt={_fmt_dt(time.perf_counter()-t0)}",
    )
    return out
