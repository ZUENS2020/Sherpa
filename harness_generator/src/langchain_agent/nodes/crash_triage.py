"""Carved from workflow_graph.py - '_node_crash_triage' LangGraph node."""

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
    _normalize_crash_triage_label,
    _opencode_cli_retries,
    _record_constraint_memory_observation,
    _record_decision_trace,
    _remaining_time_budget_sec,
    _render_opencode_prompt_safe,
    _trim_feedback_text,
    _write_crash_vuln_candidate,
)


def _node_crash_triage(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "crash-triage")
    if stop_now:
        return state

    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> crash-triage")

    repo_root = gen.repo_root
    crash_info = repo_root / "crash_info.md"
    crash_analysis = repo_root / "crash_analysis.md"
    triage_json_path = repo_root / "crash_triage.json"
    triage_md_path = repo_root / "crash_triage.md"
    re_build_report = repo_root / "re_build_report.md"
    re_run_report = repo_root / "re_run_report.md"

    info_text = crash_info.read_text(encoding="utf-8", errors="replace") if crash_info.is_file() else ""
    analysis_text = crash_analysis.read_text(encoding="utf-8", errors="replace") if crash_analysis.is_file() else ""
    re_build_text = re_build_report.read_text(encoding="utf-8", errors="replace") if re_build_report.is_file() else ""
    re_run_text = re_run_report.read_text(encoding="utf-8", errors="replace") if re_run_report.is_file() else ""
    stderr_tail = str(state.get("repair_stderr_tail") or "")[:4000]

    prompt_render_issue = ""
    prompt, render_issue = _render_opencode_prompt_safe(
        "crash_triage_with_hint",
        fallback_name="analysis_with_hint",
        fallback_hint=str(state.get("codex_hint") or ""),
        known_issues=["crash-triage prompt render degraded"],
        hint=str(state.get("codex_hint") or ""),
    )
    if render_issue:
        prompt_render_issue = str(render_issue)
        _wf_log(cast(dict[str, Any], state), f"crash-triage prompt degraded: {prompt_render_issue}")
    context_parts = [
        f"last_fuzzer: {str(state.get('last_fuzzer') or '').strip()}",
        f"last_crash_artifact: {str(state.get('last_crash_artifact') or '').strip()}",
        f"crash_signature: {str(state.get('crash_signature') or '').strip()}",
    ]
    if info_text:
        context_parts.append("=== crash_info.md ===\n" + info_text)
    if analysis_text:
        context_parts.append("=== crash_analysis.md ===\n" + analysis_text)
    if re_build_text:
        context_parts.append("=== re_build_report.md ===\n" + _trim_feedback_text(re_build_text))
    if re_run_text:
        context_parts.append("=== re_run_report.md ===\n" + _trim_feedback_text(re_run_text))
    if stderr_tail:
        context_parts.append("=== repair_stderr_tail ===\n" + stderr_tail)

    # Contract-aware triage: surface documented preconditions of the crashing
    # functions so out-of-contract crashes (harness fed input the API docs
    # forbid) are classified as harness_bug, not upstream_bug/vulnerability.
    if (os.environ.get("SHERPA_CONTRACT_TRIAGE") or "1").strip().lower() not in ("0", "false", "no", "off"):
        try:
            import contract_analysis  # local import; best-effort
            crash_text = "\n".join(filter(None, [info_text, analysis_text, stderr_tail]))
            contract_block = contract_analysis.build_contract_triage_context(
                repo_root, crash_text,
                fuzzer_name=str(state.get("last_fuzzer") or "").strip(),
            )
            if contract_block:
                context_parts.append(contract_block)
                _wf_log(cast(dict[str, Any], state), "crash-triage: injected documented API preconditions (contract-aware)")
        except Exception as _e:
            _wf_log(cast(dict[str, Any], state), f"crash-triage contract context skipped: {_e}")

    context = "\n\n".join(context_parts)

    label = "inconclusive"
    confidence = 0.35
    reason = "model output invalid/incomplete"
    signal_lines: list[str] = []
    model_output_valid = False
    try:
        gen.patcher.run_codex_command(
            prompt,
            additional_context=context or None,
            stage_skill="crash_triage",
            timeout=_remaining_time_budget_sec(state),
            max_attempts=1,
            max_cli_retries=_opencode_cli_retries(),
            activity_watch_paths=("crash_triage.json", "done"),
        )
        # Grace period for filesystem flush — OpenCode (Node.js) uses async I/O,
        # so the output file may not be flushed to disk when the done sentinel fires.
        _grace_wait_for_file(triage_json_path, max_sec=5)
        parsed: dict[str, Any] = {}
        if triage_json_path.is_file():
            try:
                raw = triage_json_path.read_text(encoding="utf-8", errors="replace").strip()
                if raw:
                    parsed = json.loads(raw)
            except Exception:
                parsed = {}
        if isinstance(parsed, dict) and parsed:
            label = _normalize_crash_triage_label(parsed.get("label"))
            raw_conf = parsed.get("confidence")
            try:
                confidence = max(0.0, min(float(raw_conf), 1.0))
            except Exception:
                confidence = 0.35
            reason = str(parsed.get("reason") or "").strip()
            evidence = parsed.get("evidence")
            if not evidence:
                evidence = parsed.get("signals")
            signal_lines = [str(x).strip() for x in (evidence or []) if str(x).strip()]
            model_output_valid = bool(reason and signal_lines)
    except Exception as e:
        reason = f"model output invalid/incomplete: {e}"
        model_output_valid = False

    if not model_output_valid:
        label = "inconclusive"
        confidence = min(confidence, 0.35)
        if not reason.startswith("model output invalid/incomplete"):
            reason = "model output invalid/incomplete"
        signal_lines = ["model output invalid/incomplete"]

    triage_doc = {
        "label": label,
        "confidence": confidence,
        "reason": reason,
        "evidence": signal_lines,
        "last_fuzzer": str(state.get("last_fuzzer") or ""),
        "last_crash_artifact": str(state.get("last_crash_artifact") or ""),
        "crash_signature": str(state.get("crash_signature") or ""),
    }
    triage_json_path.write_text(json.dumps(triage_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    triage_md = [
        "# Crash Triage Report",
        "",
        f"- label: {label}",
        f"- confidence: {confidence:.2f}",
        f"- reason: {reason}",
        "",
        "## Evidence",
    ]
    if signal_lines:
        triage_md.extend([f"- {line}" for line in signal_lines])
    else:
        triage_md.append("- (none)")
    triage_md_path.write_text("\n".join(triage_md) + "\n", encoding="utf-8")

    constraint_count = int(state.get("constraint_memory_count") or 0)
    constraint_path = str(_constraint_memory_path(repo_root))
    if str(state.get("crash_signature") or "").strip():
        try:
            constraint_count, constraint_path, _ = _record_constraint_memory_observation(
                repo_root=repo_root,
                signature=str(state.get("crash_signature") or ""),
                stage="crash-triage",
                classification=label,
                reason=reason,
                evidence=signal_lines,
                confidence=float(confidence),
                repeats=int(state.get("same_crash_repeats") or 0) + 1,
            )
        except Exception as exc:
            _wf_log(cast(dict[str, Any], state), f"crash-triage: constraint memory update skipped: {exc}")

    crash_vuln = _write_crash_vuln_candidate(
        repo_root,
        cast(dict[str, Any], state),
        stage="crash-triage",
        classification=label,
        confidence=float(confidence),
        reason=reason,
        evidence=signal_lines,
        triage_label=label,
        info_text=info_text,
        runtime_text=re_run_text,
    )

    out = {
        **state,
        "last_step": "crash-triage",
        "last_error": "",
        "crash_triage_done": True,
        "crash_triage_label": label,
        "crash_triage_confidence": float(confidence),
        "crash_triage_reason": reason,
        "crash_triage_signal_lines": signal_lines,
        "crash_triage_report_path": str(triage_md_path),
        "crash_triage_json_path": str(triage_json_path),
        "repair_mode": label == "harness_bug",
        "repair_origin_stage": "fix-harness" if label == "harness_bug" else str(state.get("repair_origin_stage") or ""),
        "repair_error_kind": "harness_bug" if label == "harness_bug" else str(state.get("repair_error_kind") or ""),
        "repair_error_code": "crash_triage_harness_bug" if label == "harness_bug" else str(state.get("repair_error_code") or ""),
        "repair_signature": (
            str(state.get("crash_signature") or "")[:12]
            if label == "harness_bug"
            else str(state.get("repair_signature") or "")
        ),
        "repair_stdout_tail": str(state.get("repair_stdout_tail") or ""),
        "repair_stderr_tail": str(state.get("repair_stderr_tail") or ""),
        "repair_attempt_index": (
            int(state.get("repair_attempt_index") or 0) + 1
            if label == "harness_bug"
            else int(state.get("repair_attempt_index") or 0)
        ),
        "repair_strategy_force_change": bool(label == "harness_bug"),
        "repair_error_digest": (
            {
                "error_code": "crash_triage_harness_bug",
                "error_kind": "harness_bug",
                "signature": str(state.get("crash_signature") or "")[:12],
                "failing_files": [],
                "symbols": [],
                "first_seen": int(time.time()),
                "latest_seen": int(time.time()),
                "top_trace": signal_lines[0] if signal_lines else reason[:256],
            }
            if label == "harness_bug"
            else dict(state.get("repair_error_digest") or {})
        ),
        "repair_recent_attempts": (
            (list(state.get("repair_recent_attempts") or []) + [{
                "step": "crash-triage",
                "origin": "fix-harness",
                "error_kind": "harness_bug",
                "error_code": "crash_triage_harness_bug",
                "signature": str(state.get("crash_signature") or "")[:12],
                "attempt_index": int(state.get("repair_attempt_index") or 0) + 1,
                "message": reason[:512],
            }])[-5:]
            if label == "harness_bug"
            else list(state.get("repair_recent_attempts") or [])
        ),
        "constraint_memory_count": constraint_count,
        "constraint_memory_path": constraint_path,
        "crash_signature_dedup_hit": bool(int(state.get("same_crash_repeats") or 0) > 0),
        "vuln_candidates_path": str(crash_vuln.get("path") or ""),
        "crash_vuln_report_path": str(crash_vuln.get("report_path") or ""),
        "latest_crash_vuln_candidate": dict(crash_vuln.get("candidate") or {}),
        "crash_vuln_candidate_count": int(crash_vuln.get("candidate_count") or 0),
        "message": f"crash triage classified as {label}",
    }
    choose_repair_snapshot = {
        "kind": "choose_repair",
        "classification_stage": "crash-triage",
        "classification": label,
        "confidence": float(confidence),
        "repair_mode": bool(out.get("repair_mode") or False),
        "repair_origin_stage": str(out.get("repair_origin_stage") or ""),
        "repair_signature": str(out.get("repair_signature") or ""),
        "constraint_memory_count": int(constraint_count),
        "vuln_candidate_status": str(dict(crash_vuln.get("candidate") or {}).get("validation_status") or ""),
        "degraded_reason": "" if model_output_valid else "model_output_invalid_or_incomplete",
    }
    out = _attach_prompt_render_status(out, issue=prompt_render_issue)
    out = _record_decision_trace(
        out,
        stage="crash-triage",
        tool="opencode",
        model=str(state.get("model") or ""),
        latency_ms=int(max(0.0, (time.perf_counter() - t0) * 1000.0)),
        error_kind="" if model_output_valid else "model_output_invalid",
        error_code="" if model_output_valid else "model_output_invalid",
        retry_count=0,
        decision_snapshot=choose_repair_snapshot,
    )
    _wf_log(cast(dict[str, Any], out), f"<- crash-triage label={label} conf={confidence:.2f} dt={_fmt_dt(time.perf_counter()-t0)}")
    return out
