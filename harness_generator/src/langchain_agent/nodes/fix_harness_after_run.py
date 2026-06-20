"""Carved from workflow_graph.py - '_node_fix_harness_after_run' LangGraph node."""

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
    _grace_wait_for_file,
    _opencode_cli_retries,
    _remaining_time_budget_sec,
    _render_opencode_prompt_safe,
)


def _node_fix_harness_after_run(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "fix-harness")
    if stop_now:
        return state

    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> fix-harness")

    repo_root = gen.repo_root
    snapshot = snapshot_repo_text(repo_root)
    crash_info = repo_root / "crash_info.md"
    crash_analysis = repo_root / "crash_analysis.md"
    triage_json = repo_root / "crash_triage.json"
    info_text = crash_info.read_text(encoding="utf-8", errors="replace") if crash_info.is_file() else ""
    analysis_text = crash_analysis.read_text(encoding="utf-8", errors="replace") if crash_analysis.is_file() else ""
    triage_text = triage_json.read_text(encoding="utf-8", errors="replace") if triage_json.is_file() else ""

    prompt_render_issue = ""
    prompt, render_issue = _render_opencode_prompt_safe(
        "fix_harness_after_run",
        fallback_name="synthesize_repair_fix_harness_with_hint",
        fallback_hint=str(state.get("codex_hint") or ""),
        known_issues=["fix-harness prompt render degraded"],
        hint=str(state.get("codex_hint") or ""),
    )
    if render_issue:
        prompt_render_issue = str(render_issue)
        _wf_log(cast(dict[str, Any], state), f"fix-harness prompt degraded: {prompt_render_issue}")
    ctx_parts: list[str] = []
    if info_text:
        ctx_parts.append("=== crash_info.md ===\n" + info_text)
    if analysis_text:
        ctx_parts.append("=== crash_analysis.md ===\n" + analysis_text)
    if triage_text:
        ctx_parts.append("=== crash_triage.json ===\n" + triage_text)
    if str(state.get("repair_stderr_tail") or "").strip():
        ctx_parts.append("=== repair_stderr_tail ===\n" + str(state.get("repair_stderr_tail") or ""))
    context = "\n\n".join(ctx_parts)

    attempts = int(state.get("fix_harness_attempts") or 0) + 1
    try:
        gen.patcher.run_codex_command(
            prompt,
            additional_context=context or None,
            stage_skill="fix_harness_after_run",
            timeout=_remaining_time_budget_sec(state),
            max_attempts=1,
            max_cli_retries=_opencode_cli_retries(),
            activity_watch_paths=("fix.patch", "done"),
        )
        patch_path = repo_root / "fix.patch"
        _grace_wait_for_file(patch_path, max_sec=5, min_size=0)
        changed_files = write_patch_from_snapshot(snapshot, repo_root, patch_path)
        patch_bytes = patch_path.stat().st_size if patch_path.exists() else 0
        if not changed_files:
            out = {
                **state,
                "last_step": "fix-harness",
                "last_error": "fix-harness made no textual file changes",
                "fix_harness_attempts": attempts,
                "restart_to_plan": True,
                "restart_to_plan_reason": "fix_harness_noop",
                "restart_to_plan_stage": "fix-harness",
                "restart_to_plan_error_text": "fix-harness no-op",
                "message": "fix-harness no-op",
                "fix_patch_path": str(patch_path) if patch_path.exists() else "",
                "fix_patch_files": [],
                "fix_patch_bytes": int(patch_bytes),
            }
            out = _attach_prompt_render_status(out, issue=prompt_render_issue)
            _wf_log(cast(dict[str, Any], out), f"<- fix-harness err=no-op dt={_fmt_dt(time.perf_counter()-t0)}")
            return out
        out = {
            **state,
            "last_step": "fix-harness",
            "last_error": "",
            "fix_harness_attempts": attempts,
            "restart_to_plan": False,
            "restart_to_plan_reason": "",
            "restart_to_plan_stage": "",
            "restart_to_plan_error_text": "",
            "message": "harness fix applied",
            "fix_patch_path": str(patch_path) if patch_path.exists() else "",
            "fix_patch_files": changed_files,
            "fix_patch_bytes": int(patch_bytes),
        }
        out = _attach_prompt_render_status(out, issue=prompt_render_issue)
        _wf_log(cast(dict[str, Any], out), f"<- fix-harness ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    except Exception as e:
        out = {
            **state,
            "last_step": "fix-harness",
            "last_error": str(e),
            "fix_harness_attempts": attempts,
            "restart_to_plan": True,
            "restart_to_plan_reason": "fix_harness_failed",
            "restart_to_plan_stage": "fix-harness",
            "restart_to_plan_error_text": str(e),
            "message": "fix-harness failed",
        }
        out = _attach_prompt_render_status(out, issue=prompt_render_issue or str(e))
        _wf_log(cast(dict[str, Any], out), f"<- fix-harness err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
