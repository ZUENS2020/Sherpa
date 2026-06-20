"""Carved from workflow_graph.py - '_node_fix_crash' LangGraph node."""

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


def _node_fix_crash(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "fix_crash")
    if stop_now:
        return state

    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> fix_crash")
    prompt_render_issue = ""

    repo_root = gen.repo_root
    snapshot = snapshot_repo_text(repo_root)
    crash_info = repo_root / "crash_info.md"
    crash_analysis = repo_root / "crash_analysis.md"
    last_artifact = (state.get("last_crash_artifact") or "").strip()
    last_fuzzer = (state.get("last_fuzzer") or "").strip()

    info_text = crash_info.read_text(encoding="utf-8", errors="replace") if crash_info.is_file() else ""
    analysis_text = crash_analysis.read_text(encoding="utf-8", errors="replace") if crash_analysis.is_file() else ""
    harness_error = bool(re.search(r"HARNESS ERROR", analysis_text, re.IGNORECASE))

    if harness_error:
        prompt, render_issue = _render_opencode_prompt_safe(
            "fix_crash_harness_error",
            fallback_name="fix_crash_upstream_bug",
        )
    else:
        prompt, render_issue = _render_opencode_prompt_safe(
            "fix_crash_upstream_bug",
            fallback_name="fix_crash_harness_error",
        )
    if render_issue:
        prompt_render_issue = str(render_issue)
        _wf_log(cast(dict[str, Any], state), f"fix-crash: prompt render degraded -> {render_issue}")

    ctx_parts: list[str] = []
    if last_fuzzer:
        ctx_parts.append(f"fuzzer: {last_fuzzer}")
    if last_artifact:
        ctx_parts.append(f"crashing_artifact: {last_artifact}")
    if info_text:
        ctx_parts.append("=== crash_info.md ===\n" + info_text)
    if analysis_text:
        ctx_parts.append("=== crash_analysis.md ===\n" + analysis_text)
    context = "\n\n".join(ctx_parts)

    attempts = int(state.get("crash_fix_attempts") or 0) + 1
    try:
        gen.patcher.run_codex_command(
            prompt,
            additional_context=context or None,
            stage_skill=("fix_crash_harness_error" if harness_error else "fix_crash_upstream_bug"),
            timeout=_remaining_time_budget_sec(state),
            max_attempts=1,
            max_cli_retries=_opencode_cli_retries(),
            activity_watch_paths=("fix.patch", "fix_summary.md", "done"),
        )
        patch_path = repo_root / "fix.patch"
        _grace_wait_for_file(patch_path, max_sec=5, min_size=0)
        fix_summary_path = repo_root / "fix_summary.md"
        changed_files = write_patch_from_snapshot(snapshot, repo_root, patch_path)
        patch_bytes = patch_path.stat().st_size if patch_path.exists() else 0
        if not changed_files:
            out = {
                **state,
                "last_step": "fix_crash",
                "last_error": "opencode fix_crash made no textual file changes",
                "crash_fix_attempts": attempts,
                "message": "opencode fix_crash no-op",
                "fix_patch_path": str(patch_path) if patch_path.exists() else "",
                "fix_patch_files": [],
                "fix_patch_bytes": int(patch_bytes),
            }
            out = _attach_prompt_render_status(out, issue=prompt_render_issue)
            _wf_log(cast(dict[str, Any], out), f"<- fix_crash err=no-op dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        # Write a concise fix summary for downstream triage.
        summary_lines = [
            "# Fix Patch Summary",
            "",
            f"- Fix type: {'harness_error' if harness_error else 'upstream_bug'}",
            f"- Patch file: {patch_path}",
            f"- Files changed: {len(changed_files)}",
            "",
        ]
        if changed_files:
            summary_lines.append("## Files")
            summary_lines.extend([f"- {p}" for p in changed_files])
        else:
            summary_lines.append("_No textual changes detected for patch generation._")
        fix_summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8", errors="replace")

        # If a challenge bundle already exists, attach patch artifacts.
        for child in repo_root.iterdir():
            if not child.is_dir():
                continue
            if child.name.startswith(("challenge_bundle", "false_positive", "unreproducible")):
                if patch_path.exists():
                    shutil.copy2(patch_path, child / patch_path.name)
                if fix_summary_path.exists():
                    shutil.copy2(fix_summary_path, child / fix_summary_path.name)

        out = {
            **state,
            "last_step": "fix_crash",
            "last_error": "",
            "crash_fix_attempts": attempts,
            "message": "opencode fixed crash" if not harness_error else "opencode fixed harness error",
            "fix_patch_path": str(patch_path) if patch_path.exists() else "",
            "fix_patch_files": changed_files,
            "fix_patch_bytes": int(patch_bytes),
        }
        out = _attach_prompt_render_status(out, issue=prompt_render_issue)
        _wf_log(cast(dict[str, Any], out), f"<- fix_crash ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    except Exception as e:
        out = {
            **state,
            "last_step": "fix_crash",
            "last_error": str(e),
            "crash_fix_attempts": attempts,
            "message": "opencode fix_crash failed",
        }
        out = _attach_prompt_render_status(out, issue=prompt_render_issue or str(e))
        _wf_log(cast(dict[str, Any], out), f"<- fix_crash err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
