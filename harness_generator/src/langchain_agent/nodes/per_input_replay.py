"""Carved from workflow_graph.py - '_node_per_input_replay' LangGraph node."""

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
    _execution_plan_targets,
    _execution_target_identity,
    _preferred_execution_target,
    _resolve_per_input_replay_binary,
    _target_type_from_run_details,
)


def _node_per_input_replay(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "per-input-replay")
    if stop_now:
        return state
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> per-input-replay")
    try:
        run_details = list(state.get("run_details") or [])
        execution_targets = _execution_plan_targets(gen.repo_root)
        preferred_target = _preferred_execution_target(
            execution_targets,
            cast(dict[str, Any], state),
            run_details=run_details,
        )
        preferred_identity = _execution_target_identity(preferred_target) if preferred_target else {}
        fuzzer_name = str(preferred_identity.get("expected_fuzzer_name") or "").strip()
        if not fuzzer_name and run_details:
            fuzzer_name = str(run_details[0].get("fuzzer") or "").strip()
        if not fuzzer_name:
            fuzzer_name = str(state.get("last_fuzzer") or state.get("coverage_target_name") or "").strip()

        replay_binary = _resolve_per_input_replay_binary(gen.repo_root, fuzzer_name) if fuzzer_name else None
        if not fuzzer_name or replay_binary is None:
            out = {
                **state,
                "last_step": "per-input-replay",
                "last_error": "",
                "coverage_per_input_manifest_path": str(state.get("coverage_per_input_manifest_path") or ""),
                "coverage_frontier_path": str(state.get("coverage_frontier_path") or ""),
                "coverage_frontier_summary": dict(state.get("coverage_frontier_summary") or {}),
                "coverage_replay_runtime_sec": 0.0,
                "coverage_replay_binary_hash": "",
                "coverage_replay_stage_success": False,
                "coverage_replay_error": "replay_binary_missing" if fuzzer_name else "replay_target_missing",
                "coverage_replay_manifest_fresh_for_current_binary": False,
                "coverage_replay_queue_drained": False,
                "coverage_replay_pending_inputs": 0,
                "coverage_replay_failed_inputs": 0,
                "coverage_replay_processed_inputs": 0,
                "coverage_replay_total_inputs": 0,
                "message": "per-input replay skipped",
            }
            _wf_log(
                cast(dict[str, Any], out),
                f"<- per-input-replay skip reason={out['coverage_replay_error']} dt={_fmt_dt(time.perf_counter()-t0)}",
            )
            return out

        replay = collect_per_input_frontier(
            repo_root=gen.repo_root,
            fuzzer_name=fuzzer_name,
            replay_binary=replay_binary,
            target_api=str(
                preferred_identity.get("target_api")
                or state.get("coverage_target_api")
                or state.get("selected_target_api")
                or state.get("synthesize_selected_target_api")
                or ""
            ).strip(),
        )
        out = {
            **state,
            "last_step": "per-input-replay",
            "last_error": "",
            "coverage_per_input_manifest_path": replay.manifest_path,
            "coverage_frontier_path": replay.frontier_path,
            "coverage_frontier_summary": dict(replay.frontier_summary or {}),
            "coverage_target_name": str(preferred_identity.get("target_name") or state.get("coverage_target_name") or ""),
            "coverage_target_api": str(preferred_identity.get("target_api") or state.get("coverage_target_api") or ""),
            "coverage_target_type": str(
                preferred_identity.get("target_type")
                or state.get("coverage_target_type")
                or _target_type_from_run_details(
                    run_details,
                    target_name=str(preferred_identity.get("target_name") or state.get("coverage_target_name") or ""),
                    target_api=str(preferred_identity.get("target_api") or state.get("coverage_target_api") or ""),
                    fuzzer_name=fuzzer_name,
                )
                or ""
            ),
            "coverage_replay_runtime_sec": float(replay.runtime_sec),
            "coverage_replay_binary_hash": str(replay.binary_hash or ""),
            "coverage_replay_stage_success": bool(replay.stage_success),
            "coverage_replay_error": str(replay.stage_error or ""),
            "coverage_replay_manifest_fresh_for_current_binary": bool(
                replay.manifest_fresh_for_current_binary
            ),
            "coverage_replay_queue_drained": bool(replay.replay_queue_drained),
            "coverage_replay_pending_inputs": int(replay.pending_inputs),
            "coverage_replay_failed_inputs": int(replay.failed_inputs),
            "coverage_replay_processed_inputs": int(replay.processed_inputs),
            "coverage_replay_total_inputs": int(replay.total_inputs),
            "message": "per-input replay completed",
        }
        _wf_log(
            cast(dict[str, Any], out),
            (
                "<- per-input-replay ok"
                f" stage_success={int(replay.stage_success)}"
                f" pending={int(replay.pending_inputs)}"
                f" failed={int(replay.failed_inputs)}"
                f" dt={_fmt_dt(time.perf_counter()-t0)}"
            ),
        )
        return out
    except Exception as e:
        out = {
            **state,
            "last_step": "per-input-replay",
            "last_error": "",
            "coverage_replay_stage_success": False,
            "coverage_replay_error": f"replay_stage_error:{type(e).__name__}",
            "message": "per-input replay failed",
        }
        _wf_log(cast(dict[str, Any], out), f"<- per-input-replay err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
