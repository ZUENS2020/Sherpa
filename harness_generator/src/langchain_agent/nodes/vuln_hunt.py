"""Carved from workflow_graph.py - '_node_vuln_hunt' LangGraph node."""

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
    _clear_error_markers_on_success,
    _run_vuln_hunt_subphase,
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


def _node_vuln_hunt(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "vuln-hunt")
    if stop_now:
        return state
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> vuln-hunt")
    try:
        out = dict(_run_vuln_hunt_subphase(state))
        out.update(
            {
                "last_step": "vuln-hunt",
                "last_error": "",
                "failed": False,
                "message": "vuln hunt done",
            }
        )
        out = _clear_error_markers_on_success(out)
        _wf_log(cast(dict[str, Any], out), f"<- vuln-hunt ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return cast(FuzzWorkflowRuntimeState, out)
    except Exception as exc:
        # Hunt is advisory. Keep the workflow fail-open but make degradation visible.
        out = {
            **state,
            "last_step": "vuln-hunt",
            "last_error": "",
            "failed": False,
            "message": "vuln hunt degraded",
            "vuln_hunt_enabled": bool(_vuln_hunting_enabled()),
            "vuln_hunt_degraded": True,
            "vuln_hunt_last_reason": f"vuln_hunt_failed:{exc}",
        }
        out = _attach_prompt_render_status(out, issue=f"vuln_hunt_failed:{exc}")
        _wf_log(cast(dict[str, Any], out), f"<- vuln-hunt degraded={exc} dt={_fmt_dt(time.perf_counter()-t0)}")
        return cast(FuzzWorkflowRuntimeState, out)
