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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypedDict, cast

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from persistent_config import load_config

import workflow_common as _wf_common
import workflow_summary as _wf_summary

from fuzz_unharnessed_repo import (
    FuzzerRunResult,
    HarnessGeneratorError,
    NonOssFuzzHarnessGenerator,
    RepoSpec,
    _seed_families_for_target,
    extract_crash_stack_signature,
    snapshot_repo_text,
    write_patch_from_snapshot,
)

_RECOVERABLE_RUN_ERROR_KINDS = {
    "run_no_progress",
    "run_seed_rejected",
    "run_idle_timeout",
    "run_timeout",
    "run_finalize_timeout",
    "run_resource_exhaustion",
}

_FATAL_RUN_ERROR_KINDS = {
    "run_exception",
    "nonzero_exit_without_crash",
    "workflow_time_budget_exceeded",
}


class FuzzWorkflowState(TypedDict, total=False):
    repo_url: str
    email: Optional[str]
    time_budget: int
    run_time_budget: int
    max_len: int
    docker_image: Optional[str]
    ai_key_path: str
    workflow_started_at: float
    resume_from_step: str
    resume_repo_root: str
    stop_after_step: str
    coverage_loop_max_rounds: int
    coverage_loop_round: int
    coverage_should_improve: bool
    coverage_improve_reason: str
    coverage_history: list[dict[str, Any]]
    coverage_target_name: str
    coverage_target_api: str
    coverage_seed_profile: str
    coverage_target_depth_score: int
    coverage_target_depth_class: str
    coverage_selection_bias_reason: str
    coverage_plateau_streak: int
    coverage_last_max_cov: int
    coverage_last_ft: int
    coverage_replan_required: bool
    coverage_replan_effective: bool
    coverage_replan_reason: str
    coverage_improve_mode: str
    coverage_round_budget_exhausted: bool
    coverage_stop_reason: str
    coverage_corpus_sources: list[str]
    coverage_seed_counts: dict[str, int]
    coverage_seed_counts_raw: dict[str, int]
    coverage_seed_counts_filtered: dict[str, int]
    coverage_seed_noise_rejected_count: int
    coverage_missing_execution_targets: list[str]
    coverage_seed_family_coverage: dict[str, Any]
    coverage_seed_feedback: dict[str, Any]
    coverage_harness_feedback: dict[str, Any]
    coverage_quality_oracle: str
    coverage_bottleneck_kind: str
    coverage_bottleneck_reason: str
    coverage_source_report: dict[str, Any]
    coverage_uncovered_functions: list[str]
    coverage_exhausted_targets: list[str]
    coverage_feedback_for_plan: str
    crash_stack_signature: str
    crash_stack_type: str
    crash_stack_top_frames: str
    dry_run_result: dict[str, Any]
    seed_pre_check_result: dict[str, Any]
    antlr_context_path: str
    antlr_context_summary: str
    target_analysis_path: str
    target_analysis_summary: str
    analysis_context_path: str
    analysis_done: bool
    analysis_degraded: bool
    analysis_error: str
    analysis_report_path: str
    analysis_evidence_count: int
    selected_targets_path: str
    execution_plan_path: str
    harness_index_path: str
    repo_understanding_path: str
    build_strategy_path: str
    build_mode: str
    build_target_source: str
    selected_target_api: str
    selected_target_runtime_viability: str
    coverage_seed_quality: dict[str, Any]
    coverage_seed_families_required: list[str]
    coverage_seed_families_covered: list[str]
    coverage_seed_families_missing: list[str]
    coverage_quality_flags: list[str]
    plan_retry_reason: str
    plan_targets_schema_valid_before_retry: bool
    plan_targets_schema_valid_after_retry: bool
    plan_used_fallback_targets: bool
    replan_effective: bool
    replan_stop_reason: str

    step_count: int
    max_steps: int
    last_step: str
    last_error: str
    build_rc: int
    build_stdout_tail: str
    build_stderr_tail: str
    build_full_log_path: str
    build_template_cache_path: str
    build_error_signature: str
    build_error_signature_before: str
    build_error_signature_after: str
    same_build_error_repeats: int
    same_error_max_retries: int
    build_error_kind: str
    build_error_code: str
    build_error_signature_short: str
    build_attempts: int
    fix_build_attempts: int
    max_fix_rounds: int
    fix_build_noop_streak: int
    fix_build_attempt_history: list[dict[str, Any]]
    fix_build_rule_hits: list[str]
    fix_build_terminal_reason: str
    fix_build_last_diff_paths: list[str]
    fix_action_type: str
    fix_effect: str
    codex_hint: str
    failed: bool
    repo_root: str
    run_rc: int
    crash_evidence: str
    run_error_kind: str
    run_terminal_reason: str
    run_idle_seconds: int
    synthesize_selected_target_name: str
    synthesize_selected_target_api: str
    synthesize_observed_target_api: str
    synthesize_observed_harness: str
    synthesize_target_drifted: bool
    synthesize_target_drift_reason: str
    synthesize_target_relation: str
    synthesize_target_runtime_viability: str
    run_children_exit_count: int
    run_details: list[dict[str, Any]]
    run_batch_plan: list[dict[str, Any]]
    last_crash_artifact: str
    last_fuzzer: str
    crash_signature: str
    same_crash_repeats: int
    timeout_signature: str
    same_timeout_repeats: int
    crash_fix_attempts: int
    crash_repro_done: bool
    crash_repro_ok: bool
    crash_repro_rc: int
    crash_repro_report_path: str
    crash_repro_json_path: str
    crash_triage_done: bool
    crash_triage_label: str
    crash_triage_confidence: float
    crash_triage_reason: str
    crash_triage_signal_lines: list[str]
    crash_triage_report_path: str
    crash_triage_json_path: str
    crash_analysis_done: bool
    crash_analysis_verdict: str
    crash_analysis_reason: str
    crash_analysis_report_path: str
    crash_analysis_json_path: str
    re_build_done: bool
    re_build_ok: bool
    re_build_rc: int
    re_build_report_path: str
    re_build_json_path: str
    re_run_done: bool
    re_run_ok: bool
    re_run_rc: int
    re_run_report_path: str
    re_run_json_path: str
    re_workspace_root: str
    restart_to_plan: bool
    restart_to_plan_reason: str
    restart_to_plan_stage: str
    restart_to_plan_error_text: str
    restart_to_plan_report_path: str
    restart_to_plan_count: int
    fix_harness_attempts: int
    next: str
    fix_patch_path: str
    fix_patch_files: list[str]
    fix_patch_bytes: int
    summary_path: str
    summary_json_path: str
    plan_fix_on_crash: bool
    plan_max_fix_rounds: int
    repair_mode: bool
    repair_origin_stage: str
    repair_error_kind: str
    repair_error_code: str
    repair_signature: str
    repair_stdout_tail: str
    repair_stderr_tail: str
    repair_recent_attempts: list[dict[str, Any]]
    repair_error_digest: dict[str, Any]
    repair_attempt_index: int
    repair_strategy_force_change: bool
    target_scoring_enabled: bool
    constraint_memory_count: int
    constraint_memory_path: str
    error: dict[str, Any]


class FuzzWorkflowRuntimeState(FuzzWorkflowState, total=False):
    generator: NonOssFuzzHarnessGenerator
    crash_found: bool
    message: str


def _has_error_payload(err: dict[str, Any] | None) -> bool:
    if not isinstance(err, dict):
        return False
    return bool(
        str(err.get("code") or "").strip()
        or str(err.get("message") or "").strip()
        or bool(err.get("terminal"))
    )


def _coerce_error_payload(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    err = {
        "stage": str(raw.get("stage") or "").strip().lower(),
        "kind": str(raw.get("kind") or "").strip().lower(),
        "code": str(raw.get("code") or "").strip().lower(),
        "message": str(raw.get("message") or "").strip(),
        "detail": str(raw.get("detail") or "").strip(),
        "signature": str(raw.get("signature") or "").strip(),
        "retryable": bool(raw.get("retryable")),
        "terminal": bool(raw.get("terminal")),
        "at": int(raw.get("at") or 0),
    }
    if err["at"] <= 0:
        err["at"] = int(time.time())
    return err


def _derive_error_from_legacy(state: dict[str, Any]) -> dict[str, Any]:
    stage = str(state.get("last_step") or "").strip().lower()
    code = str(
        state.get("build_error_code")
        or state.get("run_error_kind")
        or state.get("restart_to_plan_reason")
        or state.get("error_code")
        or ""
    ).strip().lower()
    kind = str(
        state.get("build_error_kind")
        or state.get("run_error_kind")
        or state.get("repair_error_kind")
        or state.get("error_kind")
        or ""
    ).strip().lower()
    message = str(state.get("last_error") or "").strip()
    if not message and bool(state.get("failed")):
        message = str(state.get("message") or "").strip()
    signature = str(
        state.get("build_error_signature_short")
        or state.get("build_error_signature")
        or state.get("timeout_signature")
        or state.get("crash_signature")
        or state.get("error_signature")
        or ""
    ).strip()
    terminal = bool(state.get("failed"))
    if not code and (message or terminal):
        code = "unknown_error"
    if not kind and code:
        if code.startswith("run_"):
            kind = "run"
        elif code.startswith("build_") or "build" in code:
            kind = "build"
        elif "crash" in code:
            kind = "crash"
        elif "timeout" in code:
            kind = "timeout"
        else:
            kind = "generic_failure"
    retryable = bool(code) and not terminal
    return {
        "stage": stage,
        "kind": kind,
        "code": code,
        "message": message,
        "detail": message,
        "signature": signature,
        "retryable": retryable,
        "terminal": terminal,
        "at": int(time.time()),
    }


def _project_error_legacy_fields(state: dict[str, Any], err: dict[str, Any]) -> dict[str, Any]:
    out = dict(state)
    if not _has_error_payload(err):
        return out
    code = str(err.get("code") or "").strip().lower()
    kind = str(err.get("kind") or "").strip().lower()
    message = str(err.get("message") or "").strip()
    signature = str(err.get("signature") or "").strip()
    stage = str(err.get("stage") or "").strip().lower()
    if message:
        out["last_error"] = message
    out["error_code"] = code
    out["error_kind"] = kind
    if signature:
        out["error_signature"] = signature
    if not str(out.get("repair_error_code") or "").strip() and code:
        out["repair_error_code"] = code
    if not str(out.get("repair_error_kind") or "").strip() and kind:
        out["repair_error_kind"] = kind
    if bool(out.get("restart_to_plan")) and not str(out.get("restart_to_plan_error_text") or "").strip() and message:
        out["restart_to_plan_error_text"] = message
    if stage == "run":
        if code and not str(out.get("run_error_kind") or "").strip():
            out["run_error_kind"] = code
        if code and not str(out.get("run_terminal_reason") or "").strip():
            out["run_terminal_reason"] = code
    if stage == "build" or kind == "build":
        if code and not str(out.get("build_error_code") or "").strip():
            out["build_error_code"] = code
        if kind and not str(out.get("build_error_kind") or "").strip():
            out["build_error_kind"] = kind
        if signature and not str(out.get("build_error_signature_short") or "").strip():
            out["build_error_signature_short"] = signature[:12]
    return out


def _normalize_error_state(state: dict[str, Any]) -> dict[str, Any]:
    out = dict(state)
    existing = _coerce_error_payload(out.get("error"))
    derived = _derive_error_from_legacy(out)
    if _has_error_payload(existing):
        err = {**derived, **existing}
    else:
        err = derived
    if _has_error_payload(err):
        out["error"] = err
        out = _project_error_legacy_fields(out, err)
    else:
        out["error"] = {}
    return out

def _clear_error_markers_on_success(state: dict[str, Any]) -> dict[str, Any]:
    """Clear stale error markers after a stage succeeds.

    This prevents previous recoverable errors (for example an old compile_error)
    from polluting the next stage routing/summary when current stage output is valid.
    """
    out = dict(state)
    out["error"] = {}
    out["last_error"] = ""
    out["error_code"] = ""
    out["error_kind"] = ""
    out["error_signature"] = ""
    out["build_error_kind"] = ""
    out["build_error_code"] = ""
    out["build_error_signature"] = ""
    out["build_error_signature_before"] = str(out.get("build_error_signature_before") or "")
    out["build_error_signature_after"] = ""
    out["build_error_signature_short"] = ""
    out["run_error_kind"] = ""
    out["run_terminal_reason"] = ""
    return out


def _wf_log(state: dict[str, Any] | None, msg: str) -> None:
    _wf_common.wf_log(state, msg)


def _emit_fuzz_metrics(state: dict[str, Any]) -> None:
    """Emit a structured ``[wf-metrics]`` JSON line so that the control-plane
    (main.py) can capture per-fuzzer performance data and expose it via the API.
    """
    run_details = list(state.get("run_details") or [])
    coverage_history = list(state.get("coverage_history") or [])

    # Build per-fuzzer metrics keyed by fuzzer name
    fuzzers: dict[str, dict[str, Any]] = {}
    for detail in run_details:
        name = str(detail.get("fuzzer") or "unknown")
        fuzzers[name] = {
            "fuzzer": name,
            "final_cov": int(detail.get("final_cov") or 0),
            "final_ft": int(detail.get("final_ft") or 0),
            "final_execs_per_sec": int(detail.get("final_execs_per_sec") or 0),
            "final_iteration": int(detail.get("final_iteration") or 0),
            "final_rss_mb": int(detail.get("final_rss_mb") or 0),
            "final_corpus_files": int(detail.get("final_corpus_files") or 0),
            "final_corpus_size_bytes": int(detail.get("final_corpus_size_bytes") or 0),
            "corpus_files": int(detail.get("corpus_files") or 0),
            "corpus_size_bytes": int(detail.get("corpus_size_bytes") or 0),
            "crash_found": bool(detail.get("crash_found")),
            "rc": int(detail.get("rc") or 0),
            "run_error_kind": str(detail.get("run_error_kind") or ""),
            "terminal_reason": str(detail.get("terminal_reason") or ""),
            "plateau_detected": bool(detail.get("plateau_detected")),
            "plateau_idle_seconds": int(detail.get("plateau_idle_seconds") or 0),
            "seed_quality": dict(detail.get("seed_quality") or {}),
        }

    # Aggregate summary
    max_cov = max((f["final_cov"] for f in fuzzers.values()), default=0)
    max_ft = max((f["final_ft"] for f in fuzzers.values()), default=0)
    total_execs = sum(f["final_execs_per_sec"] for f in fuzzers.values())
    any_crash = any(f["crash_found"] for f in fuzzers.values())

    payload = {
        "ts": int(time.time()),
        "stage": str(state.get("last_step") or ""),
        "coverage_loop_round": int(state.get("coverage_loop_round") or 0),
        "coverage_loop_max_rounds": int(state.get("coverage_loop_max_rounds") or 0),
        "max_cov": max_cov,
        "max_ft": max_ft,
        "total_execs_per_sec": total_execs,
        "crash_found": any_crash,
        "fuzzers": fuzzers,
        "coverage_history": coverage_history,
        "coverage_source_report": dict(state.get("coverage_source_report") or {}),
        "coverage_plateau_streak": int(state.get("coverage_plateau_streak") or 0),
        "coverage_seed_profile": str(state.get("coverage_seed_profile") or ""),
        "coverage_quality_flags": list(state.get("coverage_quality_flags") or []),
        "coverage_bottleneck_kind": str(state.get("coverage_bottleneck_kind") or ""),
        "coverage_bottleneck_reason": str(state.get("coverage_bottleneck_reason") or ""),
        "analysis_evidence_count": int(state.get("analysis_evidence_count") or 0),
        "target_scoring_enabled": bool(state.get("target_scoring_enabled") or False),
        "constraint_memory_count": int(state.get("constraint_memory_count") or 0),
    }
    try:
        line = json.dumps(payload, separators=(",", ":"), default=str)
    except Exception:
        return
    logger.info(f"[wf-metrics] {line}", flush=True)


def _fmt_dt(seconds: float) -> str:
    return _wf_common.fmt_dt(seconds)


def _calc_parallel_batch_budget(
    *,
    pending_count: int,
    max_parallel: int,
    remaining_for_run: int,
    configured_run_time_budget: int,
    total_budget_unlimited: bool,
) -> tuple[int, int, int]:
    rounds_left = (pending_count + max_parallel - 1) // max_parallel
    base_round_budget = max(1, remaining_for_run // max(1, rounds_left))
    if configured_run_time_budget <= 0:
        if total_budget_unlimited:
            # Unlimited workflow budgets can still produce pathological multi-hour
            # single-fuzzer runs; cap each run round by default unless explicitly disabled.
            unlimited_round_cap = _run_unlimited_round_budget_sec()
            if unlimited_round_cap <= 0:
                round_budget = 0
                hard_timeout = 0
                return rounds_left, round_budget, hard_timeout
            round_budget = unlimited_round_cap
            hard_timeout = max(60, round_budget + 120)
            return rounds_left, round_budget, hard_timeout
        round_budget = base_round_budget
    else:
        round_budget = min(configured_run_time_budget, base_round_budget)

    if total_budget_unlimited:
        hard_timeout = max(60, round_budget + 120)
    else:
        hard_timeout = min(max(60, round_budget + 120), max(60, remaining_for_run + 30))
    return rounds_left, round_budget, hard_timeout


def _llm_or_none() -> ChatOpenAI | None:
    openai_key = os.environ.get("OPENAI_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    cfg = None
    if not (openai_key or openrouter_key):
        try:
            cfg = load_config()
            openai_key = cfg.openai_api_key or ""
            openrouter_key = cfg.openrouter_api_key or ""
        except Exception:
            cfg = None

    key = (openai_key or openrouter_key or "").strip()
    if not key:
        return None

    if openai_key and openai_key.strip():
        model = (
            os.environ.get("OPENAI_MODEL")
            or os.environ.get("OPENCODE_MODEL")
            or "deepseek-reasoner"
        ).strip()
        base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip()
        if not base_url and cfg is not None:
            base_url = (cfg.openai_base_url or "").strip()
    else:
        model = (os.environ.get("OPENROUTER_MODEL") or "").strip()
        base_url = (os.environ.get("OPENROUTER_BASE_URL") or "").strip()
        if cfg is not None:
            if not model:
                model = (cfg.openrouter_model or "").strip()
            if not base_url:
                base_url = (cfg.openrouter_base_url or "").strip()
        if not model:
            model = "anthropic/claude-3.5-sonnet"
        if not base_url:
            base_url = "https://openrouter.ai/api/v1"

    # NOTE: langchain_openai.ChatOpenAI signature has changed across versions.
    # Build kwargs dynamically to avoid type-checker false positives.
    params: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "max_tokens": 600,
        "timeout": 30,
        "openai_api_key": key.strip(),
        "openai_api_base": base_url,
    }
    return ChatOpenAI(**params)


def _repro_context_path(repo_root: Path) -> Path:
    return repo_root / "repro_context.json"


def _read_repro_context(repo_root: Path) -> dict[str, Any]:
    path = _repro_context_path(repo_root)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_repro_context(
    repo_root: Path,
    *,
    repo_url: str = "",
    last_fuzzer: str = "",
    last_crash_artifact: str = "",
    crash_signature: str = "",
    re_workspace_root: str = "",
) -> None:
    previous = _read_repro_context(repo_root)
    payload = {
        "repo_url": repo_url or str(previous.get("repo_url") or ""),
        "last_fuzzer": last_fuzzer or str(previous.get("last_fuzzer") or ""),
        "last_crash_artifact": last_crash_artifact or str(previous.get("last_crash_artifact") or ""),
        "crash_signature": crash_signature or str(previous.get("crash_signature") or ""),
        "re_workspace_root": re_workspace_root or str(previous.get("re_workspace_root") or ""),
        "updated_at": time.time(),
    }
    try:
        _repro_context_path(repo_root).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def _extract_json_object(text: str) -> dict[str, Any] | None:
    return _wf_common.extract_json_object(text)


def _sha256_text(text: str) -> str:
    return _wf_common.sha256_text(text)


def _validate_targets_json(repo_root: Path) -> tuple[bool, str]:
    return _wf_common.validate_targets_json(repo_root)


def _infer_target_type(*parts: str) -> str:
    text = " ".join(p for p in parts if p).lower()
    if any(tok in text for tok in ("parse", "parser", "scan", "scanner", "yaml", "json", "xml", "token", "lex")):
        return "parser"
    if any(tok in text for tok in ("archive", "untar", "unzip", "tar", "zip", "rar", "7z", "inflate", "deflate", "gzip", "zlib", "lz", "zstd")):
        return "archive"
    if any(tok in text for tok in ("decode", "decoder", "decompress", "unpack")):
        return "decoder"
    if re.search(r"\bread_(?:string|line|token|field|record|key|value)\b", text):
        return "parser"
    if any(tok in text for tok in ("read string", "read_line", "readline", "reader")):
        return "parser"
    if any(tok in text for tok in ("png", "jpeg", "jpg", "gif", "bmp", "image", "pixel")):
        return "image"
    if any(tok in text for tok in ("pdf", "doc", "document", "html", "markdown")):
        return "document"
    if any(tok in text for tok in ("socket", "packet", "http", "tls", "dns", "frame", "request", "response")):
        return "network"
    if any(tok in text for tok in ("sql", "query", "db", "database", "sqlite", "record")):
        return "database"
    if any(tok in text for tok in ("emit", "dump", "serialize", "serializer", "write")):
        return "serializer"
    if any(tok in text for tok in ("eval", "vm", "execute", "compile", "bytecode", "script", "interp")):
        return "interpreter"
    return "generic"


def _opencode_done_path(repo_root: Path) -> Path:
    return repo_root / "done"


def _opencode_feedback_dir(repo_root: Path) -> Path:
    return repo_root / ".git" / "sherpa-opencode" / "feedback"


def _feedback_group_for_stage(stage: str) -> str:
    s = str(stage or "").strip().lower()
    if s in {
        "plan",
        "plan_fix_targets_schema",
        "synthesize",
        "synthesize_complete_scaffold",
        "plan_repair_build",
        "synthesize_repair_build",
        "plan_repair_crash",
        "synthesize_repair_crash",
    }:
        return "planning_synth"
    if s == "fix_build":
        return "fix_build"
    if s in {"crash_triage", "fix_harness_after_run"}:
        return "crash_triage"
    if s in {"fix_crash_harness_error", "fix_crash_upstream_bug"}:
        return "fix_crash"
    return s or "default"


def _feedback_file_for_stage(repo_root: Path, stage: str) -> Path:
    safe = re.sub(r"[^a-z0-9_.-]+", "-", str(stage or "unknown").strip().lower()).strip("-") or "unknown"
    return _opencode_feedback_dir(repo_root) / f"{safe}.md"


def _feedback_text_limits() -> tuple[int, int]:
    raw_lines = (os.environ.get("SHERPA_OPENCODE_FEEDBACK_MAX_LINES") or "50").strip()
    raw_chars = (os.environ.get("SHERPA_OPENCODE_FEEDBACK_MAX_CHARS") or "6000").strip()
    try:
        max_lines = max(20, min(int(raw_lines), 600))
    except Exception:
        max_lines = 50
    try:
        max_chars = max(512, min(int(raw_chars), 200000))
    except Exception:
        max_chars = 6000
    return max_lines, max_chars


def _trim_feedback_text(text: str) -> str:
    src = str(text or "").strip()
    if not src:
        return ""
    max_lines, max_chars = _feedback_text_limits()
    lines = src.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    out = "\n".join(lines).strip()
    if len(out) > max_chars:
        out = out[-max_chars:].lstrip()
    return out


def _write_stage_feedback(
    repo_root: Path,
    *,
    stage: str,
    error_text: str,
    state: dict[str, Any] | None = None,
) -> str:
    state = _normalize_error_state(state or {})
    err = dict(state.get("error") or {})
    parts: list[str] = [
        f"# Stage Failure Feedback: {stage}",
        "",
        f"- stage: {stage}",
        f"- group: {_feedback_group_for_stage(stage)}",
        f"- ts: {int(time.time())}",
    ]
    for k in ("restart_to_plan_reason", "build_error_kind", "build_error_code", "run_error_kind"):
        v = str(state.get(k) or "").strip()
        if v:
            parts.append(f"- {k}: {v}")
    structured = {
        "stage": str(stage or "").strip(),
        "error_code": str(
            err.get("code")
            or state.get("build_error_code")
            or state.get("run_error_kind")
            or state.get("restart_to_plan_reason")
            or ""
        ).strip(),
        "signature": str(err.get("signature") or state.get("build_error_signature_short") or "").strip(),
        "action_taken": str(state.get("fix_action_type") or "").strip(),
        "diff_paths": list(state.get("fix_build_last_diff_paths") or []),
    }
    parts.extend(
        [
            "",
            "## Structured Summary",
            "",
            "```json",
            json.dumps(structured, ensure_ascii=False, indent=2),
            "```",
        ]
    )
    err = _trim_feedback_text(error_text)
    if err:
        parts.extend(["", "## Error", "", "```text", err, "```"])
    stdout_tail = _trim_feedback_text(str(state.get("build_stdout_tail") or ""))
    stderr_tail = _trim_feedback_text(str(state.get("build_stderr_tail") or ""))
    if stdout_tail:
        parts.extend(["", "## Build Stdout Tail", "", "```text", stdout_tail, "```"])
    if stderr_tail:
        parts.extend(["", "## Build Stderr Tail", "", "```text", stderr_tail, "```"])
    body = "\n".join(parts).strip() + "\n"
    path = _feedback_file_for_stage(repo_root, stage)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8", errors="replace")
        return str(path)
    except Exception:
        return ""


def _collect_feedback_for_group(repo_root: Path, group: str, *, limit: int = 3) -> str:
    group_name = str(group or "").strip().lower()
    stage_groups = {
        "plan": _feedback_group_for_stage("plan"),
        "plan_fix_targets_schema": _feedback_group_for_stage("plan_fix_targets_schema"),
        "synthesize": _feedback_group_for_stage("synthesize"),
        "synthesize_complete_scaffold": _feedback_group_for_stage("synthesize_complete_scaffold"),
        "plan_repair_build": _feedback_group_for_stage("plan_repair_build"),
        "synthesize_repair_build": _feedback_group_for_stage("synthesize_repair_build"),
        "plan_repair_crash": _feedback_group_for_stage("plan_repair_crash"),
        "synthesize_repair_crash": _feedback_group_for_stage("synthesize_repair_crash"),
        "fix_build": _feedback_group_for_stage("fix_build"),
        "crash_triage": _feedback_group_for_stage("crash_triage"),
        "fix_harness_after_run": _feedback_group_for_stage("fix_harness_after_run"),
        "fix_crash_harness_error": _feedback_group_for_stage("fix_crash_harness_error"),
        "fix_crash_upstream_bug": _feedback_group_for_stage("fix_crash_upstream_bug"),
    }
    picked: list[Path] = []
    for stage, g in stage_groups.items():
        if g != group_name:
            continue
        p = _feedback_file_for_stage(repo_root, stage)
        if p.is_file():
            picked.append(p)
    if not picked:
        return ""
    picked.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    texts: list[str] = []
    for p in picked[: max(1, int(limit))]:
        try:
            txt = p.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            txt = ""
        if txt:
            texts.append(f"=== {p.name} ===\n{_trim_feedback_text(txt)}")
    return "\n\n".join(texts).strip()


def _clear_opencode_done_sentinel(repo_root: Path) -> bool:
    done_path = _opencode_done_path(repo_root)
    if not done_path.exists():
        return False
    try:
        done_path.unlink()
        return True
    except Exception:
        return False


def _infer_repair_origin_stage(state: dict[str, Any]) -> str:
    explicit = str(state.get("repair_origin_stage") or "").strip().lower()
    if explicit in {"build", "crash", "coverage"}:
        return explicit
    restart_stage = str(state.get("restart_to_plan_stage") or "").strip().lower()
    if restart_stage == "build":
        return "build"
    if restart_stage in {"run", "crash-triage", "fix-harness", "re-build", "re-run", "fix_crash"}:
        return "crash"
    if restart_stage in {"coverage-analysis", "improve-harness"}:
        return "coverage"
    last_step = str(state.get("last_step") or "").strip().lower()
    if last_step == "build":
        return "build"
    if last_step in {"run", "crash-triage", "fix-harness", "re-build", "re-run", "fix_crash"}:
        return "crash"
    if last_step in {"coverage-analysis", "improve-harness"}:
        return "coverage"
    if bool(state.get("crash_found")):
        return "crash"
    return "build"


def _repair_mode_active(state: dict[str, Any]) -> bool:
    state = _normalize_error_state(state)
    err = dict(state.get("error") or {})
    if bool(state.get("repair_mode")):
        return True
    if bool(state.get("restart_to_plan")):
        return True
    return _has_error_payload(err) or bool(str(state.get("last_error") or "").strip())


def _constraint_memory_path(repo_root: Path) -> Path:
    return repo_root / "fuzz" / "constraint_memory.json"


def _constraint_repeat_threshold() -> int:
    raw = (os.environ.get("SHERPA_CONSTRAINT_MEMORY_REPEAT_THRESHOLD") or "2").strip()
    try:
        return max(2, min(int(raw), 10))
    except Exception:
        return 2


def _load_constraint_memory(repo_root: Path) -> dict[str, Any]:
    path = _constraint_memory_path(repo_root)
    if not path.is_file():
        return {"schema_version": 1, "updated_at": 0, "entries": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {"schema_version": 1, "updated_at": 0, "entries": {}}
    if not isinstance(raw, dict):
        return {"schema_version": 1, "updated_at": 0, "entries": {}}
    entries = raw.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    return {
        "schema_version": 1,
        "updated_at": int(raw.get("updated_at") or 0),
        "entries": entries,
    }


def _write_constraint_memory(repo_root: Path, doc: dict[str, Any]) -> str:
    path = _constraint_memory_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "updated_at": int(time.time()),
        "entries": dict(doc.get("entries") or {}),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def _constraint_fix_hint(label_or_verdict: str) -> str:
    key = str(label_or_verdict or "").strip().lower()
    if key in {"harness_bug", "false_positive"}:
        return "Validate harness preconditions and replace brittle behavior with stable public API usage."
    if key in {"upstream_bug", "real_bug"}:
        return "Keep reproducer stable and preserve upstream crash evidence for triage/reporting."
    return "Collect stronger evidence before committing a narrow fix."


def _record_constraint_memory_observation(
    *,
    repo_root: Path,
    signature: str,
    stage: str,
    classification: str,
    reason: str,
    evidence: list[str],
    confidence: float,
    repeats: int,
) -> tuple[int, str, dict[str, Any]]:
    signature_key = str(signature or "").strip()
    doc = _load_constraint_memory(repo_root)
    entries = dict(doc.get("entries") or {})
    if not signature_key or repeats < _constraint_repeat_threshold():
        return len(entries), str(_constraint_memory_path(repo_root)), {}
    now = int(time.time())
    prev = dict(entries.get(signature_key) or {})
    entry = {
        "signature": signature_key,
        "source_stage": str(stage or "").strip() or "unknown",
        "classification": str(classification or "").strip() or "unknown",
        "reason": str(reason or "").strip()[:1024],
        "evidence": [str(x).strip()[:512] for x in list(evidence or []) if str(x).strip()][:12],
        "confidence": max(0.0, min(float(confidence), 1.0)),
        "suspected_precondition": str(reason or "").strip()[:512],
        "fix_hint": _constraint_fix_hint(classification),
        "first_seen": int(prev.get("first_seen") or now),
        "latest_seen": now,
        "occurrence_count": int(prev.get("occurrence_count") or 0) + 1,
    }
    entries[signature_key] = entry
    _write_constraint_memory(repo_root, {"entries": entries})
    return len(entries), str(_constraint_memory_path(repo_root)), entry


def _constraint_memory_snapshot_from_state(state: dict[str, Any]) -> tuple[dict[str, Any], int, str]:
    repo_root_text = str(state.get("repo_root") or "").strip()
    if not repo_root_text:
        return {}, 0, ""
    try:
        repo_root = Path(repo_root_text)
    except Exception:
        return {}, 0, ""
    doc = _load_constraint_memory(repo_root)
    entries = dict(doc.get("entries") or {})
    if not entries:
        return {}, 0, str(_constraint_memory_path(repo_root))
    candidates = [
        str(state.get("repair_signature") or "").strip(),
        str(state.get("crash_signature") or "").strip(),
        str(state.get("timeout_signature") or "").strip(),
    ]
    for sig in candidates:
        if sig and sig in entries and isinstance(entries[sig], dict):
            return dict(entries[sig]), len(entries), str(_constraint_memory_path(repo_root))
    latest_entry: dict[str, Any] = {}
    latest_ts = 0
    for raw in entries.values():
        if not isinstance(raw, dict):
            continue
        ts = int(raw.get("latest_seen") or 0)
        if ts >= latest_ts:
            latest_ts = ts
            latest_entry = dict(raw)
    return latest_entry, len(entries), str(_constraint_memory_path(repo_root))


def _build_repair_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    state = _normalize_error_state(state)
    err = dict(state.get("error") or {})
    origin = _infer_repair_origin_stage(state)
    error_text = (
        str(state.get("restart_to_plan_error_text") or "").strip()
        or str(err.get("message") or "").strip()
        or str(state.get("last_error") or "").strip()
    )
    snapshot = {
        "repair_mode": _repair_mode_active(state),
        "repair_origin_stage": origin,
        "repair_error_kind": str(
            err.get("kind")
            or state.get("repair_error_kind")
            or state.get("build_error_kind")
            or state.get("run_error_kind")
            or "generic_failure"
        ).strip() or "generic_failure",
        "repair_error_code": str(
            err.get("code")
            or state.get("repair_error_code")
            or state.get("build_error_code")
            or state.get("restart_to_plan_reason")
            or ""
        ).strip(),
        "repair_signature": str(
            err.get("signature")
            or state.get("repair_signature")
            or state.get("build_error_signature_short")
            or state.get("crash_signature")
            or ""
        ).strip(),
        "repair_stdout_tail": str(state.get("repair_stdout_tail") or state.get("build_stdout_tail") or "").strip(),
        "repair_stderr_tail": str(state.get("repair_stderr_tail") or state.get("build_stderr_tail") or "").strip(),
        "repair_error_text": error_text,
        "repair_recent_attempts": list(state.get("repair_recent_attempts") or []),
        "repair_attempt_index": int(state.get("repair_attempt_index") or 0),
        "repair_strategy_force_change": bool(state.get("repair_strategy_force_change") or False),
        "repair_error_digest": dict(state.get("repair_error_digest") or {}),
    }
    constraint_entry, constraint_count, constraint_path = _constraint_memory_snapshot_from_state(state)
    snapshot["constraint_memory_entry"] = constraint_entry
    snapshot["constraint_memory_count"] = int(constraint_count)
    snapshot["constraint_memory_path"] = constraint_path
    return snapshot


def _infer_target_lang_from_repo(repo_root: Path, *, file_hint: str = "") -> str:
    hint = file_hint.lower()
    if hint.endswith(".java"):
        return "java"
    try:
        for p in repo_root.rglob("*"):
            if not p.is_file():
                continue
            suffix = p.suffix.lower()
            if suffix == ".java":
                return "java"
            if suffix in {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}:
                return "c-cpp"
    except Exception:
        pass
    return "c-cpp"


def _infer_seed_profile(name: str, context: str, *, target_type: str) -> str:
    text = f"{name}\n{context}".lower()
    if target_type == "parser":
        if any(tok in text for tok in ("arg_id", "argument id", "positional", "named argument", "named arg", "number", "numeric")):
            return "parser-numeric"
        if any(tok in text for tok in ("format", "replacement field", "specifier", "brace", "printf", "fmt")):
            return "parser-format"
        if any(tok in text for tok in ("token", "lexer", "lex", "scan", "scanner", "read_", "readline", "read line")):
            return "parser-token"
        return "parser-structure"
    mapping = {
        "decoder": "decoder-binary",
        "archive": "archive-container",
        "serializer": "serializer-structured",
        "document": "document-text",
        "network": "network-message",
    }
    return mapping.get(target_type, "generic")


def _score_target_depth(
    name: str,
    context: str,
    *,
    target_type: str,
    risk_signals: list[str] | None = None,
) -> tuple[int, str, str]:
    text = f"{name}\n{context}".lower()
    score = 0
    reasons: list[str] = []
    positive_weights = {
        "parse": 5,
        "parser": 5,
        "scan": 4,
        "scanner": 5,
        "decode": 5,
        "inflate": 5,
        "deflate": 4,
        "read": 3,
        "load": 3,
        "stream": 3,
        "archive": 4,
        "reader": 4,
        "container": 4,
        "process": 2,
        "consume": 3,
    }
    negative_weights = {
        "adler": -7,
        "crc": -6,
        "hash": -5,
        "checksum": -6,
        "bound": -5,
        "combine": -5,
        "version": -4,
        "copy": -3,
        "helper": -4,
        "util": -3,
        "utility": -3,
    }
    for token, weight in positive_weights.items():
        if token in text:
            score += weight
            reasons.append(f"+{token}")
    for token, weight in negative_weights.items():
        if token in text:
            score += weight
            reasons.append(token)
    if target_type in {"parser", "decoder", "archive", "document"}:
        score += 4
        reasons.append(f"type:{target_type}")
    elif target_type in {"serializer", "network"}:
        score += 2
        reasons.append(f"type:{target_type}")
    signals = list(risk_signals or [])
    score += min(len(signals), 4)
    if "state-machine" in signals:
        score += 2
        reasons.append("state-machine")
    if "parser-like" in signals:
        score += 2
        reasons.append("parser-like")
    if score >= 8:
        depth_class = "deep"
    elif score >= 3:
        depth_class = "medium"
    else:
        depth_class = "shallow"
    return score, depth_class, ", ".join(reasons[:5]) or "neutral"


def _runtime_viability_details(name: str, context: str, *, file_hint: str = "") -> tuple[str, str, list[str]]:
    text = f"{name}\n{context}\n{file_hint}".lower()
    reasons: list[str] = []
    replacements: list[str] = []
    score = 0
    if any(tok in text for tok in ("test/fuzzing", "/fuzz", "fuzzing", "oss-fuzz")):
        score += 4
        reasons.append("existing-fuzz-infra")
    if any(tok in text for tok in ("println", "logger.info(", "format_to", "vformat", "fmt::format", "fmt::print", "fmt::println")):
        score += 5
        reasons.append("public-runtime-api")
    if any(tok in text for tok in ("fmt/compile.h", "fmt::compile::", " constexpr", "consteval")):
        score -= 8
        reasons.append("compile-time-only")
        replacements.extend(["fmt::println", "fmt::print", "fmt::format_to", "fmt::vformat", "fmt::format"])
    if any(tok in text for tok in ("fmt::detail::", "/detail/", " detail::")):
        score -= 5
        reasons.append("detail-helper")
        replacements.extend(["fmt::println", "fmt::print", "fmt::format_to", "fmt::vformat"])
    if any(tok in text for tok in ("helper", "setter", "set_", "value(", " arg_mapper", " container", " map_")):
        score -= 3
        reasons.append("helper-like")
    if any(tok in text for tok in ("parse_", "parser", "replacement_field", "arg_id")) and "fmt" in text:
        score -= 2
        reasons.append("fmt-parser-helper")
        replacements.extend(["fmt::format_to", "fmt::vformat", "fmt::println"])
    if score >= 4:
        viability = "high"
    elif score >= 0:
        viability = "medium"
    else:
        viability = "low"
    seen: set[str] = set()
    deduped = []
    for item in replacements:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    rationale = ", ".join(reasons[:5]) or "neutral-runtime-signal"
    return viability, rationale, deduped


def _load_targets_doc(repo_root: Path) -> list[dict[str, Any]]:
    targets_path = repo_root / "fuzz" / "targets.json"
    if not targets_path.is_file():
        return []
    try:
        data = json.loads(targets_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _select_primary_target(repo_root: Path) -> dict[str, Any]:
    targets = _load_targets_doc(repo_root)
    return dict(targets[0]) if targets else {}


def _selected_targets_path(repo_root: Path) -> Path:
    return repo_root / "fuzz" / "selected_targets.json"


def _execution_plan_path(repo_root: Path) -> Path:
    return repo_root / "fuzz" / "execution_plan.json"


def _harness_index_path(repo_root: Path) -> Path:
    return repo_root / "fuzz" / "harness_index.json"


def _observed_target_path(repo_root: Path) -> Path:
    return repo_root / "fuzz" / "observed_target.json"


def _execution_targets_max() -> int:
    raw = (os.environ.get("SHERPA_EXECUTION_TARGETS_MAX") or "3").strip()
    try:
        return max(1, min(int(raw), 8))
    except Exception:
        return 3


def _execution_targets_min_required() -> int:
    raw = (os.environ.get("SHERPA_EXECUTION_TARGETS_MIN_REQUIRED") or "2").strip()
    try:
        return max(1, min(int(raw), 8))
    except Exception:
        return 2


def _runtime_viability_rank(value: str) -> int:
    lowered = str(value or "").strip().lower()
    if lowered == "high":
        return 2
    if lowered == "medium":
        return 1
    return 0


def _target_scoring_weights() -> dict[str, float]:
    return {
        "coverage_gap": 0.35,
        "complexity": 0.25,
        "api_relevance": 0.25,
        "consumer_order_support": 0.15,
    }


def _clamp_score(value: float, *, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, float(value)))


def _target_component_coverage_gap(item: dict[str, Any]) -> float:
    explicit = item.get("coverage_gap")
    if explicit is not None:
        try:
            return _clamp_score(float(explicit))
        except Exception:
            pass
    depth_score = max(0, int(item.get("depth_score") or 0))
    depth_class = str(item.get("depth_class") or "").strip().lower()
    target_type = str(item.get("target_type") or "").strip().lower()
    base = min(7.0, float(depth_score) / 2.0)
    if depth_class == "deep":
        base += 2.0
    elif depth_class == "medium":
        base += 1.0
    if target_type in {"parser", "decoder", "archive"}:
        base += 1.0
    return _clamp_score(base)


def _target_component_complexity(item: dict[str, Any]) -> float:
    depth_score = max(0, int(item.get("depth_score") or 0))
    risk_signals = list(item.get("risk_signals") or [])
    base = min(8.0, float(depth_score) / 3.0)
    base += min(2.0, 0.4 * float(len(risk_signals)))
    return _clamp_score(base)


def _target_component_api_relevance(item: dict[str, Any]) -> float:
    runtime_rank = _runtime_viability_rank(str(item.get("runtime_viability") or ""))
    target_type = str(item.get("target_type") or "").strip().lower()
    api = str(item.get("api") or "")
    score = 2.0 + float(runtime_rank) * 2.5
    if target_type in {"parser", "decoder", "archive"}:
        score += 1.5
    if "::" in api or re.search(r"[A-Za-z_][A-Za-z0-9_]*", api):
        score += 1.0
    return _clamp_score(score)


def _target_component_consumer_order_support(item: dict[str, Any]) -> float:
    target_type = str(item.get("target_type") or "").strip().lower()
    rationale = str(item.get("selection_rationale") or "").lower()
    bias = str(item.get("selection_bias_reason") or "").lower()
    signals = " ".join(str(x).lower() for x in (item.get("risk_signals") or []))
    score = 2.0
    if target_type in {"parser", "archive", "decoder"}:
        score += 2.0
    if any(tok in rationale for tok in ("runtime", "entrypoint", "stream", "state")):
        score += 2.0
    if any(tok in bias for tok in ("state", "parse", "decode", "deep")):
        score += 1.5
    if "state-machine" in signals or "parser-like" in signals:
        score += 1.5
    return _clamp_score(score)


def _target_score_breakdown(item: dict[str, Any]) -> dict[str, Any]:
    weights = _target_scoring_weights()
    coverage_gap = _target_component_coverage_gap(item)
    complexity = _target_component_complexity(item)
    api_relevance = _target_component_api_relevance(item)
    consumer_order_support = _target_component_consumer_order_support(item)
    weighted_total = (
        coverage_gap * float(weights["coverage_gap"])
        + complexity * float(weights["complexity"])
        + api_relevance * float(weights["api_relevance"])
        + consumer_order_support * float(weights["consumer_order_support"])
    )
    return {
        "coverage_gap": round(coverage_gap, 4),
        "complexity": round(complexity, 4),
        "api_relevance": round(api_relevance, 4),
        "consumer_order_support": round(consumer_order_support, 4),
        "weights": {k: round(float(v), 4) for k, v in weights.items()},
        "weighted_total": round(weighted_total, 6),
    }


def _build_selected_targets_doc(repo_root: Path) -> list[dict[str, Any]]:
    ranked_items: list[dict[str, Any]] = []
    for item in _load_targets_doc(repo_root):
        target_name = str(item.get("name") or "").strip()
        api = str(item.get("api") or target_name).strip()
        target_type = str(item.get("target_type") or "generic").strip().lower()
        seed_profile = str(item.get("seed_profile") or "generic").strip().lower()
        required, optional = _seed_families_for_target(seed_profile, target_name, api)
        runtime_viability = str(item.get("runtime_viability") or "").strip().lower()
        selection_rationale = str(item.get("selection_rationale") or "").strip()
        runtime_replacement_candidates = list(item.get("runtime_replacement_candidates") or [])
        if not runtime_viability:
            runtime_viability, auto_rationale, auto_replacements = _runtime_viability_details(
                target_name,
                api,
                file_hint=str(item.get("file") or ""),
            )
            selection_rationale = selection_rationale or auto_rationale
            runtime_replacement_candidates = runtime_replacement_candidates or auto_replacements
        scoring_source = {
            "api": api,
            "target_type": target_type,
            "depth_score": int(item.get("depth_score") or 0),
            "depth_class": str(item.get("depth_class") or ""),
            "selection_bias_reason": str(item.get("selection_bias_reason") or ""),
            "runtime_viability": runtime_viability,
            "selection_rationale": selection_rationale,
            "risk_signals": list(item.get("risk_signals") or []),
            "coverage_gap": item.get("coverage_gap"),
        }
        score_breakdown = _target_score_breakdown(scoring_source)
        ranked_items.append(
            {
                "target_name": target_name,
                "name": target_name,
                "api": api,
                "lang": str(item.get("lang") or ""),
                "target_type": target_type,
                "seed_profile": seed_profile,
                "depth_score": int(item.get("depth_score") or 0),
                "depth_class": str(item.get("depth_class") or ""),
                "selection_bias_reason": str(item.get("selection_bias_reason") or ""),
                "runtime_viability": runtime_viability,
                "selection_rationale": selection_rationale,
                "runtime_replacement_candidates": runtime_replacement_candidates,
                "seed_families_required": required,
                "seed_families_optional": optional,
                "wrapper_fuzzer_name": str(item.get("wrapper_fuzzer_name") or ""),
                "target_score_breakdown": score_breakdown,
                "target_score": float(score_breakdown.get("weighted_total") or 0.0),
                "target_scoring_enabled": True,
            }
        )
    ranked_items.sort(
        key=lambda row: (
            -float(row.get("target_score") or 0.0),
            -int(row.get("depth_score") or 0),
            -_runtime_viability_rank(str(row.get("runtime_viability") or "")),
            str(row.get("target_name") or ""),
        )
    )
    out: list[dict[str, Any]] = []
    max_targets = _execution_targets_max()
    for idx, row in enumerate(ranked_items):
        row["execution_priority"] = int(idx + 1) if idx < max_targets else 0
        target_type = str(row.get("target_type") or "").strip().lower()
        row["must_run"] = bool(
            idx < max_targets and (idx == 0 or target_type in {"archive", "parser", "decoder"})
        )
        out.append(row)
    return out


def _write_selected_targets_doc(repo_root: Path) -> tuple[str, list[dict[str, Any]]]:
    path = _selected_targets_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = _build_selected_targets_doc(repo_root)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path), doc


def _load_selected_targets_doc(repo_root: Path) -> list[dict[str, Any]]:
    path = _selected_targets_path(repo_root)
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _build_execution_plan_doc(repo_root: Path, selected_doc: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    selected = list(selected_doc or _load_selected_targets_doc(repo_root))
    max_targets = _execution_targets_max()
    min_required = _execution_targets_min_required()
    execution_targets: list[dict[str, Any]] = []
    for item in selected:
        prio = int(item.get("execution_priority") or 0)
        if prio <= 0 or prio > max_targets:
            continue
        target_name = str(item.get("target_name") or item.get("name") or "").strip()
        expected_bin = str(item.get("wrapper_fuzzer_name") or target_name).strip()
        execution_targets.append(
            {
                "target_name": target_name,
                "expected_fuzzer_name": expected_bin,
                "api": str(item.get("api") or "").strip(),
                "seed_profile": str(item.get("seed_profile") or "").strip(),
                "target_type": str(item.get("target_type") or "").strip(),
                "must_run": bool(item.get("must_run") or False),
                "execution_priority": prio,
            }
        )
    execution_targets.sort(key=lambda row: int(row.get("execution_priority") or 999))
    execution_targets = execution_targets[:max_targets]
    required_floor = max(1, min_required)
    required_built = min(max(required_floor, 1), max(1, len(execution_targets))) if execution_targets else 1
    return {
        "schema_version": 1,
        "max_targets": max_targets,
        "min_required_built_targets": required_built,
        "execution_targets": execution_targets,
    }


def _write_execution_plan_doc(repo_root: Path, selected_doc: list[dict[str, Any]] | None = None) -> tuple[str, dict[str, Any]]:
    path = _execution_plan_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = _build_execution_plan_doc(repo_root, selected_doc=selected_doc)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path), doc


def _load_execution_plan_doc(repo_root: Path) -> dict[str, Any]:
    path = _execution_plan_path(repo_root)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return dict(raw) if isinstance(raw, dict) else {}


def _discover_harness_sources(repo_root: Path) -> list[Path]:
    fuzz_dir = repo_root / "fuzz"
    if not fuzz_dir.is_dir():
        return []
    out: list[Path] = []
    try:
        for p in fuzz_dir.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(fuzz_dir).as_posix()
            if (
                rel.startswith("out/")
                or rel.startswith("corpus/")
                or rel.startswith("build-work/")
                or "/CMakeFiles/" in rel
            ):
                continue
            if p.suffix.lower() in {".c", ".cc", ".cpp", ".cxx", ".java"}:
                out.append(p)
    except Exception:
        return []
    return sorted(out)


def _normalize_exec_target_token(value: str) -> str:
    s = str(value or "").strip().lower()
    if not s:
        return ""
    s = Path(s).name
    s = re.sub(r"\.(?:c|cc|cpp|cxx|java)$", "", s)
    s = re.sub(r"_fuzz(?:er)?$", "", s)
    s = re.sub(r"[^a-z0-9_]+", "_", s).strip("_")
    return s


def _token_overlap_ratio(a: str, b: str) -> float:
    """Return the ratio of overlapping character trigrams between two strings.
    Used as a lightweight fuzzy match for target-to-harness name mapping."""
    if not a or not b:
        return 0.0
    trigrams_a = {a[i:i+3] for i in range(max(1, len(a) - 2))}
    trigrams_b = {b[i:i+3] for i in range(max(1, len(b) - 2))}
    if not trigrams_a or not trigrams_b:
        return 0.0
    overlap = len(trigrams_a & trigrams_b)
    return float(overlap) / float(max(len(trigrams_a), len(trigrams_b)))


def _build_harness_index_doc(repo_root: Path, execution_plan_doc: dict[str, Any] | None = None) -> dict[str, Any]:
    execution_plan = dict(execution_plan_doc or _load_execution_plan_doc(repo_root))
    execution_targets = [
        item for item in list(execution_plan.get("execution_targets") or [])
        if isinstance(item, dict)
    ]
    sources = _discover_harness_sources(repo_root)
    by_norm: dict[str, str] = {}
    all_harness_rel: list[str] = []
    for src in sources:
        rel = src.relative_to(repo_root).as_posix()
        all_harness_rel.append(rel)
        norm = _normalize_exec_target_token(src.stem)
        if norm and norm not in by_norm:
            by_norm[norm] = rel

    mappings: list[dict[str, Any]] = []
    missing_targets: list[str] = []
    used_sources: set[str] = set()
    for item in execution_targets:
        target_name = str(item.get("target_name") or "").strip()
        expected = str(item.get("expected_fuzzer_name") or target_name).strip()
        api = str(item.get("api") or "").strip()
        candidates: list[tuple[str, str]] = [
            (_normalize_exec_target_token(expected), "expected_fuzzer_name"),
            (_normalize_exec_target_token(target_name), "target_name"),
            (_normalize_exec_target_token(api), "api"),
        ]
        source_path = ""
        matched_by = ""
        # Phase 1: exact normalized match
        for normalized, origin in candidates:
            if not normalized:
                continue
            found = by_norm.get(normalized)
            if found:
                source_path = found
                matched_by = origin
                break
        # Phase 2: substring/contains/prefix/fuzzy fallback match
        # Handles cases where harness name is related but not identical
        # (e.g., target="inflateBack9" but harness="infback9_fuzz.c",
        #  or target="decode" when harness is "blast_fuzz.c" from blast API)
        if not source_path:
            best_score = 0.0
            best_src = ""
            best_origin = ""
            for normalized, origin in candidates:
                if not normalized or len(normalized) < 3:
                    continue
                for norm_key, src_rel in by_norm.items():
                    if src_rel in used_sources:
                        continue
                    score = 0.0
                    # Exact substring match
                    if normalized in norm_key or norm_key in normalized:
                        score = 0.8
                    else:
                        # Shared prefix (at least 3 chars)
                        prefix_len = 0
                        for i in range(min(len(normalized), len(norm_key))):
                            if normalized[i] == norm_key[i]:
                                prefix_len += 1
                            else:
                                break
                        if prefix_len >= 3:
                            score = max(score, 0.3 + 0.4 * (prefix_len / max(len(normalized), len(norm_key))))
                        # Trigram overlap
                        overlap = _token_overlap_ratio(normalized, norm_key)
                        score = max(score, overlap)
                    if score > best_score and score >= 0.35:
                        best_score = score
                        best_src = src_rel
                        best_origin = f"{origin}(fuzzy:{score:.2f})"
            if best_src:
                source_path = best_src
                matched_by = best_origin
        if source_path:
            used_sources.add(source_path)
        else:
            label = target_name or expected or api
            if label:
                missing_targets.append(label)
        mappings.append(
            {
                "target_name": target_name,
                "expected_fuzzer_name": expected,
                "api": api,
                "must_run": bool(item.get("must_run") or False),
                "source_path": source_path,
                "matched_by": matched_by,
            }
        )

    extra_harnesses = [rel for rel in all_harness_rel if rel not in used_sources]

    # Phase 3: positional fallback — when we have exactly as many unmatched
    # targets as extra harnesses, pair them by order (best-effort).
    # This handles cases where the AI chose completely different names but
    # the target count matches the harness count.
    if missing_targets and extra_harnesses and len(missing_targets) == len(extra_harnesses):
        for i, label in enumerate(list(missing_targets)):
            fallback_src = extra_harnesses[i]
            for m in mappings:
                tname = m.get("target_name") or m.get("expected_fuzzer_name") or ""
                if tname == label and not m.get("source_path"):
                    m["source_path"] = fallback_src
                    m["matched_by"] = "positional_fallback"
                    used_sources.add(fallback_src)
                    break
        missing_targets = [
            label for label in missing_targets
            if not any(
                m.get("source_path") and (m.get("target_name") == label or m.get("expected_fuzzer_name") == label)
                for m in mappings
            )
        ]
        extra_harnesses = [rel for rel in all_harness_rel if rel not in used_sources]

    return {
        "schema_version": 1,
        "execution_plan_path": _execution_plan_path(repo_root).relative_to(repo_root).as_posix(),
        "mappings": mappings,
        "missing_targets": missing_targets,
        "extra_harnesses": extra_harnesses,
    }


def _write_harness_index_doc(repo_root: Path, execution_plan_doc: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    path = _harness_index_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = _build_harness_index_doc(repo_root, execution_plan_doc=execution_plan_doc)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path), doc


def _load_harness_index_doc(repo_root: Path) -> dict[str, Any]:
    path = _harness_index_path(repo_root)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return dict(raw) if isinstance(raw, dict) else {}


def _write_observed_target_doc(
    repo_root: Path,
    *,
    expected_target_name: str,
    expected_api: str,
    observed_api: str,
    observed_harness: str,
    drifted: bool,
    drift_reason: str,
    relation: str,
    runtime_viability: str,
) -> tuple[str, dict[str, Any]]:
    path = _observed_target_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "selected_target_name": str(expected_target_name or ""),
        "selected_target_api": str(expected_api or ""),
        "observed_target_api": str(observed_api or ""),
        "observed_harness": str(observed_harness or ""),
        "drifted": bool(drifted),
        "drift_reason": str(drift_reason or ""),
        "relation": str(relation or ""),
        "runtime_viability": str(runtime_viability or ""),
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path), doc


def _load_observed_target_doc(repo_root: Path) -> dict[str, Any]:
    path = _observed_target_path(repo_root)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return dict(raw) if isinstance(raw, dict) else {}


def _infer_harness_primary_api(text: str) -> str:
    keywords = {
        "if",
        "for",
        "while",
        "switch",
        "return",
        "sizeof",
        "catch",
        "static_cast",
        "reinterpret_cast",
        "const_cast",
        "dynamic_cast",
    }
    candidates: list[str] = []
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_:]*)\s*\(", text):
        name = str(match.group(1) or "").strip()
        lowered = name.lower()
        leaf = lowered.split("::")[-1]
        if not lowered or leaf in keywords:
            continue
        if leaf == "llvmfuzzertestoneinput":
            continue
        if leaf.startswith(("is_", "has_", "check_", "validate_", "balanced_", "helper_", "local_")):
            continue
        candidates.append(lowered)
    for candidate in candidates:
        if "::" in candidate and not candidate.startswith(("std::", "absl::")):
            return candidate
    if candidates:
        return candidates[0]
    return ""


def _readme_drift_status(repo_root: Path, alignment: dict[str, Any]) -> dict[str, Any]:
    readme = repo_root / "fuzz" / "README.md"
    if not readme.is_file():
        return {
            "complete": False,
            "missing": ["selected_target", "final_target", "technical_reason", "relation"],
            "relation": "",
            "reason": "",
        }
    text = readme.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    selected = str(alignment.get("expected_api") or alignment.get("expected_target_name") or "").strip().lower()
    observed = str(alignment.get("observed_api") or "").strip().lower()
    relation = ""
    reason = ""
    relation_match = re.search(r"(?:relation|关系)\s*[:：]\s*(.+)", text, re.IGNORECASE)
    if relation_match:
        relation = str(relation_match.group(1) or "").strip()
    reason_match = re.search(r"(?:technical reason|reason|原因)\s*[:：]\s*(.+)", text, re.IGNORECASE)
    if reason_match:
        reason = str(reason_match.group(1) or "").strip()
    missing: list[str] = []
    if selected and selected not in lowered:
        missing.append("selected_target")
    if observed and observed not in lowered:
        missing.append("final_target")
    if not reason:
        missing.append("technical_reason")
    if not relation:
        missing.append("relation")
    return {
        "complete": not missing,
        "missing": missing,
        "relation": relation,
        "reason": reason,
    }
def _analyze_harness_target_alignment(repo_root: Path) -> dict[str, Any]:
    selected_doc = _load_selected_targets_doc(repo_root)
    if not selected_doc:
        return {
            "matched": True,
            "drifted": False,
            "expected_target_name": "",
            "expected_api": "",
            "observed_api": "",
            "observed_harness": "",
            "reason": "",
        }
    primary = selected_doc[0]
    target_name = str(primary.get("target_name") or primary.get("name") or "").strip()
    api = str(primary.get("api") or "").strip()
    fuzz_dir = repo_root / "fuzz"
    harnesses = [
        p for p in fuzz_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".c", ".cc", ".cpp", ".cxx", ".java"}
        and not str(p.relative_to(fuzz_dir)).startswith(("out/", "corpus/"))
    ]
    if not harnesses:
        return {
            "matched": True,
            "drifted": False,
            "expected_target_name": target_name,
            "expected_api": api,
            "observed_api": "",
            "observed_harness": "",
            "reason": "",
        }
    normalized_target = re.sub(r"_fuzz(?:er)?$", "", target_name.lower())
    for harness in harnesses:
        rel = str(harness.relative_to(fuzz_dir)).replace("\\", "/")
        text = harness.read_text(encoding="utf-8", errors="replace").lower()
        name = harness.stem.lower()
        if api and api.lower() in text:
            return {
                "matched": True,
                "drifted": False,
                "expected_target_name": target_name,
                "expected_api": api,
                "observed_api": api.lower(),
                "observed_harness": rel,
                "reason": "",
            }
        if normalized_target and (normalized_target in name or name in normalized_target):
            return {
                "matched": True,
                "drifted": False,
                "expected_target_name": target_name,
                "expected_api": api,
                "observed_api": _infer_harness_primary_api(text),
                "observed_harness": rel,
                "reason": "",
            }
        if target_name and target_name.lower() in text:
            return {
                "matched": True,
                "drifted": False,
                "expected_target_name": target_name,
                "expected_api": api,
                "observed_api": _infer_harness_primary_api(text),
                "observed_harness": rel,
                "reason": "",
            }
    first_harness = harnesses[0]
    first_rel = str(first_harness.relative_to(fuzz_dir)).replace("\\", "/")
    first_text = first_harness.read_text(encoding="utf-8", errors="replace").lower()
    observed_api = _infer_harness_primary_api(first_text)
    expected = api or target_name
    reason = f"selected target drift: expected api `{expected}` but observed `{observed_api or 'unknown'}`"
    return {
        "matched": False,
        "drifted": True,
        "expected_target_name": target_name,
        "expected_api": api,
        "observed_api": observed_api,
        "observed_harness": first_rel,
        "reason": reason,
    }


def _build_fallback_targets_doc(
    repo_root: Path,
    *,
    antlr_context_path: str = "",
    target_analysis_path: str = "",
) -> list[dict[str, str]]:
    ctx_doc: dict[str, Any] = {}
    ctx_path = Path(antlr_context_path).expanduser().resolve() if antlr_context_path else None
    if ctx_path and ctx_path.is_file():
        try:
            loaded = json.loads(ctx_path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(loaded, dict):
                ctx_doc = loaded
        except Exception:
            ctx_doc = {}
    analysis_doc: dict[str, Any] = {}
    analysis_path = Path(target_analysis_path).expanduser().resolve() if target_analysis_path else None
    if analysis_path and analysis_path.is_file():
        try:
            loaded = json.loads(analysis_path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(loaded, dict):
                analysis_doc = loaded
        except Exception:
            analysis_doc = {}

    candidates: list[dict[str, str]] = []
    raw_candidates = (
        list(analysis_doc.get("recommended_targets") or [])
        + list(ctx_doc.get("entrypoint_candidates") or [])
        + list(ctx_doc.get("candidate_functions") or [])
    )
    seen: set[tuple[str, str]] = set()
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        file_hint = str(item.get("file") or "").strip()
        lang = _infer_target_lang_from_repo(repo_root, file_hint=file_hint)
        key = (name, lang)
        if key in seen:
            continue
        seen.add(key)
        _raw_type = str(item.get("target_type") or "").strip().lower()
        target_type = _raw_type if _raw_type and _raw_type != "pending" else _infer_target_type(name, file_hint)
        _raw_sp = str(item.get("seed_profile") or "").strip().lower()
        depth_score = int(item.get("depth_score") or 0)
        depth_class = str(item.get("depth_class") or "shallow")
        selection_bias_reason = str(item.get("selection_bias_reason") or "")
        if not selection_bias_reason:
            depth_score, depth_class, selection_bias_reason = _score_target_depth(
                name,
                file_hint,
                target_type=target_type,
                risk_signals=list(item.get("risk_signals") or []),
            )
        runtime_viability = str(item.get("runtime_viability") or "").strip().lower()
        selection_rationale = str(item.get("selection_rationale") or "").strip()
        runtime_replacement_candidates = list(item.get("runtime_replacement_candidates") or [])
        if not runtime_viability:
            runtime_viability, auto_rationale, auto_replacements = _runtime_viability_details(
                name,
                file_hint,
                file_hint=file_hint,
            )
            selection_rationale = selection_rationale or auto_rationale
            runtime_replacement_candidates = runtime_replacement_candidates or auto_replacements
        candidates.append(
            {
                "name": name,
                "api": name,
                "lang": lang,
                "target_type": target_type,
                "seed_profile": _raw_sp if _raw_sp and _raw_sp != "pending" else _infer_seed_profile(name, file_hint, target_type=target_type),
                "depth_score": depth_score,
                "depth_class": depth_class,
                "selection_bias_reason": selection_bias_reason,
                "runtime_viability": runtime_viability,
                "selection_rationale": selection_rationale,
                "runtime_replacement_candidates": runtime_replacement_candidates,
            }
        )
        if len(candidates) >= 3:
            break

    if candidates:
        has_deep = any(str(item.get("depth_class") or "") == "deep" for item in candidates)
        if has_deep:
            candidates = [item for item in candidates if str(item.get("depth_class") or "") != "shallow"]
        candidates.sort(
            key=lambda item: (
                -{"high": 2, "medium": 1, "low": 0}.get(str(item.get("runtime_viability") or "").lower(), 0),
                -int(item.get("depth_score") or 0),
                str(item.get("name") or ""),
            )
        )
        return candidates

    return [
        {
            "name": "default_target",
            "api": "default_target",
            "lang": _infer_target_lang_from_repo(repo_root),
            "target_type": "generic",
            "seed_profile": "generic",
            "depth_score": 0,
            "depth_class": "shallow",
            "selection_bias_reason": "fallback-default",
            "runtime_viability": "medium",
            "selection_rationale": "fallback-default",
            "runtime_replacement_candidates": [],
        }
    ]


def _write_fallback_targets_json(
    repo_root: Path,
    *,
    antlr_context_path: str = "",
    target_analysis_path: str = "",
) -> bool:
    fuzz_dir = repo_root / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    targets_path = fuzz_dir / "targets.json"
    doc = _build_fallback_targets_doc(
        repo_root,
        antlr_context_path=antlr_context_path,
        target_analysis_path=target_analysis_path,
    )
    try:
        targets_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        return False
    ok, _err = _validate_targets_json(repo_root)
    return ok


def _summarize_build_error(last_error: str, stdout_tail: str, stderr_tail: str) -> dict[str, str]:
    return _wf_common.summarize_build_error(last_error, stdout_tail, stderr_tail)


def _extract_actionable_build_locations(
    last_error: str,
    stdout_tail: str,
    stderr_tail: str,
    *,
    limit: int = 4,
) -> list[dict[str, str]]:
    text = "\n".join([str(last_error or ""), str(stdout_tail or ""), str(stderr_tail or "")])
    lines = text.splitlines()
    path_re = re.compile(
        r"(?P<path>(?:/|(?:\./))?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.(?:cxx|cpp|cc|c|hpp|h|py|java))(?:[:(](?P<line>\d+))?"
    )
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in lines:
        for m in path_re.finditer(line):
            path = str(m.group("path") or "").lstrip("./")
            if not path:
                continue
            ln = str(m.group("line") or "").strip()
            key = (path, ln)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "path": path,
                    "line": ln,
                    "evidence": line.strip()[:500],
                }
            )
            if len(out) >= limit:
                return out
    return out


def _build_file_targeted_fix_lines(
    repo_root: Path,
    last_error: str,
    stdout_tail: str,
    stderr_tail: str,
) -> list[str]:
    hits = _extract_actionable_build_locations(last_error, stdout_tail, stderr_tail, limit=3)
    if not hits:
        return []
    full_diag = "\n".join([str(last_error or ""), str(stdout_tail or ""), str(stderr_tail or "")]).lower()
    lines: list[str] = ["Prioritize file-targeted fixes from diagnostics:"]
    for item in hits:
        raw_path = str(item.get("path") or "").strip()
        if not raw_path:
            continue
        path = raw_path.replace("\\", "/")
        if "/build-work/" in path or "/CMakeFiles/" in path or path.startswith("build-work/"):
            continue
        if path.startswith("/"):
            abs_path = path
        else:
            abs_path = str((repo_root / path.lstrip("./")).resolve())
        ln = item.get("line") or ""
        loc = f"{abs_path}:{ln}" if ln else abs_path
        if "include" in full_diag and ("not declared" in full_diag or "not a member" in full_diag or "undeclared" in full_diag):
            lines.append(f"- Read and fix `{loc}` (header/symbol declaration mismatch; add the required include or declaration).")
        elif "undefined reference" in full_diag or "cannot find -l" in full_diag:
            lines.append(f"- Read and fix `{loc}` (linkage/build glue mismatch; align build.py/link inputs with this source usage).")
        else:
            lines.append(f"- Read and fix `{loc}` based on the failing diagnostic evidence.")
    return lines if len(lines) > 1 else []


def _repair_strategy_repeat_threshold() -> int:
    raw = (os.environ.get("SHERPA_REPAIR_STRATEGY_REPEAT_THRESHOLD") or "3").strip()
    try:
        return max(2, min(int(raw), 10))
    except Exception:
        return 3


def _extract_repair_symbols(text: str, *, limit: int = 12) -> list[str]:
    buf = str(text or "")
    patterns = [
        re.compile(r"undefined reference to [`'\"]([^`'\"]+)[`'\"]", re.IGNORECASE),
        re.compile(r"no (?:member|type) named ['`]([^'`]+)['`]", re.IGNORECASE),
        re.compile(r"cannot find -l([A-Za-z0-9_+.-]+)", re.IGNORECASE),
    ]
    out: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        for m in pat.finditer(buf):
            symbol = str(m.group(1) or "").strip()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            out.append(symbol)
            if len(out) >= limit:
                return out
    return out


def _extract_repair_top_trace(error_text: str, stdout_tail: str, stderr_tail: str, *, limit: int = 12) -> list[str]:
    lines = []
    for chunk in (stderr_tail, error_text, stdout_tail):
        for ln in str(chunk or "").replace("\r", "\n").splitlines():
            txt = ln.strip()
            if not txt:
                continue
            low = txt.lower()
            if any(
                token in low
                for token in (
                    "error:",
                    "fatal:",
                    "traceback",
                    "calledprocesserror",
                    "undefined reference",
                    "cannot find -l",
                    "no rule to make target",
                    "permission denied",
                )
            ):
                lines.append(txt[:500])
                if len(lines) >= limit:
                    return lines
    return lines


def _build_repair_error_digest(
    *,
    repo_root: Path,
    error_kind: str,
    error_code: str,
    signature: str,
    error_text: str,
    stdout_tail: str,
    stderr_tail: str,
    prev_digest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prev = dict(prev_digest or {})
    now = int(time.time())
    prev_sig = str(prev.get("signature") or "").strip()
    files = _extract_actionable_build_locations(error_text, stdout_tail, stderr_tail, limit=12)
    failing_files: list[str] = []
    seen_files: set[str] = set()
    for item in files:
        raw_path = str(item.get("path") or "").strip()
        if not raw_path:
            continue
        normalized = raw_path.replace("\\", "/")
        if normalized.startswith("/"):
            abs_path = normalized
        else:
            abs_path = str((repo_root / normalized.lstrip("./")).resolve())
        if abs_path in seen_files:
            continue
        seen_files.add(abs_path)
        failing_files.append(abs_path)
    return {
        "error_code": str(error_code or ""),
        "error_kind": str(error_kind or ""),
        "signature": str(signature or "")[:12],
        "failing_files": failing_files,
        "symbols": _extract_repair_symbols("\n".join([error_text, stdout_tail, stderr_tail])),
        "first_seen": int(prev.get("first_seen") or now) if prev_sig and prev_sig == str(signature or "")[:12] else now,
        "latest_seen": now,
        "top_trace": _extract_repair_top_trace(error_text, stdout_tail, stderr_tail),
    }


def _validate_execution_plan_harness_consistency(
    repo_root: Path,
    *,
    execution_plan_doc: dict[str, Any] | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    doc = _build_harness_index_doc(repo_root, execution_plan_doc=execution_plan_doc)
    missing_targets = [str(x).strip() for x in list(doc.get("missing_targets") or []) if str(x).strip()]
    if missing_targets:
        extras = [str(x).strip() for x in list(doc.get("extra_harnesses") or []) if str(x).strip()]
        msg = (
            "execution_plan_harness_mismatch: missing harness source for targets="
            + ",".join(missing_targets)
            + (f"; extra_harnesses={','.join(extras[:8])}" if extras else "")
        )
        return False, msg, doc
    return True, "", doc


def _validate_build_repair_contract(
    repo_root: Path,
    state: FuzzWorkflowRuntimeState,
    harness_index_doc: dict[str, Any],
) -> tuple[bool, str]:
    if not bool(state.get("repair_mode")):
        return True, ""
    if str(state.get("repair_origin_stage") or "").strip() != "build":
        return True, ""
    error_code = str(state.get("repair_error_code") or "").strip()
    if not error_code:
        return True, ""

    mappings = [m for m in list(harness_index_doc.get("mappings") or []) if isinstance(m, dict)]
    source_paths = [str(m.get("source_path") or "").strip() for m in mappings]
    source_paths = [p for p in source_paths if p]
    if not source_paths:
        return False, "repair contract failed: no harness source mapped in fuzz/harness_index.json"

    if error_code == "missing_llvmfuzzer_entrypoint":
        missing_entrypoints: list[str] = []
        for rel in source_paths:
            p = (repo_root / rel).resolve()
            if not p.is_file():
                missing_entrypoints.append(rel)
                continue
            txt = p.read_text(encoding="utf-8", errors="replace")
            if "LLVMFuzzerTestOneInput" not in txt:
                missing_entrypoints.append(rel)
        if missing_entrypoints:
            return (
                False,
                "repair contract failed: missing LLVMFuzzerTestOneInput in harness source(s): "
                + ",".join(missing_entrypoints),
            )

    if error_code in {"cxx_for_c_source_mismatch", "c_compiler_for_cpp_source_mismatch"}:
        build_py = (repo_root / "fuzz" / "build.py")
        if not build_py.is_file():
            return False, "repair contract failed: fuzz/build.py missing for compiler mismatch repair"
        build_txt = build_py.read_text(encoding="utf-8", errors="replace")
        needs_c = any(Path(rel).suffix.lower() == ".c" for rel in source_paths)
        needs_cxx = any(Path(rel).suffix.lower() in {".cc", ".cpp", ".cxx"} for rel in source_paths)
        if needs_c and "clang" not in build_txt:
            return False, "repair contract failed: build.py lacks C compiler invocation hints for .c harnesses"
        if needs_cxx and "clang++" not in build_txt:
            return False, "repair contract failed: build.py lacks C++ compiler invocation hints for C++ harnesses"

    return True, ""


def _validate_harness_source_contract(
    repo_root: Path,
    harness_index_doc: dict[str, Any],
) -> tuple[bool, str]:
    mappings = [m for m in list(harness_index_doc.get("mappings") or []) if isinstance(m, dict)]
    source_paths = [str(m.get("source_path") or "").strip() for m in mappings]
    source_paths = [p for p in source_paths if p]
    if not source_paths:
        return True, ""

    violations: list[str] = []
    for rel in source_paths:
        p = (repo_root / rel).resolve()
        if not p.is_file():
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        lowered = txt.lower()
        if re.search(r"\b(?:int|auto|void)\s+main\s*\(", txt):
            violations.append(f"{rel}: custom main() is forbidden")
        if re.search(r"\bfopen\s*\(\s*argv\s*\[\s*1\s*\]", lowered):
            violations.append(f"{rel}: forbidden corpus-file entry pattern fopen(argv[1], ...)")
        if re.search(r"\b(?:open|read)\s*\(\s*argv\s*\[\s*1\s*\]", lowered):
            violations.append(f"{rel}: forbidden argv[1]-driven read/open entry pattern")
        if "reinterpret_cast<file*>" in lowered or "(file*)data" in lowered:
            violations.append(f"{rel}: FILE* cast from fuzz input is forbidden")

    if violations:
        limited = "; ".join(violations[:6])
        if len(violations) > 6:
            limited += f"; ...(+{len(violations) - 6} more)"
        return False, f"harness contract failed: {limited}"
    return True, ""


def _classify_build_failure(
    last_error: str,
    stdout_tail: str,
    stderr_tail: str,
    *,
    build_rc: int,
    has_fuzzer_binaries: bool,
) -> tuple[str, str]:
    return _wf_common.classify_build_failure(
        last_error,
        stdout_tail,
        stderr_tail,
        build_rc=build_rc,
        has_fuzzer_binaries=has_fuzzer_binaries,
    )


def _build_failure_recovery_advice(error_kind: str, error_code: str) -> str:
    return _wf_common.build_failure_recovery_advice(error_kind, error_code)


def _collect_key_artifact_hashes(repo_root: Path) -> dict[str, str]:
    return _wf_common.collect_key_artifact_hashes(repo_root)


def _has_codex_key() -> bool:
    return _wf_common.has_codex_key()


def _build_seed_feedback(state: dict[str, Any]) -> dict[str, Any]:
    quality = dict(state.get("coverage_seed_quality") or {})
    return {
        "seed_profile": str(state.get("coverage_seed_profile") or ""),
        "initial_inited_cov": int(quality.get("initial_inited_cov") or 0),
        "final_cov": int(quality.get("final_cov") or 0),
        "cov_delta": int(quality.get("cov_delta") or 0),
        "initial_inited_ft": int(quality.get("initial_inited_ft") or 0),
        "final_ft": int(quality.get("final_ft") or 0),
        "ft_delta": int(quality.get("ft_delta") or 0),
        "early_new_units_30s": int(quality.get("early_new_units_30s") or 0),
        "early_new_units_60s": int(quality.get("early_new_units_60s") or 0),
        "initial_corpus_files": int(quality.get("initial_corpus_files") or 0),
        "final_corpus_files": int(quality.get("final_corpus_files") or 0),
        "cold_start_failure": bool(quality.get("cold_start_failure") or False),
        "merge_retained_ratio_files": float(quality.get("merge_retained_ratio_files") or 1.0),
        "merge_retained_ratio_bytes": float(quality.get("merge_retained_ratio_bytes") or 1.0),
        "required_families": list(state.get("coverage_seed_families_required") or []),
        "covered_families": list(state.get("coverage_seed_families_covered") or []),
        "missing_families": list(state.get("coverage_seed_families_missing") or []),
        "quality_flags": list(state.get("coverage_quality_flags") or quality.get("quality_flags") or []),
        "seed_score": float(quality.get("seed_score") or 0.0),
        "seed_score_components": dict(quality.get("seed_score_components") or {}),
        "seed_counts_raw": dict(state.get("coverage_seed_counts_raw") or {}),
        "seed_counts_filtered": dict(state.get("coverage_seed_counts_filtered") or {}),
        "seed_noise_rejected_count": int(state.get("coverage_seed_noise_rejected_count") or 0),
        "seed_generation_failed_count": int(state.get("coverage_seed_generation_failed_count") or 0),
        "seed_generation_failed_fuzzers": list(state.get("coverage_seed_generation_failed_fuzzers") or []),
        "seed_generation_degraded": bool(state.get("coverage_seed_generation_degraded") or False),
        "corpus_sources": list(state.get("coverage_corpus_sources") or []),
    }


def _build_harness_feedback(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "execution_plan_path": str(state.get("execution_plan_path") or ""),
        "harness_index_path": str(state.get("harness_index_path") or ""),
        "selected_target_api": str(state.get("selected_target_api") or ""),
        "coverage_target_api": str(state.get("coverage_target_api") or ""),
        "missing_execution_targets": list(state.get("coverage_missing_execution_targets") or []),
        "built_targets": list(state.get("built_targets") or []),
        "missing_targets": list(state.get("missing_targets") or []),
        "target_build_matrix": list(state.get("target_build_matrix") or []),
    }


def _slug_from_repo_url(repo_url: str) -> str:
    return _wf_common.slug_from_repo_url(repo_url)


def _alloc_output_workdir(repo_url: str) -> Path | None:
    return _wf_common.alloc_output_workdir(repo_url)


def _opencode_defunct_threshold() -> int:
    raw = (os.environ.get("SHERPA_OPENCODE_DEFUNCT_THRESHOLD") or "3").strip()
    try:
        return max(0, min(int(raw), 200))
    except Exception:
        return 3


def _count_opencode_defunct_processes() -> int:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "stat=,args="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return 0
    if int(proc.returncode or 0) != 0:
        return 0
    count = 0
    for raw_line in str(proc.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        stat = parts[0] if parts else ""
        cmd = parts[1] if len(parts) > 1 else ""
        cmd_l = cmd.lower()
        if "opencode" not in cmd_l:
            continue
        if "<defunct>" in cmd_l or stat.startswith("Z"):
            count += 1
    return count


def _enter_step(state: FuzzWorkflowRuntimeState, step_name: str) -> tuple[FuzzWorkflowRuntimeState, bool]:
    normalized_in = _normalize_error_state(cast(dict[str, Any], state))
    out, stop = _wf_common.enter_step(normalized_in, step_name)
    next_state = cast(FuzzWorkflowRuntimeState, _normalize_error_state(out))
    if stop:
        return next_state, stop
    defunct_count = _count_opencode_defunct_processes()
    next_state = cast(FuzzWorkflowRuntimeState, {**next_state, "opencode_defunct_count": defunct_count})
    threshold = _opencode_defunct_threshold()
    if threshold > 0 and defunct_count > threshold:
        msg = (
            f"opencode defunct process count exceeded threshold: "
            f"{defunct_count}>{threshold}; fail-fast to avoid stage hang"
        )
        guarded = cast(
            FuzzWorkflowRuntimeState,
            _normalize_error_state({
                **next_state,
                "last_step": step_name,
                "failed": True,
                "last_error": msg,
                "message": "workflow stopped (opencode defunct safeguard)",
                "error": {
                    "stage": step_name,
                    "kind": "infra",
                    "code": "opencode_defunct_safeguard",
                    "message": msg,
                    "detail": msg,
                    "signature": "",
                    "retryable": False,
                    "terminal": True,
                    "at": int(time.time()),
                },
            }),
        )
        _wf_log(cast(dict[str, Any], guarded), f"<- {step_name} stop=opencode-defunct count={defunct_count} threshold={threshold}")
        return guarded, True
    if defunct_count > 0:
        _wf_log(cast(dict[str, Any], next_state), f"{step_name}: opencode_defunct_count={defunct_count}")
    return next_state, False


def _remaining_time_budget_sec(state: FuzzWorkflowRuntimeState, *, min_timeout: int = 5) -> int:
    return _wf_common.remaining_time_budget_sec(cast(dict[str, Any], state), min_timeout=min_timeout)


def _opencode_cli_retries() -> int:
    raw = (os.environ.get("SHERPA_WORKFLOW_OPENCODE_CLI_RETRIES") or "2").strip()
    try:
        return max(1, min(int(raw), 8))
    except Exception:
        return 2


def _fix_build_max_noop_streak() -> int:
    raw = (os.environ.get("SHERPA_FIX_BUILD_MAX_NOOP_STREAK") or "3").strip()
    try:
        return max(1, min(int(raw), 20))
    except Exception:
        return 3


def _fix_build_max_attempts() -> int:
    raw = (os.environ.get("SHERPA_FIX_BUILD_MAX_ATTEMPTS") or "0").strip()
    try:
        return max(0, min(int(raw), 50_000))
    except Exception:
        return 0


def _effective_max_fix_rounds(state: FuzzWorkflowRuntimeState) -> int:
    # Fixed unlimited mode: fix_build is bounded only by time/stage budget.
    _ = state
    return 0


def _effective_same_error_retry_limit(state: FuzzWorkflowRuntimeState) -> int:
    # Fixed unlimited mode: do not stop/restart by same-error repeat count.
    _ = state
    return 0


def _fix_build_feedback_history_limit() -> int:
    raw = (os.environ.get("SHERPA_FIX_BUILD_FEEDBACK_HISTORY") or "6").strip()
    try:
        return max(1, min(int(raw), 30))
    except Exception:
        return 6


def _fix_build_context_max_chars() -> int:
    raw = (os.environ.get("SHERPA_FIX_BUILD_CONTEXT_MAX_CHARS") or "65536").strip()
    try:
        return max(4000, min(int(raw), 300000))
    except Exception:
        return 65536


def _fix_build_stdout_max_chars() -> int:
    raw = (os.environ.get("SHERPA_FIX_BUILD_STDOUT_MAX_CHARS") or "12000").strip()
    try:
        return max(1000, min(int(raw), 120000))
    except Exception:
        return 12000


def _fix_build_stderr_max_chars() -> int:
    raw = (os.environ.get("SHERPA_FIX_BUILD_STDERR_MAX_CHARS") or "42000").strip()
    try:
        return max(2000, min(int(raw), 220000))
    except Exception:
        return 42000


def _fix_build_keep_recent_errors() -> int:
    raw = (os.environ.get("SHERPA_FIX_BUILD_KEEP_RECENT_ERRORS") or "3").strip()
    try:
        return max(1, min(int(raw), 12))
    except Exception:
        return 3


def _fix_build_context_history_limit() -> int:
    raw = (os.environ.get("SHERPA_FIX_BUILD_CONTEXT_MAX_HISTORY") or "3").strip()
    try:
        return max(1, min(int(raw), 20))
    except Exception:
        return 3


def _fix_build_ruleset() -> str:
    raw = (os.environ.get("SHERPA_FIX_BUILD_RULESET") or "extended").strip().lower()
    if raw in {"legacy", "extended"}:
        return raw
    return "extended"


def _run_idle_timeout_sec() -> int:
    raw = (os.environ.get("SHERPA_RUN_IDLE_TIMEOUT_SEC") or "120").strip()
    try:
        return max(0, min(int(raw), 86400))
    except Exception:
        return 120


def _synthesize_opencode_idle_timeout_sec() -> int:
    raw = (os.environ.get("SHERPA_OPENCODE_IDLE_TIMEOUT_SYNTH_SEC") or "300").strip()
    try:
        return max(0, min(int(raw), 86_400))
    except Exception:
        return 300


def _synthesize_opencode_attempts() -> int:
    raw = (os.environ.get("SHERPA_OPENCODE_SYNTH_MAX_ATTEMPTS") or "2").strip()
    try:
        return max(1, min(int(raw), 4))
    except Exception:
        return 2


def _fix_build_same_signature_plan_threshold() -> int:
    raw = (os.environ.get("SHERPA_FIX_BUILD_SAME_SIGNATURE_TO_PLAN") or "3").strip()
    try:
        return max(1, min(int(raw), 20))
    except Exception:
        return 3


def _contains_cjk_text(text: str) -> bool:
    try:
        return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))
    except Exception:
        return False


def _synthesize_activity_watch_paths() -> list[str]:
    return [
        "fuzz/repo_understanding.json",
        "fuzz/build_strategy.json",
        "fuzz/build.py",
        "fuzz/README.md",
        "fuzz/system_packages.txt",
        "fuzz/*.c",
        "fuzz/*.cc",
        "fuzz/*.cpp",
        "fuzz/*.cxx",
        "fuzz/*.java",
        "fuzz/**/*.c",
        "fuzz/**/*.cc",
        "fuzz/**/*.cpp",
        "fuzz/**/*.cxx",
        "fuzz/**/*.java",
    ]


def _build_scaffold_path(repo_root: Path) -> Path:
    return repo_root / "fuzz" / "build_strategy.json"


def _build_template_cache_path(repo_root: Path) -> Path:
    return repo_root / "fuzz" / "build_template_cache.json"


def _find_static_lib(repo_root: Path, lib_name_pattern: str) -> Path | None:
    pattern = str(lib_name_pattern or "").strip()
    if not pattern:
        return None
    patterns = [
        f"**/{pattern}",
        f"**/libarchive/{pattern}",
        "**/libarchive/libarchive*.a",
        "**/.libs/libarchive*.a",
    ]
    seen: set[str] = set()
    for glob_pat in patterns:
        try:
            for match in repo_root.glob(glob_pat):
                if not match.is_file():
                    continue
                key = str(match)
                if key in seen:
                    continue
                seen.add(key)
                return match
        except Exception:
            continue
    return None


def _load_build_template_cache_doc(repo_root: Path) -> dict[str, Any]:
    path = _build_template_cache_path(repo_root)
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return doc if isinstance(doc, dict) else {}


def _write_build_template_cache_doc(repo_root: Path, doc: dict[str, Any]) -> str:
    path = _build_template_cache_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def _cache_successful_build_template(
    repo_root: Path,
    *,
    binaries: list[Path] | None = None,
    target_build_matrix: list[dict[str, Any]] | None = None,
) -> str:
    fuzz_dir = repo_root / "fuzz"
    build_py = fuzz_dir / "build.py"
    if not build_py.is_file():
        return ""
    try:
        build_text = build_py.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    strategy = _load_build_strategy_doc(repo_root)
    doc: dict[str, Any] = {
        "schema_version": 2,
        "saved_at": int(time.time()),
        "build_py": build_text,
        "build_strategy": strategy if isinstance(strategy, dict) else {},
        "binary_names": [p.name for p in (binaries or []) if isinstance(p, Path)],
        "target_build_matrix": list(target_build_matrix or []),
    }
    try:
        return _write_build_template_cache_doc(repo_root, doc)
    except Exception:
        return ""


def _restore_cached_build_template_if_missing(repo_root: Path) -> bool:
    fuzz_dir = repo_root / "fuzz"
    build_py = fuzz_dir / "build.py"
    if build_py.is_file():
        return False
    cache_doc = _load_build_template_cache_doc(repo_root)
    build_text = str(cache_doc.get("build_py") or "")
    if not build_text.strip():
        return False
    try:
        fuzz_dir.mkdir(parents=True, exist_ok=True)
        build_py.write_text(build_text, encoding="utf-8")
    except Exception:
        return False
    strategy = cache_doc.get("build_strategy")
    if isinstance(strategy, dict) and strategy:
        try:
            _build_scaffold_path(repo_root).write_text(
                json.dumps(strategy, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
    return True


def _build_runtime_facts_path(repo_root: Path) -> Path:
    return repo_root / "fuzz" / "build_runtime_facts.json"


def _repo_understanding_path(repo_root: Path) -> Path:
    return repo_root / "fuzz" / "repo_understanding.json"


def _load_repo_understanding_doc(repo_root: Path) -> dict[str, Any]:
    path = _repo_understanding_path(repo_root)
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _repo_understanding_is_complete(doc: dict[str, Any]) -> tuple[bool, str]:
    if not doc:
        return False, "missing fuzz/repo_understanding.json"
    for key in ("build_system", "chosen_target_api", "chosen_target_reason", "fuzzer_entry_strategy"):
        if not str(doc.get(key) or "").strip():
            return False, f"repo understanding missing `{key}`"
    if str(doc.get("build_system") or "").strip().lower() == "unknown":
        return False, "repo understanding must identify a concrete build_system"
    evidence = doc.get("evidence")
    if not isinstance(evidence, list) or not any(str(item or "").strip() for item in evidence):
        return False, "repo understanding must include non-empty evidence"
    return True, ""


def _load_build_strategy_doc(repo_root: Path) -> dict[str, Any]:
    path = _build_scaffold_path(repo_root)
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _load_build_runtime_facts_doc(repo_root: Path) -> dict[str, Any]:
    path = _build_runtime_facts_path(repo_root)
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _contains_forbidden_repo_fuzz_target_usage(text: str) -> bool:
    lowered = (text or "").lower()
    patterns = [
        r"--target[^a-z0-9]*(?:[a-z0-9._-]*)(?:fuzz|fuzzer)[a-z0-9._-]*",
        r"\b(?:make|gmake|ninja)[^a-z0-9]*(?:[a-z0-9._-]*)(?:fuzz|fuzzer)[a-z0-9._-]*",
    ]
    return any(re.search(pat, lowered, re.IGNORECASE) for pat in patterns)


def _extract_repo_fuzz_target_usages(text: str) -> list[str]:
    usages: list[str] = []
    for match in re.finditer(r"--target\s+([A-Za-z0-9._+-]+)", text or "", re.IGNORECASE):
        usages.append(str(match.group(1) or "").strip())
    for match in re.finditer(r"['\"]--target['\"]\s*,\s*['\"]([A-Za-z0-9._+-]+)['\"]", text or "", re.IGNORECASE):
        usages.append(str(match.group(1) or "").strip())
    for match in re.finditer(r"\b(?:make|gmake|ninja)\s+([A-Za-z0-9._+-]+)", text or "", re.IGNORECASE):
        usages.append(str(match.group(1) or "").strip())
    return [u for u in usages if u]


def _allowed_repo_fuzz_targets(repo_root: Path) -> set[str]:
    allowed: set[str] = set()
    repo_understanding = _load_repo_understanding_doc(repo_root)
    build_strategy = _load_build_strategy_doc(repo_root)
    for item in list(repo_understanding.get("repo_fuzz_targets") or []) + list(build_strategy.get("repo_fuzz_targets") or []):
        target = str(item or "").strip()
        if target:
            allowed.add(target)
    selected = str(build_strategy.get("selected_repo_target") or repo_understanding.get("selected_repo_target") or "").strip()
    if selected:
        allowed.add(selected)
    return allowed


def _infer_fuzzer_entry_strategy(build_text: str) -> str:
    lowered = (build_text or "").lower()
    if "-fsanitize=fuzzer" in lowered:
        return "sanitizer_fuzzer"
    if "main.cc" in lowered or "fuzzer-common.h" in lowered:
        return "repo_main_source"
    return "custom_main_source"


def _write_build_strategy_doc(repo_root: Path) -> tuple[str, dict[str, Any]]:
    fuzz_dir = repo_root / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    build_py = fuzz_dir / "build.py"
    build_sh = fuzz_dir / "build.sh"
    build_text = ""
    if build_py.is_file():
        build_text = build_py.read_text(encoding="utf-8", errors="replace")
    elif build_sh.is_file():
        build_text = build_sh.read_text(encoding="utf-8", errors="replace")
    path = _build_scaffold_path(repo_root)
    existing = _load_build_strategy_doc(repo_root)
    repo_understanding = _load_repo_understanding_doc(repo_root)
    build_mode = str(existing.get("build_mode") or "").strip() or "library_link"
    if build_mode not in {"repo_target", "library_link", "custom_script"}:
        build_mode = "library_link"
    reason = str(existing.get("reason") or "").strip() or "default external harness/library-link strategy"
    if not build_text.strip():
        build_mode = "custom_script"
        reason = str(existing.get("reason") or "").strip() or "no readable build scaffold found"
    elif _contains_forbidden_repo_fuzz_target_usage(build_text):
        reason = str(existing.get("reason") or "").strip() or "scaffold references repository fuzz targets; still recorded as external strategy for repair"
    entry = str(existing.get("fuzzer_entry_strategy") or "").strip() or _infer_fuzzer_entry_strategy(build_text)
    doc: dict[str, Any] = {
        "build_system": str(existing.get("build_system") or repo_understanding.get("build_system") or "unknown"),
        "build_mode": build_mode,
        "library_targets": list(existing.get("library_targets") or []),
        "library_artifacts": list(existing.get("library_artifacts") or []),
        "include_dirs": list(existing.get("include_dirs") or repo_understanding.get("include_dirs") or []),
        "extra_sources": list(existing.get("extra_sources") or repo_understanding.get("extra_sources") or []),
        "fuzzer_entry_strategy": entry,
        "reason": reason,
        "evidence": list(existing.get("evidence") or repo_understanding.get("evidence") or []),
        "repo_fuzz_targets": list(existing.get("repo_fuzz_targets") or repo_understanding.get("repo_fuzz_targets") or []),
        "selected_repo_target": str(existing.get("selected_repo_target") or repo_understanding.get("selected_repo_target") or ""),
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path), doc


def _build_scaffold_precheck(repo_root: Path) -> dict[str, Any]:
    # Precheck is intentionally disabled: build/fix loop should decide based on
    # real build outcomes instead of fail-fast gating.
    return {"ok": True, "code": "", "reason": "disabled"}

    # Legacy logic retained below for reference (unreachable).
    fuzz_dir = repo_root / "fuzz"
    build_py = fuzz_dir / "build.py"
    build_sh = fuzz_dir / "build.sh"
    build_text = ""
    if build_py.is_file():
        build_text = build_py.read_text(encoding="utf-8", errors="replace")
    elif build_sh.is_file():
        build_text = build_sh.read_text(encoding="utf-8", errors="replace")
    strategy = _load_build_strategy_doc(repo_root)
    usages = _extract_repo_fuzz_target_usages(build_text)
    if usages:
        allowed_targets = _allowed_repo_fuzz_targets(repo_root)
        unknown = [u for u in usages if u not in allowed_targets]
        if not allowed_targets or unknown:
            return {
                "ok": False,
                "code": "build_strategy_mismatch",
                "reason": "build scaffold references undocumented or guessed repository fuzz targets",
            }
    understanding = _load_repo_understanding_doc(repo_root)
    understanding_ok, understanding_reason = _repo_understanding_is_complete(understanding)
    if not understanding_ok:
        return {
            "ok": False,
            "code": "insufficient_repo_understanding",
            "reason": understanding_reason,
        }
    if strategy and not str(strategy.get("fuzzer_entry_strategy") or "").strip():
        return {
            "ok": False,
            "code": "missing_fuzzer_main",
            "reason": "build strategy missing fuzzer entry strategy",
        }
    return {"ok": True, "code": "", "reason": ""}


def _run_finalize_timeout_sec() -> int:
    raw = (os.environ.get("SHERPA_RUN_FINALIZE_TIMEOUT_SEC") or "60").strip()
    try:
        return max(0, min(int(raw), 3600))
    except Exception:
        return 60


def _run_unlimited_round_budget_sec() -> int:
    raw = (os.environ.get("SHERPA_RUN_UNLIMITED_ROUND_BUDGET_SEC") or "7200").strip()
    try:
        # 0 means fully unlimited (legacy behavior).
        return max(0, min(int(raw), 86400))
    except Exception:
        return 7200


def _verify_stage_no_ai() -> bool:
    raw = (os.environ.get("SHERPA_VERIFY_STAGE_NO_AI") or "1").strip().lower()
    if not raw:
        return True
    return raw in {"1", "true", "yes", "on"}


def _max_same_timeout_repeats() -> int:
    raw = (os.environ.get("SHERPA_WORKFLOW_MAX_SAME_TIMEOUT_REPEATS") or "1").strip()
    try:
        return max(0, min(int(raw), 10))
    except Exception:
        return 1


def _run_stop_on_first_crash() -> bool:
    raw = (os.environ.get("SHERPA_RUN_STOP_ON_FIRST_CRASH") or "1").strip().lower()
    if not raw:
        return True
    return raw in {"1", "true", "yes", "on"}


def _run_parallel_early_stop_enabled() -> bool:
    raw = (os.environ.get("SHERPA_RUN_PARALLEL_EARLY_STOP_ENABLED") or "1").strip().lower()
    if not raw:
        return True
    return raw in {"1", "true", "yes", "on"}


def _run_cpu_budget() -> int:
    raw = (os.environ.get("SHERPA_RUN_CPU_BUDGET") or "").strip()
    if raw:
        try:
            return max(1, min(int(raw), 1024))
        except Exception:
            pass
    return max(1, int(os.cpu_count() or 1))


def _run_outer_parallelism_max(default_parallel: int) -> int:
    raw = (os.environ.get("SHERPA_RUN_OUTER_PARALLELISM_MAX") or str(default_parallel)).strip()
    try:
        return max(1, min(int(raw), 64))
    except Exception:
        return max(1, default_parallel)


def _run_inner_workers_min() -> int:
    raw = (os.environ.get("SHERPA_RUN_INNER_WORKERS_MIN") or "1").strip()
    try:
        return max(1, min(int(raw), 64))
    except Exception:
        return 1


def _run_inner_workers_target() -> int:
    run_inner_raw = (os.environ.get("SHERPA_RUN_INNER_WORKERS") or "").strip()
    legacy_fork_raw = (os.environ.get("SHERPA_FUZZ_FORK") or "").strip()
    raw = run_inner_raw
    if not raw:
        raw = legacy_fork_raw or "1"
        if legacy_fork_raw:
            logger.info(
                "[warn] SHERPA_FUZZ_FORK is deprecated for run parallel config. "
                "Prefer SHERPA_RUN_INNER_WORKERS + SHERPA_RUN_PARALLEL_ENGINE."
            )
    try:
        return max(1, min(int(raw), 128))
    except Exception:
        return 1


def _run_parallel_engine() -> str:
    raw = (os.environ.get("SHERPA_RUN_PARALLEL_ENGINE") or "auto").strip().lower()
    if raw in {"auto", "fork", "jobs_workers", "single"}:
        return raw
    return "auto"


def _run_ignore_non_fatal_enabled() -> bool:
    raw = (os.environ.get("SHERPA_RUN_IGNORE_NON_FATAL") or "0").strip().lower()
    if not raw:
        return False
    return raw in {"1", "true", "yes", "on"}


def _auto_stop_policy() -> str:
    raw = (os.environ.get("SHERPA_AUTO_STOP_POLICY") or "hard_fail_only").strip().lower()
    if raw in {"hard_fail_only", "legacy_mixed"}:
        return raw
    return "hard_fail_only"


def _coverage_underutilized_execs_threshold() -> int:
    raw = (os.environ.get("SHERPA_COVERAGE_UNDERUTILIZED_EXECS_THRESHOLD") or "100").strip()
    try:
        return max(0, min(int(raw), 10_000_000))
    except Exception:
        return 100


def _solve_parallelism(
    *,
    cpu_budget: int,
    n_targets: int,
    requested_outer: int,
    outer_parallelism_max: int,
    inner_workers_min: int,
    requested_inner: int,
    engine: str,
    sanitizer: str,
) -> dict[str, Any]:
    cpu = max(1, int(cpu_budget))
    targets = max(1, int(n_targets))
    outer_cap = max(1, min(int(requested_outer), int(outer_parallelism_max), targets, cpu))
    inner_min = max(1, int(inner_workers_min))
    inner_req = max(inner_min, int(requested_inner))
    sanitizer_l = (sanitizer or "").strip().lower()

    # ASAN/MSAN/TSAN are memory-heavy in multi-process mode; cap inner fanout.
    if sanitizer_l in {"address", "memory", "thread"}:
        inner_cap = max(1, min(cpu, 2))
    else:
        inner_cap = max(1, cpu)

    resolved_engine = engine if engine in {"auto", "fork", "jobs_workers", "single"} else "auto"
    reload_enabled = False

    if resolved_engine == "auto":
        if targets > 1:
            resolved_engine = "single"
            outer = outer_cap
            inner = 1
        else:
            resolved_engine = "fork"
            outer = 1
            inner = max(inner_min, min(inner_req, inner_cap, cpu))
    elif resolved_engine == "single":
        outer = outer_cap
        inner = 1
    elif resolved_engine == "fork":
        if targets > 1:
            outer = outer_cap
            inner = max(inner_min, min(inner_req, inner_cap, max(1, cpu // max(1, outer))))
        else:
            outer = 1
            inner = max(inner_min, min(inner_req, inner_cap, cpu))
    else:  # jobs_workers
        reload_enabled = True
        if targets > 1:
            outer = outer_cap
            inner = max(inner_min, min(inner_req, inner_cap, max(1, cpu // max(1, outer))))
        else:
            outer = 1
            inner = max(inner_min, min(inner_req, inner_cap, cpu))

    warning = ""
    pre_clamp_outer = int(outer)
    pre_clamp_inner = int(inner)
    while outer * inner > cpu and inner > inner_min:
        inner -= 1
    while outer * inner > cpu and outer > 1:
        outer -= 1
    if outer * inner > cpu:
        inner = 1
        outer = min(outer, cpu)
    if pre_clamp_outer != int(outer) or pre_clamp_inner != int(inner):
        warning = (
            f"parallel_budget_clamped requested_outer={requested_outer} requested_inner={requested_inner} "
            f"cpu_budget={cpu} pre_outer={pre_clamp_outer} pre_inner={pre_clamp_inner} "
            f"resolved_outer={outer} resolved_inner={inner}"
        )

    if inner <= 1 and resolved_engine != "single":
        resolved_engine = "single"
        reload_enabled = False

    return {
        "outer_parallelism": max(1, outer),
        "inner_workers": max(1, inner),
        "parallel_engine": resolved_engine,
        "reload_enabled": bool(reload_enabled),
        "warning": warning,
    }


def _time_budget_exceeded_state(state: FuzzWorkflowRuntimeState, *, step_name: str) -> FuzzWorkflowRuntimeState:
    return cast(FuzzWorkflowRuntimeState, _wf_common.time_budget_exceeded_state(cast(dict[str, Any], state), step_name=step_name))


def _make_plan_hint(repo_root: Path) -> str:
    return _wf_common.make_plan_hint(repo_root)


def _derive_plan_policy(repo_root: Path) -> tuple[bool, int]:
    return _wf_common.derive_plan_policy(repo_root)


def _load_opencode_prompt_templates() -> dict[str, str]:
    return _wf_common.load_opencode_prompt_templates()


def _render_opencode_prompt(name: str, **kwargs: object) -> str:
    return _wf_common.render_opencode_prompt(name, **kwargs)


def _default_run_rss_limit_mb() -> int:
    raw = (os.environ.get("SHERPA_RUN_RSS_LIMIT_MB") or "").strip()
    try:
        return max(256, int(raw))
    except Exception:
        pass

    def _parse_k8s_mem_mb(text: str) -> int:
        src = str(text or "").strip().lower()
        if not src:
            return 0
        m = re.fullmatch(r"([0-9]+)([a-z]+)?", src)
        if not m:
            return 0
        val = int(m.group(1) or 0)
        unit = str(m.group(2) or "")
        if unit in {"gi", "g"}:
            return val * 1024
        if unit in {"mi", "m"}:
            return val
        if unit in {"ki", "k"}:
            return max(1, val // 1024)
        if unit in {"ti", "t"}:
            return val * 1024 * 1024
        if unit == "":
            return max(1, val // (1024 * 1024))
        return 0

    limit_mb = _parse_k8s_mem_mb(os.environ.get("SHERPA_K8S_JOB_MEMORY_LIMIT", ""))
    if limit_mb > 0:
        return max(256, int(limit_mb * 0.8))
    return 131072


def _antlr_assist_enabled() -> bool:
    raw = (os.environ.get("SHERPA_ANTLR_ASSIST_ENABLED") or "1").strip().lower()
    if not raw:
        return True
    return raw in {"1", "true", "yes", "on"}


def _antlr_assist_max_files() -> int:
    raw = (os.environ.get("SHERPA_ANTLR_ASSIST_MAX_FILES") or "120").strip()
    try:
        return max(20, min(int(raw), 1000))
    except Exception:
        return 120


def _collect_antlr_assist_context(repo_root: Path) -> dict[str, Any]:
    source_exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".java"}
    skip_prefixes = (
        ".git/",
        "fuzz/out/",
        "fuzz/build/",
        "fuzz/corpus/",
        "node_modules/",
        ".next/",
        "dist/",
    )
    source_files: list[Path] = []
    grammar_files: list[Path] = []
    max_files = _antlr_assist_max_files()

    for p in sorted(repo_root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(repo_root)).replace("\\", "/")
        if any(rel.startswith(pref) for pref in skip_prefixes):
            continue
        if p.suffix.lower() in source_exts:
            source_files.append(p)
        elif p.suffix.lower() == ".g4":
            grammar_files.append(p)
        if len(source_files) >= max_files and len(grammar_files) >= 40:
            break

    def _extract_function_candidates(path: Path, text: str) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        ext = path.suffix.lower()
        if ext in {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}:
            pat = re.compile(
                r"(?m)^\s*(?:static\s+|inline\s+|extern\s+|virtual\s+|const\s+|constexpr\s+|unsigned\s+|signed\s+|long\s+|short\s+|struct\s+|class\s+|template\s*<[^>]+>\s*)*"
                r"[A-Za-z_][A-Za-z0-9_:<>\s\*&]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^;\n{}]*)\)\s*\{"
            )
            for m in pat.finditer(text):
                name = str(m.group(1) or "").strip()
                args = " ".join(str(m.group(2) or "").split())
                if name in {"if", "for", "while", "switch", "catch"}:
                    continue
                if len(name) < 2:
                    continue
                out.append(
                    {
                        "name": name,
                        "signature": f"{name}({args})"[:240],
                        "file": str(path.relative_to(repo_root)).replace("\\", "/"),
                    }
                )
                if len(out) >= 30:
                    break
        elif ext == ".java":
            pat = re.compile(
                r"(?m)^\s*(?:public|protected|private|static|final|native|synchronized|abstract|\s)+"
                r"[A-Za-z_][A-Za-z0-9_<>\[\]]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*\{"
            )
            for m in pat.finditer(text):
                name = str(m.group(1) or "").strip()
                args = " ".join(str(m.group(2) or "").split())
                out.append(
                    {
                        "name": name,
                        "signature": f"{name}({args})"[:240],
                        "file": str(path.relative_to(repo_root)).replace("\\", "/"),
                    }
                )
                if len(out) >= 30:
                    break
        return out

    function_candidates: list[dict[str, str]] = []
    parser_rules: list[str] = []
    lexer_rules: list[str] = []
    grammar_start_rules: list[dict[str, str]] = []

    for p in source_files[:max_files]:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        function_candidates.extend(_extract_function_candidates(p, text))
        if len(function_candidates) >= 300:
            break

    for g4 in grammar_files[:40]:
        try:
            text = g4.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        prules = re.findall(r"(?m)^\s*([a-z][A-Za-z0-9_]*)\s*:", text)
        lrules = re.findall(r"(?m)^\s*([A-Z][A-Z0-9_]*)\s*:", text)
        if prules:
            grammar_start_rules.append(
                {
                    "grammar": str(g4.relative_to(repo_root)).replace("\\", "/"),
                    "start_rule": prules[0],
                }
            )
        parser_rules.extend(prules[:50])
        lexer_rules.extend(lrules[:80])

    unique_funcs: list[dict[str, str]] = []
    seen_func = set()
    for item in function_candidates:
        key = (item.get("name"), item.get("file"))
        if key in seen_func:
            continue
        seen_func.add(key)
        unique_funcs.append(item)
    unique_funcs = unique_funcs[:120]

    entrypoint_keywords = ("parse", "decode", "read", "load", "process", "handle", "consume")
    entrypoint_candidates = [
        item for item in unique_funcs if any(k in str(item.get("name") or "").lower() for k in entrypoint_keywords)
    ][:30]

    return {
        "mode": "antlr-assisted-static-context",
        "enabled": True,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "repo_root": str(repo_root),
        "source_files_scanned": [str(p.relative_to(repo_root)).replace("\\", "/") for p in source_files[:max_files]],
        "grammar_files": [str(p.relative_to(repo_root)).replace("\\", "/") for p in grammar_files[:40]],
        "antlr_grammar_start_rules": grammar_start_rules,
        "parser_rules": sorted(set(parser_rules))[:200],
        "lexer_rules": sorted(set(lexer_rules))[:200],
        "candidate_functions": unique_funcs,
        "entrypoint_candidates": entrypoint_candidates,
    }


def _prepare_antlr_assist_context(repo_root: Path) -> tuple[str, str]:
    if not _antlr_assist_enabled():
        return "", ""
    try:
        doc = _collect_antlr_assist_context(repo_root)
        fuzz_dir = repo_root / "fuzz"
        fuzz_dir.mkdir(parents=True, exist_ok=True)
        ctx_path = fuzz_dir / "antlr_plan_context.json"
        ctx_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        top_funcs = [str(x.get("name") or "") for x in (doc.get("entrypoint_candidates") or [])[:8] if x.get("name")]
        summary = (
            f"antlr_context_file=fuzz/antlr_plan_context.json; "
            f"grammar_files={len(doc.get('grammar_files') or [])}; "
            f"candidate_functions={len(doc.get('candidate_functions') or [])}; "
            f"entrypoints={', '.join(top_funcs) if top_funcs else 'n/a'}"
        )
        return str(ctx_path), summary
    except Exception:
        return "", ""


def _collect_target_analysis_context(repo_root: Path) -> dict[str, Any]:
    def _ext_to_ts_language(ext: str) -> str:
        ext = ext.lower()
        if ext in {".c", ".h"}:
            return "c"
        if ext in {".cc", ".cpp", ".cxx", ".hh", ".hpp"}:
            return "cpp"
        if ext == ".java":
            return "java"
        return ""

    def _safe_get_parser(language: str, timeout_sec: float = 5.0) -> Any:
        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
            result = {}

            def _get_parser_worker():
                tslp = importlib.import_module("tree_sitter_language_pack")
                get_parser = getattr(tslp, "get_parser", None)
                if callable(get_parser):
                    result["parser"] = get_parser(language)

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_get_parser_worker)
                future.result(timeout=timeout_sec)
                return result.get("parser")
        except (FuturesTimeoutError, Exception):
            return None

    def _extract_tree_sitter_functions(path: Path, rel: str) -> list[dict[str, Any]]:
        try:
            language = _ext_to_ts_language(path.suffix)
            if not language:
                return []
            parser = _safe_get_parser(language, timeout_sec=5.0)
            if parser is None:
                return []
            data = path.read_bytes()
            tree = parser.parse(data)
            out: list[dict[str, Any]] = []

            def _node_text(node: Any) -> str:
                try:
                    return data[int(node.start_byte) : int(node.end_byte)].decode("utf-8", errors="replace")
                except Exception:
                    return ""

            def _walk(node: Any) -> None:
                if len(out) >= 80:
                    return
                node_type = str(getattr(node, "type", "") or "")
                if node_type in {"function_definition", "method_declaration"}:
                    snippet = _node_text(node)
                    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", snippet)
                    name = str(m.group(1) or "").strip() if m else ""
                    if name and name not in {"if", "for", "while", "switch", "catch"}:
                        out.append(
                            {
                                "name": name,
                                "signature": " ".join(snippet.split())[:240],
                                "file": rel,
                                "line": int(getattr(node, "start_point", (0, 0))[0]) + 1,
                                "target_type": "pending",
                                "seed_profile": "pending",
                                "risk_signals": [],
                                "analysis_source": "tree-sitter",
                            }
                        )
                for child in getattr(node, "children", []) or []:
                    _walk(child)

            _walk(tree.root_node)
            return out
        except Exception:
            return []

    def _run_semgrep_rules(root: Path) -> tuple[bool, dict[str, list[str]]]:
        semgrep_bin = shutil.which("semgrep")
        if not semgrep_bin:
            logger.info("[semgrep] not found on PATH, skipping")
            return False, {}
        tmp_path = ""
        _SEMGREP_TIMEOUT = 60  # seconds — hard cap to avoid blocking analysis
        rules_doc = {
            "rules": [
                {
                    "id": "parser-like",
                    "languages": ["c", "cpp", "java"],
                    "message": "parser-like",
                    "severity": "INFO",
                    "pattern-regex": r"(parse|scan|lexer|token|load|decode|emit|dump|serialize|format|arg_id)",
                },
                {
                    "id": "bounds",
                    "languages": ["c", "cpp", "java"],
                    "message": "bounds",
                    "severity": "INFO",
                    "pattern-regex": r"(memcpy|memmove|strncpy|size_t|length|len|offset|index)",
                },
                {
                    "id": "state-machine",
                    "languages": ["c", "cpp", "java"],
                    "message": "state-machine",
                    "severity": "INFO",
                    "pattern-regex": r"(state|transition|consume|next|advance|dispatch|handler)",
                },
            ]
        }
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".yml", encoding="utf-8", delete=False) as fh:
                json.dump(rules_doc, fh)
                tmp_path = fh.name
            logger.info(f"[semgrep] scanning {root} (timeout={_SEMGREP_TIMEOUT}s)")
            cmd = [
                semgrep_bin, "scan", "--json",
                "--metrics=off",             # prevent telemetry network call (blocks in containers)
                "--disable-version-check",   # prevent update check network call
                "--config", tmp_path,
                str(root),
            ]
            # Use Popen + process group so timeout kills the entire process tree
            # (semgrep forks workers that subprocess.run timeout may not reach)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,       # creates a new process group
            )
            try:
                stdout, stderr = proc.communicate(timeout=_SEMGREP_TIMEOUT)
            except subprocess.TimeoutExpired:
                # Kill the entire process group (semgrep + its workers)
                import signal
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
                proc.wait(timeout=5)
                logger.info(f"[semgrep] TIMEOUT after {_SEMGREP_TIMEOUT}s, skipping")
                return True, {}
            if proc.returncode not in {0, 1}:
                logger.info(f"[semgrep] exited with code {proc.returncode}, stderr: {(stderr or '')[:200]}")
                return True, {}
            logger.info(f"[semgrep] scan completed (rc={proc.returncode})")
            doc = json.loads(stdout or "{}")
            result_map: dict[str, list[str]] = {}
            for item in doc.get("results") or []:
                path = str(((item.get("path") or "") if isinstance(item, dict) else "")).strip()
                rule_id = str(((item.get("check_id") or "") if isinstance(item, dict) else "")).strip()
                if not path or not rule_id:
                    continue
                rel = str(Path(path).resolve().relative_to(root.resolve())).replace("\\", "/") if Path(path).is_absolute() else path.replace("\\", "/")
                result_map.setdefault(rel, [])
                if rule_id not in result_map[rel]:
                    result_map[rel].append(rule_id)
            logger.info(f"[semgrep] found hits in {len(result_map)} files")
            return True, result_map
        except Exception as exc:
            logger.info(f"[semgrep] unexpected error: {exc}")
            return True, {}
        finally:
            try:
                if tmp_path:
                    os.unlink(tmp_path)
            except Exception:
                pass

    tree_sitter_enabled = importlib.util.find_spec("tree_sitter_language_pack") is not None
    semgrep_enabled, semgrep_hits = _run_semgrep_rules(repo_root)

    source_exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".java"}
    skip_prefixes = (
        ".git/",
        "fuzz/out/",
        "fuzz/build/",
        "fuzz/corpus/",
        "node_modules/",
        ".next/",
        "dist/",
    )
    source_files: list[Path] = []
    for p in sorted(repo_root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(repo_root)).replace("\\", "/")
        if any(rel.startswith(pref) for pref in skip_prefixes):
            continue
        if p.suffix.lower() in source_exts:
            source_files.append(p)
        if len(source_files) >= 120:
            break

    semgrep_rules = [
        {"id": "parser-like", "pattern": r"(parse|scan|lexer|token|load|decode|emit|dump|serialize|format|arg_id)"},
        {"id": "bounds", "pattern": r"(memcpy|memmove|strncpy|size_t|length|len|offset|index)"},
        {"id": "state-machine", "pattern": r"(state|transition|consume|next|advance|dispatch|handler)"},
    ]
    candidate_functions: list[dict[str, Any]] = []
    for p in source_files:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rel = str(p.relative_to(repo_root)).replace("\\", "/")
        ts_candidates = _extract_tree_sitter_functions(p, rel) if tree_sitter_enabled else []
        if ts_candidates:
            candidate_functions.extend(ts_candidates[:40])
            if len(candidate_functions) >= 240:
                break

        matches = re.finditer(
            r"(?m)^\s*(?:static\s+|inline\s+|extern\s+|virtual\s+|const\s+|constexpr\s+|unsigned\s+|signed\s+|long\s+|short\s+|struct\s+|class\s+|template\s*<[^>]+>\s*)*"
            r"[A-Za-z_][A-Za-z0-9_:<>\s\*&]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^;\n{}]*)\)\s*\{",
            text,
        )
        for m in matches:
            name = str(m.group(1) or "").strip()
            if name in {"if", "for", "while", "switch", "catch"} or len(name) < 2:
                continue
            signature = f"{name}({' '.join(str(m.group(2) or '').split())})"[:240]
            line_no = text[: m.start()].count("\n") + 1
            risk_signals = [rule["id"] for rule in semgrep_rules if re.search(rule["pattern"], f"{name}\n{signature}", re.IGNORECASE)]
            for rule_id in semgrep_hits.get(rel, []):
                if rule_id not in risk_signals:
                    risk_signals.append(rule_id)
            candidate_functions.append(
                {
                    "name": name,
                    "signature": signature,
                    "file": rel,
                    "line": line_no,
                    "target_type": "pending",
                    "seed_profile": "pending",
                    "risk_signals": risk_signals,
                    "analysis_source": "regex",
                }
            )
            if len(candidate_functions) >= 240:
                break
        if len(candidate_functions) >= 240:
            break

    for item in candidate_functions:
        depth_score, depth_class, selection_bias_reason = _score_target_depth(
            str(item.get("name") or ""),
            str(item.get("signature") or ""),
            target_type=str(item.get("target_type") or "generic"),
            risk_signals=list(item.get("risk_signals") or []),
        )
        runtime_viability, selection_rationale, replacement_candidates = _runtime_viability_details(
            str(item.get("name") or ""),
            str(item.get("signature") or ""),
            file_hint=str(item.get("file") or ""),
        )
        item["depth_score"] = depth_score
        item["depth_class"] = depth_class
        item["selection_bias_reason"] = selection_bias_reason
        item["runtime_viability"] = runtime_viability
        item["selection_rationale"] = selection_rationale
        item["runtime_replacement_candidates"] = replacement_candidates

    candidate_functions.sort(
        key=lambda item: (
            {"high": 2, "medium": 1, "low": 0}.get(str(item.get("runtime_viability") or "").lower(), 0),
            int(item.get("depth_score") or 0),
            len(list(item.get("risk_signals") or [])),
            str(item.get("name") or ""),
        ),
        reverse=True,
    )

    recommended_targets = []
    seen: set[tuple[str, str]] = set()
    has_deep = any(str(item.get("depth_class") or "") == "deep" for item in candidate_functions)
    for item in candidate_functions:
        risk = list(item.get("risk_signals") or [])
        if not risk and str(item.get("target_type") or "") == "generic":
            continue
        if has_deep and str(item.get("depth_class") or "") == "shallow":
            continue
        key = (str(item.get("name") or ""), str(item.get("file") or ""))
        if key in seen:
            continue
        seen.add(key)
        recommended_targets.append(
            {
                "name": str(item.get("name") or ""),
                "api": str(item.get("name") or ""),
                "lang": _infer_target_lang_from_repo(repo_root, file_hint=str(item.get("file") or "")),
                "target_type": str(item.get("target_type") or "generic"),
                "seed_profile": str(item.get("seed_profile") or "generic"),
                "risk_signals": risk,
                "file": str(item.get("file") or ""),
                "depth_score": int(item.get("depth_score") or 0),
                "depth_class": str(item.get("depth_class") or "shallow"),
                "selection_bias_reason": str(item.get("selection_bias_reason") or ""),
                "runtime_viability": str(item.get("runtime_viability") or ""),
                "selection_rationale": str(item.get("selection_rationale") or ""),
                "runtime_replacement_candidates": list(item.get("runtime_replacement_candidates") or []),
            }
        )
        if len(recommended_targets) >= 24:
            break

    return {
        "mode": "tool-assisted-target-analysis",
        "enabled": True,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "repo_root": str(repo_root),
        "source_files_scanned": [str(p.relative_to(repo_root)).replace("\\", "/") for p in source_files],
        "candidate_functions": candidate_functions,
        "recommended_targets": recommended_targets,
        "rules": semgrep_rules,
        "tree_sitter_enabled": tree_sitter_enabled,
        "semgrep_enabled": semgrep_enabled,
        "analysis_backend": "regex-fallback",
    }


def _prepare_target_analysis_context(repo_root: Path) -> tuple[str, str]:
    try:
        doc = _collect_target_analysis_context(repo_root)
        fuzz_dir = repo_root / "fuzz"
        fuzz_dir.mkdir(parents=True, exist_ok=True)
        ctx_path = fuzz_dir / "target_analysis.json"
        ctx_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        top_targets = [
            f"{str(x.get('name') or '')}:{str(x.get('seed_profile') or '')}"
            for x in (doc.get("recommended_targets") or [])[:8]
            if x.get("name")
        ]
        summary = (
            f"target_analysis_file=fuzz/target_analysis.json; "
            f"candidates={len(doc.get('candidate_functions') or [])}; "
            f"recommended={', '.join(top_targets) if top_targets else 'n/a'}"
        )
        return str(ctx_path), summary
    except Exception:
        return "", ""


def _collect_analysis_companion_context() -> tuple[dict[str, Any], str]:
    job_id = str(os.environ.get("SHERPA_JOB_ID") or "").strip()
    base_output = Path(os.environ.get("SHERPA_OUTPUT_DIR", "/shared/output")).expanduser()
    companion_root = (
        (base_output / "_k8s_jobs" / job_id / "promefuzz").resolve()
        if job_id
        else None
    )
    out: dict[str, Any] = {
        "job_id": job_id,
        "companion_root": str(companion_root) if companion_root else "",
        "artifacts": {},
    }
    if not companion_root or not companion_root.is_dir():
        return out, "companion_artifacts=0"

    artifacts: dict[str, Any] = {}
    found = 0
    for name in ("status.json", "preprocess.json", "coverage_hints.json"):
        p = companion_root / name
        doc: dict[str, Any] = {
            "path": str(p),
            "exists": p.is_file(),
        }
        if p.is_file():
            found += 1
            try:
                raw = p.read_text(encoding="utf-8", errors="replace")
                parsed = json.loads(raw)
                if isinstance(parsed, (dict, list)):
                    doc["json"] = parsed
                else:
                    doc["text"] = str(parsed)[:4000]
            except Exception:
                try:
                    doc["text"] = p.read_text(encoding="utf-8", errors="replace")[-4000:]
                except Exception:
                    doc["text"] = ""
        artifacts[name] = doc
    out["artifacts"] = artifacts
    summary_parts = [f"companion_artifacts={found}", f"companion_root={companion_root}"]
    status_doc = ((artifacts.get("status.json") or {}) if isinstance(artifacts, dict) else {}).get("json")
    if isinstance(status_doc, dict):
        state_val = str(status_doc.get("state") or "").strip()
        backend_val = str(status_doc.get("analysis_backend") or "").strip()
        candidate_count = status_doc.get("candidate_count")
        embedding_ok = status_doc.get("embedding_ok")
        rag_degraded = status_doc.get("rag_degraded")
        semantic_hit_rate = status_doc.get("semantic_hit_rate")
        if state_val:
            summary_parts.append(f"state={state_val}")
        if backend_val:
            summary_parts.append(f"backend={backend_val}")
        if candidate_count is not None:
            try:
                summary_parts.append(f"candidates={int(candidate_count)}")
            except Exception:
                pass
        if embedding_ok is not None:
            summary_parts.append(f"embedding_ok={int(bool(embedding_ok))}")
        if rag_degraded is not None:
            summary_parts.append(f"rag_degraded={int(bool(rag_degraded))}")
        if semantic_hit_rate is not None:
            try:
                summary_parts.append(f"semantic_hit_rate={round(float(semantic_hit_rate), 3)}")
            except Exception:
                pass
    hints_doc = ((artifacts.get("coverage_hints.json") or {}) if isinstance(artifacts, dict) else {}).get("json")
    if isinstance(hints_doc, dict):
        targets = hints_doc.get("recommended_targets")
        if isinstance(targets, list):
            summary_parts.append(f"hint_targets={len(targets)}")
    return out, "; ".join(summary_parts)


def _read_json_doc(path_text: str) -> dict[str, Any]:
    path = Path(str(path_text or "").strip())
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return dict(raw) if isinstance(raw, dict) else {}


def _build_analysis_evidence_index(
    *,
    repo_root: Path,
    antlr_doc: dict[str, Any],
    target_doc: dict[str, Any],
    companion_doc: dict[str, Any],
) -> dict[str, Any]:
    evidence_counter = 0
    evidence_index: dict[str, dict[str, Any]] = {}

    def _new_evidence_id() -> str:
        nonlocal evidence_counter
        evidence_counter += 1
        return f"EV{evidence_counter:04d}"

    def _add_evidence(
        *,
        kind: str,
        source_path: str,
        summary: str,
        score: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        ev_id = _new_evidence_id()
        evidence_index[ev_id] = {
            "id": ev_id,
            "kind": str(kind or "").strip() or "unknown",
            "source_path": str(source_path or "").strip(),
            "summary": str(summary or "").strip()[:800],
            "score": float(score) if score is not None else None,
            "payload": dict(payload or {}),
        }
        return ev_id

    api_inventory: list[dict[str, Any]] = []
    seen_api_keys: set[tuple[str, str, str]] = set()
    for item in list(target_doc.get("recommended_targets") or [])[:80]:
        if not isinstance(item, dict):
            continue
        api = str(item.get("api") or item.get("name") or "").strip()
        file_hint = str(item.get("file") or "").strip()
        target_type = str(item.get("target_type") or "").strip().lower()
        if not api:
            continue
        key = (api, file_hint, target_type)
        if key in seen_api_keys:
            continue
        seen_api_keys.add(key)
        ev_id = _add_evidence(
            kind="target_analysis",
            source_path="fuzz/target_analysis.json",
            summary=f"recommended target `{api}` ({target_type or 'generic'})",
            score=float(item.get("depth_score") or 0.0),
            payload={
                "target_type": target_type,
                "seed_profile": str(item.get("seed_profile") or ""),
                "runtime_viability": str(item.get("runtime_viability") or ""),
                "file": file_hint,
            },
        )
        api_inventory.append(
            {
                "evidence_id": ev_id,
                "api": api,
                "file": file_hint,
                "target_type": target_type or "generic",
                "seed_profile": str(item.get("seed_profile") or ""),
                "runtime_viability": str(item.get("runtime_viability") or ""),
            }
        )

    for item in list(antlr_doc.get("entrypoint_candidates") or [])[:80]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        file_hint = str(item.get("file") or "").strip()
        if not name:
            continue
        key = (name, file_hint, "entrypoint")
        if key in seen_api_keys:
            continue
        seen_api_keys.add(key)
        ev_id = _add_evidence(
            kind="antlr_entrypoint",
            source_path="fuzz/antlr_plan_context.json",
            summary=f"antlr/static entrypoint candidate `{name}`",
            payload={
                "file": file_hint,
                "signature": str(item.get("signature") or ""),
                "line": int(item.get("line") or 0),
            },
        )
        api_inventory.append(
            {
                "evidence_id": ev_id,
                "api": name,
                "file": file_hint,
                "target_type": str(item.get("target_type") or "generic"),
                "seed_profile": str(item.get("seed_profile") or ""),
                "runtime_viability": str(item.get("runtime_viability") or ""),
            }
        )

    callgraph_summary: list[dict[str, Any]] = []
    artifacts = dict(companion_doc.get("artifacts") or {})
    coverage_hints = dict(((artifacts.get("coverage_hints.json") or {}) if isinstance(artifacts, dict) else {}).get("json") or {})
    preprocess_doc = dict(((artifacts.get("preprocess.json") or {}) if isinstance(artifacts, dict) else {}).get("json") or {})
    status_doc = dict(((artifacts.get("status.json") or {}) if isinstance(artifacts, dict) else {}).get("json") or {})

    for item in list(coverage_hints.get("callgraph_summary") or [])[:40]:
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary") or item.get("edge") or "").strip()
        if not summary:
            continue
        ev_id = _add_evidence(
            kind="callgraph_summary",
            source_path="promefuzz/coverage_hints.json",
            summary=summary,
            score=float(item.get("score") or 0.0) if item.get("score") is not None else None,
            payload=item,
        )
        callgraph_summary.append({"evidence_id": ev_id, **item})

    consumer_patterns: list[dict[str, Any]] = []
    for item in list(preprocess_doc.get("consumer_patterns") or preprocess_doc.get("api_usage_patterns") or [])[:40]:
        if not isinstance(item, dict):
            continue
        pattern = str(item.get("pattern") or item.get("summary") or item.get("api") or "").strip()
        if not pattern:
            continue
        ev_id = _add_evidence(
            kind="consumer_pattern",
            source_path="promefuzz/preprocess.json",
            summary=pattern,
            payload=item,
        )
        consumer_patterns.append({"evidence_id": ev_id, **item})

    semantic_evidence: list[dict[str, Any]] = []
    semantic_sources: list[Any] = []
    for key in ("semantic_evidence", "semantic_findings", "retrieved_documents"):
        value = coverage_hints.get(key)
        if isinstance(value, list):
            semantic_sources.extend(value[:50])
    for item in semantic_sources[:80]:
        if not isinstance(item, dict):
            continue
        summary = str(item.get("snippet") or item.get("summary") or item.get("claim") or "").strip()
        if not summary:
            continue
        score_raw = item.get("score")
        score: float | None = None
        if score_raw is not None:
            try:
                score = float(score_raw)
            except Exception:
                score = None
        ev_id = _add_evidence(
            kind="semantic_evidence",
            source_path=str(item.get("source_path") or item.get("source") or "promefuzz/coverage_hints.json"),
            summary=summary,
            score=score,
            payload=item,
        )
        semantic_evidence.append({"evidence_id": ev_id, **item})

    if status_doc:
        _add_evidence(
            kind="companion_status",
            source_path="promefuzz/status.json",
            summary=(
                f"companion state={status_doc.get('state') or 'unknown'}, "
                f"backend={status_doc.get('analysis_backend') or 'unknown'}, "
                f"rag_ok={int(bool(status_doc.get('rag_ok')))}"
            ),
            payload={
                "state": str(status_doc.get("state") or ""),
                "analysis_backend": str(status_doc.get("analysis_backend") or ""),
                "semantic_hit_rate": status_doc.get("semantic_hit_rate"),
                "cache_hit_rate": status_doc.get("cache_hit_rate"),
            },
        )

    return {
        "analysis_version": 2,
        "generated_at": int(time.time()),
        "repo_root": str(repo_root),
        "api_inventory": api_inventory,
        "callgraph_summary": callgraph_summary,
        "consumer_patterns": consumer_patterns,
        "semantic_evidence": semantic_evidence,
        "evidence_index": evidence_index,
        "summary": {
            "evidence_count": len(evidence_index),
            "api_inventory_count": len(api_inventory),
            "callgraph_summary_count": len(callgraph_summary),
            "consumer_pattern_count": len(consumer_patterns),
            "semantic_evidence_count": len(semantic_evidence),
        },
    }


@dataclass(frozen=True)
class FuzzWorkflowInput:
    repo_url: str
    email: Optional[str]
    time_budget: int
    run_time_budget: int
    max_len: int
    docker_image: Optional[str]
    ai_key_path: Path
    model: Optional[str] = None
    resume_from_step: Optional[str] = None
    resume_repo_root: Optional[Path] = None
    stop_after_step: Optional[str] = None
    last_fuzzer: Optional[str] = None
    last_crash_artifact: Optional[str] = None
    re_workspace_root: Optional[str] = None
    coverage_loop_max_rounds: int = 0
    max_fix_rounds: int = 0
    same_error_max_retries: int = 0
    restart_to_plan_reason: str = ""
    restart_to_plan_stage: str = ""
    restart_to_plan_error_text: str = ""
    restart_to_plan_report_path: str = ""


def _analysis_companion_enabled() -> bool:
    raw = str(os.environ.get("SHERPA_K8S_ANALYSIS_COMPANION_ENABLED", "1") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _promefuzz_mcp_root_exists() -> bool:
    root = Path(str(os.environ.get("SHERPA_PROMEFUZZ_MCP_ROOT") or "/app/promefuzz-mcp")).expanduser()
    return root.exists() and root.is_dir()


def _check_promefuzz_runtime_deps() -> tuple[bool, str]:
    # PromeFuzz C++ processors now depend on system nlohmann-json3-dev.
    candidates = [
        Path("/usr/include/nlohmann/json.hpp"),
        Path("/usr/local/include/nlohmann/json.hpp"),
    ]
    for path in candidates:
        if path.is_file():
            return True, ""
    return (
        False,
        "missing system header nlohmann/json.hpp; install nlohmann-json3-dev in the runtime image",
    )


def _node_init(state: FuzzWorkflowState) -> FuzzWorkflowRuntimeState:
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> init")
    repo_url = (state.get("repo_url") or "").strip()
    if not repo_url:
        raise ValueError("repo_url is required")

    ai_key_path = Path(state.get("ai_key_path") or "").expanduser().resolve()
    if not ai_key_path:
        raise ValueError("ai_key_path is required")

    time_budget = _wf_common.parse_budget_value(state.get("time_budget"), default=900)
    run_time_budget_raw = state.get("run_time_budget")
    if run_time_budget_raw is None:
        run_time_budget = time_budget
    else:
        run_time_budget = _wf_common.parse_budget_value(run_time_budget_raw, default=time_budget)
    if time_budget < 0:
        raise ValueError("time_budget must be >= 0")
    if run_time_budget < 0:
        raise ValueError("run_time_budget must be >= 0")
    max_len_raw = state.get("max_len")
    max_len = int(max_len_raw) if max_len_raw is not None else 0
    docker_image = (state.get("docker_image") or "").strip() or None
    codex_cli = (os.environ.get("SHERPA_CODEX_CLI") or os.environ.get("CODEX_CLI") or "opencode").strip()

    if _analysis_companion_enabled() and _promefuzz_mcp_root_exists():
        dep_ok, dep_err = _check_promefuzz_runtime_deps()
        if not dep_ok:
            raise RuntimeError(f"init prerequisite failed: {dep_err}")

    raw_resume_repo_root = (state.get("resume_repo_root") or "").strip()
    workdir: Path | None = None
    if raw_resume_repo_root:
        candidate = Path(raw_resume_repo_root).expanduser().resolve()
        if candidate.exists() and candidate.is_dir():
            workdir = candidate
    if workdir is None:
        workdir = _alloc_output_workdir(repo_url)
    generator = NonOssFuzzHarnessGenerator(
        repo_spec=RepoSpec(url=repo_url, workdir=workdir),
        ai_key_path=ai_key_path,
        max_len=max_len,
        time_budget_per_target=run_time_budget,
        rss_limit_mb=_default_run_rss_limit_mb(),
        docker_image=docker_image,
        codex_cli=codex_cli,
    )

    resume_step = (state.get("resume_from_step") or "").strip().lower()

    out = cast(
        FuzzWorkflowRuntimeState,
        {
            **state,
            "generator": generator,
            "crash_found": False,
            "message": "initialized",
            "plan_retry_reason": str(state.get("plan_retry_reason") or ""),
            "plan_targets_schema_valid_before_retry": bool(state.get("plan_targets_schema_valid_before_retry") or False),
            "plan_targets_schema_valid_after_retry": bool(state.get("plan_targets_schema_valid_after_retry") or False),
            "plan_used_fallback_targets": bool(state.get("plan_used_fallback_targets") or False),
            "step_count": int(state.get("step_count") or 0),
            "max_steps": int(state.get("max_steps")) if state.get("max_steps") is not None else 0,
            "last_step": "init",
            "last_error": "",
            "build_rc": 0,
            "build_stdout_tail": "",
            "build_stderr_tail": "",
            "build_full_log_path": "",
            "build_error_signature": "",
            "build_error_signature_before": "",
            "build_error_signature_after": "",
            "same_build_error_repeats": 0,
            "same_error_max_retries": max(
                0,
                int(
                    state.get("same_error_max_retries")
                    if state.get("same_error_max_retries") is not None
                    else 0
                ),
            ),
            "build_error_kind": "",
            "build_error_code": "",
            "build_error_signature_short": "",
            "build_attempts": int(state.get("build_attempts") or 0),
            "fix_build_attempts": int(state.get("fix_build_attempts") or 0),
            "max_fix_rounds": max(
                0,
                int(state.get("max_fix_rounds") if state.get("max_fix_rounds") is not None else 0),
            ),
            "fix_build_noop_streak": int(state.get("fix_build_noop_streak") or 0),
            "fix_build_attempt_history": list(state.get("fix_build_attempt_history") or []),
            "fix_build_rule_hits": list(state.get("fix_build_rule_hits") or []),
            "fix_build_terminal_reason": str(state.get("fix_build_terminal_reason") or ""),
            "fix_build_last_diff_paths": list(state.get("fix_build_last_diff_paths") or []),
            "fix_action_type": "",
            "fix_effect": "",
            "codex_hint": "",
            "failed": False,
            "repo_root": str(generator.repo_root),
            "run_rc": 0,
            "crash_evidence": "none",
            "run_error_kind": "",
            "run_terminal_reason": "",
            "run_idle_seconds": 0,
            "run_children_exit_count": 0,
            "last_crash_artifact": str(state.get("last_crash_artifact") or ""),
            "last_fuzzer": str(state.get("last_fuzzer") or ""),
            "crash_signature": "",
            "same_crash_repeats": 0,
            "crash_fix_attempts": int(state.get("crash_fix_attempts") or 0),
            "crash_repro_done": bool(state.get("crash_repro_done") or False),
            "crash_repro_ok": bool(state.get("crash_repro_ok") or False),
            "crash_repro_rc": int(state.get("crash_repro_rc") or 0),
            "crash_repro_report_path": str(state.get("crash_repro_report_path") or ""),
            "crash_repro_json_path": str(state.get("crash_repro_json_path") or ""),
            "crash_triage_done": bool(state.get("crash_triage_done") or False),
            "crash_triage_label": str(state.get("crash_triage_label") or ""),
            "crash_triage_confidence": float(state.get("crash_triage_confidence") or 0.0),
            "crash_triage_reason": str(state.get("crash_triage_reason") or ""),
            "crash_triage_signal_lines": list(state.get("crash_triage_signal_lines") or []),
            "crash_triage_report_path": str(state.get("crash_triage_report_path") or ""),
            "crash_triage_json_path": str(state.get("crash_triage_json_path") or ""),
            "re_build_done": bool(state.get("re_build_done") or False),
            "re_build_ok": bool(state.get("re_build_ok") or False),
            "re_build_rc": int(state.get("re_build_rc") or 0),
            "re_build_report_path": str(state.get("re_build_report_path") or ""),
            "re_build_json_path": str(state.get("re_build_json_path") or ""),
            "re_run_done": bool(state.get("re_run_done") or False),
            "re_run_ok": bool(state.get("re_run_ok") or False),
            "re_run_rc": int(state.get("re_run_rc") or 0),
            "re_run_report_path": str(state.get("re_run_report_path") or ""),
            "re_run_json_path": str(state.get("re_run_json_path") or ""),
            "re_workspace_root": str(state.get("re_workspace_root") or ""),
            "restart_to_plan": bool(state.get("restart_to_plan") or False),
            "restart_to_plan_reason": str(state.get("restart_to_plan_reason") or ""),
            "restart_to_plan_stage": str(state.get("restart_to_plan_stage") or ""),
            "restart_to_plan_error_text": str(state.get("restart_to_plan_error_text") or ""),
            "restart_to_plan_report_path": str(state.get("restart_to_plan_report_path") or ""),
            "restart_to_plan_count": int(state.get("restart_to_plan_count") or 0),
            "fix_harness_attempts": int(state.get("fix_harness_attempts") or 0),
            "plan_fix_on_crash": True,
            "plan_max_fix_rounds": 0,
            "repair_mode": bool(state.get("repair_mode") or False),
            "repair_origin_stage": str(state.get("repair_origin_stage") or ""),
            "repair_error_kind": str(state.get("repair_error_kind") or ""),
            "repair_error_code": str(state.get("repair_error_code") or ""),
            "repair_signature": str(state.get("repair_signature") or ""),
            "repair_stdout_tail": str(state.get("repair_stdout_tail") or ""),
            "repair_stderr_tail": str(state.get("repair_stderr_tail") or ""),
            "repair_recent_attempts": list(state.get("repair_recent_attempts") or []),
            "coverage_loop_max_rounds": max(
                0,
                int(
                    state.get("coverage_loop_max_rounds")
                    if state.get("coverage_loop_max_rounds") is not None
                    else 0
                ),
            ),
            "coverage_loop_round": int(state.get("coverage_loop_round") or 0),
            "coverage_should_improve": bool(state.get("coverage_should_improve") or False),
            "coverage_improve_reason": str(state.get("coverage_improve_reason") or ""),
            "coverage_bottleneck_kind": str(state.get("coverage_bottleneck_kind") or ""),
            "coverage_bottleneck_reason": str(state.get("coverage_bottleneck_reason") or ""),
            "coverage_history": list(state.get("coverage_history") or []),
            "coverage_target_name": str(state.get("coverage_target_name") or ""),
            "coverage_seed_profile": str(state.get("coverage_seed_profile") or ""),
            "coverage_plateau_streak": int(state.get("coverage_plateau_streak") or 0),
            "coverage_last_max_cov": int(state.get("coverage_last_max_cov") or 0),
            "coverage_last_ft": int(state.get("coverage_last_ft") or 0),
            "coverage_replan_required": bool(state.get("coverage_replan_required") or False),
            "coverage_improve_mode": str(state.get("coverage_improve_mode") or ""),
            "coverage_round_budget_exhausted": bool(state.get("coverage_round_budget_exhausted") or False),
            "coverage_stop_reason": str(state.get("coverage_stop_reason") or ""),
            "coverage_corpus_sources": list(state.get("coverage_corpus_sources") or []),
            "coverage_seed_counts": dict(state.get("coverage_seed_counts") or {}),
            "analysis_done": bool(state.get("analysis_done") or False),
            "analysis_degraded": bool(state.get("analysis_degraded") or False),
            "analysis_error": str(state.get("analysis_error") or ""),
            "analysis_report_path": str(state.get("analysis_report_path") or ""),
            "analysis_context_path": str(state.get("analysis_context_path") or ""),
            "analysis_evidence_count": int(state.get("analysis_evidence_count") or 0),
            "antlr_context_path": str(state.get("antlr_context_path") or ""),
            "antlr_context_summary": str(state.get("antlr_context_summary") or ""),
            "target_analysis_path": str(state.get("target_analysis_path") or ""),
            "target_analysis_summary": str(state.get("target_analysis_summary") or ""),
            "target_scoring_enabled": bool(state.get("target_scoring_enabled") or False),
            "constraint_memory_count": int(state.get("constraint_memory_count") or 0),
            "constraint_memory_path": str(state.get("constraint_memory_path") or ""),
        },
    )

    # Restore crash context from previous run stage when repro/fix is resumed
    # as a separate k8s stage job. Without this, init resets crash state and
    # repro_crash would be incorrectly skipped.
    if resume_step in {
        "analysis",
        "plan",
        "synthesize",
        "build",
        "run",
        "crash-triage",
        "fix-harness",
        "coverage-analysis",
        "improve-harness",
        "repro_crash",
        "re-build",
        "re-run",
    }:
        try:
            repro_doc = _read_repro_context(generator.repo_root)
            if isinstance(repro_doc, dict):
                if not str(out.get("last_fuzzer") or "").strip():
                    out["last_fuzzer"] = str(repro_doc.get("last_fuzzer") or "")
                if not str(out.get("last_crash_artifact") or "").strip():
                    out["last_crash_artifact"] = str(repro_doc.get("last_crash_artifact") or "")
                if not str(out.get("re_workspace_root") or "").strip():
                    out["re_workspace_root"] = str(repro_doc.get("re_workspace_root") or "")
            summary_json = generator.repo_root / "run_summary.json"
            if summary_json.is_file():
                doc = json.loads(summary_json.read_text(encoding="utf-8", errors="replace"))
                if isinstance(doc, dict):
                    out["crash_found"] = bool(doc.get("crash_found") or False)
                    out["run_error_kind"] = str(doc.get("run_error_kind") or "")
                    out["run_details"] = list(doc.get("run_details") or [])
                    if not str(out.get("last_fuzzer") or "").strip():
                        out["last_fuzzer"] = str(doc.get("last_fuzzer") or "")
                    if not str(out.get("last_crash_artifact") or "").strip():
                        out["last_crash_artifact"] = str(doc.get("last_crash_artifact") or "")
                    out["crash_evidence"] = str(doc.get("crash_evidence") or "none")
                    out["run_rc"] = int(doc.get("run_rc") or 0)
                    coverage_loop = doc.get("coverage_loop")
                    if isinstance(coverage_loop, dict):
                        out["coverage_loop_max_rounds"] = max(
                            0,
                            int(
                                coverage_loop.get("max_rounds")
                                if coverage_loop.get("max_rounds") is not None
                                else (
                                    out.get("coverage_loop_max_rounds")
                                    if out.get("coverage_loop_max_rounds") is not None
                                    else 0
                                )
                            ),
                        )
                        out["coverage_loop_round"] = int(coverage_loop.get("round") or out.get("coverage_loop_round") or 0)
                        out["coverage_should_improve"] = bool(
                            coverage_loop.get("should_improve") or out.get("coverage_should_improve") or False
                        )
                        out["coverage_improve_reason"] = str(
                            coverage_loop.get("reason") or out.get("coverage_improve_reason") or ""
                        )
                        out["coverage_history"] = list(
                            coverage_loop.get("history") or out.get("coverage_history") or []
                        )
                        out["coverage_target_name"] = str(coverage_loop.get("target_name") or out.get("coverage_target_name") or "")
                        out["coverage_seed_profile"] = str(coverage_loop.get("seed_profile") or out.get("coverage_seed_profile") or "")
                        out["coverage_target_depth_score"] = int(
                            coverage_loop.get("target_depth_score") or out.get("coverage_target_depth_score") or 0
                        )
                        out["coverage_target_depth_class"] = str(
                            coverage_loop.get("target_depth_class") or out.get("coverage_target_depth_class") or ""
                        )
                        out["coverage_selection_bias_reason"] = str(
                            coverage_loop.get("selection_bias_reason") or out.get("coverage_selection_bias_reason") or ""
                        )
                        out["coverage_plateau_streak"] = int(coverage_loop.get("plateau_streak") or out.get("coverage_plateau_streak") or 0)
                        out["coverage_last_max_cov"] = int(coverage_loop.get("last_max_cov") or out.get("coverage_last_max_cov") or 0)
                        out["coverage_last_ft"] = int(coverage_loop.get("last_ft") or out.get("coverage_last_ft") or 0)
                        out["coverage_replan_required"] = bool(coverage_loop.get("replan_required") or out.get("coverage_replan_required") or False)
                        out["coverage_replan_effective"] = bool(
                            coverage_loop.get("replan_effective") if "replan_effective" in coverage_loop else out.get("coverage_replan_effective") or False
                        )
                        out["coverage_replan_reason"] = str(
                            coverage_loop.get("replan_reason") or out.get("coverage_replan_reason") or ""
                        )
                        out["coverage_improve_mode"] = str(coverage_loop.get("improve_mode") or out.get("coverage_improve_mode") or "")
                        out["coverage_round_budget_exhausted"] = bool(
                            coverage_loop.get("round_budget_exhausted") or out.get("coverage_round_budget_exhausted") or False
                        )
                        out["coverage_stop_reason"] = str(
                            coverage_loop.get("stop_reason") or out.get("coverage_stop_reason") or ""
                        )
                        out["coverage_corpus_sources"] = list(coverage_loop.get("corpus_sources") or out.get("coverage_corpus_sources") or [])
                        out["coverage_seed_counts"] = dict(coverage_loop.get("seed_counts") or out.get("coverage_seed_counts") or {})
                        out["coverage_repo_examples_filtered"] = bool(
                            coverage_loop.get("repo_examples_filtered")
                            if "repo_examples_filtered" in coverage_loop
                            else out.get("coverage_repo_examples_filtered") or False
                        )
                        out["coverage_repo_examples_rejected_count"] = int(
                            coverage_loop.get("repo_examples_rejected_count")
                            or out.get("coverage_repo_examples_rejected_count")
                            or 0
                        )
                        out["coverage_repo_examples_accepted_count"] = int(
                            coverage_loop.get("repo_examples_accepted_count")
                            or out.get("coverage_repo_examples_accepted_count")
                            or 0
                        )
                    plan_policy = doc.get("plan_policy")
                    if isinstance(plan_policy, dict):
                        out["plan_fix_on_crash"] = bool(plan_policy.get("fix_on_crash", out["plan_fix_on_crash"]))
                    out["plan_max_fix_rounds"] = 0
                    build_fix_policy = doc.get("build_fix_policy")
                    if isinstance(build_fix_policy, dict):
                        _ = build_fix_policy
                    out["max_fix_rounds"] = 0
                    out["same_error_max_retries"] = 0
                    re_stage = doc.get("re_stage")
                    if isinstance(re_stage, dict):
                        if not str(out.get("re_workspace_root") or "").strip():
                            out["re_workspace_root"] = str(re_stage.get("workspace_root") or "")
                        out["re_build_done"] = bool(re_stage.get("re_build_done") or False)
                        out["re_build_ok"] = bool(re_stage.get("re_build_ok") or False)
                        out["re_build_rc"] = int(re_stage.get("re_build_rc") or 0)
                        out["re_build_report_path"] = str(re_stage.get("re_build_report_path") or "")
                        out["re_build_json_path"] = str(re_stage.get("re_build_json_path") or "")
                        out["re_run_done"] = bool(re_stage.get("re_run_done") or False)
                        out["re_run_ok"] = bool(re_stage.get("re_run_ok") or False)
                        out["re_run_rc"] = int(re_stage.get("re_run_rc") or 0)
                        out["re_run_report_path"] = str(re_stage.get("re_run_report_path") or "")
                        out["re_run_json_path"] = str(re_stage.get("re_run_json_path") or "")
                    restart_ctx = doc.get("restart_to_plan")
                    if isinstance(restart_ctx, dict):
                        out["restart_to_plan"] = bool(restart_ctx.get("active") or False)
                        out["restart_to_plan_reason"] = str(restart_ctx.get("reason") or "")
                        out["restart_to_plan_stage"] = str(restart_ctx.get("stage") or "")
                        out["restart_to_plan_error_text"] = str(restart_ctx.get("error_text") or "")
                        out["restart_to_plan_report_path"] = str(restart_ctx.get("report_path") or "")
                        out["restart_to_plan_count"] = int(restart_ctx.get("count") or 0)
        except Exception:
            pass

    out = cast(FuzzWorkflowRuntimeState, _normalize_error_state(cast(dict[str, Any], out)))
    _wf_log(cast(dict[str, Any], out), f"<- init ok repo_root={out.get('repo_root')} dt={_fmt_dt(time.perf_counter()-t0)}")
    return out


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
                "companion": companion_doc,
                "analysis_evidence": evidence_doc,
            }
            analysis_path = fuzz_dir / "analysis_context.json"
            analysis_path.write_text(json.dumps(analysis_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            analysis_context_path = str(analysis_path)
            analysis_report_path = str(analysis_path)

            if _has_codex_key():
                analysis_lines: list[str] = [
                    "Generate analysis artifacts for downstream planning.",
                    "Write/update `fuzz/analysis_context.json` with concise actionable signals.",
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
                prompt = _render_opencode_prompt("analysis_with_hint", hint=analysis_hint)
                gen.patcher.run_codex_command(
                    prompt,
                    stage_skill="analysis",
                    timeout=_remaining_time_budget_sec(state),
                    max_attempts=1,
                    max_cli_retries=_opencode_cli_retries(),
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
                "analysis_report_path": analysis_report_path,
                "analysis_context_path": analysis_context_path,
                "analysis_evidence_count": analysis_evidence_count,
                "antlr_context_path": antlr_context_path,
                "antlr_context_summary": antlr_context_summary,
                "target_analysis_path": target_analysis_path,
                "target_analysis_summary": target_analysis_summary,
                "message": "analysis completed",
            }
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
        "antlr_context_path": antlr_context_path,
        "antlr_context_summary": antlr_context_summary,
        "target_analysis_path": target_analysis_path,
        "target_analysis_summary": target_analysis_summary,
        "message": "analysis degraded",
    }
    out = _clear_error_markers_on_success(out)
    _wf_log(cast(dict[str, Any], out), f"<- analysis degraded err={fallback_error} dt={_fmt_dt(time.perf_counter()-t0)}")
    return out


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
    if not antlr_context_path and not antlr_context_summary:
        antlr_context_path, antlr_context_summary = _prepare_antlr_assist_context(gen.repo_root)
    if not target_analysis_path and not target_analysis_summary:
        target_analysis_path, target_analysis_summary = _prepare_target_analysis_context(gen.repo_root)
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
        repair_hint = "\n\n".join(part for part in repair_blocks if part.strip())
        hint = (hint + "\n\n" + repair_hint).strip() if hint else repair_hint
    seed_feedback = dict(state.get("coverage_seed_feedback") or {})
    harness_feedback = dict(state.get("coverage_harness_feedback") or {})
    quality_oracle = str(state.get("coverage_quality_oracle") or "").strip()
    if seed_feedback or harness_feedback or quality_oracle:
        feedback_lines: list[str] = ["Coverage feedback signals for planning:"]
        if quality_oracle:
            feedback_lines.append(f"- quality_oracle: {quality_oracle}")
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
        _wf_log(cast(dict[str, Any], out), f"<- plan err=missing-key dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    try:
        plan_template_name = "plan_with_hint"
        plan_stage_skill = "plan"
        if repair_mode:
            if repair_origin_stage == "crash":
                plan_template_name = "plan_repair_crash_with_hint"
                plan_stage_skill = "plan_repair_crash"
            elif repair_origin_stage == "coverage":
                plan_template_name = "plan_repair_coverage_with_hint"
                plan_stage_skill = "plan_repair_coverage"
            else:
                plan_template_name = "plan_repair_build_with_hint"
                plan_stage_skill = "plan_repair_build"
        if hint:
            prompt = _render_opencode_prompt(plan_template_name, hint=hint)
            gen.patcher.run_codex_command(
                prompt,
                stage_skill=plan_stage_skill,
                timeout=_remaining_time_budget_sec(state),
                max_attempts=1,
                max_cli_retries=_opencode_cli_retries(),
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
            prompt = _render_opencode_prompt("plan_fix_targets_schema", schema_error=targets_err)
            gen.patcher.run_codex_command(
                prompt,
                stage_skill="plan_fix_targets_schema",
                timeout=_remaining_time_budget_sec(state),
                max_attempts=1,
                max_cli_retries=_opencode_cli_retries(),
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
                if not ok_targets:
                    _wf_log(cast(dict[str, Any], out), f"<- plan err=targets-schema dt={_fmt_dt(time.perf_counter()-t0)}")
                    return out

        fix_on_crash, _ = _derive_plan_policy(gen.repo_root)
        plan_hint = _make_plan_hint(gen.repo_root)
        if antlr_context_summary:
            plan_hint = (
                (plan_hint.strip() + "\n\n") if plan_hint.strip() else ""
            ) + (
                "Use `fuzz/antlr_plan_context.json` as grammar-aware grounding for API/entrypoint selection.\n"
                f"{antlr_context_summary}"
            )
        primary_target = _select_primary_target(gen.repo_root)
        selected_targets_path = ""
        execution_plan_path = ""
        try:
            selected_targets_path, selected_targets_doc = _write_selected_targets_doc(gen.repo_root)
            execution_plan_path, _ = _write_execution_plan_doc(gen.repo_root, selected_targets_doc)
        except Exception:
            selected_targets_doc = []
        new_target_name = str(primary_target.get("name") or "")
        new_target_api = str(primary_target.get("api") or new_target_name)
        new_seed_profile = str(primary_target.get("seed_profile") or "")
        new_depth_score = int(primary_target.get("depth_score") or 0)
        new_depth_class = str(primary_target.get("depth_class") or "")
        new_selection_bias_reason = str(primary_target.get("selection_bias_reason") or "")
        selected_primary = selected_targets_doc[0] if selected_targets_doc else {}
        seed_families_required = list(selected_primary.get("seed_families_required") or [])
        seed_families_optional = list(selected_primary.get("seed_families_optional") or [])
        selected_runtime_viability = str(selected_primary.get("runtime_viability") or "").strip().lower()
        target_scoring_enabled = bool(
            selected_primary.get("target_scoring_enabled")
            or any(bool(item.get("target_score_breakdown")) for item in selected_targets_doc)
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
            targets_changed = new_targets_text != prev_targets_text
            target_changed = new_target_name != prev_target_name
            depth_improved = (
                new_depth_score > prev_target_depth_score
                or depth_rank.get(new_depth_class, -1) > depth_rank.get(prev_target_depth_class, -1)
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
            "selected_targets_path": selected_targets_path,
            "execution_plan_path": execution_plan_path,
            "coverage_target_name": new_target_name or prev_target_name,
            "coverage_target_api": new_target_api or str(state.get("coverage_target_api") or ""),
            "selected_target_api": new_target_api or str(state.get("selected_target_api") or ""),
            "selected_target_runtime_viability": selected_runtime_viability or str(state.get("selected_target_runtime_viability") or ""),
            "coverage_seed_profile": new_seed_profile or str(state.get("coverage_seed_profile") or ""),
            "coverage_seed_families_required": seed_families_required or list(state.get("coverage_seed_families_required") or []),
            "coverage_seed_families_covered": list(state.get("coverage_seed_families_covered") or []),
            "coverage_seed_families_missing": list(state.get("coverage_seed_families_missing") or seed_families_required),
            "coverage_seed_quality": dict(state.get("coverage_seed_quality") or {}),
            "coverage_quality_flags": list(state.get("coverage_quality_flags") or []),
            "coverage_target_depth_score": new_depth_score,
            "coverage_target_depth_class": new_depth_class,
            "coverage_selection_bias_reason": new_selection_bias_reason,
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
            "target_scoring_enabled": target_scoring_enabled,
            "message": "planned",
        }
        out = _clear_error_markers_on_success(out)
        _wf_log(cast(dict[str, Any], out), f"<- plan ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    except Exception as e:
        _write_stage_feedback(
            gen.repo_root,
            stage="plan",
            error_text=str(e),
            state=cast(dict[str, Any], state),
        )
        out = {**state, "last_step": "plan", "last_error": str(e), "message": "plan failed"}
        _wf_log(cast(dict[str, Any], out), f"<- plan err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


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
    repair_mode = bool(state.get("repair_mode") or False)
    repair_origin_stage = str(state.get("repair_origin_stage") or "").strip().lower()
    if repair_origin_stage not in {"build", "crash", "coverage"}:
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
        raw_timeout = (os.environ.get("SHERPA_SYNTH_BUILD_VALIDATE_TIMEOUT_SEC") or "90").strip()
        try:
            cfg_timeout = max(10, min(int(raw_timeout), 600))
        except Exception:
            cfg_timeout = 90
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
            Write `fuzz/build.py` into `./done`.
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
        )
        _wf_log(cast(dict[str, Any], state), "synthesize: applied post-validation build.py repair for path/link issue")

    def _run_synthesize_completion(timeout: int) -> None:
        missing_items = "\n".join(f"- {item}" for item in _missing_synthesis_items()) or "- no missing items detected"
        prompt = _render_opencode_prompt("synthesize_complete_scaffold", missing_items=missing_items)
        gen.patcher.run_codex_command(
            prompt,
            additional_context=_completion_context() or None,
            stage_skill="synthesize_complete_scaffold",
            timeout=timeout,
            max_attempts=_synthesize_opencode_attempts(),
            max_cli_retries=_opencode_cli_retries(),
            idle_timeout_override=_synthesize_opencode_idle_timeout_sec(),
            activity_watch_paths=_synthesize_activity_watch_paths(),
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
            - Write `fuzz/out/` into `./done` before exit.
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
            - Write `fuzz/README.md` into `./done` before finishing.
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
        )

    if not _has_codex_key():
        out = {
            **state,
            "last_step": "synthesize",
            "last_error": "Missing OPENAI_API_KEY for synthesis",
            "message": "synthesize failed",
        }
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
            elif repair_origin_stage == "coverage":
                synth_template_name = "synthesize_repair_coverage_with_hint"
                synth_stage_skill = "synthesize_repair_coverage"
            else:
                synth_template_name = "synthesize_repair_build_with_hint"
                synth_stage_skill = "synthesize_repair_build"
        if hint:
            prompt = _render_opencode_prompt(synth_template_name, hint=hint)
            # Provide context from plan/targets if present.
            plan = (gen.repo_root / "fuzz" / "PLAN.md")
            targets = (gen.repo_root / "fuzz" / "targets.json")
            ctx = ""
            try:
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
        execution_plan_doc = _load_execution_plan_doc(gen.repo_root)
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
            "coverage_target_name": str(target_alignment.get("observed_api") or state.get("coverage_target_name") or ""),
            "analysis_context_path": analysis_context_path or str(state.get("analysis_context_path") or ""),
            "analysis_evidence_count": analysis_evidence_count,
            "target_scoring_enabled": bool(state.get("target_scoring_enabled") or False),
            "constraint_memory_count": int(state.get("constraint_memory_count") or 0),
            "message": "synthesized",
        }
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
        out = {**state, "last_step": "synthesize", "last_error": str(e), "message": "synthesize failed"}
        _wf_log(cast(dict[str, Any], out), f"<- synthesize err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


def _node_build(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "build")
    if stop_now:
        return state
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), f"-> build attempt={(int(state.get('build_attempts') or 0)+1)}")
    try:
        fuzz_dir = gen.repo_root / "fuzz"
        build_py = fuzz_dir / "build.py"
        build_sh = fuzz_dir / "build.sh"
        build_full_log_path = fuzz_dir / "build_full.log"

        def _tail(s: str, n: int = 120) -> str:
            lines = (s or "").replace("\r", "\n").splitlines()
            return "\n".join(lines[-n:]).strip()

        def _init_build_full_log() -> None:
            try:
                build_full_log_path.parent.mkdir(parents=True, exist_ok=True)
                header = (
                    "Sherpa build full log\n"
                    f"repo_root={gen.repo_root}\n"
                    f"generated_at={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n"
                    + "=" * 88
                    + "\n"
                )
                build_full_log_path.write_text(header, encoding="utf-8", errors="replace")
            except Exception:
                pass

        def _append_build_full_log(*, stage: str, cmd: list[str], cwd: Path, rc: int, out: str, err: str) -> None:
            try:
                lines = [
                    "",
                    "=" * 88,
                    f"stage={stage}",
                    f"cmd={' '.join(cmd)}",
                    f"cwd={cwd}",
                    f"rc={rc}",
                    "-" * 88,
                    "[stdout]",
                    out or "",
                    "-" * 88,
                    "[stderr]",
                    err or "",
                    "=" * 88,
                    "",
                ]
                with build_full_log_path.open("a", encoding="utf-8", errors="replace") as f:
                    f.write("\n".join(lines))
            except Exception:
                pass

        _init_build_full_log()

        def _build_py_supports_clean_flag(path: Path) -> bool:
            try:
                txt = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return False
            return "--clean" in txt

        def _env_bool(name: str, default: bool) -> bool:
            raw = (os.environ.get(name) or "").strip().lower()
            if not raw:
                return default
            return raw in {"1", "true", "yes", "on"}

        def _read_declared_system_packages(dep_file: Path) -> set[str]:
            alias_map = {
                "z": "zlib",
                "bz2": "bzip2",
                "lzma": "liblzma",
                "xz": "liblzma",
                "ssl": "openssl",
                "crypto": "openssl",
                "libssl": "openssl",
                "libcrypto": "openssl",
                "xml2": "libxml2",
                "libxml": "libxml2",
            }
            if not dep_file.is_file():
                return set()
            declared: set[str] = set()
            try:
                for raw_line in dep_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = raw_line.split("#", 1)[0].strip().lower()
                    if not line:
                        continue
                    if re.fullmatch(r"[a-z0-9][a-z0-9+._-]*", line):
                        declared.add(alias_map.get(line, line))
            except Exception:
                return set()
            return declared

        def _detect_missing_optional_ports(stdout_text: str, stderr_text: str) -> list[str]:
            combined = ((stdout_text or "") + "\n" + (stderr_text or "")).lower()
            signal_to_port: list[tuple[list[str], str]] = [
                (["could not find zlib", "zlib_library", "zlib_include_dir"], "zlib"),
                (["could not find bzip2", "bzip2_libraries", "bzip2_include_dir"], "bzip2"),
                (["could not find liblzma", "liblzma_library", "liblzma_include_dir"], "liblzma"),
                (["could not find lz4", "lz4_library", "lz4_include_dir"], "lz4"),
                (["could not find zstd", "zstd_library", "zstd_include_dir"], "zstd"),
                (["could not find openssl", "openssl_crypto_library", "openssl_include_dir"], "openssl"),
                (["could not find libxml2", "libxml2_library", "libxml2_include_dir"], "libxml2"),
                (["could not find expat", "expat_library", "expat_include_dir"], "expat"),
            ]
            missing: list[str] = []
            for needles, port in signal_to_port:
                if any(n in combined for n in needles) and port not in missing:
                    missing.append(port)
            return missing

        def _list_static_libs_for_diagnostics() -> str:
            build_dir = gen.repo_root / "build"
            if not build_dir.exists():
                return f"(no build dir at {build_dir})"
            libs: list[str] = []
            try:
                for p in build_dir.rglob("*"):
                    if not p.is_file():
                        continue
                    if p.suffix.lower() in {".a", ".lib", ".so", ".dylib"}:
                        try:
                            libs.append(f"{p.relative_to(gen.repo_root)} ({p.stat().st_size} bytes)")
                        except Exception:
                            libs.append(str(p.relative_to(gen.repo_root)))
                    if len(libs) >= 80:
                        break
            except Exception as e:
                return f"(failed to list libs under build/: {e})"
            return "\n".join(libs) if libs else "(no static libs found under build/)"

        build_cmd_clean: list[str] | None = None
        build_cwd = fuzz_dir
        fallback_cmd: list[str] | None = None
        fallback_cwd: Path | None = None
        if build_py.is_file():
            build_cmd = [gen._python_runner(), "build.py"]
            fallback_cmd = [gen._python_runner(), "fuzz/build.py"]
            fallback_cwd = gen.repo_root
            if _build_py_supports_clean_flag(build_py):
                build_cmd_clean = list(build_cmd) + ["--clean"]
        elif build_sh.is_file():
            shell = "bash"
            if not getattr(gen, "docker_image", None):
                if shutil.which("bash") is None:
                    if shutil.which("sh") is not None:
                        shell = "sh"
                    else:
                        raise HarnessGeneratorError("build.sh exists but neither bash nor sh is available in PATH")
            try:
                mode = build_sh.stat().st_mode
                build_sh.chmod(mode | 0o111)
            except Exception:
                pass
            build_cmd = [shell, "build.sh"]
            fallback_cmd = [shell, "fuzz/build.sh"]
            fallback_cwd = gen.repo_root
        else:
            raise HarnessGeneratorError("Missing fuzz/build.py (agent must create fuzz/build.py)")

        build_env = os.environ.copy()
        if getattr(gen, "docker_image", None):
            include_root = "/work"
            build_env.setdefault("CC", "clang")
            build_env.setdefault("CXX", "clang++")
            build_env.setdefault("CFLAGS", "-D_GNU_SOURCE")
            build_env.setdefault("CXXFLAGS", "-D_GNU_SOURCE")
            for stale_dir in (gen.repo_root / "fuzz" / "build", gen.repo_root / "build"):
                if stale_dir.exists():
                    try:
                        shutil.rmtree(stale_dir)
                    except Exception:
                        pass
        else:
            include_root = str(gen.repo_root)
        for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
            prev = build_env.get(key, "").strip()
            build_env[key] = f"{include_root}:{prev}" if prev else include_root

        retries_raw = os.environ.get("SHERPA_WORKFLOW_BUILD_LOCAL_RETRIES", "2")
        try:
            max_local_attempts = int(retries_raw)
        except Exception:
            max_local_attempts = 2
        max_local_attempts = max(1, min(max_local_attempts, 5))
        retry_with_clean = _env_bool("SHERPA_WORKFLOW_BUILD_RETRY_WITH_CLEAN", True)
        retry_delay_s = 1.0

        attempts_used = 0
        final_rc = 1
        final_out = ""
        final_err = ""
        final_bins: list[Path] = []
        def _is_repo_root_cwd_issue(out: str, err: str) -> bool:
            combined = ((out or "") + "\n" + (err or "")).lower()
            return (
                ("no such file or directory" in combined and "fuzz/" in combined)
                or "can't open file '/work/fuzz/fuzz/" in combined
                or "can't open file 'fuzz/" in combined
            )

        for attempt in range(1, max_local_attempts + 1):
            build_cmd_timeout = _remaining_time_budget_sec(state, min_timeout=0)
            if build_cmd_timeout <= 0:
                return _time_budget_exceeded_state(state, step_name="build")
            _wf_log(cast(dict[str, Any], state), f"build cmd attempt {attempt}/{max_local_attempts} -> {' '.join(build_cmd)}")
            rc, out, err = gen._run_cmd(list(build_cmd), cwd=build_cwd, env=build_env, timeout=build_cmd_timeout)
            _append_build_full_log(stage=f"attempt-{attempt}/primary", cmd=list(build_cmd), cwd=build_cwd, rc=rc, out=out, err=err)
            attempts_used += 1

            # Backward-compatibility shim: older generated scripts may hardcode "fuzz/..."
            # and therefore need repo-root cwd.
            if rc != 0 and fallback_cmd is not None and fallback_cwd is not None and _is_repo_root_cwd_issue(out, err):
                fallback_timeout = _remaining_time_budget_sec(state, min_timeout=0)
                if fallback_timeout <= 0:
                    return _time_budget_exceeded_state(state, step_name="build")
                _wf_log(
                    cast(dict[str, Any], state),
                    f"build retry from repo-root cwd -> {' '.join(fallback_cmd)}",
                )
                rc, out, err = gen._run_cmd(list(fallback_cmd), cwd=fallback_cwd, env=build_env, timeout=fallback_timeout)
                _append_build_full_log(stage=f"attempt-{attempt}/repo-root-fallback", cmd=list(fallback_cmd), cwd=fallback_cwd, rc=rc, out=out, err=err)
                attempts_used += 1

            if rc != 0 and retry_with_clean and build_cmd_clean is not None:
                combined = (out or "") + "\n" + (err or "")
                if not re.search(r"unrecognized arguments: --clean", combined, re.IGNORECASE):
                    clean_timeout = _remaining_time_budget_sec(state, min_timeout=0)
                    if clean_timeout <= 0:
                        return _time_budget_exceeded_state(state, step_name="build")
                    _wf_log(cast(dict[str, Any], state), "build failed; retrying once with --clean")
                    rc2, out2, err2 = gen._run_cmd(list(build_cmd_clean), cwd=build_cwd, env=build_env, timeout=clean_timeout)
                    _append_build_full_log(stage=f"attempt-{attempt}/clean-retry", cmd=list(build_cmd_clean), cwd=build_cwd, rc=rc2, out=out2, err=err2)
                    attempts_used += 1
                    combined2 = (out2 or "") + "\n" + (err2 or "")
                    if re.search(r"unrecognized arguments: --clean", combined2, re.IGNORECASE):
                        _wf_log(cast(dict[str, Any], state), "build.py rejected --clean; keeping original diagnostics")
                    else:
                        rc, out, err = rc2, out2, err2

            bins = gen._discover_fuzz_binaries() if rc == 0 else []
            final_rc, final_out, final_err, final_bins = rc, out, err, bins
            if rc == 0 and bins:
                break

            if attempt < max_local_attempts:
                reason = f"rc={rc}" if rc != 0 else "no fuzzer binaries generated"
                _wf_log(cast(dict[str, Any], state), f"build attempt {attempt} not ready ({reason}); retrying")
                time.sleep(retry_delay_s)

        if final_rc == 0 and not final_bins:
            libs_diag = _list_static_libs_for_diagnostics()
            if libs_diag:
                final_out = (final_out or "") + "\n\n=== build dir artifacts (static libs) ===\n" + libs_diag + "\n"

        attempts_total = int(state.get("build_attempts") or 0) + attempts_used
        next_state: FuzzWorkflowRuntimeState = {
            **state,
            "build_attempts": attempts_total,
            "build_rc": int(final_rc),
            "build_stdout_tail": _tail(final_out),
            "build_stderr_tail": _tail(final_err),
            "build_full_log_path": str(build_full_log_path),
            "harness_index_path": str(_harness_index_path(gen.repo_root)),
            "last_step": "build",
            "build_mode": str(state.get("build_mode") or ""),
            "build_target_source": str(state.get("build_target_source") or "external_scaffold"),
        }
        def _mark_build_repair_state(*, kind: str, code: str, sig: str = "") -> None:
            signature_short = sig[:12] if sig else str(next_state.get("build_error_signature_short") or "")
            attempt_index = int(state.get("repair_attempt_index") or 0) + 1
            same_signature_streak = int(next_state.get("same_build_error_repeats") or 0) + 1
            force_change = same_signature_streak >= _repair_strategy_repeat_threshold()
            next_state["repair_mode"] = True
            next_state["repair_origin_stage"] = "build"
            next_state["repair_error_kind"] = kind or "build_failure_generic"
            next_state["repair_error_code"] = code or ""
            next_state["repair_signature"] = signature_short
            next_state["repair_stdout_tail"] = str(next_state.get("build_stdout_tail") or "")
            next_state["repair_stderr_tail"] = str(next_state.get("build_stderr_tail") or "")
            next_state["repair_attempt_index"] = attempt_index
            next_state["repair_strategy_force_change"] = force_change
            if force_change:
                force_msg = (
                    " strategy_change_required: same build signature repeated; "
                    "next repair round must materially change target selection or harness/build strategy."
                )
                current_error = str(next_state.get("last_error") or "")
                if force_msg.strip() not in current_error:
                    next_state["last_error"] = (current_error + force_msg).strip()
            next_state["repair_error_digest"] = _build_repair_error_digest(
                repo_root=gen.repo_root,
                error_kind=kind or "build_failure_generic",
                error_code=code or "",
                signature=signature_short,
                error_text=str(next_state.get("last_error") or ""),
                stdout_tail=str(next_state.get("build_stdout_tail") or ""),
                stderr_tail=str(next_state.get("build_stderr_tail") or ""),
                prev_digest=dict(state.get("repair_error_digest") or {}),
            )
            recent = list(state.get("repair_recent_attempts") or [])
            recent.append(
                {
                    "step": "build",
                    "origin": "build",
                    "error_kind": kind or "build_failure_generic",
                    "error_code": code or "",
                    "signature": signature_short,
                    "attempt_index": attempt_index,
                    "force_strategy_change": force_change,
                    "message": str(next_state.get("last_error") or "")[:512],
                }
            )
            next_state["repair_recent_attempts"] = recent[-5:]
        build_error_kind, build_error_code = _classify_build_failure(
            str(next_state.get("last_error") or ""),
            str(next_state.get("build_stdout_tail") or ""),
            str(next_state.get("build_stderr_tail") or ""),
            build_rc=int(final_rc),
            has_fuzzer_binaries=bool(final_bins),
        )

        def _calc_build_error_signature() -> str:
            marker = "rc-fail" if final_rc != 0 else "no-fuzzers"
            blob = (
                marker
                + "\n"
                + _tail(final_out, n=220)
                + "\n"
                + _tail(final_err, n=220)
            )
            return _sha256_text(blob)

        prev_sig = str(state.get("build_error_signature") or "").strip()
        prev_repeats = int(state.get("same_build_error_repeats") or 0)
        max_same_repeats = _effective_same_error_retry_limit(state)

        if final_rc != 0:
            sig = _calc_build_error_signature()
            next_state["build_error_signature_short"] = sig[:12]
            repeats = (prev_repeats + 1) if (prev_sig and prev_sig == sig) else 0
            next_state["build_error_signature"] = sig
            next_state["build_error_signature_before"] = prev_sig
            next_state["build_error_signature_after"] = sig
            next_state["same_build_error_repeats"] = repeats
            next_state["build_error_kind"] = build_error_kind
            next_state["build_error_code"] = build_error_code
            advice = _build_failure_recovery_advice(build_error_kind, build_error_code)
            if max_same_repeats > 0 and repeats >= max_same_repeats:
                repeated_err = (
                    "build failed with the same error signature repeatedly "
                    f"(repeats={repeats + 1}, threshold={max_same_repeats + 1})"
                )
                next_state["failed"] = False
                next_state["last_error"] = repeated_err
                next_state["message"] = "build failed repeatedly (same error)"
                next_state["restart_to_plan"] = build_error_kind == "infra"
                next_state["restart_to_plan_reason"] = "build_same_error_repeated" if build_error_kind == "infra" else ""
                next_state["restart_to_plan_stage"] = "build" if build_error_kind == "infra" else ""
                next_state["restart_to_plan_error_text"] = repeated_err if build_error_kind == "infra" else ""
                _wf_log(
                    cast(dict[str, Any], next_state),
                    "<- build stop same-error "
                    f"repeats={repeats+1} "
                    f"signature_before={prev_sig[:12] if prev_sig else '-'} "
                    f"signature_after={sig[:12]} "
                    f"same_error_max_retries={max_same_repeats}",
                )
                return next_state
            next_state["last_error"] = f"build failed rc={final_rc} after {attempts_used} command run(s)"
            if advice:
                next_state["last_error"] += f"\nrecovery: {advice}"
            next_state["message"] = "build failed"
            next_state["restart_to_plan"] = build_error_kind == "infra"
            next_state["restart_to_plan_reason"] = "build_failed" if build_error_kind == "infra" else ""
            next_state["restart_to_plan_stage"] = "build" if build_error_kind == "infra" else ""
            next_state["restart_to_plan_error_text"] = str(next_state["last_error"]) if build_error_kind == "infra" else ""
            _mark_build_repair_state(kind=build_error_kind or "build_failure_generic", code=build_error_code, sig=sig)
            _wf_log(
                cast(dict[str, Any], next_state),
                "<- build fail "
                f"rc={final_rc} "
                f"signature_before={prev_sig[:12] if prev_sig else '-'} "
                f"signature_after={sig[:12]} "
                f"same_error_count={repeats} "
                f"same_error_max_retries={max_same_repeats} "
                f"dt={_fmt_dt(time.perf_counter()-t0)}",
            )
            return next_state

        if not final_bins:
            sig = _calc_build_error_signature()
            next_state["build_error_signature_short"] = sig[:12]
            repeats = (prev_repeats + 1) if (prev_sig and prev_sig == sig) else 0
            next_state["build_error_signature"] = sig
            next_state["build_error_signature_before"] = prev_sig
            next_state["build_error_signature_after"] = sig
            next_state["same_build_error_repeats"] = repeats
            next_state["build_error_kind"] = build_error_kind
            next_state["build_error_code"] = build_error_code
            if max_same_repeats > 0 and repeats >= max_same_repeats:
                repeated_err = (
                    "build produced no fuzzers with the same diagnostics repeatedly "
                    f"(repeats={repeats + 1}, threshold={max_same_repeats + 1})"
                )
                next_state["failed"] = False
                next_state["last_error"] = repeated_err
                next_state["message"] = "build failed repeatedly (no fuzzers)"
                next_state["restart_to_plan"] = build_error_kind == "infra"
                next_state["restart_to_plan_reason"] = "build_no_fuzzer_repeated" if build_error_kind == "infra" else ""
                next_state["restart_to_plan_stage"] = "build" if build_error_kind == "infra" else ""
                next_state["restart_to_plan_error_text"] = repeated_err if build_error_kind == "infra" else ""
                _wf_log(
                    cast(dict[str, Any], next_state),
                    "<- build stop same-no-fuzzer "
                    f"repeats={repeats+1} "
                    f"signature_before={prev_sig[:12] if prev_sig else '-'} "
                    f"signature_after={sig[:12]} "
                    f"same_error_max_retries={max_same_repeats}",
                )
                return next_state
            next_state["last_error"] = f"No fuzzer binaries found under fuzz/out/ after {attempts_used} command run(s)"
            next_state["message"] = "build produced no fuzzers"
            next_state["restart_to_plan"] = build_error_kind == "infra"
            next_state["restart_to_plan_reason"] = "build_no_fuzzers" if build_error_kind == "infra" else ""
            next_state["restart_to_plan_stage"] = "build" if build_error_kind == "infra" else ""
            next_state["restart_to_plan_error_text"] = str(next_state["last_error"]) if build_error_kind == "infra" else ""
            _mark_build_repair_state(kind=build_error_kind or "build_failure_generic", code=build_error_code, sig=sig)
            _wf_log(
                cast(dict[str, Any], next_state),
                "<- build fail no-fuzzers "
                f"signature_before={prev_sig[:12] if prev_sig else '-'} "
                f"signature_after={sig[:12]} "
                f"same_error_count={repeats} "
                f"same_error_max_retries={max_same_repeats} "
                f"dt={_fmt_dt(time.perf_counter()-t0)}",
            )
            return next_state

        execution_plan_doc = _load_execution_plan_doc(gen.repo_root)
        execution_targets = [
            item for item in list(execution_plan_doc.get("execution_targets") or [])
            if isinstance(item, dict)
        ]
        harness_index_doc = _load_harness_index_doc(gen.repo_root)
        if not harness_index_doc:
            try:
                _, harness_index_doc = _write_harness_index_doc(
                    gen.repo_root,
                    execution_plan_doc=execution_plan_doc,
                )
            except Exception:
                harness_index_doc = {}
        mapping_by_target: dict[str, dict[str, Any]] = {}
        for row in list(harness_index_doc.get("mappings") or []):
            if not isinstance(row, dict):
                continue
            key = str(row.get("target_name") or "").strip()
            if key and key not in mapping_by_target:
                mapping_by_target[key] = row
        built_names = {p.name for p in final_bins}
        built_stems = {Path(name).stem for name in built_names}
        built_norm_tokens = {
            token
            for token in (
                _normalize_exec_target_token(name)
                for name in (list(built_names) + list(built_stems))
            )
            if token
        }
        target_build_matrix: list[dict[str, Any]] = []
        built_execution_targets = 0
        for item in execution_targets:
            target_name = str(item.get("target_name") or "").strip()
            expected = str(item.get("expected_fuzzer_name") or item.get("target_name") or "").strip()
            mapping = mapping_by_target.get(target_name)
            source_path = str(mapping.get("source_path") or "").strip() if isinstance(mapping, dict) else ""
            source_stem = Path(source_path).stem if source_path else ""
            raw_candidates = {
                expected,
                f"{expected}_fuzz" if expected else "",
                f"{expected}_fuzzer" if expected else "",
                target_name,
                f"{target_name}_fuzz" if target_name else "",
                f"{target_name}_fuzzer" if target_name else "",
                source_stem,
            }
            raw_candidates = {c for c in raw_candidates if c}
            norm_candidates = {
                _normalize_exec_target_token(c)
                for c in (list(raw_candidates) + [Path(c).stem for c in raw_candidates])
            }
            norm_candidates = {c for c in norm_candidates if c}
            matched = bool(
                (raw_candidates & built_names)
                or ({Path(c).stem for c in raw_candidates} & built_stems)
                or (norm_candidates & built_norm_tokens)
            )
            has_source = bool(source_path)
            if matched and has_source:
                built_execution_targets += 1
            target_build_matrix.append(
                {
                    "target_name": target_name,
                    "expected_fuzzer_name": expected,
                    "must_run": bool(item.get("must_run") or False),
                    "source_path": source_path,
                    "has_source": has_source,
                    "built": bool(matched),
                }
            )
        min_required_built = int(execution_plan_doc.get("min_required_built_targets") or _execution_targets_min_required())
        if execution_targets and len(execution_targets) > 1 and built_execution_targets < min_required_built:
            missing_targets = [
                str(item.get("target_name") or item.get("expected_fuzzer_name") or "")
                for item in target_build_matrix
                if not (bool(item.get("built")) and bool(item.get("has_source")))
            ]
            next_state["last_error"] = (
                "partial_build_undercoverage: built "
                f"{built_execution_targets}/{len(execution_targets)} execution targets "
                f"(required>={min_required_built}); missing={','.join([x for x in missing_targets if x]) or 'unknown'}"
            )
            next_state["message"] = "build undercoverage: execution target gate not met"
            next_state["build_error_kind"] = "source"
            next_state["build_error_code"] = "partial_build_undercoverage"
            next_state["restart_to_plan"] = False
            next_state["restart_to_plan_reason"] = ""
            next_state["restart_to_plan_stage"] = ""
            next_state["restart_to_plan_error_text"] = ""
            next_state["restart_to_plan_report_path"] = ""
            _mark_build_repair_state(kind="source", code="partial_build_undercoverage", sig=str(next_state.get("build_error_signature_short") or ""))
            if isinstance(next_state.get("repair_error_digest"), dict):
                next_state["repair_error_digest"]["oracle"] = "undercoverage_gate"
            next_state["build_gate_reason"] = "partial_build_undercoverage"
            next_state["built_targets"] = sorted(built_names)
            next_state["missing_targets"] = missing_targets
            next_state["target_build_matrix"] = target_build_matrix
            _wf_log(
                cast(dict[str, Any], next_state),
                "<- build gate partial_build_undercoverage "
                f"built={built_execution_targets}/{len(execution_targets)} required={min_required_built} "
                f"dt={_fmt_dt(time.perf_counter()-t0)}",
            )
            return next_state

        next_state["build_error_signature"] = ""
        next_state["build_error_signature_before"] = prev_sig
        next_state["build_error_signature_after"] = ""
        next_state["build_error_signature_short"] = ""
        next_state["same_build_error_repeats"] = 0
        next_state["build_error_kind"] = ""
        next_state["build_error_code"] = ""
        next_state["fix_build_attempts"] = 0
        next_state["fix_build_noop_streak"] = 0
        next_state["fix_build_terminal_reason"] = ""
        next_state["fix_build_last_diff_paths"] = []
        next_state["fix_action_type"] = ""
        next_state["fix_effect"] = ""
        next_state["last_error"] = ""
        next_state["repair_mode"] = False
        next_state["repair_origin_stage"] = ""
        next_state["repair_error_kind"] = ""
        next_state["repair_error_code"] = ""
        next_state["repair_signature"] = ""
        next_state["repair_stdout_tail"] = ""
        next_state["repair_stderr_tail"] = ""
        next_state["repair_recent_attempts"] = []
        next_state["repair_error_digest"] = {}
        next_state["repair_attempt_index"] = 0
        next_state["repair_strategy_force_change"] = False

        enforce_declared_optional_deps = _env_bool("SHERPA_BUILD_ENFORCE_DECLARED_OPTIONAL_DEPS", True)
        if enforce_declared_optional_deps:
            dep_file = fuzz_dir / "system_packages.txt"
            declared_ports = _read_declared_system_packages(dep_file)
            missing_ports = [
                p for p in _detect_missing_optional_ports(final_out, final_err)
                if p not in declared_ports
            ]
            if missing_ports:
                next_state["last_error"] = (
                    "build succeeded with missing optional libraries but "
                    "fuzz/system_packages.txt does not declare required vcpkg ports: "
                    + ", ".join(missing_ports)
                )
                next_state["message"] = "build missing declared optional deps"
                next_state["build_error_kind"] = "source"
                next_state["build_error_code"] = "missing_system_packages_declared"
                next_state["restart_to_plan"] = False
                next_state["restart_to_plan_reason"] = ""
                next_state["restart_to_plan_stage"] = ""
                next_state["restart_to_plan_error_text"] = ""
                next_state["restart_to_plan_report_path"] = ""
                _mark_build_repair_state(kind="source", code="missing_system_packages_declared", sig=str(next_state.get("build_error_signature_short") or ""))
                if isinstance(next_state.get("repair_error_digest"), dict):
                    next_state["repair_error_digest"]["oracle"] = "undercoverage_gate"
                _wf_log(
                    cast(dict[str, Any], next_state),
                    "<- build gate missing-optional-deps "
                    f"ports={','.join(missing_ports)} dt={_fmt_dt(time.perf_counter()-t0)}",
                )
                return next_state

        cache_path = _cache_successful_build_template(
            gen.repo_root,
            binaries=final_bins,
            target_build_matrix=target_build_matrix,
        )
        if cache_path:
            next_state["build_template_cache_path"] = cache_path
        next_state["built_targets"] = sorted(built_names)
        next_state["missing_targets"] = [
            str(item.get("target_name") or item.get("expected_fuzzer_name") or "")
            for item in target_build_matrix
            if not bool(item.get("built"))
        ]
        next_state["target_build_matrix"] = target_build_matrix
        next_state["build_gate_reason"] = "ok"
        next_state["message"] = f"built ({len(final_bins)} fuzzers)"
        next_state = _clear_error_markers_on_success(next_state)
        _wf_log(cast(dict[str, Any], next_state), f"<- build ok fuzzers={len(final_bins)} dt={_fmt_dt(time.perf_counter()-t0)}")
        return next_state
    except Exception as e:
        out = {
            **state,
            "last_step": "build",
            "last_error": str(e),
            "message": "build failed",
            "build_error_kind": "unknown",
            "build_error_code": "build_node_exception",
            "restart_to_plan": True,
            "restart_to_plan_reason": "build_node_exception",
            "restart_to_plan_stage": "build",
            "restart_to_plan_error_text": str(e),
            "repair_mode": True,
            "repair_origin_stage": "build",
            "repair_error_kind": "build_failure_generic",
            "repair_error_code": "build_node_exception",
            "repair_signature": "",
            "repair_stdout_tail": str(state.get("build_stdout_tail") or ""),
            "repair_stderr_tail": str(state.get("build_stderr_tail") or ""),
            "repair_recent_attempts": list(state.get("repair_recent_attempts") or []),
        }
        if "build_full_log_path" in locals():
            out["build_full_log_path"] = str(build_full_log_path)
        _wf_log(cast(dict[str, Any], out), f"<- build err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


def _node_fix_build(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "fix_build")
    if stop_now:
        return state

    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> fix_build")
    fix_attempts = int(state.get("fix_build_attempts") or 0) + 1
    state = cast(FuzzWorkflowRuntimeState, {**state, "fix_build_attempts": fix_attempts})

    last_error = (state.get("last_error") or "").strip()
    stdout_tail = (state.get("build_stdout_tail") or "").strip()
    stderr_tail = (state.get("build_stderr_tail") or "").strip()
    build_error_kind = (state.get("build_error_kind") or "").strip().lower()
    build_error_code = (state.get("build_error_code") or "").strip().lower()
    repo_root = str(gen.repo_root)
    diag_text = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
    prev_noop_streak = int(state.get("fix_build_noop_streak") or 0)
    history = list(state.get("fix_build_attempt_history") or [])
    rule_hits = list(state.get("fix_build_rule_hits") or [])
    max_noop_streak = _fix_build_max_noop_streak()
    max_fix_attempts = _effective_max_fix_rounds(state)
    history_limit = _fix_build_feedback_history_limit()
    context_history_limit = _fix_build_context_history_limit()
    context_max_chars = _fix_build_context_max_chars()
    error_sig = (state.get("build_error_signature_short") or "").strip()
    if not error_sig:
        error_sig = _sha256_text("\n".join([last_error, stdout_tail, stderr_tail]))[:12]

    if max_fix_attempts > 0 and fix_attempts > max_fix_attempts:
        out = {
            **state,
            "last_step": "fix_build",
            "failed": False,
            "fix_build_terminal_reason": "fix_build_max_attempts_exceeded",
            "last_error": f"fix_build max attempts exceeded ({max_fix_attempts}); restart from plan",
            "message": "fix_build max attempts exceeded; restarting from plan",
            "restart_to_plan": True,
            "restart_to_plan_reason": "fix_build_max_attempts_exceeded",
            "restart_to_plan_stage": "fix_build",
            "restart_to_plan_error_text": str(last_error or "").strip(),
            "fix_action_type": "none",
            "fix_effect": "stalled",
        }
        _wf_log(cast(dict[str, Any], out), f"<- fix_build stop=max-attempts limit={max_fix_attempts}")
        return out

    def _append_attempt(outcome: str, *, rejection_reason: str = "", rule_hit: str = "", changed_paths_count: int = 0) -> tuple[list[dict[str, Any]], list[str]]:
        updated_rule_hits = list(rule_hits)
        if rule_hit and rule_hit not in updated_rule_hits:
            updated_rule_hits.append(rule_hit)
        row = {
            "attempt_index": fix_attempts,
            "build_error_kind": build_error_kind or "unknown",
            "build_error_code": build_error_code or "unknown",
            "classified_signature": error_sig,
            "changed_paths_count": int(changed_paths_count),
            "outcome": outcome,
            "rejection_reason": rejection_reason,
            "rule_hit": rule_hit,
        }
        updated_history = history + [row]
        if len(updated_history) > history_limit:
            updated_history = updated_history[-history_limit:]
        return updated_history, updated_rule_hits

    def _fix_build_quick_check_timeout_sec() -> int:
        raw = (os.environ.get("SHERPA_FIX_BUILD_QUICK_CHECK_TIMEOUT_SEC") or "0").strip()
        try:
            return max(0, min(int(raw), 300))
        except Exception:
            return 0

    def _run_fix_build_quick_probe() -> tuple[bool, dict[str, Any]]:
        def _tail_local(s: str, n: int = 120) -> str:
            lines = (s or "").replace("\r", "\n").splitlines()
            return "\n".join(lines[-n:]).strip()

        if not hasattr(gen, "_run_cmd"):
            return False, {"reason": "unsupported_generator"}

        fuzz_dir = gen.repo_root / "fuzz"
        build_py = fuzz_dir / "build.py"
        build_sh = fuzz_dir / "build.sh"
        if not build_py.is_file() and not build_sh.is_file():
            return False, {"reason": "missing_build_script"}

        quick_timeout = _fix_build_quick_check_timeout_sec()
        if quick_timeout <= 0:
            return False, {"reason": "disabled"}
        remaining = _remaining_time_budget_sec(state, min_timeout=0)
        if remaining <= 0:
            return False, {"reason": "no_budget"}
        timeout = min(remaining, quick_timeout)

        build_cwd = fuzz_dir
        if build_py.is_file():
            if hasattr(gen, "_python_runner"):
                cmd = [gen._python_runner(), "build.py"]
            else:
                py = shutil.which("python3") or shutil.which("python") or "python"
                cmd = [py, "build.py"]
        else:
            shell = "bash"
            if not getattr(gen, "docker_image", None):
                if shutil.which("bash") is None and shutil.which("sh") is not None:
                    shell = "sh"
            cmd = [shell, "build.sh"]

        build_env = os.environ.copy()
        if getattr(gen, "docker_image", None):
            include_root = "/work"
            build_env.setdefault("CC", "clang")
            build_env.setdefault("CXX", "clang++")
            build_env.setdefault("CFLAGS", "-D_GNU_SOURCE")
            build_env.setdefault("CXXFLAGS", "-D_GNU_SOURCE")
        else:
            include_root = str(gen.repo_root)
        for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
            prev = build_env.get(key, "").strip()
            build_env[key] = f"{include_root}:{prev}" if prev else include_root

        rc, out, err = gen._run_cmd(list(cmd), cwd=build_cwd, env=build_env, timeout=timeout)
        bins = gen._discover_fuzz_binaries() if rc == 0 else []
        marker = "rc-fail" if rc != 0 else ("ok" if bins else "no-fuzzers")
        signature = _sha256_text(marker + "\n" + _tail_local(out, n=200) + "\n" + _tail_local(err, n=200))
        kind, code = _classify_build_failure(
            "",
            _tail_local(out, n=200),
            _tail_local(err, n=200),
            build_rc=int(rc),
            has_fuzzer_binaries=bool(bins),
        )
        return True, {
            "rc": int(rc),
            "has_bins": bool(bins),
            "stdout_tail": _tail_local(out, n=200),
            "stderr_tail": _tail_local(err, n=200),
            "signature": signature,
            "kind": kind,
            "code": code,
            "cmd": " ".join(cmd),
            "timeout": timeout,
        }

    def _requires_env_rebuild(changed_paths: list[str] | None = None) -> bool:
        normalized = {
            str(p or "").strip().replace("\\", "/")
            for p in (changed_paths or [])
            if str(p or "").strip()
        }
        return "fuzz/system_packages.txt" in normalized

    def _success_out(message: str, *, outcome: str, rule_hit: str = "", changed_paths_count: int = 1, last_diff_paths: list[str] | None = None) -> FuzzWorkflowRuntimeState:
        updated_history, updated_rule_hits = _append_attempt(
            outcome,
            rule_hit=rule_hit,
            changed_paths_count=changed_paths_count,
        )
        out = cast(
            FuzzWorkflowRuntimeState,
            {
                **state,
                "last_step": "fix_build",
                "last_error": "",
                "codex_hint": "",
                "message": message,
                "fix_build_noop_streak": 0,
                "fix_build_attempt_history": updated_history,
                "fix_build_rule_hits": updated_rule_hits,
                "fix_build_terminal_reason": "",
                "fix_build_last_diff_paths": list(last_diff_paths or []),
                "fix_action_type": "rule" if rule_hit else "opencode",
                "fix_effect": "advanced",
            },
        )
        if _requires_env_rebuild(last_diff_paths):
            out["message"] = f"{message} (requires env rebuild)"
            out["fix_effect"] = "requires_env_rebuild"
            out["fix_build_terminal_reason"] = "requires_env_rebuild"
            return out
        probe_ran, probe = _run_fix_build_quick_probe()
        if probe_ran:
            probe_rc = int(probe.get("rc") or 1)
            probe_has_bins = bool(probe.get("has_bins"))
            probe_sig = str(probe.get("signature") or "")
            prev_sig_full = str(state.get("build_error_signature") or "")
            same_signature = bool(probe_sig and prev_sig_full and probe_sig == prev_sig_full)
            _wf_log(
                cast(dict[str, Any], state),
                "fix_build: quick-check "
                f"cmd={probe.get('cmd')} timeout={probe.get('timeout')}s rc={probe_rc} has_bins={probe_has_bins}",
            )
            if probe_rc == 0 and probe_has_bins:
                out["message"] = f"{message} (quick-check passed)"
                out["fix_effect"] = "advanced"
                return out

            next_noop_streak = (prev_noop_streak + 1) if same_signature else 0
            out["fix_build_noop_streak"] = next_noop_streak
            out["build_rc"] = probe_rc
            out["build_stdout_tail"] = str(probe.get("stdout_tail") or "")
            out["build_stderr_tail"] = str(probe.get("stderr_tail") or "")
            out["build_error_signature_before"] = prev_sig_full
            out["build_error_signature_after"] = probe_sig
            out["build_error_signature"] = probe_sig
            out["build_error_signature_short"] = probe_sig[:12]
            out["build_error_kind"] = str(probe.get("kind") or "")
            out["build_error_code"] = str(probe.get("code") or "")
            out["same_build_error_repeats"] = (int(state.get("same_build_error_repeats") or 0) + 1) if same_signature else 0
            out["last_error"] = (
                f"fix_build quick-check failed rc={probe_rc} "
                f"(same_signature={'yes' if same_signature else 'no'})"
            )
            out["message"] = "fix_build changed files but quick-check failed"
            out["fix_effect"] = "stalled" if same_signature else "advanced"
            if same_signature and next_noop_streak >= max_noop_streak:
                out["failed"] = False
                out["fix_build_terminal_reason"] = "fix_build_noop_streak_exceeded"
                out["last_error"] = f"fix_build no-op streak exceeded ({max_noop_streak}); restart from plan"
                out["message"] = "fix_build no-op streak exceeded; restarting from plan"
                out["restart_to_plan"] = True
                out["restart_to_plan_reason"] = "fix_build_noop_streak_exceeded"
                out["restart_to_plan_stage"] = "fix_build"
                out["restart_to_plan_error_text"] = str(out.get("last_error") or "")
        else:
            _wf_log(cast(dict[str, Any], state), f"fix_build: quick-check skipped ({probe.get('reason')})")
        return out

    def _detect_non_source_build_blocker(diag: str) -> str:
        checks: list[tuple[str, list[str]]] = [
            (
                "docker_daemon_unavailable",
                [
                    "cannot connect to the docker daemon",
                    "is the docker daemon running",
                    "lookup sherpa-docker",
                    "permission denied while trying to connect to the docker daemon",
                ],
            ),
            (
                "registry_or_network_unavailable",
                [
                    "tls handshake timeout",
                    "temporary failure in name resolution",
                    "failed to resolve source metadata",
                    "dial tcp",
                    "no such host",
                ],
            ),
            (
                "resource_exhausted",
                [
                    "no space left on device",
                    "cannot allocate memory",
                    "out of memory",
                    "killed",
                ],
            ),
            (
                "build_command_timeout",
                [
                    "[timeout] process exceeded limit and was killed",
                    "process exceeded limit and was killed",
                ],
            ),
        ]
        for reason, needles in checks:
            if any(n in diag for n in needles):
                return reason
        return ""

    stop_on_infra_raw = (os.environ.get("SHERPA_WORKFLOW_STOP_ON_INFRA_BUILD_ERROR") or "").strip().lower()
    stop_on_infra = stop_on_infra_raw in {"1", "true", "yes", "on"}
    non_source_reason = ""
    if build_error_kind == "infra":
        non_source_reason = build_error_code or _detect_non_source_build_blocker(diag_text) or "infra_build_failure"
    else:
        non_source_reason = _detect_non_source_build_blocker(diag_text)
    if stop_on_infra and non_source_reason:
        updated_history, updated_rule_hits = _append_attempt(
            "infra_blocked",
            rejection_reason=non_source_reason,
            changed_paths_count=0,
        )
        out = {
            **state,
            "last_step": "fix_build",
            "failed": False,
            "build_error_kind": "infra",
            "build_error_code": non_source_reason,
            "fix_build_terminal_reason": "fix_build_infra_blocked",
            "fix_build_attempt_history": updated_history,
            "fix_build_rule_hits": updated_rule_hits,
            "last_error": f"non-source build blocker detected: {non_source_reason}; restart from plan",
            "message": "fix_build skipped (environment/infrastructure issue), restarting from plan",
            "restart_to_plan": True,
            "restart_to_plan_reason": f"infra:{non_source_reason}",
            "restart_to_plan_stage": "fix_build",
            "restart_to_plan_error_text": str(last_error or "").strip(),
            "fix_action_type": "none",
            "fix_effect": "stalled",
        }
        _wf_log(cast(dict[str, Any], out), f"<- fix_build stop=non-source reason={non_source_reason}")
        return out

    build_log_file = ""
    raw_build_log_path = (state.get("build_full_log_path") or "").strip()
    if raw_build_log_path:
        p = Path(raw_build_log_path)
        if p.is_file():
            try:
                build_log_file = str(p.resolve().relative_to(gen.repo_root.resolve())).replace("\\", "/")
            except Exception:
                build_log_file = p.name
    if not build_log_file:
        default_log = gen.repo_root / "fuzz" / "build_full.log"
        if default_log.is_file():
            build_log_file = "fuzz/build_full.log"

    def _is_fix_build_allowed_path(rel_path: str) -> bool:
        rel = rel_path.strip().replace("\\", "/")
        if not rel:
            return False
        if rel == "done":
            return True
        return rel.startswith("fuzz/")

    def _collect_fix_step_hashes() -> dict[str, str]:
        repo_root = gen.repo_root
        out: dict[str, str] = {}
        skip_prefixes = (
            ".git/",
            "fuzz/out/",
            "fuzz/corpus/",
            "fuzz/build/",
        )
        skip_names = {"fuzz/build_full.log", "done"}
        for current_root, dirnames, filenames in os.walk(repo_root, topdown=True):
            try:
                root_rel = str(Path(current_root).relative_to(repo_root)).replace("\\", "/")
            except Exception:
                continue
            if root_rel == ".":
                root_rel = ""

            keep_dirs: list[str] = []
            for d in dirnames:
                rel_dir = f"{root_rel}/{d}" if root_rel else d
                rel_dir = rel_dir.replace("\\", "/")
                rel_prefix = f"{rel_dir}/"
                if rel_dir == ".git" or any(rel_prefix.startswith(pref) for pref in skip_prefixes):
                    continue
                keep_dirs.append(d)
            dirnames[:] = keep_dirs

            for name in filenames:
                rel = f"{root_rel}/{name}" if root_rel else name
                rel = rel.replace("\\", "/")
                if rel in skip_names:
                    continue
                if any(rel.startswith(pref) for pref in skip_prefixes):
                    continue
                path = repo_root / rel
                try:
                    if path.stat().st_size > 5_000_000:
                        continue
                    data = path.read_bytes()
                except Exception:
                    continue
                out[rel] = hashlib.sha256(data).hexdigest()
        return out

    def _collect_fix_relevant_hashes() -> dict[str, str]:
        fuzz_dir = gen.repo_root / "fuzz"
        if not fuzz_dir.is_dir():
            return {}
        out: dict[str, str] = {}
        skip_prefixes = ("fuzz/out/", "fuzz/corpus/", "fuzz/build/")
        skip_names = {"fuzz/build_full.log"}
        for p in fuzz_dir.rglob("*"):
            if not p.is_file():
                continue
            try:
                rel = str(p.relative_to(gen.repo_root)).replace("\\", "/")
            except Exception:
                continue
            if rel in skip_names:
                continue
            if any(rel.startswith(pref) for pref in skip_prefixes):
                continue
            try:
                data = p.read_bytes()
            except Exception:
                continue
            if len(data) > 5_000_000:
                continue
            out[rel] = hashlib.sha256(data).hexdigest()
        return out

    baseline_fix_hashes = _collect_fix_relevant_hashes()
    baseline_step_hashes = _collect_fix_step_hashes()

    # Fast-path hotfixes (minimal, no refactor):
    # 1) libstdc++/libc++ ABI mismatch from injected "-stdlib=libc++"
    # 2) libFuzzer main conflict when target sources define main()
    # 3) linking with `-lz` while the static library is only available by file path.
    def _repo_has_c_cpp_main() -> bool:
        exts = {".c", ".cc", ".cpp", ".cxx"}
        try:
            checked = 0
            for p in gen.repo_root.rglob("*"):
                if not p.is_file() or p.suffix.lower() not in exts:
                    continue
                checked += 1
                if checked > 200:
                    break
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if re.search(r"\bint\s+main\s*\(", txt):
                    return True
        except Exception:
            return False
        return False

    def _inject_define_into_flag_list(text: str, define_flag: str) -> tuple[str, bool]:
        if define_flag in text:
            return text, False
        lines = text.splitlines()
        changed = False
        in_flags = False
        for i, line in enumerate(lines):
            if not in_flags and re.search(r"^\s*(?:CXXFLAGS|flags)\s*=\s*\[", line):
                in_flags = True
                continue
            if not in_flags:
                continue
            if re.search(r"^\s*\]", line):
                indent_match = re.match(r"^(\s*)", line)
                indent = indent_match.group(1) if indent_match else "    "
                lines.insert(i, f'{indent}"{define_flag}",')
                changed = True
                break
        if changed:
            return "\n".join(lines) + ("\n" if text.endswith("\n") else ""), True
        # Fallback for common command pattern in generated build.py
        replaced = text.replace(
            " + [harness_cpp, VULNERABLE_CPP] + ",
            f" + ['{define_flag}', harness_cpp, VULNERABLE_CPP] + ",
        )
        if replaced != text:
            return replaced, True
        return text, False

    def _try_hotfix_stdlib_mismatch_and_main_conflict() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        abi_mismatch = any(
            token in diag
            for token in [
                "undefined reference to `std::__cxx11",
                "undefined reference to `std::",
                "vtable for std::",
                "libclang_rt.fuzzer",
            ]
        )
        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False
        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

        has_libcpp_flag = "-stdlib=libc++" in text
        multiple_main = ("multiple definition of `main'" in diag) or ("multiple definition of main" in diag)

        if not (abi_mismatch or has_libcpp_flag or multiple_main):
            return False

        changed = False
        # Avoid libc++/libstdc++ mismatch with clang/libFuzzer runtime in our base image.
        if has_libcpp_flag:
            text2 = text
            # Remove simple flag-list entries like:
            #   "-stdlib=libc++",
            #   '-stdlib=libc++',
            text2 = re.sub(r'^[ \t]*["\']-stdlib=libc\+\+["\'],?[ \t]*\n?', "", text2, flags=re.MULTILINE)
            # Remove conditional list entries like:
            #   ("-stdlib=libc++" if "clang" in cxx else ""),
            # without leaving broken syntax.
            text2 = re.sub(
                r'^[ \t]*\(\s*["\']-stdlib=libc\+\+["\']\s*if\s+.*?\s+else\s+["\']{0,1}["\']{0,1}\s*\)\s*,?[ \t]*\n?',
                "",
                text2,
                flags=re.MULTILINE,
            )
            # Repair previously broken malformed artifact:
            #   ( if "clang" in cxx else ""),
            text2 = re.sub(
                r'^[ \t]*\(\s*if\s+.*?\s+else\s+["\']{0,1}["\']{0,1}\s*\)\s*,?[ \t]*\n?',
                "",
                text2,
                flags=re.MULTILINE,
            )
            if text2 != text:
                text = text2
                changed = True

        # If sources define main(), rename it away from libFuzzer's main symbol.
        need_main_rename = multiple_main or _repo_has_c_cpp_main()
        if need_main_rename and "-Dmain=vuln_main" not in text:
            text, injected = _inject_define_into_flag_list(text, "-Dmain=vuln_main")
            changed = changed or injected

        # Keep legacy libFuzzer macro hotfix for compatibility with existing build.py patterns/tests.
        if multiple_main and "-DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION" not in text:
            text, injected = _inject_define_into_flag_list(text, "-DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION")
            changed = changed or injected

        if not changed:
            return False

        try:
            build_py.write_text(text, encoding="utf-8", errors="replace")
            _wf_log(
                cast(dict[str, Any], state),
                "fix_build: applied local hotfix for stdlib mismatch/main conflict",
            )
            return True
        except Exception:
            return False

    def _try_hotfix_libfuzzer_main_conflict() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        if "multiple definition of `main'" not in diag and "multiple definition of main" not in diag:
            return False

        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False

        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

        define_flag = "-DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION"
        if define_flag in text:
            return False

        lines = text.splitlines()
        changed = False
        in_flags = False
        for i, line in enumerate(lines):
            if not in_flags and re.search(r"^\s*flags\s*=\s*\[", line):
                in_flags = True
                continue
            if not in_flags:
                continue
            if "-fsanitize=fuzzer" in line:
                indent_match = re.match(r"^(\s*)", line)
                indent = indent_match.group(1) if indent_match else "        "
                lines.insert(i + 1, f"{indent}'{define_flag}',")
                changed = True
                break
            if re.search(r"^\s*\]", line):
                lines.insert(i, f"        '{define_flag}',")
                changed = True
                break

        if not changed:
            replaced = text.replace(
                "cmd = [cxx] + flags + [source_path, harness_path, '-o', output_path]",
                "cmd = [cxx, '-DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION'] + flags + [source_path, harness_path, '-o', output_path]",
            )
            if replaced != text:
                text = replaced
                changed = True
            else:
                return False
        else:
            text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")

        try:
            build_py.write_text(text, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), "fix_build: applied local hotfix for libfuzzer main conflict")
            return True
        except Exception:
            return False

    def _try_hotfix_missing_lz() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        if "cannot find -lz" not in diag and "undefined reference to `gz" not in diag and "undefined reference to `inflate" not in diag:
            return False

        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False

        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

        changed = False
        if "import glob" not in text:
            if "import os" in text:
                text = text.replace("import os", "import os\nimport glob", 1)
                changed = True
            elif "import subprocess" in text:
                text = text.replace("import subprocess", "import os\nimport glob\nimport subprocess", 1)
                changed = True

        # Strengthen search path first.
        if "-L' + os.path.join(build_dir, 'lib')" not in text:
            text2 = text.replace(
                "lib_path = ['-L' + build_dir]",
                "lib_path = ['-L' + build_dir, '-L' + os.path.join(build_dir, 'lib')]",
            )
            if text2 != text:
                text = text2
                changed = True

        # Prefer explicit static archive path to avoid flaky '-lz' resolution in container builds.
        if "zlib_link_arg = '-lz'" not in text:
            marker = "libs = ['-lz']"
            if marker in text:
                inject = (
                    "zlib_link_arg = '-lz'\n"
                    "    zlib_candidates = [\n"
                    "        os.path.join(build_dir, 'libz.a'),\n"
                    "        os.path.join(build_dir, 'lib', 'libz.a'),\n"
                    "    ]\n"
                    "    for p in glob.glob(os.path.join(build_dir, '**', 'libz.a'), recursive=True):\n"
                    "        if p not in zlib_candidates:\n"
                    "            zlib_candidates.append(p)\n"
                    "    for p in zlib_candidates:\n"
                    "        if os.path.exists(p):\n"
                    "            zlib_link_arg = p\n"
                    "            break\n"
                    "    libs = [zlib_link_arg]"
                )
                text = text.replace(marker, inject, 1)
                changed = True

        # Generic fallback for scripts that embed '-lz' directly in command arrays.
        replaced = re.sub(r"(['\"])\\-lz\\1", "zlib_link_arg", text)
        if replaced != text:
            if "zlib_link_arg = '-lz'" not in replaced:
                # Keep insertion local and simple for ad-hoc scripts.
                if "def build_target(" in replaced:
                    replaced = replaced.replace(
                        "def build_target(",
                        "zlib_link_arg = '-lz'\n\n\ndef build_target(",
                        1,
                    )
                else:
                    replaced = "zlib_link_arg = '-lz'\n" + replaced
            text = replaced
            changed = True

        if not changed:
            return False

        try:
            build_py.write_text(text, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), "fix_build: applied local hotfix for missing -lz")
            return True
        except Exception:
            return False

    def _try_hotfix_collapsed_include_flags() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False
        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False
        # High-frequency generation issue: '-I/a -I/b' produced as one argv token.
        has_file_signal = bool(re.search(r"['\"][^'\"]*-I[^'\"]+\s+-I[^'\"]*['\"]", text))
        has_diag_signal = ("no such file or directory" in diag and " -i/" in diag)
        if not (has_file_signal or has_diag_signal):
            return False

        def _split_token(tok: str) -> str:
            parts = [x for x in tok.strip().split() if x]
            if len(parts) <= 1 or not all(x.startswith("-I") for x in parts):
                return tok
            return ", ".join(f"'{x}'" for x in parts)

        changed = False

        def _repl_single(m: re.Match[str]) -> str:
            nonlocal changed
            inner = m.group(1)
            out = _split_token(inner)
            if out != inner:
                changed = True
                return out
            return m.group(0)

        # Single-quoted combined include flags.
        text2 = re.sub(r"'([^']*-I[^']+\s+-I[^']*)'", lambda m: _repl_single(m), text)
        # Double-quoted combined include flags.
        text3 = re.sub(r"\"([^\"]*-I[^\"]+\s+-I[^\"]*)\"", lambda m: _repl_single(m), text2)
        if text3 != text:
            text = text3
        if not changed:
            return False
        try:
            build_py.write_text(text, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), "fix_build: applied local hotfix for collapsed include flags")
            return True
        except Exception:
            return False

    def _try_hotfix_compiler_fuzzer_flag_mismatch() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False
        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False
        has_diag_signal = ("-fsanitize=" in diag and "fuzzer" in diag and "unrecognized argument" in diag)
        has_file_signal = ("gcc" in text and "-fsanitize=fuzzer" in text)
        if not (has_diag_signal or has_file_signal):
            return False
        text2 = text.replace("'gcc'", "'clang'").replace('"gcc"', '"clang"')
        text2 = text2.replace("'g++'", "'clang++'").replace('"g++"', '"clang++"')
        if text2 == text:
            return False
        try:
            build_py.write_text(text2, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), "fix_build: applied local hotfix for compiler_fuzzer_flag_mismatch")
            return True
        except Exception:
            return False

    def _try_hotfix_missing_llvmfuzzer_entrypoint() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        has_diag_signal = "undefined reference to `llvmfuzzertestoneinput'" in diag
        has_file_signal = build_error_code == "missing_llvmfuzzer_entrypoint"
        if not (has_diag_signal or has_file_signal):
            return False

        fuzz_dir = gen.repo_root / "fuzz"
        cpp_exts = {".cc", ".cpp", ".cxx"}
        entry_pat = re.compile(r"(?m)^(\s*)int\s+LLVMFuzzerTestOneInput\s*\(")
        extern_entry_pat = re.compile(r'(?m)^\s*extern\s+"C"\s+int\s+LLVMFuzzerTestOneInput\s*\(')
        for src in fuzz_dir.rglob("*"):
            if not src.is_file() or src.suffix.lower() not in cpp_exts:
                continue
            try:
                text = src.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if "LLVMFuzzerTestOneInput" not in text:
                continue
            if extern_entry_pat.search(text):
                continue
            if not entry_pat.search(text):
                continue
            text2 = entry_pat.sub(r'\1extern "C" int LLVMFuzzerTestOneInput(', text, count=1)
            if text2 == text:
                continue
            try:
                src.write_text(text2, encoding="utf-8", errors="replace")
                _wf_log(
                    cast(dict[str, Any], state),
                    f"fix_build: applied local hotfix for missing_llvmfuzzer_entrypoint in {src.relative_to(gen.repo_root)}",
                )
                return True
            except Exception:
                continue

        build_py = fuzz_dir / "build.py"
        if not build_py.is_file():
            return False
        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

        changed = False
        # Fallback for scripts that compile C harnesses with clang++ and rely on
        # libFuzzer's C linkage entrypoint.
        if "'clang++'" in text and ".c" in text:
            text2 = text.replace("'clang++'", "'clang'").replace('"clang++"', '"clang"')
            if text2 != text:
                text = text2
                changed = True
        if changed and "'-x'" not in text and '"-x"' not in text and "flags = [" in text:
            text2 = text.replace("flags = [", "flags = ['-x', 'c', ", 1)
            if text2 != text:
                text = text2
        if not changed:
            return False
        try:
            build_py.write_text(text, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), "fix_build: applied local hotfix for missing_llvmfuzzer_entrypoint")
            return True
        except Exception:
            return False

    def _try_hotfix_cxx_for_c_source_mismatch() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        if "treating 'c' input as 'c++'" not in diag and "treated as c++" not in diag:
            return False
        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False
        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False
        text2 = text.replace("'clang++'", "'clang'").replace('"clang++"', '"clang"')
        if text2 == text:
            return False
        try:
            build_py.write_text(text2, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), "fix_build: applied local hotfix for cxx_for_c_source_mismatch")
            return True
        except Exception:
            return False

    def _try_hotfix_c_compiler_for_cpp_source_mismatch() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        if (
            "invalid argument '-std=c++" not in diag
            and "this file requires compiler and library support for the iso c++" not in diag
            and "unknown type name 'namespace'" not in diag
        ):
            return False
        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False
        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False
        has_cpp_signal = any(x in text for x in [".cc", ".cpp", ".cxx", "-std=c++"])
        if not has_cpp_signal:
            return False
        text2 = re.sub(r"(['\"])clang\1", r"\1clang++\1", text)
        text2 = re.sub(r"(['\"])gcc\1", r"\1g++\1", text2)
        if text2 == text:
            return False
        try:
            build_py.write_text(text2, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), "fix_build: applied local hotfix for c_compiler_for_cpp_source_mismatch")
            return True
        except Exception:
            return False

    def _try_hotfix_missing_symbol_include() -> bool:
        diag_raw = last_error + "\n" + stdout_tail + "\n" + stderr_tail
        if "undeclared identifier" not in diag_raw.lower():
            return False

        symbol_rules: list[tuple[re.Pattern[str], str]] = [
            (re.compile(r"^archive_entry_"), "#include <archive_entry.h>"),
            (re.compile(r"^archive_(read|write|format|filter|error|version|match|util|string)_"), "#include <archive.h>"),
        ]
        include_edits: dict[Path, set[str]] = {}
        for m in re.finditer(
            r"(?m)^(?P<file>[^:\n]+(?:\.cc|\.cpp|\.cxx|\.c)):\d+:\d+:\s+error:\s+use of undeclared identifier '(?P<sym>[A-Za-z_][A-Za-z0-9_]*)'",
            diag_raw,
        ):
            raw_file = str(m.group("file")).strip()
            sym = str(m.group("sym")).strip()
            if not raw_file or not sym:
                continue
            src = Path(raw_file)
            if not src.is_absolute():
                src = gen.repo_root / src
            if not src.is_file():
                continue
            include_line = ""
            for pat, inc in symbol_rules:
                if pat.search(sym):
                    include_line = inc
                    break
            if not include_line:
                continue
            include_edits.setdefault(src, set()).add(include_line)

        if not include_edits:
            return False

        for src, include_lines in include_edits.items():
            try:
                text = src.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            lines = text.splitlines()
            insert_at = 0
            for i, line in enumerate(lines):
                if line.lstrip().startswith("#include"):
                    insert_at = i + 1
            to_insert = [inc for inc in sorted(include_lines) if inc not in text]
            if not to_insert:
                continue
            for inc in to_insert:
                lines.insert(insert_at, inc)
                insert_at += 1
            new_text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
            if new_text == text:
                continue
            try:
                src.write_text(new_text, encoding="utf-8", errors="replace")
                _wf_log(cast(dict[str, Any], state), f"fix_build: applied local hotfix for missing include(s) in {src}")
                return True
            except Exception:
                continue
        return False

    def _try_hotfix_missing_system_packages() -> bool:
        alias_map = {
            "z": "zlib",
            "bz2": "bzip2",
            "lzma": "liblzma",
            "xz": "liblzma",
            "ssl": "openssl",
            "crypto": "openssl",
            "libssl": "openssl",
            "libcrypto": "openssl",
            "xml2": "libxml2",
            "libxml": "libxml2",
        }
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        if "cannot find -lz" in diag or "undefined reference to `gz" in diag or "undefined reference to `inflate" in diag:
            # Prefer dedicated link-fix rule for zlib linker failures.
            return False
        pkg_signals: list[tuple[list[str], str]] = [
            (["zlib.h", "could not find zlib", "cannot find -lz"], "zlib"),
            (["bzlib.h", "could not find bzip2"], "bzip2"),
            (["lzma.h", "could not find liblzma"], "liblzma"),
            (["zstd.h", "could not find zstd", "one of the modules 'libzstd'"], "zstd"),
            (["lz4.h", "could not find lz4"], "lz4"),
            (["openssl/", "could not find openssl"], "openssl"),
            (["expat.h", "could not find expat"], "expat"),
            (["libxml/parser.h", "could not find libxml2"], "libxml2"),
        ]
        need_pkgs: list[str] = []
        for needles, pkg in pkg_signals:
            if any(n in diag for n in needles):
                need_pkgs.append(pkg)
        if not need_pkgs:
            return False
        dep_file = gen.repo_root / "fuzz" / "system_packages.txt"
        existing: list[str] = []
        if dep_file.is_file():
            try:
                for line in dep_file.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    token = line.split("#", 1)[0].strip().lower()
                    if not token or not re.fullmatch(r"[a-z0-9][a-z0-9+._-]*", token):
                        continue
                    existing.append(alias_map.get(token, token))
            except Exception:
                return False
        merged = sorted(set(existing) | set(need_pkgs))
        if merged == sorted(set(existing)):
            return False
        dep_file.parent.mkdir(parents=True, exist_ok=True)
        body = (
            "# Auto-maintained by fix_build hotfix rules.\n"
            "# Package names are vcpkg ports (not apt package names).\n"
            + "\n".join(merged)
            + "\n"
        )
        try:
            dep_file.write_text(body, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), f"fix_build: declared system packages in {dep_file}")
            return True
        except Exception:
            return False
    def _try_hotfix_fuzz_out_path_mismatch() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False
        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False
        has_diag_signal = ("no fuzzer binaries found under fuzz/out" in diag or "build produced no fuzzers" in diag)
        has_file_signal = ('out_dir="fuzz/out"' in text)
        if not (has_diag_signal or has_file_signal):
            return False
        changed = False
        if 'out_dir="fuzz/out"' in text:
            text = text.replace('out_dir="fuzz/out"', 'out_dir="out"')
            changed = True
        if "os.path.abspath(out_dir)" not in text and "def build_all(" in text and "os.makedirs(out_dir" in text:
            text = text.replace("os.makedirs(out_dir, exist_ok=True)", "abs_out_dir = os.path.abspath(out_dir)\n    os.makedirs(abs_out_dir, exist_ok=True)")
            text = text.replace("compile_target(name, target_info, out_dir, cc)", "compile_target(name, target_info, abs_out_dir, cc)")
            changed = True
        if not changed:
            return False
        try:
            build_py.write_text(text, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), "fix_build: applied local hotfix for fuzz_out_path_mismatch")
            return True
        except Exception:
            return False

    def _try_hotfix_source_build_dir_collision() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        collision_signals = [
            "build/version",
            "build/cmake",
            "cmakelists.txt: could not find requested file",
            "include(cmake/checkfileoffsetbits.cmake)",
        ]
        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False
        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False
        uses_repo_build = (
            "BUILD_DIR = REPO_ROOT / \"build\"" in text
            or "BUILD_DIR=REPO_ROOT / \"build\"" in text
            or "BUILD_DIR = REPO_ROOT/'build'" in text
        )
        destructive_clean = ("shutil.rmtree(BUILD_DIR" in text or "rm -rf \"$BUILD_DIR\"" in text)
        if not ((any(sig in diag for sig in collision_signals) or uses_repo_build) and uses_repo_build and destructive_clean):
            return False

        new_text = text
        changed = False
        if "BUILD_DIR = REPO_ROOT / \"build\"" in new_text:
            new_text = new_text.replace(
                "BUILD_DIR = REPO_ROOT / \"build\"",
                "BUILD_DIR = REPO_ROOT / \"fuzz\" / \"build-work\"",
            )
            changed = True
        if "BUILD_DIR=REPO_ROOT / \"build\"" in new_text:
            new_text = new_text.replace(
                "BUILD_DIR=REPO_ROOT / \"build\"",
                "BUILD_DIR=REPO_ROOT / \"fuzz\" / \"build-work\"",
            )
            changed = True
        if "BUILD_DIR = REPO_ROOT/'build'" in new_text:
            new_text = new_text.replace(
                "BUILD_DIR = REPO_ROOT/'build'",
                "BUILD_DIR = REPO_ROOT/'fuzz'/'build-work'",
            )
            changed = True

        if new_text == text or not changed:
            return False
        try:
            build_py.write_text(new_text, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), "fix_build: applied local hotfix for source_build_dir_collision")
            return True
        except Exception:
            return False

    def _try_hotfix_missing_cmake_archive_target() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        target_miss_signals = [
            "no rule to make target 'archive'",
            'no rule to make target "archive"',
            "unknown target archive",
        ]
        if not any(sig in diag for sig in target_miss_signals):
            return False

        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False
        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

        changed = False
        new_text = text
        replacements = [
            ("'--target', 'archive'", "'--target', 'all'"),
            ('"--target", "archive"', '"--target", "all"'),
            ("'--target','archive'", "'--target','all'"),
            ('"--target","archive"', '"--target","all"'),
        ]
        for old, new in replacements:
            if old in new_text:
                new_text = new_text.replace(old, new)
                changed = True
        if changed and new_text != text:
            try:
                build_py.write_text(new_text, encoding="utf-8", errors="replace")
                _wf_log(cast(dict[str, Any], state), "fix_build: replaced cmake --target archive with --target all")
            except Exception:
                return False

        pkg_signals: list[tuple[list[str], str]] = [
            (['could not find zlib', 'zlib_library', 'zlib_include_dir'], 'zlib'),
            (['could not find bzip2', 'bzip2_libraries', 'bzip2_include_dir'], 'bzip2'),
            (['could not find liblzma', 'liblzma_library', 'liblzma_include_dir'], 'liblzma'),
            (['could not find lz4', 'lz4_library', 'lz4_include_dir'], 'lz4'),
            (['could not find zstd', "one of the modules 'libzstd'", 'zstd_library'], 'zstd'),
            (['could not find openssl', 'openssl_crypto_library', 'openssl_include_dir'], 'openssl'),
            (['could not find expat', 'expat_library', 'expat_include_dir'], 'expat'),
            (['could not find libxml2', 'libxml2_library', 'libxml2_include_dir'], 'libxml2'),
        ]
        need_pkgs: list[str] = []
        for needles, pkg in pkg_signals:
            if any(n in diag for n in needles):
                need_pkgs.append(pkg)
        if need_pkgs:
            alias_map = {
                "z": "zlib",
                "bz2": "bzip2",
                "lzma": "liblzma",
                "xz": "liblzma",
                "ssl": "openssl",
                "crypto": "openssl",
                "libssl": "openssl",
                "libcrypto": "openssl",
                "xml2": "libxml2",
                "libxml": "libxml2",
            }
            dep_file = gen.repo_root / 'fuzz' / 'system_packages.txt'
            existing: list[str] = []
            if dep_file.is_file():
                try:
                    for line in dep_file.read_text(encoding='utf-8', errors='replace').splitlines():
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        token = line.split("#", 1)[0].strip().lower()
                        if not token or not re.fullmatch(r"[a-z0-9][a-z0-9+._-]*", token):
                            continue
                        existing.append(alias_map.get(token, token))
                except Exception:
                    existing = []
            merged = sorted(set(existing) | set(need_pkgs))
            if merged != sorted(set(existing)):
                dep_file.parent.mkdir(parents=True, exist_ok=True)
                body = (
                    '# Auto-maintained by fix_build hotfix rules.\n'
                    '# Package names are vcpkg ports (not apt package names).\n'
                    + '\n'.join(merged)
                    + '\n'
                )
                try:
                    dep_file.write_text(body, encoding='utf-8', errors='replace')
                    _wf_log(cast(dict[str, Any], state), f'fix_build: declared system packages in {dep_file}')
                    changed = True
                except Exception:
                    pass

        return changed

    if _fix_build_ruleset() == "extended":
        if _try_hotfix_compiler_fuzzer_flag_mismatch():
            out = _success_out(
                "local hotfix for compiler_fuzzer_flag_mismatch applied",
                outcome="rule_fixed",
                rule_hit="compiler_fuzzer_flag_mismatch",
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        if _try_hotfix_cxx_for_c_source_mismatch():
            out = _success_out(
                "local hotfix for cxx_for_c_source_mismatch applied",
                outcome="rule_fixed",
                rule_hit="cxx_for_c_source_mismatch",
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        if _try_hotfix_collapsed_include_flags():
            out = _success_out(
                "local hotfix for collapsed include flags applied",
                outcome="rule_fixed",
                rule_hit="collapsed_include_flags",
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        if _try_hotfix_missing_llvmfuzzer_entrypoint():
            out = _success_out(
                "local hotfix for missing_llvmfuzzer_entrypoint applied",
                outcome="rule_fixed",
                rule_hit="missing_llvmfuzzer_entrypoint",
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        if _try_hotfix_fuzz_out_path_mismatch():
            out = _success_out(
                "local hotfix for fuzz_out_path_mismatch applied",
                outcome="rule_fixed",
                rule_hit="fuzz_out_path_mismatch",
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        if _try_hotfix_source_build_dir_collision():
            out = _success_out(
                "local hotfix for source_build_dir_collision applied",
                outcome="rule_fixed",
                rule_hit="source_build_dir_collision",
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        if _try_hotfix_missing_cmake_archive_target():
            out = _success_out(
                "local hotfix for missing_cmake_archive_target applied",
                outcome="rule_fixed",
                rule_hit="missing_cmake_archive_target",
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        if _try_hotfix_c_compiler_for_cpp_source_mismatch():
            out = _success_out(
                "local hotfix for c_compiler_for_cpp_source_mismatch applied",
                outcome="rule_fixed",
                rule_hit="c_compiler_for_cpp_source_mismatch",
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        if _try_hotfix_missing_symbol_include():
            out = _success_out(
                "local hotfix for missing symbol include applied",
                outcome="rule_fixed",
                # Keep legacy rule name for compatibility with existing dashboards/tests.
                rule_hit="archive_entry_missing_include",
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        if _try_hotfix_missing_system_packages():
            out = _success_out(
                "local hotfix for missing system package declarations applied",
                outcome="rule_fixed",
                rule_hit="missing_system_packages_declared",
                last_diff_paths=["fuzz/system_packages.txt"],
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

    if _try_hotfix_stdlib_mismatch_and_main_conflict():
        out = _success_out(
            "local hotfix for stdlib mismatch/main conflict applied",
            outcome="rule_fixed",
            rule_hit="stdlib_mismatch_or_abi",
        )
        _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out

    if _try_hotfix_libfuzzer_main_conflict():
        out = _success_out(
            "local hotfix for libfuzzer main conflict applied",
            outcome="rule_fixed",
            rule_hit="main_symbol_conflict",
        )
        _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out

    if _try_hotfix_missing_lz():
        out = _success_out(
            "local hotfix for -lz applied",
            outcome="rule_fixed",
            rule_hit="missing_zlib_link_flag",
        )
        _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out

    stdout_for_summary = re.sub(r"(?m)^\[\s*\d+%]\s+Built target\s+.*$", "", stdout_tail).strip()
    summary = _summarize_build_error(last_error, stdout_for_summary, stderr_tail)
    recent_history = history[-history_limit:] if history else []
    build_strategy_doc = _load_build_strategy_doc(gen.repo_root)
    build_runtime_facts_doc = _load_build_runtime_facts_doc(gen.repo_root)
    repo_understanding_doc = _load_repo_understanding_doc(gen.repo_root)

    # Ask an LLM to draft an *OpenCode instruction* tailored to the diagnostics.
    llm = _llm_or_none()
    codex_hint = (state.get("codex_hint") or "").strip()

    targeted_fix_lines = _build_file_targeted_fix_lines(gen.repo_root, last_error, stdout_tail, stderr_tail)
    if not codex_hint:
        if llm is not None:
            coordinator_prompt = (
                "You are coordinating OpenCode to fix a fuzz harness build.\n"
                "Given the build diagnostics, produce a short instruction for OpenCode.\n\n"
                "Requirements for your output:\n"
                "- Output JSON only: {\"codex_hint\": \"...\"}\n"
                "- codex_hint must be 1-10 lines, concrete and minimal.\n"
                "- codex_hint must be in English only.\n"
                "- Extract concrete failing file paths from diagnostics when possible.\n"
                "- Include at least one line in the form: `Read and fix <path>[:line]` when file evidence exists.\n"
                "- Tell OpenCode to only change files under fuzz/.\n"
                "- Any change outside fuzz/ (except ./done sentinel) is rejected.\n"
                + (f"- Tell OpenCode to read full build logs from `{build_log_file}` before editing.\n" if build_log_file else "")
                + "- IMPORTANT: Tell OpenCode to NOT run any commands — only edit files.\n"
                "- Acceptance: `(cd fuzz && python build.py)` succeeds and leaves at least one executable in fuzz/out/.\n\n"
                f"repo_root={repo_root}\n"
                + f"error_type={summary['error_type']}\n"
                + (f"build_log_file={build_log_file}\n" if build_log_file else "")
                + (f"last_error={last_error}\n" if last_error else "")
                + ("\n=== STDOUT (tail) ===\n" + stdout_tail + "\n" if stdout_tail else "")
                + ("\n=== STDERR (tail) ===\n" + stderr_tail + "\n" if stderr_tail else "")
                + "\n=== STRUCTURED EVIDENCE ===\n" + summary["evidence"] + "\n"
                + "\nReturn JSON only."
            )
            try:
                resp = llm.invoke(coordinator_prompt)
                text = getattr(resp, "content", None) or str(resp)
                obj = _extract_json_object(text) or {}
                codex_hint = str(obj.get("codex_hint") or "").strip()
            except Exception:
                codex_hint = ""
            if codex_hint and _contains_cjk_text(codex_hint):
                codex_hint = ""

        if not codex_hint:
            codex_hint = (
                (f"First read `{build_log_file}` for the complete build logs, then apply the minimal fix.\n" if build_log_file else "")
                +
                "Fix the fuzz build so that running `(cd fuzz && python build.py)` succeeds and leaves at least one executable fuzzer under fuzz/out/.\n"
                "Keep the scaffold grounded in `fuzz/repo_understanding.json`; repair that file first if the current build path is underspecified.\n"
                "Only modify files under fuzz/. Any change outside fuzz/ (except ./done sentinel) will be rejected.\n"
                "Do not use `-stdlib=libc++` in this environment.\n"
                "If target sources define `main`, add a compile define such as `-Dmain=vuln_main` to avoid libFuzzer main conflicts.\n"
                "If include/link flags are wrong, fix them from fuzz/build.py or fuzz harness code only.\n"
                "Do not invoke repository-provided fuzz targets or guessed `--target ...fuzzer` build commands.\n"
                "Always build the repository library/objects and link the generated harness externally.\n"
                "Do not refactor production code or edit upstream source files."
            )
        if targeted_fix_lines:
            codex_hint = (codex_hint.strip() + "\n" + "\n".join(targeted_fix_lines)).strip()
        if build_error_code == "build_strategy_mismatch":
            codex_hint = (
                codex_hint.strip()
                + "\nThe current scaffold incorrectly depends on a repository fuzz target. Rewrite fuzz/build.py to avoid any repo fuzz target invocation and use external harness linking only."
            )
        if build_error_code == "missing_fuzzer_main":
            codex_hint = (
                codex_hint.strip()
                + "\nThe current scaffold is missing a fuzzer main strategy. Add `-fsanitize=fuzzer` or explicitly compile a repo-provided main source as a normal source input."
            )
        if build_error_code == "insufficient_repo_understanding":
            codex_hint = (
                codex_hint.strip()
                + "\nThe current scaffold lacks grounded repository understanding. Repair `fuzz/repo_understanding.json` first with concrete build facts and evidence, then make `fuzz/build.py` match it."
            )
        if build_error_code == "non_public_api_usage":
            codex_hint = (
                codex_hint.strip()
                + "\nDiagnostics indicate non-public/internal API usage in harness code. Replace offending symbols with public/stable APIs first."
                + "\nIf no public API exists, declare `api_surface_exception` in `fuzz/repo_understanding.json` with non-empty `reason` and `evidence` (and optional `approved_symbols`)."
            )
        if recent_history and any("noop" in str(x.get("outcome") or "") for x in recent_history):
            codex_hint = (
                codex_hint.strip()
                + "\nPrevious attempts were no-op; this attempt MUST produce at least one meaningful change under fuzz/."
            )

    # Build an error-heavy, noise-reduced context for fix_build.
    def _tail_lines(text: str, n: int = 120) -> str:
        lines = str(text or "").replace("\r", "\n").splitlines()
        return "\n".join(lines[-n:]).strip()

    def _tail_chars(text: str, max_chars: int) -> str:
        s = str(text or "").strip()
        if max_chars <= 0 or len(s) <= max_chars:
            return s
        return s[-max_chars:]

    def _denoise_build_stdout(text: str) -> str:
        lines = str(text or "").replace("\r", "\n").splitlines()
        if not lines:
            return ""
        noise_patterns = [
            re.compile(r"^\[\s*\d+%]\s+Built target\s+", re.IGNORECASE),
            re.compile(r"^--\s*Configuring done\b", re.IGNORECASE),
            re.compile(r"^--\s*Generating done\b", re.IGNORECASE),
            re.compile(r"^--\s*Build files have been written to:\b", re.IGNORECASE),
            re.compile(r"^\s*done\s*$", re.IGNORECASE),
        ]
        kept: list[str] = []
        for ln in lines:
            raw = ln.rstrip("\n")
            if any(p.search(raw.strip()) for p in noise_patterns):
                continue
            kept.append(raw)
        return "\n".join(kept).strip()

    def _dedupe_stderr_blocks(text: str, keep_recent: int) -> str:
        raw = str(text or "").replace("\r", "\n").strip()
        if not raw:
            return ""
        blocks = [b.strip() for b in re.split(r"\n{2,}", raw) if b.strip()]
        if not blocks:
            return raw

        seen_first: dict[str, int] = {}
        seen_latest: dict[str, int] = {}
        for i, block in enumerate(blocks):
            first_line = next((ln.strip().lower() for ln in block.splitlines() if ln.strip()), "")
            key = first_line[:220] or block[:220].lower()
            if key not in seen_first:
                seen_first[key] = i
            seen_latest[key] = i

        selected: list[int] = []
        for key, first_idx in seen_first.items():
            selected.append(first_idx)
            last_idx = seen_latest.get(key, first_idx)
            if last_idx != first_idx:
                selected.append(last_idx)
        selected = sorted(set(selected))
        if keep_recent > 0 and len(selected) > keep_recent * 2:
            selected = selected[-(keep_recent * 2) :]
        return "\n\n".join(blocks[i] for i in selected).strip()

    explicit_actions = [
        line.strip()
        for line in codex_hint.splitlines()
        if line.strip().lower().startswith("read and fix ")
    ]
    if not explicit_actions:
        explicit_actions = [line.strip() for line in targeted_fix_lines if line.strip()]

    previous_failed_attempts: list[dict[str, Any]] = []
    for row in recent_history[-context_history_limit:]:
        outcome = str(row.get("outcome") or "").strip()
        if not outcome:
            continue
        previous_failed_attempts.append(
            {
                "attempt": int(row.get("attempt_index") or 0),
                "outcome": outcome,
                "changed_paths_count": int(row.get("changed_paths_count") or 0),
                "signature": str(row.get("classified_signature") or "").strip(),
                "error_code": str(row.get("build_error_code") or "").strip(),
                "reason": str(row.get("rejection_reason") or "").strip(),
            }
        )

    file_refs: dict[str, Any] = {}
    if build_log_file:
        file_refs["build_log_file"] = build_log_file
    if isinstance(build_strategy_doc, dict):
        file_refs["fuzz/build_strategy.json"] = {
            "build_system": str(build_strategy_doc.get("build_system") or ""),
            "build_mode": str(build_strategy_doc.get("build_mode") or ""),
            "fuzzer_entry_strategy": str(build_strategy_doc.get("fuzzer_entry_strategy") or ""),
            "library_targets": list(build_strategy_doc.get("library_targets") or [])[:5],
        }
    if isinstance(build_runtime_facts_doc, dict):
        file_refs["fuzz/build_runtime_facts.json"] = {
            "build_system": str(build_runtime_facts_doc.get("build_system") or ""),
            "build_mode": str(build_runtime_facts_doc.get("build_mode") or ""),
            "required_outputs": list(build_runtime_facts_doc.get("required_outputs") or [])[:5],
        }
    if isinstance(repo_understanding_doc, dict):
        file_refs["fuzz/repo_understanding.json"] = {
            "build_system": str(repo_understanding_doc.get("build_system") or ""),
            "chosen_target_api": str(repo_understanding_doc.get("chosen_target_api") or ""),
            "fuzzer_entry_strategy": str(repo_understanding_doc.get("fuzzer_entry_strategy") or ""),
        }

    stderr_text = _tail_chars(
        _dedupe_stderr_blocks(stderr_tail, keep_recent=_fix_build_keep_recent_errors()),
        _fix_build_stderr_max_chars(),
    )
    stdout_text = _tail_chars(_denoise_build_stdout(stdout_tail), _fix_build_stdout_max_chars())

    p0_blocks: list[str] = []
    if stderr_text:
        p0_blocks.append("=== build stderr diagnostics ===\n" + stderr_text)

    p1_blocks: list[str] = ["=== structured_error ===\n" + json.dumps(summary, ensure_ascii=False, indent=2)]
    if last_error:
        p1_blocks.append("=== last_error ===\n" + _tail_lines(last_error, n=80))

    p2_blocks: list[str] = []
    if previous_failed_attempts:
        p2_blocks.append(
            "=== previous_failed_attempts ===\n" + json.dumps(previous_failed_attempts, ensure_ascii=False, indent=2)
        )
    p3_blocks: list[str] = []
    if stdout_text:
        p3_blocks.append("=== build stdout relevant ===\n" + stdout_text)
    p4_blocks: list[str] = []
    if explicit_actions:
        p4_blocks.append("=== targeted_file_actions ===\n" + "\n".join(f"- {line}" for line in explicit_actions))
    if file_refs:
        p4_blocks.append("=== context_file_refs ===\n" + json.dumps(file_refs, ensure_ascii=False, indent=2))

    mandatory = p0_blocks + p1_blocks
    optional = p2_blocks + p3_blocks + p4_blocks

    packed: list[str] = []
    current_len = 0
    for block in mandatory:
        b = str(block or "").strip()
        if not b:
            continue
        sep = 2 if packed else 0
        packed.append(b)
        current_len += len(b) + sep
    for block in optional:
        b = str(block or "").strip()
        if not b:
            continue
        sep = 2 if packed else 0
        if current_len + len(b) + sep > context_max_chars:
            continue
        packed.append(b)
        current_len += len(b) + sep
    context = "\n\n".join(packed).strip()
    if len(context) > context_max_chars:
        context = context[:context_max_chars]

    prompt = _render_opencode_prompt(
        "fix_build_execute",
        codex_hint=codex_hint.strip(),
        build_log_file=build_log_file or "fuzz/build_full.log",
    )

    try:
        _wf_log(cast(dict[str, Any], state), f"fix_build: running opencode (hint_lines={len(codex_hint.splitlines())})")
        gen.patcher.run_codex_command(
            prompt,
            additional_context=context or None,
            stage_skill="fix_build",
            timeout=_remaining_time_budget_sec(state),
            max_attempts=1,
            max_cli_retries=_opencode_cli_retries(),
        )
        post_step_hashes = _collect_fix_step_hashes()
        changed_paths = sorted(
            p
            for p in (set(baseline_step_hashes.keys()) | set(post_step_hashes.keys()))
            if baseline_step_hashes.get(p) != post_step_hashes.get(p)
        )
        effective_changed_paths = [p for p in changed_paths if str(p).strip().replace("\\", "/") != "done"]
        changed_paths_count = len(effective_changed_paths)
        llm_outcome = "llm_fixed" if changed_paths_count > 0 else "llm_noop"
        updated_history, updated_rule_hits = _append_attempt(
            llm_outcome,
            changed_paths_count=changed_paths_count,
        )
        next_noop_streak = prev_noop_streak + 1 if changed_paths_count == 0 else 0
        message = "opencode fixed build" if changed_paths_count > 0 else "opencode returned without code changes"
        last_error_text = "" if changed_paths_count > 0 else (last_error or "fix_build produced no file changes")
        fix_effect = "advanced" if changed_paths_count > 0 else "stalled"
        out = {
            **state,
            "last_step": "fix_build",
            "last_error": last_error_text,
            "codex_hint": "",
            "message": message,
            "fix_build_noop_streak": next_noop_streak,
            "fix_build_attempt_history": updated_history,
            "fix_build_rule_hits": updated_rule_hits,
            "fix_build_terminal_reason": "",
            "fix_build_last_diff_paths": effective_changed_paths,
            "fix_action_type": "opencode",
            "fix_effect": fix_effect,
        }
        if changed_paths_count == 0 and next_noop_streak >= max_noop_streak:
            out["failed"] = False
            out["fix_build_terminal_reason"] = "fix_build_noop_streak_exceeded"
            out["last_error"] = f"fix_build no-op streak exceeded ({max_noop_streak}); restart from plan"
            out["message"] = "fix_build no-op streak exceeded; restarting from plan"
            out["restart_to_plan"] = True
            out["restart_to_plan_reason"] = "fix_build_noop_streak_exceeded"
            out["restart_to_plan_stage"] = "fix_build"
            out["restart_to_plan_error_text"] = str(out.get("last_error") or "")
        if changed_paths_count > 0 and _requires_env_rebuild(effective_changed_paths):
            out["message"] = "opencode fixed build (requires env rebuild)"
            out["fix_effect"] = "requires_env_rebuild"
            out["fix_build_terminal_reason"] = "requires_env_rebuild"
        _wf_log(cast(dict[str, Any], out), f"<- fix_build ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    except Exception as e:
        updated_history, updated_rule_hits = _append_attempt(
            "exception",
            rejection_reason=str(e),
            changed_paths_count=0,
        )
        out = {
            **state,
            "last_step": "fix_build",
            "last_error": str(e),
            "message": "opencode fix_build failed",
            "fix_build_attempt_history": updated_history,
            "fix_build_rule_hits": updated_rule_hits,
            "fix_build_last_diff_paths": [],
            "fix_action_type": "opencode",
            "fix_effect": "regressed",
        }
        _wf_log(cast(dict[str, Any], out), f"<- fix_build err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


def _node_run(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "run")
    if stop_now:
        return state
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> run")
    try:
        # If we've already seen crashes in a previous round, archive old artifacts so
        # new crashes are detectable.
        fix_attempts = int(state.get("crash_fix_attempts") or 0)
        if fix_attempts:
            try:
                art_dir = gen.fuzz_out_dir / "artifacts"
                if art_dir.is_dir():
                    archive = art_dir / f"old-{fix_attempts}"
                    archive.mkdir(exist_ok=True)
                    for p in art_dir.glob("*"):
                        if p.is_file():
                            p.rename(archive / p.name)
            except Exception:
                pass

        bins = gen._discover_fuzz_binaries()
        if not bins:
            raise HarnessGeneratorError("No fuzzer binaries found under fuzz/out/")

        crash_found = False
        last_artifact = ""
        last_fuzzer = ""
        run_rc = 0
        crash_evidence = "none"
        run_error_kind = ""
        run_terminal_reason = ""
        run_idle_seconds = 0
        run_last_error = ""
        run_details: list[dict[str, Any]] = []
        run_batch_plan: list[dict[str, Any]] = []
        run_children_exit_count = 0
        run_cancel_requested_count = 0
        run_cancel_effective_count = 0
        total_time_budget = _wf_common.parse_budget_value(state.get("time_budget"), default=900)
        run_time_budget_raw = state.get("run_time_budget")
        if run_time_budget_raw is None:
            configured_run_time_budget = total_time_budget
        else:
            configured_run_time_budget = _wf_common.parse_budget_value(run_time_budget_raw, default=total_time_budget)
        if configured_run_time_budget < 0:
            raise HarnessGeneratorError("run_time_budget must be >= 0")
        total_budget_unlimited = total_time_budget <= 0
        prev_crash_sig = str(state.get("crash_signature") or "").strip()
        prev_crash_repeats = int(state.get("same_crash_repeats") or 0)
        prev_timeout_sig = str(state.get("timeout_signature") or "").strip()
        prev_timeout_repeats = int(state.get("same_timeout_repeats") or 0)
        max_same_crash_repeats_raw = os.environ.get("SHERPA_WORKFLOW_MAX_SAME_CRASH_REPEATS", "1")
        try:
            max_same_crash_repeats = max(0, min(int(max_same_crash_repeats_raw), 10))
        except Exception:
            max_same_crash_repeats = 1
        max_same_timeout_repeats = _max_same_timeout_repeats()
        auto_stop_policy = _auto_stop_policy()
        max_parallel_raw = os.environ.get("SHERPA_PARALLEL_FUZZERS", "3")
        try:
            requested_outer_parallelism = max(1, min(int(max_parallel_raw), 64))
        except Exception:
            requested_outer_parallelism = 3
        stop_on_first_crash = _run_stop_on_first_crash()
        parallel_early_stop = _run_parallel_early_stop_enabled()
        if stop_on_first_crash and len(bins) > 1 and not parallel_early_stop:
            # Compatibility mode: force serial when parallel early stop is disabled.
            requested_outer_parallelism = 1
        cpu_budget = _run_cpu_budget()
        outer_parallelism_max = _run_outer_parallelism_max(requested_outer_parallelism)
        inner_workers_min = _run_inner_workers_min()
        requested_inner_workers = _run_inner_workers_target()
        requested_engine = _run_parallel_engine()
        ignore_non_fatal = _run_ignore_non_fatal_enabled()
        solved_parallel = _solve_parallelism(
            cpu_budget=cpu_budget,
            n_targets=len(bins),
            requested_outer=requested_outer_parallelism,
            outer_parallelism_max=outer_parallelism_max,
            inner_workers_min=inner_workers_min,
            requested_inner=requested_inner_workers,
            engine=requested_engine,
            sanitizer=str(getattr(gen, "sanitizer", "") or ""),
        )
        max_parallel = int(solved_parallel.get("outer_parallelism") or 1)
        inner_workers = int(solved_parallel.get("inner_workers") or 1)
        parallel_engine = str(solved_parallel.get("parallel_engine") or "single")
        reload_enabled = bool(solved_parallel.get("reload_enabled"))
        parallel_warning = str(solved_parallel.get("warning") or "").strip()
        idle_timeout_sec = _run_idle_timeout_sec()
        finalize_timeout_sec = _run_finalize_timeout_sec()

        current_run_parallel_cfg = {
            bin_path.name: {
                "parallel_engine": parallel_engine,
                "parallel_role": "reserved",
                "outer_slot": idx % max(1, max_parallel),
                "inner_workers": inner_workers,
                "reload_enabled": reload_enabled,
                "ignore_non_fatal": ignore_non_fatal,
            }
            for idx, bin_path in enumerate(bins)
        }
        prev_run_parallel_cfg = getattr(gen, "current_run_parallel_config_by_fuzzer", None)
        setattr(gen, "current_run_parallel_config_by_fuzzer", current_run_parallel_cfg)

        _wf_log(
            cast(dict[str, Any], state),
            (
                f"run: fuzzers={len(bins)} parallel_outer={max_parallel} inner={inner_workers} "
                f"engine={parallel_engine} cpu_budget={cpu_budget} "
                f"stop_on_first_crash={int(stop_on_first_crash)} "
                f"parallel_early_stop={int(parallel_early_stop)}"
            ),
        )
        if parallel_warning:
            _wf_log(cast(dict[str, Any], state), f"run: {parallel_warning}")

        def _calc_crash_signature(fuzzer_name: str, artifact_path: str) -> str:
            parts: list[str] = [f"fuzzer={fuzzer_name}", f"artifact={artifact_path}"]
            crash_info = gen.repo_root / "crash_info.md"
            crash_analysis = gen.repo_root / "crash_analysis.md"
            combined_log = ""
            for p in (crash_info, crash_analysis):
                if not p.is_file():
                    continue
                try:
                    txt = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                tail = "\n".join(txt.splitlines()[-400:])
                parts.append(f"== {p.name} ==\n{tail}")
                combined_log += txt + "\n"
            # Also compute stack-based signature for better dedup
            stack_sig = extract_crash_stack_signature(combined_log)
            nonlocal _last_stack_sig
            _last_stack_sig = stack_sig
            return _sha256_text("\n\n".join(parts))

        _last_stack_sig: dict[str, str] = {}

        def _calc_timeout_signature(kind: str, details: list[dict[str, Any]]) -> str:
            parts: list[str] = [f"kind={kind}"]
            for d in details[:5]:
                parts.append(
                    "|".join(
                        [
                            str(d.get("fuzzer") or ""),
                            str(d.get("run_error_kind") or ""),
                            str(d.get("effective_rc") or d.get("rc") or ""),
                            str(d.get("error") or "")[:400],
                            str(d.get("first_artifact") or ""),
                        ]
                    )
                )
            return _sha256_text("\n".join(parts))

        _wf_log(cast(dict[str, Any], state), "run: generating AI seeds before fuzzing")
        # Seed generation uses OpenCode and shared repo context; keep it serial.
        prev_seed_timeout = getattr(gen, "seed_generation_timeout_sec", None)
        last_seed_profile = str(state.get("coverage_seed_profile") or "")
        seed_count_total: dict[str, int] = {"repo_examples": 0, "ai": 0, "radamsa": 0, "total": 0}
        seed_count_raw_total: dict[str, int] = {"repo_examples": 0, "ai": 0, "radamsa": 0, "total": 0}
        seed_count_filtered_total: dict[str, int] = {"repo_examples": 0, "ai": 0, "radamsa": 0, "total": 0}
        seed_generation_failed_fuzzers: list[str] = []
        seed_generation_error_by_fuzzer: dict[str, str] = {}

        def _accumulate_seed_counts(dst: dict[str, int], src: Any) -> None:
            if not isinstance(src, dict):
                return
            for key in ("repo_examples", "ai", "radamsa", "total"):
                try:
                    dst[key] = int(dst.get(key, 0)) + int(src.get(key) or 0)
                except Exception:
                    continue

        seed_sources: set[str] = set()
        repo_examples_filtered = False
        repo_examples_rejected_count = 0
        repo_examples_accepted_count = 0
        seed_noise_rejected_count = 0
        missing_execution_targets: list[str] = []
        seed_family_coverage_state: dict[str, Any] = {}
        execution_plan_doc = _load_execution_plan_doc(gen.repo_root)
        execution_targets = [
            item for item in list(execution_plan_doc.get("execution_targets") or [])
            if isinstance(item, dict)
        ]
        bins_by_name = {p.name: p for p in bins}
        bins_by_stem = {p.stem: p for p in bins}
        seed_fuzzers: list[Path] = []
        if execution_targets:
            for item in execution_targets:
                expected = str(item.get("expected_fuzzer_name") or item.get("target_name") or "").strip()
                candidate = bins_by_name.get(expected) or bins_by_stem.get(Path(expected).stem)
                if candidate is not None:
                    if candidate not in seed_fuzzers:
                        seed_fuzzers.append(candidate)
                else:
                    missing_name = str(item.get("target_name") or expected)
                    if missing_name and missing_name not in missing_execution_targets:
                        missing_execution_targets.append(missing_name)
        if not seed_fuzzers:
            seed_fuzzers = list(bins)
        try:
            for idx, bin_path in enumerate(seed_fuzzers):
                remaining_for_seed = _remaining_time_budget_sec(state, min_timeout=0)
                if remaining_for_seed <= 0:
                    return _time_budget_exceeded_state(state, step_name="run")
                fuzzers_left = len(seed_fuzzers) - idx
                per_fuzzer_budget = max(1, remaining_for_seed // max(1, fuzzers_left))
                setattr(gen, "seed_generation_timeout_sec", per_fuzzer_budget)
                fuzzer_name = bin_path.name
                try:
                    gen._pass_generate_seeds(fuzzer_name)
                    profile_map = getattr(gen, "last_seed_profile_by_fuzzer", {}) or {}
                    if not last_seed_profile:
                        last_seed_profile = str(profile_map.get(fuzzer_name) or "")
                    bootstrap_map = getattr(gen, "last_seed_bootstrap_by_fuzzer", {}) or {}
                    meta = bootstrap_map.get(fuzzer_name) or {}
                    if isinstance(meta, dict):
                        _accumulate_seed_counts(seed_count_total, meta.get("counts") or {})
                        _accumulate_seed_counts(seed_count_raw_total, meta.get("seed_counts_raw") or {})
                        _accumulate_seed_counts(seed_count_filtered_total, meta.get("seed_counts_filtered") or {})
                        sources = meta.get("sources") or []
                        if isinstance(sources, list):
                            for src in sources:
                                src_text = str(src or "").strip()
                                if src_text:
                                    seed_sources.add(src_text)
                        repo_examples_filtered = bool(meta.get("repo_examples_filtered") or repo_examples_filtered)
                        repo_examples_rejected_count += int(meta.get("repo_examples_rejected_count") or 0)
                        repo_examples_accepted_count += int(meta.get("repo_examples_accepted_count") or 0)
                        seed_noise_rejected_count += int(meta.get("seed_noise_rejected_count") or 0)
                        if not seed_family_coverage_state and isinstance(meta.get("seed_family_coverage"), dict):
                            seed_family_coverage_state = dict(meta.get("seed_family_coverage") or {})
                except Exception as e:
                    # Seed generation is best-effort; do not block fuzzing.
                    seed_generation_failed_fuzzers.append(fuzzer_name)
                    seed_generation_error_by_fuzzer[fuzzer_name] = str(e)[:400]
                    logger.info(f"[warn] seed generation skipped ({fuzzer_name}): {e}")
        finally:
            setattr(gen, "seed_generation_timeout_sec", prev_seed_timeout)

        run_results: dict[str, FuzzerRunResult] = {}
        run_exec_errors: dict[str, str] = {}
        finalized_fuzzers: set[str] = set()
        first_crash_fuzzer = ""
        early_stop_reason = ""
        early_stopped_fuzzers: list[str] = []

        def _run_one(bin_path: Path) -> tuple[str, FuzzerRunResult]:
            return bin_path.name, gen._run_fuzzer(bin_path)

        def _capture_timeout_from_error(err_text: str) -> tuple[str, int]:
            lowered = (err_text or "").lower()
            if "idle-timeout" in lowered:
                return "run_idle_timeout", idle_timeout_sec
            if "timed out after" in lowered or "[timeout]" in lowered:
                return "run_timeout", 0
            return "", 0

        # Execute fuzzers in parallel batches and cap each batch to remaining total budget.
        pending_bins = list(bins)
        prev_run_budget = getattr(gen, "current_run_time_budget_sec", None)
        prev_run_hard_timeout = getattr(gen, "current_run_hard_timeout_sec", None)
        try:
            while pending_bins:
                remaining_for_run = _remaining_time_budget_sec(state, min_timeout=0)
                if remaining_for_run <= 0:
                    if not run_last_error:
                        run_last_error = "time budget exceeded during run phase"
                    if not run_error_kind:
                        run_error_kind = "workflow_time_budget_exceeded"
                    for skipped in pending_bins:
                        run_exec_errors[skipped.name] = "skipped: workflow total time budget exhausted before execution"
                        finalized_fuzzers.add(skipped.name)
                    pending_bins = []
                    break

                rounds_left, round_budget, hard_timeout = _calc_parallel_batch_budget(
                    pending_count=len(pending_bins),
                    max_parallel=max_parallel,
                    remaining_for_run=remaining_for_run,
                    configured_run_time_budget=configured_run_time_budget,
                    total_budget_unlimited=total_budget_unlimited,
                )
                setattr(gen, "current_run_time_budget_sec", round_budget)
                setattr(gen, "current_run_hard_timeout_sec", hard_timeout)

                batch = pending_bins[:max_parallel]
                pending_bins = pending_bins[max_parallel:]
                run_batch_plan.append(
                    {
                        "round": len(run_batch_plan) + 1,
                        "batch_size": len(batch),
                        "pending_before": len(batch) + len(pending_bins),
                        "rounds_left": rounds_left,
                        "remaining_total_budget_sec": remaining_for_run,
                        "round_budget_sec": round_budget,
                        "hard_timeout_sec": hard_timeout,
                    }
                )
                _wf_log(
                    cast(dict[str, Any], state),
                    (
                        "run batch: "
                        f"size={len(batch)} round_budget={round_budget}s hard_timeout={hard_timeout}s "
                        f"remaining_total={remaining_for_run}s"
                    ),
                )

                if len(batch) <= 1:
                    for bin_path in batch:
                        try:
                            name, run = _run_one(bin_path)
                            run_results[name] = run
                            finalized_fuzzers.add(name)
                            run_children_exit_count += 1
                            if stop_on_first_crash and run.crash_found:
                                pending_bins = []
                                break
                        except Exception as e:
                            run_exec_errors[bin_path.name] = str(e)
                            finalized_fuzzers.add(bin_path.name)
                            run_children_exit_count += 1
                            detected_kind, detected_idle = _capture_timeout_from_error(str(e))
                            if detected_kind and not run_error_kind:
                                run_error_kind = detected_kind
                                run_terminal_reason = detected_kind
                                if detected_idle > 0:
                                    run_idle_seconds = detected_idle
                else:
                    with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                        futures = {pool.submit(_run_one, bin_path): bin_path for bin_path in batch}
                        batch_should_stop = False
                        processed_futures: set[Any] = set()
                        for fut in as_completed(futures):
                            bin_path = futures[fut]
                            processed_futures.add(fut)
                            try:
                                name, run = fut.result()
                                run_results[name] = run
                                finalized_fuzzers.add(name)
                                run_children_exit_count += 1
                                if (
                                    stop_on_first_crash
                                    and parallel_early_stop
                                    and run.crash_found
                                ):
                                    first_crash_fuzzer = str(name)
                                    early_stop_reason = "first_crash_parallel_early_stop"
                                    terminator = getattr(gen, "terminate_active_run_processes", None)
                                    if callable(terminator):
                                        try:
                                            terminator(reason=f"first_crash:{name}")
                                        except Exception:
                                            pass
                                    for pending_fut in futures:
                                        if pending_fut is not fut:
                                            run_cancel_requested_count += 1
                                            try:
                                                if pending_fut.cancel():
                                                    run_cancel_effective_count += 1
                                            except Exception:
                                                pass
                                    for other_bin in batch:
                                        if other_bin.name != name and other_bin.name not in early_stopped_fuzzers:
                                            early_stopped_fuzzers.append(other_bin.name)
                                    # Collect already-finished futures before leaving this batch
                                    # so early-stop does not drop completed results.
                                    for remaining_fut, remaining_bin in futures.items():
                                        if remaining_fut in processed_futures or remaining_fut is fut:
                                            continue
                                        if not remaining_fut.done():
                                            continue
                                        processed_futures.add(remaining_fut)
                                        try:
                                            rname, rrun = remaining_fut.result(timeout=0)
                                            run_results[rname] = rrun
                                            finalized_fuzzers.add(rname)
                                            run_children_exit_count += 1
                                        except Exception as e:
                                            run_exec_errors[remaining_bin.name] = str(e)
                                            finalized_fuzzers.add(remaining_bin.name)
                                            run_children_exit_count += 1
                                    batch_should_stop = True
                                    break
                            except Exception as e:
                                run_exec_errors[bin_path.name] = str(e)
                                finalized_fuzzers.add(bin_path.name)
                                run_children_exit_count += 1
                                detected_kind, detected_idle = _capture_timeout_from_error(str(e))
                                if detected_kind and not run_error_kind:
                                    run_error_kind = detected_kind
                                    run_terminal_reason = detected_kind
                                    if detected_idle > 0:
                                        run_idle_seconds = detected_idle
                        if batch_should_stop:
                            pending_bins = []
                if stop_on_first_crash and any(run.crash_found for run in run_results.values()):
                    if not first_crash_fuzzer:
                        for crash_name, crash_run in run_results.items():
                            if crash_run.crash_found:
                                first_crash_fuzzer = crash_name
                                break
                    if not early_stop_reason and first_crash_fuzzer:
                        early_stop_reason = "first_crash_stop"
                    for skipped in pending_bins:
                        if skipped.name not in early_stopped_fuzzers:
                            early_stopped_fuzzers.append(skipped.name)
                    pending_bins = []
                    break
        finally:
            setattr(gen, "current_run_time_budget_sec", prev_run_budget)
            setattr(gen, "current_run_hard_timeout_sec", prev_run_hard_timeout)
            setattr(gen, "current_run_parallel_config_by_fuzzer", prev_run_parallel_cfg)

        _wf_log(cast(dict[str, Any], state), "run children exited, collecting results...")
        finalize_started = time.perf_counter()
        finalize_deadline = (
            finalize_started + float(finalize_timeout_sec) if finalize_timeout_sec > 0 else None
        )

        def _finalize_timed_out(stage: str) -> bool:
            nonlocal run_error_kind, run_terminal_reason, run_last_error
            if finalize_deadline is None:
                return False
            if time.perf_counter() <= finalize_deadline:
                return False
            run_error_kind = "run_finalize_timeout"
            run_terminal_reason = "run_finalize_timeout"
            run_last_error = f"run finalize timed out while {stage} (>{finalize_timeout_sec}s)"
            return True

        first_nonzero_rc = 0
        crash_candidates: list[tuple[str, Path, FuzzerRunResult]] = []

        def _make_run_detail_fallback(
            *,
            fuzzer_name: str,
            rc: int,
            run_error_kind_value: str,
            exception_kind: str,
            error: str,
        ) -> dict[str, Any]:
            return {
                "fuzzer": fuzzer_name,
                "rc": rc,
                "effective_rc": rc,
                "crash_found": False,
                "crash_evidence": "none",
                "run_error_kind": run_error_kind_value,
                "exception_kind": exception_kind,
                "error": error,
                "new_artifacts": [],
                "first_artifact": "",
                "final_cov": 0,
                "final_ft": 0,
                "final_iteration": 0,
                "final_execs_per_sec": 0,
                "final_rss_mb": 0,
                "final_corpus_files": 0,
                "final_corpus_size_bytes": 0,
                "corpus_files": 0,
                "corpus_size_bytes": 0,
                "seed_quality": {},
                "parallel_engine": str((current_run_parallel_cfg.get(fuzzer_name) or {}).get("parallel_engine") or "single"),
                "parallel_role": str((current_run_parallel_cfg.get(fuzzer_name) or {}).get("parallel_role") or "reserved"),
                "outer_slot": int((current_run_parallel_cfg.get(fuzzer_name) or {}).get("outer_slot") or 0),
                "inner_workers": int((current_run_parallel_cfg.get(fuzzer_name) or {}).get("inner_workers") or 1),
                "reload_enabled": bool((current_run_parallel_cfg.get(fuzzer_name) or {}).get("reload_enabled")),
            }

        for bin_path in bins:
            if _finalize_timed_out("collecting run details"):
                break
            fuzzer_name = bin_path.name
            if fuzzer_name not in finalized_fuzzers:
                continue
            exec_err = run_exec_errors.get(fuzzer_name, "")
            if exec_err:
                detail_kind = "run_exception"
                detail_rc = 1
                if not run_last_error:
                    run_last_error = f"fuzzer run crashed for {fuzzer_name}: {exec_err}"
                if not run_error_kind:
                    run_error_kind = "run_exception"
                detected_kind, detected_idle = _capture_timeout_from_error(exec_err)
                if detected_kind and not run_terminal_reason:
                    run_terminal_reason = detected_kind
                    if detected_idle > 0:
                        run_idle_seconds = detected_idle
                    detail_kind = detected_kind
                    detail_rc = 124 if detected_kind in {"run_timeout", "run_idle_timeout"} else 1
                if first_nonzero_rc == 0:
                    first_nonzero_rc = detail_rc
                run_details.append(
                    _make_run_detail_fallback(
                        fuzzer_name=fuzzer_name,
                        rc=detail_rc,
                        run_error_kind_value=detail_kind,
                        exception_kind=detail_kind,
                        error=exec_err,
                    )
                )
                continue

            run = run_results.get(fuzzer_name)
            if run is None:
                # Defensive fallback: if a future completed without result/exception.
                if not run_last_error:
                    run_last_error = f"missing run result for {fuzzer_name}"
                if not run_error_kind:
                    run_error_kind = "run_exception"
                if first_nonzero_rc == 0:
                    first_nonzero_rc = 1
                run_details.append(
                    _make_run_detail_fallback(
                        fuzzer_name=fuzzer_name,
                        rc=1,
                        run_error_kind_value="run_exception",
                        exception_kind="run_exception",
                        error="missing run result",
                    )
                )
                continue

            rc = int(run.rc)
            if first_nonzero_rc == 0 and rc != 0:
                first_nonzero_rc = rc
            if not run_error_kind and run.run_error_kind:
                run_error_kind = run.run_error_kind
            if run.run_error_kind in {"run_idle_timeout", "run_timeout"} and not run_terminal_reason:
                run_terminal_reason = run.run_error_kind
                if run.run_error_kind == "run_idle_timeout":
                    run_idle_seconds = idle_timeout_sec
            if not run_terminal_reason and str(run.terminal_reason or "").strip():
                run_terminal_reason = str(run.terminal_reason).strip()
                if run.terminal_reason == "coverage_plateau":
                    run_idle_seconds = int(run.plateau_idle_seconds or 0)

            run_details.append(
                {
                    "fuzzer": fuzzer_name,
                    "rc": rc,
                    "effective_rc": rc,
                    "crash_found": bool(run.crash_found),
                    "crash_evidence": run.crash_evidence,
                    "run_error_kind": run.run_error_kind,
                    "exception_kind": "",
                    "error": run.error or "",
                    "log_tail": run.log_tail or "",
                    "new_artifacts": [str(p) for p in (run.new_artifacts or [])],
                    "first_artifact": run.first_artifact or "",
                    "final_cov": int(run.final_cov),
                    "final_ft": int(run.final_ft),
                    "final_iteration": int(run.final_iteration),
                    "final_execs_per_sec": int(run.final_execs_per_sec),
                    "final_rss_mb": int(run.final_rss_mb),
                    "final_corpus_files": int(run.final_corpus_files),
                    "final_corpus_size_bytes": int(run.final_corpus_size_bytes),
                    "corpus_files": int(run.corpus_files),
                    "corpus_size_bytes": int(run.corpus_size_bytes),
                    "terminal_reason": str(run.terminal_reason or ""),
                    "plateau_detected": bool(run.plateau_detected),
                    "plateau_idle_seconds": int(run.plateau_idle_seconds or 0),
                    "plateau_hit_count": int(run.plateau_hit_count or 0),
                    "plateau_last_hit_at": float(run.plateau_last_hit_at or 0.0),
                    "progress_sample_file": str(run.progress_sample_file or ""),
                    "seed_quality": dict(run.seed_quality or {}),
                    "parallel_engine": str(run.parallel_engine or "single"),
                    "parallel_role": str(run.parallel_role or "reserved"),
                    "outer_slot": int(run.outer_slot or 0),
                    "inner_workers": int(run.inner_workers or 1),
                    "reload_enabled": bool(run.reload_enabled),
                }
            )
            if run.error and not run_last_error:
                run_last_error = run.error
            if run.crash_found and run.first_artifact:
                crash_candidates.append((fuzzer_name, Path(run.first_artifact), run))

        # Detect "no real progress" runs so the workflow can repair instead of silently
        # ending in a false-success/false-running state.
        if not crash_candidates and not run_last_error:
            no_progress_fuzzers: list[str] = []
            seed_rejected_fuzzers: list[str] = []
            for detail in run_details:
                if bool(detail.get("crash_found")):
                    continue
                if int(detail.get("rc") or 0) != 0:
                    continue
                final_execs = int(detail.get("final_execs_per_sec") or 0)
                final_cov = int(detail.get("final_cov") or 0)
                final_ft = int(detail.get("final_ft") or 0)
                final_corpus_files = int(detail.get("final_corpus_files") or 0)
                final_corpus_size_bytes = int(detail.get("final_corpus_size_bytes") or 0)
                log_or_err = f"{detail.get('error') or ''}\n{detail.get('log_tail') or ''}".lower()
                warned_no_progress = (
                    "no interesting inputs were found so far" in log_or_err
                    or "inited exec/s: 0" in log_or_err
                    or "exec/s: 0" in log_or_err
                )
                if warned_no_progress and final_execs > 0 and final_cov <= 0 and final_ft <= 0 and (
                    final_corpus_files <= 1 or final_corpus_size_bytes <= 1
                ):
                    seed_rejected_fuzzers.append(str(detail.get("fuzzer") or "unknown"))
                if final_execs <= 0 and warned_no_progress:
                    no_progress_fuzzers.append(str(detail.get("fuzzer") or "unknown"))
            if seed_rejected_fuzzers:
                run_error_kind = "run_seed_rejected"
                joined = ", ".join(seed_rejected_fuzzers[:5])
                run_last_error = (
                    "fuzzer inputs were likely rejected by target parser "
                    f"(no interesting inputs, zero cov/ft, tiny corpus): {joined}"
                )
            if no_progress_fuzzers:
                if not run_error_kind:
                    run_error_kind = "run_no_progress"
                    joined = ", ".join(no_progress_fuzzers[:5])
                    run_last_error = (
                        "fuzzer run made no measurable progress "
                        f"(exec/s=0 with no-interesting-input warnings): {joined}"
                    )

        if crash_candidates:
            if _finalize_timed_out("packaging crash artifacts"):
                crash_found = False
                run_rc = 1
                crash_evidence = "none"
            else:
                last_fuzzer, first, crash_run = crash_candidates[0]
                gen._analyze_and_package(last_fuzzer, first)
                crash_found = True
                last_artifact = str(first)
                run_rc = int(crash_run.rc)
                crash_evidence = crash_run.crash_evidence
        else:
            run_rc = first_nonzero_rc
            crash_evidence = "none"

        if crash_found:
            msg = "Fuzzing completed (crash found and packaged)."
        elif run_last_error:
            msg = "Fuzzing run failed."
        else:
            msg = "Fuzzing completed."

        seed_bootstrap_all = getattr(gen, "last_seed_bootstrap_by_fuzzer", {}) or {}

        def _first_seed_meta_list(*path: str) -> list[str]:
            for meta in seed_bootstrap_all.values():
                if not isinstance(meta, dict):
                    continue
                cur: Any = meta
                ok = True
                for key in path:
                    if not isinstance(cur, dict):
                        ok = False
                        break
                    cur = cur.get(key)
                if not ok:
                    continue
                if isinstance(cur, (list, tuple, set)):
                    out_vals = [str(v).strip() for v in cur if str(v).strip()]
                    if out_vals:
                        return out_vals
            return []

        crash_signature = ""
        same_crash_repeats = 0
        if crash_found and last_fuzzer and last_artifact:
            crash_signature = _calc_crash_signature(last_fuzzer, last_artifact)
            same_crash_repeats = (prev_crash_repeats + 1) if (prev_crash_sig and crash_signature == prev_crash_sig) else 0

        timeout_signature = ""
        same_timeout_repeats = 0
        timeout_like_kinds = {"run_timeout", "run_idle_timeout", "run_finalize_timeout", "run_no_progress"}
        if run_error_kind in timeout_like_kinds:
            timeout_signature = _calc_timeout_signature(run_error_kind, run_details)
            same_timeout_repeats = (
                (prev_timeout_repeats + 1)
                if (prev_timeout_sig and timeout_signature == prev_timeout_sig)
                else 0
            )

        out = {
            **state,
            "last_step": "run",
            "last_error": run_last_error,
            "crash_found": crash_found,
            "run_rc": run_rc,
            "crash_evidence": crash_evidence,
            "run_error_kind": run_error_kind,
            "run_terminal_reason": run_terminal_reason,
            "run_idle_seconds": int(run_idle_seconds or 0),
            "run_children_exit_count": int(run_children_exit_count),
            "run_cancel_requested_count": int(run_cancel_requested_count),
            "run_cancel_effective_count": int(run_cancel_effective_count),
            "run_details": run_details,
            "run_batch_plan": run_batch_plan,
            "run_parallel_engine": parallel_engine,
            "run_parallel_outer": int(max_parallel),
            "run_parallel_inner": int(inner_workers),
            "run_parallel_cpu_budget": int(cpu_budget),
            "first_crash_fuzzer": first_crash_fuzzer,
            "early_stop_reason": early_stop_reason,
            "early_stopped_fuzzers": list(early_stopped_fuzzers),
            "last_crash_artifact": last_artifact,
            "last_fuzzer": last_fuzzer,
            "coverage_target_name": (
                str(state.get("synthesize_observed_target_api") or "").strip()
                or last_fuzzer
                or next(
                    (
                        str(detail.get("fuzzer") or "").strip()
                        for detail in run_details
                        if str(detail.get("fuzzer") or "").strip()
                    ),
                    str(state.get("coverage_target_name") or ""),
                )
            ),
            "coverage_target_api": (
                str(state.get("synthesize_observed_target_api") or "").strip()
                or str(state.get("selected_target_api") or "").strip()
            ),
            "coverage_seed_profile": last_seed_profile,
            "coverage_seed_quality": next(
                (dict(detail.get("seed_quality") or {}) for detail in run_details if detail.get("seed_quality")),
                dict(state.get("coverage_seed_quality") or {}),
            ),
            "coverage_seed_families_required": list(
                _first_seed_meta_list("seed_families_required")
                or list(state.get("coverage_seed_families_required") or [])
            ),
            "coverage_seed_families_covered": list(
                _first_seed_meta_list("seed_family_coverage", "covered")
                or list(state.get("coverage_seed_families_covered") or [])
            ),
            "coverage_seed_families_missing": list(
                _first_seed_meta_list("seed_family_coverage", "missing")
                or list(state.get("coverage_seed_families_missing") or [])
            ),
            "coverage_quality_flags": list(
                next(
                    (
                        list((detail.get("seed_quality") or {}).get("quality_flags") or [])
                        for detail in run_details
                        if detail.get("seed_quality")
                    ),
                    list(state.get("coverage_quality_flags") or []),
                )
            ),
            "coverage_target_depth_score": int(state.get("coverage_target_depth_score") or 0),
            "coverage_target_depth_class": str(state.get("coverage_target_depth_class") or ""),
            "coverage_selection_bias_reason": str(state.get("coverage_selection_bias_reason") or ""),
            "coverage_corpus_sources": sorted(seed_sources),
            "coverage_seed_counts": seed_count_total,
            "coverage_seed_counts_raw": seed_count_raw_total,
            "coverage_seed_counts_filtered": seed_count_filtered_total,
            "coverage_seed_noise_rejected_count": seed_noise_rejected_count,
            "coverage_seed_generation_failed_fuzzers": list(seed_generation_failed_fuzzers),
            "coverage_seed_generation_error_by_fuzzer": dict(seed_generation_error_by_fuzzer),
            "coverage_seed_generation_failed_count": int(len(seed_generation_failed_fuzzers)),
            "coverage_seed_generation_degraded": bool(
                seed_generation_failed_fuzzers
                or int(seed_count_filtered_total.get("total") or 0) <= 1
            ),
            "coverage_missing_execution_targets": missing_execution_targets,
            "coverage_seed_family_coverage": seed_family_coverage_state,
            "coverage_repo_examples_filtered": repo_examples_filtered,
            "coverage_repo_examples_rejected_count": repo_examples_rejected_count,
            "coverage_repo_examples_accepted_count": repo_examples_accepted_count,
            "crash_signature": crash_signature,
            "crash_stack_signature": _last_stack_sig.get("stack_signature", ""),
            "crash_stack_type": _last_stack_sig.get("crash_type", ""),
            "crash_stack_top_frames": _last_stack_sig.get("top_frames", ""),
            "same_crash_repeats": same_crash_repeats,
            "timeout_signature": timeout_signature,
            "same_timeout_repeats": same_timeout_repeats,
            "message": msg,
            "auto_stop_policy": auto_stop_policy,
            "auto_stop_blocked_reason": str(state.get("auto_stop_blocked_reason") or ""),
            "continuous_loop_count": int(state.get("continuous_loop_count") or 0),
        }
        if run_error_kind == "workflow_time_budget_exceeded":
            out["failed"] = True
            out["last_error"] = out.get("last_error") or "time budget exceeded during run phase"
            out["message"] = "workflow stopped (time budget exceeded)"
        if run_error_kind in {"run_idle_timeout", "run_timeout", "run_finalize_timeout"}:
            if run_error_kind == "run_idle_timeout":
                out["message"] = "run stalled (idle timeout), routing to plan-repair"
                if not out.get("last_error"):
                    out["last_error"] = f"run stalled: no output for >= {idle_timeout_sec}s"
            elif run_error_kind == "run_finalize_timeout":
                out["message"] = "run finalize timed out, routing to plan-repair"
            else:
                out["message"] = "run timed out, routing to plan-repair"
        if crash_found and same_crash_repeats >= max_same_crash_repeats:
            out["failed"] = True
            out["last_error"] = (
                "same crash signature repeated after crash-fix attempts "
                f"(repeats={same_crash_repeats + 1}, threshold={max_same_crash_repeats + 1})"
            )
            out["message"] = "workflow stopped (same crash repeated)"
        if run_error_kind in timeout_like_kinds and same_timeout_repeats >= max_same_timeout_repeats:
            if auto_stop_policy == "legacy_mixed":
                out["failed"] = True
                out["last_error"] = (
                    "same timeout/no-progress signature repeated "
                    f"(repeats={same_timeout_repeats + 1}, threshold={max_same_timeout_repeats + 1})"
                )
                out["message"] = "workflow stopped (same timeout/no-progress repeated)"
            else:
                out["auto_stop_blocked_reason"] = "same_timeout_repeats"
                out["continuous_loop_count"] = int(out.get("continuous_loop_count") or 0) + 1
                out["message"] = "same timeout/no-progress repeated; continue under hard_fail_only"
        if crash_found and last_fuzzer and last_artifact:
            _write_repro_context(
                gen.repo_root,
                repo_url=str(out.get("repo_url") or ""),
                last_fuzzer=last_fuzzer,
                last_crash_artifact=last_artifact,
                crash_signature=crash_signature,
                re_workspace_root=str(out.get("re_workspace_root") or ""),
            )
        quality_flags = list(out.get("coverage_quality_flags") or [])
        if bool(state.get("synthesize_target_drifted")):
            quality_flags.append("target_runtime_mismatch")
        if list(out.get("coverage_seed_families_missing") or []):
            quality_flags.append("seed_family_undercovered")
        if list(out.get("coverage_missing_execution_targets") or []):
            quality_flags.append("missing_execution_targets")
        raw_total = int((out.get("coverage_seed_counts_raw") or {}).get("total") or 0)
        noise_rejected = int(out.get("coverage_seed_noise_rejected_count") or 0)
        if noise_rejected > 0 and (raw_total <= 0 or float(noise_rejected) / float(max(raw_total, 1)) >= 0.25):
            quality_flags.append("seed_noise_high")
        observed_api = str(out.get("coverage_target_api") or "").lower()
        if observed_api in {"println", "fmt::println", "print", "fmt::print", "format", "fmt::format", "format_to", "fmt::format_to", "vformat", "fmt::vformat"}:
            quality_flags.append("generic_wrapper_fallback")
        out["coverage_quality_flags"] = sorted({flag for flag in quality_flags if flag})
        out["coverage_seed_feedback"] = _build_seed_feedback(cast(dict[str, Any], out))
        out["coverage_harness_feedback"] = _build_harness_feedback(cast(dict[str, Any], out))
        try:
            seed_feedback_path = gen.repo_root / "fuzz" / "seed_feedback.json"
            seed_feedback_path.parent.mkdir(parents=True, exist_ok=True)
            by_fuzzer: dict[str, Any] = {}
            for detail in run_details:
                fuzzer_name = str(detail.get("fuzzer") or "").strip()
                if not fuzzer_name:
                    continue
                seed_quality = dict(detail.get("seed_quality") or {})
                if not seed_quality:
                    continue
                by_fuzzer[fuzzer_name] = {
                    "seed_profile": str(seed_quality.get("seed_profile") or out.get("coverage_seed_profile") or ""),
                    "initial_inited_cov": int(seed_quality.get("initial_inited_cov") or 0),
                    "final_cov": int(seed_quality.get("final_cov") or 0),
                    "cov_delta": int(seed_quality.get("cov_delta") or 0),
                    "initial_inited_ft": int(seed_quality.get("initial_inited_ft") or 0),
                    "final_ft": int(seed_quality.get("final_ft") or 0),
                    "ft_delta": int(seed_quality.get("ft_delta") or 0),
                    "early_new_units_30s": int(seed_quality.get("early_new_units_30s") or 0),
                    "early_new_units_60s": int(seed_quality.get("early_new_units_60s") or 0),
                    "initial_corpus_files": int(seed_quality.get("initial_corpus_files") or 0),
                    "final_corpus_files": int(seed_quality.get("final_corpus_files") or 0),
                    "quality_flags": list(seed_quality.get("quality_flags") or []),
                    "missing_required_families": list(out.get("coverage_seed_families_missing") or []),
                    "merge_retained_ratio_files": float(seed_quality.get("merge_retained_ratio_files") or 1.0),
                    "merge_retained_ratio_bytes": float(seed_quality.get("merge_retained_ratio_bytes") or 1.0),
                    "cold_start_failure": bool(seed_quality.get("cold_start_failure") or False),
                    "updated_at": int(time.time()),
                }
            failed_seed_gen = set(out.get("coverage_seed_generation_failed_fuzzers") or [])
            if failed_seed_gen:
                for fuzzer_name in sorted(failed_seed_gen):
                    item = dict(by_fuzzer.get(fuzzer_name) or {})
                    item["seed_generation_failed"] = True
                    item["seed_generation_error"] = str(
                        (out.get("coverage_seed_generation_error_by_fuzzer") or {}).get(fuzzer_name) or ""
                    )
                    item["updated_at"] = int(time.time())
                    by_fuzzer[fuzzer_name] = item
            seed_feedback_doc = {
                "version": 1,
                "updated_at": int(time.time()),
                "job_id": str(state.get("job_id") or ""),
                "repo_url": str(state.get("repo_url") or ""),
                "seed_generation_degraded": bool(out.get("coverage_seed_generation_degraded") or False),
                "seed_generation_failed_count": int(out.get("coverage_seed_generation_failed_count") or 0),
                "seed_generation_failed_fuzzers": list(out.get("coverage_seed_generation_failed_fuzzers") or []),
                "by_fuzzer": by_fuzzer,
            }
            seed_feedback_path.write_text(
                json.dumps(seed_feedback_doc, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            out["coverage_seed_feedback_path"] = str(seed_feedback_path)
        except Exception as exc:
            _wf_log(cast(dict[str, Any], state), f"run: failed to write seed_feedback.json: {exc}")
        if run_error_kind:
            last_detail = run_details[-1] if run_details else {}
            attempt_index = int(state.get("repair_attempt_index") or 0) + 1
            out["repair_mode"] = True
            out["repair_origin_stage"] = "crash"
            out["repair_error_kind"] = run_error_kind
            out["repair_error_code"] = run_terminal_reason or run_error_kind
            out["repair_signature"] = str(timeout_signature or crash_signature or "")[:12]
            out["repair_stdout_tail"] = str(last_detail.get("stdout_tail") or "")
            out["repair_stderr_tail"] = str(last_detail.get("stderr_tail") or "")
            out["repair_attempt_index"] = attempt_index
            out["repair_strategy_force_change"] = False
            out["repair_error_digest"] = {
                "error_code": str(out.get("repair_error_code") or ""),
                "error_kind": run_error_kind,
                "signature": str(out.get("repair_signature") or ""),
                "failing_files": [],
                "symbols": [],
                "first_seen": int(time.time()),
                "latest_seen": int(time.time()),
                "top_trace": _extract_repair_top_trace(
                    str(out.get("last_error") or ""),
                    str(last_detail.get("stdout_tail") or ""),
                    str(last_detail.get("stderr_tail") or ""),
                ),
            }
            recent = list(state.get("repair_recent_attempts") or [])
            recent.append(
                {
                    "step": "run",
                    "origin": "crash",
                    "error_kind": run_error_kind,
                    "error_code": run_terminal_reason or run_error_kind,
                    "signature": out["repair_signature"],
                    "attempt_index": attempt_index,
                    "message": str(out.get("last_error") or "")[:512],
                }
            )
            out["repair_recent_attempts"] = recent[-5:]
        elif not crash_found:
            out["repair_mode"] = False
            out["repair_origin_stage"] = ""
            out["repair_error_kind"] = ""
            out["repair_error_code"] = ""
            out["repair_signature"] = ""
            out["repair_stdout_tail"] = ""
            out["repair_stderr_tail"] = ""
            out["repair_recent_attempts"] = []
            out["repair_error_digest"] = {}
            out["repair_attempt_index"] = 0
            out["repair_strategy_force_change"] = False
        _wf_log(
            cast(dict[str, Any], out),
            (
                f"<- run ok crash_found={crash_found} rc={run_rc} evidence={crash_evidence} "
                f"same_crash_repeats={same_crash_repeats} same_timeout_repeats={same_timeout_repeats} "
                f"dt={_fmt_dt(time.perf_counter()-t0)}"
            ),
        )
        _emit_fuzz_metrics(cast(dict[str, Any], out))
        return out
    except Exception as e:
        out = {
            **state,
            "last_step": "run",
            "last_error": str(e),
            "message": "run failed",
            "repair_mode": True,
            "repair_origin_stage": "crash",
            "repair_error_kind": "run_exception",
            "repair_error_code": "run_exception",
            "repair_signature": "",
            "repair_stdout_tail": "",
            "repair_stderr_tail": "",
            "repair_recent_attempts": list(state.get("repair_recent_attempts") or []),
        }
        _wf_log(cast(dict[str, Any], out), f"<- run err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


def _max_cov_from_run_details(run_details: list[dict[str, Any]]) -> int:
    covs: list[int] = []
    for detail in run_details or []:
        try:
            covs.append(int(detail.get("final_cov") or 0))
        except Exception:
            continue
    return max(covs) if covs else 0


def _node_coverage_analysis(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    state, stop_now = _enter_step(state, "coverage-analysis")
    if stop_now:
        return state
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> coverage-analysis")
    try:
        max_rounds = max(
            0,
            int(
                state.get("coverage_loop_max_rounds")
                if state.get("coverage_loop_max_rounds") is not None
                else 0
            ),
        )
        current_round = max(0, int(state.get("coverage_loop_round") or 0))
        unlimited_rounds = max_rounds == 0
        run_details = list(state.get("run_details") or [])
        history = list(state.get("coverage_history") or [])
        current_cov = _max_cov_from_run_details(run_details)
        current_ft = 0
        current_target_name = ""
        current_target_api = str(state.get("coverage_target_api") or "")
        if run_details:
            try:
                current_ft = max(int(detail.get("final_ft") or 0) for detail in run_details)
            except Exception:
                current_ft = 0
            current_target_name = current_target_api or str(run_details[0].get("fuzzer") or "")
        plateau_detected = any(bool(detail.get("plateau_detected")) for detail in run_details)
        plateau_idle_seconds = max(int(detail.get("plateau_idle_seconds") or 0) for detail in run_details) if run_details else 0
        prev_cov = max(0, int(state.get("coverage_last_max_cov") or 0))
        prev_ft = max(0, int(state.get("coverage_last_ft") or 0))
        prev_plateau_streak = max(0, int(state.get("coverage_plateau_streak") or 0))
        current_seed_profile = str(state.get("coverage_seed_profile") or "")
        current_depth_score = int(state.get("coverage_target_depth_score") or 0)
        current_depth_class = str(state.get("coverage_target_depth_class") or "")
        current_selection_bias_reason = str(state.get("coverage_selection_bias_reason") or "")
        seed_quality = dict(state.get("coverage_seed_quality") or {})
        seed_feedback = dict(state.get("coverage_seed_feedback") or _build_seed_feedback(cast(dict[str, Any], state)))
        quality_flags = list(state.get("coverage_quality_flags") or seed_quality.get("quality_flags") or [])
        seed_families_required = list(state.get("coverage_seed_families_required") or [])
        seed_families_covered = list(state.get("coverage_seed_families_covered") or [])
        seed_families_missing = list(state.get("coverage_seed_families_missing") or [])
        if not current_seed_profile:
            for detail in run_details:
                profile = str(detail.get("seed_profile") or "")
                if profile:
                    current_seed_profile = profile
                    break
        total_execs_per_sec = 0
        try:
            total_execs_per_sec = max(0, sum(int(detail.get("final_execs_per_sec") or 0) for detail in run_details))
        except Exception:
            total_execs_per_sec = 0
        parallel_outer = max(1, int(state.get("run_parallel_outer") or 1))
        parallel_inner = max(1, int(state.get("run_parallel_inner") or 1))
        parallel_cpu_budget = max(1, int(state.get("run_parallel_cpu_budget") or 1))
        parallel_engine = str(state.get("run_parallel_engine") or "single")
        configured_parallel_units = max(1, parallel_outer * parallel_inner)
        parallel_utilization_ratio = float(configured_parallel_units) / float(max(1, parallel_cpu_budget))
        if parallel_utilization_ratio < 0.0:
            parallel_utilization_ratio = 0.0
        if parallel_utilization_ratio > 1.0:
            parallel_utilization_ratio = 1.0
        underutilized_execs_threshold = _coverage_underutilized_execs_threshold()
        resource_underutilized = bool(
            configured_parallel_units < int(parallel_cpu_budget * 0.7)
            and total_execs_per_sec < underutilized_execs_threshold
        )
        strategy_mismatch = bool(
            plateau_detected and total_execs_per_sec > 0 and current_cov <= prev_cov and current_ft <= prev_ft
        )
        if resource_underutilized:
            parallel_diagnosis_code = "resource_underutilized"
            parallel_diagnosis = (
                "exec/s is low while configured parallel units are below cpu budget; "
                "increase outer or inner workers"
            )
        elif strategy_mismatch:
            parallel_diagnosis_code = "strategy_mismatch"
            parallel_diagnosis = (
                "exec/s is healthy but coverage/features are stalled; "
                "reduce parallelism and prioritize target/seed strategy changes"
            )
        else:
            parallel_diagnosis_code = "balanced"
            parallel_diagnosis = "parallelism looks balanced for current coverage signal"
        plateau_no_gain = plateau_detected and current_cov <= prev_cov and current_ft <= prev_ft
        plateau_streak = (prev_plateau_streak + 1) if plateau_no_gain else (1 if plateau_detected else 0)
        requested_replan = bool(
            plateau_no_gain
            and plateau_streak >= 2
            and bool(current_seed_profile)
        )
        replan_reason = ""
        improve_mode = ""
        can_in_place = unlimited_rounds or (current_round < max_rounds)
        can_replan = unlimited_rounds or ((current_round + 1) < max_rounds)
        round_budget_exhausted = False
        stop_reason = ""
        run_error_kind = str(state.get("run_error_kind") or "").strip().lower()
        cold_start_failure = bool(seed_feedback.get("cold_start_failure") or False)
        merge_retained_ratio = float(seed_feedback.get("merge_retained_ratio_files") or 1.0)
        merge_retained_low = bool(merge_retained_ratio > 0.0 and merge_retained_ratio < 0.35)
        seed_quality_issue = bool(
            any(
                flag in quality_flags
                for flag in {
                    "low_retention",
                    "low_early_yield",
                    "high_homogeneity",
                    "missing_required_families",
                    "repo_examples_missing",
                    "seed_family_undercovered",
                    "seed_noise_high",
                    "missing_execution_targets",
                }
            )
            or cold_start_failure
            or merge_retained_low
        )
        quality_degraded = bool(
            seed_quality_issue
            or list(state.get("coverage_missing_execution_targets") or [])
            or (requested_replan and plateau_no_gain)
        )
        quality_oracle = "quality_degraded" if quality_degraded else "ok"
        if seed_quality_issue:
            coverage_bottleneck_kind = "seed_limited"
            if cold_start_failure:
                coverage_bottleneck_reason = "cold_start_failure"
            elif merge_retained_low:
                coverage_bottleneck_reason = "merge_retained_low"
            elif seed_families_missing:
                coverage_bottleneck_reason = "missing_seed_families"
            else:
                coverage_bottleneck_reason = "seed_quality_flags"
        elif requested_replan or (plateau_no_gain and str(current_depth_class or "").lower() == "shallow"):
            coverage_bottleneck_kind = "target_limited"
            coverage_bottleneck_reason = "target_plateau_or_shallow_depth"
        elif plateau_no_gain:
            coverage_bottleneck_kind = "harness_limited"
            coverage_bottleneck_reason = "plateau_without_seed_or_target_signal"
        else:
            coverage_bottleneck_kind = "none"
            coverage_bottleneck_reason = ""
        auto_stop_policy = _auto_stop_policy()
        base_should_improve = (
            (not bool(state.get("crash_found")))
            and (not bool(state.get("failed")))
            and (
                (not run_error_kind)
                or (run_error_kind in _RECOVERABLE_RUN_ERROR_KINDS)
            )
        )
        should_improve = False
        replan_required = False
        if base_should_improve:
            if seed_quality_issue and can_in_place:
                should_improve = True
                improve_mode = "in_place"
                if cold_start_failure:
                    replan_reason = "seed_cold_start_failure"
                elif merge_retained_low:
                    replan_reason = "seed_merge_retained_low"
                else:
                    replan_reason = "seed_quality_issue"
            elif requested_replan:
                if can_replan:
                    should_improve = True
                    replan_required = True
                    improve_mode = "replan"
                    replan_reason = "prefer_deeper_target" if current_depth_class == "shallow" else "stalled_current_target"
                else:
                    round_budget_exhausted = True
                    stop_reason = "coverage_loop_budget_exhausted"
            elif can_in_place:
                should_improve = True
                improve_mode = "in_place"
            else:
                round_budget_exhausted = True
                stop_reason = "coverage_loop_budget_exhausted"

        next_round = current_round + (1 if should_improve else 0)
        reason = "skip coverage loop"
        if should_improve:
            if plateau_detected:
                round_budget_text = "unlimited" if unlimited_rounds else str(max_rounds)
                reason = (
                    f"coverage plateau detected; mode={improve_mode}; round={next_round}/{round_budget_text}, "
                    f"max_cov={current_cov}, prev_cov={prev_cov}, max_ft={current_ft}, prev_ft={prev_ft}, "
                    f"plateau_streak={plateau_streak}, idle_no_growth={plateau_idle_seconds}s"
                )
            else:
                round_budget_text = "unlimited" if unlimited_rounds else str(max_rounds)
                reason = (
                    f"mode=in_place; round={next_round}/{round_budget_text}, max_cov={current_cov}, prev_cov={prev_cov}, "
                    f"max_ft={current_ft}, prev_ft={prev_ft}"
                )
            if seed_quality_issue:
                reason += f"; seed_quality_flags={','.join(quality_flags) or 'none'}"
            if cold_start_failure:
                reason += "; cold_start_failure=1"
            if merge_retained_low:
                reason += f"; merge_retained_ratio_files={merge_retained_ratio:.2f}"
            if parallel_diagnosis_code != "balanced":
                reason += f"; parallel_diagnosis={parallel_diagnosis_code}"
            if coverage_bottleneck_kind != "none":
                reason += f"; bottleneck={coverage_bottleneck_kind}:{coverage_bottleneck_reason}"
        elif round_budget_exhausted:
            if requested_replan:
                reason = (
                    f"coverage plateau detected but replan budget exhausted; "
                    f"round={current_round}/{max_rounds if not unlimited_rounds else 'unlimited'}, max_cov={current_cov}, max_ft={current_ft}, "
                    f"plateau_streak={plateau_streak}"
                )
            else:
                reason = (
                    f"coverage loop budget exhausted; round={current_round}/{max_rounds if not unlimited_rounds else 'unlimited'}, "
                    f"max_cov={current_cov}, max_ft={current_ft}"
                )
        history.append(
            {
                "index": len(history) + 1,
                "round": next_round if should_improve else current_round,
                "max_rounds": max_rounds,
                "max_cov": current_cov,
                "max_ft": current_ft,
                "prev_cov": prev_cov,
                "prev_ft": prev_ft,
                "plateau_detected": plateau_detected,
                "plateau_idle_seconds": plateau_idle_seconds,
                "plateau_streak": plateau_streak,
                "seed_profile": current_seed_profile,
                "target_name": current_target_name,
                "target_api": current_target_api or current_target_name,
                "target_depth_score": current_depth_score,
                "target_depth_class": current_depth_class,
                "selection_bias_reason": current_selection_bias_reason,
                "replan_required": replan_required,
                "replan_effective": bool(state.get("coverage_replan_effective") or False),
                "replan_reason": replan_reason or str(state.get("coverage_replan_reason") or ""),
                "improve_mode": improve_mode,
                "round_budget_exhausted": round_budget_exhausted,
                "stop_reason": stop_reason,
                "corpus_sources": list(state.get("coverage_corpus_sources") or []),
                "seed_counts": dict(state.get("coverage_seed_counts") or {}),
                "seed_quality": seed_quality,
                "seed_families_required": seed_families_required,
                "seed_families_covered": seed_families_covered,
                "seed_families_missing": seed_families_missing,
                "quality_flags": quality_flags,
                "quality_oracle": quality_oracle,
                "coverage_bottleneck_kind": coverage_bottleneck_kind,
                "coverage_bottleneck_reason": coverage_bottleneck_reason,
                "parallel_diagnosis_code": parallel_diagnosis_code,
                "parallel_diagnosis": parallel_diagnosis,
                "parallel_engine": parallel_engine,
                "parallel_outer": parallel_outer,
                "parallel_inner": parallel_inner,
                "parallel_cpu_budget": parallel_cpu_budget,
                "parallel_utilization_ratio": parallel_utilization_ratio,
                "total_execs_per_sec": total_execs_per_sec,
                "underutilized_execs_threshold": underutilized_execs_threshold,
                "repo_examples_filtered": bool(state.get("coverage_repo_examples_filtered") or False),
                "repo_examples_rejected_count": int(state.get("coverage_repo_examples_rejected_count") or 0),
                "repo_examples_accepted_count": int(state.get("coverage_repo_examples_accepted_count") or 0),
                "crash_found": bool(state.get("crash_found")),
                "run_error_kind": str(state.get("run_error_kind") or ""),
                "should_improve": should_improve,
                "ts": int(time.time()),
            }
        )
        out = {
            **state,
            "last_step": "coverage-analysis",
            "last_error": "",
            "coverage_loop_max_rounds": max_rounds,
            "coverage_loop_round": next_round if should_improve else current_round,
            "coverage_should_improve": should_improve,
            "coverage_improve_reason": reason,
            "coverage_history": history,
            "coverage_target_name": current_target_name or str(state.get("coverage_target_name") or ""),
            "coverage_target_api": current_target_api or current_target_name or str(state.get("coverage_target_api") or ""),
            "coverage_seed_profile": current_seed_profile,
            "coverage_seed_quality": seed_quality,
            "coverage_seed_families_required": seed_families_required,
            "coverage_seed_families_covered": seed_families_covered,
            "coverage_seed_families_missing": seed_families_missing,
            "coverage_quality_flags": quality_flags,
            "coverage_quality_oracle": quality_oracle,
            "coverage_bottleneck_kind": coverage_bottleneck_kind,
            "coverage_bottleneck_reason": coverage_bottleneck_reason,
            "coverage_parallel_diagnosis_code": parallel_diagnosis_code,
            "coverage_parallel_diagnosis": parallel_diagnosis,
            "coverage_parallel_engine": parallel_engine,
            "coverage_parallel_outer": parallel_outer,
            "coverage_parallel_inner": parallel_inner,
            "coverage_parallel_cpu_budget": parallel_cpu_budget,
            "coverage_parallel_utilization_ratio": parallel_utilization_ratio,
            "coverage_total_execs_per_sec": total_execs_per_sec,
            "coverage_underutilized_execs_threshold": underutilized_execs_threshold,
            "coverage_target_depth_score": current_depth_score,
            "coverage_target_depth_class": current_depth_class,
            "coverage_selection_bias_reason": current_selection_bias_reason,
            "coverage_plateau_streak": plateau_streak,
            "coverage_last_max_cov": current_cov,
            "coverage_last_ft": current_ft,
            "coverage_replan_required": replan_required,
            "coverage_replan_reason": replan_reason or str(state.get("coverage_replan_reason") or ""),
            "coverage_improve_mode": improve_mode,
            "coverage_round_budget_exhausted": round_budget_exhausted,
            "coverage_stop_reason": stop_reason,
            "coverage_repo_examples_filtered": bool(state.get("coverage_repo_examples_filtered") or False),
            "coverage_repo_examples_rejected_count": int(state.get("coverage_repo_examples_rejected_count") or 0),
            "coverage_repo_examples_accepted_count": int(state.get("coverage_repo_examples_accepted_count") or 0),
            "message": "coverage analysis done",
            "auto_stop_policy": auto_stop_policy,
            "auto_stop_blocked_reason": str(state.get("auto_stop_blocked_reason") or ""),
            "continuous_loop_count": int(state.get("continuous_loop_count") or 0),
        }
        if auto_stop_policy == "hard_fail_only" and (not bool(out.get("failed"))) and not str(out.get("last_error") or "").strip() and not bool(should_improve):
            out["auto_stop_blocked_reason"] = "coverage_no_improve"
            out["continuous_loop_count"] = int(out.get("continuous_loop_count") or 0) + 1
        out["coverage_seed_feedback"] = _build_seed_feedback(cast(dict[str, Any], out))
        out["coverage_harness_feedback"] = _build_harness_feedback(cast(dict[str, Any], out))

        # Collect source coverage report (llvm-cov) for improve feedback
        source_report: dict[str, Any] | None = None
        try:
            fuzz_out = gen.repo_root / "fuzz" / "out"
            if fuzz_out.is_dir():
                bins = sorted(fuzz_out.glob("*"))
                for b in bins:
                    if b.is_file() and os.access(str(b), os.X_OK):
                        source_report = gen.collect_source_coverage(b)
                        if source_report:
                            break
        except Exception:
            pass
        if source_report:
            out["coverage_source_report"] = source_report
            out["coverage_uncovered_functions"] = list(source_report.get("uncovered_functions") or [])

        # Track exhausted targets for coverage-guided replan
        exhausted = list(state.get("coverage_exhausted_targets") or [])
        if plateau_streak >= 2 and current_target_api and current_target_api not in exhausted:
            exhausted.append(current_target_api)
        out["coverage_exhausted_targets"] = exhausted

        # Build coverage feedback for plan stage (replan context)
        if replan_required and history:
            feedback_lines = [f"Previously exhausted targets (avoid re-selecting):"]
            for t in exhausted:
                feedback_lines.append(f"  - {t}")
            for h_entry in history[-3:]:
                feedback_lines.append(
                    f"  Round {h_entry.get('round')}: target={h_entry.get('target_api')}, "
                    f"cov={h_entry.get('max_cov')}, ft={h_entry.get('max_ft')}, "
                    f"plateau={h_entry.get('plateau_detected')}"
                )
            out["coverage_feedback_for_plan"] = "\n".join(feedback_lines)

        _wf_log(
            cast(dict[str, Any], out),
            f"<- coverage-analysis improve={int(should_improve)} {reason} dt={_fmt_dt(time.perf_counter()-t0)}",
        )
        _emit_fuzz_metrics(cast(dict[str, Any], out))
        return out
    except Exception as e:
        out = {**state, "last_step": "coverage-analysis", "last_error": str(e), "message": "coverage analysis failed"}
        _wf_log(cast(dict[str, Any], out), f"<- coverage-analysis err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


def _node_improve_harness(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "improve-harness")
    if stop_now:
        return state
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> improve-harness")
    try:
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
        seed_families_required = list(state.get("coverage_seed_families_required") or [])
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
            exhausted_targets = list(state.get("coverage_exhausted_targets") or [])
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
                f"- Required seed families: {', '.join(seed_families_required) if seed_families_required else 'none'}\n"
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

        opencode_applied = False
        if improve_mode == "in_place":
            if not _has_codex_key():
                out = {
                    **state,
                    "last_step": "improve-harness",
                    "last_error": "Missing OPENAI_API_KEY for improve-harness in-place repair",
                    "message": "improve-harness failed",
                }
                _wf_log(cast(dict[str, Any], out), f"<- improve-harness err=missing-key dt={_fmt_dt(time.perf_counter()-t0)}")
                return out
            prompt = _render_opencode_prompt("improve_harness_in_place_with_hint", hint=hint)
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
                    "seed_families_required": seed_families_required,
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
        _wf_log(cast(dict[str, Any], out), f"<- improve-harness err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


def _node_fix_crash(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "fix_crash")
    if stop_now:
        return state

    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> fix_crash")

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
        prompt = _render_opencode_prompt("fix_crash_harness_error")
    else:
        prompt = _render_opencode_prompt("fix_crash_upstream_bug")

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
        )
        patch_path = repo_root / "fix.patch"
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
        _wf_log(cast(dict[str, Any], out), f"<- fix_crash err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


def _normalize_crash_triage_label(raw: str) -> str:
    val = str(raw or "").strip().lower()
    if val in {"harness_bug", "upstream_bug", "inconclusive"}:
        return val
    if val in {"harness", "harness-error", "harness_error"}:
        return "harness_bug"
    if val in {"upstream", "upstream-error", "upstream_error", "library_bug"}:
        return "upstream_bug"
    return "inconclusive"


def _normalize_crash_analysis_verdict(raw: str) -> str:
    val = str(raw or "").strip().lower()
    if val in {"false_positive", "real_bug", "unknown"}:
        return val
    if val in {"false-positive", "harness_false_positive", "falsepositive"}:
        return "false_positive"
    if val in {"upstream_bug", "realbug", "true_positive", "upstream"}:
        return "real_bug"
    return "unknown"

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

    prompt = _render_opencode_prompt("crash_triage_with_hint", hint=str(state.get("codex_hint") or ""))
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
        )
        parsed: dict[str, Any] = {}
        if triage_json_path.is_file():
            try:
                parsed = json.loads(triage_json_path.read_text(encoding="utf-8", errors="replace"))
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
        "constraint_memory_count": constraint_count,
        "constraint_memory_path": constraint_path,
        "message": f"crash triage classified as {label}",
    }
    _wf_log(cast(dict[str, Any], out), f"<- crash-triage label={label} conf={confidence:.2f} dt={_fmt_dt(time.perf_counter()-t0)}")
    return out


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

    prompt = _render_opencode_prompt("crash_analysis_with_hint", hint=str(state.get("codex_hint") or ""))
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
        )
        parsed_doc: dict[str, Any] = {}
        if analysis_json_path.is_file():
            try:
                loaded = json.loads(analysis_json_path.read_text(encoding="utf-8", errors="replace"))
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
        "message": "crash-analysis false_positive" if false_positive else "crash-analysis stop",
    }
    _wf_log(
        cast(dict[str, Any], out),
        f"<- crash-analysis verdict={verdict} action={recommended_action} dt={_fmt_dt(time.perf_counter()-t0)}",
    )
    return out


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

    prompt = _render_opencode_prompt("fix_harness_after_run")
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
            stage_skill="fix_crash_harness_error",
            timeout=_remaining_time_budget_sec(state),
            max_attempts=1,
            max_cli_retries=_opencode_cli_retries(),
        )
        patch_path = repo_root / "fix.patch"
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
        _wf_log(cast(dict[str, Any], out), f"<- fix-harness err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


def _node_re_build(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "re-build")
    if stop_now:
        return state

    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> re-build")
    repo_root = gen.repo_root
    report_md = repo_root / "re_build_report.md"
    report_json = repo_root / "re_build_report.json"

    if not bool(state.get("crash_found")):
        out = {
            **state,
            "last_step": "re-build",
            "last_error": "",
            "re_build_done": True,
            "re_build_ok": False,
            "re_build_rc": 0,
            "message": "re-build skipped (no crash found)",
            "re_build_report_path": str(report_md),
            "re_build_json_path": str(report_json),
        }
        _wf_log(cast(dict[str, Any], out), f"<- re-build skip=no-crash dt={_fmt_dt(time.perf_counter()-t0)}")
        return out

    repo_url = str(state.get("repo_url") or "").strip()
    now_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    payload: dict[str, Any] = {
        "timestamp": now_ts,
        "repo_url": repo_url,
        "fuzzer": str(state.get("last_fuzzer") or ""),
        "artifact": str(state.get("last_crash_artifact") or ""),
        "clone_repo_root": "",
        "clone_ok": False,
        "clone_rc": 1,
        "build_ok": False,
        "build_rc": 1,
        "error": "",
        "stdout_tail": "",
        "stderr_tail": "",
    }

    try:
        if not repo_url:
            raise HarnessGeneratorError("missing repo_url for re-build")

        repro_workspace = repo_root / ".repro_crash"
        repro_workspace.mkdir(parents=True, exist_ok=True)
        clone_root = repro_workspace / "workdir"
        if clone_root.exists():
            shutil.rmtree(clone_root, ignore_errors=True)

        # Reuse the same clone path as init so mirrors/proxy/retry behavior stays consistent.
        rem = _remaining_time_budget_sec(state, min_timeout=0)
        if rem <= 0:
            raise HarnessGeneratorError("re-build clone skipped: no remaining workflow budget")
        try:
            cloned_root = gen._clone_repo(RepoSpec(url=repo_url, workdir=clone_root))
        except Exception as clone_err:
            payload["clone_rc"] = 1
            payload["clone_ok"] = False
            payload["clone_repo_root"] = str(clone_root)
            payload["stderr_tail"] = str(clone_err)[-4000:]
            raise HarnessGeneratorError(f"re-build clone failed via init clone logic: {clone_err}")

        payload["clone_rc"] = 0
        payload["clone_ok"] = True
        payload["clone_repo_root"] = str(cloned_root)

        source_fuzz = repo_root / "fuzz"
        if not source_fuzz.is_dir():
            raise HarnessGeneratorError(f"run fuzz directory missing: {source_fuzz}")
        dest_fuzz = clone_root / "fuzz"
        if dest_fuzz.exists():
            shutil.rmtree(dest_fuzz, ignore_errors=True)
        shutil.copytree(
            source_fuzz,
            dest_fuzz,
            ignore=shutil.ignore_patterns(
                "build-work",   # CMake build dir (contains CMakeCache.txt with hardcoded paths)
                "CMakeFiles",   # CMake intermediate files
                "out",          # fuzzer output (corpus/crashes); re-run regenerates
                "__pycache__",
                "*.o",
                "*.a",
            ),
        )

        python_runner = "python3"
        try:
            python_runner = str(gen._python_runner() or "python3")
        except Exception:
            python_runner = "python3"

        build_cmd: list[str]
        build_cwd: Path
        if (clone_root / "fuzz" / "build.py").is_file():
            build_cmd = [python_runner, "build.py"]
            build_cwd = clone_root / "fuzz"
        elif (clone_root / "fuzz" / "build.sh").is_file():
            build_cmd = ["bash", "build.sh"]
            build_cwd = clone_root / "fuzz"
        else:
            raise HarnessGeneratorError("no fuzz/build.py or fuzz/build.sh found in cloned repo")

        rem = _remaining_time_budget_sec(state, min_timeout=15)
        build_timeout = max(30, min(rem, 600))
        build_env = os.environ.copy()
        if hasattr(gen, "_compose_vcpkg_runtime_env"):
            try:
                build_env = gen._compose_vcpkg_runtime_env(build_env, repo_root=clone_root)  # type: ignore[attr-defined]
            except Exception:
                pass
        if hasattr(gen, "_run_cmd"):
            rc, out, err = gen._run_cmd(  # type: ignore[attr-defined]
                build_cmd,
                cwd=build_cwd,
                env=build_env,
                timeout=build_timeout,
                idle_timeout=0,
            )
            payload["build_rc"] = int(rc)
            payload["build_ok"] = int(rc) == 0
            if int(rc) != 0:
                payload["stdout_tail"] = (out or "")[-4000:]
                payload["stderr_tail"] = (err or "")[-4000:]
                raise HarnessGeneratorError(f"re-build build failed (rc={rc})")
        else:
            build = subprocess.run(
                build_cmd,
                cwd=build_cwd,
                capture_output=True,
                text=True,
                timeout=build_timeout,
                env=build_env,
            )
            payload["build_rc"] = int(build.returncode)
            payload["build_ok"] = build.returncode == 0
            if build.returncode != 0:
                payload["stdout_tail"] = (build.stdout or "")[-4000:]
                payload["stderr_tail"] = (build.stderr or "")[-4000:]
                raise HarnessGeneratorError(f"re-build build failed (rc={build.returncode})")
    except Exception as e:
        payload["error"] = str(e)

    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_lines = [
        "# Re-Build Report",
        "",
        f"- timestamp: {payload['timestamp']}",
        f"- repo_url: {payload['repo_url']}",
        f"- clone_ok: {payload['clone_ok']} (rc={payload['clone_rc']})",
        f"- build_ok: {payload['build_ok']} (rc={payload['build_rc']})",
        "",
    ]
    if payload["error"]:
        md_lines.extend(["## Error", "", str(payload["error"]), ""])
    if payload["stdout_tail"]:
        md_lines.extend(["## STDOUT (tail)", "", "```text", str(payload["stdout_tail"]), "```", ""])
    if payload["stderr_tail"]:
        md_lines.extend(["## STDERR (tail)", "", "```text", str(payload["stderr_tail"]), "```", ""])
    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    re_build_ok = bool(payload["build_ok"])
    restart_reason = ""
    restart_error = ""
    restart_report = ""
    restart_stage = ""
    restart_count = int(state.get("restart_to_plan_count") or 0)
    if not re_build_ok:
        restart_reason = "re_build_failed"
        restart_stage = "re-build"
        restart_error = str(payload.get("error") or payload.get("stderr_tail") or payload.get("stdout_tail") or "")[:4096]
        restart_report = str(report_md)
        restart_count += 1
    restart_limit = _re_restart_limit()
    restart_exceeded = (not re_build_ok) and restart_count > restart_limit
    if re_build_ok:
        _write_repro_context(
            repo_root,
            repo_url=repo_url,
            last_fuzzer=str(state.get("last_fuzzer") or ""),
            last_crash_artifact=str(state.get("last_crash_artifact") or ""),
            crash_signature=str(state.get("crash_signature") or ""),
            re_workspace_root=str(payload.get("clone_repo_root") or ""),
        )

    out = {
        **state,
        "last_step": "re-build",
        "last_error": "" if re_build_ok else restart_error,
        "re_build_done": True,
        "re_build_ok": re_build_ok,
        "re_build_rc": int(payload["build_rc"]),
        "re_build_report_path": str(report_md),
        "re_build_json_path": str(report_json),
        "re_workspace_root": str(payload.get("clone_repo_root") or ""),
        "restart_to_plan": not re_build_ok,
        "restart_to_plan_reason": restart_reason,
        "restart_to_plan_stage": restart_stage,
        "restart_to_plan_error_text": restart_error,
        "restart_to_plan_report_path": restart_report,
        "restart_to_plan_count": restart_count,
        "failed": bool(state.get("failed")) or restart_exceeded,
        "run_terminal_reason": "re_restart_limit_exceeded" if restart_exceeded else str(state.get("run_terminal_reason") or ""),
        "message": "re-build validated" if re_build_ok else "re-build failed",
        "repair_mode": (not re_build_ok),
        "repair_origin_stage": "crash" if not re_build_ok else "",
        "repair_error_kind": "re_build_failed" if not re_build_ok else "",
        "repair_error_code": restart_reason if not re_build_ok else "",
        "repair_signature": str(state.get("crash_signature") or "")[:12] if not re_build_ok else "",
        "repair_stdout_tail": str(payload.get("stdout_tail") or "") if not re_build_ok else "",
        "repair_stderr_tail": str(payload.get("stderr_tail") or "") if not re_build_ok else "",
        "repair_attempt_index": (int(state.get("repair_attempt_index") or 0) + 1) if not re_build_ok else 0,
        "repair_strategy_force_change": False,
        "repair_error_digest": (
            {
                "error_code": restart_reason,
                "error_kind": "re_build_failed",
                "signature": str(state.get("crash_signature") or "")[:12],
                "failing_files": [],
                "symbols": [],
                "first_seen": int(time.time()),
                "latest_seen": int(time.time()),
                "top_trace": _extract_repair_top_trace(
                    restart_error,
                    str(payload.get("stdout_tail") or ""),
                    str(payload.get("stderr_tail") or ""),
                ),
            }
            if not re_build_ok
            else {}
        ),
        "repair_recent_attempts": (
            (list(state.get("repair_recent_attempts") or []) + [{
                "step": "re-build",
                "origin": "crash",
                "error_kind": "re_build_failed",
                "error_code": restart_reason,
                "signature": str(state.get("crash_signature") or "")[:12],
                "attempt_index": int(state.get("repair_attempt_index") or 0) + 1,
                "message": restart_error[:512],
            }])[-5:]
            if not re_build_ok
            else []
        ),
    }
    if restart_exceeded:
        out["last_error"] = f"re failed and restart-to-plan limit exceeded ({restart_limit})"
    _wf_log(
        cast(dict[str, Any], out),
        (
            "<- re-build "
            f"ok={re_build_ok} clone_rc={payload['clone_rc']} build_rc={payload['build_rc']} "
            f"dt={_fmt_dt(time.perf_counter()-t0)}"
        ),
    )
    return out


def _node_re_run(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "re-run")
    if stop_now:
        return state
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> re-run")
    _wf_log(cast(dict[str, Any], state), "re-run: reusing run-stage corpus from fuzz/corpus; no new seeds will be generated")

    repo_root = gen.repo_root
    report_md = repo_root / "re_run_report.md"
    report_json = repo_root / "re_run_report.json"
    last_fuzzer = str(state.get("last_fuzzer") or "").strip()
    last_artifact = str(state.get("last_crash_artifact") or "").strip()
    workspace_root = str(state.get("re_workspace_root") or "").strip() or str((repo_root / ".repro_crash" / "workdir"))
    artifact_path = Path(last_artifact) if last_artifact else None

    def _recover_artifact_path() -> tuple[str, Path | None]:
        recovered = last_artifact
        if not recovered:
            repro_doc = _read_repro_context(repo_root)
            if isinstance(repro_doc, dict):
                recovered = str(repro_doc.get("last_crash_artifact") or "").strip()
        if not recovered and (repo_root / "re_build_report.json").is_file():
            try:
                re_build_doc = json.loads((repo_root / "re_build_report.json").read_text(encoding="utf-8", errors="replace"))
                if isinstance(re_build_doc, dict):
                    recovered = str(re_build_doc.get("artifact") or "").strip()
            except Exception:
                pass
        if not recovered and (repo_root / "run_summary.json").is_file():
            try:
                summary_doc = json.loads((repo_root / "run_summary.json").read_text(encoding="utf-8", errors="replace"))
                if isinstance(summary_doc, dict):
                    recovered = str(summary_doc.get("last_crash_artifact") or "").strip()
            except Exception:
                pass
        if not recovered:
            artifacts_dir = repo_root / "fuzz" / "out" / "artifacts"
            if artifacts_dir.is_dir():
                candidates: list[Path] = []
                for p in artifacts_dir.iterdir():
                    if not p.is_file():
                        continue
                    name = p.name.lower()
                    if name.startswith("crash-") or "crash" in name:
                        candidates.append(p)
                if not candidates:
                    for p in artifacts_dir.iterdir():
                        if p.is_file():
                            candidates.append(p)
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    recovered = str(candidates[0])
        return recovered, (Path(recovered) if recovered else None)
    def _rebuild_workspace_from_init_clone() -> Path:
        repo_url = str(state.get("repo_url") or "").strip()
        if not repo_url:
            raise HarnessGeneratorError("missing repo_url for re-run workspace rebuild")
        repro_workspace = repo_root / ".repro_crash"
        repro_workspace.mkdir(parents=True, exist_ok=True)
        clone_root = repro_workspace / "workdir"
        if clone_root.exists():
            shutil.rmtree(clone_root, ignore_errors=True)

        rem = _remaining_time_budget_sec(state, min_timeout=15)
        if rem <= 0:
            raise HarnessGeneratorError("re-run workspace rebuild skipped: no remaining workflow budget")
        clone_result = gen._clone_repo(RepoSpec(url=repo_url, workdir=clone_root))
        clone_root = Path(clone_result).expanduser().resolve()
        source_fuzz = repo_root / "fuzz"
        if not source_fuzz.is_dir():
            raise HarnessGeneratorError(f"run fuzz directory missing: {source_fuzz}")
        dest_fuzz = clone_root / "fuzz"
        if dest_fuzz.exists():
            shutil.rmtree(dest_fuzz, ignore_errors=True)
        shutil.copytree(
            source_fuzz,
            dest_fuzz,
            ignore=shutil.ignore_patterns(
                "build-work",   # CMake build dir (contains CMakeCache.txt with hardcoded paths)
                "CMakeFiles",   # CMake intermediate files
                "out",          # fuzzer output (corpus/crashes); re-run regenerates
                "__pycache__",
                "*.o",
                "*.a",
            ),
        )

        python_runner = "python3"
        try:
            python_runner = str(gen._python_runner() or "python3")
        except Exception:
            python_runner = "python3"

        build_cmd: list[str]
        build_cwd: Path
        if (clone_root / "fuzz" / "build.py").is_file():
            build_cmd = [python_runner, "build.py"]
            build_cwd = clone_root / "fuzz"
        elif (clone_root / "fuzz" / "build.sh").is_file():
            build_cmd = ["bash", "build.sh"]
            build_cwd = clone_root / "fuzz"
        else:
            raise HarnessGeneratorError("no fuzz/build.py or fuzz/build.sh found in re-run workspace rebuild")

        build_timeout = max(30, min(rem, 600))
        build_env = os.environ.copy()
        if hasattr(gen, "_compose_vcpkg_runtime_env"):
            try:
                build_env = gen._compose_vcpkg_runtime_env(build_env, repo_root=clone_root)  # type: ignore[attr-defined]
            except Exception:
                pass
        if hasattr(gen, "_run_cmd"):
            rc, out, err = gen._run_cmd(  # type: ignore[attr-defined]
                build_cmd,
                cwd=build_cwd,
                env=build_env,
                timeout=build_timeout,
                idle_timeout=0,
            )
            if int(rc) != 0:
                err_tail = ((err or "") + "\n" + (out or ""))[-1200:]
                raise HarnessGeneratorError(f"re-run workspace rebuild build failed (rc={rc}): {err_tail}")
        else:
            build = subprocess.run(
                build_cmd,
                cwd=build_cwd,
                capture_output=True,
                text=True,
                timeout=build_timeout,
                env=build_env,
            )
            if build.returncode != 0:
                err_tail = ((build.stderr or "") + "\n" + (build.stdout or ""))[-1200:]
                raise HarnessGeneratorError(f"re-run workspace rebuild build failed (rc={build.returncode}): {err_tail}")
        return clone_root

    def _guess_fuzzer_from_workspace(workdir: Path) -> str:
        out_dir = workdir / "fuzz" / "out"
        if not out_dir.is_dir():
            return ""
        candidates: list[Path] = []
        for p in out_dir.iterdir():
            if not p.is_file():
                continue
            name = p.name
            if name.startswith("."):
                continue
            if name.startswith(("crash-", "timeout-", "slow-unit-")):
                continue
            if name.endswith((".md", ".json", ".txt", ".log", ".patch", ".py")):
                continue
            if os.access(p, os.X_OK):
                candidates.append(p)
        if len(candidates) == 1:
            return candidates[0].name
        # Prefer common fuzzer naming if multiple binaries are present.
        named = [p for p in candidates if "fuzz" in p.name.lower()]
        if len(named) == 1:
            return named[0].name
        return ""

    now_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    payload: dict[str, Any] = {
        "timestamp": now_ts,
        "fuzzer": last_fuzzer,
        "artifact": last_artifact,
        "workspace_root": workspace_root,
        "reproduce_ok": False,
        "reproduce_rc": 1,
        "error": "",
        "stdout_tail": "",
        "stderr_tail": "",
    }
    try:
        workdir = Path(workspace_root)
        if not last_fuzzer or not last_artifact or not str(state.get("re_workspace_root") or "").strip():
            repro_doc = _read_repro_context(repo_root)
            if isinstance(repro_doc, dict):
                if not last_fuzzer:
                    last_fuzzer = str(repro_doc.get("last_fuzzer") or "").strip()
                    payload["fuzzer"] = last_fuzzer
                if not last_artifact:
                    last_artifact = str(repro_doc.get("last_crash_artifact") or "").strip()
                    payload["artifact"] = last_artifact
                    if last_artifact:
                        artifact_path = Path(last_artifact)
                restored_workspace = str(repro_doc.get("re_workspace_root") or "").strip()
                if restored_workspace and not workdir.is_dir():
                    workspace_root = restored_workspace
                    payload["workspace_root"] = restored_workspace
                    workdir = Path(restored_workspace)
        if not workdir.is_dir():
            _wf_log(cast(dict[str, Any], state), f"re-run: workspace missing, attempting rebuild via init clone logic: {workdir}")
            workdir = _rebuild_workspace_from_init_clone()
            workspace_root = str(workdir)
            payload["workspace_root"] = workspace_root
            _write_repro_context(
                repo_root,
                repo_url=str(state.get("repo_url") or ""),
                re_workspace_root=workspace_root,
            )
        if (not last_fuzzer or not last_artifact) and (repo_root / "re_build_report.json").is_file():
            try:
                re_build_doc = json.loads((repo_root / "re_build_report.json").read_text(encoding="utf-8", errors="replace"))
                if isinstance(re_build_doc, dict):
                    if not last_fuzzer:
                        last_fuzzer = str(re_build_doc.get("fuzzer") or "").strip()
                        payload["fuzzer"] = last_fuzzer
                    if not last_artifact:
                        last_artifact = str(re_build_doc.get("artifact") or "").strip()
                        payload["artifact"] = last_artifact
                        if last_artifact:
                            artifact_path = Path(last_artifact)
            except Exception:
                pass
        if artifact_path is None or not artifact_path.is_file():
            recovered_artifact, recovered_path = _recover_artifact_path()
            if recovered_artifact:
                last_artifact = recovered_artifact
                artifact_path = recovered_path
                payload["artifact"] = recovered_artifact
        if not last_fuzzer:
            # Stage resume can occasionally lose last_fuzzer in state; recover from workspace.
            last_fuzzer = _guess_fuzzer_from_workspace(workdir)
            payload["fuzzer"] = last_fuzzer
        if not last_fuzzer:
            _wf_log(cast(dict[str, Any], state), "re-run: last_fuzzer missing, attempting workspace rebuild before failing")
            workdir = _rebuild_workspace_from_init_clone()
            workspace_root = str(workdir)
            payload["workspace_root"] = workspace_root
            last_fuzzer = _guess_fuzzer_from_workspace(workdir)
            payload["fuzzer"] = last_fuzzer
        if not last_fuzzer:
            raise HarnessGeneratorError("missing last_fuzzer for re-run after workspace rebuild")
        if artifact_path is None or not artifact_path.is_file():
            recovered_artifact, recovered_path = _recover_artifact_path()
            if recovered_artifact:
                last_artifact = recovered_artifact
                artifact_path = recovered_path
                payload["artifact"] = recovered_artifact
        if artifact_path is None or not artifact_path.is_file():
            raise HarnessGeneratorError(f"crash artifact not found: {last_artifact}")
        fuzzer_bin = workdir / "fuzz" / "out" / last_fuzzer
        if not fuzzer_bin.is_file():
            _wf_log(cast(dict[str, Any], state), f"re-run: fuzzer binary missing, attempting workspace rebuild: {fuzzer_bin}")
            workdir = _rebuild_workspace_from_init_clone()
            workspace_root = str(workdir)
            payload["workspace_root"] = workspace_root
            if not last_fuzzer:
                last_fuzzer = _guess_fuzzer_from_workspace(workdir)
                payload["fuzzer"] = last_fuzzer
            fuzzer_bin = workdir / "fuzz" / "out" / last_fuzzer
            if not fuzzer_bin.is_file():
                raise HarnessGeneratorError(f"re-run fuzzer binary not found after workspace rebuild: {fuzzer_bin}")
        rem = _remaining_time_budget_sec(state, min_timeout=15)
        repro_timeout = max(20, min(rem, 180))
        repro_env = os.environ.copy()
        if hasattr(gen, "_compose_vcpkg_runtime_env"):
            try:
                repro_env = gen._compose_vcpkg_runtime_env(repro_env, repo_root=workdir)  # type: ignore[attr-defined]
            except Exception:
                pass
        repro_cmd = [str(fuzzer_bin), "-runs=1", str(artifact_path)]
        if hasattr(gen, "_run_cmd"):
            rc, out, err = gen._run_cmd(  # type: ignore[attr-defined]
                repro_cmd,
                cwd=workdir,
                env=repro_env,
                timeout=repro_timeout,
                idle_timeout=0,
            )
            payload["reproduce_rc"] = int(rc)
            payload["reproduce_ok"] = int(rc) != 0
            payload["stdout_tail"] = (out or "")[-4000:]
            payload["stderr_tail"] = (err or "")[-4000:]
        else:
            repro = subprocess.run(
                repro_cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=repro_timeout,
                env=repro_env,
            )
            payload["reproduce_rc"] = int(repro.returncode)
            payload["reproduce_ok"] = repro.returncode != 0
            payload["stdout_tail"] = (repro.stdout or "")[-4000:]
            payload["stderr_tail"] = (repro.stderr or "")[-4000:]
        _write_repro_context(
            repo_root,
            repo_url=str(state.get("repo_url") or ""),
            last_fuzzer=last_fuzzer,
            last_crash_artifact=last_artifact,
            crash_signature=str(state.get("crash_signature") or ""),
            re_workspace_root=workspace_root,
        )
    except Exception as e:
        payload["error"] = str(e)

    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_lines = [
        "# Re-Run Report",
        "",
        f"- timestamp: {payload['timestamp']}",
        f"- fuzzer: {payload['fuzzer']}",
        f"- artifact: {payload['artifact']}",
        f"- workspace_root: {payload['workspace_root']}",
        f"- reproduce_ok: {payload['reproduce_ok']} (rc={payload['reproduce_rc']})",
        "",
    ]
    if payload["error"]:
        md_lines.extend(["## Error", "", str(payload["error"]), ""])
    if payload["stdout_tail"]:
        md_lines.extend(["## STDOUT (tail)", "", "```text", str(payload["stdout_tail"]), "```", ""])
    if payload["stderr_tail"]:
        md_lines.extend(["## STDERR (tail)", "", "```text", str(payload["stderr_tail"]), "```", ""])
    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    re_run_ok = bool(payload["reproduce_ok"])
    restart_reason = ""
    restart_error = ""
    restart_report = ""
    restart_stage = ""
    restart_count = int(state.get("restart_to_plan_count") or 0)
    if not re_run_ok:
        restart_reason = "re_run_failed"
        restart_stage = "re-run"
        restart_error = str(payload.get("error") or payload.get("stderr_tail") or payload.get("stdout_tail") or "")[:4096]
        restart_report = str(report_md)
        restart_count += 1
    restart_limit = _re_restart_limit()
    restart_exceeded = (not re_run_ok) and restart_count > restart_limit

    out = {
        **state,
        "last_step": "re-run",
        "last_error": "" if re_run_ok else restart_error,
        "re_run_done": True,
        "re_run_ok": re_run_ok,
        "re_run_rc": int(payload["reproduce_rc"]),
        "re_run_report_path": str(report_md),
        "re_run_json_path": str(report_json),
        "crash_repro_done": True,
        "crash_repro_ok": re_run_ok,
        "crash_repro_rc": int(payload["reproduce_rc"]),
        "crash_repro_report_path": str(report_md),
        "crash_repro_json_path": str(report_json),
        "restart_to_plan": not re_run_ok,
        "restart_to_plan_reason": restart_reason,
        "restart_to_plan_stage": restart_stage,
        "restart_to_plan_error_text": restart_error,
        "restart_to_plan_report_path": restart_report,
        "restart_to_plan_count": restart_count,
        "failed": bool(state.get("failed")) or restart_exceeded,
        "run_terminal_reason": "re_restart_limit_exceeded" if restart_exceeded else str(state.get("run_terminal_reason") or ""),
        "message": "re-run validated" if re_run_ok else "re-run failed",
        "repair_mode": (not re_run_ok),
        "repair_origin_stage": "crash" if not re_run_ok else "",
        "repair_error_kind": "re_run_failed" if not re_run_ok else "",
        "repair_error_code": restart_reason if not re_run_ok else "",
        "repair_signature": str(state.get("crash_signature") or "")[:12] if not re_run_ok else "",
        "repair_stdout_tail": str(payload.get("stdout_tail") or "") if not re_run_ok else "",
        "repair_stderr_tail": str(payload.get("stderr_tail") or "") if not re_run_ok else "",
        "repair_attempt_index": (int(state.get("repair_attempt_index") or 0) + 1) if not re_run_ok else 0,
        "repair_strategy_force_change": False,
        "repair_error_digest": (
            {
                "error_code": restart_reason,
                "error_kind": "re_run_failed",
                "signature": str(state.get("crash_signature") or "")[:12],
                "failing_files": [],
                "symbols": [],
                "first_seen": int(time.time()),
                "latest_seen": int(time.time()),
                "top_trace": _extract_repair_top_trace(
                    restart_error,
                    str(payload.get("stdout_tail") or ""),
                    str(payload.get("stderr_tail") or ""),
                ),
            }
            if not re_run_ok
            else {}
        ),
        "repair_recent_attempts": (
            (list(state.get("repair_recent_attempts") or []) + [{
                "step": "re-run",
                "origin": "crash",
                "error_kind": "re_run_failed",
                "error_code": restart_reason,
                "signature": str(state.get("crash_signature") or "")[:12],
                "attempt_index": int(state.get("repair_attempt_index") or 0) + 1,
                "message": restart_error[:512],
            }])[-5:]
            if not re_run_ok
            else []
        ),
    }
    if restart_exceeded:
        out["last_error"] = f"re failed and restart-to-plan limit exceeded ({restart_limit})"
    _wf_log(
        cast(dict[str, Any], out),
        (
            "<- re-run "
            f"ok={re_run_ok} rc={payload['reproduce_rc']} "
            f"dt={_fmt_dt(time.perf_counter()-t0)}"
        ),
    )
    return out


# Backward-compatible alias for legacy step name.
def _node_repro_crash(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    return _node_re_build(state)


def _route_after_build_state(state: FuzzWorkflowRuntimeState) -> str:
    state = cast(FuzzWorkflowRuntimeState, _normalize_error_state(cast(dict[str, Any], state)))
    if bool(state.get("restart_to_plan")):
        return "plan"
    err = dict(state.get("error") or {})
    if not _has_error_payload(err):
        return "run"
    return "plan"


def _route_after_run_state(state: FuzzWorkflowRuntimeState) -> str:
    state = cast(FuzzWorkflowRuntimeState, _normalize_error_state(cast(dict[str, Any], state)))
    if bool(state.get("restart_to_plan")):
        return "plan"
    if bool(state.get("crash_found")):
        return "crash-triage"
    err = dict(state.get("error") or {})
    terminal_reason = str(state.get("run_terminal_reason") or err.get("code") or "").strip().lower()
    # Coverage plateau is a coverage signal, not a hard run failure.
    # Let coverage-analysis decide in_place vs replan.
    if terminal_reason == "coverage_plateau":
        return "coverage-analysis"
    run_error_kind = str(state.get("run_error_kind") or err.get("code") or "").strip().lower()
    if run_error_kind in _RECOVERABLE_RUN_ERROR_KINDS:
        return "coverage-analysis"
    if run_error_kind in _FATAL_RUN_ERROR_KINDS:
        return "plan"
    if run_error_kind:
        return "plan"
    return "coverage-analysis"


def _route_after_coverage_analysis_state(state: FuzzWorkflowRuntimeState) -> str:
    state = cast(FuzzWorkflowRuntimeState, _normalize_error_state(cast(dict[str, Any], state)))
    err = dict(state.get("error") or {})
    if bool(state.get("failed")) or bool(err.get("terminal")):
        return "stop"
    if str(state.get("last_error") or err.get("message") or "").strip():
        return "stop"
    if bool(state.get("coverage_should_improve")):
        return "improve-harness"
    if _auto_stop_policy() == "hard_fail_only":
        return "run"
    return "stop"


def _route_after_improve_harness_state(state: FuzzWorkflowRuntimeState) -> str:
    state = cast(FuzzWorkflowRuntimeState, _normalize_error_state(cast(dict[str, Any], state)))
    err = dict(state.get("error") or {})
    if bool(state.get("failed")) or bool(err.get("terminal")):
        return "stop"
    if str(state.get("last_error") or err.get("message") or "").strip():
        return "stop"
    if str(state.get("coverage_improve_mode") or "").strip() == "replan" and not bool(
        state.get("coverage_replan_effective", True)
    ):
        if _auto_stop_policy() == "hard_fail_only":
            return "plan"
        return "stop"
    if bool(state.get("coverage_round_budget_exhausted")):
        if _auto_stop_policy() == "hard_fail_only":
            return "plan"
        return "stop"
    if bool(state.get("coverage_should_improve")):
        if str(state.get("coverage_improve_mode") or "").strip() == "in_place":
            return "build"
        return "plan"
    return "stop"


def _route_after_analysis_state(state: FuzzWorkflowRuntimeState) -> str:
    state = cast(FuzzWorkflowRuntimeState, _normalize_error_state(cast(dict[str, Any], state)))
    err = dict(state.get("error") or {})
    if bool(state.get("failed")) or bool(err.get("terminal")):
        return "stop"
    if str(state.get("last_error") or err.get("message") or "").strip() and not bool(state.get("analysis_degraded")):
        return "stop"
    return "plan"


def _route_after_plan_state(state: FuzzWorkflowRuntimeState) -> str:
    state = cast(FuzzWorkflowRuntimeState, _normalize_error_state(cast(dict[str, Any], state)))
    err = dict(state.get("error") or {})
    if bool(state.get("failed")) or bool(err.get("terminal")) or str(state.get("last_error") or err.get("message") or "").strip():
        return "stop"
    return "synthesize"


def _route_after_synthesize_state(state: FuzzWorkflowRuntimeState) -> str:
    state = cast(FuzzWorkflowRuntimeState, _normalize_error_state(cast(dict[str, Any], state)))
    err = dict(state.get("error") or {})
    if bool(state.get("failed")) or bool(err.get("terminal")) or str(state.get("last_error") or err.get("message") or "").strip():
        return "stop"
    return "build"


def _route_after_fix_build_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("restart_to_plan")):
        return "plan"
    if int(state.get("same_build_error_repeats") or 0) >= _fix_build_same_signature_plan_threshold():
        return "plan"
    terminal_reason = (state.get("fix_build_terminal_reason") or "").strip()
    if terminal_reason == "requires_env_rebuild":
        return "build"
    if terminal_reason:
        return "fix_build"
    if (state.get("last_error") or "").strip():
        return "fix_build"
    return "build"


def _route_after_fix_crash_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")):
        return "stop"
    if (state.get("last_error") or "").strip():
        return "stop"
    return "build"


def _route_after_crash_triage_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")):
        return "stop"
    if bool(state.get("restart_to_plan")):
        return "plan"
    label = _normalize_crash_triage_label(str(state.get("crash_triage_label") or ""))
    if label == "harness_bug":
        return "fix-harness"
    if label == "upstream_bug":
        return "re-build"
    return "plan"


def _route_after_fix_harness_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")):
        return "stop"
    if bool(state.get("restart_to_plan")):
        return "plan"
    if (state.get("last_error") or "").strip():
        return "plan"
    return "build"


def _re_restart_limit() -> int:
    raw = (os.environ.get("SHERPA_RESTART_FROM_PLAN_MAX") or "1").strip()
    try:
        return max(0, min(int(raw), 10))
    except Exception:
        return 1


def _route_after_re_build_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")):
        return "stop"
    if not bool(state.get("crash_found")):
        return "stop"
    if bool(state.get("restart_to_plan")):
        if int(state.get("restart_to_plan_count") or 0) > _re_restart_limit():
            return "stop"
        return "plan"
    if bool(state.get("re_build_done")) and bool(state.get("re_build_ok")):
        return "re-run"
    return "stop"


def _route_after_re_run_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")):
        return "stop"
    if not bool(state.get("crash_found")):
        return "stop"
    if bool(state.get("restart_to_plan")):
        if int(state.get("restart_to_plan_count") or 0) > _re_restart_limit():
            return "stop"
        return "plan"
    if bool(state.get("crash_repro_done")) and not bool(state.get("crash_repro_ok")):
        return "plan"
    if bool(state.get("crash_repro_done")) and bool(state.get("crash_repro_ok")):
        return "crash-analysis"
    return "stop"


def _route_after_crash_analysis_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")):
        return "stop"
    if bool(state.get("restart_to_plan")):
        if int(state.get("restart_to_plan_count") or 0) > _re_restart_limit():
            return "stop"
        return "plan"
    verdict = _normalize_crash_analysis_verdict(str(state.get("crash_analysis_verdict") or ""))
    if verdict == "false_positive":
        return "plan"
    return "stop"


def _recommended_next_step(state: FuzzWorkflowRuntimeState) -> str:
    state = cast(FuzzWorkflowRuntimeState, _normalize_error_state(cast(dict[str, Any], state)))
    last_step = str(state.get("last_step") or "").strip().lower()
    if not last_step:
        return "stop"
    if last_step == "init":
        return _route_after_init_state(state)
    if last_step == "analysis":
        return _route_after_analysis_state(state)
    if last_step == "plan":
        return _route_after_plan_state(state)
    if last_step == "synthesize":
        return _route_after_synthesize_state(state)
    if last_step == "build":
        return _route_after_build_state(state)
    if last_step == "fix_build":
        return _route_after_fix_build_state(state)
    if last_step == "run":
        return _route_after_run_state(state)
    if last_step == "fix_crash":
        return _route_after_fix_crash_state(state)
    if last_step == "crash-triage":
        return _route_after_crash_triage_state(state)
    if last_step == "fix-harness":
        return _route_after_fix_harness_state(state)
    if last_step == "coverage-analysis":
        return _route_after_coverage_analysis_state(state)
    if last_step == "improve-harness":
        return _route_after_improve_harness_state(state)
    if last_step in {"repro_crash", "re-build"}:
        return _route_after_re_build_state(state)
    if last_step == "re-run":
        return _route_after_re_run_state(state)
    if last_step == "crash-analysis":
        return _route_after_crash_analysis_state(state)
    return "stop"


def _route_after_init_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")) or (state.get("last_error") or "").strip():
        return "stop"
    raw = (state.get("resume_from_step") or "").strip().lower()
    allowed = {
        "analysis",
        "plan",
        "synthesize",
        "build",
        "fix_build",
        "run",
        "fix_crash",
        "crash-triage",
        "fix-harness",
        "coverage-analysis",
        "improve-harness",
        "re-build",
        "re-run",
        "repro_crash",
        "crash-analysis",
    }
    if raw == "repro_crash":
        raw = "re-build"
    if raw in allowed:
        return raw
    return "analysis"


def _should_stage_stop(state: FuzzWorkflowRuntimeState, step_name: str) -> bool:
    target = (state.get("stop_after_step") or "").strip().lower()
    return bool(target) and target == step_name


def _apply_stage_stop_guard(state: FuzzWorkflowRuntimeState, step_name: str, next_step: str) -> str:
    if _should_stage_stop(state, step_name):
        return "stop"
    return next_step


def build_fuzz_workflow() -> StateGraph:
    graph: StateGraph = StateGraph(FuzzWorkflowRuntimeState)

    graph.add_node("init", _node_init)
    graph.add_node("analysis", _node_analysis)
    graph.add_node("plan", _node_plan)
    graph.add_node("synthesize", _node_synthesize)
    graph.add_node("build", _node_build)
    graph.add_node("fix_build", _node_fix_build)
    graph.add_node("coverage-analysis", _node_coverage_analysis)
    graph.add_node("improve-harness", _node_improve_harness)
    graph.add_node("re-build", _node_re_build)
    graph.add_node("re-run", _node_re_run)
    graph.add_node("crash-analysis", _node_crash_analysis)
    graph.add_node("repro_crash", _node_repro_crash)
    graph.add_node("run", _node_run)
    graph.add_node("fix_crash", _node_fix_crash)
    graph.add_node("crash-triage", _node_crash_triage)
    graph.add_node("fix-harness", _node_fix_harness_after_run)

    graph.set_entry_point("init")

    def _route_after_plan(state: FuzzWorkflowRuntimeState) -> str:
        if (state.get("last_error") or "").strip():
            return "stop"
        if _should_stage_stop(state, "plan"):
            return "stop"
        return "synthesize"

    def _route_after_analysis(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_analysis_state(state)
        return _apply_stage_stop_guard(state, "analysis", nxt)

    def _route_after_synthesize(state: FuzzWorkflowRuntimeState) -> str:
        if (state.get("last_error") or "").strip():
            return "stop"
        if _should_stage_stop(state, "synthesize"):
            return "stop"
        return "build"

    def _route_after_build(state: FuzzWorkflowRuntimeState) -> str:
        if not (state.get("last_error") or "").strip():
            if _should_stage_stop(state, "build"):
                return "stop"
        return _route_after_build_state(state)

    def _route_after_run(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_run_state(state)
        return _apply_stage_stop_guard(state, "run", nxt)

    def _route_after_fix_build(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_fix_build_state(state)
        return _apply_stage_stop_guard(state, "fix_build", nxt)

    def _route_after_fix_crash(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_fix_crash_state(state)
        return _apply_stage_stop_guard(state, "fix_crash", nxt)

    def _route_after_crash_triage(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_crash_triage_state(state)
        return _apply_stage_stop_guard(state, "crash-triage", nxt)

    def _route_after_fix_harness(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_fix_harness_state(state)
        return _apply_stage_stop_guard(state, "fix-harness", nxt)

    def _route_after_coverage_analysis(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_coverage_analysis_state(state)
        return _apply_stage_stop_guard(state, "coverage-analysis", nxt)

    def _route_after_improve_harness(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_improve_harness_state(state)
        return _apply_stage_stop_guard(state, "improve-harness", nxt)

    def _route_after_re_build(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_re_build_state(state)
        return _apply_stage_stop_guard(state, "re-build", nxt)

    def _route_after_re_run(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_re_run_state(state)
        return _apply_stage_stop_guard(state, "re-run", nxt)

    def _route_after_crash_analysis(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_crash_analysis_state(state)
        return _apply_stage_stop_guard(state, "crash-analysis", nxt)

    graph.add_conditional_edges(
        "init",
        _route_after_init_state,
        {
            "analysis": "analysis",
            "plan": "plan",
            "synthesize": "synthesize",
            "build": "build",
            "fix_build": "fix_build",
            "run": "run",
            "fix_crash": "fix_crash",
            "crash-triage": "crash-triage",
            "fix-harness": "fix-harness",
            "coverage-analysis": "coverage-analysis",
            "improve-harness": "improve-harness",
            "re-build": "re-build",
            "re-run": "re-run",
            "crash-analysis": "crash-analysis",
            "repro_crash": "re-build",
            "stop": END,
        },
    )
    graph.add_conditional_edges("analysis", _route_after_analysis, {"plan": "plan", "stop": END})
    graph.add_conditional_edges("plan", _route_after_plan, {"synthesize": "synthesize", "stop": END})
    graph.add_conditional_edges("synthesize", _route_after_synthesize, {"build": "build", "stop": END})
    graph.add_conditional_edges(
        "build",
        _route_after_build,
        {"run": "run", "plan": "plan", "fix_build": "fix_build", "stop": END},
    )
    graph.add_conditional_edges(
        "fix_build",
        _route_after_fix_build,
        {"build": "build", "fix_build": "fix_build", "plan": "plan", "stop": END},
    )
    graph.add_conditional_edges(
        "run",
        _route_after_run,
        {
            "coverage-analysis": "coverage-analysis",
            "crash-triage": "crash-triage",
            "fix_crash": "fix_crash",
            "plan": "plan",
            "stop": END,
        },
    )
    graph.add_conditional_edges(
        "fix_crash",
        _route_after_fix_crash,
        {"build": "build", "fix_crash": "fix_crash", "stop": END},
    )
    graph.add_conditional_edges(
        "crash-triage",
        _route_after_crash_triage,
        {"fix-harness": "fix-harness", "re-build": "re-build", "plan": "plan", "stop": END},
    )
    graph.add_conditional_edges(
        "fix-harness",
        _route_after_fix_harness,
        {"build": "build", "plan": "plan", "stop": END},
    )
    graph.add_conditional_edges(
        "coverage-analysis",
        _route_after_coverage_analysis,
        {"improve-harness": "improve-harness", "stop": END},
    )
    graph.add_conditional_edges(
        "improve-harness",
        _route_after_improve_harness,
        {"plan": "plan", "stop": END},
    )
    graph.add_conditional_edges(
        "re-build",
        _route_after_re_build,
        {"re-run": "re-run", "fix_crash": "fix_crash", "plan": "plan", "stop": END},
    )
    graph.add_conditional_edges(
        "re-run",
        _route_after_re_run,
        {"crash-analysis": "crash-analysis", "plan": "plan", "fix_crash": "fix_crash", "stop": END},
    )
    graph.add_conditional_edges(
        "crash-analysis",
        _route_after_crash_analysis,
        {"plan": "plan", "stop": END},
    )
    graph.add_conditional_edges(
        "repro_crash",
        _route_after_re_build,
        {"re-run": "re-run", "fix_crash": "fix_crash", "plan": "plan", "stop": END},
    )

    return graph


def _detect_harness_error(repo_root: Path) -> bool:
    return _wf_summary.detect_harness_error(repo_root)


def _bytes_human(num_bytes: int) -> str:
    return _wf_summary.bytes_human(num_bytes)


def _tree_file_stats(root: Path) -> tuple[int, int]:
    return _wf_summary.tree_file_stats(root)


def _collect_fuzz_inventory(repo_root: Path) -> dict[str, Any]:
    return _wf_summary.collect_fuzz_inventory(repo_root)


def _write_run_summary(out: dict[str, Any]) -> None:
    _wf_summary.write_run_summary(out)


def run_fuzz_workflow(inp: FuzzWorkflowInput) -> dict[str, Any]:
    total_budget_log = "unlimited" if int(inp.time_budget) == 0 else f"{int(inp.time_budget)}s"
    run_budget_log = "unlimited" if int(inp.run_time_budget) == 0 else f"{int(inp.run_time_budget)}s"
    resume_step = (inp.resume_from_step or "").strip().lower()
    if resume_step == "repro_crash":
        resume_step = "re-build"
    resume_root = str(inp.resume_repo_root or "").strip()
    stop_after_step = (inp.stop_after_step or "").strip().lower()
    _wf_log(
        None,
        "workflow start "
        f"repo={inp.repo_url} docker_image={inp.docker_image or '(native)'} "
        f"time_budget={total_budget_log} run_time_budget={run_budget_log} "
        f"resume_step={resume_step or '-'} resume_root={resume_root or '-'} "
        f"stop_after_step={stop_after_step or '-'}",
    )
    t0 = time.perf_counter()
    try:
        max_steps_env = int(os.environ.get("SHERPA_WORKFLOW_MAX_STEPS", "0"))
    except Exception:
        max_steps_env = 0
    # max_steps <= 0 means unlimited workflow steps.
    max_steps = 0 if max_steps_env <= 0 else max(3, max_steps_env)
    wf = build_fuzz_workflow().compile()
    raw: Any = wf.invoke(
        {
            "repo_url": inp.repo_url,
            "email": inp.email,
            "time_budget": inp.time_budget,
            "run_time_budget": inp.run_time_budget,
            "workflow_started_at": time.time(),
            "max_len": inp.max_len,
            "docker_image": inp.docker_image,
            "ai_key_path": str(inp.ai_key_path),
            "resume_from_step": resume_step,
            "resume_repo_root": str(inp.resume_repo_root or ""),
            "stop_after_step": stop_after_step,
            "coverage_loop_max_rounds": max(
                0,
                int(inp.coverage_loop_max_rounds if inp.coverage_loop_max_rounds is not None else 0),
            ),
            "max_fix_rounds": max(
                0,
                int(inp.max_fix_rounds if inp.max_fix_rounds is not None else 0),
            ),
            "same_error_max_retries": max(
                0,
                int(inp.same_error_max_retries if inp.same_error_max_retries is not None else 0),
            ),
            "restart_to_plan": bool(
                str(inp.restart_to_plan_reason or "").strip()
                or str(inp.restart_to_plan_error_text or "").strip()
                or str(inp.restart_to_plan_stage or "").strip()
            ),
            "restart_to_plan_reason": str(inp.restart_to_plan_reason or ""),
            "restart_to_plan_stage": str(inp.restart_to_plan_stage or ""),
            "restart_to_plan_error_text": str(inp.restart_to_plan_error_text or ""),
            "restart_to_plan_report_path": str(inp.restart_to_plan_report_path or ""),
            "last_fuzzer": str(inp.last_fuzzer or ""),
            "last_crash_artifact": str(inp.last_crash_artifact or ""),
            "re_workspace_root": str(inp.re_workspace_root or ""),
            "max_steps": max_steps,
        }
    )
    out = _normalize_error_state(cast(dict[str, Any], raw) if isinstance(raw, dict) else {})
    try:
        _write_run_summary(out)
    except Exception:
        pass
    msg = str(out.get("message") or "Fuzzing completed.").strip()
    recommended_next = _recommended_next_step(cast(FuzzWorkflowRuntimeState, out))
    err = dict(out.get("error") or {})
    if bool(out.get("failed")) or bool(err.get("terminal")):
        _wf_log(out, f"workflow end status=failed dt={_fmt_dt(time.perf_counter()-t0)}")
        terminal_reason = str(out.get("run_terminal_reason") or err.get("code") or "").strip() or str(
            out.get("fix_build_terminal_reason") or ""
        ).strip()
        if terminal_reason:
            msg = f"{terminal_reason}: {msg}"
        raise RuntimeError(msg or "workflow failed")
    # If we stopped due to an error but didn't mark failed, still surface it.
    last_error = str(out.get("last_error") or err.get("message") or "").strip()
    if last_error and not bool(out.get("crash_found")):
        if stop_after_step and recommended_next != "stop":
            _wf_log(
                out,
                (
                    "workflow end status=stage_recoverable "
                    f"next={recommended_next} dt={_fmt_dt(time.perf_counter()-t0)}"
                ),
            )
        else:
            _wf_log(out, f"workflow end status=error dt={_fmt_dt(time.perf_counter()-t0)}")
            raise RuntimeError(last_error)

    if not (last_error and not bool(out.get("crash_found")) and stop_after_step and recommended_next != "stop"):
        _wf_log(out, f"workflow end status=ok dt={_fmt_dt(time.perf_counter()-t0)}")
    return {
        "message": msg,
        "error": dict(out.get("error") or {}),
        "repo_root": str(out.get("repo_root") or ""),
        "workflow_last_step": str(out.get("last_step") or ""),
        "workflow_active_step": str(out.get("next") or ""),
        "workflow_recommended_next": str(recommended_next or ""),
        "stop_after_step": stop_after_step,
        "fix_build_terminal_reason": str(out.get("fix_build_terminal_reason") or ""),
        "fix_build_attempts": int(out.get("fix_build_attempts") or 0),
        "fix_build_noop_streak": int(out.get("fix_build_noop_streak") or 0),
        "fix_build_rule_hits": list(out.get("fix_build_rule_hits") or []),
        "run_error_kind": str(out.get("run_error_kind") or ""),
        "run_terminal_reason": str(out.get("run_terminal_reason") or ""),
        "run_idle_seconds": int(out.get("run_idle_seconds") or 0),
        "run_children_exit_count": int(out.get("run_children_exit_count") or 0),
        "coverage_loop_max_rounds": int(
            out.get("coverage_loop_max_rounds")
            if out.get("coverage_loop_max_rounds") is not None
            else 0
        ),
        "coverage_loop_round": int(out.get("coverage_loop_round") or 0),
        "coverage_should_improve": bool(out.get("coverage_should_improve") or False),
        "coverage_improve_reason": str(out.get("coverage_improve_reason") or ""),
        "coverage_bottleneck_kind": str(out.get("coverage_bottleneck_kind") or ""),
        "coverage_bottleneck_reason": str(out.get("coverage_bottleneck_reason") or ""),
        "coverage_history": list(out.get("coverage_history") or []),
        "analysis_evidence_count": int(out.get("analysis_evidence_count") or 0),
        "target_scoring_enabled": bool(out.get("target_scoring_enabled") or False),
        "constraint_memory_count": int(out.get("constraint_memory_count") or 0),
        "constraint_memory_path": str(out.get("constraint_memory_path") or ""),
        "plan_retry_reason": str(out.get("plan_retry_reason") or ""),
        "plan_targets_schema_valid_before_retry": bool(out.get("plan_targets_schema_valid_before_retry") or False),
        "plan_targets_schema_valid_after_retry": bool(out.get("plan_targets_schema_valid_after_retry") or False),
        "plan_used_fallback_targets": bool(out.get("plan_used_fallback_targets") or False),
        "max_fix_rounds": int(out.get("max_fix_rounds") if out.get("max_fix_rounds") is not None else 0),
        "same_error_max_retries": int(
            out.get("same_error_max_retries")
            if out.get("same_error_max_retries") is not None
            else 0
        ),
        "fix_action_type": str(out.get("fix_action_type") or ""),
        "fix_effect": str(out.get("fix_effect") or ""),
        "build_error_signature_before": str(out.get("build_error_signature_before") or ""),
        "build_error_signature_after": str(out.get("build_error_signature_after") or ""),
        "crash_repro_done": bool(out.get("crash_repro_done") or False),
        "crash_repro_ok": bool(out.get("crash_repro_ok") or False),
        "crash_repro_rc": int(out.get("crash_repro_rc") or 0),
        "crash_repro_report_path": str(out.get("crash_repro_report_path") or ""),
        "crash_repro_json_path": str(out.get("crash_repro_json_path") or ""),
        "crash_triage_done": bool(out.get("crash_triage_done") or False),
        "crash_triage_label": str(out.get("crash_triage_label") or ""),
        "crash_triage_confidence": float(out.get("crash_triage_confidence") or 0.0),
        "crash_triage_reason": str(out.get("crash_triage_reason") or ""),
        "crash_triage_report_path": str(out.get("crash_triage_report_path") or ""),
        "crash_triage_json_path": str(out.get("crash_triage_json_path") or ""),
        "re_build_done": bool(out.get("re_build_done") or False),
        "re_build_ok": bool(out.get("re_build_ok") or False),
        "re_build_rc": int(out.get("re_build_rc") or 0),
        "re_build_report_path": str(out.get("re_build_report_path") or ""),
        "re_build_json_path": str(out.get("re_build_json_path") or ""),
        "re_run_done": bool(out.get("re_run_done") or False),
        "re_run_ok": bool(out.get("re_run_ok") or False),
        "re_run_rc": int(out.get("re_run_rc") or 0),
        "re_run_report_path": str(out.get("re_run_report_path") or ""),
        "re_run_json_path": str(out.get("re_run_json_path") or ""),
        "crash_analysis_done": bool(out.get("crash_analysis_done") or False),
        "crash_analysis_verdict": str(out.get("crash_analysis_verdict") or ""),
        "crash_analysis_reason": str(out.get("crash_analysis_reason") or ""),
        "crash_analysis_report_path": str(out.get("crash_analysis_report_path") or ""),
        "crash_analysis_json_path": str(out.get("crash_analysis_json_path") or ""),
        "re_workspace_root": str(out.get("re_workspace_root") or ""),
        "last_fuzzer": str(out.get("last_fuzzer") or ""),
        "last_crash_artifact": str(out.get("last_crash_artifact") or ""),
        "restart_to_plan": bool(out.get("restart_to_plan") or False),
        "restart_to_plan_reason": str(out.get("restart_to_plan_reason") or ""),
        "restart_to_plan_stage": str(out.get("restart_to_plan_stage") or ""),
        "restart_to_plan_error_text": str(out.get("restart_to_plan_error_text") or ""),
        "restart_to_plan_report_path": str(out.get("restart_to_plan_report_path") or ""),
        "restart_to_plan_count": int(out.get("restart_to_plan_count") or 0),
        "build_error_kind": str(out.get("build_error_kind") or ""),
        "build_error_code": str(out.get("build_error_code") or ""),
    }
