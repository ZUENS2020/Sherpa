"""Carved from workflow_graph.py - shared workflow helper library. Single self-contained module: the planning domains are tightly coupled, so splitting them further would create import cycles."""

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
    _bounded_float,
    _has_error_payload,
    _normalize_error_state,
    _wf_log,
)
from workflow_public_api import (  # noqa: E402
    _NON_PUBLIC_API_NAME_PATTERNS,
    _PUBLIC_API_SYMBOL_CACHE,
    _CALLGRAPH_REVERSE_CACHE,
    _is_internal_api_symbol,
    _vuln_public_api_enforce,
    _companion_work_dir,
    _load_public_api_symbols,
    _load_api_loc_index,
    _api_surface_class,
    _is_non_public_api,
    _is_unlinkable_binding_api,
    _load_callgraph_reverse,
    _load_callgraph_forward,
    _nearest_public_entry,
    coverage_potential,
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


def _effective_run_error_kind(state: dict[str, Any]) -> str:
    """Normalize run error kind for routing/repair decisions.

    nonzero_exit_without_crash is usually fatal, but if one fuzzer timed out
    (timeout artifact) while at least one sibling fuzzer completed normally,
    treat it as recoverable timeout-like signal for the repair loop.
    """
    kind = str(state.get("run_error_kind") or "").strip().lower()
    if kind != "nonzero_exit_without_crash":
        return kind
    run_details = list(state.get("run_details") or [])
    if not run_details:
        return kind

    has_timeout_artifact = False
    has_clean_success = False
    for detail in run_details:
        if str(detail.get("crash_evidence") or "").strip().lower() == "timeout_artifact":
            has_timeout_artifact = True
        if int(detail.get("rc") or 0) == 0 and not bool(detail.get("crash_found")):
            has_clean_success = True

    if has_timeout_artifact and has_clean_success:
        return "run_timeout"
    return kind


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


def _record_decision_trace(
    state: dict[str, Any],
    *,
    stage: str,
    tool: str = "",
    model: str = "",
    latency_ms: int | None = None,
    token_usage: dict[str, Any] | None = None,
    error_kind: str = "",
    error_code: str = "",
    retry_count: int = 0,
    decision_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _wf_obs.record_decision_trace(
        state,
        stage=stage,
        tool=tool,
        model=model,
        latency_ms=latency_ms,
        token_usage=token_usage,
        error_kind=error_kind,
        error_code=error_code,
        retry_count=retry_count,
        decision_snapshot=decision_snapshot,
    )


def _emit_fuzz_metrics(state: dict[str, Any]) -> None:
    _wf_obs.emit_fuzz_metrics(state)


def _grace_wait_for_file(path: Path, max_sec: int = 5, *, min_size: int = 1) -> bool:
    """Wait up to max_sec for a file to appear on disk with minimum size.

    This handles filesystem flush delays when OpenCode (Node.js async I/O)
    signals completion via ./done before output files are fully written.
    """
    if max_sec <= 0:
        return path.is_file() and path.stat().st_size >= min_size
    deadline = time.time() + max_sec
    while time.time() < deadline:
        if path.is_file() and path.stat().st_size >= min_size:
            return True
        time.sleep(0.5)
    return path.is_file() and path.stat().st_size >= min_size


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
        "plan_repair_fix_harness",
        "synthesize_repair_fix_harness",
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


def _build_fix_harness_crash_context(
    repo_root: Path,
    *,
    include_contents: bool,
) -> tuple[str, list[str]]:
    crash_info = repo_root / "crash_info.md"
    crash_analysis = repo_root / "crash_analysis.md"
    triage_json = repo_root / "crash_triage.json"
    known_issues: list[str] = []
    lines = [
        "Repo-root crash evidence for fix-harness repair:",
        f"- crash_info_path: {crash_info}",
        f"- crash_triage_json_path: {triage_json}",
        f"- crash_analysis_path: {crash_analysis}",
        "- Do not guess `fuzz/crash_*` paths; these artifacts live at repo root unless an explicit path says otherwise.",
    ]
    info_text = crash_info.read_text(encoding="utf-8", errors="replace") if crash_info.is_file() else ""
    triage_text = triage_json.read_text(encoding="utf-8", errors="replace") if triage_json.is_file() else ""
    analysis_text = crash_analysis.read_text(encoding="utf-8", errors="replace") if crash_analysis.is_file() else ""
    if not crash_analysis.is_file():
        known_issues.append("crash_analysis_not_available_yet")
        lines.append("- crash_analysis_status: unavailable during crash-triage repair path; rely on crash_info.md and crash_triage.json first.")
    if include_contents:
        if info_text:
            lines.append("=== repo-root crash_info.md ===\n" + _trim_feedback_text(info_text))
        if triage_text:
            lines.append("=== repo-root crash_triage.json ===\n" + _trim_feedback_text(triage_text))
        if analysis_text:
            lines.append("=== repo-root crash_analysis.md ===\n" + _trim_feedback_text(analysis_text))
    return "\n".join(lines).strip(), known_issues


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
        "plan_repair_fix_harness": _feedback_group_for_stage("plan_repair_fix_harness"),
        "synthesize_repair_fix_harness": _feedback_group_for_stage("synthesize_repair_fix_harness"),
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


def _try_hotfix_missing_decl(state: dict[str, Any], build_py_path: str) -> bool:
    """Detect implicit-function-decl / undeclared-identifier build errors
    and insert ``extern`` declarations into harness source files.

    Called from the build retry loop so header-only / single-file library
    harnesses can be patched in-place without going through a full
    plan→synthesize→build cycle.
    """
    last_error = str(state.get("last_error") or state.get("build_stdout_tail") or "").strip()
    stdout_tail = str(state.get("build_stdout_tail") or "").strip()
    stderr_tail = str(state.get("build_stderr_tail") or "").strip()
    diag_raw = (last_error + "\n" + stdout_tail + "\n" + stderr_tail)
    diag_lower = diag_raw.lower()
    if "implicit declaration of function" not in diag_lower and "undeclared identifier" not in diag_lower:
        return False

    gen = state.get("generator")
    if gen is None:
        return False
    repo_root = gen.repo_root

    import re as _re_local
    _changes = 0
    _diag_lines = diag_raw.splitlines()
    for _dl in _diag_lines:
        _m = _re_local.search(
            r"(?P<file>[^\s:]+\.(?:c|cc|cpp|cxx)):\d+:\d+:\s+(?:error|fatal error):\s+.*?(?:undeclared identifier|implicit declaration of function).*?(?:'|\")(?P<sym>[A-Za-z_][A-Za-z0-9_]*)(?:'|\")",
            _dl,
        )
        if not _m:
            continue
        _src_name = _m.group("file")
        _symbol = _m.group("sym")
        _src_path = Path(_src_name)
        if not _src_path.is_absolute():
            _src_path = repo_root / _src_path
        if not _src_path.is_file() or _src_path.parent.name != "fuzz":
            continue
        try:
            _src_text = _src_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "extern " in _src_text and _symbol in _src_text:
            continue
        _lines = _src_text.splitlines()
        _insert_at = 0
        for _j, _line in enumerate(_lines):
            if _line.lstrip().startswith("#include") or _line.lstrip().startswith("#define"):
                _insert_at = _j + 1
        if _insert_at == 0:
            for _j, _line in enumerate(_lines):
                if _line.lstrip().startswith("#if"):
                    _insert_at = _j + 1
        _decl = f"extern void {_symbol}(void);  /* sherpa-hotfix: forward declaration for header-only lib */"
        _lines.insert(_insert_at, _decl)
        _new_text = "\n".join(_lines) + ("\n" if _src_text.endswith("\n") else "")
        if _new_text == _src_text:
            continue
        try:
            _src_path.write_text(_new_text, encoding="utf-8", errors="replace")
            _changes += 1
            _wf_log(state, f"build: added extern {_symbol}() to {_src_path.relative_to(repo_root)}")
        except Exception:
            continue
    return _changes > 0


def _install_coverage_cc_wrapper(repo_root: Path) -> Path:
    """Install a compiler wrapper that auto-injects -fsanitize-coverage flags.

    Returns the wrapper directory path (add it to ``PATH`` before the real
    compiler).  The wrapper delegates every call to the system clang/clang++
    and silently appends ``-fsanitize-coverage=inline-8bit-counters,pc-table``
    unless the command already contains ``-fsanitize-coverage`` or
    ``-fprofile-instr-generate`` (replay builds).

    The wrapper directory contains symlinks ``clang`` and ``clang++`` pointing
    to the wrapper script, so build scripts that hardcode ``clang`` pick it up
    automatically when the directory is first in ``PATH``.
    """
    wrapper_dir = repo_root / "fuzz" / ".sherpa-cc"
    wrapper_dir.mkdir(parents=True, exist_ok=True)

    _COV_FLAGS = "-fsanitize-coverage=inline-8bit-counters,pc-table"

    wrapper_sh = wrapper_dir / "sherpa-cc-wrapper.sh"
    wrapper_sh.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        set -euo pipefail
        # sherpa-cc-wrapper: inject coverage flags for primary fuzz builds.
        # Symlinked as 'clang' and 'clang++'; delegates to the real compiler
        # while appending coverage instrumentation unless the command is a
        # replay build (already has -fprofile-instr-generate).
        COV_FLAGS="{_COV_FLAGS}"
        HAS_COV=0
        HAS_REPLAY=0
        # Rewrite any deprecated trace-pc-guard coverage flag to the modern set.
        # clang>=14 libFuzzer refuses trace-pc-guard binaries at runtime, so a
        # flag from build.py/CMake/anywhere must be normalized at compile time.
        ARGS=()
        for arg in "$@"; do
            case "$arg" in
                *trace-pc-guard*) arg="{_COV_FLAGS}" ;;
            esac
            ARGS+=("$arg")
            [[ "$arg" == *fsanitize-coverage* ]] && HAS_COV=1
            [[ "$arg" == *fprofile-instr-generate* ]] && HAS_REPLAY=1
        done
        REAL_CC=$(basename "$0")
        REAL_PATH=$(command -v "$REAL_CC" 2>/dev/null || echo "")
        if [[ -z "$REAL_PATH" || "$REAL_PATH" == "$0" ]]; then
            # Fall back to PATH search excluding this directory
            REAL_PATH=$(PATH=${{PATH#*:}} command -v "$REAL_CC" 2>/dev/null || echo "/usr/bin/$REAL_CC")
        fi
        if [[ $HAS_COV -eq 0 && $HAS_REPLAY -eq 0 ]]; then
            exec "$REAL_PATH" "${{ARGS[@]}}" $COV_FLAGS
        else
            exec "$REAL_PATH" "${{ARGS[@]}}"
        fi
    """))
    wrapper_sh.chmod(0o755)

    # Create symlinks so that 'clang' and 'clang++' in this directory
    # resolve to the wrapper.
    for name in ("clang", "clang++"):
        link = wrapper_dir / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(wrapper_sh.name)

    return wrapper_dir


def _apply_coverage_cc_wrapper_env(build_env: dict[str, str], repo_root: Path) -> dict[str, str]:
    """Prepend the coverage cc-wrapper to a build env's PATH so EVERY bare
    `clang`/`clang++` invocation in build.py (library objects included, not just
    the harness link) gets `-fsanitize-coverage` instrumentation.

    Without this, build.py compiles the target library WITHOUT coverage and
    libFuzzer is blind to it — the fuzzer cannot guide mutation into the library
    and coverage flatlines. The main build stage already wired this; the
    run-stage workspace rebuild (which produces the binary that is actually
    fuzzed) did not, so the fuzzed binary was uninstrumented. Best-effort."""
    try:
        wrapper_dir = _install_coverage_cc_wrapper(repo_root)
        build_env["PATH"] = f"{wrapper_dir}:{build_env.get('PATH', '')}"
        build_env.setdefault("CC", "clang")
        build_env.setdefault("CXX", "clang++")
    except Exception:
        pass
    return build_env


def _inject_coverage_instrumentation(build_py_path: str, state: dict[str, Any]) -> None:
    """Inject -fsanitize-coverage flags into build.py primary fuzz link commands.

    LibFuzzer requires coverage feedback instrumentation to guide its mutation
    engine.  Without ``-fsanitize-coverage=inline-8bit-counters,pc-table``
    the fuzzer runs blind (corp:1/1b, no corpus growth).  The synthesize agent
    occasionally omits these flags even when the SKILL.md contract requires them.
    This function inserts a separate ``'-fsanitize-coverage=...'`` list element
    after the ``-fsanitize=fuzzer`` element so that clang receives them as
    distinct arguments.
    """
    COVERAGE_FLAGS = "-fsanitize-coverage=inline-8bit-counters,pc-table"
    bp = Path(build_py_path)
    if not bp.is_file():
        return
    try:
        text = bp.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return

    # Normalize any deprecated trace-pc-guard the synthesize agent may have
    # written directly. Modern libFuzzer (clang >= ~14) removed trace-pc-guard
    # and refuses to run such binaries at startup
    # ("trace-pc-guard is no longer supported by libFuzzer"). Rewrite it to the
    # supported inline-8bit-counters,pc-table set in place.
    if "trace-pc-guard" in text:
        normalized = re.sub(
            r"-fsanitize-coverage=[\w,\-]*trace-pc-guard[\w,\-]*",
            COVERAGE_FLAGS,
            text,
        )
        if normalized != text:
            text = normalized
            try:
                bp.write_text(text, encoding="utf-8", errors="replace")
                _wf_log(state, "build: normalized deprecated trace-pc-guard -> inline-8bit-counters,pc-table")
            except Exception:
                pass

    # --- Pass A: instrument the target *library* build ---------------------
    # Many agent build.py's compile the target library via `make` with a
    # hardcoded compile-flags string that lacks coverage, e.g.:
    #     cflags = "-std=c99 -Wall -Wextra -fpic -O2 -DNDEBUG"
    #     env["CFLAGS"] = cflags
    # That object is then linked into the fuzzer, but without
    # -fsanitize-coverage libFuzzer is BLIND to the library and coverage
    # flatlines at the handful of harness edges (observed: cov stuck at ~7 even
    # on valid inputs that should drive the whole parser). The PATH cc-wrapper
    # is best-effort and does not reliably reach `make` subprocesses, so here we
    # deterministically append coverage to every compile-flags string literal
    # that is NOT a replay build (-fprofile-instr-generate) and does not already
    # carry coverage. This is the reliable, format-agnostic library hook.
    def _augment_cflags(m: "re.Match[str]") -> str:
        head, quote, body = m.group(1), m.group(2), m.group(3)
        if (
            "-fprofile-instr-generate" in body
            or "-fcoverage-mapping" in body
            or "-fsanitize-coverage" in body
            or "-fsanitize=fuzzer" in body
            or "-" not in body
        ):
            return m.group(0)
        return f"{head}{quote}{body} {COVERAGE_FLAGS}{quote}"

    lib_cflags_re = re.compile(
        r"((?:\b(?:cflags|c_flags|cxxflags|cxx_flags)\b\s*=\s*"
        r"|\[\s*['\"](?:CFLAGS|CXXFLAGS)['\"]\s*\]\s*=\s*))"
        r"(['\"])([^'\"]*)\2",
        re.IGNORECASE,
    )
    lib_text = lib_cflags_re.sub(_augment_cflags, text)
    if lib_text != text:
        text = lib_text
        try:
            bp.write_text(text, encoding="utf-8", errors="replace")
            _wf_log(state, f"build: injected {COVERAGE_FLAGS} into library CFLAGS in build.py")
        except Exception:
            pass

    # --- Pass C: force coverage onto the library `make` build via CC=wrapper -
    # Pass A only helps when build.py passes a CFLAGS string the Makefile
    # honours. Many real Makefiles (e.g. tomlc99) HARD-ASSIGN `CFLAGS = ...`,
    # which overrides the environment, so env/CFLAGS-string coverage never
    # reaches the library and coverage flatlines (observed: cov plateaus at ~23
    # with ft in the thousands — value-profile churns on a couple dozen edges).
    # A command-line `make CC=<wrapper>` overrides even a hard-assigned Makefile
    # variable, and the wrapper appends -fsanitize-coverage to every compile
    # while preserving the project's own CFLAGS. The wrapper internally skips
    # replay builds (-fprofile-instr-generate), and we only touch make calls
    # WITHOUT an `env=` kwarg, which targets the primary (non-replay) library
    # build and leaves coverage/replay make calls (which pass env=) alone.
    try:
        repo_root = bp.parent.parent
        wrapper_dir = _install_coverage_cc_wrapper(repo_root)
        cc_path = str(wrapper_dir / "clang")
        cxx_path = str(wrapper_dir / "clang++")
        if cc_path not in text:
            # Match `run([... "make" ...])` / `subprocess.run([... "make" ...])`
            # where the list is closed immediately by `)` (no env= kwarg).
            make_call_re = re.compile(
                r"((?:subprocess\.)?run\(\s*\[\s*['\"]make['\"][^\]]*?)(\]\s*\))"
            )

            def _augment_make(m: "re.Match[str]") -> str:
                head, tail = m.group(1), m.group(2)
                if ".sherpa-cc" in head:
                    return m.group(0)
                return f'{head}, "CC={cc_path}", "CXX={cxx_path}"{tail}'

            new_make = make_call_re.sub(_augment_make, text)
            if new_make != text:
                text = new_make
                bp.write_text(text, encoding="utf-8", errors="replace")
                _wf_log(state, "build: forced coverage onto library make build via CC=<cc-wrapper>")
    except Exception:
        pass

    # --- Pass D: instrument direct `clang -c` library/object compiles --------
    # Some build.py's skip make entirely and compile the library with explicit
    # clang commands that carry no coverage and no -fsanitize=fuzzer, e.g.
    #     cmd = ["clang", "-c", "-std=c99", "-Wall", "-Wextra"]
    # The resulting object is linked into the fuzzer uninstrumented and coverage
    # flatlines. Append coverage to any single-line clang/clang++ command list
    # that starts with the compiler, compiles (-c), and is neither a libFuzzer
    # build (-fsanitize=fuzzer) nor a replay build (-fprofile-instr-generate)
    # nor already instrumented.
    def _augment_clang_compile(m: "re.Match[str]") -> str:
        seg, close = m.group(1), m.group(2)
        if not ('"-c"' in seg or "'-c'" in seg):
            return m.group(0)
        if (
            "-fsanitize=fuzzer" in seg
            or "-fprofile-instr-generate" in seg
            or "-fcoverage-mapping" in seg
            or "-fsanitize-coverage" in seg
        ):
            return m.group(0)
        return f'{seg}, "{COVERAGE_FLAGS}"{close}'

    clang_compile_re = re.compile(
        r"(\[\s*['\"]clang(?:\+\+)?['\"][^\[\]\n]*?)(\])"
    )
    new_clang = clang_compile_re.sub(_augment_clang_compile, text)
    if new_clang != text:
        text = new_clang
        try:
            bp.write_text(text, encoding="utf-8", errors="replace")
            _wf_log(state, f"build: injected {COVERAGE_FLAGS} into direct clang -c compile in build.py")
        except Exception:
            pass

    # --- Pass B: instrument the harness link (clang -fsanitize=fuzzer ...) ---
    # Idempotent: per-line guards below skip lines that already carry coverage
    # or are replay builds, so this is safe even after Pass A added coverage.
    lines = text.splitlines()
    changed = False
    for i, line in enumerate(lines):
        if "-fsanitize=fuzzer" not in line:
            continue
        if "-fsanitize-coverage" in line:
            continue
        if "-fprofile-instr-generate" in line or "-fcoverage-mapping" in line:
            continue

        indent = line[:len(line) - len(line.lstrip())]

        # Case 1: standalone flag line: "    '-fsanitize=fuzzer,address,undefined',"
        m = re.match(r"^(\s*)(['\"])(-fsanitize=fuzzer[^'\"]*)\2\s*,?\s*$", line)
        if m:
            q = m.group(2)
            lines.insert(i + 1, f"{indent}{q}{COVERAGE_FLAGS}{q},")
            changed = True
            break

        # Case 2: flag inside a longer line (e.g. a Python list element) — insert
        # the coverage flag as a SEPARATE quoted element. Capture the element and
        # any trailing separator separately so a comma is guaranteed between
        # them: without it two adjacent Python string literals silently
        # concatenate into one malformed flag (e.g.
        # "...undefined""-fsanitize-coverage=..." -> a single invalid
        # `-fsanitize=...undefined-fsanitize-coverage` arg that fails to compile).
        m2 = re.search(r"(['\"]-fsanitize=fuzzer[^'\"]*['\"])(\s*,?\s*)", line)
        if m2:
            q = m2.group(1)[0]
            sep = m2.group(2)
            if "," in sep:
                # already comma-separated: insert after the separator
                ins = f"{q}{COVERAGE_FLAGS}{q}, "
                lines[i] = line[:m2.end()] + ins + line[m2.end():]
            else:
                # no trailing comma (last/only list element): add one before the
                # inserted element so the two literals don't concatenate
                ins = f", {q}{COVERAGE_FLAGS}{q}"
                lines[i] = line[:m2.end(1)] + ins + line[m2.end(1):]
            changed = True
            break

    if not changed:
        for i, line in enumerate(lines):
            if "-fsanitize=fuzzer" in line and COVERAGE_FLAGS not in line and "replay" not in line.lower() and "-fprofile" not in line:
                lines[i] = line.replace(
                    "-fsanitize=fuzzer,address,undefined",
                    f"-fsanitize=fuzzer,address,undefined -fsanitize-coverage=inline-8bit-counters,pc-table",
                )
                if lines[i] != line:
                    changed = True
                    break

    if changed:
        new_text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        try:
            bp.write_text(new_text, encoding="utf-8", errors="replace")
            _wf_log(state, f"build: injected {COVERAGE_FLAGS} into build.py")
        except Exception:
            pass


def _clear_opencode_done_sentinel(repo_root: Path) -> bool:
    done_path = _opencode_done_path(repo_root)
    if not done_path.exists():
        return False
    try:
        if done_path.is_dir():
            shutil.rmtree(done_path)
            return True
        done_path.unlink()
        return True
    except Exception:
        return False


def _infer_repair_origin_stage(state: dict[str, Any]) -> str:
    explicit = str(state.get("repair_origin_stage") or "").strip().lower()
    if explicit in {"build", "crash", "coverage", "fix-harness"}:
        return explicit
    restart_stage = str(state.get("restart_to_plan_stage") or "").strip().lower()
    if restart_stage == "build":
        return "build"
    if restart_stage == "fix-harness":
        return "fix-harness"
    if restart_stage in {"run", "crash-triage", "re-build", "re-run", "fix_crash"}:
        return "crash"
    if restart_stage in {"per-input-replay", "coverage-analysis", "improve-harness"}:
        return "coverage"
    last_step = str(state.get("last_step") or "").strip().lower()
    if last_step == "build":
        return "build"
    if last_step == "fix-harness":
        return "fix-harness"
    if last_step in {"run", "crash-triage", "re-build", "re-run", "fix_crash"}:
        return "crash"
    if last_step in {"per-input-replay", "coverage-analysis", "improve-harness"}:
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
        "source": str(stage or "").strip() or "unknown",
        "source_stage": str(stage or "").strip() or "unknown",
        "classification": str(classification or "").strip() or "unknown",
        "reason": str(reason or "").strip()[:1024],
        "evidence": [str(x).strip()[:512] for x in list(evidence or []) if str(x).strip()][:12],
        "confidence": max(0.0, min(float(confidence), 1.0)),
        "suspected_precondition": str(reason or "").strip()[:512],
        "fix_hint": _constraint_fix_hint(classification),
        "first_seen": int(prev.get("first_seen") or now),
        "last_seen": now,
        "latest_seen": now,
        "count": int(prev.get("count") or prev.get("occurrence_count") or 0) + 1,
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


def _procedural_memory_library_class(repo_root: Path) -> str:
    """Coarse build-system label used as procedural-memory scope."""
    for rel in ("fuzz/build_strategy.json", "fuzz/build_runtime_facts.json", "fuzz/repo_understanding.json"):
        try:
            p = repo_root / rel
            if not p.is_file():
                continue
            doc = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            bs = str((doc or {}).get("build_system") or "").strip().lower()
            if bs and bs != "unknown":
                return bs
        except Exception:
            continue
    return ""


def _procedural_memory_system_packages_nonempty(repo_root: Path) -> bool:
    try:
        p = repo_root / "fuzz" / "system_packages.txt"
        if not p.is_file():
            return False
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.split("#", 1)[0].strip():
                return True
    except Exception:
        return False
    return False


def _record_procedural_memory(state: dict[str, Any], repair_snapshot: dict[str, Any]) -> None:
    """Reflexion-style write path: when a stage fails into repair, distill a
    known failure class into a cross-job lesson. No-op unless
    SHERPA_PROCEDURAL_MEMORY is enabled; never raises. (Phase 2: accumulate
    only — injection into prompts is gated for Phase 3.)"""
    if not _proc_mem.memory_enabled():
        return
    repo_root_text = str(state.get("repo_root") or "").strip()
    if not repo_root_text:
        return
    try:
        repo_root = Path(repo_root_text)
        stage = str(repair_snapshot.get("repair_origin_stage") or "").strip()
        if not stage:
            return
        lesson = _proc_mem.classify_stage_failure(
            stage=stage,
            error_code=str(repair_snapshot.get("repair_error_code") or ""),
            error_kind=str(repair_snapshot.get("repair_error_kind") or ""),
            diagnostics=" ".join(
                [
                    str(repair_snapshot.get("repair_error_text") or ""),
                    str(repair_snapshot.get("repair_stderr_tail") or ""),
                ]
            )[:4000],
            system_packages_nonempty=_procedural_memory_system_packages_nonempty(repo_root),
            api_surface_exception_used=bool(state.get("api_surface_exception_used")),
            library_class=_procedural_memory_library_class(repo_root),
        )
        if lesson:
            _proc_mem.record_lesson(job_id=str(os.environ.get("SHERPA_JOB_ID") or ""), **lesson)
    except Exception:
        return


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
    dedup_count = int(
        constraint_entry.get("count")
        or constraint_entry.get("occurrence_count")
        or 0
    )
    if dedup_count >= 2:
        snapshot["repair_strategy_force_change"] = True
        snapshot["crash_signature_dedup_hit"] = True
    _record_procedural_memory(state, snapshot)
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
    return _wf_norm.infer_seed_profile(
        name,
        context,
        target_type=str(target_type or "").strip().lower(),
    )


def _normalize_seed_profile(
    seed_profile: str,
    *,
    target_type: str,
    name: str,
    context: str,
) -> str:
    return _wf_norm.normalize_seed_profile(
        seed_profile,
        target_type=str(target_type or "").strip().lower(),
        name=name,
        context=context,
    )


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
    if _is_test_or_demo_helper_target(name=name, api=context, file_hint=file_hint):
        score -= 8
        reasons.append("test-demo-helper")
        if "png" in text:
            replacements.extend(["png_read_image", "png_process_data", "png_read_info"])
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


def _is_test_or_demo_helper_target(*, name: str, api: str, file_hint: str = "") -> bool:
    text = f"{name}\n{api}\n{file_hint}".lower()
    basename = Path(str(file_hint or "")).name.lower()
    symbol = str(name or api or "").strip().lower()
    public_like_symbol = symbol.startswith(("png_", "xml", "yaml_", "json_", "sqlite3_"))
    helper_name = (
        str(name or "").lower().startswith(("test_", "do_test"))
        or str(api or "").lower().startswith(("test_", "do_test"))
        or str(name or "").lower() in {"testonefile", "test_one_file"}
        or str(api or "").lower() in {"testonefile", "test_one_file"}
    )
    test_file = any(
        token in text
        for token in (
            "/contrib/libtests/",
            "contrib/libtests/",
            "/tests/",
            "tests/",
            "/test/",
            "test/",
            "/contrib/gregbook/",
            "contrib/gregbook/",
            "/contrib/examples/",
            "contrib/examples/",
            "/examples/",
            "examples/",
            "/demo/",
            "demo/",
            "/demos/",
            "demos/",
            "/deprecated/",
            "deprecated/",
            "/legacy/",
            "legacy/",
            "examples/",
        )
    )
    demo_file = any(token in basename for token in ("test", "demo", "example", "deprecated", "legacy"))
    return bool((helper_name and (test_file or demo_file)) or ((test_file or demo_file) and not public_like_symbol))


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


def _enrich_targets_depth(repo_root: Path) -> None:
    """Back-fill depth_score / depth_class on every target in targets.json.

    OpenCode often omits these fields.  Without them all targets look equal
    and _select_primary_target cannot prefer deeper ones on replan.
    """
    targets_path = repo_root / "fuzz" / "targets.json"
    if not targets_path.is_file():
        return
    try:
        data = json.loads(targets_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return
    if not isinstance(data, list) or not data:
        return
    changed = False
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("depth_score") and item.get("depth_class"):
            continue
        name = str(item.get("name") or "")
        desc = str(item.get("description") or "")
        ttype = str(item.get("target_type") or "")
        score, depth_class, reason = _score_target_depth(
            name, desc, target_type=ttype,
        )
        item["depth_score"] = score
        item["depth_class"] = depth_class
        item["selection_bias_reason"] = reason
        changed = True
    if changed:
        try:
            targets_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass


def _select_primary_target(
    repo_root: Path,
    *,
    exclude_names: list[str] | None = None,
    prefer_deeper: bool = False,
) -> dict[str, Any]:
    targets = _load_targets_doc(repo_root)
    if not targets:
        return {}
    candidates = targets
    if exclude_names:
        filtered = [t for t in candidates if t.get("name") not in set(exclude_names)]
        if filtered:
            candidates = filtered
    if prefer_deeper:
        _depth_order = {"deep": 0, "medium": 1, "shallow": 2}
        candidates = sorted(
            candidates,
            key=lambda t: _depth_order.get(
                str(t.get("depth_class", "shallow")).lower(), 2
            ),
        )
    return dict(candidates[0])


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
        "coverage_gap": 0.30,
        "complexity": 0.30,
        "api_relevance": 0.25,
        "consumer_order_support": 0.15,
    }


def _clamp_score(value: float, *, lo: float = 0.0, hi: float = 10.0) -> float:
    return _wf_target_scoring.clamp_score(value, lo=lo, hi=hi)


def _target_component_coverage_gap(item: dict[str, Any]) -> float:
    return _wf_target_scoring.target_component_coverage_gap(item)


def _target_component_complexity(item: dict[str, Any]) -> float:
    return _wf_target_scoring.target_component_complexity(item)


def _target_component_api_relevance(item: dict[str, Any]) -> float:
    return _wf_target_scoring.target_component_api_relevance(
        item,
        runtime_viability_rank_fn=_runtime_viability_rank,
    )


def _target_component_consumer_order_support(item: dict[str, Any]) -> float:
    return _wf_target_scoring.target_component_consumer_order_support(item)


def _target_score_breakdown(item: dict[str, Any]) -> dict[str, Any]:
    return _wf_target_scoring.target_score_breakdown(
        item,
        weights=_target_scoring_weights(),
        runtime_viability_rank_fn=_runtime_viability_rank,
    )


def _load_seed_feedback_by_fuzzer(repo_root: Path) -> dict[str, dict[str, Any]]:
    path = repo_root / "fuzz" / "seed_feedback.json"
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    by_fuzzer = raw.get("by_fuzzer") if isinstance(raw, dict) else {}
    if not isinstance(by_fuzzer, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in by_fuzzer.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        out[key] = dict(value)
    return out


def _load_target_runtime_cooldown_index(repo_root: Path) -> dict[str, Any]:
    context_dir = context_dir_for_repo_root(repo_root)
    control_doc, workflow_doc = read_context_docs(context_dir, job_id="")
    control = strip_meta(control_doc)
    workflow = strip_meta(workflow_doc)
    exhausted_targets: set[str] = set()
    raw_exhausted = list(workflow.get("coverage_exhausted_targets") or [])
    for item in raw_exhausted:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
        else:
            name = str(item or "").strip()
        if name:
            exhausted_targets.update(_target_analysis_lookup_keys(name, name))

    low_yield_targets: set[str] = set()
    plateau_counts: dict[str, int] = {}
    for entry in list(workflow.get("coverage_history") or [])[-8:]:
        if not isinstance(entry, dict):
            continue
        target_api = str(entry.get("target_api") or entry.get("target_name") or "").strip()
        if not target_api:
            continue
        plateau_no_gain = bool(
            entry.get("plateau_detected")
            and int(entry.get("max_cov") or 0) <= int(entry.get("prev_cov") or 0)
            and int(entry.get("max_ft") or 0) <= int(entry.get("prev_ft") or 0)
        )
        if plateau_no_gain:
            plateau_counts[target_api] = plateau_counts.get(target_api, 0) + 1
    for target_api, count in plateau_counts.items():
        if count >= 2:
            low_yield_targets.update(_target_analysis_lookup_keys(target_api, target_api))

    oom_fuzzers: set[str] = set()
    timeout_fuzzers: set[str] = set()
    last_fuzzer = str(workflow.get("last_fuzzer") or control.get("last_fuzzer") or "").strip()
    run_error_kind = str(
        workflow.get("coverage_run_error_kind_effective")
        or workflow.get("run_error_kind")
        or workflow.get("error_code")
        or ""
    ).strip().lower()
    if last_fuzzer:
        if run_error_kind == "oom_killed":
            oom_fuzzers.add(last_fuzzer)
        elif run_error_kind in {"k8s_job_timeout", "timeout", "run_timeout", "run_idle_timeout"}:
            timeout_fuzzers.add(last_fuzzer)

    return {
        "exhausted_targets": exhausted_targets,
        "low_yield_targets": low_yield_targets,
        "oom_fuzzers": oom_fuzzers,
        "timeout_fuzzers": timeout_fuzzers,
    }


def _target_runtime_penalty(
    repo_root: Path,
    wrapper_fuzzer_name: str,
    *,
    target_name: str = "",
    api: str = "",
) -> dict[str, Any]:
    feedback = _load_seed_feedback_by_fuzzer(repo_root).get(wrapper_fuzzer_name) or {}
    base = _wf_target_scoring.runtime_penalty_from_feedback(feedback)
    penalty = float(base.get("score_penalty") or 0.0)
    reasons: list[str] = []
    if str(base.get("reason") or "").strip():
        reasons.append(str(base.get("reason") or "").strip())

    cooldown = _load_target_runtime_cooldown_index(repo_root)
    lookup_keys = _target_analysis_lookup_keys(target_name or wrapper_fuzzer_name, api or target_name or wrapper_fuzzer_name)
    if lookup_keys & set(cooldown.get("exhausted_targets") or set()):
        penalty += 1.2
        reasons.append("coverage_exhausted_target")
    if lookup_keys & set(cooldown.get("low_yield_targets") or set()):
        penalty += 0.9
        reasons.append("persistent_low_yield_target")
    if wrapper_fuzzer_name and wrapper_fuzzer_name in set(cooldown.get("oom_fuzzers") or set()):
        penalty += 2.5
        reasons.append("recent_oom_killed")
    elif wrapper_fuzzer_name and wrapper_fuzzer_name in set(cooldown.get("timeout_fuzzers") or set()):
        penalty += 1.4
        reasons.append("recent_timeout")

    return {
        "score_penalty": round(min(max(penalty, 0.0), 4.5), 4),
        "reason": ";".join(dict.fromkeys(x for x in reasons if x)),
        "seed_feedback": dict(feedback),
    }


def _selection_target_key(name: str) -> str:
    return str(name or "").strip().lower()


_ENTRYPOINT_HINTS_SUFFIX = (
    "_parse", "_parse_file", "_parse_string", "_parse_buffer",
    "_loads", "_load", "_load_file", "_decode", "_deserialize", "_unmarshal",
    "_read_file", "_read_buffer", "_from_string", "_from_buffer", "_fromjson",
    # whole-input decoder read-entrypoints (e.g. libpng png_read_image /
    # png_read_png / png_read_info). Kept specific so low-level readers
    # (png_read_filter_row -> _row, png_read_data -> _data, *_chunk/_byte) are
    # excluded as leaves rather than promoted.
    "_read_image", "_read_png", "_read_info", "_read_document", "_read_memory",
    "_read_stream", "_readimage", "_readfile",
)


_ENTRYPOINT_HINTS_EXACT = {
    "parse", "loads", "load", "decode", "deserialize", "unmarshal",
    "parse_file", "parse_string", "parse_buffer",
}


_ENTRYPOINT_LEAF_HINTS = (
    "scan_", "lex", "next_token", "nexttoken", "next_char", "_digit", "_char",
    "_byte", "peek", "advance", "getc", "ungetc",
)


def _vuln_entrypoint_bias_weight() -> float:
    raw = os.environ.get("SHERPA_VULN_ENTRYPOINT_BIAS")
    if raw is None or str(raw).strip() == "":
        return 0.15
    try:
        return max(0.0, min(float(raw), 0.5))
    except Exception:
        return 0.15


def _is_library_entrypoint(api: str) -> bool:
    name = str(api or "").strip().lower().split("::")[-1].split(".")[-1]
    if not name:
        return False
    if any(h in name for h in _ENTRYPOINT_LEAF_HINTS):
        return False
    if name in _ENTRYPOINT_HINTS_EXACT:
        return True
    return any(name.endswith(s) for s in _ENTRYPOINT_HINTS_SUFFIX)


_NON_HARNESSABLE_EXACT = {
    "malloc", "calloc", "realloc", "free", "memcpy", "memmove", "memset",
    "strdup", "strndup", "xmalloc", "xcalloc", "xrealloc", "xfree",
}


_NON_HARNESSABLE_TOKENS = (
    "_alloc", "alloc_", "_free", "_realloc", "_ptrarr", "_grow", "expand_",
    "_resize", "_reserve", "_xmalloc", "_memdup", "_strdup",
)


def _is_non_harnessable_target(api: str) -> bool:
    """True for allocator/memory-infra helpers and alloc macros that should not
    be selected as standalone fuzz targets. Conservative: only matches clear
    allocation/growth infrastructure, not parse/decode logic."""
    name = str(api or "").strip().lower().split("::")[-1].split(".")[-1]
    if not name:
        return False
    # ALL-CAPS macros like CALLOC / MALLOC / REALLOC / FREE
    bare = str(api or "").strip().split("::")[-1].split(".")[-1]
    if bare.isupper() and any(k in name for k in ("alloc", "free", "realloc")):
        return True
    if name in _NON_HARNESSABLE_EXACT:
        return True
    return any(tok in name for tok in _NON_HARNESSABLE_TOKENS)


def _library_entrypoint_bias(api: str, target_type: str) -> float:
    """Positive risk bias for whole-input library entrypoints (0 otherwise).
    Strongest for parser/decoder/archive libraries. Disable with
    SHERPA_VULN_ENTRYPOINT_BIAS=0.

    This is the NAME heuristic — used as a fallback when the call graph is
    unavailable (see _entrypoint_risk_bias for the primary structural signal)."""
    if not _is_library_entrypoint(api):
        return 0.0
    weight = _vuln_entrypoint_bias_weight()
    if weight <= 0.0:
        return 0.0
    if str(target_type or "").strip().lower() in {"parser", "decoder", "archive", "deserializer"}:
        return round(weight, 4)
    return round(weight * 0.6, 4)


def _selection_mode() -> str:
    """Target-selection mode (hybrid experiment).

    - "score" (default): deterministic value arithmetic re-ranks candidates
      (effective_risk = vuln_likelihood - penalties + entrypoint_bias).
    - "llm_first": trust the agent's own risk judgement — order by the LLM
      dimensions only, keep the feedback-gating + hard guardrail-drop pillars,
      drop the value arithmetic. Lets us A/B whether the scorer helps or fights
      the LLM."""
    mode = (os.environ.get("SHERPA_SELECTION_MODE") or "score").strip().lower()
    return "llm_first" if mode in {"llm_first", "llm-first", "llm"} else "score"


def _coverage_potential_enabled() -> bool:
    raw = (os.environ.get("SHERPA_VULN_COVERAGE_POTENTIAL") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _coverage_potential_weight() -> float:
    raw = os.environ.get("SHERPA_VULN_COVERAGE_POTENTIAL_WEIGHT")
    if raw is None or str(raw).strip() == "":
        return 0.25
    try:
        return max(0.0, min(float(raw), 0.5))
    except Exception:
        return 0.25


def _entrypoint_risk_bias(api: str, target_type: str, repo_root: Path) -> tuple[float, float]:
    """Combined entrypoint risk bias + the raw coverage_potential signal.

    Primary signal is structural: a target's call-graph reachable fan-out
    ('coverage_potential' in [0,1]) — a whole-library entrypoint drives much more
    reachable bug surface than an isolated leaf. The name heuristic
    (_library_entrypoint_bias) is the fallback used when the call graph is
    empty/degraded. We take the MAX of the two (never sum) so neither
    double-counts. Returns (bias, coverage_potential)."""
    name_bias = _library_entrypoint_bias(api, target_type)
    cov_pot = 0.0
    if _coverage_potential_enabled():
        try:
            cov_pot = coverage_potential(api, _load_callgraph_forward(repo_root))
        except Exception:
            cov_pot = 0.0
    structural_bias = round(_coverage_potential_weight() * cov_pot, 4)
    return max(structural_bias, name_bias), round(cov_pot, 4)


def _execution_depth_bias(
    *,
    target_name: str,
    api: str,
    target_type: str,
    depth_class: str,
    depth_score: int,
    selection_rationale: str,
) -> dict[str, Any]:
    text = " ".join(
        [
            str(target_name or ""),
            str(api or ""),
            str(selection_rationale or ""),
        ]
    ).lower()
    callback_penalty = 0.0
    wrapper_penalty = 0.0
    if any(token in text for token in ("callback", "init", "cleanup", "helper", "check_sig", "sig")):
        callback_penalty = 0.12
    if any(token in text for token in ("helper", "wrapper", "adapter")):
        wrapper_penalty = max(wrapper_penalty, 0.08)
    deep_bonus = 0.0
    if any(
        token in text
        for token in ("decode", "parse", "read", "load", "stream", "row", "image", "chunk", "process", "transform")
    ):
        deep_bonus += 0.08
    if str(target_type or "").strip().lower() in {"parser", "decoder", "archive"}:
        deep_bonus += 0.04
    depth_class_l = str(depth_class or "").strip().lower()
    if depth_class_l == "deep":
        deep_bonus += 0.05
    elif depth_class_l == "medium":
        deep_bonus += 0.02
    if int(depth_score or 0) >= 18:
        deep_bonus += 0.03
    execution_bias = round(deep_bonus - callback_penalty - wrapper_penalty, 4)
    return {
        "execution_depth_bias": execution_bias,
        "callback_penalty": round(callback_penalty, 4),
        "wrapper_penalty": round(wrapper_penalty, 4),
    }


def _target_surface_penalty(
    *,
    target_name: str,
    api: str,
    source_path: str,
    runtime_replacement_reason: str = "",
) -> dict[str, Any]:
    if str(runtime_replacement_reason or "").strip() == "test_demo_helper_public_surrogate":
        return {"target_surface_penalty": 0.0, "target_surface_penalty_reason": ""}
    normalized_path = str(source_path or "").replace("\\", "/").strip().lower()
    basename = Path(normalized_path).name.lower()
    symbol_text = f"{target_name} {api}".strip().lower()
    penalty = 0.0
    reasons: list[str] = []
    non_core_tokens = (
        "contrib/arm",
        "contrib/intel",
        "contrib/mips",
        "contrib/powerpc",
        "contrib/riscv",
        "contrib/loongarch",
        "/arm-neon/",
        "arm-neon/",
        "linux-auxv",
        "/auxv",
    )
    if any(token in normalized_path for token in non_core_tokens):
        penalty += 0.45
        reasons.append("non_core_auxiliary_source")
    elif normalized_path.startswith("contrib/") and not any(
        token in normalized_path
        for token in ("contrib/oss-fuzz/", "contrib/libtests/", "contrib/examples/", "contrib/gregbook/")
    ):
        penalty += 0.25
        reasons.append("contrib_auxiliary_source")
    helper_tokens = ("safe_read", "auxv", "helper", "wrapper", "callback", "platform", "cpuinfo")
    if any(token in symbol_text for token in helper_tokens) or any(token in basename for token in helper_tokens):
        penalty += 0.12
        reasons.append("helper_surface")
    return {
        "target_surface_penalty": round(min(max(penalty, 0.0), 0.75), 4),
        "target_surface_penalty_reason": ";".join(dict.fromkeys(reasons)),
    }


def _apply_selected_target_filters(
    ranked_items: list[dict[str, Any]],
    *,
    exclude_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows = list(ranked_items)
    # Hard-drop structurally-unlinkable language bindings that the selection
    # gate flagged (wasm/napi/jni shims with no public caller). Keep at least
    # one row as a safety net so we never hand an empty selection downstream.
    if any(row.get("unlinkable_binding_dropped") for row in rows):
        linkable = [row for row in rows if not row.get("unlinkable_binding_dropped")]
        rows = linkable or rows
    # Hard-drop allocator/infra helpers (CALLOC macro, expand_ptrarr, ...) that
    # cannot be harnessed standalone — selecting them as must_run only produces
    # execution_plan_harness_mismatch and churns the loop.
    if any(row.get("non_harnessable_dropped") for row in rows):
        harnessable = [row for row in rows if not row.get("non_harnessable_dropped")]
        rows = harnessable or rows
    excluded = {
        _selection_target_key(name)
        for name in list(exclude_names or [])
        if _selection_target_key(name)
    }
    if not excluded:
        return rows
    filtered = [
        row
        for row in rows
        if _selection_target_key(
            str(row.get("target_name") or row.get("target") or row.get("name") or "")
        )
        not in excluded
    ]
    return filtered or rows


def _target_analysis_lookup_keys(target_name: str, api: str) -> set[str]:
    keys: set[str] = set()
    for raw in (target_name, api):
        norm = _normalize_exec_target_token(raw)
        if norm:
            keys.add(norm)
        if raw:
            tail = str(raw).split("::")[-1].split(".")[-1].strip()
            norm_tail = _normalize_exec_target_token(tail)
            if norm_tail:
                keys.add(norm_tail)
    return keys


def _targets_material_signature(targets_text: str) -> tuple[tuple[str, str, str, str, str], ...] | None:
    """
    Build a semantic signature from strict required target keys.
    This avoids false replan "changes" caused only by auto-enriched metadata
    (e.g. depth_score/selection_bias_reason) or JSON formatting differences.
    """
    try:
        parsed = json.loads(targets_text or "[]")
    except Exception:
        return None
    if not isinstance(parsed, list):
        return None
    sig: list[tuple[str, str, str, str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        sig.append(
            (
                str(item.get("name") or "").strip(),
                str(item.get("api") or "").strip(),
                str(item.get("lang") or "").strip(),
                str(item.get("target_type") or "").strip(),
                str(item.get("seed_profile") or "").strip(),
            )
        )
    return tuple(sig)


def _load_target_analysis_security_index(repo_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    target_path = repo_root / "fuzz" / "target_analysis.json"
    try:
        target_doc = json.loads(target_path.read_text(encoding="utf-8", errors="replace")) if target_path.is_file() else {}
    except Exception:
        target_doc = {}
    for item in list((target_doc.get("recommended_targets") if isinstance(target_doc, dict) else []) or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        api = str(item.get("api") or name).strip()
        for key in _target_analysis_lookup_keys(name, api):
            out.setdefault(key, dict(item))

    analysis_path = repo_root / "fuzz" / "analysis_context.json"
    try:
        analysis_doc = json.loads(analysis_path.read_text(encoding="utf-8", errors="replace")) if analysis_path.is_file() else {}
    except Exception:
        analysis_doc = {}
    analysis_evidence = dict((analysis_doc.get("analysis_evidence") if isinstance(analysis_doc, dict) else {}) or {})
    for item in list(analysis_evidence.get("vuln_candidate_inventory") or []):
        if not isinstance(item, dict):
            continue
        api = str(item.get("api") or "").strip()
        name = str(item.get("name") or api).strip()
        for key in _target_analysis_lookup_keys(name, api):
            merged = dict(out.get(key) or {})
            merged.update(item)
            out[key] = merged
    vuln_doc = _load_vuln_candidates_doc(repo_root)
    for item in list(vuln_doc.get("candidates") or []):
        if not isinstance(item, dict):
            continue
        status = str(item.get("validation_status") or "").strip().lower()
        if status in {"exhausted", "cooling"}:
            continue
        api = str(item.get("target_api") or item.get("api") or "").strip()
        name = str(item.get("target_name") or item.get("name") or api).strip()
        if not (api or name):
            continue
        normalized = dict(item)
        normalized.setdefault("api", api)
        normalized.setdefault("name", name)
        normalized.setdefault("file", str(item.get("target_file") or item.get("file") or ""))
        normalized.setdefault("candidate_origin", str(item.get("candidate_origin") or item.get("source_stage") or "vuln_candidates"))
        for key in _target_analysis_lookup_keys(name, api):
            merged = dict(out.get(key) or {})
            merged.update(normalized)
            out[key] = merged
    return out


def _load_security_evidence_list(
    repo_root: Path,
    analysis_context_path: str,
) -> tuple[list[dict[str, Any]], str]:
    """
    Load security evidence from analysis_context using a strict list-only contract.

    Contract:
      analysis_context.json.analysis_evidence.security_evidence must be list[object].
    Any non-list schema returns empty evidence with a structured issue code.
    """
    path_text = str(analysis_context_path or "").strip()
    if not path_text:
        return [], ""
    ctx_path = Path(path_text)
    if not ctx_path.is_absolute():
        ctx_path = repo_root / ctx_path
    if not ctx_path.is_file():
        return [], ""
    try:
        raw_doc = json.loads(ctx_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return [], f"security_evidence_load_error:{exc}"
    if not isinstance(raw_doc, dict):
        return [], "security_evidence_schema_invalid:analysis_context_not_object"
    analysis_evidence = raw_doc.get("analysis_evidence")
    if analysis_evidence is None:
        return [], ""
    if not isinstance(analysis_evidence, dict):
        return [], "security_evidence_schema_invalid:analysis_evidence_not_object"
    security_evidence = analysis_evidence.get("security_evidence")
    if security_evidence is None:
        return [], ""
    if not isinstance(security_evidence, list):
        return [], "security_evidence_schema_invalid:security_evidence_not_list"
    normalized: list[dict[str, Any]] = []
    for item in security_evidence:
        if isinstance(item, dict):
            normalized.append(dict(item))
    return normalized, ""


def _load_vuln_candidate_inventory(
    repo_root: Path,
    analysis_context_path: str,
) -> tuple[list[dict[str, Any]], str]:
    """
    Load vulnerability candidates from analysis_context using a strict list-only contract.

    Contract:
      analysis_context.json.analysis_evidence.vuln_candidate_inventory must be list[object].
    """
    path_text = str(analysis_context_path or "").strip()
    if not path_text:
        return [], ""
    ctx_path = Path(path_text)
    if not ctx_path.is_absolute():
        ctx_path = repo_root / ctx_path
    if not ctx_path.is_file():
        return [], ""
    try:
        raw_doc = json.loads(ctx_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return [], f"vuln_candidate_inventory_load_error:{exc}"
    if not isinstance(raw_doc, dict):
        return [], "vuln_candidate_inventory_schema_invalid:analysis_context_not_object"
    analysis_evidence = raw_doc.get("analysis_evidence")
    if analysis_evidence is None:
        return [], ""
    if not isinstance(analysis_evidence, dict):
        return [], "vuln_candidate_inventory_schema_invalid:analysis_evidence_not_object"
    inventory = analysis_evidence.get("vuln_candidate_inventory")
    if inventory is None:
        return [], ""
    if not isinstance(inventory, list):
        return [], "vuln_candidate_inventory_schema_invalid:vuln_candidate_inventory_not_list"
    normalized: list[dict[str, Any]] = []
    for item in inventory:
        if isinstance(item, dict):
            normalized.append(dict(item))
    return normalized, ""


def _vuln_candidates_path(repo_root: Path) -> Path:
    return repo_root / "fuzz" / "vuln_candidates.json"


def _load_vuln_candidates_doc(repo_root: Path) -> dict[str, Any]:
    path = _vuln_candidates_path(repo_root)
    if not path.is_file():
        return {"schema_version": 1, "updated_at": 0, "candidate_count": 0, "candidates": []}
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {"schema_version": 1, "updated_at": 0, "candidate_count": 0, "candidates": []}
    if not isinstance(raw, dict):
        return {"schema_version": 1, "updated_at": 0, "candidate_count": 0, "candidates": []}
    candidates = [dict(x) for x in list(raw.get("candidates") or []) if isinstance(x, dict)]
    return {
        "schema_version": int(raw.get("schema_version") or 1),
        "updated_at": int(raw.get("updated_at") or 0),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _active_vuln_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocked = {"exhausted", "cooling"}
    return [
        dict(item)
        for item in candidates
        if str(item.get("validation_status") or "pending").strip().lower() not in blocked
    ]


def _write_vuln_candidates_doc(repo_root: Path, candidates: list[dict[str, Any]], *, degraded_reason: str = "") -> str:
    path = _vuln_candidates_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": 1,
        "updated_at": int(time.time()),
        "candidate_count": len(candidates),
        "degraded_reason": degraded_reason,
        "candidates": candidates,
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def _vuln_candidate_matches_feedback(candidate: dict[str, Any], event: dict[str, Any], active_candidate_id: str) -> bool:
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    if active_candidate_id and candidate_id == active_candidate_id:
        return True
    event_api = str(event.get("target_api") or "").strip()
    event_name = str(event.get("target_name") or "").strip()
    values = {
        str(candidate.get("target_api") or "").strip(),
        str(candidate.get("api") or "").strip(),
        str(candidate.get("target_name") or "").strip(),
        str(candidate.get("name") or "").strip(),
    }
    return bool((event_api and event_api in values) or (event_name and event_name in values))


def _feedback_status_for_vuln_candidate(event: dict[str, Any], *, plateau_streak: int) -> str:
    event_type = str(event.get("event_type") or "").strip()
    if event_type == "coverage_plateau":
        return "exhausted" if plateau_streak >= _vuln_max_iterations_per_candidate() else "cooling"
    if event_type in {"seed_generation_degraded", "harness_bug", "repair_feedback"}:
        return "inconclusive"
    return "pending"


def _update_vuln_candidate_feedback(repo_root: Path, state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    if not event:
        return {}
    doc = _load_vuln_candidates_doc(repo_root)
    candidates = [dict(x) for x in list(doc.get("candidates") or []) if isinstance(x, dict)]
    if not candidates:
        return {}
    active_candidate_id = str(state.get("vuln_hunt_active_candidate_id") or "").strip()
    plateau_streak = int(state.get("coverage_plateau_streak") or event.get("coverage_plateau_streak") or 0)
    changed = False
    updated: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        if _vuln_candidate_matches_feedback(item, event, active_candidate_id):
            attempts = int(item.get("attempt_count") or 0) + 1
            status = _feedback_status_for_vuln_candidate(event, plateau_streak=plateau_streak)
            item.update(
                {
                    "attempt_count": attempts,
                    "validation_status": status,
                    "last_result": {
                        "ts": int(event.get("ts") or time.time()),
                        "event_type": str(event.get("event_type") or ""),
                        "target_name": str(event.get("target_name") or ""),
                        "target_api": str(event.get("target_api") or ""),
                        "coverage_plateau_streak": plateau_streak,
                        "coverage_quality_flags": list(event.get("coverage_quality_flags") or []),
                        "coverage_seed_generation_degraded": bool(
                            event.get("coverage_seed_generation_degraded") or False
                        ),
                        "crash_triage_label": str(event.get("crash_triage_label") or ""),
                    },
                    "updated_at": int(time.time()),
                }
            )
            changed = True
        updated.append(item)
    if not changed:
        return {}
    path = _write_vuln_candidates_doc(repo_root, updated)
    active = _active_vuln_candidates(updated)
    active_candidate = dict(active[0]) if active else {}
    return {
        "vuln_candidates_path": path,
        "vuln_candidate_count": len(updated),
        "vuln_hunt_candidate_count": len(updated),
        "vuln_hunt_active_candidate_id": str(active_candidate.get("candidate_id") or ""),
        "vuln_hunt_rerun_requested": True,
    }


def _vuln_candidate_id(value: str, idx: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip()).strip("_").lower()
    return f"analysis_{slug or 'candidate'}_{idx + 1}"


def _normalize_analysis_vuln_candidate(
    item: dict[str, Any],
    *,
    idx: int,
    evidence_by_id: dict[str, dict[str, Any]],
    public_set: frozenset[str] | None = None,
    api_loc_index: dict[str, tuple[str, int]] | None = None,
) -> dict[str, Any]:
    api = str(item.get("api") or item.get("target_api") or item.get("name") or "").strip()
    name = str(item.get("name") or item.get("target_name") or api).strip()
    target_type = str(item.get("target_type") or "generic").strip().lower()
    evidence_ids = [
        str(x).strip()
        for x in list(item.get("evidence_ids") or [])
        if str(x).strip()
    ]
    evidence_refs = [dict(evidence_by_id[x]) for x in evidence_ids if x in evidence_by_id]
    signal_type = str(item.get("signal_type") or item.get("signal_id") or "").strip()
    if not signal_type and evidence_refs:
        signal_type = str(evidence_refs[0].get("signal_id") or "").strip()
    if not signal_type:
        signal_type = "mem_oob_candidate"
    vuln_likelihood = _bounded_float(item.get("vuln_likelihood"), 0.0)
    exploitability = _bounded_float(item.get("exploitability"), 0.0)
    reachability = _bounded_float(item.get("reachability_confidence"), 0.0)
    signal_score = max(_bounded_float(item.get("signal_score"), 0.0), vuln_likelihood)
    reason = str(item.get("security_priority_reason") or item.get("summary") or "").strip()
    source_path = str(item.get("file") or item.get("target_file") or item.get("source_path") or "").strip()
    try:
        line = int(item.get("line") or 0)
    except Exception:
        line = 0

    # Classify the candidate against the public-API surface. Non-public
    # symbols (WASM/binding shims, statics, internal-only) keep their evidence
    # but are reachability-capped so public candidates rank ahead of them; the
    # high-vuln-likelihood escape hatch still applies later at target selection.
    public_set = public_set if public_set is not None else frozenset()
    api_surface = _api_surface_class(api, public_set)
    if api_surface == "internal":
        cap = _vuln_non_public_reachability_cap()
        if reachability > cap:
            reachability = cap

    # Backfill source location from the public-API index when the candidate
    # arrived without a concrete line (line == 0) but names a known symbol.
    if (line <= 0 or not source_path) and api_loc_index:
        loc = api_loc_index.get(api)
        if loc:
            loc_path, loc_line = loc
            if not source_path and loc_path:
                source_path = loc_path
            if line <= 0 and loc_line > 0:
                line = loc_line
    attack_hint = dict(item.get("attack_hint") or {})
    if not attack_hint:
        attack_hint = _candidate_attack_hint(
            api=api,
            target_type=target_type,
            signal_id=signal_type,
            source_path=source_path,
            security_reason=reason,
        )
    candidate_id = str(item.get("candidate_id") or "").strip() or _vuln_candidate_id(api or name, idx)
    risk_signal_source_breakdown = dict(item.get("risk_signal_source_breakdown") or {})
    security_signal_scores = dict(item.get("security_signal_scores") or {})
    return {
        "candidate_id": candidate_id,
        "source_stage": "analysis",
        "candidate_origin": "analysis_context",
        "validation_status": str(item.get("validation_status") or "pending"),
        "target_api": api,
        "api": api,
        "target_name": name,
        "name": name,
        "target_file": source_path,
        "file": source_path,
        "source_path": source_path,
        "line": line,
        "target_type": target_type,
        "signal_type": signal_type,
        "risk_type": signal_type,
        "api_surface": api_surface,
        "signal_score": round(max(0.0, min(signal_score, 1.0)), 4),
        "vuln_likelihood": round(vuln_likelihood, 4),
        "exploitability": round(exploitability, 4),
        "reachability_confidence": round(reachability, 4),
        "detectability_confidence": round(
            max(0.0, min(max(reachability, signal_score, vuln_likelihood * 0.8), 1.0)),
            4,
        ),
        "priority": _candidate_priority(
            vuln_likelihood=vuln_likelihood,
            exploitability=exploitability,
            reachability_confidence=reachability,
            evidence_count=len(evidence_ids),
            signal_score=signal_score,
        ),
        "security_priority_reason": reason,
        "evidence_ids": evidence_ids,
        "evidence": evidence_refs,
        "attack_hint": attack_hint,
        "security_signal_scores": {k: float(v) for k, v in security_signal_scores.items()},
        "risk_signal_source_breakdown": risk_signal_source_breakdown,
        "attempt_count": int(item.get("attempt_count") or 0),
        "last_result": dict(item.get("last_result") or {}),
        "created_at": int(item.get("created_at") or time.time()),
        "updated_at": int(time.time()),
    }


def _write_analysis_vuln_candidates(repo_root: Path, analysis_context_path: str) -> dict[str, Any]:
    evidence, evidence_issue = _load_security_evidence_list(repo_root, analysis_context_path)
    inventory, inventory_issue = _load_vuln_candidate_inventory(repo_root, analysis_context_path)
    path = _vuln_candidates_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_doc = _load_vuln_candidates_doc(repo_root)
    evidence_by_id = {
        str(item.get("evidence_id") or "").strip(): dict(item)
        for item in evidence
        if str(item.get("evidence_id") or "").strip()
    }
    existing_candidates = [
        dict(x)
        for x in list(existing_doc.get("candidates") or [])
        if isinstance(x, dict) and str(x.get("source_stage") or "") != "analysis"
    ]
    topk = max(1, int(_vuln_topk()))
    public_set = _load_public_api_symbols(repo_root)
    api_loc_index = _load_api_loc_index(repo_root)
    analysis_candidates = [
        _normalize_analysis_vuln_candidate(
            item,
            idx=idx,
            evidence_by_id=evidence_by_id,
            public_set=public_set,
            api_loc_index=api_loc_index,
        )
        for idx, item in enumerate(inventory[:topk])
        if isinstance(item, dict)
    ]
    by_id: dict[str, dict[str, Any]] = {}
    for candidate in existing_candidates + analysis_candidates:
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if candidate_id:
            by_id[candidate_id] = candidate
    candidates = sorted(
        by_id.values(),
        key=lambda x: (
            -float(x.get("priority") or 0.0),
            -float(x.get("vuln_likelihood") or 0.0),
            str(x.get("candidate_id") or ""),
        ),
    )
    issue = ";".join(x for x in (evidence_issue, inventory_issue) if x)
    doc = {
        "schema_version": 1,
        "updated_at": int(time.time()),
        "candidate_count": len(candidates),
        "analysis_candidate_count": len(analysis_candidates),
        "degraded_reason": issue,
        "candidates": candidates,
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"path": str(path), "candidate_count": len(candidates), "analysis_candidate_count": len(analysis_candidates), "issue": issue}


def _vuln_hunt_summary_path(repo_root: Path) -> Path:
    return repo_root / "fuzz" / "vuln_hunt_summary.md"


def _vuln_hunt_events_path(repo_root: Path) -> Path:
    return repo_root / "fuzz" / "vuln_hunt_events.jsonl"


def _vuln_hunt_event_from_state(state: dict[str, Any]) -> dict[str, Any]:
    event_type = ""
    if int(state.get("coverage_plateau_streak") or 0) > 0:
        event_type = "coverage_plateau"
    elif bool(state.get("coverage_seed_generation_degraded") or False):
        event_type = "seed_generation_degraded"
    elif str(state.get("crash_triage_label") or "").strip() == "harness_bug":
        event_type = "harness_bug"
    elif bool(state.get("repair_mode") or False):
        event_type = "repair_feedback"
    # Always emit at least a coverage_normal event so the vuln-hunt agent
    # receives current run/coverage data for iterative refinement.
    if not event_type:
        event_type = "coverage_normal"
    seed_quality = dict(state.get("coverage_seed_quality") or {}) if isinstance(state.get("coverage_seed_quality"), dict) else {}
    return {
        "ts": int(time.time()),
        "event_type": event_type,
        "target_name": str(state.get("coverage_target_name") or ""),
        "target_api": str(state.get("coverage_target_api") or state.get("selected_target_api") or ""),
        "coverage_plateau_streak": int(state.get("coverage_plateau_streak") or 0),
        "coverage_quality_flags": list(state.get("coverage_quality_flags") or []),
        "coverage_seed_generation_degraded": bool(state.get("coverage_seed_generation_degraded") or False),
        "crash_triage_label": str(state.get("crash_triage_label") or ""),
        "repair_origin_stage": str(state.get("repair_origin_stage") or ""),
        "coverage_loop_round": int(state.get("coverage_loop_round") or 0),
        "coverage_should_improve": bool(state.get("coverage_should_improve") or False),
        "cov_delta": int(seed_quality.get("cov_delta") or 0),
        "ft_delta": int(seed_quality.get("ft_delta") or 0),
        "seed_score": float(seed_quality.get("seed_score") or 0.0),
        "coverage_bottleneck_kind": str(state.get("coverage_bottleneck_kind") or ""),
    }


def _append_vuln_hunt_event(repo_root: Path, event: dict[str, Any]) -> str:
    if not event:
        return ""
    path = _vuln_hunt_events_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return str(path)


def _write_vuln_hunt_summary(repo_root: Path, candidates: list[dict[str, Any]], event: dict[str, Any]) -> str:
    path = _vuln_hunt_summary_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    active = _active_vuln_candidates(candidates)
    lines = [
        "# Vulnerability Hunt Summary",
        "",
        f"- generated_at: {int(time.time())}",
        f"- candidate_count: {len(candidates)}",
        f"- active_candidate_count: {len(active)}",
        f"- event_type: {event.get('event_type') or 'initial_hunt'}",
        "",
        "## Top Candidates",
        "",
    ]
    for idx, item in enumerate(active[:10], start=1):
        lines.extend(
            [
                f"{idx}. `{item.get('candidate_id') or ''}`",
                f"   - api: `{item.get('target_api') or item.get('api') or ''}`",
                f"   - risk_type: `{item.get('risk_type') or item.get('signal_type') or ''}`",
                f"   - priority: {float(item.get('priority') or 0.0):.4f}",
                f"   - status: `{item.get('validation_status') or 'pending'}`",
            ]
        )
        reason = str(item.get("security_priority_reason") or "").strip()
        if reason:
            lines.append(f"   - reason: {reason[:240]}")
    if not active:
        lines.append("- no active candidates")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _run_vuln_hunt_subphase(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    """Refresh vulnerability candidates without mutating control-plane truth."""
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    repo_root = gen.repo_root
    enabled = bool(_vuln_hunting_enabled())
    analysis_context_path = str(state.get("analysis_context_path") or repo_root / "fuzz" / "analysis_context.json")
    analysis_ctx_obj = Path(analysis_context_path)
    if not analysis_ctx_obj.is_absolute():
        analysis_ctx_obj = repo_root / analysis_ctx_obj
    has_hunt_input = bool(analysis_ctx_obj.is_file() or _vuln_candidates_path(repo_root).is_file())
    issue = ""
    result: dict[str, Any] = {"path": str(_vuln_candidates_path(repo_root)), "candidate_count": 0, "issue": ""}
    if enabled:
        try:
            result = _write_analysis_vuln_candidates(repo_root, analysis_context_path)
        except Exception as exc:
            issue = f"vuln_hunt_candidate_materialize_error:{exc}"
        if has_hunt_input and _has_codex_key() and getattr(gen, "patcher", None) is not None:
            try:
                # Snapshot vuln_candidates.json before the agent runs so we can
                # roll back if it is killed mid-write and leaves a corrupt file.
                # The done-file flush grace period only waits for the file named
                # in ./done (vuln_hunt_summary.md), not vuln_candidates.json.
                _vc_path = _vuln_candidates_path(repo_root)
                _vc_snap_raw: bytes | None = None
                _vc_snap_count = 0
                try:
                    if _vc_path.is_file():
                        _vc_snap_raw = _vc_path.read_bytes()
                        _vc_snap_count = len(
                            (_load_vuln_candidates_doc(repo_root).get("candidates") or [])
                        )
                except Exception:
                    _vc_snap_raw = None

                _clear_opencode_done_sentinel(repo_root)
                hunt_hint_lines = [
                    f"- analysis_context_path: {analysis_context_path}",
                    "- read `fuzz/analysis_context.json` and `fuzz/vuln_candidates.json` first",
                    "- update only advisory vulnerability candidates and hunt summary",
                    "- preserve validation_status/attempt_count/last_result for existing candidates",
                ]
                # Inject run/coverage feedback paths so the agent can refine
                # vulnerability assessments with actual fuzz data.
                run_feedback_path = str(state.get("fuzz_coverage_run_feedback_path") or "")
                if run_feedback_path:
                    hunt_hint_lines.append(f"- run_feedback_path: {run_feedback_path}")
                coverage_frontier = str(state.get("fuzz_coverage_frontier_path") or "")
                if coverage_frontier:
                    hunt_hint_lines.append(f"- coverage_frontier_path: {coverage_frontier}")
                replay_manifest = str(state.get("coverage_per_input_manifest_path") or "")
                if replay_manifest:
                    hunt_hint_lines.append(f"- per_input_manifest_path: {replay_manifest}")
                # Attach run summary if available.
                run_summary_path = str(state.get("fuzz_coverage_run_feedback_summary") or "")
                if run_summary_path:
                    hunt_hint_lines.append(f"- coverage_run_feedback_summary: {run_summary_path}")
                hunt_hint = "\n".join(hunt_hint_lines)
                event = _vuln_hunt_event_from_state(cast(dict[str, Any], state))
                if event:
                    hunt_hint += "\n- latest_feedback_event: " + json.dumps(event, ensure_ascii=False, sort_keys=True)
                prompt, render_issue = _render_opencode_prompt_safe(
                    "vuln_hunt_with_hint",
                    fallback_name="analysis_with_hint",
                    hint=hunt_hint,
                    fallback_hint=hunt_hint,
                )
                if render_issue:
                    issue = "; ".join(x for x in [issue, render_issue] if str(x).strip())
                gen.patcher.run_codex_command(
                    prompt,
                    stage_skill="vuln_hunt",
                    timeout=_remaining_time_budget_sec(state),
                    max_attempts=1,
                    max_cli_retries=_opencode_cli_retries(),
                    idle_timeout_override=_vuln_hunt_idle_timeout_sec(),
                )

                # Validate vuln_candidates.json; restore snapshot if agent
                # left a corrupt file (truncated write due to process kill).
                if _vc_snap_raw is not None:
                    _post_ok = False
                    _post_count = 0
                    try:
                        _raw = json.loads(_vc_path.read_text(encoding="utf-8", errors="replace"))
                        _post_ok = isinstance(_raw, dict)
                        _post_count = len(_raw.get("candidates") or []) if _post_ok else 0
                    except Exception:
                        pass
                    if not _post_ok:
                        try:
                            _vc_path.write_bytes(_vc_snap_raw)
                            issue = "; ".join(
                                x for x in [
                                    issue,
                                    f"vuln_candidates_corrupt_restored(pre={_vc_snap_count},post={_post_count})",
                                ]
                                if str(x).strip()
                            )
                        except Exception as _restore_exc:
                            issue = "; ".join(
                                x for x in [issue, f"vuln_candidates_restore_failed:{_restore_exc}"]
                                if str(x).strip()
                            )
            except Exception as exc:
                issue = "; ".join(
                    x for x in [issue, f"vuln_hunt_opencode_error:{exc}"] if str(x).strip()
                )
    doc = _load_vuln_candidates_doc(repo_root)
    candidates = list(doc.get("candidates") or [])
    active = _active_vuln_candidates(candidates)
    active_candidate = dict(active[0]) if active else {}
    event = _vuln_hunt_event_from_state(cast(dict[str, Any], state))
    events_path = _append_vuln_hunt_event(repo_root, event)
    summary_path = _write_vuln_hunt_summary(repo_root, candidates, event)
    degraded_reason = str(issue or result.get("issue") or "").strip()
    out: dict[str, Any] = {
        **state,
        "vuln_hunt_enabled": enabled,
        "vuln_hunt_iteration": int(state.get("vuln_hunt_iteration") or 0) + 1,
        "vuln_hunt_active_candidate_id": str(active_candidate.get("candidate_id") or ""),
        "vuln_hunt_highest_priority": float(active_candidate.get("priority") or 0.0) if active else 0.0,
        "vuln_hunt_candidate_count": len(candidates),
        "vuln_hunt_degraded": bool(degraded_reason),
        "vuln_hunt_last_reason": degraded_reason,
        "vuln_hunt_summary_path": summary_path,
        "vuln_hunt_events_path": events_path,
        "vuln_candidates_path": str(result.get("path") or _vuln_candidates_path(repo_root)),
        "vuln_candidate_count": len(candidates),
    }
    snapshot = {
        "kind": "choose_vuln_candidate",
        "candidate_id": str(active_candidate.get("candidate_id") or ""),
        "target_api": str(active_candidate.get("target_api") or active_candidate.get("api") or ""),
        "priority": float(active_candidate.get("priority") or 0.0),
        "validation_status": str(active_candidate.get("validation_status") or ""),
        "candidate_count": len(candidates),
        "active_candidate_count": len(active),
        "event_type": str(event.get("event_type") or "initial_hunt"),
        "degraded_reason": degraded_reason,
    }
    out = _record_decision_trace(
        out,
        stage="plan",
        tool="system",
        model="vuln-hunt",
        latency_ms=0,
        decision_snapshot=snapshot,
    )
    return _attach_prompt_render_status(out, issue=degraded_reason)


def _lookup_target_security_candidate(
    *,
    target_name: str,
    api: str,
    index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    for key in _target_analysis_lookup_keys(target_name, api):
        if key in index:
            return dict(index.get(key) or {})
    return {}


def _build_selected_target_row(
    *,
    repo_root: Path,
    item: dict[str, Any],
    security_lookup: dict[str, dict[str, Any]],
    security_priority_mode: bool,
    degrade_reason: str,
    score_weights: dict[str, float],
    vuln_priority_by_api: dict[str, float] | None = None,
) -> dict[str, Any]:
    item = _wf_norm.normalize_target_row(item)
    explicit_security_breakdown = (
        dict(item.get("security_score_breakdown") or {})
        if isinstance(item.get("security_score_breakdown"), dict)
        else {}
    )
    target_name = str(item.get("target_name") or item.get("name") or "").strip()
    api = str(item.get("api") or target_name).strip()
    target_type = str(item.get("target_type") or "generic").strip().lower()

    # Internal sink -> nearest public entry mapping. When the requested API is
    # non-public but the call graph shows a public caller that reaches it, fuzz
    # the public entry instead (it's linkable) and keep the sink as an attack
    # hint so synthesize still steers toward the dangerous path. Falls back to
    # the escape-hatch handling below when no public caller is found.
    _sink_remap: dict[str, str] = {}
    if _vuln_public_api_enforce() and api:
        _ps = _load_public_api_symbols(repo_root)
        if _is_non_public_api(api, _ps):
            _mapped = _nearest_public_entry(api, _ps, _load_callgraph_reverse(repo_root))
            if _mapped and _mapped != api:
                _sink_remap = {"original_api": api, "mapped_api": _mapped}
                if not target_name or target_name == api:
                    target_name = _mapped
                api = _mapped

    seed_profile = _normalize_seed_profile(
        str(item.get("seed_profile") or "generic"),
        target_type=target_type,
        name=target_name,
        context=api,
    )
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
    security_candidate = _lookup_target_security_candidate(
        target_name=target_name,
        api=api,
        index=security_lookup,
    )
    source_hint = str(item.get("file") or security_candidate.get("file") or "")
    advisory_origin_target_name = ""
    advisory_origin_api = ""
    runtime_replacement_reason = ""
    if _is_test_or_demo_helper_target(name=target_name, api=api, file_hint=source_hint):
        if not runtime_replacement_candidates:
            _, _, runtime_replacement_candidates = _runtime_viability_details(
                target_name,
                api,
                file_hint=source_hint,
            )
        public_replacements = [
            str(x).strip()
            for x in runtime_replacement_candidates
            if str(x).strip() and not _is_internal_api_symbol(str(x))
        ]
        if public_replacements:
            advisory_origin_target_name = target_name
            advisory_origin_api = api
            target_name = public_replacements[0]
            api = public_replacements[0]
            runtime_viability = "high"
            runtime_replacement_reason = "test_demo_helper_public_surrogate"
            selection_rationale = (
                f"{selection_rationale};public-validation-surrogate"
                if selection_rationale
                else "public-validation-surrogate"
            )
    security_scores = _extract_security_scores(item)
    if not any(float(v) > 0.0 for v in security_scores.values()):
        security_scores = _extract_security_scores(security_candidate)
    if not any(float(v) > 0.0 for v in security_scores.values()):
        security_scores = _compute_security_signal_scores(
            name=target_name,
            signature=f"{api} {selection_rationale}",
            file_hint=source_hint,
            risk_signals=list(item.get("risk_signals") or security_candidate.get("risk_signals") or []),
            risk_signal_source_breakdown=dict(
                item.get("risk_signal_source_breakdown")
                or security_candidate.get("risk_signal_source_breakdown")
                or {}
            ),
        )
    # If plan emitted an aggregate public validation target with explicit
    # security_score_breakdown, treat that as the advisory risk source for this
    # row. Exact candidate matches are often internal/helper APIs and must not
    # overwrite the agent's selected public entrypoint risk summary.
    vuln_likelihood_raw = explicit_security_breakdown.get(
        "vuln_likelihood",
        security_candidate.get("vuln_likelihood", item.get("vuln_likelihood")),
    )
    exploitability_raw = explicit_security_breakdown.get(
        "exploitability",
        security_candidate.get("exploitability", item.get("exploitability")),
    )
    reachability_raw = explicit_security_breakdown.get(
        "reachability_confidence",
        security_candidate.get("reachability_confidence", item.get("reachability_confidence")),
    )
    security_reason = str(
        security_candidate.get("security_priority_reason")
        or item.get("security_priority_reason")
        or ""
    ).strip()
    evidence_ids_source = (
        list(item.get("evidence_ids") or [])
        if explicit_security_breakdown
        else (list(security_candidate.get("evidence_ids") or []) or list(item.get("evidence_ids") or []))
    )
    evidence_ids = list(dict.fromkeys(evidence_ids_source))
    evidence_refs = (
        list(item.get("evidence") or [])
        if explicit_security_breakdown and item.get("evidence")
        else list(security_candidate.get("evidence") or item.get("evidence") or [])
    )
    signal_type = str(security_candidate.get("signal_type") or item.get("signal_type") or "").strip()
    try:
        vuln_likelihood = max(0.0, min(float(vuln_likelihood_raw), 1.0))
        exploitability = max(0.0, min(float(exploitability_raw), 1.0))
        reachability_confidence = max(0.0, min(float(reachability_raw), 1.0))
    except Exception:
        vuln_likelihood, exploitability, reachability_confidence, derived_reason = _derive_security_priority(
            target_type=target_type,
            runtime_viability=runtime_viability,
            security_scores=security_scores,
        )
        if not security_reason:
            security_reason = derived_reason
    # Override vuln_likelihood with vuln_candidate priority when a
    # matching candidate exists.  This bridges vuln-hunt findings
    # into plan target selection so the highest-priority vuln candidate
    # becomes the selected target.
    _vuln_override = vuln_priority_by_api or {}
    _vc_prio = float(_vuln_override.get(api, _vuln_override.get(target_name, 0.0)) or 0.0)
    if _vc_prio > 0:
        vuln_likelihood = max(vuln_likelihood, _vc_prio)
    if not security_reason:
        _, _, _, security_reason = _derive_security_priority(
            target_type=target_type,
            runtime_viability=runtime_viability,
            security_scores=security_scores,
        )
    if not signal_type:
        top_signals = _top_security_signals(security_scores, threshold=0.0)
        signal_type = top_signals[0] if top_signals else "mem_oob_candidate"
    attack_hint = _normalize_attack_hint(
        security_candidate.get("attack_hint") or item.get("attack_hint"),
        api=api,
        target_type=target_type,
        signal_id=signal_type,
        source_path=source_hint,
        security_reason=security_reason,
    )
    if not attack_hint:
        attack_hint = _candidate_attack_hint(
            api=api,
            target_type=target_type,
            signal_id=signal_type,
            source_path=str(item.get("file") or security_candidate.get("file") or ""),
            security_reason=security_reason,
        )
    # When we remapped an internal sink to a public entry, keep the sink on the
    # attack-hint key code path so synthesize steers the public harness toward
    # the dangerous internal function.
    if _sink_remap:
        attack_hint = dict(attack_hint or {})
        kcp = [str(x).strip() for x in list(attack_hint.get("key_code_path") or []) if str(x).strip()]
        sink = _sink_remap.get("original_api", "")
        if sink and sink not in kcp:
            kcp.insert(0, sink)
        attack_hint["key_code_path"] = kcp
        attack_hint["mapped_from_internal_sink"] = sink
        attack_hint["public_entry"] = _sink_remap.get("mapped_api", "")
    signal_score = max(0.0, min(float(security_scores.get(signal_type) or 0.0), 1.0))
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
    wrapper_fuzzer_name = str(item.get("wrapper_fuzzer_name") or "")
    runtime_penalty = _target_runtime_penalty(
        repo_root,
        wrapper_fuzzer_name,
        target_name=target_name,
        api=api,
    )
    execution_bias = _execution_depth_bias(
        target_name=target_name,
        api=api,
        target_type=target_type,
        depth_class=str(item.get("depth_class") or ""),
        depth_score=int(item.get("depth_score") or 0),
        selection_rationale=selection_rationale,
    )
    surface_penalty = _target_surface_penalty(
        target_name=target_name,
        api=api,
        source_path=source_hint,
        runtime_replacement_reason=runtime_replacement_reason,
    )
    score_penalty = float(runtime_penalty.get("score_penalty") or 0.0)
    target_surface_penalty = float(surface_penalty.get("target_surface_penalty") or 0.0)
    entrypoint_bias, coverage_potential_score = _entrypoint_risk_bias(api, target_type, repo_root)
    non_harnessable_dropped = _is_non_harnessable_target(api)
    score_breakdown["recent_yield_penalty"] = round(score_penalty + target_surface_penalty, 4)
    score_total = (
        float(score_weights.get("vuln_likelihood", 0.50)) * float(vuln_likelihood)
        + float(score_weights.get("exploitability", 0.30)) * float(exploitability)
        + float(score_weights.get("reachability_confidence", 0.20)) * float(reachability_confidence)
        + float(execution_bias.get("execution_depth_bias") or 0.0)
        + float(entrypoint_bias)
        - float(score_penalty)
        - float(target_surface_penalty)
    )
    if explicit_security_breakdown and item.get("score_total") is not None:
        try:
            score_total = (
                float(item.get("score_total") or 0.0)
                - float(score_penalty)
                - float(target_surface_penalty)
            )
        except Exception:
            pass
    adjusted_target_score = max(0.0, float(score_total))
    # Non-public/non-linkable symbols (WASM/binding shims, statics, internal)
    # are gated here using the public-API surface oracle, not just the legacy
    # name heuristic — so they get the internal-API penalty + escape-hatch
    # treatment *before* a wasted build cycle exposes them as non_public_api_usage.
    _public_set = _load_public_api_symbols(repo_root)
    internal_api = _is_non_public_api(api, _public_set)
    internal_min = _vuln_internal_api_min_score()
    api_surface_exception = {"used": False, "reason": "", "evidence_ids": []}
    # Structurally-unlinkable language bindings (wasm/napi/jni/emscripten shims)
    # have no native public caller. The internal->public remap above already
    # tried to find one; an empty `_sink_remap` means it failed. Such a symbol
    # can never link into a native harness regardless of its vuln score, so the
    # high-risk escape-hatch must NOT admit it — flag it for a hard drop instead
    # (filtered in `_apply_selected_target_filters`) so it can't waste a
    # plan/synthesize/build/repair cycle. Honour the enforce kill-switch.
    unlinkable_binding_dropped = bool(
        _vuln_public_api_enforce()
        and internal_api
        and not _sink_remap
        and _is_unlinkable_binding_api(api)
    )
    if internal_api:
        if unlinkable_binding_dropped:
            adjusted_target_score = 0.0
            api_surface_exception = {
                "used": False,
                "reason": f"unlinkable_binding_no_public_entry({api})",
                "evidence_ids": list(security_candidate.get("evidence_ids") or []),
            }
            if not runtime_penalty.get("reason"):
                runtime_penalty["reason"] = "unlinkable_binding_dropped"
            elif "unlinkable_binding_dropped" not in str(runtime_penalty.get("reason") or ""):
                runtime_penalty["reason"] = (
                    f"{runtime_penalty.get('reason')};unlinkable_binding_dropped"
                )
        elif security_priority_mode and vuln_likelihood >= internal_min:
            api_surface_exception = {
                "used": True,
                "reason": f"risk_first_allow_internal(vuln_likelihood={vuln_likelihood:.2f})",
                "evidence_ids": list(security_candidate.get("evidence_ids") or []),
            }
        else:
            adjusted_target_score = max(0.0, adjusted_target_score - 0.75)
            if not runtime_penalty.get("reason"):
                runtime_penalty["reason"] = "internal_api_below_vuln_threshold"
            elif "internal_api_below_vuln_threshold" not in str(runtime_penalty.get("reason") or ""):
                runtime_penalty["reason"] = (
                    f"{runtime_penalty.get('reason')};internal_api_below_vuln_threshold"
                )
    priority_base = _candidate_priority(
        vuln_likelihood=vuln_likelihood,
        exploitability=exploitability,
        reachability_confidence=reachability_confidence,
        evidence_count=max(len(evidence_ids), len(evidence_refs)),
        signal_score=signal_score,
    )
    priority_penalty = min(0.95, (max(0.0, score_penalty) * 0.55) + (target_surface_penalty * 0.9))
    penalty_parts = [
        str(runtime_penalty.get("reason") or "").strip(),
        str(surface_penalty.get("target_surface_penalty_reason") or "").strip(),
    ]
    penalty_reason = ";".join(dict.fromkeys(x for x in penalty_parts if x))
    if any(
        token in penalty_reason
        for token in (
            "persistent_low_yield_target",
            "coverage_exhausted_target",
            "cold_start_low_yield",
            "very_low_seed_score",
        )
    ):
        priority_penalty = max(priority_penalty, 0.5)
    effective_priority = round(
        max(
            0.0,
            min(
                1.0,
                float(priority_base)
                - float(priority_penalty)
                + float(execution_bias.get("execution_depth_bias") or 0.0)
                + float(entrypoint_bias),
            ),
        ),
        4,
    )
    score_breakdown_fixed = {
        "vuln_likelihood": float(vuln_likelihood),
        "exploitability": float(exploitability),
        "reachability_confidence": float(reachability_confidence),
        "recent_yield_penalty": float(score_breakdown.get("recent_yield_penalty") or 0.0),
    }
    return {
        "target_name": target_name,
        "name": target_name,
        "target": target_name,
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
        "runtime_replacement_reason": runtime_replacement_reason,
        "advisory_origin_target_name": advisory_origin_target_name,
        "advisory_origin_api": advisory_origin_api,
        "seed_families_suggested": required,
        "seed_families_optional": optional,
        "wrapper_fuzzer_name": wrapper_fuzzer_name,
        "risk_signal_source_breakdown": dict(
            item.get("risk_signal_source_breakdown")
            or security_candidate.get("risk_signal_source_breakdown")
            or {}
        ),
        "score_total": float(adjusted_target_score),
        "score_breakdown": score_breakdown_fixed,
        "penalty_reason": penalty_reason,
        "security_score_breakdown": {
            "vuln_likelihood": float(vuln_likelihood),
            "exploitability": float(exploitability),
            "reachability_confidence": float(reachability_confidence),
            "recent_yield_penalty": float(score_penalty + target_surface_penalty),
            "weights": {k: float(v) for k, v in score_weights.items()},
        },
        "security_priority_mode": bool(security_priority_mode),
        "degraded_reason": str(degrade_reason),
        "vuln_likelihood": float(vuln_likelihood),
        "exploitability": float(exploitability),
        "reachability_confidence": float(reachability_confidence),
        "signal_type": signal_type,
        "signal_score": float(signal_score),
        "priority": float(priority_base),
        "effective_priority": float(effective_priority),
        "execution_depth_bias": float(execution_bias.get("execution_depth_bias") or 0.0),
        "entrypoint_bias": float(entrypoint_bias),
        "coverage_potential": float(coverage_potential_score),
        "callback_penalty": float(execution_bias.get("callback_penalty") or 0.0),
        "wrapper_penalty": float(execution_bias.get("wrapper_penalty") or 0.0),
        "target_surface_penalty": float(target_surface_penalty),
        "target_surface_penalty_reason": str(surface_penalty.get("target_surface_penalty_reason") or ""),
        "evidence": list(evidence_refs),
        "evidence_ids": evidence_ids,
        "vuln_candidate_id": str(security_candidate.get("candidate_id") or item.get("candidate_id") or ""),
        "vuln_candidate_priority": float(security_candidate.get("priority") or item.get("priority") or 0.0),
        "candidate_origin": str(security_candidate.get("candidate_origin") or item.get("candidate_origin") or "analysis_context"),
        "validation_status": str(security_candidate.get("validation_status") or item.get("validation_status") or "pending"),
        "attack_hint": attack_hint,
        "sink_remap": dict(_sink_remap),
        "security_priority_reason": security_reason,
        "security_signals": _top_security_signals(security_scores),
        "security_signal_scores": {k: float(v) for k, v in security_scores.items()},
        "api_surface_exception": api_surface_exception,
        "unlinkable_binding_dropped": bool(unlinkable_binding_dropped),
        "non_harnessable_dropped": bool(non_harnessable_dropped),
        "target_score_breakdown": score_breakdown,
        "target_score": float(adjusted_target_score),
        "target_score_penalty": float(score_penalty + target_surface_penalty),
        "target_score_penalty_reason": penalty_reason,
        "target_score_breakdown_available": True,
        "target_scoring_enabled": True,
        "vuln_hunting_enabled": bool(_vuln_hunting_enabled()),
        "vuln_focus_profile": "broad_high_risk",
        "target_surface_policy": "risk_first",
    }


def _build_selected_targets_doc(
    repo_root: Path,
    *,
    exclude_names: list[str] | None = None,
    prefer_deeper: bool = False,
) -> list[dict[str, Any]]:
    security_lookup = _load_target_analysis_security_index(repo_root)
    # Build vuln_candidate priority override: map API→priority from
    # vuln_candidates.json so vuln-hunt findings directly influence
    # target scoring, not just target_analysis static signals.
    _vuln_priority_by_api: dict[str, float] = {}
    if _vuln_hunting_enabled():
        try:
            _vc_doc = _load_vuln_candidates_doc(repo_root)
            for _vc in _vc_doc.get("candidates") or []:
                # Exhausted/cooling candidates have already been driven to a
                # plateau; do not let their stale priority override re-boost the
                # same target (and re-inject it below), or risk-first ranking
                # keeps reselecting a target the fuzzer can't make progress on.
                _vc_status = str(_vc.get("validation_status") or "").strip().lower()
                if _vc_status in {"exhausted", "cooling"}:
                    continue
                _vc_api = str(_vc.get("api") or _vc.get("target_api") or "").strip()
                _vc_prio = float(_vc.get("priority") or 0.0)
                if _vc_api and _vc_prio > 0:
                    _existing = _vuln_priority_by_api.get(_vc_api, 0.0)
                    _vuln_priority_by_api[_vc_api] = max(_existing, _vc_prio)
        except Exception:
            pass
    security_priority_mode = bool(_vuln_hunting_enabled() and _vuln_score_mode() == "risk_first_v1")
    degrade_reason = ""
    if not _vuln_hunting_enabled():
        degrade_reason = "vuln_hunting_disabled"
    elif _vuln_score_mode() != "risk_first_v1":
        degrade_reason = "unsupported_vuln_score_mode"
    score_weights = _vuln_score_weights()
    ranked_items: list[dict[str, Any]] = []
    for item in _load_targets_doc(repo_root):
        ranked_items.append(
            _build_selected_target_row(
                repo_root=repo_root,
                item=item,
                security_lookup=security_lookup,
                security_priority_mode=security_priority_mode,
                degrade_reason=degrade_reason,
                score_weights=score_weights,
                vuln_priority_by_api=_vuln_priority_by_api,
            )
        )
    # Inject high-priority vuln candidates that are missing from targets.json.
    # The plan stage's OpenCode may not include internal helper functions
    # that vuln-hunt identified as attack surfaces.  Create target entries
    # for them so they participate in ranking.
    _seen_apis: set[str] = {str(r.get("api") or "").strip().lower() for r in ranked_items}
    _seen_apis.update(str(r.get("target_name") or "").strip().lower() for r in ranked_items)
    for _vc_api, _vc_prio in sorted(_vuln_priority_by_api.items(), key=lambda x: -x[1]):
        if _vc_api.lower() not in _seen_apis and len(ranked_items) < _execution_targets_max() * 2:
            _synthetic_item = {
                "name": _vc_api,
                "api": _vc_api,
                "lang": "c",
                "target_type": "generic",
                "seed_profile": "generic",
                "priority": _vc_prio,
                "vuln_likelihood": _vc_prio,
                "exploitability": 0.5,
                "reachability_confidence": 0.5,
                "_synthetic_from_vuln_candidate": True,
            }
            ranked_items.append(
                _build_selected_target_row(
                    repo_root=repo_root,
                    item=_synthetic_item,
                    security_lookup=security_lookup,
                    security_priority_mode=security_priority_mode,
                    degrade_reason=degrade_reason,
                    score_weights=score_weights,
                    vuln_priority_by_api=_vuln_priority_by_api,
                )
            )
            _seen_apis.add(_vc_api.lower())
    # Inject the library's public entrypoints (whole-input parse/decode/load
    # APIs) as candidates. vuln-hunt is a "sniper" that proposes risky internal
    # helpers but rarely the top-level public entry (e.g. toml_parse), which
    # drives the whole parser for far more coverage. With the entry present, the
    # entrypoint bias can promote it into the execution plan. No-op when the
    # public-API oracle is empty (degraded) or enforcement is disabled.
    if _vuln_public_api_enforce():
        try:
            _public_syms = _load_public_api_symbols(repo_root)
            _entry_syms = sorted(n for n in _public_syms if _is_library_entrypoint(n))
            for _ep in _entry_syms:
                if _ep.lower() in _seen_apis or len(ranked_items) >= _execution_targets_max() * 2:
                    continue
                _ep_item = {
                    "name": _ep,
                    "api": _ep,
                    "lang": "c",
                    "target_type": "parser",
                    "seed_profile": "generic",
                    "vuln_likelihood": 0.62,
                    "exploitability": 0.5,
                    "reachability_confidence": 0.65,
                    "_synthetic_public_entrypoint": True,
                }
                ranked_items.append(
                    _build_selected_target_row(
                        repo_root=repo_root,
                        item=_ep_item,
                        security_lookup=security_lookup,
                        security_priority_mode=security_priority_mode,
                        degrade_reason=degrade_reason,
                        score_weights=score_weights,
                        vuln_priority_by_api=_vuln_priority_by_api,
                    )
                )
                _seen_apis.add(_ep.lower())
        except Exception:
            pass
    # In risk-first mode, ranking is driven by security risk directly.
    # `score_total` is still emitted for observability/reference, not as the
    # primary ordering key.
    ranked_items = _wf_target_selection.sort_ranked_items(
        ranked_items,
        security_priority_mode=security_priority_mode,
        is_internal_api_symbol_fn=_is_internal_api_symbol,
        runtime_viability_rank_fn=_runtime_viability_rank,
        prefer_deeper=prefer_deeper,
        selection_mode=_selection_mode(),
    )
    ranked_items = _apply_selected_target_filters(
        ranked_items,
        exclude_names=exclude_names,
    )
    max_targets = _execution_targets_max()
    out = _wf_target_selection.assign_execution_priority(
        ranked_items,
        max_targets=max_targets,
        security_priority_mode=security_priority_mode,
    )
    return out


def _write_selected_targets_doc(
    repo_root: Path,
    *,
    exclude_names: list[str] | None = None,
    prefer_deeper: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    path = _selected_targets_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = _build_selected_targets_doc(
        repo_root,
        exclude_names=exclude_names,
        prefer_deeper=prefer_deeper,
    )
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
    seen_identity_tokens: set[str] = set()
    for item in selected:
        prio = int(item.get("execution_priority") or 0)
        if prio <= 0 or prio > max_targets:
            continue
        target_name = str(item.get("target_name") or item.get("name") or "").strip()
        api = str(item.get("api") or "").strip()
        identity_token = _normalize_exec_target_token(api) or _normalize_exec_target_token(target_name)
        if identity_token and identity_token in seen_identity_tokens:
            continue
        if identity_token:
            seen_identity_tokens.add(identity_token)
        expected_bin = str(item.get("wrapper_fuzzer_name") or target_name).strip()
        execution_targets.append(
            {
                "target_name": target_name,
                "expected_fuzzer_name": expected_bin,
                "api": api,
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


def _sync_execution_plan_doc_from_selected_targets(repo_root: Path) -> tuple[str, dict[str, Any]]:
    """Make execution_plan.json a derived artifact of selected_targets.json.

    Agent-authored repair plans may edit execution_plan.json directly. The
    control-plane contract treats selected_targets.json as the normalized ranked
    target source, so stage boundaries must re-derive execution_plan.json from it
    when it exists.
    """
    selected_doc = _load_selected_targets_doc(repo_root)
    if not selected_doc:
        return str(_execution_plan_path(repo_root)), _load_execution_plan_doc(repo_root)
    return _write_execution_plan_doc(repo_root, selected_doc)


def _selected_target_row_for_execution_target(
    selected_doc: list[dict[str, Any]],
    execution_target: dict[str, Any],
) -> dict[str, Any]:
    if not selected_doc:
        return {}
    target_tokens = [
        str(execution_target.get("target_name") or ""),
        str(execution_target.get("expected_fuzzer_name") or ""),
        str(execution_target.get("api") or ""),
    ]
    for row in selected_doc:
        row_tokens = [
            str(row.get("target_name") or ""),
            str(row.get("target") or ""),
            str(row.get("name") or ""),
            str(row.get("wrapper_fuzzer_name") or ""),
            str(row.get("api") or ""),
        ]
        for target_token in target_tokens:
            if target_token and _execution_target_matches_token(
                {
                    "target_name": target_token,
                    "expected_fuzzer_name": target_token,
                    "api": target_token,
                },
                " ".join(row_tokens),
            ):
                return dict(row)
        normalized_target_tokens = {
            _normalize_exec_target_token(token)
            for token in target_tokens
            if str(token or "").strip()
        }
        normalized_row_tokens = {
            _normalize_exec_target_token(token)
            for token in row_tokens
            if str(token or "").strip()
        }
        if normalized_target_tokens.intersection(normalized_row_tokens):
            return dict(row)
    return dict(selected_doc[0])


def _workflow_target_state_from_execution_plan(
    repo_root: Path,
    execution_plan_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    execution_doc = execution_plan_doc if isinstance(execution_plan_doc, dict) else _load_execution_plan_doc(repo_root)
    execution_targets = [
        item for item in list(execution_doc.get("execution_targets") or [])
        if isinstance(item, dict)
    ]
    if not execution_targets:
        return {}
    selected_doc = _load_selected_targets_doc(repo_root)
    primary = _primary_execution_target(execution_targets)
    selected = _selected_target_row_for_execution_target(selected_doc, primary)
    target_name = str(
        primary.get("target_name")
        or selected.get("target_name")
        or selected.get("target")
        or selected.get("name")
        or ""
    ).strip()
    target_api = str(primary.get("api") or selected.get("api") or target_name).strip()
    target_type = str(primary.get("target_type") or selected.get("target_type") or "").strip().lower()
    seed_profile = _normalize_seed_profile(
        str(primary.get("seed_profile") or selected.get("seed_profile") or ""),
        target_type=target_type,
        name=target_name,
        context=target_api,
    )
    suggested_families = list(selected.get("seed_families_suggested") or [])
    score_breakdown = dict(
        selected.get("score_breakdown")
        or selected.get("target_score_breakdown")
        or {}
    )
    selected_targets_path = str(_selected_targets_path(repo_root))
    decision_snapshot = {
        "kind": "choose_target",
        "selected_target": str(selected.get("target") or target_name or ""),
        "selected_api": str(selected.get("api") or target_api or ""),
        "score_total": float(selected.get("score_total") or selected.get("target_score") or 0.0),
        "score_breakdown": score_breakdown,
        "penalty_reason": str(
            selected.get("penalty_reason")
            or selected.get("target_score_penalty_reason")
            or ""
        ),
        "selected_targets_path": selected_targets_path,
        "degraded_reason": "" if selected_doc else "selected_targets_missing_or_empty",
        "security_priority_mode": bool(selected.get("security_priority_mode") or False),
        "top_vuln_candidate": str(selected.get("target") or target_name or ""),
        "security_score_breakdown": dict(selected.get("security_score_breakdown") or {}),
        "api_surface_exception_used": bool(
            dict(selected.get("api_surface_exception") or {}).get("used") or False
        ),
        "tie_break_reason": str(selected.get("tie_break_reason") or ""),
        "selection_delta_vs_runner_up": dict(selected.get("selection_delta_vs_runner_up") or {}),
    }
    out: dict[str, Any] = {
        "coverage_target_name": target_name,
        "coverage_target_api": target_api,
        "coverage_target_type": target_type,
        "selected_target_api": target_api,
        "coverage_seed_profile": seed_profile,
        "coverage_seed_families_suggested": suggested_families,
        "coverage_seed_families_missing": suggested_families,
        "coverage_target_score_breakdown": score_breakdown,
        "target_scoring_enabled": bool(selected.get("target_scoring_enabled") or False),
        "target_score_breakdown_available": bool(
            selected.get("target_score_breakdown_available")
            or selected.get("score_breakdown")
            or selected.get("target_score_breakdown")
        ),
        "selected_targets_path": selected_targets_path if selected_doc else "",
        "execution_plan_path": str(_execution_plan_path(repo_root)),
        "latest_decision_snapshot": decision_snapshot,
    }
    if bool(selected.get("security_priority_mode") or False):
        out["latest_vuln_decision_snapshot"] = {
            "kind": "choose_target",
            "selected_target": str(decision_snapshot.get("selected_target") or ""),
            "selected_api": str(decision_snapshot.get("selected_api") or ""),
            "security_priority_mode": True,
            "top_vuln_candidate": str(decision_snapshot.get("top_vuln_candidate") or ""),
            "security_score_breakdown": dict(decision_snapshot.get("security_score_breakdown") or {}),
            "api_surface_exception_used": bool(decision_snapshot.get("api_surface_exception_used") or False),
            "tie_break_reason": str(decision_snapshot.get("tie_break_reason") or ""),
            "selection_delta_vs_runner_up": dict(decision_snapshot.get("selection_delta_vs_runner_up") or {}),
        }
        out["security_priority_mode"] = True
    return out


def _load_execution_plan_doc(repo_root: Path) -> dict[str, Any]:
    path = _execution_plan_path(repo_root)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return dict(raw) if isinstance(raw, dict) else {}


def _execution_plan_targets(repo_root: Path) -> list[dict[str, Any]]:
    doc = _load_execution_plan_doc(repo_root)
    return [item for item in list(doc.get("execution_targets") or []) if isinstance(item, dict)]


def _execution_target_sort_key(item: dict[str, Any], index: int) -> tuple[int, int]:
    try:
        priority = int(item.get("execution_priority") or index + 1)
    except Exception:
        priority = index + 1
    must_run_rank = 0 if bool(item.get("must_run", True)) else 1
    return (must_run_rank, priority)


def _primary_execution_target(execution_targets: list[dict[str, Any]]) -> dict[str, Any]:
    if not execution_targets:
        return {}
    indexed = [(item, idx) for idx, item in enumerate(execution_targets)]
    indexed.sort(key=lambda pair: _execution_target_sort_key(pair[0], pair[1]))
    return dict(indexed[0][0])


def _execution_target_fuzzer_name(item: dict[str, Any]) -> str:
    return str(
        item.get("expected_fuzzer_name")
        or item.get("wrapper_fuzzer_name")
        or item.get("target_name")
        or item.get("name")
        or ""
    ).strip()


def _execution_target_fuzzer_aliases(item: dict[str, Any]) -> list[str]:
    aliases: list[str] = []
    for value in (
        _execution_target_fuzzer_name(item),
        item.get("wrapper_fuzzer_name"),
        item.get("target_name"),
        item.get("name"),
        item.get("api"),
    ):
        text = str(value or "").strip()
        if not text:
            continue
        base = Path(text).name
        stem = Path(base).stem
        for candidate in (text, base, stem):
            candidate = str(candidate or "").strip()
            if candidate and candidate not in aliases:
                aliases.append(candidate)
            if candidate and not re.search(r"_fuzz(?:er)?$", candidate):
                # Match both common harness naming conventions: `<api>_fuzz`
                # and `<api>_fuzzer`. Without the `_fuzzer` variant a binary
                # named e.g. `png_read_image_fuzzer` fails to match the
                # execution-plan target `png_read_image`, so the run stage finds
                # "no binaries matching execution_plan" and records nothing.
                for suffix in ("_fuzz", "_fuzzer"):
                    fuzz_candidate = f"{candidate}{suffix}"
                    if fuzz_candidate not in aliases:
                        aliases.append(fuzz_candidate)
            stripped = re.sub(r"_fuzz(?:er)?$", "", candidate)
            if stripped and stripped not in aliases:
                aliases.append(stripped)
    return aliases


def _execution_target_identity(item: dict[str, Any]) -> dict[str, str]:
    target_name = str(item.get("target_name") or item.get("name") or "").strip()
    target_api = str(item.get("api") or target_name).strip()
    target_type = str(item.get("target_type") or "").strip()
    seed_profile = str(item.get("seed_profile") or "").strip()
    expected_fuzzer_name = _execution_target_fuzzer_name(item)
    identity = _wf_norm.normalize_target_identity(target_name=target_name, target_api=target_api)
    return {
        "target_name": identity["target_name"],
        "target_api": identity["target_api"],
        "target_type": target_type,
        "seed_profile": seed_profile,
        "expected_fuzzer_name": expected_fuzzer_name,
    }


def _execution_target_matches_token(item: dict[str, Any], token: str) -> bool:
    normalized = _normalize_exec_target_token(token)
    if not normalized:
        return False
    identity = _execution_target_identity(item)
    candidates = [
        identity.get("target_name", ""),
        identity.get("target_api", ""),
        identity.get("expected_fuzzer_name", ""),
        str(item.get("wrapper_fuzzer_name") or ""),
        str(item.get("source_path") or ""),
    ]
    return normalized in {_normalize_exec_target_token(value) for value in candidates if str(value or "").strip()}


def _find_execution_target_for_tokens(
    execution_targets: list[dict[str, Any]],
    tokens: list[str],
) -> dict[str, Any]:
    for token in tokens:
        for item in execution_targets:
            if _execution_target_matches_token(item, token):
                return dict(item)
    return {}


def _preferred_execution_target(
    execution_targets: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    run_details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not execution_targets:
        return {}
    tokens: list[str] = []
    for snap_key in ("latest_vuln_decision_snapshot", "latest_decision_snapshot"):
        snap = state.get(snap_key)
        if isinstance(snap, dict):
            tokens.extend(
                [
                    str(snap.get("selected_target") or ""),
                    str(snap.get("selected_api") or ""),
                    str(snap.get("top_vuln_candidate") or ""),
                ]
            )
    tokens.extend(
        [
            str(state.get("coverage_target_name") or ""),
            str(state.get("coverage_target_api") or ""),
            str(state.get("selected_target_api") or ""),
            str(state.get("synthesize_selected_target_name") or ""),
            str(state.get("synthesize_selected_target_api") or ""),
            str(state.get("last_fuzzer") or ""),
        ]
    )
    for detail in run_details or []:
        if isinstance(detail, dict):
            tokens.append(str(detail.get("fuzzer") or ""))
    matched = _find_execution_target_for_tokens(execution_targets, tokens)
    return matched or _primary_execution_target(execution_targets)


def _target_type_from_run_details(
    run_details: list[dict[str, Any]] | None,
    *,
    target_name: str = "",
    target_api: str = "",
    fuzzer_name: str = "",
) -> str:
    details = [detail for detail in list(run_details or []) if isinstance(detail, dict)]
    if not details:
        return ""
    wanted = {
        _normalize_exec_target_token(target_name),
        _normalize_exec_target_token(target_api),
        _normalize_exec_target_token(fuzzer_name),
    }
    wanted.discard("")
    for detail in details:
        candidates = {
            _normalize_exec_target_token(str(detail.get("target_name") or "")),
            _normalize_exec_target_token(str(detail.get("target_api") or "")),
            _normalize_exec_target_token(str(detail.get("fuzzer") or "")),
        }
        if wanted and not (wanted & candidates):
            continue
        target_type = str(detail.get("target_type") or "").strip()
        if target_type:
            return target_type
    for detail in details:
        target_type = str(detail.get("target_type") or "").strip()
        if target_type:
            return target_type
    return ""


def _matching_run_details_for_target(
    run_details: list[dict[str, Any]] | None,
    *,
    target_name: str = "",
    target_api: str = "",
    fuzzer_name: str = "",
) -> list[dict[str, Any]]:
    details = [detail for detail in list(run_details or []) if isinstance(detail, dict)]
    wanted = {
        _normalize_exec_target_token(target_name),
        _normalize_exec_target_token(target_api),
        _normalize_exec_target_token(fuzzer_name),
    }
    wanted.discard("")
    if not details or not wanted:
        return []
    matched: list[dict[str, Any]] = []
    for detail in details:
        candidates = {
            _normalize_exec_target_token(str(detail.get("target_name") or "")),
            _normalize_exec_target_token(str(detail.get("target_api") or "")),
            _normalize_exec_target_token(str(detail.get("fuzzer") or "")),
        }
        candidates.discard("")
        if wanted & candidates:
            matched.append(detail)
    return matched


def _order_fuzzer_bins_by_execution_plan(bins: list[Path], execution_targets: list[dict[str, Any]]) -> list[Path]:
    if not bins or not execution_targets:
        return list(bins)
    by_name = {p.name: p for p in bins}
    by_stem = {p.stem: p for p in bins}
    ordered: list[Path] = []
    for item in sorted(
        enumerate(execution_targets),
        key=lambda pair: _execution_target_sort_key(pair[1], pair[0]),
    ):
        candidate = None
        for alias in _execution_target_fuzzer_aliases(item[1]):
            candidate = by_name.get(alias) or by_stem.get(Path(alias).stem)
            if candidate is not None:
                break
        if candidate is not None and candidate not in ordered:
            ordered.append(candidate)
    for p in bins:
        if p not in ordered:
            ordered.append(p)
    return ordered


def _filter_fuzzer_bins_by_execution_plan(bins: list[Path], execution_targets: list[dict[str, Any]]) -> list[Path]:
    """Return only fuzzer binaries that are current execution-plan targets.

    `fuzz/out` can contain binaries from earlier plan/replan rounds. Running those
    stale binaries pollutes run_details and can make coverage-analysis optimize a
    target that is no longer in the active plan.
    """
    if not bins or not execution_targets:
        return list(bins)
    by_name = {p.name: p for p in bins}
    by_stem = {p.stem: p for p in bins}
    filtered: list[Path] = []
    for item in sorted(
        enumerate(execution_targets),
        key=lambda pair: _execution_target_sort_key(pair[1], pair[0]),
    ):
        candidate = None
        for alias in _execution_target_fuzzer_aliases(item[1]):
            candidate = by_name.get(alias) or by_stem.get(Path(alias).stem)
            if candidate is not None:
                break
        if candidate is not None and candidate not in filtered:
            filtered.append(candidate)
    return filtered


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
            api_norm = _normalize_exec_target_token(api)
            for prior in mappings:
                if (
                    api_norm
                    and api_norm == _normalize_exec_target_token(str(prior.get("api") or ""))
                    and str(prior.get("source_path") or "").strip()
                ):
                    source_path = str(prior.get("source_path") or "").strip()
                    matched_by = "api_equivalent"
                    break
            if not source_path:
                label = target_name or expected or api
                if label:
                    missing_targets.append(label)
        mappings.append(
            {
                "target_name": target_name,
                "expected_fuzzer_name": expected,
                "api": api,
                "target_type": str(item.get("target_type") or "").strip(),
                "seed_profile": str(item.get("seed_profile") or "").strip(),
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
                "seed_profile": _normalize_seed_profile(
                    _raw_sp,
                    target_type=target_type,
                    name=name,
                    context=file_hint,
                ),
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


def _splice_sources_list(text: str, new_paths: list[str]) -> tuple[str, bool]:
    """Insert string-literal entries before the closing ']' of a SOURCES list.

    Recognizes two template shapes used by fuzz/build.py:
      - Python:  ``SOURCES = [ "a.c", "b.c" ]``
      - JSON-ish: ``"sources": [ "a.c", "b.c" ]``
    Skips entries already present (substring match). Returns ``(new_text, ok)``
    where ``ok`` is False when no SOURCES list could be located.
    """
    if not new_paths:
        return text, False

    pat = re.compile(r"(?P<head>SOURCES\s*=\s*\[|[\"']sources[\"']\s*:\s*\[)", re.IGNORECASE)
    m = pat.search(text)
    if not m:
        return text, False

    open_idx = m.end() - 1  # position of '['
    depth = 0
    close_idx = -1
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
    if close_idx < 0:
        return text, False

    inner = text[open_idx + 1 : close_idx]
    additions = [p for p in new_paths if p not in inner]
    if not additions:
        return text, False

    # Match indentation of last non-empty line inside the list (best-effort).
    inner_rstripped = inner.rstrip()
    last_nl = inner_rstripped.rfind("\n")
    indent = "    "
    if last_nl >= 0:
        tail_line = inner_rstripped[last_nl + 1 :]
        lead = len(tail_line) - len(tail_line.lstrip(" \t"))
        if lead > 0:
            indent = tail_line[:lead]

    sep = "," if inner_rstripped and not inner_rstripped.endswith(",") else ""
    insertion = sep + "\n" + "\n".join(f'{indent}"{p}",' for p in additions) + "\n"
    new_text = text[: open_idx + 1] + inner_rstripped + insertion + text[close_idx:]
    return new_text, True


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
    missing_all = [str(x).strip() for x in list(doc.get("missing_targets") or []) if str(x).strip()]
    if not missing_all:
        return True, "", doc
    # Only fail if must_run targets are missing; extra harnesses are harmless.
    ep = dict(execution_plan_doc or _load_execution_plan_doc(repo_root))
    must_run_names: set[str] = set()
    for t in ep.get("execution_targets") or []:
        if isinstance(t, dict) and t.get("must_run"):
            name = str(t.get("target_name") or t.get("expected_fuzzer_name") or "").strip()
            if name:
                must_run_names.add(name)
    must_run_missing = [n for n in missing_all if n in must_run_names]
    if must_run_missing:
        # A non-harnessable infra helper (CALLOC macro, expand_ptrarr, ...) that
        # the plan wrongly marked must_run should not fail a synthesize that
        # produced valid harnesses for the real targets — the synthesizer
        # correctly declined to harness it. Tolerate missing must_run targets
        # only when ALL of them are non-harnessable; genuine missing targets
        # still fail (preserving execution_plan_harness_mismatch detection).
        residual = [n for n in must_run_missing if not _is_non_harnessable_target(n)]
        if not residual:
            doc = dict(doc)
            doc["dropped_non_harnessable_must_run"] = must_run_missing
            return True, "", doc
        extras = [str(x).strip() for x in list(doc.get("extra_harnesses") or []) if str(x).strip()]
        msg = (
            "execution_plan_harness_mismatch: missing harness source for must_run targets="
            + ",".join(residual)
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
        "suggested_families": list(state.get("coverage_seed_families_suggested") or []),
        "covered_families": list(state.get("coverage_seed_families_covered") or []),
        "missing_suggested_families": list(state.get("coverage_seed_families_missing") or []),
        "attack_hint_total_count": int(quality.get("attack_hint_total_count") or 0),
        "attack_hint_covered_count": int(quality.get("attack_hint_covered_count") or 0),
        "attack_hint_missing_values": list(quality.get("attack_hint_missing_values") or []),
        "attack_hint_coverage_ratio": float(quality.get("attack_hint_coverage_ratio") or 1.0),
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


def _coverage_attack_hint_feedback_lines(seed_feedback: dict[str, Any]) -> list[str]:
    missing_values = [
        str(x).strip()
        for x in list(seed_feedback.get("attack_hint_missing_values") or [])
        if str(x).strip()
    ]
    if not missing_values:
        return []
    ratio = float(seed_feedback.get("attack_hint_coverage_ratio") or 0.0)
    return [
        (
            "- attack_hint_gap: missing boundary-oriented seeds for "
            + ", ".join(missing_values[:6])
            + (" ..." if len(missing_values) > 6 else "")
        ),
        (
            "- attack_hint_repair_directive: add or preserve format-valid seeds that encode those boundary values "
            f"(coverage_ratio={ratio:.2f}) before broadening generic mutations."
        ),
    ]


def _coverage_frontier_feedback_lines(frontier_summary: dict[str, Any]) -> list[str]:
    top_inputs = [
        dict(item)
        for item in list(frontier_summary.get("top_inputs") or [])
        if isinstance(item, dict)
    ]
    if not top_inputs:
        return []
    lines = ["- per_input_frontier: strongest replayed inputs so far:"]
    for item in top_inputs[:3]:
        rel = str(item.get("input_relpath") or "").strip() or "(unknown)"
        fn_count = int(item.get("covered_function_count") or 0)
        region_count = int(item.get("covered_region_count") or 0)
        frontier_score = float(item.get("frontier_score") or 0.0)
        unique_frontier_functions = int(item.get("unique_frontier_functions") or 0)
        nearby_uncovered_regions = int(item.get("nearby_uncovered_regions") or 0)
        target_relevance_count = int(item.get("target_relevance_count") or 0)
        closest_target_distance = int(item.get("closest_target_distance") or 0)
        funcs = [
            str(x).strip()
            for x in list(item.get("covered_functions_sample") or [])
            if str(x).strip()
        ]
        line = (
            f"  * {rel}: functions={fn_count}, regions={region_count}, "
            f"frontier_score={frontier_score:.3f}, frontier_functions={unique_frontier_functions}, "
            f"uncovered_regions_nearby={nearby_uncovered_regions}, "
            f"target_relevance={target_relevance_count}, target_distance={closest_target_distance}"
        )
        if funcs:
            line += f", sample={', '.join(funcs[:4])}"
        lines.append(line)
        frontier_functions = [
            dict(fn)
            for fn in list(item.get("frontier_functions") or [])
            if isinstance(fn, dict)
        ]
        for fn in frontier_functions[:3]:
            fn_name = str(fn.get("name") or "").strip()
            if not fn_name:
                continue
            fn_file = str(fn.get("file") or "").strip()
            fn_line = int(fn.get("line") or 0)
            fn_uncovered = int(fn.get("uncovered_regions_nearby") or 0)
            fn_ratio = float(fn.get("region_coverage_ratio") or 0.0)
            lines.append(
                "    - "
                f"{fn_name} ({fn_file}:{fn_line}) uncovered_regions={fn_uncovered}, "
                f"coverage_ratio={fn_ratio:.2f}"
            )
    top_frontier_functions = [
        dict(item)
        for item in list(frontier_summary.get("top_frontier_functions") or [])
        if isinstance(item, dict)
    ]
    if top_frontier_functions:
        lines.append("- per_input_frontier_inverse_index:")
        for item in top_frontier_functions[:5]:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            refs = [
                str(x).strip()
                for x in list(item.get("input_relpaths") or [])
                if str(x).strip()
            ]
            best_distance = int(item.get("best_distance_to_target") or 0)
            lines.append(f"    - {name}: inputs={len(refs)} [{', '.join(refs[:3])}], best_target_distance={best_distance}")
    pending = int(frontier_summary.get("pending_input_count") or 0)
    failed = int(frontier_summary.get("failed_input_count") or 0)
    if pending > 0 or failed > 0:
        lines.append(f"- per_input_frontier_status: pending={pending}, failed={failed}")
    return lines


def _build_run_feedback_summary(
    *,
    repo_root: Path,
    source_report: dict[str, Any],
    frontier_summary: dict[str, Any],
) -> dict[str, Any]:
    function_gaps: list[dict[str, Any]] = []
    for item in list(source_report.get("uncovered_function_details") or []):
        if not isinstance(item, dict):
            continue
        function_gaps.append(
            {
                "name": str(item.get("name") or "").strip(),
                "file": str(item.get("file") or "").strip(),
                "line": int(item.get("line") or 0),
                "kind": "uncovered",
                "execution_count": int(item.get("execution_count") or 0),
                "region_coverage_ratio": float(item.get("region_coverage_ratio") or 0.0),
            }
        )
    for item in list(source_report.get("partial_function_details") or []):
        if not isinstance(item, dict):
            continue
        function_gaps.append(
            {
                "name": str(item.get("name") or "").strip(),
                "file": str(item.get("file") or "").strip(),
                "line": int(item.get("line") or 0),
                "kind": "partial",
                "execution_count": int(item.get("execution_count") or 0),
                "region_coverage_ratio": float(item.get("region_coverage_ratio") or 0.0),
            }
        )
    function_gaps = [x for x in function_gaps if x["name"]]
    function_gaps.sort(
        key=lambda x: (
            0 if str(x.get("kind") or "") == "uncovered" else 1,
            float(x.get("region_coverage_ratio") or 0.0),
            str(x.get("name") or ""),
        )
    )

    path_frontiers: list[dict[str, Any]] = []
    for item in list(frontier_summary.get("top_inputs") or []):
        if not isinstance(item, dict):
            continue
        path_frontiers.append(
            {
                "input_relpath": str(item.get("input_relpath") or "").strip(),
                "frontier_score": float(item.get("frontier_score") or 0.0),
                "covered_function_count": int(item.get("covered_function_count") or 0),
                "covered_region_count": int(item.get("covered_region_count") or 0),
                "covered_functions_sample": [
                    str(x).strip()
                    for x in list(item.get("covered_functions_sample") or [])
                    if str(x).strip()
                ][:5],
                "frontier_functions": [
                    {
                        "name": str(fn.get("name") or "").strip(),
                        "file": str(fn.get("file") or "").strip(),
                        "line": int(fn.get("line") or 0),
                        "uncovered_regions_nearby": int(fn.get("uncovered_regions_nearby") or 0),
                        "region_coverage_ratio": float(fn.get("region_coverage_ratio") or 0.0),
                    }
                    for fn in list(item.get("frontier_functions") or [])
                    if isinstance(fn, dict) and str(fn.get("name") or "").strip()
                ][:5],
            }
        )
    path_frontiers = [x for x in path_frontiers if x["input_relpath"]]

    top_frontier_functions = []
    for item in list(frontier_summary.get("top_frontier_functions") or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        top_frontier_functions.append(
            {
                "name": name,
                "file": str(item.get("file") or "").strip(),
                "line": int(item.get("line") or 0),
                "input_count": int(item.get("input_count") or len(list(item.get("input_relpaths") or []))),
                "best_distance_to_target": int(item.get("best_distance_to_target") or 0),
                "input_relpaths": [
                    str(x).strip()
                    for x in list(item.get("input_relpaths") or [])
                    if str(x).strip()
                ][:5],
            }
        )

    files_index: dict[str, dict[str, Any]] = {}
    for item in function_gaps[:20]:
        file_path = str(item.get("file") or "").strip()
        if not file_path:
            continue
        bucket = files_index.setdefault(file_path, {"path": file_path, "issue_count": 0, "functions": []})
        bucket["issue_count"] = int(bucket.get("issue_count") or 0) + 1
        bucket["functions"].append({"name": item["name"], "line": item["line"], "kind": item["kind"]})
    for item in top_frontier_functions[:20]:
        file_path = str(item.get("file") or "").strip()
        if not file_path:
            continue
        bucket = files_index.setdefault(file_path, {"path": file_path, "issue_count": 0, "functions": []})
        bucket["issue_count"] = int(bucket.get("issue_count") or 0) + 1
        bucket["functions"].append({"name": item["name"], "line": item["line"], "kind": "frontier"})
    top_files = sorted(
        files_index.values(),
        key=lambda x: (-int(x.get("issue_count") or 0), str(x.get("path") or "")),
    )[:10]

    return {
        "repo_root": str(repo_root),
        "generated_at": int(time.time()),
        "function_gap_count": len(function_gaps),
        "path_frontier_count": len(path_frontiers),
        "frontier_function_count": len(top_frontier_functions),
        "top_function_gaps": function_gaps[:12],
        "top_path_frontiers": path_frontiers[:8],
        "top_frontier_functions": top_frontier_functions[:12],
        "top_files": top_files,
        "coverage_pct": float(source_report.get("coverage_pct") or 0.0),
        "covered_functions": int(source_report.get("covered_functions") or 0),
        "total_functions": int(source_report.get("total_functions") or 0),
    }


def _write_run_feedback_artifact(
    *,
    repo_root: Path,
    source_report: dict[str, Any],
    frontier_summary: dict[str, Any],
) -> dict[str, Any]:
    summary = _build_run_feedback_summary(
        repo_root=repo_root,
        source_report=source_report,
        frontier_summary=frontier_summary,
    )
    fuzz_dir = repo_root / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    path = fuzz_dir / "run_feedback.json"
    doc = {
        "schema_version": 1,
        "updated_at": int(time.time()),
        "summary": summary,
        "source_report": dict(source_report or {}),
        "frontier_summary": dict(frontier_summary or {}),
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"path": str(path), "summary": summary}


def _resolve_current_coverage_binary(repo_root: Path, state: FuzzWorkflowRuntimeState) -> Path | None:
    fuzz_out = repo_root / "fuzz" / "out"
    if not fuzz_out.is_dir():
        return None

    candidate_names: list[str] = []
    for detail in list(state.get("run_details") or []):
        name = str(detail.get("fuzzer") or "").strip()
        if name:
            candidate_names.append(Path(name).name)
    for raw in (
        state.get("last_fuzzer"),
        state.get("coverage_target_name"),
        state.get("selected_target_name"),
    ):
        name = str(raw or "").strip()
        if name:
            candidate_names.append(Path(name).name)

    seen: set[str] = set()
    for name in candidate_names:
        if not name or name in seen:
            continue
        seen.add(name)
        candidate = fuzz_out / name
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return candidate

    bins = sorted(
        p for p in fuzz_out.glob("*")
        if p.is_file() and os.access(str(p), os.X_OK)
    )
    return bins[0] if bins else None


def _replay_out_dir(repo_root: Path) -> Path:
    return repo_root / "fuzz" / "out" / "replay"


def _binary_looks_profile_instrumented(path: Path) -> bool:
    try:
        if not path.is_file() or not os.access(str(path), os.X_OK):
            return False
        data = path.read_bytes()
    except OSError:
        return False
    return any(
        marker in data
        for marker in (
            b"__llvm_prf",
            b"LLVM_PROFILE_FILE",
            b"__llvm_profile",
        )
    )


def _materialize_replay_binaries(repo_root: Path, bin_paths: list[Path]) -> list[Path]:
    replay_dir = _replay_out_dir(repo_root)
    replay_dir.mkdir(parents=True, exist_ok=True)
    fuzz_out = repo_root / "fuzz" / "out"
    primary_names = {
        p.name
        for p in bin_paths
        if p.is_file() and p.parent == fuzz_out
    }
    for stale in replay_dir.iterdir():
        if stale.is_file() and stale.name not in primary_names:
            try:
                stale.unlink()
            except OSError:
                pass

    created: list[Path] = []
    for name in sorted(primary_names):
        dest = replay_dir / name
        candidates = [
            dest,
            replay_dir / f"{name}.exe",
            fuzz_out / f"{name}_replay",
            fuzz_out / f"{name}-replay",
            fuzz_out / f"{name}.replay",
        ]
        src = next((p for p in candidates if _binary_looks_profile_instrumented(p)), None)
        if src is None:
            try:
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
            except OSError:
                pass
            continue
        if src == dest:
            created.append(dest)
            continue
        try:
            if dest.exists() or dest.is_symlink():
                dest.unlink()
        except OSError:
            pass
        try:
            rel_src = os.path.relpath(src, replay_dir)
            os.symlink(rel_src, dest)
        except OSError:
            shutil.copy2(src, dest)
            try:
                mode = src.stat().st_mode
                os.chmod(dest, mode)
            except OSError:
                pass
        if _binary_looks_profile_instrumented(dest):
            created.append(dest)
    return created


def _resolve_per_input_replay_binary(repo_root: Path, fuzzer_name: str) -> Path | None:
    replay_dir = _replay_out_dir(repo_root)
    fuzz_out = repo_root / "fuzz" / "out"
    candidates = [
        replay_dir / fuzzer_name,
        replay_dir / f"{fuzzer_name}.exe",
        fuzz_out / f"{fuzzer_name}_replay",
        fuzz_out / f"{fuzzer_name}-replay",
        fuzz_out / f"{fuzzer_name}.replay",
    ]
    for candidate in candidates:
        if _binary_looks_profile_instrumented(candidate):
            return candidate
    return None


def _aggregate_seed_quality_from_run_details(
    run_details: list[dict[str, Any]],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    quality_docs: list[dict[str, Any]] = [
        dict(detail.get("seed_quality") or {})
        for detail in (run_details or [])
        if isinstance(detail.get("seed_quality"), dict) and detail.get("seed_quality")
    ]
    if not quality_docs:
        return dict(fallback or {})

    merged = dict(quality_docs[0])

    def _min_float(key: str) -> None:
        vals: list[float] = []
        for doc in quality_docs:
            try:
                vals.append(float(doc.get(key) or 0.0))
            except Exception:
                continue
        if vals:
            merged[key] = min(vals)

    def _max_float(key: str) -> None:
        vals: list[float] = []
        for doc in quality_docs:
            try:
                vals.append(float(doc.get(key) or 0.0))
            except Exception:
                continue
        if vals:
            merged[key] = max(vals)

    def _min_int(key: str) -> None:
        vals: list[int] = []
        for doc in quality_docs:
            try:
                vals.append(int(doc.get(key) or 0))
            except Exception:
                continue
        if vals:
            merged[key] = min(vals)

    def _max_int(key: str) -> None:
        vals: list[int] = []
        for doc in quality_docs:
            try:
                vals.append(int(doc.get(key) or 0))
            except Exception:
                continue
        if vals:
            merged[key] = max(vals)

    _min_float("seed_score")
    _min_float("merge_retained_ratio_files")
    _min_float("merge_retained_ratio_bytes")
    _min_int("early_new_units_30s")
    _min_int("early_new_units_60s")
    _max_int("initial_inited_cov")
    _max_int("final_cov")
    _max_int("cov_delta")
    _max_int("initial_inited_ft")
    _max_int("final_ft")
    _max_int("ft_delta")
    _max_int("initial_corpus_files")
    _max_int("final_corpus_files")
    merged["cold_start_failure"] = any(bool(doc.get("cold_start_failure") or False) for doc in quality_docs)

    all_flags: set[str] = set()
    for doc in quality_docs:
        for flag in list(doc.get("quality_flags") or []):
            sval = str(flag or "").strip()
            if sval:
                all_flags.add(sval)
    if all_flags:
        merged["quality_flags"] = sorted(all_flags)

    return merged


def _seed_quality_from_run_details_for_target(
    run_details: list[dict[str, Any]],
    fallback: dict[str, Any],
    *,
    target_name: str = "",
    target_api: str = "",
    fuzzer_name: str = "",
) -> dict[str, Any]:
    matched = _matching_run_details_for_target(
        run_details,
        target_name=target_name,
        target_api=target_api,
        fuzzer_name=fuzzer_name,
    )
    if matched:
        return _aggregate_seed_quality_from_run_details(matched, fallback)
    return _aggregate_seed_quality_from_run_details(run_details, fallback)


def _quality_flags_from_seed_quality(seed_quality: dict[str, Any]) -> list[str]:
    flags: set[str] = set()
    for flag in list(seed_quality.get("quality_flags") or []):
        sval = str(flag or "").strip()
        if sval:
            flags.add(sval)
    return sorted(flags)


def _build_harness_feedback(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "execution_plan_path": str(state.get("execution_plan_path") or ""),
        "harness_index_path": str(state.get("harness_index_path") or ""),
        "selected_target_api": str(state.get("selected_target_api") or ""),
        "coverage_target_api": str(state.get("coverage_target_api") or ""),
        "coverage_frontier_path": str(state.get("coverage_frontier_path") or ""),
        "coverage_frontier_summary": dict(state.get("coverage_frontier_summary") or {}),
        "coverage_replay_error": str(state.get("coverage_replay_error") or ""),
        "missing_execution_targets": list(state.get("coverage_missing_execution_targets") or []),
        "built_targets": list(state.get("built_targets") or []),
        "missing_targets": list(state.get("missing_targets") or []),
        "target_build_matrix": list(state.get("target_build_matrix") or []),
    }


def _slug_from_repo_url(repo_url: str) -> str:
    return _wf_common.slug_from_repo_url(repo_url)


def _alloc_output_workdir(repo_url: str) -> Path | None:
    return _wf_common.alloc_output_workdir(repo_url)


def _remaining_time_budget_sec(state: FuzzWorkflowRuntimeState, *, min_timeout: int = 5) -> int:
    return _wf_common.remaining_time_budget_sec(cast(dict[str, Any], state), min_timeout=min_timeout)


def _opencode_cli_retries() -> int:
    raw = (os.environ.get("SHERPA_WORKFLOW_OPENCODE_CLI_RETRIES") or "2").strip()
    try:
        return max(1, min(int(raw), 8))
    except Exception:
        return 2


def _analysis_opencode_advisory_enabled() -> bool:
    raw = (os.environ.get("SHERPA_ANALYSIS_OPENCODE_ADVISORY_ENABLED") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _analysis_opencode_timeout_sec(state: FuzzWorkflowRuntimeState) -> int:
    raw = (os.environ.get("SHERPA_ANALYSIS_OPENCODE_TIMEOUT_SEC") or "120").strip()
    try:
        configured = max(15, min(int(raw), 900))
    except Exception:
        configured = 120
    remaining = _remaining_time_budget_sec(state)
    return max(15, min(configured, remaining))


def _analysis_opencode_idle_timeout_sec() -> int:
    raw = (os.environ.get("SHERPA_ANALYSIS_OPENCODE_IDLE_TIMEOUT_SEC") or "300").strip()
    try:
        return max(10, min(int(raw), 1200))
    except Exception:
        return 300


def _vuln_hunt_idle_timeout_sec() -> int:
    raw = (os.environ.get("SHERPA_OPENCODE_IDLE_TIMEOUT_VULN_HUNT_SEC") or "1800").strip()
    try:
        return max(60, min(int(raw), 7200))
    except Exception:
        return 1800


def _plan_idle_timeout_sec() -> int:
    raw = (os.environ.get("SHERPA_OPENCODE_IDLE_TIMEOUT_PLAN_SEC") or "1200").strip()
    try:
        return max(60, min(int(raw), 7200))
    except Exception:
        return 1200


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
    raw = (os.environ.get("SHERPA_OPENCODE_IDLE_TIMEOUT_SYNTH_SEC") or "600").strip()
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


def _cold_start_seed_replan_quality_threshold() -> float:
    raw = (os.environ.get("SHERPA_RUN_COLD_START_SEED_REPLAN_QUALITY_THRESHOLD") or "0.55").strip()
    try:
        return max(0.0, min(float(raw), 1.0))
    except Exception:
        return 0.55


def _cold_start_seed_replan_early_units_30s_threshold() -> int:
    raw = (os.environ.get("SHERPA_RUN_COLD_START_SEED_REPLAN_EARLY_UNITS_30S_THRESHOLD") or "0").strip()
    try:
        return max(0, min(int(raw), 1_000_000))
    except Exception:
        return 0


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
    # Allow up to 4 workers – modern pods typically have ≥4 CPU and ASAN
    # overhead is ~2×, so 4 workers still fit within typical memory budgets.
    if sanitizer_l in {"address", "memory", "thread"}:
        inner_cap = max(1, min(cpu, 4))
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


def _procedural_stage_for_prompt(name: str) -> str:
    """Map an opencode prompt/template name to its pipeline stage for
    procedural-memory lesson retrieval."""
    n = str(name or "").lower()
    if "synthesize" in n:
        return "synthesize"
    if "plan" in n:
        return "plan"
    if "fix_build" in n or "build" in n:
        return "build"
    if "vuln" in n or "hunt" in n:
        return "vuln-hunt"
    if "analysis" in n:
        return "analysis"
    if "crash" in n:
        return "crash-triage"
    return ""


def _inject_procedural_lessons(name: str, kwargs: dict[str, object]) -> None:
    """Phase 3 read path: prepend learned 'known pitfalls' for this stage into
    the prompt hint, so cross-job lessons steer the agent away from repeating
    past failures. No-op unless SHERPA_PROCEDURAL_MEMORY is enabled, only when
    the template already takes a `hint`, and never raises."""
    try:
        if "hint" not in kwargs or not _proc_mem.memory_enabled():
            return
        stage = _procedural_stage_for_prompt(name)
        if not stage:
            return
        lessons = _proc_mem.retrieve(stage=stage, library_class="")
        block = _proc_mem.render_lessons_block(lessons)
        if not block:
            return
        base = str(kwargs.get("hint") or "").strip()
        kwargs["hint"] = (block + "\n\n" + base).strip() if base else block
    except Exception:
        return


def _render_opencode_prompt_safe(
    name: str,
    *,
    fallback_name: str = "",
    fallback_hint: str = "",
    known_issues: list[str] | None = None,
    **kwargs: object,
) -> tuple[str, str]:
    """
    Render prompt templates with a non-throwing fallback path.

    Returns:
      (rendered_prompt, render_issue)
      render_issue is empty when primary render succeeds.
    """
    _inject_procedural_lessons(name, kwargs)
    try:
        return _render_opencode_prompt(name, **kwargs), ""
    except Exception as e:
        issue = f"prompt-render:{name} failed: {e}"
        merged_issues = [str(x).strip() for x in (known_issues or []) if str(x).strip()]
        merged_issues.append(issue)
        fallback_issue_block = "Known Issues:\n" + "\n".join(f"- {x}" for x in merged_issues)
        hint_txt = str(fallback_hint or kwargs.get("hint") or "").strip()
        degraded_hint = (hint_txt + "\n\n" + fallback_issue_block).strip() if hint_txt else fallback_issue_block
        if fallback_name:
            try:
                return _render_opencode_prompt(fallback_name, hint=degraded_hint), issue
            except Exception as e2:
                issue = f"{issue}; fallback={fallback_name} failed: {e2}"
        # Final fallback: plain prompt text to avoid hard crash in plan/synthesize nodes.
        return (
            (
                "Template render degraded. Continue with repair planning using current diagnostics.\n\n"
                f"{fallback_issue_block}\n\n"
                "Do not run commands. Read diagnostics first and output concrete file-level changes."
            ),
            issue,
        )


def _attach_prompt_render_status(
    out: dict[str, Any],
    *,
    issue: str = "",
) -> dict[str, Any]:
    issue_text = str(issue or "").strip()
    if issue_text:
        prev = str(out.get("prompt_render_issue") or "").strip()
        merged = issue_text
        if prev and issue_text not in prev:
            merged = f"{prev}; {issue_text}"
        out["prompt_render_degraded"] = True
        out["prompt_render_issue"] = merged[:4096]
        for snapshot_key in ("latest_decision_snapshot", "latest_vuln_decision_snapshot"):
            snapshot = out.get(snapshot_key)
            if not isinstance(snapshot, dict):
                continue
            snapshot_doc = dict(snapshot)
            degraded_prev = str(snapshot_doc.get("degraded_reason") or "").strip()
            if not degraded_prev:
                snapshot_doc["degraded_reason"] = issue_text
            elif issue_text not in degraded_prev:
                snapshot_doc["degraded_reason"] = f"{degraded_prev}; {issue_text}"
            out[snapshot_key] = snapshot_doc
        return out
    out["prompt_render_degraded"] = bool(out.get("prompt_render_degraded") or False)
    out["prompt_render_issue"] = str(out.get("prompt_render_issue") or "")
    return out


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
        executor = None
        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
            result = {}

            def _get_parser_worker():
                tslp = importlib.import_module("tree_sitter_language_pack")
                get_parser = getattr(tslp, "get_parser", None)
                if callable(get_parser):
                    result["parser"] = get_parser(language)

            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_get_parser_worker)
            future.result(timeout=timeout_sec)
            return result.get("parser")
        except (FuturesTimeoutError, Exception):
            return None
        finally:
            if executor is not None:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    executor.shutdown(wait=False)
                except Exception:
                    pass

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
                                "security_signals": [],
                                "security_signal_scores": _empty_security_scores(),
                                "vuln_likelihood": 0.0,
                                "exploitability": 0.0,
                                "reachability_confidence": 0.0,
                                "security_priority_reason": "",
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

    tree_sitter_opt_in = str(os.environ.get("SHERPA_TARGET_ANALYSIS_TREE_SITTER") or "").strip().lower()
    tree_sitter_enabled = (
        tree_sitter_opt_in in {"1", "true", "yes", "on"}
        and importlib.util.find_spec("tree_sitter_language_pack") is not None
    )
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
        {"id": "mem_oob_candidate", "pattern": r"(memcpy|memmove|strcpy|strncpy|strcat|strncat|offset|index|bounds?)"},
        {"id": "integer_overflow_candidate", "pattern": r"(overflow|underflow|size_t|uint|int32_t|int64_t|length|count|\*)"},
        {"id": "format_string_candidate", "pattern": r"(printf|fprintf|sprintf|snprintf|vsnprintf|vprintf|format|string_format|fmt::)"},
        {"id": "path_traversal_candidate", "pattern": r"(path|filepath|filename|fopen|open\(|readfile|writefile|\.\./)"},
        {"id": "command_injection_candidate", "pattern": r"(system\(|popen\(|exec\(|spawn\(|shell|command)"},
        {"id": "authz_bypass_candidate", "pattern": r"(auth|authorize|permission|acl|role|token|session|bypass|skip[_-]?check)"},
        {"id": "null_deref_candidate", "pattern": r"(null|nullptr|optional|dereference|->)"},
        {"id": "uaf_candidate", "pattern": r"(free\(|delete|release|destroy|dispose|dangling)"},
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
            regex_signals = [
                rule["id"] for rule in semgrep_rules if re.search(rule["pattern"], f"{name}\n{signature}", re.IGNORECASE)
            ]
            weak_file_signals = [
                rule_id for rule_id in semgrep_hits.get(rel, []) if rule_id not in regex_signals
            ]
            risk_signals = list(regex_signals)
            candidate_functions.append(
                {
                    "name": name,
                    "signature": signature,
                    "file": rel,
                    "line": line_no,
                    "target_type": "pending",
                    "seed_profile": "pending",
                    "risk_signals": risk_signals,
                    "risk_signal_source_breakdown": {
                        "regex": list(regex_signals),
                        "weak_file": list(weak_file_signals),
                    },
                    "security_signals": [],
                    "security_signal_scores": _empty_security_scores(),
                    "vuln_likelihood": 0.0,
                    "exploitability": 0.0,
                    "reachability_confidence": 0.0,
                    "security_priority_reason": "",
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
        security_scores = _compute_security_signal_scores(
            name=str(item.get("name") or ""),
            signature=str(item.get("signature") or ""),
            file_hint=str(item.get("file") or ""),
            risk_signals=list(item.get("risk_signals") or []),
            risk_signal_source_breakdown=dict(item.get("risk_signal_source_breakdown") or {}),
        )
        vuln_likelihood, exploitability, reachability_confidence, security_reason = _derive_security_priority(
            target_type=str(item.get("target_type") or "generic"),
            runtime_viability=runtime_viability,
            security_scores=security_scores,
        )
        item["security_signal_scores"] = security_scores
        item["security_signals"] = _top_security_signals(security_scores)
        item["vuln_likelihood"] = vuln_likelihood
        item["exploitability"] = exploitability
        item["reachability_confidence"] = reachability_confidence
        item["security_priority_reason"] = security_reason

    if _vuln_hunting_enabled() and _vuln_score_mode() == "risk_first_v1":
        candidate_functions.sort(
            key=lambda item: (
                float(item.get("vuln_likelihood") or 0.0),
                float(item.get("exploitability") or 0.0),
                float(item.get("reachability_confidence") or 0.0),
                {"high": 2, "medium": 1, "low": 0}.get(str(item.get("runtime_viability") or "").lower(), 0),
                int(item.get("depth_score") or 0),
                len(list(item.get("risk_signals") or [])),
                str(item.get("name") or ""),
            ),
            reverse=True,
        )
    else:
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
                "risk_signal_source_breakdown": dict(item.get("risk_signal_source_breakdown") or {}),
                "file": str(item.get("file") or ""),
                "depth_score": int(item.get("depth_score") or 0),
                "depth_class": str(item.get("depth_class") or "shallow"),
                "selection_bias_reason": str(item.get("selection_bias_reason") or ""),
                "runtime_viability": str(item.get("runtime_viability") or ""),
                "selection_rationale": str(item.get("selection_rationale") or ""),
                "runtime_replacement_candidates": list(item.get("runtime_replacement_candidates") or []),
                "security_signals": list(item.get("security_signals") or []),
                "security_signal_scores": dict(item.get("security_signal_scores") or {}),
                "vuln_likelihood": float(item.get("vuln_likelihood") or 0.0),
                "exploitability": float(item.get("exploitability") or 0.0),
                "reachability_confidence": float(item.get("reachability_confidence") or 0.0),
                "security_priority_reason": str(item.get("security_priority_reason") or ""),
                "vuln_hunting_enabled": bool(_vuln_hunting_enabled()),
                "vuln_focus_profile": "broad_high_risk",
                "target_surface_policy": "risk_first",
            }
        )
        if len(recommended_targets) >= _vuln_topk():
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
        "vuln_hunting_enabled": bool(_vuln_hunting_enabled()),
        "vuln_focus_profile": "broad_high_risk",
        "target_surface_policy": "risk_first",
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


def _materialize_analysis_context_from_companion(
    *,
    repo_root: Path,
    antlr_context_path: str,
    antlr_context_summary: str,
    target_analysis_path: str,
    target_analysis_summary: str,
) -> tuple[str, int, int, int, str]:
    companion_doc, companion_summary = _collect_analysis_companion_context()
    artifacts = dict(companion_doc.get("artifacts") or {})
    found_artifacts = sum(
        1
        for name in ("status.json", "preprocess.json", "coverage_hints.json")
        if isinstance(artifacts.get(name), dict) and bool((artifacts.get(name) or {}).get("exists"))
    )
    if found_artifacts <= 0:
        return "", 0, 0, 0, companion_summary

    antlr_doc = _read_json_doc(antlr_context_path)
    target_doc = _read_json_doc(target_analysis_path)
    evidence_doc = _build_analysis_evidence_index(
        repo_root=repo_root,
        antlr_doc=antlr_doc,
        target_doc=target_doc,
        companion_doc=companion_doc,
    )
    fuzz_dir = repo_root / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    analysis_doc = {
        "mode": "companion-fallback-analysis",
        "generated_at": int(time.time()),
        "repo_root": str(repo_root),
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
    summary = dict(evidence_doc.get("summary") or {})
    return (
        str(analysis_path),
        int(summary.get("evidence_count") or 0),
        int(summary.get("security_evidence_count") or 0),
        int(summary.get("vuln_candidate_count") or 0),
        companion_summary,
    )


def _build_analysis_evidence_index(
    *,
    repo_root: Path,
    antlr_doc: dict[str, Any],
    target_doc: dict[str, Any],
    companion_doc: dict[str, Any],
) -> dict[str, Any]:
    evidence_counter = 0
    candidate_counter = 0
    evidence_index: dict[str, dict[str, Any]] = {}
    security_evidence: list[dict[str, Any]] = []
    vuln_candidate_inventory: list[dict[str, Any]] = []
    min_confidence = _vuln_min_evidence_confidence()

    def _new_evidence_id() -> str:
        nonlocal evidence_counter
        evidence_counter += 1
        return f"EV{evidence_counter:04d}"

    def _new_candidate_id(signal_id: str) -> str:
        nonlocal candidate_counter
        candidate_counter += 1
        return f"{_signal_slug(signal_id)}_{candidate_counter:03d}"

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
        security_scores = _extract_security_scores(item)
        security_signals = list(item.get("security_signals") or _top_security_signals(security_scores))
        if not security_signals:
            security_signals = _top_security_signals(security_scores, threshold=min_confidence)
        candidate_evidence_ids: list[str] = [ev_id]
        for signal_id in security_signals:
            try:
                signal_score = max(0.0, min(float(security_scores.get(signal_id) or 0.0), 1.0))
            except Exception:
                signal_score = 0.0
            if signal_score < min_confidence:
                continue
            source_line = item.get("line")
            try:
                source_line_int = int(source_line) if source_line is not None else 0
            except Exception:
                source_line_int = 0
            sec_ev_id = _add_evidence(
                kind="security_signal",
                source_path=file_hint or "fuzz/target_analysis.json",
                summary=f"security signal `{signal_id}` on `{api}`",
                score=signal_score,
                payload={
                    "api": api,
                    "signal_id": signal_id,
                    "security_priority_reason": str(item.get("security_priority_reason") or ""),
                    "target_type": target_type or "generic",
                },
            )
            candidate_evidence_ids.append(sec_ev_id)
            security_evidence.append(
                {
                    "evidence_id": sec_ev_id,
                    "target_api": api,
                    "signal_id": signal_id,
                    "severity": "high" if signal_score >= 0.75 else ("medium" if signal_score >= 0.55 else "low"),
                    "confidence": round(signal_score, 4),
                    "source_path": file_hint or "fuzz/target_analysis.json",
                    "line": source_line_int,
                    "summary": f"`{api}` matched {signal_id} (score={signal_score:.2f})",
                }
            )
        primary_signal = security_signals[0] if security_signals else "mem_oob_candidate"
        evidence_refs = [{"evidence_id": ref} for ref in list(dict.fromkeys(candidate_evidence_ids))]
        attack_hint = _candidate_attack_hint(
            api=api,
            target_type=target_type or "generic",
            signal_id=primary_signal,
            source_path=file_hint,
            security_reason=str(item.get("security_priority_reason") or ""),
        )
        vuln_likelihood = float(item.get("vuln_likelihood") or 0.0)
        exploitability = float(item.get("exploitability") or 0.0)
        reachability_confidence = float(item.get("reachability_confidence") or 0.0)
        primary_signal_score = float(security_scores.get(primary_signal) or 0.0)
        vuln_candidate_inventory.append(
            {
                "candidate_id": _new_candidate_id(primary_signal),
                "api": api,
                "name": str(item.get("name") or api),
                "file": file_hint,
                "target_type": target_type or "generic",
                "target_api": api,
                "target_file": file_hint,
                "signal_type": primary_signal,
                "signal_score": round(primary_signal_score, 4),
                "evidence": evidence_refs,
                "attack_hint": attack_hint,
                "candidate_origin": "analysis_context",
                "validation_status": "pending",
                "vuln_likelihood": vuln_likelihood,
                "exploitability": exploitability,
                "reachability_confidence": reachability_confidence,
                "priority": _candidate_priority(
                    vuln_likelihood=vuln_likelihood,
                    exploitability=exploitability,
                    reachability_confidence=reachability_confidence,
                    evidence_count=len(evidence_refs),
                    signal_score=primary_signal_score,
                ),
                "evidence_ids": list(dict.fromkeys(candidate_evidence_ids)),
                "security_signal_scores": {k: float(v) for k, v in security_scores.items()},
                "risk_signal_source_breakdown": dict(item.get("risk_signal_source_breakdown") or {}),
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
        "security_evidence": security_evidence,
        "vuln_candidate_inventory": vuln_candidate_inventory,
        "evidence_index": evidence_index,
        "summary": {
            "evidence_count": len(evidence_index),
            "api_inventory_count": len(api_inventory),
            "callgraph_summary_count": len(callgraph_summary),
            "consumer_pattern_count": len(consumer_patterns),
            "semantic_evidence_count": len(semantic_evidence),
            "security_evidence_count": len(security_evidence),
            "vuln_candidate_count": len(vuln_candidate_inventory),
            "security_mode": "risk_first_v1",
            "vuln_focus_profile": "broad_high_risk",
            "target_surface_policy": "risk_first",
        },
    }


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


def _max_cov_from_run_details(run_details: list[dict[str, Any]]) -> int:
    covs: list[int] = []
    for detail in run_details or []:
        try:
            covs.append(int(detail.get("final_cov") or 0))
        except Exception:
            continue
    return max(covs) if covs else 0


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


def _crash_vuln_status(*, stage: str, classification: str) -> str:
    stage_l = str(stage or "").strip().lower()
    cls = str(classification or "").strip().lower()
    if stage_l == "crash-analysis":
        if cls == "real_bug":
            return "real_bug"
        if cls == "false_positive":
            return "false_positive"
        return "inconclusive"
    if cls == "upstream_bug":
        return "likely_bug"
    if cls == "harness_bug":
        return "false_positive"
    return "inconclusive"


def _crash_vuln_confidence(*, status: str, raw_confidence: float) -> float:
    status_l = str(status or "").strip().lower()
    base = max(0.0, min(float(raw_confidence or 0.0), 1.0))
    floor_by_status = {
        "real_bug": 0.80,
        "likely_bug": 0.65,
        "false_positive": 0.70,
        "inconclusive": 0.35,
    }
    return max(base, float(floor_by_status.get(status_l, 0.35)))


def _crash_vuln_sanitizer_signal(*texts: str) -> tuple[str, str]:
    joined = "\n".join(str(t or "") for t in texts if str(t or "").strip())
    m = re.search(
        r"ERROR:\s*(Address|Undefined|Memory|Thread|Leak)Sanitizer:\s*([^\s]+)",
        joined,
        flags=re.IGNORECASE,
    )
    if m:
        sanitizer = f"{m.group(1)}Sanitizer"
        crash_type = str(m.group(2) or "unknown").strip()
        return sanitizer, crash_type
    m = re.search(
        r"SUMMARY:\s*(Address|Undefined|Memory|Thread|Leak)Sanitizer:\s*([^\s]+)",
        joined,
        flags=re.IGNORECASE,
    )
    if m:
        sanitizer = f"{m.group(1)}Sanitizer"
        crash_type = str(m.group(2) or "unknown").strip()
        return sanitizer, crash_type
    return "", str("unknown")


def _crash_vuln_selected_target(repo_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    fuzzer = str(state.get("last_fuzzer") or "").strip()
    selected = _load_selected_targets_doc(repo_root)
    if fuzzer:
        for item in selected:
            if str(item.get("wrapper_fuzzer_name") or "").strip() == fuzzer:
                return dict(item)
            if str(item.get("target_name") or item.get("name") or "").strip() == fuzzer:
                return dict(item)
    return dict(selected[0]) if selected else {}


def _write_crash_vuln_candidate(
    repo_root: Path,
    state: dict[str, Any],
    *,
    stage: str,
    classification: str,
    reason: str,
    evidence: list[str],
    confidence: float,
    triage_label: str = "",
    analysis_verdict: str = "",
    info_text: str = "",
    runtime_text: str = "",
) -> dict[str, Any]:
    fuzz_dir = repo_root / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    path = fuzz_dir / "vuln_candidates.json"
    report_path = repo_root / "crash_vuln_report.md"

    selected_target = _crash_vuln_selected_target(repo_root, state)
    attack_hint = dict(selected_target.get("attack_hint") or {})
    sanitizer, crash_type = _crash_vuln_sanitizer_signal(
        info_text,
        runtime_text,
        "\n".join(evidence),
        str(state.get("crash_stack_type") or ""),
    )
    if not sanitizer and str(state.get("crash_stack_type") or "").strip():
        crash_type = str(state.get("crash_stack_type") or "").strip()

    signature = str(state.get("crash_signature") or state.get("crash_stack_signature") or "").strip()
    signature_short = re.sub(r"[^A-Za-z0-9]+", "", signature)[:12] or "unknown"
    status = _crash_vuln_status(stage=stage, classification=classification)
    candidate = {
        "candidate_id": f"crash_{signature_short}",
        "source_stage": stage,
        "validation_status": status,
        "classification": classification,
        "confidence": _crash_vuln_confidence(status=status, raw_confidence=confidence),
        "reason": str(reason or "").strip(),
        "evidence": [str(x).strip() for x in list(evidence or []) if str(x).strip()],
        "target_api": str(
            selected_target.get("target_api")
            or selected_target.get("api")
            or state.get("selected_target_api")
            or state.get("coverage_target_api")
            or ""
        ).strip(),
        "target_name": str(
            selected_target.get("target_name")
            or selected_target.get("name")
            or state.get("coverage_target_name")
            or ""
        ).strip(),
        "target_file": str(selected_target.get("target_file") or selected_target.get("file") or ""),
        "fuzzer": str(state.get("last_fuzzer") or ""),
        "artifact": str(state.get("last_crash_artifact") or ""),
        "crash_signature": signature,
        "sanitizer": sanitizer,
        "crash_type": crash_type,
        "triage_label": triage_label,
        "analysis_verdict": analysis_verdict,
        "reproduction_status": "reproduced" if bool(state.get("re_run_ok") or state.get("crash_repro_ok")) else "observed",
        "attack_hint": attack_hint,
        "created_at": int(time.time()),
    }

    doc: dict[str, Any] = {"schema_version": 1, "candidates": []}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(loaded, dict):
                doc = loaded
        except Exception:
            doc = {"schema_version": 1, "candidates": []}
    candidates = [dict(x) for x in list(doc.get("candidates") or []) if isinstance(x, dict)]
    candidates = [x for x in candidates if str(x.get("candidate_id") or "") != candidate["candidate_id"]]
    candidates.append(candidate)
    doc = {
        "schema_version": 1,
        "updated_at": int(time.time()),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "# Crash Vulnerability Candidate Report",
        "",
        f"- candidate_id: {candidate['candidate_id']}",
        f"- validation_status: {candidate['validation_status']}",
        f"- classification: {candidate['classification']}",
        f"- confidence: {float(candidate['confidence']):.2f}",
        f"- target_api: {candidate['target_api'] or '(unknown)'}",
        f"- fuzzer: {candidate['fuzzer'] or '(unknown)'}",
        f"- sanitizer: {candidate['sanitizer'] or '(unknown)'}",
        f"- crash_type: {candidate['crash_type'] or '(unknown)'}",
        f"- crash_signature: {candidate['crash_signature'] or '(unknown)'}",
        "",
        "## Reason",
        "",
        candidate["reason"] or "(none)",
        "",
        "## Evidence",
        "",
    ]
    report_lines.extend([f"- {line}" for line in candidate["evidence"]] or ["- (none)"])
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "candidate": candidate,
        "path": str(path),
        "report_path": str(report_path),
        "candidate_count": len(candidates),
    }


def _re_restart_limit() -> int:
    raw = (os.environ.get("SHERPA_RESTART_FROM_PLAN_MAX") or "1").strip()
    try:
        return max(0, min(int(raw), 10))
    except Exception:
        return 1


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
