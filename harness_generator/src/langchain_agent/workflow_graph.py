from __future__ import annotations

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
    snapshot_repo_text,
    write_patch_from_snapshot,
)


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
    coverage_seed_family_coverage: dict[str, Any]
    antlr_context_path: str
    antlr_context_summary: str
    target_analysis_path: str
    target_analysis_summary: str
    selected_targets_path: str
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
    next: str
    fix_patch_path: str
    fix_patch_files: list[str]
    fix_patch_bytes: int
    summary_path: str
    summary_json_path: str
    plan_fix_on_crash: bool
    plan_max_fix_rounds: int


class FuzzWorkflowRuntimeState(FuzzWorkflowState, total=False):
    generator: NonOssFuzzHarnessGenerator
    crash_found: bool
    message: str


def _wf_log(state: dict[str, Any] | None, msg: str) -> None:
    _wf_common.wf_log(state, msg)


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
    if any(tok in text for tok in ("parse", "parser", "scan", "scanner", "yaml", "json", "xml", "token", "lex", "reader")):
        return "parser"
    if any(tok in text for tok in ("decode", "decoder", "decompress", "inflate", "unpack")):
        return "decoder"
    if any(tok in text for tok in ("archive", "untar", "unzip", "tar", "zip", "rar", "7z")):
        return "archive"
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


def _clear_opencode_done_sentinel(repo_root: Path) -> bool:
    done_path = _opencode_done_path(repo_root)
    if not done_path.exists():
        return False
    try:
        done_path.unlink()
        return True
    except Exception:
        return False


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
        if any(tok in text for tok in ("token", "lexer", "lex", "scan", "scanner")):
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
    if any(tok in text for tok in ("println", "print(", "format_to", "vformat", "fmt::format", "fmt::print", "fmt::println")):
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


def _build_selected_targets_doc(repo_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
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
        out.append(
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
            }
        )
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
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_:]*)\s*\(", text):
        name = str(match.group(1) or "").strip()
        lowered = name.lower()
        leaf = lowered.split("::")[-1]
        if not lowered or leaf in keywords:
            continue
        if leaf == "llvmfuzzertestoneinput":
            continue
        return lowered
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
        target_type = str(item.get("target_type") or _infer_target_type(name, file_hint))
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
                "seed_profile": str(item.get("seed_profile") or _infer_seed_profile(name, file_hint, target_type=target_type)),
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


def _slug_from_repo_url(repo_url: str) -> str:
    return _wf_common.slug_from_repo_url(repo_url)


def _alloc_output_workdir(repo_url: str) -> Path | None:
    return _wf_common.alloc_output_workdir(repo_url)


def _enter_step(state: FuzzWorkflowRuntimeState, step_name: str) -> tuple[FuzzWorkflowRuntimeState, bool]:
    out, stop = _wf_common.enter_step(cast(dict[str, Any], state), step_name)
    return cast(FuzzWorkflowRuntimeState, out), stop


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
    raw = (os.environ.get("SHERPA_FIX_BUILD_MAX_ATTEMPTS") or "8").strip()
    try:
        return max(1, min(int(raw), 50))
    except Exception:
        return 8


def _effective_max_fix_rounds(state: FuzzWorkflowRuntimeState) -> int:
    configured = int(state.get("max_fix_rounds") or 0)
    if configured > 0:
        return max(1, min(configured, 20))
    return _fix_build_max_attempts()


def _effective_same_error_retry_limit(state: FuzzWorkflowRuntimeState) -> int:
    if "same_error_max_retries" in state:
        configured = int(state.get("same_error_max_retries") or 0)
    else:
        configured = 1
    return max(0, min(configured, 10))


def _fix_build_feedback_history_limit() -> int:
    raw = (os.environ.get("SHERPA_FIX_BUILD_FEEDBACK_HISTORY") or "6").strip()
    try:
        return max(1, min(int(raw), 30))
    except Exception:
        return 6


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
    raw = (os.environ.get("SHERPA_OPENCODE_IDLE_TIMEOUT_SYNTH_SEC") or "900").strip()
    try:
        return max(0, min(int(raw), 86_400))
    except Exception:
        return 900


def _synthesize_activity_watch_paths() -> list[str]:
    return [
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

    def _extract_tree_sitter_functions(path: Path, rel: str) -> list[dict[str, Any]]:
        try:
            tslp = importlib.import_module("tree_sitter_language_pack")
            get_parser = getattr(tslp, "get_parser", None)
            if not callable(get_parser):
                return []
            language = _ext_to_ts_language(path.suffix)
            if not language:
                return []
            parser = get_parser(language)
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
                        target_type = _infer_target_type(name, rel)
                        out.append(
                            {
                                "name": name,
                                "signature": " ".join(snippet.split())[:240],
                                "file": rel,
                                "line": int(getattr(node, "start_point", (0, 0))[0]) + 1,
                                "target_type": target_type,
                                "seed_profile": _infer_seed_profile(name, snippet, target_type=target_type),
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
            return False, {}
        tmp_path = ""
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
            proc = subprocess.run(
                [semgrep_bin, "scan", "--json", "--config", tmp_path, str(root)],
                capture_output=True,
                text=True,
                timeout=90,
            )
            if proc.returncode not in {0, 1}:
                return True, {}
            doc = json.loads(proc.stdout or "{}")
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
            return True, result_map
        except Exception:
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
            target_type = _infer_target_type(name, rel)
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
                    "target_type": target_type,
                    "seed_profile": _infer_seed_profile(name, signature, target_type=target_type),
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
    coverage_loop_max_rounds: int = 3
    max_fix_rounds: int = 3
    same_error_max_retries: int = 1


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
    max_len = int(state.get("max_len") or 1024)
    docker_image = (state.get("docker_image") or "").strip() or None
    codex_cli = (os.environ.get("SHERPA_CODEX_CLI") or os.environ.get("CODEX_CLI") or "opencode").strip()

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
            "max_steps": int(state.get("max_steps") or 10),
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
            "same_error_max_retries": max(0, min(int(state.get("same_error_max_retries") or 1), 10)),
            "build_error_kind": "",
            "build_error_code": "",
            "build_error_signature_short": "",
            "build_attempts": int(state.get("build_attempts") or 0),
            "fix_build_attempts": int(state.get("fix_build_attempts") or 0),
            "max_fix_rounds": max(0, min(int(state.get("max_fix_rounds") or 3), 20)),
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
            "plan_fix_on_crash": True,
            "plan_max_fix_rounds": 1,
            "coverage_loop_max_rounds": max(1, min(int(state.get("coverage_loop_max_rounds") or 3), 5)),
            "coverage_loop_round": int(state.get("coverage_loop_round") or 0),
            "coverage_should_improve": bool(state.get("coverage_should_improve") or False),
            "coverage_improve_reason": str(state.get("coverage_improve_reason") or ""),
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
            "antlr_context_path": str(state.get("antlr_context_path") or ""),
            "antlr_context_summary": str(state.get("antlr_context_summary") or ""),
            "target_analysis_path": str(state.get("target_analysis_path") or ""),
            "target_analysis_summary": str(state.get("target_analysis_summary") or ""),
        },
    )

    # Restore crash context from previous run stage when repro/fix is resumed
    # as a separate k8s stage job. Without this, init resets crash state and
    # repro_crash would be incorrectly skipped.
    if resume_step in {"plan", "synthesize", "build", "run", "coverage-analysis", "improve-harness", "repro_crash", "re-build", "re-run", "fix_crash"}:
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
                            1,
                            min(int(coverage_loop.get("max_rounds") or out.get("coverage_loop_max_rounds") or 3), 5),
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
                        out["plan_max_fix_rounds"] = int(plan_policy.get("max_fix_rounds") or out["plan_max_fix_rounds"])
                    build_fix_policy = doc.get("build_fix_policy")
                    if isinstance(build_fix_policy, dict):
                        out["max_fix_rounds"] = max(
                            0,
                            min(int(build_fix_policy.get("max_fix_rounds") or out.get("max_fix_rounds") or 3), 20),
                        )
                        out["same_error_max_retries"] = max(
                            0,
                            min(
                                int(
                                    build_fix_policy.get("same_error_max_retries")
                                    or out.get("same_error_max_retries")
                                    or 1
                                ),
                                10,
                            ),
                        )
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

    _wf_log(cast(dict[str, Any], out), f"<- init ok repo_root={out.get('repo_root')} dt={_fmt_dt(time.perf_counter()-t0)}")
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
    antlr_context_path, antlr_context_summary = _prepare_antlr_assist_context(gen.repo_root)
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
            "上轮 re 阶段失败，需要优先修复该根因后再规划：\n"
            f"- stage: {restart_stage or 'unknown'}\n"
            f"- reason: {restart_reason or 'unknown'}\n"
            f"- error: {(restart_error_text or 'n/a')[:4096]}\n"
        )
        if report_tail:
            injected_ctx += "\n=== re failure report tail ===\n" + report_tail + "\n"
        hint = (hint + "\n\n" + injected_ctx).strip() if hint else injected_ctx
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
        if hint:
            prompt = _render_opencode_prompt("plan_with_hint", hint=hint)
            gen.patcher.run_codex_command(
                prompt,
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

        fix_on_crash, max_fix_rounds = _derive_plan_policy(gen.repo_root)
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
        try:
            selected_targets_path, selected_targets_doc = _write_selected_targets_doc(gen.repo_root)
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
        out = {
            **state,
            "last_step": "plan",
            "last_error": "",
            "codex_hint": plan_hint,
            "plan_fix_on_crash": fix_on_crash,
            "plan_max_fix_rounds": max_fix_rounds,
            "plan_retry_reason": plan_retry_reason,
            "plan_targets_schema_valid_before_retry": plan_targets_schema_valid_before_retry,
            "plan_targets_schema_valid_after_retry": plan_targets_schema_valid_after_retry,
            "plan_used_fallback_targets": plan_used_fallback_targets,
            "antlr_context_path": antlr_context_path,
            "antlr_context_summary": antlr_context_summary,
            "target_analysis_path": target_analysis_path,
            "target_analysis_summary": target_analysis_summary,
            "selected_targets_path": selected_targets_path,
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
            "restart_to_plan": restart_to_plan,
            "restart_to_plan_reason": restart_reason,
            "restart_to_plan_stage": restart_stage,
            "restart_to_plan_error_text": restart_error_text,
            "restart_to_plan_report_path": restart_report_path,
            "message": "planned",
        }
        _wf_log(cast(dict[str, Any], out), f"<- plan ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    except Exception as e:
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
    antlr_context_path = str(state.get("antlr_context_path") or "").strip()
    antlr_context_summary = str(state.get("antlr_context_summary") or "").strip()
    target_analysis_path = str(state.get("target_analysis_path") or "").strip()
    target_analysis_summary = str(state.get("target_analysis_summary") or "").strip()
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

    def _synthesis_output_status() -> dict[str, Any]:
        fuzz_dir = gen.repo_root / "fuzz"
        harnesses: list[str] = []
        has_build_script = False
        has_readme = False
        try:
            for p in fuzz_dir.rglob("*"):
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
        except Exception:
            return {
                "harnesses": [],
                "has_harness": False,
                "has_build_script": False,
                "has_readme": False,
                "has_required": False,
                "has_partial": False,
            }
        return {
            "harnesses": harnesses,
            "has_harness": bool(harnesses),
            "has_build_script": has_build_script,
            "has_readme": has_readme,
            "has_required": bool(harnesses) and has_build_script,
            "has_partial": bool(harnesses) or has_build_script or has_readme,
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
            build_sh = gen.repo_root / "fuzz" / "build.sh"
            if build_sh.is_file():
                parts.append("=== existing fuzz/build.sh ===\n" + build_sh.read_text(encoding="utf-8", errors="replace"))
            readme = gen.repo_root / "fuzz" / "README.md"
            if readme.is_file():
                parts.append("=== existing fuzz/README.md ===\n" + readme.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass
        return "\n\n".join(parts)

    def _run_synthesize_completion(timeout: int) -> None:
        missing_items = "\n".join(f"- {item}" for item in _missing_synthesis_items()) or "- no missing items detected"
        prompt = _render_opencode_prompt("synthesize_complete_scaffold", missing_items=missing_items)
        gen.patcher.run_codex_command(
            prompt,
            additional_context=_completion_context() or None,
            timeout=timeout,
            max_attempts=1,
            max_cli_retries=_opencode_cli_retries(),
            idle_timeout_override=_synthesize_opencode_idle_timeout_sec(),
            activity_watch_paths=_synthesize_activity_watch_paths(),
        )

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
            timeout=timeout,
            max_attempts=1,
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

        if hint:
            prompt = _render_opencode_prompt("synthesize_with_hint", hint=hint)
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
                timeout=_remaining_time_budget_sec(state),
                max_attempts=1,
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
            raise HarnessGeneratorError("synthesize incomplete: missing harness source under fuzz/")
        if not _has_required_synthesis_outputs():
            missing = ", ".join(_missing_synthesis_items()) or "unknown required files"
            raise HarnessGeneratorError(f"synthesize incomplete: missing required scaffold items: {missing}")
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
        out = {
            **state,
            "last_step": "synthesize",
            "last_error": "",
            "codex_hint": "",
            "restart_to_plan": False,
            "restart_to_plan_reason": "",
            "restart_to_plan_stage": "",
            "restart_to_plan_error_text": "",
            "restart_to_plan_report_path": "",
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
            "message": "synthesized",
        }
        _wf_log(cast(dict[str, Any], out), f"<- synthesize ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    except Exception as e:
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
            "last_step": "build",
        }
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
            if repeats >= max_same_repeats:
                next_state["failed"] = True
                next_state["last_error"] = (
                    "build failed with the same error signature repeatedly "
                    f"(repeats={repeats + 1}, threshold={max_same_repeats + 1})"
                )
                next_state["message"] = "build failed repeatedly (same error)"
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
            if repeats >= max_same_repeats:
                next_state["failed"] = True
                next_state["last_error"] = (
                    "build produced no fuzzers with the same diagnostics repeatedly "
                    f"(repeats={repeats + 1}, threshold={max_same_repeats + 1})"
                )
                next_state["message"] = "build failed repeatedly (no fuzzers)"
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
        next_state["message"] = f"built ({len(final_bins)} fuzzers)"
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
    error_sig = (state.get("build_error_signature_short") or "").strip()
    if not error_sig:
        error_sig = _sha256_text("\n".join([last_error, stdout_tail, stderr_tail]))[:12]

    if fix_attempts > max_fix_attempts:
        out = {
            **state,
            "last_step": "fix_build",
            "failed": True,
            "fix_build_terminal_reason": "fix_build_max_attempts_exceeded",
            "last_error": f"fix_build stopped: max attempts exceeded ({max_fix_attempts})",
            "message": "fix_build stopped (max attempts exceeded)",
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
        raw = (os.environ.get("SHERPA_FIX_BUILD_QUICK_CHECK_TIMEOUT_SEC") or "45").strip()
        try:
            return max(0, min(int(raw), 300))
        except Exception:
            return 45

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
                out["failed"] = True
                out["fix_build_terminal_reason"] = "fix_build_noop_streak_exceeded"
                out["last_error"] = f"fix_build stopped: no-op streak exceeded ({max_noop_streak})"
                out["message"] = "fix_build stopped (no-op streak exceeded)"
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
    stop_on_infra = stop_on_infra_raw not in {"0", "false", "no", "off"}
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
            "failed": True,
            "build_error_kind": "infra",
            "build_error_code": non_source_reason,
            "fix_build_terminal_reason": "fix_build_infra_blocked",
            "fix_build_attempt_history": updated_history,
            "fix_build_rule_hits": updated_rule_hits,
            "last_error": f"non-source build blocker detected: {non_source_reason}",
            "message": "fix_build skipped (environment/infrastructure issue)",
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
        skip_names = {"fuzz/build_full.log"}
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
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        if "cannot find -lz" in diag or "undefined reference to `gz" in diag or "undefined reference to `inflate" in diag:
            # Prefer dedicated link-fix rule for zlib linker failures.
            return False
        pkg_signals: list[tuple[list[str], str]] = [
            (["zlib.h", "could not find zlib", "cannot find -lz"], "zlib1g-dev"),
            (["bzlib.h", "could not find bzip2"], "libbz2-dev"),
            (["lzma.h", "could not find liblzma"], "liblzma-dev"),
            (["zstd.h", "could not find zstd", "one of the modules 'libzstd'"], "libzstd-dev"),
            (["lz4.h", "could not find lz4"], "liblz4-dev"),
            (["openssl/", "could not find openssl"], "libssl-dev"),
            (["expat.h", "could not find expat"], "libexpat1-dev"),
            (["libxml/parser.h", "could not find libxml2"], "libxml2-dev"),
            (["aclocal: not found", "automake: not found", "missing tools: automake"], "automake"),
            (["autoconf: not found", "missing tools: autoconf"], "autoconf"),
            (["libtool: not found", "libtoolize: not found", "missing tools: libtool"], "libtool"),
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
                    existing.append(line)
            except Exception:
                return False
        merged = sorted(set(existing) | set(need_pkgs))
        if merged == sorted(set(existing)):
            return False
        dep_file.parent.mkdir(parents=True, exist_ok=True)
        body = (
            "# Auto-maintained by fix_build hotfix rules.\n"
            "# Package names are Debian/Ubuntu apt identifiers.\n"
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

    summary = _summarize_build_error(last_error, stdout_tail, stderr_tail)
    recent_history = history[-history_limit:] if history else []

    # Ask an LLM to draft an *OpenCode instruction* tailored to the diagnostics.
    llm = _llm_or_none()
    codex_hint = (state.get("codex_hint") or "").strip()

    if not codex_hint:
        if llm is not None:
            coordinator_prompt = (
                "You are coordinating OpenCode to fix a fuzz harness build.\n"
                "Given the build diagnostics, produce a short instruction for OpenCode.\n\n"
                "Requirements for your output:\n"
                "- Output JSON only: {\"codex_hint\": \"...\"}\n"
                "- codex_hint must be 1-10 lines, concrete and minimal.\n"
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

        if not codex_hint:
            codex_hint = (
                (f"First read `{build_log_file}` for the complete build logs, then apply the minimal fix.\n" if build_log_file else "")
                +
                "Fix the fuzz build so that running `(cd fuzz && python build.py)` succeeds and leaves at least one executable fuzzer under fuzz/out/.\n"
                "Only modify files under fuzz/. Any change outside fuzz/ (except ./done sentinel) will be rejected.\n"
                "Do not use `-stdlib=libc++` in this environment.\n"
                "If target sources define `main`, add a compile define such as `-Dmain=vuln_main` to avoid libFuzzer main conflicts.\n"
                "If include/link flags are wrong, fix them from fuzz/build.py or fuzz harness code only.\n"
                "Do not refactor production code or edit upstream source files."
            )
        if recent_history and any(str(x.get("outcome") or "") == "noop" for x in recent_history):
            codex_hint = (
                codex_hint.strip()
                + "\nPrevious attempts were no-op; this attempt MUST produce at least one meaningful change under fuzz/."
            )

    # Now call OpenCode with a purpose-built prompt including diagnostics.
    context_parts: list[str] = []
    if build_log_file:
        context_parts.append("=== full build log file ===\n" + build_log_file)
    context_parts.append("=== structured_error ===\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    if recent_history:
        context_parts.append("=== fix_build_attempt_history (recent) ===\n" + json.dumps(recent_history, ensure_ascii=False, indent=2))
    if last_error:
        context_parts.append("=== last_error ===\n" + last_error)
    if stdout_tail:
        context_parts.append("=== build stdout (tail) ===\n" + stdout_tail)
    if stderr_tail:
        context_parts.append("=== build stderr (tail) ===\n" + stderr_tail)
    context = "\n\n".join(context_parts)

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
            timeout=_remaining_time_budget_sec(state),
            max_attempts=1,
            max_cli_retries=_opencode_cli_retries(),
        )
        post_fix_hashes = _collect_fix_relevant_hashes()
        post_step_hashes = _collect_fix_step_hashes()
        changed_paths = sorted(
            p
            for p in (set(baseline_step_hashes.keys()) | set(post_step_hashes.keys()))
            if baseline_step_hashes.get(p) != post_step_hashes.get(p)
        )
        changed_paths_count = len(changed_paths)
        disallowed_paths = [p for p in changed_paths if not _is_fix_build_allowed_path(p)]
        if disallowed_paths:
            preview = ", ".join(disallowed_paths[:5])
            suffix = ""
            if len(disallowed_paths) > 5:
                suffix = f" (+{len(disallowed_paths) - 5} more)"
            err = (
                "opencode fix_build touched disallowed path(s): "
                f"{preview}{suffix}. Only `fuzz/` and `done` are allowed"
            )
            updated_history, updated_rule_hits = _append_attempt(
                "disallowed_paths",
                rejection_reason="disallowed_paths",
                changed_paths_count=changed_paths_count,
            )
            out = {
                **state,
                "last_step": "fix_build",
                "last_error": err,
                "message": "opencode fix_build touched disallowed files",
                "fix_build_noop_streak": 0,
                "fix_build_attempt_history": updated_history,
                "fix_build_rule_hits": updated_rule_hits,
                "fix_build_last_diff_paths": changed_paths,
                "fix_action_type": "opencode",
                "fix_effect": "regressed",
            }
            _wf_log(
                cast(dict[str, Any], out),
                f"<- fix_build err=disallowed paths={len(disallowed_paths)} dt={_fmt_dt(time.perf_counter()-t0)}",
            )
            return out
        if post_fix_hashes == baseline_fix_hashes:
            updated_noop_streak = prev_noop_streak + 1
            updated_history, updated_rule_hits = _append_attempt(
                "noop",
                rejection_reason="no_relevant_file_changes",
                changed_paths_count=changed_paths_count,
            )
            terminal_reason = ""
            failed = False
            msg = "opencode fix_build no-op"
            err = "opencode fix_build made no relevant file changes"
            if updated_noop_streak >= max_noop_streak:
                failed = True
                terminal_reason = "fix_build_noop_streak_exceeded"
                msg = "fix_build stopped (no-op streak exceeded)"
                err = f"fix_build stopped: no-op streak exceeded ({max_noop_streak})"
            out = {
                **state,
                "last_step": "fix_build",
                "last_error": err,
                "message": msg,
                "failed": failed,
                "fix_build_terminal_reason": terminal_reason,
                "fix_build_noop_streak": updated_noop_streak,
                "fix_build_attempt_history": updated_history,
                "fix_build_rule_hits": updated_rule_hits,
                "fix_build_last_diff_paths": changed_paths,
                "fix_action_type": "opencode",
                "fix_effect": "stalled",
            }
            _wf_log(cast(dict[str, Any], out), f"<- fix_build err=no-op dt={_fmt_dt(time.perf_counter()-t0)}")
            return out
        updated_history, updated_rule_hits = _append_attempt(
            "llm_fixed",
            changed_paths_count=changed_paths_count,
        )
        out = {
            **state,
            "last_step": "fix_build",
            "last_error": "",
            "codex_hint": "",
            "message": "opencode fixed build",
            "fix_build_noop_streak": 0,
            "fix_build_attempt_history": updated_history,
            "fix_build_rule_hits": updated_rule_hits,
            "fix_build_terminal_reason": "",
            "fix_build_last_diff_paths": changed_paths,
            "fix_action_type": "opencode",
            "fix_effect": "advanced",
        }
        if _requires_env_rebuild(changed_paths):
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
        max_parallel_raw = os.environ.get("SHERPA_PARALLEL_FUZZERS", "2")
        try:
            max_parallel = max(1, min(int(max_parallel_raw), 16))
        except Exception:
            max_parallel = 2
        stop_on_first_crash = _run_stop_on_first_crash()
        if stop_on_first_crash and len(bins) > 1:
            # Run sequentially so a proven crash can terminate the stage
            # immediately instead of leaving sibling fuzzers consuming the full
            # run budget before the stage result is written back.
            max_parallel = 1
        if len(bins) <= 1:
            max_parallel = 1
        idle_timeout_sec = _run_idle_timeout_sec()
        finalize_timeout_sec = _run_finalize_timeout_sec()

        _wf_log(
            cast(dict[str, Any], state),
            (
                f"run: fuzzers={len(bins)} parallel={max_parallel} "
                f"stop_on_first_crash={int(stop_on_first_crash)}"
            ),
        )

        def _calc_crash_signature(fuzzer_name: str, artifact_path: str) -> str:
            parts: list[str] = [f"fuzzer={fuzzer_name}", f"artifact={artifact_path}"]
            crash_info = gen.repo_root / "crash_info.md"
            crash_analysis = gen.repo_root / "crash_analysis.md"
            for p in (crash_info, crash_analysis):
                if not p.is_file():
                    continue
                try:
                    txt = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                tail = "\n".join(txt.splitlines()[-400:])
                parts.append(f"== {p.name} ==\n{tail}")
            return _sha256_text("\n\n".join(parts))

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
        seed_sources: set[str] = set()
        repo_examples_filtered = False
        repo_examples_rejected_count = 0
        repo_examples_accepted_count = 0
        seed_noise_rejected_count = 0
        seed_family_coverage_state: dict[str, Any] = {}
        try:
            for bin_path in bins:
                remaining_for_seed = _remaining_time_budget_sec(state, min_timeout=0)
                if remaining_for_seed <= 0:
                    return _time_budget_exceeded_state(state, step_name="run")
                setattr(gen, "seed_generation_timeout_sec", max(1, remaining_for_seed))
                fuzzer_name = bin_path.name
                try:
                    gen._pass_generate_seeds(fuzzer_name)
                    profile_map = getattr(gen, "last_seed_profile_by_fuzzer", {}) or {}
                    if not last_seed_profile:
                        last_seed_profile = str(profile_map.get(fuzzer_name) or "")
                    bootstrap_map = getattr(gen, "last_seed_bootstrap_by_fuzzer", {}) or {}
                    meta = bootstrap_map.get(fuzzer_name) or {}
                    if isinstance(meta, dict):
                        counts = meta.get("counts") or {}
                        if isinstance(counts, dict):
                            for key in ("repo_examples", "ai", "radamsa", "total"):
                                try:
                                    seed_count_total[key] = int(seed_count_total.get(key, 0)) + int(counts.get(key) or 0)
                                except Exception:
                                    continue
                        raw_counts = meta.get("seed_counts_raw") or {}
                        if isinstance(raw_counts, dict):
                            for key in ("repo_examples", "ai", "radamsa", "total"):
                                try:
                                    seed_count_raw_total[key] = int(seed_count_raw_total.get(key, 0)) + int(raw_counts.get(key) or 0)
                                except Exception:
                                    continue
                        filtered_counts = meta.get("seed_counts_filtered") or {}
                        if isinstance(filtered_counts, dict):
                            for key in ("repo_examples", "ai", "radamsa", "total"):
                                try:
                                    seed_count_filtered_total[key] = int(seed_count_filtered_total.get(key, 0)) + int(filtered_counts.get(key) or 0)
                                except Exception:
                                    continue
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
                    print(f"[warn] seed generation skipped ({fuzzer_name}): {e}")
        finally:
            setattr(gen, "seed_generation_timeout_sec", prev_seed_timeout)

        run_results: dict[str, FuzzerRunResult] = {}
        run_exec_errors: dict[str, str] = {}
        finalized_fuzzers: set[str] = set()

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
                        for fut in as_completed(futures):
                            bin_path = futures[fut]
                            try:
                                name, run = fut.result()
                                run_results[name] = run
                                finalized_fuzzers.add(name)
                                run_children_exit_count += 1
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
                if stop_on_first_crash and any(run.crash_found for run in run_results.values()):
                    pending_bins = []
                    break
        finally:
            setattr(gen, "current_run_time_budget_sec", prev_run_budget)
            setattr(gen, "current_run_hard_timeout_sec", prev_run_hard_timeout)

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
                    {
                        "fuzzer": fuzzer_name,
                        "rc": detail_rc,
                        "effective_rc": detail_rc,
                        "crash_found": False,
                        "crash_evidence": "none",
                        "run_error_kind": detail_kind,
                        "exception_kind": detail_kind,
                        "error": exec_err,
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
                    }
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
                    {
                        "fuzzer": fuzzer_name,
                        "rc": 1,
                        "effective_rc": 1,
                        "crash_found": False,
                        "crash_evidence": "none",
                        "run_error_kind": "run_exception",
                        "exception_kind": "run_exception",
                        "error": "missing run result",
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
                    }
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
                    "seed_quality": dict(run.seed_quality or {}),
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
            for detail in run_details:
                if bool(detail.get("crash_found")):
                    continue
                if int(detail.get("rc") or 0) != 0:
                    continue
                final_execs = int(detail.get("final_execs_per_sec") or 0)
                log_or_err = f"{detail.get('error') or ''}\n{detail.get('log_tail') or ''}".lower()
                warned_no_progress = (
                    "no interesting inputs were found so far" in log_or_err
                    or "inited exec/s: 0" in log_or_err
                    or "exec/s: 0" in log_or_err
                )
                if final_execs <= 0 and warned_no_progress:
                    no_progress_fuzzers.append(str(detail.get("fuzzer") or "unknown"))
            if no_progress_fuzzers:
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
            "run_details": run_details,
            "run_batch_plan": run_batch_plan,
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
                next(
                    (
                        list((meta.get("seed_families_required") or []))
                        for meta in (getattr(gen, "last_seed_bootstrap_by_fuzzer", {}) or {}).values()
                        if isinstance(meta, dict) and meta.get("seed_families_required")
                    ),
                    list(state.get("coverage_seed_families_required") or []),
                )
            ),
            "coverage_seed_families_covered": list(
                next(
                    (
                        list(((meta.get("seed_family_coverage") or {}).get("covered") or []))
                        for meta in (getattr(gen, "last_seed_bootstrap_by_fuzzer", {}) or {}).values()
                        if isinstance(meta, dict) and (meta.get("seed_family_coverage") or {}).get("covered")
                    ),
                    list(state.get("coverage_seed_families_covered") or []),
                )
            ),
            "coverage_seed_families_missing": list(
                next(
                    (
                        list(((meta.get("seed_family_coverage") or {}).get("missing") or []))
                        for meta in (getattr(gen, "last_seed_bootstrap_by_fuzzer", {}) or {}).values()
                        if isinstance(meta, dict) and (meta.get("seed_family_coverage") or {}).get("missing") is not None
                    ),
                    list(state.get("coverage_seed_families_missing") or []),
                )
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
            "coverage_seed_family_coverage": seed_family_coverage_state,
            "coverage_repo_examples_filtered": repo_examples_filtered,
            "coverage_repo_examples_rejected_count": repo_examples_rejected_count,
            "coverage_repo_examples_accepted_count": repo_examples_accepted_count,
            "crash_signature": crash_signature,
            "same_crash_repeats": same_crash_repeats,
            "timeout_signature": timeout_signature,
            "same_timeout_repeats": same_timeout_repeats,
            "message": msg,
        }
        if run_error_kind == "workflow_time_budget_exceeded":
            out["failed"] = True
            out["last_error"] = out.get("last_error") or "time budget exceeded during run phase"
            out["message"] = "workflow stopped (time budget exceeded)"
        if run_error_kind in {"run_idle_timeout", "run_timeout", "run_finalize_timeout"}:
            if run_error_kind == "run_idle_timeout":
                out["message"] = "run stalled (idle timeout), routing to fix_build"
                if not out.get("last_error"):
                    out["last_error"] = f"run stalled: no output for >= {idle_timeout_sec}s"
            elif run_error_kind == "run_finalize_timeout":
                out["message"] = "run finalize timed out, routing to fix_build"
            else:
                out["message"] = "run timed out, routing to fix_build"
        if crash_found and same_crash_repeats >= max_same_crash_repeats:
            out["failed"] = True
            out["last_error"] = (
                "same crash signature repeated after crash-fix attempts "
                f"(repeats={same_crash_repeats + 1}, threshold={max_same_crash_repeats + 1})"
            )
            out["message"] = "workflow stopped (same crash repeated)"
        if run_error_kind in timeout_like_kinds and same_timeout_repeats >= max_same_timeout_repeats:
            out["failed"] = True
            out["last_error"] = (
                "same timeout/no-progress signature repeated "
                f"(repeats={same_timeout_repeats + 1}, threshold={max_same_timeout_repeats + 1})"
            )
            out["message"] = "workflow stopped (same timeout/no-progress repeated)"
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
        raw_total = int((out.get("coverage_seed_counts_raw") or {}).get("total") or 0)
        noise_rejected = int(out.get("coverage_seed_noise_rejected_count") or 0)
        if noise_rejected > 0 and (raw_total <= 0 or float(noise_rejected) / float(max(raw_total, 1)) >= 0.25):
            quality_flags.append("seed_noise_high")
        observed_api = str(out.get("coverage_target_api") or "").lower()
        if observed_api in {"println", "fmt::println", "print", "fmt::print", "format", "fmt::format", "format_to", "fmt::format_to", "vformat", "fmt::vformat"}:
            quality_flags.append("generic_wrapper_fallback")
        out["coverage_quality_flags"] = sorted({flag for flag in quality_flags if flag})
        _wf_log(
            cast(dict[str, Any], out),
            (
                f"<- run ok crash_found={crash_found} rc={run_rc} evidence={crash_evidence} "
                f"same_crash_repeats={same_crash_repeats} same_timeout_repeats={same_timeout_repeats} "
                f"dt={_fmt_dt(time.perf_counter()-t0)}"
            ),
        )
        return out
    except Exception as e:
        out = {**state, "last_step": "run", "last_error": str(e), "message": "run failed"}
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
        max_rounds = max(1, min(int(state.get("coverage_loop_max_rounds") or 3), 5))
        current_round = max(0, int(state.get("coverage_loop_round") or 0))
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
        plateau_no_gain = plateau_detected and current_cov <= prev_cov and current_ft <= prev_ft
        plateau_streak = (prev_plateau_streak + 1) if plateau_no_gain else (1 if plateau_detected else 0)
        requested_replan = bool(
            plateau_no_gain
            and plateau_streak >= 2
            and bool(current_seed_profile)
        )
        replan_reason = ""
        improve_mode = ""
        can_in_place = current_round < max_rounds
        can_replan = (current_round + 1) < max_rounds
        round_budget_exhausted = False
        stop_reason = ""
        run_error_kind = str(state.get("run_error_kind") or "").strip().lower()
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
                }
            )
        )
        base_should_improve = (
            (not bool(state.get("crash_found")))
            and (not bool(state.get("failed")))
            and (not run_error_kind or run_error_kind in {"run_resource_exhaustion"})
        )
        should_improve = False
        replan_required = False
        if base_should_improve:
            if seed_quality_issue and can_in_place:
                should_improve = True
                improve_mode = "in_place"
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
                reason = (
                    f"coverage plateau detected; mode={improve_mode}; round={next_round}/{max_rounds}, "
                    f"max_cov={current_cov}, prev_cov={prev_cov}, max_ft={current_ft}, prev_ft={prev_ft}, "
                    f"plateau_streak={plateau_streak}, idle_no_growth={plateau_idle_seconds}s"
                )
            else:
                reason = (
                    f"mode=in_place; round={next_round}/{max_rounds}, max_cov={current_cov}, prev_cov={prev_cov}, "
                    f"max_ft={current_ft}, prev_ft={prev_ft}"
                )
            if seed_quality_issue:
                reason += f"; seed_quality_flags={','.join(quality_flags) or 'none'}"
        elif round_budget_exhausted:
            if requested_replan:
                reason = (
                    f"coverage plateau detected but replan budget exhausted; "
                    f"round={current_round}/{max_rounds}, max_cov={current_cov}, max_ft={current_ft}, "
                    f"plateau_streak={plateau_streak}"
                )
            else:
                reason = (
                    f"coverage loop budget exhausted; round={current_round}/{max_rounds}, "
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
        }
        _wf_log(
            cast(dict[str, Any], out),
            f"<- coverage-analysis improve={int(should_improve)} {reason} dt={_fmt_dt(time.perf_counter()-t0)}",
        )
        return out
    except Exception as e:
        out = {**state, "last_step": "coverage-analysis", "last_error": str(e), "message": "coverage analysis failed"}
        _wf_log(cast(dict[str, Any], out), f"<- coverage-analysis err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


def _node_improve_harness(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
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
        improve_mode = str(state.get("coverage_improve_mode") or "").strip() or ("replan" if replan_required else "in_place")
        if replan_required:
            hint = (
                "覆盖率闭环改进任务（重新选 target）：\n"
                "- 当前 target 已连续多轮进入平台期，且 coverage/features 未继续提升。\n"
                "- 允许重新评估 targets.json，并重新选择更可能提高覆盖率或发现漏洞的 target。\n"
                "- 结合 fuzz/target_analysis.json 与 fuzz/antlr_plan_context.json 重新规划。\n"
                f"- 当前 target: {target_name or 'unknown'}\n"
                f"- 当前 target api: {target_api or selected_target_api or 'unknown'}\n"
                f"- 当前 seed_profile: {seed_profile or 'generic'}\n"
                f"- 当前深度: {depth_class or 'unknown'} (score={depth_score})\n"
                f"- 当前选择原因: {selection_bias_reason or 'n/a'}\n"
                f"- replan 原因: {replan_reason or cov_reason or 'coverage plateau'}\n"
                "- 如果当前 target 属于 shallow，优先选择 medium/deep 候选，不要再次落到 checksum/hash/helper/bound/combine/version/copy 类浅目标。\n"
                "- 优先考虑 decode/inflate/deflate/parse/read/load/scan/archive/stream 这类更深入口。"
            )
        else:
            hint = (
                "覆盖率闭环改进任务（当前 target 原地改进）：\n"
                "- 只允许修改当前 fuzzer 相关的 fuzz/ 下文件，不要改业务源码。\n"
                "- 不要修改 targets.json，不要新增第二个 target。\n"
                "- 优先补 seed 生成、dictionary、输入建模、调用顺序、边界值与 corpus bootstrap。\n"
                "- 保持可构建与可运行。\n"
                f"- 当前 target: {target_name or 'unknown'}\n"
                f"- 当前 target api: {target_api or selected_target_api or 'unknown'}\n"
                f"- 当前 seed_profile: {seed_profile or 'generic'}\n"
                f"- seed quality flags: {', '.join(quality_flags) if quality_flags else 'none'}\n"
                f"- required seed families: {', '.join(seed_families_required) if seed_families_required else 'none'}\n"
                f"- covered seed families: {', '.join(seed_families_covered) if seed_families_covered else 'none'}\n"
                f"- missing seed families: {', '.join(seed_families_missing) if seed_families_missing else 'none'}\n"
                f"- seed raw counts: {seed_counts_raw or {}}\n"
                f"- seed filtered counts: {seed_counts_filtered or {}}\n"
                f"- seed noise rejected: {seed_noise_rejected_count}\n"
                f"- seed quality summary: {json.dumps(seed_quality, ensure_ascii=False) if seed_quality else '{}'}\n"
                f"- 当前深度: {depth_class or 'unknown'} (score={depth_score})\n"
                f"- 当前选择原因: {selection_bias_reason or 'n/a'}\n"
                f"- 诊断: {cov_reason}"
            )
        out = {
            **state,
            "last_step": "improve-harness",
            "last_error": "",
            "codex_hint": hint,
            "coverage_improve_mode": improve_mode,
            "message": "improve-harness prepared plan hint",
        }
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
        shutil.copytree(source_fuzz, dest_fuzz)

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
        build = subprocess.run(
            build_cmd,
            cwd=build_cwd,
            capture_output=True,
            text=True,
            timeout=build_timeout,
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
        gen._clone_repo(RepoSpec(url=repo_url, workdir=clone_root))
        source_fuzz = repo_root / "fuzz"
        if not source_fuzz.is_dir():
            raise HarnessGeneratorError(f"run fuzz directory missing: {source_fuzz}")
        dest_fuzz = clone_root / "fuzz"
        if dest_fuzz.exists():
            shutil.rmtree(dest_fuzz, ignore_errors=True)
        shutil.copytree(source_fuzz, dest_fuzz)

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
        build = subprocess.run(
            build_cmd,
            cwd=build_cwd,
            capture_output=True,
            text=True,
            timeout=build_timeout,
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
        repro = subprocess.run(
            [str(fuzzer_bin), "-runs=1", str(artifact_path)],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=repro_timeout,
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
    if bool(state.get("failed")):
        return "stop"
    if not (state.get("last_error") or "").strip():
        return "run"
    if (state.get("build_error_kind") or "").strip().lower() == "infra":
        return "stop"
    return "fix_build"


def _route_after_run_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")):
        return "stop"
    run_error_kind = (state.get("run_error_kind") or "").strip().lower()
    if run_error_kind in {
        "run_no_progress",
        "nonzero_exit_without_crash",
        "run_exception",
    }:
        return "fix_build"
    if run_error_kind in {"run_idle_timeout", "run_timeout", "run_finalize_timeout"}:
        return "stop"
    if run_error_kind in {"run_resource_exhaustion"}:
        return "coverage-analysis"
    if run_error_kind:
        return "stop"
    if bool(state.get("crash_found")):
        return "re-build"
    return "coverage-analysis"


def _route_after_coverage_analysis_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")):
        return "stop"
    if (state.get("last_error") or "").strip():
        return "stop"
    if bool(state.get("coverage_should_improve")):
        return "improve-harness"
    return "stop"


def _route_after_improve_harness_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")):
        return "stop"
    if (state.get("last_error") or "").strip():
        return "stop"
    if str(state.get("coverage_improve_mode") or "").strip() == "replan" and not bool(
        state.get("coverage_replan_effective", True)
    ):
        return "stop"
    if bool(state.get("coverage_round_budget_exhausted")):
        return "stop"
    if bool(state.get("coverage_should_improve")):
        if str(state.get("coverage_improve_mode") or "").strip() == "in_place":
            return "build"
        return "plan"
    return "stop"


def _route_after_plan_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")) or (state.get("last_error") or "").strip():
        return "stop"
    return "synthesize"


def _route_after_synthesize_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")) or (state.get("last_error") or "").strip():
        return "stop"
    return "build"


def _route_after_fix_build_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")):
        return "stop"
    if (state.get("fix_build_terminal_reason") or "").strip():
        return "stop"
    if (state.get("last_error") or "").strip():
        return "stop"
    return "build"


def _route_after_fix_crash_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")):
        return "stop"
    if (state.get("last_error") or "").strip():
        return "stop"
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
        return "stop"
    fix_on_crash = bool(state.get("plan_fix_on_crash", True))
    max_fix_rounds = max(0, int(state.get("plan_max_fix_rounds") or 1))
    attempts = int(state.get("crash_fix_attempts") or 0)
    if fix_on_crash and attempts < max_fix_rounds:
        return "fix_crash"
    return "stop"


def _recommended_next_step(state: FuzzWorkflowRuntimeState) -> str:
    last_step = str(state.get("last_step") or "").strip().lower()
    if not last_step:
        return "stop"
    if last_step == "init":
        return _route_after_init_state(state)
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
    if last_step == "coverage-analysis":
        return _route_after_coverage_analysis_state(state)
    if last_step == "improve-harness":
        return _route_after_improve_harness_state(state)
    if last_step in {"repro_crash", "re-build"}:
        return _route_after_re_build_state(state)
    if last_step == "re-run":
        return _route_after_re_run_state(state)
    if last_step == "fix_crash":
        return _route_after_fix_crash_state(state)
    return "stop"


def _route_after_init_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")) or (state.get("last_error") or "").strip():
        return "stop"
    raw = (state.get("resume_from_step") or "").strip().lower()
    allowed = {
        "plan",
        "synthesize",
        "build",
        "fix_build",
        "run",
        "coverage-analysis",
        "improve-harness",
        "re-build",
        "re-run",
        "repro_crash",
        "fix_crash",
    }
    if raw == "repro_crash":
        raw = "re-build"
    if raw in allowed:
        return raw
    return "plan"


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
    graph.add_node("plan", _node_plan)
    graph.add_node("synthesize", _node_synthesize)
    graph.add_node("build", _node_build)
    graph.add_node("fix_build", _node_fix_build)
    graph.add_node("coverage-analysis", _node_coverage_analysis)
    graph.add_node("improve-harness", _node_improve_harness)
    graph.add_node("re-build", _node_re_build)
    graph.add_node("re-run", _node_re_run)
    graph.add_node("repro_crash", _node_repro_crash)
    graph.add_node("fix_crash", _node_fix_crash)
    graph.add_node("run", _node_run)

    graph.set_entry_point("init")

    def _route_after_plan(state: FuzzWorkflowRuntimeState) -> str:
        if bool(state.get("failed")) or (state.get("last_error") or "").strip():
            return "stop"
        if _should_stage_stop(state, "plan"):
            return "stop"
        return "synthesize"

    def _route_after_synthesize(state: FuzzWorkflowRuntimeState) -> str:
        if bool(state.get("failed")) or (state.get("last_error") or "").strip():
            return "stop"
        if _should_stage_stop(state, "synthesize"):
            return "stop"
        return "build"

    def _route_after_build(state: FuzzWorkflowRuntimeState) -> str:
        if not bool(state.get("failed")) and not (state.get("last_error") or "").strip():
            if _should_stage_stop(state, "build"):
                return "stop"
        return _route_after_build_state(state)

    def _route_after_fix_build(state: FuzzWorkflowRuntimeState) -> str:
        if bool(state.get("failed")):
            return "stop"
        if (state.get("fix_build_terminal_reason") or "").strip():
            return "stop"
        if (state.get("last_error") or "").strip():
            return "stop"
        if _should_stage_stop(state, "fix_build"):
            return "stop"
        return "build"

    def _route_after_run(state: FuzzWorkflowRuntimeState) -> str:
        nxt = _route_after_run_state(state)
        return _apply_stage_stop_guard(state, "run", nxt)

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

    def _route_after_fix_crash(state: FuzzWorkflowRuntimeState) -> str:
        if bool(state.get("failed")):
            return "stop"
        if (state.get("last_error") or "").strip():
            return "stop"
        if _should_stage_stop(state, "fix_crash"):
            return "stop"
        return "build"

    graph.add_conditional_edges(
        "init",
        _route_after_init_state,
        {
            "plan": "plan",
            "synthesize": "synthesize",
            "build": "build",
            "fix_build": "fix_build",
            "run": "run",
            "coverage-analysis": "coverage-analysis",
            "improve-harness": "improve-harness",
            "re-build": "re-build",
            "re-run": "re-run",
            "repro_crash": "re-build",
            "fix_crash": "fix_crash",
            "stop": END,
        },
    )
    graph.add_conditional_edges("plan", _route_after_plan, {"synthesize": "synthesize", "stop": END})
    graph.add_conditional_edges("synthesize", _route_after_synthesize, {"build": "build", "stop": END})
    graph.add_conditional_edges("build", _route_after_build, {"run": "run", "fix_build": "fix_build", "stop": END})
    graph.add_conditional_edges("fix_build", _route_after_fix_build, {"build": "build", "stop": END})
    graph.add_conditional_edges(
        "run",
        _route_after_run,
        {
            "coverage-analysis": "coverage-analysis",
            "re-build": "re-build",
            "fix_crash": "fix_crash",
            "fix_build": "fix_build",
            "stop": END,
        },
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
    graph.add_conditional_edges("re-build", _route_after_re_build, {"re-run": "re-run", "plan": "plan", "stop": END})
    graph.add_conditional_edges("re-run", _route_after_re_run, {"fix_crash": "fix_crash", "plan": "plan", "stop": END})
    graph.add_conditional_edges("repro_crash", _route_after_re_build, {"re-run": "re-run", "plan": "plan", "stop": END})
    graph.add_conditional_edges("fix_crash", _route_after_fix_crash, {"build": "build", "stop": END})

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
        max_steps_env = int(os.environ.get("SHERPA_WORKFLOW_MAX_STEPS", "20"))
    except Exception:
        max_steps_env = 20
    max_steps = max(3, min(max_steps_env, 100))
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
            "coverage_loop_max_rounds": max(1, min(int(inp.coverage_loop_max_rounds or 3), 5)),
            "max_fix_rounds": max(0, min(int(inp.max_fix_rounds or 3), 20)),
            "same_error_max_retries": max(0, min(int(inp.same_error_max_retries or 1), 10)),
            "last_fuzzer": str(inp.last_fuzzer or ""),
            "last_crash_artifact": str(inp.last_crash_artifact or ""),
            "re_workspace_root": str(inp.re_workspace_root or ""),
            "max_steps": max_steps,
        }
    )
    out = cast(dict[str, Any], raw) if isinstance(raw, dict) else {}
    try:
        _write_run_summary(out)
    except Exception:
        pass
    msg = str(out.get("message") or "Fuzzing completed.").strip()
    recommended_next = _recommended_next_step(cast(FuzzWorkflowRuntimeState, out))
    if bool(out.get("failed")):
        _wf_log(out, f"workflow end status=failed dt={_fmt_dt(time.perf_counter()-t0)}")
        terminal_reason = str(out.get("run_terminal_reason") or "").strip() or str(
            out.get("fix_build_terminal_reason") or ""
        ).strip()
        if terminal_reason:
            msg = f"{terminal_reason}: {msg}"
        raise RuntimeError(msg or "workflow failed")
    # If we stopped due to an error but didn't mark failed, still surface it.
    last_error = str(out.get("last_error") or "").strip()
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
        "coverage_loop_max_rounds": int(out.get("coverage_loop_max_rounds") or 3),
        "coverage_loop_round": int(out.get("coverage_loop_round") or 0),
        "coverage_should_improve": bool(out.get("coverage_should_improve") or False),
        "coverage_improve_reason": str(out.get("coverage_improve_reason") or ""),
        "coverage_history": list(out.get("coverage_history") or []),
        "plan_retry_reason": str(out.get("plan_retry_reason") or ""),
        "plan_targets_schema_valid_before_retry": bool(out.get("plan_targets_schema_valid_before_retry") or False),
        "plan_targets_schema_valid_after_retry": bool(out.get("plan_targets_schema_valid_after_retry") or False),
        "plan_used_fallback_targets": bool(out.get("plan_used_fallback_targets") or False),
        "max_fix_rounds": int(out.get("max_fix_rounds") or 3),
        "same_error_max_retries": int(out.get("same_error_max_retries") or 1),
        "fix_action_type": str(out.get("fix_action_type") or ""),
        "fix_effect": str(out.get("fix_effect") or ""),
        "build_error_signature_before": str(out.get("build_error_signature_before") or ""),
        "build_error_signature_after": str(out.get("build_error_signature_after") or ""),
        "crash_repro_done": bool(out.get("crash_repro_done") or False),
        "crash_repro_ok": bool(out.get("crash_repro_ok") or False),
        "crash_repro_rc": int(out.get("crash_repro_rc") or 0),
        "crash_repro_report_path": str(out.get("crash_repro_report_path") or ""),
        "crash_repro_json_path": str(out.get("crash_repro_json_path") or ""),
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
