from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from loguru import logger


def decision_snapshot_from_state(state: dict[str, Any]) -> dict[str, Any]:
    """Build a compact decision snapshot for stage-level observability."""
    return {
        "repo_root": str(state.get("repo_root") or ""),
        "coverage_loop_round": int(state.get("coverage_loop_round") or 0),
        "coverage_loop_max_rounds": int(state.get("coverage_loop_max_rounds") or 0),
        "coverage_should_improve": bool(state.get("coverage_should_improve") or False),
        "coverage_improve_mode": str(state.get("coverage_improve_mode") or ""),
        "coverage_replan_required": bool(state.get("coverage_replan_required") or False),
        "run_error_kind": str(state.get("run_error_kind") or ""),
        "repair_mode": bool(state.get("repair_mode") or False),
        "repair_origin_stage": str(state.get("repair_origin_stage") or ""),
        "repair_error_kind": str(state.get("repair_error_kind") or ""),
        "repair_error_code": str(state.get("repair_error_code") or ""),
        "crash_triage_label": str(state.get("crash_triage_label") or ""),
        "crash_triage_confidence": float(state.get("crash_triage_confidence") or 0.0),
        "security_priority_mode": bool(state.get("security_priority_mode") or False),
        "vuln_hunting_enabled": bool(state.get("vuln_hunting_enabled") or False),
        "vuln_candidate_count": int(state.get("vuln_candidate_count") or 0),
        "security_evidence_count": int(state.get("security_evidence_count") or 0),
        "analysis_evidence_count": int(state.get("analysis_evidence_count") or 0),
    }


def _decision_trace_path(state: dict[str, Any]) -> Path | None:
    repo_root = str(state.get("repo_root") or "").strip()
    if not repo_root:
        gen = state.get("generator")
        if gen is not None:
            try:
                repo_root = str(getattr(gen, "repo_root", "") or "").strip()
            except Exception:
                repo_root = ""
    if not repo_root:
        return None
    try:
        return Path(repo_root) / "fuzz" / "decision_trace.jsonl"
    except Exception:
        return None


def _decision_trace_max_items() -> int:
    raw = (os.environ.get("SHERPA_DECISION_TRACE_MAX_ITEMS") or "200").strip()
    try:
        return max(20, min(int(raw), 2000))
    except Exception:
        return 200


def record_decision_trace(
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
    out = dict(state)
    traces = list(out.get("decision_traces") or [])
    existing_count = max(int(out.get("decision_trace_count") or 0), len(traces))
    trace = {
        "ts": int(time.time()),
        "stage": str(stage or "").strip(),
        "tool": str(tool or "").strip(),
        "model": str(model or "").strip(),
        "latency_ms": int(latency_ms or 0),
        "token_usage": dict(token_usage or {}),
        "error_kind": str(error_kind or "").strip(),
        "error_code": str(error_code or "").strip(),
        "retry_count": int(retry_count or 0),
        "decision_snapshot": dict(decision_snapshot or {}),
    }
    traces.append(trace)
    max_items = _decision_trace_max_items()
    if len(traces) > max_items:
        traces = traces[-max_items:]
    out["decision_traces"] = traces
    out["decision_trace_count"] = int(max(existing_count + 1, len(traces)))
    out["latest_decision_snapshot"] = dict(decision_snapshot or {})
    trace_path = _decision_trace_path(out)
    if trace_path is not None:
        try:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(trace, ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception:
            pass
    return out


def emit_fuzz_metrics(state: dict[str, Any]) -> None:
    """Emit a structured ``[wf-metrics]`` line for control-plane ingestion."""
    run_details = list(state.get("run_details") or [])
    coverage_history = list(state.get("coverage_history") or [])

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

    payload = {
        "ts": int(time.time()),
        "stage": str(state.get("last_step") or ""),
        "coverage_loop_round": int(state.get("coverage_loop_round") or 0),
        "coverage_loop_max_rounds": int(state.get("coverage_loop_max_rounds") or 0),
        "max_cov": max((f["final_cov"] for f in fuzzers.values()), default=0),
        "max_ft": max((f["final_ft"] for f in fuzzers.values()), default=0),
        "total_execs_per_sec": sum(f["final_execs_per_sec"] for f in fuzzers.values()),
        "crash_found": any(f["crash_found"] for f in fuzzers.values()),
        "fuzzers": fuzzers,
        "coverage_history": coverage_history,
        "coverage_source_report": dict(state.get("coverage_source_report") or {}),
        "coverage_run_feedback_path": str(state.get("coverage_run_feedback_path") or ""),
        "coverage_run_feedback_summary": dict(state.get("coverage_run_feedback_summary") or {}),
        "coverage_per_input_manifest_path": str(state.get("coverage_per_input_manifest_path") or ""),
        "coverage_frontier_path": str(state.get("coverage_frontier_path") or ""),
        "coverage_frontier_summary": dict(state.get("coverage_frontier_summary") or {}),
        "coverage_plateau_streak": int(state.get("coverage_plateau_streak") or 0),
        "coverage_seed_profile": str(state.get("coverage_seed_profile") or ""),
        "coverage_quality_flags": list(state.get("coverage_quality_flags") or []),
        "coverage_bottleneck_kind": str(state.get("coverage_bottleneck_kind") or ""),
        "coverage_bottleneck_reason": str(state.get("coverage_bottleneck_reason") or ""),
        "coverage_replay_runtime_sec": float(state.get("coverage_replay_runtime_sec") or 0.0),
        "coverage_replay_binary_hash": str(state.get("coverage_replay_binary_hash") or ""),
        "coverage_replay_binary_dir": str(state.get("coverage_replay_binary_dir") or ""),
        "coverage_replay_binary_count": int(state.get("coverage_replay_binary_count") or 0),
        "coverage_replay_stage_success": bool(state.get("coverage_replay_stage_success") or False),
        "coverage_replay_error": str(state.get("coverage_replay_error") or ""),
        "coverage_replay_manifest_fresh_for_current_binary": bool(
            state.get("coverage_replay_manifest_fresh_for_current_binary") or False
        ),
        "coverage_replay_queue_drained": bool(state.get("coverage_replay_queue_drained") or False),
        "coverage_replay_pending_inputs": int(state.get("coverage_replay_pending_inputs") or 0),
        "coverage_replay_failed_inputs": int(state.get("coverage_replay_failed_inputs") or 0),
        "coverage_replay_processed_inputs": int(state.get("coverage_replay_processed_inputs") or 0),
        "coverage_replay_total_inputs": int(state.get("coverage_replay_total_inputs") or 0),
        "analysis_evidence_count": int(state.get("analysis_evidence_count") or 0),
        "security_evidence_count": int(state.get("security_evidence_count") or 0),
        "vuln_candidate_count": int(state.get("vuln_candidate_count") or 0),
        "vuln_hunting_enabled": bool(state.get("vuln_hunting_enabled") or False),
        "security_priority_mode": bool(state.get("security_priority_mode") or False),
        "latest_vuln_decision_snapshot": dict(state.get("latest_vuln_decision_snapshot") or {}),
        "target_scoring_enabled": bool(state.get("target_scoring_enabled") or False),
        "target_score_breakdown_available": bool(state.get("target_score_breakdown_available") or False),
        "constraint_memory_count": int(state.get("constraint_memory_count") or 0),
        "decision_trace_count": int(state.get("decision_trace_count") or 0),
        "latest_decision_snapshot": dict(state.get("latest_decision_snapshot") or {}),
        "crash_signature_dedup_hit": bool(state.get("crash_signature_dedup_hit") or False),
    }
    try:
        line = json.dumps(payload, separators=(",", ":"), default=str)
    except Exception:
        return
    logger.info("[wf-metrics] {}", line)
