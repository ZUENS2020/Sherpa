from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1

CONTROL_CONTEXT_KEYS = {
    "resume_from_step",
    "stop_after_step",
    "time_budget",
    "run_time_budget",
    "coverage_loop_max_rounds",
    "max_fix_rounds",
    "same_error_max_retries",
    "run_oom_retry_count",
    "run_rss_limit_mb_override",
    "run_parallel_fuzzers_override",
    "run_timeout_budget_sec_override",
    "target_node_name",
    "resume_repo_root",
    "last_fuzzer",
    "last_crash_artifact",
    "re_workspace_root",
}

WORKFLOW_CONTEXT_KEYS = {
    "crash_triage_label",
    "crash_triage_confidence",
    "crash_triage_reason",
    "crash_triage_done",
    "repair_mode",
    "repair_origin_stage",
    "repair_error_kind",
    "repair_error_code",
    "repair_signature",
    "repair_recent_attempts",
    "repair_error_digest",
    "decision_trace_count",
    "latest_decision_snapshot",
    "crash_signature_dedup_hit",
    "target_score_breakdown_available",
    "coverage_bottleneck_kind",
    "coverage_bottleneck_reason",
    "restart_to_plan",
    "restart_to_plan_reason",
    "restart_to_plan_stage",
    "restart_to_plan_error_text",
    "restart_to_plan_report_path",
    "restart_to_plan_count",
}

_META_KEYS = {"schema_version", "updated_at", "job_id"}


def context_dir_for_repo_root(repo_root: str | Path | None) -> Path | None:
    txt = str(repo_root or "").strip()
    if not txt:
        return None
    return Path(txt).expanduser() / "fuzz" / "context"


def context_paths(context_dir: str | Path) -> tuple[Path, Path]:
    root = Path(context_dir).expanduser()
    return (root / "control_context.json", root / "workflow_context.json")


def _base_doc(job_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": int(time.time()),
        "job_id": str(job_id or "").strip(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _sanitize_doc(doc: dict[str, Any], allowed_keys: set[str], job_id: str) -> dict[str, Any]:
    out = _base_doc(job_id)
    for k in allowed_keys:
        if k in doc:
            out[k] = doc[k]
    return out


def read_context_docs(context_dir: str | Path | None, *, job_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not context_dir:
        return _base_doc(job_id), _base_doc(job_id)
    ctrl_path, wf_path = context_paths(context_dir)
    ctrl_raw = _read_json(ctrl_path)
    wf_raw = _read_json(wf_path)
    return (
        _sanitize_doc(ctrl_raw, CONTROL_CONTEXT_KEYS, job_id),
        _sanitize_doc(wf_raw, WORKFLOW_CONTEXT_KEYS, job_id),
    )


def write_context_docs(context_dir: str | Path | None, *, control: dict[str, Any], workflow: dict[str, Any], job_id: str) -> None:
    if not context_dir:
        return
    root = Path(context_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    ctrl_path, wf_path = context_paths(root)
    ctrl_doc = _sanitize_doc(control, CONTROL_CONTEXT_KEYS, job_id)
    wf_doc = _sanitize_doc(workflow, WORKFLOW_CONTEXT_KEYS, job_id)
    now = int(time.time())
    ctrl_doc["updated_at"] = now
    wf_doc["updated_at"] = now
    ctrl_path.write_text(json.dumps(ctrl_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    wf_path.write_text(json.dumps(wf_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def merge_result_into_contexts(result: dict[str, Any], *, control: dict[str, Any], workflow: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    out_control = dict(control or {})
    out_workflow = dict(workflow or {})
    for k in CONTROL_CONTEXT_KEYS:
        if k in result and result.get(k) is not None:
            out_control[k] = result.get(k)
    for k in WORKFLOW_CONTEXT_KEYS:
        if k in result and result.get(k) is not None:
            out_workflow[k] = result.get(k)
    return out_control, out_workflow


def strip_meta(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in (doc or {}).items() if k not in _META_KEYS}

