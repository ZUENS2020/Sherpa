from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1

CONTROL_CONTEXT_KEYS = {
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

CONTROL_CONTEXT_DYNAMIC_SUFFIXES = (
    "_timeout_retry_count",
    "_timeout_wait_sec_override",
)

WORKFLOW_CONTEXT_KEYS = {
    "last_step",
    "last_error",
    "message",
    "failed",
    "next",
    "decision_trace_count",
    "latest_decision_snapshot",
    "target_score_breakdown_available",
    "prompt_render_degraded",
    "prompt_render_issue",
    "restart_to_plan",
    "restart_to_plan_reason",
    "restart_to_plan_stage",
    "restart_to_plan_error_text",
    "restart_to_plan_report_path",
    "restart_to_plan_count",
    "cold_start_seed_replan_triggered",
    "degraded_seed_replan_triggered",
    "cold_start_seed_replan_skipped_budget",
    "cold_start_trigger_snapshot",
}

WORKFLOW_CONTEXT_PREFIXES = (
    "analysis_",
    "antlr_",
    "auto_stop_",
    "build_",
    "codex_",
    "constraint_",
    "continuous_",
    "coverage_",
    "crash_",
    "decision_",
    "degraded_",
    "early_",
    "fix_",
    "first_",
    "plan_",
    "prompt_",
    "re_",
    "repair_",
    "restart_",
    "run_",
    "same_",
    "selected_",
    "synthesize_",
    "target_",
)

WORKFLOW_CONTEXT_KEY_SUFFIXES = (
    "_path",
    "_summary",
    "_count",
    "_reason",
    "_kind",
    "_signature",
)

CONTEXT_REJECT_KEYS = {
    "generator",
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


def _coerce_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out[str(k)] = _coerce_json_value(v)
        return out
    if isinstance(value, (list, tuple, set)):
        return [_coerce_json_value(x) for x in value]
    return str(value)


def _is_control_key(key: str) -> bool:
    if key in CONTROL_CONTEXT_KEYS:
        return True
    return key.endswith(CONTROL_CONTEXT_DYNAMIC_SUFFIXES)


def _is_workflow_key(key: str) -> bool:
    if key in WORKFLOW_CONTEXT_KEYS:
        return True
    if key.startswith(WORKFLOW_CONTEXT_PREFIXES):
        return True
    if key.endswith(WORKFLOW_CONTEXT_KEY_SUFFIXES):
        return True
    return False


def _sanitize_doc(
    doc: dict[str, Any],
    *,
    kind: str,
    job_id: str,
) -> dict[str, Any]:
    out = _base_doc(job_id)
    if not isinstance(doc, dict):
        return out
    for raw_key, raw_value in doc.items():
        key = str(raw_key or "").strip()
        if not key or key in _META_KEYS or key in CONTEXT_REJECT_KEYS:
            continue
        if kind == "control":
            if not _is_control_key(key):
                continue
        else:
            if not _is_workflow_key(key):
                continue
        out[key] = _coerce_json_value(raw_value)
    return out


def read_context_docs(context_dir: str | Path | None, *, job_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not context_dir:
        return _base_doc(job_id), _base_doc(job_id)
    ctrl_path, wf_path = context_paths(context_dir)
    ctrl_raw = _read_json(ctrl_path)
    wf_raw = _read_json(wf_path)
    return (
        _sanitize_doc(ctrl_raw, kind="control", job_id=job_id),
        _sanitize_doc(wf_raw, kind="workflow", job_id=job_id),
    )


def write_context_docs(context_dir: str | Path | None, *, control: dict[str, Any], workflow: dict[str, Any], job_id: str) -> None:
    if not context_dir:
        return
    root = Path(context_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    ctrl_path, wf_path = context_paths(root)
    ctrl_doc = _sanitize_doc(control, kind="control", job_id=job_id)
    wf_doc = _sanitize_doc(workflow, kind="workflow", job_id=job_id)
    now = int(time.time())
    ctrl_doc["updated_at"] = now
    wf_doc["updated_at"] = now
    ctrl_path.write_text(json.dumps(ctrl_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    wf_path.write_text(json.dumps(wf_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def merge_result_into_contexts(result: dict[str, Any], *, control: dict[str, Any], workflow: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    out_control = dict(control or {})
    out_workflow = dict(workflow or {})
    for raw_key, raw_value in (result or {}).items():
        key = str(raw_key or "").strip()
        if not key or raw_value is None or key in _META_KEYS or key in CONTEXT_REJECT_KEYS:
            continue
        if _is_control_key(key):
            out_control[key] = _coerce_json_value(raw_value)
            continue
        if _is_workflow_key(key):
            out_workflow[key] = _coerce_json_value(raw_value)
    return out_control, out_workflow


def strip_meta(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in (doc or {}).items() if k not in _META_KEYS}
