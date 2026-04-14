from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "harness_generator" / "src" / "langchain_agent" / "main.py"
WORKER = ROOT / "harness_generator" / "src" / "langchain_agent" / "k8s_job_worker.py"


def test_stage_payload_includes_restart_and_runtime_override_fields() -> None:
    text = MAIN.read_text(encoding="utf-8")
    required = [
        '"context_dir": (context_dir or None)',
        '"run_unlimited_round_budget_sec": int(',
        '"target_node_name": (current_node_name if can_pin_node else None)',
        "control_ctx[\"run_oom_retry_count\"] = str(oom_retry_count + 1)",
        "control_ctx[\"run_rss_limit_mb_override\"] = str(retry_rss)",
        "control_ctx[\"run_parallel_fuzzers_override\"] = \"1\"",
    ]
    for marker in required:
        assert marker in text


def test_worker_forwards_restart_and_decision_fields_to_fuzz_logic() -> None:
    text = WORKER.read_text(encoding="utf-8")
    required = [
        'context_dir = str(payload.get("context_dir") or "").strip()',
        "control_ctx = strip_meta(control_doc)",
        'run_rss_limit_mb_override = str(control_ctx.get("run_rss_limit_mb_override") or "").strip()',
        "context_dir=(context_dir or None),",
    ]
    for marker in required:
        assert marker in text
