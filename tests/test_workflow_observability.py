from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
SRC_DIR = ROOT / "harness_generator" / "src"
for p in (APP_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import workflow_observability as wo


def test_record_decision_trace_writes_jsonl_and_updates_state(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SHERPA_DECISION_TRACE_MAX_ITEMS", "20")
    state = {"repo_root": str(tmp_path)}

    out = wo.record_decision_trace(
        state,
        stage="plan",
        tool="opencode",
        model="GLM-5",
        latency_ms=123,
        decision_snapshot={"kind": "choose_target"},
    )
    out = wo.record_decision_trace(
        out,
        stage="synthesize",
        decision_snapshot={"kind": "choose_repair"},
    )
    out = wo.record_decision_trace(
        out,
        stage="build",
        decision_snapshot={"kind": "choose_seed"},
    )

    assert int(out.get("decision_trace_count") or 0) >= 3
    traces = list(out.get("decision_traces") or [])
    assert len(traces) == 3
    assert traces[-1]["stage"] == "build"
    assert out.get("latest_decision_snapshot") == {"kind": "choose_seed"}

    trace_path = tmp_path / "fuzz" / "decision_trace.jsonl"
    assert trace_path.exists()
    lines = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 3
    assert lines[0]["stage"] == "plan"
    assert lines[-1]["stage"] == "build"


def test_decision_snapshot_from_state_has_expected_keys() -> None:
    snapshot = wo.decision_snapshot_from_state(
        {
            "repo_root": "/tmp/repo",
            "coverage_loop_round": 2,
            "coverage_loop_max_rounds": 8,
            "coverage_should_improve": True,
            "crash_triage_label": "harness_bug",
            "security_priority_mode": True,
            "vuln_candidate_count": 5,
        }
    )
    assert snapshot["repo_root"] == "/tmp/repo"
    assert snapshot["coverage_loop_round"] == 2
    assert snapshot["coverage_should_improve"] is True
    assert snapshot["crash_triage_label"] == "harness_bug"
    assert snapshot["security_priority_mode"] is True
    assert snapshot["vuln_candidate_count"] == 5


def test_emit_fuzz_metrics_does_not_raise() -> None:
    wo.emit_fuzz_metrics(
        {
            "last_step": "run",
            "run_details": [
                {
                    "fuzzer": "a",
                    "final_cov": 10,
                    "final_ft": 22,
                    "final_execs_per_sec": 1234,
                    "crash_found": False,
                }
            ],
            "coverage_history": [],
            "coverage_run_feedback_summary": {
                "function_gap_count": 2,
                "path_frontier_count": 1,
            },
        }
    )
