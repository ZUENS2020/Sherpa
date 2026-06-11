from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "harness_generator" / "src" / "langchain_agent"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import workflow_observability as obs  # noqa: E402


def _state(tmp_path: Path) -> dict:
    (tmp_path / "fuzz").mkdir(parents=True, exist_ok=True)
    return {
        "repo_root": str(tmp_path),
        "fuzz_max_cov": 25,
        "fuzz_max_ft": 109,
        "fuzz_total_execs_per_sec": 130,
        "fuzz_crash_found": False,
        "fuzz_coverage_plateau_streak": 3,
        "fuzz_coverage_bottleneck_kind": "harness_limited",
        "vuln_candidate_count": 24,
        "vuln_hunt_candidate_count": 7,
        "crash_vuln_candidate_count": 1,
        "analysis_companion_ready": True,
        "analysis_companion_backend": "promefuzz",
        "constraint_memory_count": 2,
        "workflow_active_step": "synthesize",
    }


def test_snapshot_includes_cross_stage_fields(tmp_path):
    snap = obs.decision_snapshot_from_state(_state(tmp_path))
    # newly added cross-stage fields are present and typed
    assert snap["fuzz_max_cov"] == 25
    assert snap["fuzz_max_ft"] == 109
    assert snap["fuzz_total_execs_per_sec"] == 130
    assert snap["fuzz_coverage_bottleneck_kind"] == "harness_limited"
    assert snap["vuln_hunt_candidate_count"] == 7
    assert snap["crash_vuln_candidate_count"] == 1
    assert snap["analysis_companion_ready"] is True
    assert snap["analysis_companion_backend"] == "promefuzz"
    assert snap["constraint_memory_count"] == 2
    assert snap["workflow_active_step"] == "synthesize"


def test_record_embeds_cross_stage_and_writes_jsonl(tmp_path):
    out = obs.record_decision_trace(_state(tmp_path), stage="synthesize", tool="opencode", latency_ms=1200)
    # cross_stage embedded on the in-memory trace
    traces = out["decision_traces"]
    assert traces[-1]["cross_stage"]["fuzz_max_cov"] == 25
    assert traces[-1]["stage"] == "synthesize"
    # persisted to fuzz/decision_trace.jsonl with the cross_stage payload
    path = tmp_path / "fuzz" / "decision_trace.jsonl"
    assert path.is_file()
    last = json.loads(path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert last["cross_stage"]["vuln_hunt_candidate_count"] == 7
    assert last["tool"] == "opencode"
    assert last["latency_ms"] == 1200


def test_record_is_safe_without_repo_root():
    # no repo_root -> no file write, no exception, still returns updated state
    out = obs.record_decision_trace({"fuzz_max_cov": 5}, stage="run")
    assert out["decision_traces"][-1]["cross_stage"]["fuzz_max_cov"] == 5
