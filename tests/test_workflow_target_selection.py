from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
SRC_DIR = ROOT / "harness_generator" / "src"
for p in (APP_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import workflow_target_selection as sel


def test_sort_ranked_items_risk_first_prefers_vuln_metrics() -> None:
    rows = [
        {
            "target_name": "safe",
            "api": "safe_api",
            "vuln_likelihood": 0.2,
            "exploitability": 0.2,
            "reachability_confidence": 0.2,
            "security_signals": [],
            "target_score": 9.0,
            "depth_score": 9,
            "runtime_viability": "native_high",
            "api_surface_exception": {"used": True},
        },
        {
            "target_name": "risk",
            "api": "risk_api",
            "vuln_likelihood": 0.9,
            "exploitability": 0.8,
            "reachability_confidence": 0.7,
            "security_signals": ["uaf_candidate"],
            "target_score": 6.0,
            "depth_score": 7,
            "runtime_viability": "native_high",
            "api_surface_exception": {"used": True},
        },
    ]
    sorted_rows = sel.sort_ranked_items(
        rows,
        security_priority_mode=True,
        is_internal_api_symbol_fn=lambda _: False,
        runtime_viability_rank_fn=lambda _: 3,
    )
    assert sorted_rows[0]["target_name"] == "risk"


def test_sort_ranked_items_risk_first_uses_priority_and_evidence_count() -> None:
    rows = [
        {
            "target_name": "low-priority",
            "api": "risk_api_a",
            "priority": 0.61,
            "vuln_likelihood": 0.82,
            "exploitability": 0.75,
            "reachability_confidence": 0.70,
            "evidence_ids": ["EV1"],
            "security_signals": ["mem_oob_candidate"],
            "target_score": 5.9,
            "depth_score": 6,
            "runtime_viability": "native_high",
            "api_surface_exception": {"used": True},
        },
        {
            "target_name": "high-priority",
            "api": "risk_api_b",
            "priority": 0.93,
            "vuln_likelihood": 0.82,
            "exploitability": 0.75,
            "reachability_confidence": 0.70,
            "evidence_ids": ["EV1", "EV2", "EV3"],
            "security_signals": ["mem_oob_candidate"],
            "target_score": 5.9,
            "depth_score": 6,
            "runtime_viability": "native_high",
            "api_surface_exception": {"used": True},
        },
    ]
    sorted_rows = sel.sort_ranked_items(
        rows,
        security_priority_mode=True,
        is_internal_api_symbol_fn=lambda _: False,
        runtime_viability_rank_fn=lambda _: 3,
    )
    assert sorted_rows[0]["target_name"] == "high-priority"


def test_sort_ranked_items_risk_first_breaks_ties_with_execution_depth_and_callback_penalty() -> None:
    rows = [
        {
            "target_name": "readpng2_end_callback",
            "api": "readpng2_end_callback",
            "priority": 0.91,
            "vuln_likelihood": 0.78,
            "exploitability": 0.18,
            "reachability_confidence": 0.62,
            "evidence_ids": ["EV1", "EV2"],
            "security_signals": ["integer_overflow_candidate"],
            "target_score": 0.82,
            "depth_score": 3,
            "runtime_viability": "medium",
            "api_surface_exception": {"used": True},
            "execution_depth_bias": -0.2,
            "callback_penalty": 0.35,
            "target_score_penalty": 0.0,
        },
        {
            "target_name": "readpng2_decode_row",
            "api": "readpng2_decode_row",
            "priority": 0.91,
            "vuln_likelihood": 0.78,
            "exploitability": 0.18,
            "reachability_confidence": 0.62,
            "evidence_ids": ["EV1", "EV2"],
            "security_signals": ["integer_overflow_candidate"],
            "target_score": 0.82,
            "depth_score": 3,
            "runtime_viability": "medium",
            "api_surface_exception": {"used": True},
            "execution_depth_bias": 0.35,
            "callback_penalty": 0.0,
            "target_score_penalty": 0.0,
        },
    ]

    sorted_rows = sel.sort_ranked_items(
        rows,
        security_priority_mode=True,
        is_internal_api_symbol_fn=lambda _: False,
        runtime_viability_rank_fn=lambda _: 3,
    )
    assert sorted_rows[0]["target_name"] == "readpng2_decode_row"


def test_assign_execution_priority_sets_rank_and_must_run() -> None:
    rows = [
        {"target_name": "a", "target_type": "generic"},
        {"target_name": "b", "target_type": "parser"},
        {"target_name": "c", "target_type": "generic"},
    ]
    out = sel.assign_execution_priority(rows, max_targets=2)
    assert out[0]["rank"] == 1
    assert out[0]["execution_priority"] == 1
    assert out[0]["must_run"] is True
    assert out[1]["execution_priority"] == 2
    assert out[1]["must_run"] is True
    assert out[2]["execution_priority"] == 0
