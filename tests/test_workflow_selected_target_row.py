from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
SRC_DIR = ROOT / "harness_generator" / "src"
for p in (APP_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import workflow_graph


def test_build_selected_target_row_keeps_contract_shape(tmp_path: Path) -> None:
    row = workflow_graph._build_selected_target_row(
        repo_root=tmp_path,
        item={
            "name": "parse_packet",
            "api": "parse_packet",
            "target_type": "parser",
            "seed_profile": "parser-token",
            "lang": "c",
            "depth_score": 8,
            "depth_class": "deep",
            "selection_bias_reason": "risk",
            "selection_rationale": "runtime parser",
            "risk_signals": ["uaf_candidate"],
            "coverage_gap": 6.0,
        },
        security_lookup={},
        security_priority_mode=True,
        degrade_reason="",
        score_weights=workflow_graph._vuln_score_weights(),
    )

    assert row["target_name"] == "parse_packet"
    assert row["target"] == "parse_packet"
    assert isinstance(row["score_breakdown"], dict)
    assert isinstance(row["security_score_breakdown"], dict)
    assert isinstance(row["security_signals"], list)
    assert isinstance(row["security_signal_scores"], dict)
    assert isinstance(row["api_surface_exception"], dict)
    assert row["target_score_breakdown_available"] is True
