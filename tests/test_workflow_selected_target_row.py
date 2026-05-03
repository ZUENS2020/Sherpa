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
import workflow_context_store as ctx_store


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
    assert isinstance(row["attack_hint"], dict)
    assert row["attack_hint"]["trigger_condition"]
    assert isinstance(row["attack_hint"]["key_code_path"], list)
    assert isinstance(row["attack_hint"]["boundary_values"], list)
    assert row["signal_type"]
    assert isinstance(row["evidence_ids"], list)
    assert row["validation_status"] == "pending"
    assert row["target_score_breakdown_available"] is True


def test_build_selected_target_row_penalizes_exhausted_and_oom_target(tmp_path: Path) -> None:
    context_dir = ctx_store.context_dir_for_repo_root(tmp_path)
    assert context_dir is not None
    ctx_store.write_context_docs(
        context_dir,
        control={"last_fuzzer": "png_read_image_fuzz"},
        workflow={
            "run_error_kind": "oom_killed",
            "coverage_exhausted_targets": [{"name": "png_read_image", "round": 4}],
            "coverage_history": [
                {
                    "target_api": "png_read_image",
                    "target_name": "png_read_image",
                    "plateau_detected": True,
                    "max_cov": 11,
                    "prev_cov": 11,
                    "max_ft": 24,
                    "prev_ft": 24,
                },
                {
                    "target_api": "png_read_image",
                    "target_name": "png_read_image",
                    "plateau_detected": True,
                    "max_cov": 11,
                    "prev_cov": 11,
                    "max_ft": 24,
                    "prev_ft": 24,
                },
            ],
        },
        job_id="job-1",
    )

    row = workflow_graph._build_selected_target_row(
        repo_root=tmp_path,
        item={
            "name": "png_read_image",
            "api": "png_read_image",
            "target_type": "decoder",
            "seed_profile": "decoder-binary",
            "lang": "c",
            "wrapper_fuzzer_name": "png_read_image_fuzz",
            "depth_score": 8,
            "selection_rationale": "runtime decoder",
            "risk_signals": ["mem_oob_candidate"],
            "coverage_gap": 6.0,
            "vuln_likelihood": 0.9,
            "exploitability": 0.8,
            "reachability_confidence": 0.85,
        },
        security_lookup={},
        security_priority_mode=True,
        degrade_reason="",
        score_weights=workflow_graph._vuln_score_weights(),
    )

    assert row["target_score_penalty"] >= 4.0
    reason = str(row["target_score_penalty_reason"] or "")
    assert "coverage_exhausted_target" in reason
    assert "persistent_low_yield_target" in reason
    assert "recent_oom_killed" in reason


def test_build_selected_target_row_exposes_signal_sources_and_callback_penalty(tmp_path: Path) -> None:
    row = workflow_graph._build_selected_target_row(
        repo_root=tmp_path,
        item={
            "name": "readpng2_end_callback",
            "api": "readpng2_end_callback",
            "target_type": "decoder",
            "seed_profile": "decoder-binary",
            "lang": "c",
            "depth_score": 3,
            "depth_class": "medium",
            "selection_bias_reason": "+read",
            "selection_rationale": "callback wrapper",
            "risk_signals": ["integer_overflow_candidate"],
            "risk_signal_source_breakdown": {
                "regex": ["integer_overflow_candidate"],
                "weak_file": [],
            },
            "security_signal_scores": {
                "mem_oob_candidate": 0.0,
                "integer_overflow_candidate": 0.63,
                "format_string_candidate": 0.0,
                "path_traversal_candidate": 0.0,
                "command_injection_candidate": 0.0,
                "authz_bypass_candidate": 0.0,
                "null_deref_candidate": 0.0,
                "uaf_candidate": 0.0,
            },
            "coverage_gap": 2.5,
        },
        security_lookup={},
        security_priority_mode=True,
        degrade_reason="",
        score_weights=workflow_graph._vuln_score_weights(),
    )

    assert isinstance(row["risk_signal_source_breakdown"], dict)
    assert float(row["callback_penalty"]) > 0.0
    assert float(row["execution_depth_bias"]) < 0.0
