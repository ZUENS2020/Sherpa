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


def test_build_selected_target_row_normalizes_parser_seed_profile_for_decoder(tmp_path: Path) -> None:
    row = workflow_graph._build_selected_target_row(
        repo_root=tmp_path,
        item={
            "name": "readpng_init",
            "api": "readpng_init",
            "target_type": "decoder",
            "seed_profile": "parser-structure",
            "lang": "c",
            "depth_score": 7,
            "depth_class": "medium",
        },
        security_lookup={},
        security_priority_mode=True,
        degrade_reason="",
        score_weights=workflow_graph._vuln_score_weights(),
    )

    assert row["seed_profile"] == "decoder-binary"
    assert row["seed_families_suggested"] == []


def test_build_selected_target_row_normalizes_binary_parser_seed_profile(tmp_path: Path) -> None:
    row = workflow_graph._build_selected_target_row(
        repo_root=tmp_path,
        item={
            "name": "png_read_info",
            "api": "png_read_info",
            "target_type": "parser",
            "seed_profile": "parser-structure",
            "lang": "c",
            "depth_score": 7,
            "depth_class": "medium",
        },
        security_lookup={},
        security_priority_mode=True,
        degrade_reason="",
        score_weights=workflow_graph._vuln_score_weights(),
    )

    assert row["target_type"] == "parser"
    assert row["seed_profile"] == "decoder-binary"
    assert row["seed_families_suggested"] == []
    assert "document_markers" not in row["seed_families_optional"]
    assert "png_chunk_order" in row["seed_families_optional"]


def test_build_selected_target_row_accepts_string_attack_hint(tmp_path: Path) -> None:
    row = workflow_graph._build_selected_target_row(
        repo_root=tmp_path,
        item={
            "name": "png_read_image",
            "api": "png_read_image",
            "target_type": "image",
            "seed_profile": "decoder-binary",
            "lang": "c",
            "attack_hint": "Craft PNG images with oversized IHDR dimensions.",
            "security_score_breakdown": {
                "vuln_likelihood": 0.78,
                "exploitability": 0.65,
                "reachability_confidence": 0.75,
            },
        },
        security_lookup={},
        security_priority_mode=True,
        degrade_reason="",
        score_weights=workflow_graph._vuln_score_weights(),
    )

    assert isinstance(row["attack_hint"], dict)
    assert row["attack_hint"]["trigger_condition"] == "Craft PNG images with oversized IHDR dimensions."
    assert row["attack_hint"]["vuln_category"]


def test_build_selected_target_row_replaces_test_helper_with_public_surrogate(tmp_path: Path) -> None:
    row = workflow_graph._build_selected_target_row(
        repo_root=tmp_path,
        item={
            "name": "test_one_file",
            "api": "test_one_file",
            "file": "contrib/libtests/pngimage.c",
            "target_type": "image",
            "seed_profile": "decoder-binary",
            "lang": "c",
            "vuln_likelihood": 0.9,
            "exploitability": 0.7,
            "reachability_confidence": 0.62,
            "security_signal_scores": {
                "integer_overflow_candidate": 0.78,
            },
        },
        security_lookup={},
        security_priority_mode=True,
        degrade_reason="",
        score_weights=workflow_graph._vuln_score_weights(),
    )

    assert row["target_name"] == "png_read_image"
    assert row["api"] == "png_read_image"
    assert row["advisory_origin_target_name"] == "test_one_file"
    assert row["advisory_origin_api"] == "test_one_file"
    assert row["runtime_replacement_reason"] == "test_demo_helper_public_surrogate"
    assert row["runtime_viability"] == "high"


def test_selected_targets_use_analysis_file_hint_for_public_surrogate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SHERPA_VULN_HUNTING_ENABLED", "1")
    monkeypatch.setenv("SHERPA_VULN_SCORE_MODE", "risk_first_v1")
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True)
    (fuzz_dir / "targets.json").write_text(
        json.dumps(
            [
                {
                    "name": "test_one_file",
                    "api": "test_one_file",
                    "lang": "c",
                    "target_type": "image",
                    "seed_profile": "decoder-binary",
                }
            ]
        ),
        encoding="utf-8",
    )
    (fuzz_dir / "vuln_candidates.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "name": "test_one_file",
                        "target_api": "test_one_file",
                        "signature": "test_one_file(struct display *dp, const char *filename)",
                        "target_file": "contrib/libtests/pngimage.c",
                        "security_signal_scores": {"integer_overflow_candidate": 0.78},
                        "vuln_likelihood": 0.9,
                        "exploitability": 0.7,
                        "reachability_confidence": 0.62,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    selected = workflow_graph._build_selected_targets_doc(tmp_path)

    assert selected[0]["target_name"] == "png_read_image"
    assert selected[0]["advisory_origin_target_name"] == "test_one_file"
    assert selected[0]["runtime_replacement_reason"] == "test_demo_helper_public_surrogate"


def test_selected_targets_replace_example_helper_with_public_surrogate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SHERPA_VULN_HUNTING_ENABLED", "1")
    monkeypatch.setenv("SHERPA_VULN_SCORE_MODE", "risk_first_v1")
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True)
    (fuzz_dir / "targets.json").write_text(
        json.dumps(
            [
                {
                    "name": "readpng2_init",
                    "api": "readpng2_init",
                    "lang": "c",
                    "target_type": "image",
                    "seed_profile": "decoder-binary",
                }
            ]
        ),
        encoding="utf-8",
    )
    (fuzz_dir / "vuln_candidates.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "name": "readpng2_init",
                        "target_api": "readpng2_init",
                        "target_file": "contrib/gregbook/rpng2-x.c",
                        "security_signal_scores": {"integer_overflow_candidate": 0.78},
                        "vuln_likelihood": 0.86,
                        "exploitability": 0.64,
                        "reachability_confidence": 0.60,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    selected = workflow_graph._build_selected_targets_doc(tmp_path)

    assert selected[0]["target_name"] == "png_read_image"
    assert selected[0]["advisory_origin_target_name"] == "readpng2_init"
    assert selected[0]["runtime_replacement_reason"] == "test_demo_helper_public_surrogate"


def test_selected_targets_demote_platform_contrib_helpers_below_core_decoder(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SHERPA_VULN_HUNTING_ENABLED", "1")
    monkeypatch.setenv("SHERPA_VULN_SCORE_MODE", "risk_first_v1")
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True)
    (fuzz_dir / "targets.json").write_text(
        json.dumps(
            [
                {
                    "name": "safe_read",
                    "api": "safe_read",
                    "file": "contrib/arm-neon/linux-auxv.c",
                    "target_type": "decoder",
                    "seed_profile": "decoder-binary",
                    "depth_score": 7,
                    "depth_class": "medium",
                    "runtime_viability": "medium",
                    "vuln_likelihood": 0.78,
                    "exploitability": 0.52,
                    "reachability_confidence": 0.62,
                    "security_signal_scores": {
                        "integer_overflow_candidate": 0.78,
                        "mem_oob_candidate": 0.68,
                    },
                    "evidence_ids": ["EV_SAFE"],
                },
                {
                    "name": "png_read_image",
                    "api": "png_read_image",
                    "file": "pngread.c",
                    "target_type": "decoder",
                    "seed_profile": "decoder-binary",
                    "depth_score": 7,
                    "depth_class": "medium",
                    "runtime_viability": "medium",
                    "vuln_likelihood": 0.74,
                    "exploitability": 0.50,
                    "reachability_confidence": 0.62,
                    "security_signal_scores": {
                        "integer_overflow_candidate": 0.72,
                        "mem_oob_candidate": 0.66,
                    },
                    "evidence_ids": ["EV_PNG"],
                },
            ]
        ),
        encoding="utf-8",
    )

    selected = workflow_graph._build_selected_targets_doc(tmp_path)

    assert selected[0]["target_name"] == "png_read_image"
    safe_read = next(row for row in selected if row["target_name"] == "safe_read")
    assert "non_core_auxiliary_source" in safe_read["penalty_reason"]
    assert safe_read["target_surface_penalty"] > 0


def test_selected_targets_preserve_agent_risk_breakdown_over_exact_candidate_match(tmp_path: Path) -> None:
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True)
    (fuzz_dir / "targets.json").write_text(
        json.dumps(
            [
                {
                    "name": "png_read_png",
                    "api": "png_read_png",
                    "lang": "c-cpp",
                    "target_type": "decoder",
                    "seed_profile": "decoder-binary",
                    "security_priority_mode": True,
                    "security_score_breakdown": {
                        "vuln_likelihood": 0.80,
                        "exploitability": 0.65,
                        "reachability_confidence": 0.80,
                        "coverage_gap": 0.50,
                        "complexity_depth": 0.40,
                        "api_relevance": 0.90,
                        "consumer_order_support": 0.50,
                        "recent_yield_penalty": 0.0,
                    },
                    "score_total": 0.7305,
                    "vuln_candidate_ids": ["mem_oob_025", "integer_overflow_025"],
                    "evidence_ids": ["EV0098", "EV0097", "EV0096", "EV0099"],
                    "depth_score": 7,
                    "depth_class": "medium",
                },
                {
                    "name": "readpng_init",
                    "api": "readpng_init",
                    "lang": "c-cpp",
                    "target_type": "image",
                    "seed_profile": "parser-structure",
                    "security_priority_mode": True,
                    "security_score_breakdown": {
                        "vuln_likelihood": 0.78,
                        "exploitability": 0.1716,
                        "reachability_confidence": 0.62,
                        "coverage_gap": 0.35,
                        "complexity_depth": 0.40,
                        "api_relevance": 0.60,
                        "consumer_order_support": 0.40,
                        "recent_yield_penalty": 0.0,
                    },
                    "score_total": 0.5376,
                    "vuln_candidate_ids": ["integer_overflow_001"],
                    "evidence_ids": ["EV0002"],
                    "depth_score": 3,
                    "depth_class": "medium",
                },
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (fuzz_dir / "target_analysis.json").write_text(
        json.dumps(
            {
                "recommended_targets": [
                    {
                        "name": "readpng_init",
                        "api": "readpng_init",
                        "target_type": "image",
                        "vuln_likelihood": 0.78,
                        "exploitability": 0.1716,
                        "reachability_confidence": 0.62,
                        "security_signal_scores": {
                            "integer_overflow_candidate": 0.78,
                        },
                        "evidence_ids": ["EV0002"],
                    }
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    selected = workflow_graph._build_selected_targets_doc(tmp_path)

    assert selected[0]["target_name"] == "png_read_png"
    assert selected[0]["security_score_breakdown"]["vuln_likelihood"] == 0.80
    assert selected[1]["target_name"] == "readpng_init"
