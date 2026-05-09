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


def test_analysis_context_writes_vuln_candidate_worklist(tmp_path: Path) -> None:
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True)
    analysis_context = fuzz_dir / "analysis_context.json"
    analysis_context.write_text(
        json.dumps(
            {
                "analysis_evidence": {
                    "security_evidence": [
                        {
                            "evidence_id": "EV-1",
                            "signal_id": "mem_oob_candidate",
                            "severity": "high",
                            "confidence": 0.9,
                            "source_path": "src/pngread.c",
                            "line": 42,
                            "summary": "attacker controlled length reaches memcpy",
                        }
                    ],
                    "vuln_candidate_inventory": [
                        {
                            "candidate_id": "cand_png_read",
                            "api": "png_read_image",
                            "file": "src/pngread.c",
                            "target_type": "decoder",
                            "vuln_likelihood": 0.88,
                            "exploitability": 0.75,
                            "reachability_confidence": 0.82,
                            "evidence_ids": ["EV-1"],
                            "security_priority_reason": "unchecked image row length",
                        }
                    ],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = workflow_graph._write_analysis_vuln_candidates(tmp_path, str(analysis_context))
    doc = json.loads((fuzz_dir / "vuln_candidates.json").read_text(encoding="utf-8"))

    assert result["analysis_candidate_count"] == 1
    assert doc["candidate_count"] == 1
    candidate = doc["candidates"][0]
    assert candidate["candidate_id"] == "cand_png_read"
    assert candidate["source_stage"] == "analysis"
    assert candidate["validation_status"] == "pending"
    assert candidate["target_api"] == "png_read_image"
    assert candidate["priority"] > 0.8
    assert candidate["attack_hint"]["trigger_condition"]
    assert candidate["evidence"][0]["evidence_id"] == "EV-1"


def test_selected_targets_consume_vuln_candidate_worklist(tmp_path: Path) -> None:
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True)
    (fuzz_dir / "targets.json").write_text(
        json.dumps(
            [
                {
                    "name": "safe_api",
                    "api": "safe_api",
                    "lang": "c",
                    "target_type": "generic",
                    "seed_profile": "generic",
                },
                {
                    "name": "png_read_image",
                    "api": "png_read_image",
                    "lang": "c",
                    "target_type": "decoder",
                    "seed_profile": "decoder-binary",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (fuzz_dir / "vuln_candidates.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_count": 1,
                "candidates": [
                    {
                        "candidate_id": "cand_png_read",
                        "source_stage": "analysis",
                        "validation_status": "pending",
                        "target_api": "png_read_image",
                        "target_name": "png_read_image",
                        "target_type": "decoder",
                        "signal_type": "mem_oob_candidate",
                        "vuln_likelihood": 0.95,
                        "exploitability": 0.86,
                        "reachability_confidence": 0.83,
                        "priority": 0.91,
                        "evidence_ids": ["EV-1"],
                        "attack_hint": {
                            "trigger_condition": "row bytes overflow allocation",
                            "key_code_path": ["png_read_image"],
                            "boundary_values": ["width=0xFFFFFFFF"],
                            "vuln_category": "heap-buffer-overflow",
                            "sanitizer_hint": "address",
                        },
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = workflow_graph._build_selected_targets_doc(tmp_path)

    assert rows[0]["target_name"] == "png_read_image"
    assert rows[0]["vuln_candidate_id"] == "cand_png_read"
    assert rows[0]["vuln_candidate_priority"] == 0.91
    assert rows[0]["attack_hint"]["trigger_condition"] == "row bytes overflow allocation"
    assert rows[0]["security_priority_mode"] is True


def test_vuln_hunt_materializes_candidates_summary_and_events(tmp_path: Path) -> None:
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True)
    analysis_context = fuzz_dir / "analysis_context.json"
    analysis_context.write_text(
        json.dumps(
            {
                "analysis_evidence": {
                    "security_evidence": [
                        {
                            "evidence_id": "EV-1",
                            "signal_id": "integer_overflow_candidate",
                            "severity": "high",
                            "confidence": 0.88,
                            "source_path": "pngrutil.c",
                            "line": 77,
                            "summary": "chunk length participates in allocation sizing",
                        }
                    ],
                    "vuln_candidate_inventory": [
                        {
                            "candidate_id": "cand_chunk_len",
                            "api": "png_handle_iCCP",
                            "file": "pngrutil.c",
                            "line": 77,
                            "target_type": "decoder",
                            "signal_type": "integer_overflow_candidate",
                            "vuln_likelihood": 0.9,
                            "exploitability": 0.8,
                            "reachability_confidence": 0.7,
                            "evidence_ids": ["EV-1"],
                        }
                    ],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class Gen:
        repo_root = tmp_path

    out = workflow_graph._node_vuln_hunt(
        {
            "generator": Gen(),
            "analysis_context_path": str(analysis_context),
            "coverage_plateau_streak": 3,
            "coverage_seed_generation_degraded": True,
            "coverage_target_name": "old_target",
        }
    )

    candidates = json.loads((fuzz_dir / "vuln_candidates.json").read_text(encoding="utf-8"))
    candidate = candidates["candidates"][0]
    assert candidate["candidate_id"] == "cand_chunk_len"
    assert candidate["source_path"] == "pngrutil.c"
    assert candidate["line"] == 77
    assert candidate["risk_type"] == "integer_overflow_candidate"
    assert candidate["detectability_confidence"] > 0
    assert candidate["attempt_count"] == 0
    assert candidate["last_result"] == {}

    summary = (fuzz_dir / "vuln_hunt_summary.md").read_text(encoding="utf-8")
    assert "cand_chunk_len" in summary
    assert "png_handle_iCCP" in summary

    events = (fuzz_dir / "vuln_hunt_events.jsonl").read_text(encoding="utf-8").splitlines()
    assert events
    event = json.loads(events[-1])
    assert event["event_type"] == "coverage_plateau"
    assert event["target_name"] == "old_target"

    assert out["vuln_hunt_enabled"] is True
    assert out["vuln_hunt_candidate_count"] == 1
    assert out["vuln_hunt_active_candidate_id"] == "cand_chunk_len"
    assert out["vuln_hunt_summary_path"].endswith("fuzz/vuln_hunt_summary.md")


def test_vuln_hunt_invokes_opencode_skill_when_key_is_available(tmp_path: Path, monkeypatch) -> None:
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True)
    analysis_context = fuzz_dir / "analysis_context.json"
    analysis_context.write_text(
        json.dumps(
            {
                "analysis_evidence": {
                    "security_evidence": [
                        {
                            "evidence_id": "EV-1",
                            "signal_id": "mem_oob_candidate",
                            "confidence": 0.9,
                            "source_path": "decode.c",
                            "line": 10,
                            "summary": "bounds-sensitive decode path",
                        }
                    ],
                    "vuln_candidate_inventory": [],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    class _Patcher:
        def run_codex_command(self, prompt: str, **kwargs):
            calls.append({"prompt": prompt, **kwargs})
            (fuzz_dir / "vuln_candidates.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "candidate_count": 1,
                        "candidates": [
                            {
                                "candidate_id": "ai_decode_bounds",
                                "source_stage": "vuln_hunt",
                                "validation_status": "pending",
                                "target_api": "decode",
                                "target_name": "decode",
                                "source_path": "decode.c",
                                "line": 10,
                                "risk_type": "mem_oob_candidate",
                                "vuln_likelihood": 0.91,
                                "exploitability": 0.82,
                                "reachability_confidence": 0.77,
                                "detectability_confidence": 0.8,
                                "priority": 0.89,
                                "evidence_ids": ["EV-1"],
                                "attack_hint": {"trigger_condition": "oversized length"},
                                "attempt_count": 0,
                                "last_result": {},
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (tmp_path / "done").write_text("fuzz/vuln_hunt_summary.md\n", encoding="utf-8")
            return None

    class Gen:
        repo_root = tmp_path
        patcher = _Patcher()

    monkeypatch.setattr(workflow_graph, "_has_codex_key", lambda: True)

    out = workflow_graph._node_vuln_hunt(
        {"generator": Gen(), "analysis_context_path": str(analysis_context)}
    )

    assert calls
    assert calls[0]["stage_skill"] == "vuln_hunt"
    assert "fuzz/vuln_candidates.json" in str(calls[0]["prompt"])
    assert out["vuln_hunt_active_candidate_id"] == "ai_decode_bounds"


def test_selected_targets_skip_exhausted_vuln_candidates(tmp_path: Path) -> None:
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True)
    (fuzz_dir / "targets.json").write_text(
        json.dumps(
            [
                {
                    "name": "exhausted_api",
                    "api": "exhausted_api",
                    "lang": "c",
                    "target_type": "decoder",
                    "seed_profile": "decoder-binary",
                },
                {
                    "name": "pending_api",
                    "api": "pending_api",
                    "lang": "c",
                    "target_type": "decoder",
                    "seed_profile": "decoder-binary",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (fuzz_dir / "vuln_candidates.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_count": 2,
                "candidates": [
                    {
                        "candidate_id": "cand_exhausted",
                        "validation_status": "exhausted",
                        "target_api": "exhausted_api",
                        "target_name": "exhausted_api",
                        "target_type": "decoder",
                        "vuln_likelihood": 0.99,
                        "exploitability": 0.99,
                        "reachability_confidence": 0.99,
                        "priority": 0.99,
                    },
                    {
                        "candidate_id": "cand_pending",
                        "validation_status": "pending",
                        "target_api": "pending_api",
                        "target_name": "pending_api",
                        "target_type": "decoder",
                        "vuln_likelihood": 0.7,
                        "exploitability": 0.7,
                        "reachability_confidence": 0.7,
                        "priority": 0.7,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = workflow_graph._build_selected_targets_doc(tmp_path)

    assert rows[0]["target_name"] == "pending_api"
    assert rows[0]["vuln_candidate_id"] == "cand_pending"


def test_vuln_candidate_feedback_marks_plateau_candidate_cooling(tmp_path: Path) -> None:
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True)
    (fuzz_dir / "vuln_candidates.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_count": 1,
                "candidates": [
                    {
                        "candidate_id": "cand_png",
                        "validation_status": "pending",
                        "target_api": "png_read_image",
                        "target_name": "png_read_image",
                        "priority": 0.9,
                        "attempt_count": 0,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    event = {
        "ts": 123,
        "event_type": "coverage_plateau",
        "target_name": "png_read_image",
        "target_api": "png_read_image",
        "coverage_plateau_streak": 2,
        "coverage_quality_flags": ["low_early_yield"],
    }

    out = workflow_graph._update_vuln_candidate_feedback(
        tmp_path,
        {"vuln_hunt_active_candidate_id": "cand_png", "coverage_plateau_streak": 2},
        event,
    )
    doc = json.loads((fuzz_dir / "vuln_candidates.json").read_text(encoding="utf-8"))
    candidate = doc["candidates"][0]

    assert out["vuln_hunt_rerun_requested"] is True
    assert candidate["validation_status"] == "cooling"
    assert candidate["attempt_count"] == 1
    assert candidate["last_result"]["event_type"] == "coverage_plateau"


def test_write_analysis_vuln_candidates_does_not_promote_weak_file_signal_to_high_score(
    tmp_path: Path,
) -> None:
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True)
    analysis_context = fuzz_dir / "analysis_context.json"
    analysis_context.write_text(
        json.dumps(
            {
                "analysis_evidence": {
                    "security_evidence": [],
                    "vuln_candidate_inventory": [
                        {
                            "candidate_id": "cand_png_callback",
                            "api": "readpng2_end_callback",
                            "file": "contrib/gregbook/readpng2.c",
                            "target_type": "decoder",
                            "vuln_likelihood": 0.63,
                            "exploitability": 0.18,
                            "reachability_confidence": 0.62,
                            "evidence_ids": [],
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
                            "risk_signal_source_breakdown": {
                                "regex": [],
                                "semantic": [],
                                "weak_file": ["integer_overflow_candidate"],
                            },
                        }
                    ],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = workflow_graph._write_analysis_vuln_candidates(tmp_path, str(analysis_context))
    doc = json.loads((fuzz_dir / "vuln_candidates.json").read_text(encoding="utf-8"))

    assert result["candidate_count"] == 1
    candidate = doc["candidates"][0]
    assert candidate["candidate_id"] == "cand_png_callback"
    assert candidate["security_signal_scores"]["integer_overflow_candidate"] == 0.63
    assert candidate["risk_signal_source_breakdown"]["weak_file"] == ["integer_overflow_candidate"]
