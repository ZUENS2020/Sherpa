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
