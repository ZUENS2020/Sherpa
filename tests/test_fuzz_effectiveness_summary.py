from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
SRC_DIR = ROOT / "harness_generator" / "src"
for p in (APP_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import workflow_graph
from fuzz_unharnessed_repo import parse_libfuzzer_final_stats


def test_parse_libfuzzer_final_stats_uses_last_progress_line() -> None:
    log = "\n".join(
        [
            "#10 NEW cov: 101 ft: 202 corp: 11/12Kb lim: 1000 exec/s: 333 rss: 44Mb",
            "#11 REDUCE cov: 111 ft: 222 corp: 12/1Mb lim: 1000 exec/s: 444 rss: 55Mb",
        ]
    )

    stats = parse_libfuzzer_final_stats(log)

    assert stats["iteration"] == 11
    assert stats["cov"] == 111
    assert stats["ft"] == 222
    assert stats["corpus_files"] == 12
    assert stats["corpus_size_bytes"] == 1024 * 1024
    assert stats["execs_per_sec"] == 444
    assert stats["rss_mb"] == 55


def test_write_run_summary_emits_fuzz_effectiveness_artifacts(tmp_path: Path) -> None:
    repo_root = tmp_path
    out_dir = repo_root / "fuzz" / "out"
    corpus_dir = repo_root / "fuzz" / "corpus" / "demo_fuzz"
    artifacts_dir = out_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    fuzzer_bin = out_dir / "demo_fuzz"
    fuzzer_bin.write_text("", encoding="utf-8")
    os.chmod(fuzzer_bin, 0o755)
    (out_dir / "demo_fuzz.options").write_text("[libfuzzer]\n", encoding="utf-8")
    (artifacts_dir / "crash-1").write_text("boom", encoding="utf-8")
    (corpus_dir / "seed1").write_bytes(b"AAAA")
    (corpus_dir / "seed2").write_bytes(b"BBBBBB")

    workflow_graph._write_run_summary(
        {
            "repo_url": "https://example.com/repo.git",
            "repo_root": str(repo_root),
            "last_step": "run",
            "step_count": 10,
            "build_attempts": 2,
            "build_rc": 0,
            "build_error_kind": "",
            "build_error_code": "",
            "run_rc": 0,
            "last_error": "",
            "crash_found": False,
            "crash_evidence": "none",
            "run_error_kind": "",
            "message": "ok",
            "run_details": [
                {
                    "fuzzer": "demo_fuzz",
                    "rc": 0,
                    "crash_found": False,
                    "crash_evidence": "none",
                    "run_error_kind": "",
                    "new_artifacts": [],
                    "first_artifact": "",
                    "final_cov": 123,
                    "final_ft": 456,
                    "final_iteration": 789,
                    "final_execs_per_sec": 99,
                    "final_rss_mb": 64,
                    "final_corpus_files": 2,
                    "final_corpus_size_bytes": 10,
                    "corpus_files": 2,
                    "corpus_size_bytes": 10,
                }
            ],
            "coverage_seed_quality": {
                "initial_corpus_files": 2,
                "initial_corpus_bytes": 10,
                "initial_inited_cov": 100,
                "initial_inited_ft": 200,
                "early_new_units_30s": 0,
                "early_new_units_60s": 0,
                "final_corpus_files": 2,
                "final_corpus_bytes": 10,
                "corpus_retention_ratio_files": 1.0,
                "corpus_retention_ratio_bytes": 1.0,
                "cov_growth_slope_pre_plateau": 0.0,
                "ft_growth_slope_pre_plateau": 0.0,
                "plateau_after_sec": 180,
                "quality_flags": ["high_homogeneity"],
            },
            "coverage_seed_families_required": ["document_markers", "flow_structures"],
            "coverage_seed_families_covered": ["document_markers"],
            "coverage_seed_families_missing": ["flow_structures"],
            "coverage_target_api": "fmt::println",
            "coverage_seed_counts_raw": {"repo_examples": 2, "ai": 3, "radamsa": 4, "total": 9},
            "coverage_seed_counts_filtered": {"repo_examples": 1, "ai": 2, "radamsa": 1, "total": 4},
            "coverage_seed_noise_rejected_count": 5,
            "coverage_seed_family_coverage": {"covered": ["replacement_fields"], "missing": ["width_precision"]},
            "synthesize_selected_target_name": "parse_replacement_field_then_tail",
            "synthesize_selected_target_api": "parse_replacement_field_then_tail",
            "synthesize_observed_target_api": "fmt::println",
            "synthesize_observed_harness": "println_fuzz.cc",
            "synthesize_target_drifted": True,
            "synthesize_target_drift_reason": "selected target is not a runtime entrypoint",
            "synthesize_target_relation": "runtime wrapper for same formatting path",
            "synthesize_target_runtime_viability": "low",
            "synthesize_readme_consistent": True,
            "synthesize_build_scaffold_consistent": True,
        }
    )

    run_summary_json = repo_root / "run_summary.json"
    fuzz_effectiveness_json = out_dir / "fuzz_effectiveness.json"
    fuzz_effectiveness_md = out_dir / "fuzz_effectiveness.md"

    assert run_summary_json.is_file()
    assert fuzz_effectiveness_json.is_file()
    assert fuzz_effectiveness_md.is_file()

    summary = json.loads(run_summary_json.read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["fuzz_inventory"]["fuzzer_count"] == 1
    assert summary["fuzz_inventory"]["corpus_total_files"] == 2
    assert summary["fuzz_inventory"]["artifact_count"] == 1
    assert summary["seed_quality"]["initial_corpus_files"] == 2
    assert summary["seed_family_coverage"]["missing"] == ["flow_structures"]
    assert summary["seed_bootstrap"]["noise_rejected_count"] == 5
    assert summary["synthesize_target"]["relation"] == "runtime wrapper for same formatting path"
    assert summary["synthesize_target"]["readme_consistent"] is True
    assert summary["synthesize_target"]["build_scaffold_consistent"] is True
    assert summary["coverage_loop"]["target_api"] == "fmt::println"
    assert summary["build_error_kind"] == ""
    assert summary["build_error_code"] == ""
    assert len(summary["run_details"]) == 1

    effectiveness = json.loads(fuzz_effectiveness_json.read_text(encoding="utf-8"))
    assert effectiveness["status"] == "ok"
    assert effectiveness["fuzz_inventory"]["fuzzer_count"] == 1
    assert effectiveness["run_details"][0]["final_cov"] == 123
