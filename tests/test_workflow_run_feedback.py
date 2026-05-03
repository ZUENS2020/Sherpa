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


def test_build_run_feedback_summary_merges_function_and_path_views() -> None:
    summary = workflow_graph._build_run_feedback_summary(
        repo_root=Path("/tmp/repo"),
        source_report={
            "coverage_pct": 31.2,
            "covered_functions": 12,
            "total_functions": 38,
            "uncovered_function_details": [
                {
                    "name": "png_read_info",
                    "file": "pngrutil.c",
                    "line": 412,
                    "execution_count": 0,
                    "region_coverage_ratio": 0.0,
                },
                {
                    "name": "png_handle_PLTE",
                    "file": "pngread.c",
                    "line": 188,
                    "execution_count": 0,
                    "region_coverage_ratio": 0.1,
                },
            ],
        },
        frontier_summary={
            "top_input_count": 2,
            "top_inputs": [
                {
                    "input_relpath": "fuzz/corpus/png/a.bin",
                    "frontier_score": 4.8,
                    "covered_function_count": 11,
                    "covered_region_count": 33,
                    "covered_functions_sample": ["png_read_info"],
                    "frontier_functions": [
                        {
                            "name": "png_read_info",
                            "file": "pngrutil.c",
                            "line": 412,
                            "uncovered_regions_nearby": 7,
                            "region_coverage_ratio": 0.2,
                        }
                    ],
                }
            ],
            "top_frontier_functions": [
                {
                    "name": "png_read_info",
                    "file": "pngrutil.c",
                    "line": 412,
                    "best_distance_to_target": 1,
                    "input_relpaths": ["fuzz/corpus/png/a.bin"],
                    "input_count": 1,
                }
            ],
        },
    )

    assert summary["function_gap_count"] == 2
    assert summary["path_frontier_count"] == 1
    assert summary["top_function_gaps"][0]["name"] == "png_read_info"
    assert summary["top_function_gaps"][0]["file"] == "pngrutil.c"
    assert summary["top_path_frontiers"][0]["input_relpath"] == "fuzz/corpus/png/a.bin"
    assert summary["top_files"][0]["path"] == "pngrutil.c"


def test_write_run_feedback_artifact_persists_summary(tmp_path: Path) -> None:
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    result = workflow_graph._write_run_feedback_artifact(
        repo_root=tmp_path,
        source_report={
            "coverage_pct": 25.0,
            "covered_functions": 5,
            "total_functions": 20,
            "uncovered_function_details": [
                {"name": "inflate", "file": "inflate.c", "line": 99, "execution_count": 0}
            ],
        },
        frontier_summary={
            "top_input_count": 1,
            "top_inputs": [{"input_relpath": "fuzz/corpus/inflate/a.bin", "frontier_score": 1.5}],
            "top_frontier_functions": [],
        },
    )

    path = fuzz_dir / "run_feedback.json"
    assert result["path"] == str(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1
    assert doc["summary"]["top_function_gaps"][0]["name"] == "inflate"
    assert doc["summary"]["top_path_frontiers"][0]["input_relpath"] == "fuzz/corpus/inflate/a.bin"
