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

import coverage_replay


def _make_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    corpus_dir = repo_root / "fuzz" / "corpus" / "demo_fuzz"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    replay_binary = repo_root / "fuzz" / "out" / "demo_fuzz"
    replay_binary.parent.mkdir(parents=True, exist_ok=True)
    replay_binary.write_text("bin-v1", encoding="utf-8")
    replay_binary.chmod(0o755)
    return repo_root, replay_binary


def test_summarize_export_doc_extracts_frontier_functions(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src").mkdir(parents=True, exist_ok=True)
    source_path = repo_root / "src" / "pngrutil.c"
    source_path.write_text("int x;\n", encoding="utf-8")

    export_doc = {
        "data": [
            {
                "files": [
                    {
                        "filename": str(source_path),
                        "summary": {"regions": {"covered": 4, "count": 8}},
                    }
                ],
                "functions": [
                    {
                        "name": "png_handle_iCCP",
                        "count": 1,
                        "filenames": [str(source_path)],
                        "regions": [
                            [1843, 1, 1843, 12, 1, 0, 0, 0],
                            [1844, 1, 1844, 12, 0, 0, 0, 0],
                            [1845, 1, 1845, 12, 0, 0, 0, 0],
                        ],
                    },
                    {
                        "name": "png_crc_finish",
                        "count": 1,
                        "filenames": [str(source_path)],
                        "regions": [
                            [1900, 1, 1900, 12, 1, 0, 0, 0],
                            [1901, 1, 1901, 12, 1, 0, 0, 0],
                        ],
                    },
                ],
            }
        ]
    }

    summary = coverage_replay._summarize_export_doc(export_doc, repo_root)
    assert summary["covered_function_count"] == 2
    assert summary["repo_file_count"] == 1
    assert summary["unique_frontier_functions"] == 1
    assert summary["nearby_uncovered_regions"] == 2
    assert summary["frontier_functions"][0]["name"] == "png_handle_iCCP"
    assert summary["frontier_functions"][0]["uncovered_regions_nearby"] == 2


def test_frontier_focus_uses_analysis_context_for_target_relevance(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    fuzz_dir = repo_root / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "analysis_context.json").write_text(
        json.dumps(
            {
                "analysis_evidence": {
                    "vuln_candidate_inventory": [
                        {
                            "target_api": "parse_zip",
                            "attack_hint": {
                                "key_code_path": ["parse_zip", "copy_entry_data", "memcpy"],
                            },
                        }
                    ],
                    "callgraph_summary": [
                        {"summary": "entry -> parse_zip -> copy_entry_data"},
                    ],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    focus = coverage_replay._frontier_focus(repo_root, "parse_zip")
    assert "parse_zip" in focus["focus_function_names"]
    assert "copy_entry_data" in focus["focus_function_names"]
    assert "parse" in focus["focus_tokens"] or "parse_zip" in focus["focus_function_names"]


def test_build_frontier_summary_prefers_partial_frontier_signal(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    replay_binary = repo_root / "fuzz" / "out" / "demo_fuzz"
    replay_binary.parent.mkdir(parents=True, exist_ok=True)
    replay_binary.write_text("bin-v1", encoding="utf-8")

    summary, frontier_doc = coverage_replay._build_frontier_summary(
        manifest_inputs=[
            {
                "input_relpath": "fuzz/corpus/demo_fuzz/slow.bin",
                "size_bytes": 1024,
                "exec_time_us": 5000,
                "covered_function_count": 20,
                "covered_region_count": 30,
                "covered_functions_sample": ["fn_slow"],
                "repo_file_count": 1,
                "replay_status": "ok",
                "unique_frontier_functions": 0,
                "nearby_uncovered_regions": 0,
                "target_relevance_count": 0,
                "closest_target_distance": 2,
                "frontier_functions": [],
            },
            {
                "input_relpath": "fuzz/corpus/demo_fuzz/frontier.bin",
                "size_bytes": 256,
                "exec_time_us": 200,
                "covered_function_count": 8,
                "covered_region_count": 12,
                "covered_functions_sample": ["fn_frontier"],
                "repo_file_count": 1,
                "replay_status": "ok",
                "unique_frontier_functions": 2,
                "nearby_uncovered_regions": 6,
                "target_relevance_count": 1,
                "closest_target_distance": 0,
                "frontier_functions": [
                    {
                        "name": "png_handle_iCCP",
                        "file": "src/pngrutil.c",
                        "line": 1843,
                        "uncovered_regions_nearby": 6,
                        "region_coverage_ratio": 0.33,
                    }
                ],
            },
        ],
        repo_root=repo_root,
        replay_binary=replay_binary,
        binary_hash="sha256:demo",
        pending_inputs=0,
        failed_inputs=0,
        focus={"target_api": "parse_zip", "focus_function_names": {"png_handle_iCCP"}, "focus_tokens": {"png", "iccp"}},
    )
    assert summary["top_inputs"][0]["input_relpath"].endswith("frontier.bin")
    assert summary["top_inputs"][0]["frontier_score"] > summary["top_inputs"][1]["frontier_score"]
    assert summary["top_inputs"][0]["target_relevance_count"] >= 1
    assert summary["top_frontier_functions"][0]["best_distance_to_target"] >= 0
    assert frontier_doc["function_to_inputs"]["fn_frontier"] == ["fuzz/corpus/demo_fuzz/frontier.bin"]


def test_collect_per_input_frontier_tracks_pending_and_deletes_removed_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root, replay_binary = _make_repo(tmp_path)
    corpus_dir = repo_root / "fuzz" / "corpus" / "demo_fuzz"
    for name in ("a.bin", "b.bin", "c.bin"):
        (corpus_dir / name).write_bytes(name.encode("utf-8"))

    monkeypatch.setattr(coverage_replay, "_llvm_tools", lambda: ("llvm-profdata", "llvm-cov"))

    def fake_export(**kwargs):
        input_path = Path(kwargs["input_path"])
        return {
            "replay_status": "ok",
            "replay_error": "",
            "exec_time_us": 100,
            "exit_code": 0,
            "covered_function_count": 10 if input_path.name == "a.bin" else 5,
            "covered_region_count": 20 if input_path.name == "a.bin" else 8,
            "total_region_count": 30,
            "covered_functions_sample": [f"fn_{input_path.stem}"],
            "uncovered_functions_sample": [],
            "repo_file_count": 1,
        }

    monkeypatch.setattr(coverage_replay, "_export_input_coverage", fake_export)
    monkeypatch.setenv("SHERPA_PER_INPUT_REPLAY_MAX_INPUTS", "2")

    first = coverage_replay.collect_per_input_frontier(
        repo_root=repo_root,
        fuzzer_name="demo_fuzz",
        replay_binary=replay_binary,
    )
    assert first.stage_success is True
    assert first.pending_inputs == 1
    assert first.processed_inputs == 2
    assert first.frontier_summary["top_frontier_function_count"] >= 1
    assert first.frontier_summary["top_frontier_functions"][0]["name"].startswith("fn_")
    manifest_doc = json.loads(Path(first.manifest_path).read_text(encoding="utf-8"))
    statuses = {
        item["input_relpath"]: item["replay_status"]
        for item in manifest_doc["inputs"]
    }
    assert len(statuses) == 3
    assert list(statuses.values()).count("pending") == 1

    os.unlink(corpus_dir / "a.bin")
    monkeypatch.setenv("SHERPA_PER_INPUT_REPLAY_MAX_INPUTS", "8")
    second = coverage_replay.collect_per_input_frontier(
        repo_root=repo_root,
        fuzzer_name="demo_fuzz",
        replay_binary=replay_binary,
    )
    assert second.pending_inputs == 0
    frontier_doc = json.loads(Path(second.frontier_path).read_text(encoding="utf-8"))
    assert "function_to_inputs" in frontier_doc
    manifest_doc = json.loads(Path(second.manifest_path).read_text(encoding="utf-8"))
    relpaths = [item["input_relpath"] for item in manifest_doc["inputs"]]
    assert all(not rel.endswith("a.bin") for rel in relpaths)
    assert second.total_inputs == 2


def test_collect_per_input_frontier_replays_all_inputs_after_binary_hash_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root, replay_binary = _make_repo(tmp_path)
    corpus_dir = repo_root / "fuzz" / "corpus" / "demo_fuzz"
    (corpus_dir / "seed.bin").write_bytes(b"seed")

    monkeypatch.setattr(coverage_replay, "_llvm_tools", lambda: ("llvm-profdata", "llvm-cov"))
    calls: list[str] = []

    def fake_export(**kwargs):
        input_path = Path(kwargs["input_path"])
        calls.append(input_path.name)
        return {
            "replay_status": "ok",
            "replay_error": "",
            "exec_time_us": 10,
            "exit_code": 0,
            "covered_function_count": 1,
            "covered_region_count": 2,
            "total_region_count": 2,
            "covered_functions_sample": ["fn_seed"],
            "uncovered_functions_sample": [],
            "repo_file_count": 1,
        }

    monkeypatch.setattr(coverage_replay, "_export_input_coverage", fake_export)

    coverage_replay.collect_per_input_frontier(
        repo_root=repo_root,
        fuzzer_name="demo_fuzz",
        replay_binary=replay_binary,
    )
    replay_binary.write_text("bin-v2", encoding="utf-8")
    coverage_replay.collect_per_input_frontier(
        repo_root=repo_root,
        fuzzer_name="demo_fuzz",
        replay_binary=replay_binary,
    )
    assert calls == ["seed.bin", "seed.bin"]
