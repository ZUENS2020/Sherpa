from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
SRC_DIR = ROOT / "harness_generator" / "src"
for p in (APP_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import workflow_graph  # noqa: E402


def _write_public_api_surface(
    output_dir: Path,
    job_id: str,
    functions: list[dict],
) -> None:
    """Create companion work/api_functions.json under the fake output dir."""
    work = output_dir / "_k8s_jobs" / job_id / "promefuzz" / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "api_functions.json").write_text(
        json.dumps({"count": len(functions), "functions": functions}, indent=2),
        encoding="utf-8",
    )


def _write_callgraph(output_dir: Path, job_id: str, edges: list[dict]) -> None:
    promefuzz = output_dir / "_k8s_jobs" / job_id / "promefuzz"
    promefuzz.mkdir(parents=True, exist_ok=True)
    (promefuzz / "coverage_hints.json").write_text(
        json.dumps({"schema_version": 1, "callgraph_summary": edges}, indent=2),
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _clear_caches():
    workflow_graph._PUBLIC_API_SYMBOL_CACHE.clear()
    workflow_graph._CALLGRAPH_REVERSE_CACHE.clear()
    yield
    workflow_graph._PUBLIC_API_SYMBOL_CACHE.clear()
    workflow_graph._CALLGRAPH_REVERSE_CACHE.clear()


def _setup_env(monkeypatch, tmp_path: Path, job_id: str = "job123") -> Path:
    output_dir = tmp_path / "shared_output"
    monkeypatch.setenv("SHERPA_JOB_ID", job_id)
    monkeypatch.setenv("SHERPA_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("SHERPA_VULN_PUBLIC_API_ENFORCE", "1")
    return output_dir


def test_non_public_candidate_tagged_and_reachability_capped(tmp_path, monkeypatch) -> None:
    output_dir = _setup_env(monkeypatch, tmp_path)
    _write_public_api_surface(
        output_dir,
        "job123",
        [{"name": "ts_parser_parse_string", "is_public": True, "decl_loc": "api.h:10:1"}],
    )
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
                            "candidate_id": "cand_wasm",
                            "api": "ts_node_start_index_wasm",
                            "file": "binding_web/lib/tree-sitter.c",
                            "target_type": "parser",
                            "vuln_likelihood": 0.78,
                            "exploitability": 0.56,
                            "reachability_confidence": 0.62,
                            "evidence_ids": [],
                        },
                        {
                            "candidate_id": "cand_public",
                            "api": "ts_parser_parse_string",
                            "file": "lib/include/tree_sitter/api.h",
                            "target_type": "parser",
                            "vuln_likelihood": 0.70,
                            "exploitability": 0.55,
                            "reachability_confidence": 0.60,
                            "evidence_ids": [],
                        },
                    ],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    workflow_graph._write_analysis_vuln_candidates(tmp_path, str(analysis_context))
    doc = json.loads((fuzz_dir / "vuln_candidates.json").read_text(encoding="utf-8"))
    by_id = {c["candidate_id"]: c for c in doc["candidates"]}

    wasm = by_id["cand_wasm"]
    pub = by_id["cand_public"]
    assert wasm["api_surface"] == "internal"
    assert pub["api_surface"] == "public"
    # reachability capped for the non-public binding symbol
    assert wasm["reachability_confidence"] <= 0.30
    # public candidate now ranks ahead despite lower raw vuln_likelihood
    assert pub["priority"] > wasm["priority"]
    assert doc["candidates"][0]["candidate_id"] == "cand_public"


def test_line_backfill_from_public_api(tmp_path, monkeypatch) -> None:
    output_dir = _setup_env(monkeypatch, tmp_path)
    _write_public_api_surface(
        output_dir,
        "job123",
        [
            {
                "name": "ts_parser_parse_string",
                "is_public": True,
                "decl_loc": "/repo/lib/include/tree_sitter/api.h:123:5",
            }
        ],
    )
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
                            "candidate_id": "cand_public",
                            "api": "ts_parser_parse_string",
                            "file": "",
                            "line": 0,
                            "target_type": "parser",
                            "vuln_likelihood": 0.7,
                            "exploitability": 0.5,
                            "reachability_confidence": 0.6,
                            "evidence_ids": [],
                        }
                    ],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    workflow_graph._write_analysis_vuln_candidates(tmp_path, str(analysis_context))
    doc = json.loads((fuzz_dir / "vuln_candidates.json").read_text(encoding="utf-8"))
    cand = doc["candidates"][0]
    assert cand["line"] == 123
    assert cand["source_path"].endswith("api.h")


def test_empty_public_set_does_not_misclassify(tmp_path, monkeypatch) -> None:
    # No api_functions.json present -> public set empty -> degrade to legacy
    # heuristic, must NOT flag an ordinary public-looking symbol as internal.
    _setup_env(monkeypatch, tmp_path, job_id="nojob")
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
                            "candidate_id": "cand_plain",
                            "api": "png_read_image",
                            "file": "pngread.c",
                            "target_type": "decoder",
                            "vuln_likelihood": 0.8,
                            "exploitability": 0.7,
                            "reachability_confidence": 0.75,
                            "evidence_ids": [],
                        }
                    ],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    workflow_graph._write_analysis_vuln_candidates(tmp_path, str(analysis_context))
    doc = json.loads((fuzz_dir / "vuln_candidates.json").read_text(encoding="utf-8"))
    cand = doc["candidates"][0]
    # unknown (no oracle) — not wrongly capped
    assert cand["api_surface"] == "unknown"
    assert cand["reachability_confidence"] == pytest.approx(0.75, abs=1e-6)


def test_is_non_public_api_helpers(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SHERPA_VULN_PUBLIC_API_ENFORCE", "1")
    public = frozenset({"ts_parser_parse_string", "png_read_image"})
    # binding pattern
    assert workflow_graph._is_non_public_api("ts_node_index_wasm", public) is True
    # absent from non-empty public set
    assert workflow_graph._is_non_public_api("unmarshal_node_at", public) is True
    # present in public set
    assert workflow_graph._is_non_public_api("png_read_image", public) is False
    # empty public set + benign name -> legacy heuristic says public
    assert workflow_graph._is_non_public_api("png_read_image", frozenset()) is False


def test_nearest_public_entry_via_callgraph() -> None:
    public = frozenset({"public_entry"})
    # public_entry -> mid -> sink
    reverse = {
        "sink": {"mid"},
        "mid": {"public_entry"},
    }
    assert workflow_graph._nearest_public_entry("sink", public, reverse) == "public_entry"
    # no path
    assert workflow_graph._nearest_public_entry("orphan", public, reverse) == ""
    # no callgraph
    assert workflow_graph._nearest_public_entry("sink", public, {}) == ""
