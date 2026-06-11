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


@pytest.fixture(autouse=True)
def _clear_caches():
    workflow_graph._PUBLIC_API_SYMBOL_CACHE.clear()
    workflow_graph._CALLGRAPH_REVERSE_CACHE.clear()
    yield
    workflow_graph._PUBLIC_API_SYMBOL_CACHE.clear()
    workflow_graph._CALLGRAPH_REVERSE_CACHE.clear()


def _write_public_api_surface(output_dir: Path, job_id: str, functions: list[dict]) -> None:
    work = output_dir / "_k8s_jobs" / job_id / "promefuzz" / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "api_functions.json").write_text(
        json.dumps({"count": len(functions), "functions": functions}, indent=2),
        encoding="utf-8",
    )


def _setup_env(monkeypatch, tmp_path: Path, job_id: str = "jobUB") -> Path:
    output_dir = tmp_path / "shared_output"
    monkeypatch.setenv("SHERPA_JOB_ID", job_id)
    monkeypatch.setenv("SHERPA_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("SHERPA_VULN_PUBLIC_API_ENFORCE", "1")
    return output_dir


def test_is_unlinkable_binding_api():
    assert workflow_graph._is_unlinkable_binding_api("ts_node_start_index_wasm") is True
    assert workflow_graph._is_unlinkable_binding_api("napi_create_object") is True
    assert workflow_graph._is_unlinkable_binding_api("Java_com_x_foo_jni") is True
    # plain static / internal symbols are NOT unlinkable bindings (owning-source
    # linkage can pull them in) -> they keep the escape-hatch
    assert workflow_graph._is_unlinkable_binding_api("unmarshal_node_at") is False
    assert workflow_graph._is_unlinkable_binding_api("png_read_image") is False
    assert workflow_graph._is_unlinkable_binding_api("") is False


def _row(api: str, *, repo_root: Path, monkeypatch, tmp_path) -> dict:
    return workflow_graph._build_selected_target_row(
        repo_root=repo_root,
        item={
            "name": api,
            "api": api,
            "target_type": "parser",
            "vuln_likelihood": 0.78,
            "exploitability": 0.6,
            "reachability_confidence": 0.62,
        },
        security_lookup={},
        security_priority_mode=True,
        degrade_reason="",
        score_weights={"vuln_likelihood": 0.5, "exploitability": 0.3, "reachability_confidence": 0.2},
        vuln_priority_by_api={},
    )


def test_high_risk_wasm_binding_is_dropped_not_excepted(tmp_path, monkeypatch):
    # No callgraph -> the internal->public remap cannot fire. A high-vuln wasm
    # binding must be flagged for drop rather than admitted via the escape-hatch.
    output_dir = _setup_env(monkeypatch, tmp_path)
    _write_public_api_surface(
        output_dir, "jobUB", [{"name": "ts_parser_parse_string", "is_public": True}]
    )
    row = _row("ts_node_start_index_wasm", repo_root=tmp_path, monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert row["unlinkable_binding_dropped"] is True
    assert row["api_surface_exception"]["used"] is False
    assert "unlinkable_binding" in row["api_surface_exception"]["reason"]


def test_high_risk_plain_static_still_uses_escape_hatch(tmp_path, monkeypatch):
    output_dir = _setup_env(monkeypatch, tmp_path)
    _write_public_api_surface(
        output_dir, "jobUB", [{"name": "ts_parser_parse_string", "is_public": True}]
    )
    row = _row("unmarshal_node_at", repo_root=tmp_path, monkeypatch=monkeypatch, tmp_path=tmp_path)
    # static internal sink with vuln_likelihood>=0.75 keeps the escape-hatch
    assert row["unlinkable_binding_dropped"] is False
    assert row["api_surface_exception"]["used"] is True


def test_filter_drops_flagged_rows_but_keeps_at_least_one():
    rows = [
        {"target_name": "a_wasm", "unlinkable_binding_dropped": True},
        {"target_name": "public_api", "unlinkable_binding_dropped": False},
        {"target_name": "b_wasm", "unlinkable_binding_dropped": True},
    ]
    out = workflow_graph._apply_selected_target_filters(rows)
    names = [r["target_name"] for r in out]
    assert names == ["public_api"]

    # all-unlinkable -> keep them (never hand downstream an empty selection)
    all_bad = [
        {"target_name": "a_wasm", "unlinkable_binding_dropped": True},
        {"target_name": "b_wasm", "unlinkable_binding_dropped": True},
    ]
    out2 = workflow_graph._apply_selected_target_filters(all_bad)
    assert len(out2) == 2


def test_enforce_off_restores_escape_hatch(tmp_path, monkeypatch):
    output_dir = _setup_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SHERPA_VULN_PUBLIC_API_ENFORCE", "0")
    _write_public_api_surface(
        output_dir, "jobUB", [{"name": "ts_parser_parse_string", "is_public": True}]
    )
    row = _row("ts_node_start_index_wasm", repo_root=tmp_path, monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert row["unlinkable_binding_dropped"] is False
