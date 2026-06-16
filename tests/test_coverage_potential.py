from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "harness_generator" / "src" / "langchain_agent"
SRC = ROOT / "harness_generator" / "src"
for p in (APP, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import workflow_public_api as pub  # noqa: E402
import workflow_graph as wg  # noqa: E402


def _edges(pairs):
    return [{"caller": a, "callee": b, "summary": f"{a}->{b}"} for a, b in pairs]


def _write_callgraph(out_dir: Path, job_id: str, pairs):
    p = out_dir / "_k8s_jobs" / job_id / "promefuzz"
    p.mkdir(parents=True, exist_ok=True)
    (p / "coverage_hints.json").write_text(
        json.dumps({"callgraph_summary": _edges(pairs)}), encoding="utf-8"
    )


def _setup(monkeypatch, tmp_path, pairs, job="jobCP"):
    out = tmp_path / "shared_output"
    monkeypatch.setenv("SHERPA_JOB_ID", job)
    monkeypatch.setenv("SHERPA_OUTPUT_DIR", str(out))
    monkeypatch.delenv("SHERPA_VULN_COVERAGE_POTENTIAL", raising=False)
    monkeypatch.delenv("SHERPA_VULN_COVERAGE_POTENTIAL_NORM", raising=False)
    pub._CALLGRAPH_FORWARD_CACHE.clear()
    _write_callgraph(out, job, pairs)
    return Path(tmp_path)


# whole-parser entry reaches many; a leaf reaches ~0
_PARSER_CG = [
    ("toml_parse", "next_token"),
    ("toml_parse", "parse_keyval"),
    ("parse_keyval", "parse_value"),
    ("parse_value", "parse_array"),
    ("parse_array", "scan_string"),
    ("parse_value", "scan_time"),
    ("parse_value", "scan_digits"),
]


def test_forward_loader_builds_caller_to_callees(tmp_path, monkeypatch):
    root = _setup(monkeypatch, tmp_path, _PARSER_CG)
    fwd = pub._load_callgraph_forward(root)
    assert fwd["toml_parse"] == {"next_token", "parse_keyval"}
    assert fwd["parse_value"] == {"parse_array", "scan_time", "scan_digits"}


def test_reachable_fanout_entry_vs_leaf(tmp_path, monkeypatch):
    root = _setup(monkeypatch, tmp_path, _PARSER_CG)
    fwd = pub._load_callgraph_forward(root)
    entry = pub._reachable_fanout("toml_parse", fwd)
    leaf = pub._reachable_fanout("scan_time", fwd)
    assert entry >= 6  # reaches the whole tree
    assert leaf == 0  # leaf scanner reaches nothing
    assert entry > leaf


def test_reachable_fanout_caps(tmp_path, monkeypatch):
    # a long chain a0->a1->...->a200; cap at max_nodes keeps it bounded
    pairs = [(f"a{i}", f"a{i+1}") for i in range(200)]
    root = _setup(monkeypatch, tmp_path, pairs)
    fwd = pub._load_callgraph_forward(root)
    n = pub._reachable_fanout("a0", fwd, max_nodes=50, max_depth=8)
    assert n <= 50


def test_coverage_potential_normalized(tmp_path, monkeypatch):
    root = _setup(monkeypatch, tmp_path, _PARSER_CG)
    monkeypatch.setenv("SHERPA_VULN_COVERAGE_POTENTIAL_NORM", "6")
    fwd = pub._load_callgraph_forward(root)
    cp_entry = pub.coverage_potential("toml_parse", fwd)
    cp_leaf = pub.coverage_potential("scan_time", fwd)
    assert cp_entry == 1.0  # fanout >= norm -> clamps to 1.0
    assert cp_leaf == 0.0
    assert pub.coverage_potential("toml_parse", {}) == 0.0  # empty CG -> 0


def test_structural_bias_beats_name_heuristic(tmp_path, monkeypatch):
    root = _setup(monkeypatch, tmp_path, _PARSER_CG)
    monkeypatch.setenv("SHERPA_VULN_COVERAGE_POTENTIAL_NORM", "6")
    monkeypatch.setenv("SHERPA_VULN_COVERAGE_POTENTIAL_WEIGHT", "0.3")
    bias, cov = wg._entrypoint_risk_bias("toml_parse", "parser", root)
    assert cov == 1.0
    assert bias == 0.3  # 0.3*1.0 structural > 0.15 name heuristic


def test_falls_back_to_name_when_callgraph_empty(tmp_path, monkeypatch):
    # no callgraph file -> coverage_potential 0 -> name heuristic fallback
    out = tmp_path / "shared_output"
    monkeypatch.setenv("SHERPA_JOB_ID", "nocg")
    monkeypatch.setenv("SHERPA_OUTPUT_DIR", str(out))
    monkeypatch.delenv("SHERPA_VULN_ENTRYPOINT_BIAS", raising=False)
    pub._CALLGRAPH_FORWARD_CACHE.clear()
    bias, cov = wg._entrypoint_risk_bias("toml_parse", "parser", Path(tmp_path))
    assert cov == 0.0
    assert bias == 0.15  # name heuristic (parser type, default weight)


def test_kill_switch_disables_structural_signal(tmp_path, monkeypatch):
    root = _setup(monkeypatch, tmp_path, _PARSER_CG)
    monkeypatch.setenv("SHERPA_VULN_COVERAGE_POTENTIAL", "0")
    bias, cov = wg._entrypoint_risk_bias("toml_parse", "parser", root)
    assert cov == 0.0  # structural signal off
    assert bias == 0.15  # name heuristic still applies
