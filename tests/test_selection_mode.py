from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "harness_generator" / "src" / "langchain_agent"
SRC = ROOT / "harness_generator" / "src"
for p in (APP, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import workflow_graph as wg  # noqa: E402
import workflow_target_selection as wts  # noqa: E402


def _sort(rows, mode):
    return wts.sort_ranked_items(
        rows,
        security_priority_mode=True,
        is_internal_api_symbol_fn=lambda s: False,
        runtime_viability_rank_fn=lambda s: 0,
        selection_mode=mode,
    )


def test_mode_env_parsing(monkeypatch):
    monkeypatch.delenv("SHERPA_SELECTION_MODE", raising=False)
    assert wg._selection_mode() == "score"
    monkeypatch.setenv("SHERPA_SELECTION_MODE", "llm_first")
    assert wg._selection_mode() == "llm_first"
    monkeypatch.setenv("SHERPA_SELECTION_MODE", "LLM-FIRST")
    assert wg._selection_mode() == "llm_first"
    monkeypatch.setenv("SHERPA_SELECTION_MODE", "whatever")
    assert wg._selection_mode() == "score"


def test_llm_first_orders_by_llm_risk_ignoring_bias_and_penalty():
    # In score mode, the entrypoint_bias would lift the lower-likelihood entry;
    # in llm_first the agent's raw vuln_likelihood wins instead.
    leaf = {"target_name": "scan_time", "vuln_likelihood": 0.85, "entrypoint_bias": 0.0,
            "target_score_penalty": 0.0}
    entry = {"target_name": "toml_parse", "vuln_likelihood": 0.62, "entrypoint_bias": 0.25,
             "target_score_penalty": 0.0}
    # score mode: entry 0.62+0.25=0.87 > leaf 0.85 -> entry first
    assert _sort([leaf, entry], "score")[0]["target_name"] == "toml_parse"
    # llm_first: pure likelihood 0.85 > 0.62 -> leaf first (bias ignored)
    assert _sort([leaf, entry], "llm_first")[0]["target_name"] == "scan_time"


def test_llm_first_keeps_feedback_gating():
    # a spent/exhausted target sinks even with a higher raw likelihood
    spent = {"target_name": "hot", "vuln_likelihood": 0.95,
             "penalty_reason": "coverage_exhausted_target"}
    fresh = {"target_name": "fresh", "vuln_likelihood": 0.70, "penalty_reason": ""}
    ranked = _sort([spent, fresh], "llm_first")
    assert ranked[0]["target_name"] == "fresh"  # exhausted demoted despite higher vl


def test_score_mode_unchanged_by_default():
    # default path still uses effective_risk (penalty + entrypoint_bias)
    a = {"target_name": "a", "vuln_likelihood": 0.8, "target_surface_penalty": 0.5}
    b = {"target_name": "b", "vuln_likelihood": 0.7, "target_surface_penalty": 0.0}
    # score: a eff 0.3 vs b 0.7 -> b first (penalty applied)
    assert _sort([a, b], "score")[0]["target_name"] == "b"
    # llm_first: pure 0.8 > 0.7 -> a first (penalty ignored in key)
    assert _sort([a, b], "llm_first")[0]["target_name"] == "a"
