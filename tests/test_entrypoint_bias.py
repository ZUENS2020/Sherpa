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


def test_detects_top_level_parse_entry_not_subparser_or_leaf():
    # whole-input entrypoints
    assert wg._is_library_entrypoint("toml_parse")
    assert wg._is_library_entrypoint("json_loads")
    assert wg._is_library_entrypoint("yaml_load")
    assert wg._is_library_entrypoint("png_decode")
    assert wg._is_library_entrypoint("parse")
    # decoder read-entrypoints (callgraph often incomplete for generated-header
    # libs like libpng, so the name fallback must still recognize these)
    assert wg._is_library_entrypoint("png_read_image")
    assert wg._is_library_entrypoint("png_read_info")
    assert wg._is_library_entrypoint("png_read_png")
    # but low-level readers stay leaves, not promoted
    assert not wg._is_library_entrypoint("png_read_filter_row")
    assert not wg._is_library_entrypoint("png_read_data")
    assert not wg._is_library_entrypoint("png_read_chunk")
    # sub-parsers (must NOT count — these are the leaves we want to de-prioritize)
    assert not wg._is_library_entrypoint("parse_array")
    assert not wg._is_library_entrypoint("parse_keyval")
    assert not wg._is_library_entrypoint("parse_inline_table")
    # leaf scanners
    assert not wg._is_library_entrypoint("scan_time")
    assert not wg._is_library_entrypoint("scan_digits")
    assert not wg._is_library_entrypoint("next_token")


def test_bias_strongest_for_parser_type(monkeypatch):
    monkeypatch.delenv("SHERPA_VULN_ENTRYPOINT_BIAS", raising=False)
    assert wg._library_entrypoint_bias("toml_parse", "parser") == 0.15
    assert wg._library_entrypoint_bias("toml_parse", "generic") == 0.09
    assert wg._library_entrypoint_bias("scan_time", "parser") == 0.0


def test_bias_disabled_by_env(monkeypatch):
    monkeypatch.setenv("SHERPA_VULN_ENTRYPOINT_BIAS", "0")
    assert wg._library_entrypoint_bias("toml_parse", "parser") == 0.0


def test_entrypoint_outranks_higher_likelihood_leaf_in_risk_first():
    # leaf scanner with higher raw likelihood vs entrypoint with the bias
    leaf = {"target_name": "scan_time", "vuln_likelihood": 0.78, "entrypoint_bias": 0.0}
    entry = {"target_name": "toml_parse", "vuln_likelihood": 0.66, "entrypoint_bias": 0.15}
    ranked = wts.sort_ranked_items(
        [leaf, entry],
        security_priority_mode=True,
        is_internal_api_symbol_fn=lambda s: False,
        runtime_viability_rank_fn=lambda s: 0,
    )
    # effective risk: scan_time 0.78 vs toml_parse 0.66+0.15=0.81 -> entry first
    assert ranked[0]["target_name"] == "toml_parse"


def test_leaf_still_wins_when_likelihood_gap_large():
    leaf = {"target_name": "scan_time", "vuln_likelihood": 0.95, "entrypoint_bias": 0.0}
    entry = {"target_name": "toml_parse", "vuln_likelihood": 0.60, "entrypoint_bias": 0.15}
    ranked = wts.sort_ranked_items(
        [leaf, entry],
        security_priority_mode=True,
        is_internal_api_symbol_fn=lambda s: False,
        runtime_viability_rank_fn=lambda s: 0,
    )
    # 0.95 vs 0.75 -> a genuinely much riskier leaf is not overridden
    assert ranked[0]["target_name"] == "scan_time"
