from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "harness_generator" / "src" / "langchain_agent"
SRC = ROOT / "harness_generator" / "src"
for p in (APP, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import workflow_graph as wg  # noqa: E402


# ---- A: non-harnessable target detection + filter ---------------------------
def test_non_harnessable_detection():
    for junk in ("CALLOC", "MALLOC", "malloc", "calloc", "free", "expand_ptrarr", "norm_ptrarr_grow", "xrealloc"):
        assert wg._is_non_harnessable_target(junk), junk
    for ok in ("toml_parse", "parse_keyval", "parse_inline_table", "scan_time", "png_read_info"):
        assert not wg._is_non_harnessable_target(ok), ok


def test_filter_drops_non_harnessable_keep_one():
    rows = [
        {"target_name": "CALLOC", "non_harnessable_dropped": True},
        {"target_name": "parse_keyval", "non_harnessable_dropped": False},
        {"target_name": "expand_ptrarr", "non_harnessable_dropped": True},
    ]
    out = wg._apply_selected_target_filters(rows)
    assert [r["target_name"] for r in out] == ["parse_keyval"]
    # all-junk -> keep them rather than hand downstream an empty selection
    alljunk = [{"target_name": "CALLOC", "non_harnessable_dropped": True}]
    assert len(wg._apply_selected_target_filters(alljunk)) == 1


# ---- B: validator tolerates non-harnessable must_run, not real targets ------
def _write_plan(tmp_path: Path, targets: list[dict], harness_files: list[str]):
    fuzz = tmp_path / "fuzz"
    fuzz.mkdir(parents=True, exist_ok=True)
    (fuzz / "execution_plan.json").write_text(
        json.dumps({"execution_targets": targets}), encoding="utf-8"
    )
    for h in harness_files:
        (fuzz / h).write_text("int x;", encoding="utf-8")


def test_validator_tolerates_missing_nonharnessable_must_run(tmp_path):
    # plan marks CALLOC (a macro) must_run but only parse_keyval was harnessed
    _write_plan(
        tmp_path,
        [
            {"target_name": "parse_keyval", "expected_fuzzer_name": "parse_keyval", "must_run": True},
            {"target_name": "CALLOC", "expected_fuzzer_name": "CALLOC", "must_run": True},
        ],
        ["parse_keyval.c"],
    )
    ok, reason, doc = wg._validate_execution_plan_harness_consistency(tmp_path)
    assert ok is True, reason
    assert "CALLOC" in (doc.get("dropped_non_harnessable_must_run") or [])


def test_validator_still_fails_on_real_missing_must_run(tmp_path):
    _write_plan(
        tmp_path,
        [
            {"target_name": "parse_keyval", "expected_fuzzer_name": "parse_keyval", "must_run": True},
            {"target_name": "vformat", "expected_fuzzer_name": "vformat", "must_run": True},
        ],
        ["parse_keyval.c"],
    )
    ok, reason, _ = wg._validate_execution_plan_harness_consistency(tmp_path)
    assert ok is False
    assert "vformat" in reason


# ---- C: public entrypoint injected as a candidate ---------------------------
def _setup_public_api(tmp_path: Path, monkeypatch, funcs: list[dict]):
    out = tmp_path / "shared_output"
    monkeypatch.setenv("SHERPA_JOB_ID", "jobC")
    monkeypatch.setenv("SHERPA_OUTPUT_DIR", str(out))
    monkeypatch.setenv("SHERPA_VULN_PUBLIC_API_ENFORCE", "1")
    monkeypatch.setenv("SHERPA_VULN_HUNTING_ENABLED", "1")
    monkeypatch.setenv("SHERPA_VULN_SCORE_MODE", "risk_first_v1")
    work = out / "_k8s_jobs" / "jobC" / "promefuzz" / "work"
    work.mkdir(parents=True)
    (work / "api_functions.json").write_text(
        json.dumps({"count": len(funcs), "functions": funcs}), encoding="utf-8"
    )
    wg._PUBLIC_API_SYMBOL_CACHE.clear()


def test_public_entrypoint_injected_as_candidate(tmp_path, monkeypatch):
    _setup_public_api(
        tmp_path,
        monkeypatch,
        [
            {"name": "toml_parse", "is_public": True, "decl_loc": "toml.h:10:1"},
            {"name": "parse_keyval", "is_public": False},
        ],
    )
    fuzz = tmp_path / "fuzz"
    fuzz.mkdir(parents=True, exist_ok=True)
    # plan proposed only the internal sub-parser; toml_parse is NOT in targets.json
    (fuzz / "targets.json").write_text(
        json.dumps(
            [{"name": "parse_keyval", "api": "parse_keyval", "lang": "c",
              "target_type": "parser", "seed_profile": "generic",
              "vuln_likelihood": 0.78, "exploitability": 0.5, "reachability_confidence": 0.6}]
        ),
        encoding="utf-8",
    )
    rows = wg._build_selected_targets_doc(tmp_path)
    apis = {str(r.get("api")) for r in rows}
    assert "toml_parse" in apis, "public entrypoint should be injected as a candidate"
    entry = next(r for r in rows if r.get("api") == "toml_parse")
    assert float(entry.get("entrypoint_bias") or 0) > 0
