from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "harness_generator" / "src" / "langchain_agent"
SRC = ROOT / "harness_generator" / "src"
for p in (APP, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import procedural_memory as pm  # noqa: E402
import workflow_graph as wg  # noqa: E402


def test_stage_mapping():
    assert wg._procedural_stage_for_prompt("synthesize") == "synthesize"
    assert wg._procedural_stage_for_prompt("synthesize_repair_build") == "synthesize"
    assert wg._procedural_stage_for_prompt("plan_with_hint") == "plan"
    assert wg._procedural_stage_for_prompt("fix_build") == "build"
    assert wg._procedural_stage_for_prompt("vuln_hunt") == "vuln-hunt"
    assert wg._procedural_stage_for_prompt("analysis_with_hint") == "analysis"
    assert wg._procedural_stage_for_prompt("unknown_thing") == ""


def _seed_lesson(path, monkeypatch):
    monkeypatch.setenv("SHERPA_PROCEDURAL_MEMORY", "1")
    monkeypatch.setenv("SHERPA_PROCEDURAL_MEMORY_PATH", str(path))
    monkeypatch.setenv("SHERPA_PROCEDURAL_MEMORY_MIN_OCCURRENCE", "2")
    pm._STORE_CACHE.clear()
    kw = dict(
        stage="synthesize",
        error_class="vcpkg_overdeclare",
        scope="global",
        signature="sig",
        lesson="don't declare vcpkg for self-contained libs",
        path=path,
    )
    pm.record_lesson(**kw)
    pm.record_lesson(**kw)  # reach min_occurrence -> active


def test_injects_lessons_into_hint(tmp_path, monkeypatch):
    _seed_lesson(tmp_path / "m.json", monkeypatch)
    kwargs = {"hint": "do the synthesize work"}
    wg._inject_procedural_lessons("synthesize_repair_build", kwargs)
    assert "Known pitfalls" in kwargs["hint"]
    assert "vcpkg" in kwargs["hint"].lower()
    assert kwargs["hint"].strip().endswith("do the synthesize work")


def test_no_injection_when_disabled(tmp_path, monkeypatch):
    _seed_lesson(tmp_path / "m.json", monkeypatch)
    monkeypatch.setenv("SHERPA_PROCEDURAL_MEMORY", "0")
    pm._STORE_CACHE.clear()
    kwargs = {"hint": "x"}
    wg._inject_procedural_lessons("synthesize", kwargs)
    assert kwargs["hint"] == "x"


def test_no_injection_without_hint_kwarg(tmp_path, monkeypatch):
    _seed_lesson(tmp_path / "m.json", monkeypatch)
    kwargs = {"other": "y"}
    wg._inject_procedural_lessons("synthesize", kwargs)
    assert "hint" not in kwargs


def test_no_injection_for_unmapped_stage(tmp_path, monkeypatch):
    _seed_lesson(tmp_path / "m.json", monkeypatch)
    kwargs = {"hint": "x"}
    wg._inject_procedural_lessons("some_random_template", kwargs)
    assert kwargs["hint"] == "x"
