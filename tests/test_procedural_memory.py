from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "harness_generator" / "src" / "langchain_agent"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import procedural_memory as pm  # noqa: E402


@pytest.fixture()
def store(tmp_path, monkeypatch) -> Path:
    p = tmp_path / "procedural_memory.json"
    monkeypatch.setenv("SHERPA_PROCEDURAL_MEMORY_PATH", str(p))
    monkeypatch.setenv("SHERPA_PROCEDURAL_MEMORY", "1")
    monkeypatch.delenv("SHERPA_PROCEDURAL_MEMORY_READONLY", raising=False)
    monkeypatch.setenv("SHERPA_PROCEDURAL_MEMORY_MIN_OCCURRENCE", "2")
    return p


def _record(path, **over):
    base = dict(
        stage="synthesize",
        error_class="vcpkg_overdeclare",
        scope="library_class:makefile-selfcontained",
        signature="sig",
        lesson="don't declare vcpkg for self-contained libs",
        job_id="job1",
        path=path,
    )
    base.update(over)
    return pm.record_lesson(**base)


def test_enabled_by_default(monkeypatch):
    # Phase 3: memory is ON unless explicitly disabled.
    monkeypatch.delenv("SHERPA_PROCEDURAL_MEMORY", raising=False)
    assert pm.memory_enabled() is True


def test_disabled_when_off(tmp_path, monkeypatch):
    monkeypatch.setenv("SHERPA_PROCEDURAL_MEMORY", "0")
    monkeypatch.setenv("SHERPA_PROCEDURAL_MEMORY_PATH", str(tmp_path / "m.json"))
    assert _record(tmp_path / "m.json") is None
    assert pm.retrieve(stage="synthesize") == []


def test_min_occurrence_gate_then_active(store):
    # first occurrence: stored but not yet active (occurrence 1 < 2)
    _record(store)
    assert pm.retrieve(stage="synthesize", library_class="makefile-selfcontained") == []
    # second occurrence: now active and retrievable
    e = _record(store)
    assert e["occurrence_count"] == 2
    hits = pm.retrieve(stage="synthesize", library_class="makefile-selfcontained")
    assert len(hits) == 1
    assert hits[0]["error_class"] == "vcpkg_overdeclare"
    assert hits[0]["confidence"] > e["confidence"] - 0.001


def test_readonly_suppresses_writes_but_allows_reads(store, monkeypatch):
    _record(store)
    _record(store)  # active now
    monkeypatch.setenv("SHERPA_PROCEDURAL_MEMORY_READONLY", "1")
    assert _record(store, job_id="job2") is None  # write suppressed
    assert len(pm.retrieve(stage="synthesize", library_class="makefile-selfcontained")) == 1


def test_decay_excludes_stale(store, monkeypatch):
    _record(store)
    _record(store)
    # force last_seen far in the past
    doc = pm.load_store(store)
    for e in doc["lessons"].values():
        e["last_seen"] = 1
    pm._atomic_write(store, doc)
    assert pm.retrieve(stage="synthesize", library_class="makefile-selfcontained") == []


def test_scope_isolation(store):
    _record(store, scope="library_class:cmake")
    _record(store, scope="library_class:cmake")
    # querying a different library class with no global lesson -> still returns
    # because non-global lessons surface when no exact match? guard: only global
    # or matching scope. Here scope is cmake, query makefile -> should NOT match.
    hits = pm.retrieve(stage="synthesize", library_class="makefile-selfcontained")
    assert all(h["scope"] != "library_class:cmake" for h in hits)
    # querying cmake matches
    hits2 = pm.retrieve(stage="synthesize", library_class="cmake")
    assert len(hits2) == 1


def test_global_scope_always_matches(store):
    _record(store, scope="global")
    _record(store, scope="global")
    hits = pm.retrieve(stage="synthesize", library_class="anything")
    assert len(hits) == 1


def test_note_success_decays_confidence(store):
    _record(store)
    _record(store)
    before = pm.retrieve(stage="synthesize", library_class="makefile-selfcontained")[0]["confidence"]
    pm.note_success("synthesize", "library_class:makefile-selfcontained", path=store)
    doc = pm.load_store(store)
    after = list(doc["lessons"].values())[0]["confidence"]
    assert after == pytest.approx(max(0.0, before - 0.2), abs=1e-6)


def test_classify_vcpkg_overdeclare(store):
    res = pm.classify_stage_failure(
        stage="synthesize",
        error_code="k8s_job_failed",
        error_kind="unknown_error",
        diagnostics="[error] (native/deps) vcpkg unavailable while required ports are declared",
        system_packages_nonempty=True,
        library_class="makefile-selfcontained",
    )
    assert res and res["error_class"] == "vcpkg_overdeclare"
    assert res["scope"] == "library_class:makefile-selfcontained"
    # round-trips into record_lesson
    e = pm.record_lesson(job_id="j", path=store, **res)
    assert e["error_class"] == "vcpkg_overdeclare"


def test_classify_non_public_selection(store):
    res = pm.classify_stage_failure(
        stage="build",
        error_code="non_public_api_usage",
        api_surface_exception_used=True,
        library_class="cmake",
    )
    assert res and res["error_class"] == "non_public_api_selection"


def test_classify_synthesize_incomplete_repo_understanding(store):
    res = pm.classify_stage_failure(
        stage="synthesize",
        error_code="",
        error_kind="generic_failure",
        diagnostics="synthesize incomplete: repo understanding missing `chosen_target_api`",
        library_class="makefile-selfcontained",
    )
    assert res and res["error_class"] == "synthesize_incomplete_repo_understanding"
    assert "repo_understanding.json" in res["lesson"]
    e = pm.record_lesson(job_id="j", path=store, **res)
    assert e["error_class"] == "synthesize_incomplete_repo_understanding"


def test_classify_unknown_returns_none(store):
    assert (
        pm.classify_stage_failure(stage="build", error_code="cxx_for_c_source_mismatch")
        is None
    )


def test_render_block(store):
    _record(store)
    _record(store)
    hits = pm.retrieve(stage="synthesize", library_class="makefile-selfcontained")
    block = pm.render_lessons_block(hits)
    assert "Known pitfalls" in block
    assert "vcpkg" in block.lower()
    assert pm.render_lessons_block([]) == ""
