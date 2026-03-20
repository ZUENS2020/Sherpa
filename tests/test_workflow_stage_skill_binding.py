from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "harness_generator" / "src" / "langchain_agent" / "workflow_graph.py"
LEGACY = ROOT / "harness_generator" / "src" / "fuzz_unharnessed_repo.py"


def test_workflow_graph_binds_stage_skills_for_all_opencode_calls() -> None:
    text = WF.read_text(encoding="utf-8")
    expected = [
        'stage_skill="plan"',
        'stage_skill="plan_fix_targets_schema"',
        'stage_skill="synthesize"',
        'stage_skill="synthesize_complete_scaffold"',
        'stage_skill="fix_build"',
        'stage_skill=("fix_crash_harness_error" if harness_error else "fix_crash_upstream_bug")',
    ]
    for token in expected:
        assert token in text


def test_main_workflow_stage_skills_exist() -> None:
    root = ROOT / "harness_generator" / "src" / "langchain_agent" / "opencode_skills"
    required = [
        "plan",
        "plan_fix_targets_schema",
        "synthesize",
        "synthesize_complete_scaffold",
        "fix_build",
        "fix_crash_harness_error",
        "fix_crash_upstream_bug",
        "seed_generation",
    ]
    for stage in required:
        skill = root / stage / "SKILL.md"
        assert skill.is_file(), f"missing {skill}"


def test_legacy_passes_also_bind_stage_skills_for_plan_and_synthesize() -> None:
    text = LEGACY.read_text(encoding="utf-8")
    assert 'stage_skill="plan"' in text
    assert 'stage_skill="synthesize"' in text
    assert 'stage_skill="seed_generation"' in text


def test_workflow_attempts_forced_harness_repair_before_missing_harness_error() -> None:
    text = WF.read_text(encoding="utf-8")
    repair_hint = "synthesize: harness missing after grace wait; running forced harness repair"
    error_hint = "synthesize incomplete: missing harness source under fuzz/"
    repair_pos = text.find(repair_hint)
    error_pos = text.find(error_hint)
    assert repair_pos != -1
    assert error_pos != -1
    assert repair_pos < error_pos


def test_workflow_plan_and_synthesize_use_group_feedback_context() -> None:
    text = WF.read_text(encoding="utf-8")
    assert '_collect_feedback_for_group(gen.repo_root, "planning_synth", limit=3)' in text
    assert "_write_stage_feedback(" in text
    assert 'stage="plan"' in text
    assert 'stage="synthesize"' in text
