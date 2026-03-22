from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "harness_generator" / "src" / "langchain_agent" / "workflow_graph.py"
LEGACY = ROOT / "harness_generator" / "src" / "fuzz_unharnessed_repo.py"


def test_workflow_graph_binds_stage_skills_for_all_opencode_calls() -> None:
    text = WF.read_text(encoding="utf-8")
    expected = [
        'stage_skill="plan_fix_targets_schema"',
        'stage_skill="synthesize_complete_scaffold"',
        'stage_skill="crash_triage"',
        'plan_stage_skill = "plan"',
        'synth_stage_skill = "synthesize"',
        'plan_stage_skill = "plan_repair_build"',
        'plan_stage_skill = "plan_repair_crash"',
        'synth_stage_skill = "synthesize_repair_build"',
        'synth_stage_skill = "synthesize_repair_crash"',
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
        "plan_repair_build",
        "plan_repair_crash",
        "synthesize_repair_build",
        "synthesize_repair_crash",
        "seed_generation",
        "crash_triage",
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


def test_workflow_build_failures_route_to_plan_without_fix_nodes() -> None:
    text = WF.read_text(encoding="utf-8")
    assert 'return "plan"' in text
    assert '"fix_build": "fix_build"' not in text
    assert 'graph.add_node("fix_build", _node_fix_build)' not in text
    assert 'graph.add_node("fix_crash", _node_fix_crash)' not in text


def test_workflow_synthesize_uses_configurable_opencode_attempts() -> None:
    text = WF.read_text(encoding="utf-8")
    assert "def _synthesize_opencode_attempts() -> int:" in text
    assert "max_attempts=_synthesize_opencode_attempts()" in text
