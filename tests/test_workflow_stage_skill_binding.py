from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "harness_generator" / "src" / "langchain_agent" / "workflow_graph.py"
LEGACY = ROOT / "harness_generator" / "src" / "fuzz_unharnessed_repo.py"


def test_workflow_graph_binds_stage_skills_for_all_opencode_calls() -> None:
    text = WF.read_text(encoding="utf-8")
    expected = [
        'stage_skill="analysis"',
        'stage_skill="plan_fix_targets_schema"',
        'stage_skill="synthesize_complete_scaffold"',
        'stage_skill="crash_triage"',
        'stage_skill="crash_analysis"',
        'plan_stage_skill = "plan"',
        'synth_stage_skill = "synthesize"',
        'plan_stage_skill = "plan_repair_build"',
        'plan_stage_skill = "plan_repair_crash"',
        'plan_stage_skill = "plan_repair_coverage"',
        'plan_stage_skill = "plan_repair_fix_harness"',
        'synth_stage_skill = "synthesize_repair_build"',
        'synth_stage_skill = "synthesize_repair_crash"',
        'synth_stage_skill = "synthesize_repair_coverage"',
        'synth_stage_skill = "synthesize_repair_fix_harness"',
        'plan_template_name = "plan_repair_coverage_with_hint"',
        'plan_template_name = "plan_repair_fix_harness_with_hint"',
        'synth_template_name = "synthesize_repair_coverage_with_hint"',
        'synth_template_name = "synthesize_repair_fix_harness_with_hint"',
        'stage_skill="improve_harness_in_place"',
    ]
    for token in expected:
        assert token in text


def test_main_workflow_stage_skills_exist() -> None:
    root = ROOT / "harness_generator" / "src" / "langchain_agent" / "opencode_skills"
    required = [
        "plan",
        "analysis",
        "plan_fix_targets_schema",
        "synthesize",
        "synthesize_complete_scaffold",
        "plan_repair_build",
        "plan_repair_crash",
        "plan_repair_coverage",
        "plan_repair_fix_harness",
        "synthesize_repair_build",
        "synthesize_repair_crash",
        "synthesize_repair_coverage",
        "synthesize_repair_fix_harness",
        "improve_harness_in_place",
        "seed_generation",
        "crash_triage",
        "crash_analysis",
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


def test_workflow_build_and_crash_failures_route_to_plan_repair_loop() -> None:
    text = WF.read_text(encoding="utf-8")
    assert 'graph.add_node("fix_build", _node_fix_build)' not in text
    assert 'graph.add_node("fix_crash", _node_fix_crash)' not in text
    assert '{"run": "run", "plan": "plan", "stop": END}' in text
    assert '{"re-run": "re-run", "plan": "plan", "stop": END}' in text
    assert '{"crash-analysis": "crash-analysis", "plan": "plan", "stop": END}' in text


def test_workflow_fix_harness_node_is_legacy_only() -> None:
    text = WF.read_text(encoding="utf-8")
    assert 'graph.add_node("fix-harness", _node_fix_harness_after_run)' not in text
    assert '{"fix-harness": "fix-harness"' not in text


def test_workflow_synthesize_uses_configurable_opencode_attempts() -> None:
    text = WF.read_text(encoding="utf-8")
    assert "def _synthesize_opencode_attempts() -> int:" in text
    assert "max_attempts=_synthesize_opencode_attempts()" in text


def test_workflow_marks_coverage_replan_as_coverage_repair_origin() -> None:
    text = WF.read_text(encoding="utf-8")
    assert '"repair_origin_stage": "coverage" if replan_required' in text
    assert '"repair_error_kind": "coverage_plateau" if replan_required' in text
    assert '"repair_error_code": "coverage_replan_required" if replan_required' in text
