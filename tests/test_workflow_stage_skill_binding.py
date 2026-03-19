from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "harness_generator" / "src" / "langchain_agent" / "workflow_graph.py"


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
    ]
    for stage in required:
        skill = root / stage / "SKILL.md"
        assert skill.is_file(), f"missing {skill}"
