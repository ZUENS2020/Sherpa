from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "harness_generator" / "src" / "langchain_agent" / "workflow_graph.py"
PROMPTS = ROOT / "harness_generator" / "src" / "langchain_agent" / "prompts" / "opencode_prompts.md"
SKILL_ROOT = ROOT / "harness_generator" / "src" / "langchain_agent" / "opencode_skills"


def test_workflow_has_execution_plan_helpers_and_build_gate() -> None:
    text = WF.read_text(encoding="utf-8")
    assert "def _execution_plan_path(repo_root: Path) -> Path:" in text
    assert "def _build_execution_plan_doc(" in text
    assert "def _write_execution_plan_doc(" in text
    assert "def _load_execution_plan_doc(" in text
    assert "def _harness_index_path(repo_root: Path) -> Path:" in text
    assert "def _build_harness_index_doc(" in text
    assert "def _write_harness_index_doc(" in text
    assert "def _validate_execution_plan_harness_consistency(" in text
    assert "partial_build_undercoverage" in text
    assert "SHERPA_EXECUTION_TARGETS_MIN_REQUIRED" in text
    assert "coverage_missing_execution_targets" in text


def test_prompts_and_skills_reference_execution_plan_contract() -> None:
    prompts = PROMPTS.read_text(encoding="utf-8")
    assert "fuzz/execution_plan.json" in prompts
    assert "fuzz/harness_index.json" in prompts
    assert "execution_priority" in prompts
    assert "must_run" in prompts

    plan = (SKILL_ROOT / "plan" / "SKILL.md").read_text(encoding="utf-8")
    synth = (SKILL_ROOT / "synthesize" / "SKILL.md").read_text(encoding="utf-8")
    fix_build = (SKILL_ROOT / "fix_build" / "SKILL.md").read_text(encoding="utf-8")

    assert "fuzz/execution_plan.json" in plan
    assert "min_required_built_targets" in plan
    assert "fuzz/execution_plan.json" in synth
    assert "multiple execution targets" in synth
    assert "fuzz/harness_index.json" in synth
    assert "fuzz/execution_plan.json" in fix_build
