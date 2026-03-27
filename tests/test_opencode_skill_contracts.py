from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "harness_generator" / "src" / "langchain_agent" / "opencode_skills"


def _load(stage: str) -> str:
    return (SKILL_ROOT / stage / "SKILL.md").read_text(encoding="utf-8")


def test_all_skills_use_frontmatter_and_standard_sections() -> None:
    required_sections = [
        "## What this skill does",
        "## When to use this skill",
        "## Required inputs",
        "## Required outputs",
        "## Workflow",
        "## Command policy",
        "## Done contract",
    ]
    for skill in sorted(SKILL_ROOT.glob("*/SKILL.md")):
        text = skill.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert "\nname:" in text
        assert "\ndescription:" in text
        assert "\ncompatibility:" in text
        for section in required_sections:
            assert section in text, f"{skill} missing section: {section}"


def test_synthesize_contract_keeps_harness_and_build_requirements() -> None:
    synth = _load("synthesize")
    assert "at least one harness source file under `fuzz/`" in synth
    assert "`fuzz/harness_index.json` aligned to `fuzz/execution_plan.json`" in synth
    assert "chosen_target_api" in synth
    assert "build_system" in synth
    assert "fuzzer_entry_strategy" in synth
    assert "DEFAULT_CMAKE_ARGS" in synth
    assert "def find_static_lib(repo_root):" in synth
    assert "use `clang` for `.c` sources" in synth
    assert "use `clang++` for `.cc`, `.cpp`, `.cxx` sources" in synth
    assert "api_surface_exception" in synth


def test_synthesize_complete_scaffold_requires_missing_item_repair() -> None:
    complete = _load("synthesize_complete_scaffold")
    assert "create at least one harness source file" in complete
    assert "before doc/json-only fixes" in complete or "before only-doc/json fixes" in complete
    assert "repo_understanding.json" in complete
    assert "chosen_target_api" in complete
    assert "build_system.lower() != \"unknown\"" in complete or "build_system" in complete
    assert "fuzz/harness_index.json" in complete


def test_plan_and_schema_fix_contracts_keep_target_semantics() -> None:
    plan = _load("plan")
    plan_fix = _load("plan_fix_targets_schema")
    assert "LLVMFuzzerTestOneInput" in plan
    assert "`api` must describe an API identifier" in plan
    assert "fuzz/execution_plan.json" in plan
    assert "min_required_built_targets" in plan
    assert "semantic reminder: do not rewrite `api` to harness paths" in plan_fix.lower()


def test_fix_build_contract_keeps_vcpkg_and_compiler_rules() -> None:
    fix_build = _load("fix_build")
    assert "canonical vcpkg names" in fix_build
    assert "zlib" in fix_build and "bzip2" in fix_build and "liblzma" in fix_build
    assert "previous_failed_attempts" in fix_build
    assert "pure no-op is invalid" in fix_build
    assert "stale `./done` without fresh diff is invalid" in fix_build.lower()
    assert "compile `.c` with `clang`" in fix_build
    assert "compile `.cc/.cpp/.cxx` with `clang++`" in fix_build
    assert "api_surface_exception" in fix_build
    assert "Read and fix <path>[:line]" in fix_build


def test_crash_skills_contracts_are_classification_and_analysis_driven() -> None:
    triage = _load("crash_triage")
    analysis = _load("crash_analysis")
    fix_h = _load("fix_crash_harness_error")
    fix_u = _load("fix_crash_upstream_bug")
    assert "classification-only" in triage
    assert "harness_bug|upstream_bug|inconclusive" in triage
    assert "analysis-only" in analysis
    assert "false_positive|real_bug|unknown" in analysis
    assert "pure no-op is invalid" in fix_h
    assert "pure no-op is invalid" in fix_u
    assert "Read and fix <path>[:line]" in fix_h
    assert "Read and fix <path>[:line]" in fix_u


def test_seed_and_repair_skills_keep_feedback_and_api_surface_constraints() -> None:
    seed = _load("seed_generation")
    plan_repair_build = _load("plan_repair_build")
    plan_repair_crash = _load("plan_repair_crash")
    plan_repair_coverage = _load("plan_repair_coverage")
    synth_repair_build = _load("synthesize_repair_build")
    synth_repair_crash = _load("synthesize_repair_crash")
    synth_repair_coverage = _load("synthesize_repair_coverage")
    improve_in_place = _load("improve_harness_in_place")

    assert "real archive samples first" in seed
    assert "contrib/oss-fuzz/corpus.zip" in seed
    assert "avoid hand-crafted magic-only files" in seed.lower()
    assert "malformed/truncated seeds <= 30%" in seed

    assert "strategy change" in plan_repair_build.lower()
    assert "strategy change" in plan_repair_crash.lower()
    assert "strategy-diff" in plan_repair_coverage.lower()
    assert "api_surface_exception" in plan_repair_build
    assert "api_surface_exception" in plan_repair_crash
    assert "api_surface_exception" in synth_repair_build
    assert "api_surface_exception" in synth_repair_crash
    assert "non_public_api_usage" in synth_repair_build
    assert "non_public_api_usage" in synth_repair_crash
    assert "no target switching" in improve_in_place.lower() or "without switching targets" in improve_in_place.lower()
    assert "coverage" in synth_repair_coverage.lower()
