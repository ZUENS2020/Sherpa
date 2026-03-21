from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import workflow_common


def test_load_opencode_prompt_templates_parses_markdown_templates():
    workflow_common.load_opencode_prompt_templates.cache_clear()
    templates = workflow_common.load_opencode_prompt_templates()

    assert "plan_with_hint" in templates
    assert "plan_repair_build_with_hint" in templates
    assert "plan_repair_crash_with_hint" in templates
    assert "synthesize_with_hint" in templates
    assert "synthesize_repair_build_with_hint" in templates
    assert "synthesize_repair_crash_with_hint" in templates
    assert "synthesize_complete_scaffold" in templates
    assert "plan_fix_targets_schema" in templates
    assert "./done" in templates["plan_with_hint"]
    assert "./done" in templates["synthesize_with_hint"]
    assert "./done" in templates["synthesize_complete_scaffold"]
    assert "TEMPLATE:" not in templates["plan_with_hint"]


def test_render_opencode_prompt_replaces_placeholders():
    workflow_common.load_opencode_prompt_templates.cache_clear()
    out = workflow_common.render_opencode_prompt("plan_with_hint", hint="hello-hint")
    assert "hello-hint" in out
    assert "{{hint}}" not in out


def test_plan_prompt_references_stage_skill_and_schema_contract():
    workflow_common.load_opencode_prompt_templates.cache_clear()
    out = workflow_common.render_opencode_prompt("plan_with_hint", hint="schema-check")

    assert "Follow the STAGE SKILL loaded by the runner as primary instructions." in out
    assert "strict-schema `fuzz/targets.json`" in out
    assert "`name`, `api`, `lang`, `target_type`, `seed_profile`" in out
    assert "Keep runtime-viable/public entrypoints first." in out


def test_repair_plan_prompts_are_split_by_origin() -> None:
    workflow_common.load_opencode_prompt_templates.cache_clear()
    build_repair = workflow_common.render_opencode_prompt("plan_repair_build_with_hint", hint="build-diag")
    crash_repair = workflow_common.render_opencode_prompt("plan_repair_crash_with_hint", hint="crash-diag")

    assert "build-stage failure" in build_repair
    assert "crash/repro stage failure" in crash_repair
    assert "build-diag" in build_repair
    assert "crash-diag" in crash_repair


def test_synthesize_prompts_keep_stage_contracts_but_are_short():
    workflow_common.load_opencode_prompt_templates.cache_clear()
    synth = workflow_common.render_opencode_prompt("synthesize_with_hint", hint="runtime-first")
    scaffold = workflow_common.render_opencode_prompt("synthesize_complete_scaffold", missing_items="- fuzz/build.py")

    assert "Follow the STAGE SKILL loaded by the runner as primary instructions." in synth
    assert "`fuzz/repo_understanding.json`" in synth
    assert "`fuzz/build_strategy.json`" in synth
    assert "`fuzz/build_runtime_facts.json`" in synth
    assert "DEFAULT_CMAKE_ARGS" in synth
    assert "-DENABLE_TEST=OFF" in synth
    assert "-DENABLE_INSTALL=OFF" in synth
    assert "read-only exploration commands are allowed" in synth.lower()
    assert "Do NOT run build/execute commands." in synth
    assert "Prefer public/stable repository APIs for harness logic." in synth

    assert "Follow the STAGE SKILL loaded by the runner as primary instructions." in scaffold
    assert "partial scaffold" in scaffold
    assert "fuzz/build_runtime_facts.json" in scaffold
    assert "missing items" in scaffold.lower()

    synth_build_repair = workflow_common.render_opencode_prompt("synthesize_repair_build_with_hint", hint="build-fail")
    synth_crash_repair = workflow_common.render_opencode_prompt("synthesize_repair_crash_with_hint", hint="crash-fail")
    assert "after a build-stage failure" in synth_build_repair
    assert "after a crash/repro-stage failure" in synth_crash_repair
    assert "build-fail" in synth_build_repair
    assert "crash-fail" in synth_crash_repair


def test_global_policy_document_contains_core_rules():
    policy = (
        ROOT
        / "harness_generator"
        / "src"
        / "langchain_agent"
        / "prompts"
        / "opencode_global_policy.md"
    ).read_text(encoding="utf-8")

    assert "Default to minimal linking" in policy
    assert "Do not hardcode a single build artifact path." in policy
    assert "Allowed: read-only inspection commands" in policy
    assert "Forbidden: `name = \"LLVMFuzzerTestOneInput\"`." in policy
    assert "Archive Seed Policy" in policy
    assert "use real repository samples first" in policy
    assert "Keep malformed/truncated archive seeds as a minority" in policy


def test_stage_skills_include_exact_build_template_block():
    skill_root = ROOT / "harness_generator" / "src" / "langchain_agent" / "opencode_skills"
    required_stages = ["synthesize", "fix_build"]
    for stage in required_stages:
        text = (skill_root / stage / "SKILL.md").read_text(encoding="utf-8")
        assert 'DEFAULT_CMAKE_ARGS = ["-DENABLE_TEST=OFF", "-DENABLE_INSTALL=OFF"]' in text
        assert "def find_static_lib(repo_root):" in text
        assert '["find", str(repo_root), "-name", "*.a", "-type", "f"]' in text
        assert "capture_output=True, text=True, timeout=60" in text
        assert 'for p in result.stdout.strip().split("\\n"):' in text
        assert 'if "test" not in p.name.lower() and p.exists():' in text


def test_synthesize_skills_require_harness_output_and_self_check():
    skill_root = ROOT / "harness_generator" / "src" / "langchain_agent" / "opencode_skills"
    synth = (skill_root / "synthesize" / "SKILL.md").read_text(encoding="utf-8")
    complete = (skill_root / "synthesize_complete_scaffold" / "SKILL.md").read_text(encoding="utf-8")

    assert "harness-first contract" in synth
    assert "harness file count is >= 1" in synth
    assert "Harness file:` points to an existing harness file under `fuzz/`." in synth
    assert "chosen_target_api" in synth
    assert "chosen_target_reason" in synth
    assert "fuzzer_entry_strategy" in synth
    assert "evidence" in synth
    assert "minimal valid template" in synth
    assert "must be a target API identifier" in synth
    assert "forbidden examples: `fuzz/xxx_fuzz.cc`" in synth
    assert "build_system` must not be `unknown`" in synth
    assert "evidence` must be a non-empty string array" in synth
    assert "never use shell substitutions like `$(nproc)`" in synth
    assert '["-j", str(os.cpu_count() or 1)]' in synth
    assert "def build_fuzzers():" in synth
    assert "static_lib = find_static_lib(BUILD_DIR) or find_static_lib(REPO_ROOT)" in synth
    assert '"clang++"' in synth
    assert "-fsanitize=address,undefined,fuzzer" in synth
    assert '"cmake", "-S", str(REPO_ROOT), "-B", str(BUILD_DIR)' in synth

    assert "if harness source is missing" in complete
    assert "if harness was missing before this step, harness exists after this step." in complete
    assert "repo_understanding.json" in complete
    assert "repair it in place" in complete
    assert "minimal valid shape example" in complete
    assert "semantically invalid" in complete
    assert "not a harness file path" in complete
    assert 'build_system.lower() != "unknown"' in complete
    assert "`$(nproc)`" in complete
    assert '["-j", str(os.cpu_count() or 1)]' in complete


def test_other_stage_skills_include_runtime_contract_clauses():
    skill_root = ROOT / "harness_generator" / "src" / "langchain_agent" / "opencode_skills"
    plan = (skill_root / "plan" / "SKILL.md").read_text(encoding="utf-8")
    plan_fix = (skill_root / "plan_fix_targets_schema" / "SKILL.md").read_text(encoding="utf-8")
    plan_repair_build = (skill_root / "plan_repair_build" / "SKILL.md").read_text(encoding="utf-8")
    plan_repair_crash = (skill_root / "plan_repair_crash" / "SKILL.md").read_text(encoding="utf-8")
    synth_repair_build = (skill_root / "synthesize_repair_build" / "SKILL.md").read_text(encoding="utf-8")
    synth_repair_crash = (skill_root / "synthesize_repair_crash" / "SKILL.md").read_text(encoding="utf-8")

    assert "forbidden: `name = LLVMFuzzerTestOneInput`" in plan
    assert "`api` must describe a target API identifier" in plan
    assert "forbidden: `name = LLVMFuzzerTestOneInput`" in plan_fix
    assert "semantic reminder: do not rewrite `api` to harness file paths" in plan_fix
    assert "build-stage failure" in plan_repair_build
    assert "crash/repro-stage failure" in plan_repair_crash
    assert "build-failure-driven" in synth_repair_build
    assert "crash/repro evidence" in synth_repair_crash
