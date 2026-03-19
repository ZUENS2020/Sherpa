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
    assert "synthesize_with_hint" in templates
    assert "synthesize_complete_scaffold" in templates
    assert "fix_build_execute" in templates
    assert "./done" in templates["plan_with_hint"]
    assert "./done" in templates["synthesize_with_hint"]
    assert "./done" in templates["synthesize_complete_scaffold"]
    assert "./done" in templates["fix_build_execute"]
    assert "./done" in templates["fix_crash_harness_error"]
    assert "./done" in templates["fix_crash_upstream_bug"]
    assert "./done" in templates["plan_fix_targets_schema"]
    assert "TEMPLATE:" not in templates["plan_with_hint"]


def test_render_opencode_prompt_replaces_placeholders():
    workflow_common.load_opencode_prompt_templates.cache_clear()
    out = workflow_common.render_opencode_prompt("plan_with_hint", hint="hello-hint")
    assert "hello-hint" in out
    assert "{{hint}}" not in out


def test_plan_prompt_hardens_targets_schema_on_first_attempt():
    workflow_common.load_opencode_prompt_templates.cache_clear()
    out = workflow_common.render_opencode_prompt("plan_with_hint", hint="schema-check")

    assert "MUST be plain JSON" in out
    assert "MUST be a JSON array" in out
    assert "Never wrap the array inside another object" in out
    assert '"lang": "c-cpp"' in out
    assert '"target_type": "parser"' in out
    assert "Put the best runtime target first" in out
    assert "Do not put compile-only, constexpr-only, detail/helper" in out


def test_synthesize_prompts_require_observed_target_alignment():
    workflow_common.load_opencode_prompt_templates.cache_clear()
    synth = workflow_common.render_opencode_prompt("synthesize_with_hint", hint="runtime-first")
    scaffold = workflow_common.render_opencode_prompt("synthesize_complete_scaffold", missing_items="- fuzz/build.py")

    assert "fuzz/observed_target.json" in synth
    assert "fuzz/repo_understanding.json" in synth
    assert "fuzz/build_runtime_facts.json" in synth
    assert "must agree on one final external/library API" in synth
    assert "`Harness file: ...`" in synth
    assert "local helper/checker/wrapper" in synth
    assert "Do not optimize for early artifact output" in synth
    assert "MANDATORY OUTPUT CHECKLIST" in synth
    assert "If blocked, still create minimal valid versions" in synth
    assert "First create minimal skeleton artifacts immediately" not in synth
    assert "You may use a repository-provided fuzz target only when" in synth
    assert "drift only when repository facts prove it is not directly fuzzable" in synth
    assert "`rejected_targets`" in synth
    assert "near-miss candidates and why they were rejected" in synth
    assert "enumerate at least 3 concrete seed families" in synth
    assert "actual corpus example or planned corpus file pattern" in synth
    assert "do not use `-lfuzzer`" in synth
    assert "input-size guard" in synth
    assert "if (size > 8192) return 0;" in synth
    assert "FIRST-PASS QUALITY GATE" in synth
    assert "you MUST declare matching vcpkg ports" in synth
    assert "Never keep contradictory \"feature disabled but still linked\" states" in synth
    assert "do not hardcode a single static library path" in synth
    assert "subprocess.run([\"find\", str(REPO_ROOT), \"-name\", \"*.a\", \"-type\", \"f\"], ...)" in synth
    assert "find_static_lib(...)" in synth
    assert "DEFAULT_CMAKE_ARGS = [" in synth
    assert "\"-DENABLE_TEST=OFF\"" in synth
    assert "\"-DENABLE_INSTALL=OFF\"" in synth
    assert "fuzz/observed_target.json" in scaffold
    assert "fuzz/repo_understanding.json" in scaffold
    assert "fuzz/build_runtime_facts.json" in scaffold
    assert "MANDATORY OUTPUT CHECKLIST" in scaffold
    assert "If `fuzz/build_strategy.json` is missing" in scaffold
    assert "never reference missing scaffold files" in scaffold
    assert "record the rejected original target and repository-grounded reason" in scaffold
    assert "avoid high-level repository summaries with no execution consequences" in scaffold


def test_fix_build_prompt_prefers_target_alignment_and_concrete_seed_repairs():
    workflow_common.load_opencode_prompt_templates.cache_clear()
    out = workflow_common.render_opencode_prompt(
        "fix_build_execute",
        build_log_file="fuzz/build_full.log",
        codex_hint="tighten scaffold",
    )

    assert "If the selected target and observed target disagree" in out
    assert "repair that mismatch before incremental build tweaks" in out
    assert "rejected alternatives" in out
    assert "concrete seed families tied to target semantics" in out
    assert "If `fuzz/build_runtime_facts.json` is missing or weak" in out
    assert "forbidden link flags" in out
    assert "input-size guard" in out
    assert "cannot find -l..." in out
    assert "MUST create or update `fuzz/system_packages.txt` in the same attempt" in out
    assert "First repair pass must be build-ready" in out
    assert "runtime command discovery" in out
