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
    assert "The final target must be the actual external/library API" in synth
    assert "`Harness file: ...`" in synth
    assert "local helper, checker, wrapper utility" in synth
    assert "Do not optimize for early artifact output" in synth
    assert "First create minimal skeleton artifacts immediately" not in synth
    assert "You may invoke a repository-provided fuzz target only if" in synth
    assert "fuzz/observed_target.json" in scaffold
    assert "fuzz/repo_understanding.json" in scaffold
    assert "never reference missing scaffold files" in scaffold
