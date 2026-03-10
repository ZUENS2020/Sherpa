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
    assert "fix_build_execute" in templates
    assert "./done" in templates["plan_with_hint"]
    assert "./done" in templates["synthesize_with_hint"]
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
