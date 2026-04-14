from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
SRC_DIR = ROOT / "harness_generator" / "src"
for p in (APP_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import workflow_graph


class _NoopPatcher:
    def run_codex_command(self, _prompt: str, **_kwargs) -> None:
        return None


def _broken_render(*_args, **_kwargs) -> str:
    raise ValueError("Single '}' encountered in format string")


def test_crash_triage_degrades_when_prompt_template_is_invalid(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(workflow_graph, "_render_opencode_prompt", _broken_render)
    gen = SimpleNamespace(repo_root=tmp_path, patcher=_NoopPatcher())

    out = workflow_graph._node_crash_triage(
        {
            "generator": gen,
            "codex_hint": "triage",
            "model": "GLM-5",
        }
    )

    assert out["last_step"] == "crash-triage"
    assert out["crash_triage_done"] is True
    assert out["prompt_render_degraded"] is True
    assert "Single '}' encountered in format string" in out["prompt_render_issue"]


def test_analysis_degrades_when_prompt_template_is_invalid(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "fuzz").mkdir(parents=True, exist_ok=True)
    antlr_path = tmp_path / "fuzz" / "antlr_plan_context.json"
    target_path = tmp_path / "fuzz" / "target_analysis.json"
    antlr_path.write_text("{}", encoding="utf-8")
    target_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(workflow_graph, "_render_opencode_prompt", _broken_render)
    monkeypatch.setattr(
        workflow_graph,
        "_prepare_antlr_assist_context",
        lambda _repo_root: (str(antlr_path), "antlr-ok"),
    )
    monkeypatch.setattr(
        workflow_graph,
        "_prepare_target_analysis_context",
        lambda _repo_root: (str(target_path), "target-ok"),
    )
    monkeypatch.setattr(workflow_graph, "_collect_analysis_companion_context", lambda: ({}, ""))
    monkeypatch.setattr(
        workflow_graph,
        "_build_analysis_evidence_index",
        lambda **_kwargs: {"summary": {"evidence_count": 1}},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    gen = SimpleNamespace(repo_root=tmp_path, patcher=_NoopPatcher())

    out = workflow_graph._node_analysis(
        {
            "generator": gen,
            "codex_hint": "analysis",
        }
    )

    assert out["last_step"] == "analysis"
    assert out["analysis_done"] is True
    assert out["prompt_render_degraded"] is True
    assert "Single '}' encountered in format string" in out["prompt_render_issue"]


def test_crash_analysis_degrades_when_prompt_template_is_invalid(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(workflow_graph, "_render_opencode_prompt", _broken_render)
    gen = SimpleNamespace(repo_root=tmp_path, patcher=_NoopPatcher())

    out = workflow_graph._node_crash_analysis(
        {
            "generator": gen,
            "codex_hint": "analysis",
            "model": "GLM-5",
        }
    )

    assert out["last_step"] == "crash-analysis"
    assert out["crash_analysis_done"] is True
    assert out["prompt_render_degraded"] is True
    assert "Single '}' encountered in format string" in out["prompt_render_issue"]


def test_fix_harness_degrades_when_prompt_template_is_invalid(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(workflow_graph, "_render_opencode_prompt", _broken_render)
    gen = SimpleNamespace(repo_root=tmp_path, patcher=_NoopPatcher())

    out = workflow_graph._node_fix_harness_after_run(
        {
            "generator": gen,
            "codex_hint": "repair",
        }
    )

    assert out["last_step"] == "fix-harness"
    assert out["prompt_render_degraded"] is True
    assert "Single '}' encountered in format string" in out["prompt_render_issue"]
