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


def test_collect_antlr_assist_context_extracts_symbols(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "demo.c").write_text(
        "int parse_zip(const char* s) { return s ? 1 : 0; }\n"
        "int helper(int x) { return x + 1; }\n",
        encoding="utf-8",
    )
    (src / "Mini.g4").write_text(
        "grammar Mini;\n"
        "startrule: expr EOF;\n"
        "expr: INT;\n"
        "INT: [0-9]+;\n"
        "WS: [ \\t\\r\\n]+ -> skip;\n",
        encoding="utf-8",
    )

    doc = workflow_graph._collect_antlr_assist_context(tmp_path)
    names = {str(x.get("name") or "") for x in (doc.get("candidate_functions") or [])}
    parser_rules = set(doc.get("parser_rules") or [])
    grammar_files = set(doc.get("grammar_files") or [])

    assert "parse_zip" in names
    assert "startrule" in parser_rules
    assert "src/Mini.g4" in grammar_files


def test_node_plan_writes_antlr_context_and_hint(tmp_path: Path, monkeypatch):
    class _Patcher:
        def run_codex_command(self, _prompt: str, **kwargs):
            _pass_plan_targets(timeout=int(kwargs.get("timeout") or 1))
            return None

    def _pass_plan_targets(*, timeout: int) -> None:
        fuzz_dir = tmp_path / "fuzz"
        fuzz_dir.mkdir(parents=True, exist_ok=True)
        (fuzz_dir / "PLAN.md").write_text("# plan\n", encoding="utf-8")
        (fuzz_dir / "targets.json").write_text('[{"name":"a","api":"b","lang":"c-cpp","target_type":"parser"}]\n', encoding="utf-8")

    gen = SimpleNamespace(repo_root=tmp_path, _pass_plan_targets=_pass_plan_targets, patcher=_Patcher())
    monkeypatch.setattr(workflow_graph, "_has_codex_key", lambda: True)
    monkeypatch.setattr(workflow_graph, "_make_plan_hint", lambda _repo_root: "base plan hint")
    monkeypatch.setenv("SHERPA_PLAN_STRICT_TARGETS_SCHEMA", "0")

    out = workflow_graph._node_plan({"generator": gen, "codex_hint": ""})
    assert out["last_error"] == ""
    assert "antlr_context_path" in out
    antlr_ctx = Path(str(out.get("antlr_context_path") or ""))
    assert antlr_ctx.is_file()
    assert "antlr_plan_context.json" in str(out.get("codex_hint") or "")


def test_node_synthesize_injects_antlr_context_into_additional_context(tmp_path: Path, monkeypatch):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "PLAN.md").write_text("# plan\n", encoding="utf-8")
    (fuzz_dir / "targets.json").write_text('[{"name":"a","api":"b","lang":"c-cpp","target_type":"parser"}]\n', encoding="utf-8")
    antlr_ctx = fuzz_dir / "antlr_plan_context.json"
    antlr_ctx.write_text('{"entrypoint_candidates":[{"name":"parse_zip"}]}\n', encoding="utf-8")

    captured: dict[str, str] = {}

    class _Patcher:
        def run_codex_command(self, _prompt: str, **kwargs):
            captured["additional_context"] = str(kwargs.get("additional_context") or "")
            # Produce minimal synth outputs to satisfy guard.
            (fuzz_dir / "harness.cc").write_text("int LLVMFuzzerTestOneInput(const unsigned char*, unsigned long){return 0;}\n", encoding="utf-8")
            (fuzz_dir / "build.py").write_text("print('ok')\n", encoding="utf-8")
            return None

    gen = SimpleNamespace(repo_root=tmp_path, patcher=_Patcher(), _pass_synthesize_harness=lambda timeout: None)
    monkeypatch.setattr(workflow_graph, "_has_codex_key", lambda: True)
    monkeypatch.setenv("SHERPA_SYNTHESIZE_GRACE_SEC", "0")

    out = workflow_graph._node_synthesize(
        {
            "generator": gen,
            "codex_hint": "use target hints",
            "antlr_context_path": str(antlr_ctx),
            "antlr_context_summary": "antlr_context_file=fuzz/antlr_plan_context.json",
        }
    )
    assert out["last_error"] == ""
    assert "fuzz/antlr_plan_context.json" in captured.get("additional_context", "")


def test_node_plan_clears_stale_done_before_schema_retry(tmp_path: Path, monkeypatch):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "done").write_text("fuzz/PLAN.md\n", encoding="utf-8")
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "demo.c").write_text(
        "int parse_yaml(const char* s) { return s ? 1 : 0; }\n",
        encoding="utf-8",
    )

    sentinel_seen: list[bool] = []
    call_count = {"n": 0}

    class _Patcher:
        def run_codex_command(self, _prompt: str, **kwargs):
            call_count["n"] += 1
            sentinel_seen.append((tmp_path / "done").exists())
            (fuzz_dir / "PLAN.md").write_text("# plan\n", encoding="utf-8")
            if call_count["n"] == 1:
                (fuzz_dir / "targets.json").write_text('{"targets":[]}\n', encoding="utf-8")
            else:
                (fuzz_dir / "targets.json").write_text(
                    '[{"name":"parse_yaml","api":"parse_yaml","lang":"c-cpp","target_type":"parser"}]\n',
                    encoding="utf-8",
                )
            return None

    gen = SimpleNamespace(repo_root=tmp_path, patcher=_Patcher(), _pass_plan_targets=lambda timeout: None)
    monkeypatch.setattr(workflow_graph, "_has_codex_key", lambda: True)
    monkeypatch.setattr(workflow_graph, "_make_plan_hint", lambda _repo_root: "base plan hint")
    monkeypatch.setenv("SHERPA_PLAN_STRICT_TARGETS_SCHEMA", "1")

    out = workflow_graph._node_plan({"generator": gen, "codex_hint": ""})

    assert out["last_error"] == ""
    assert call_count["n"] == 2
    assert sentinel_seen == [True, False]
    assert out["plan_retry_reason"] == "targets-schema"
    assert out["plan_targets_schema_valid_before_retry"] is False
    assert out["plan_targets_schema_valid_after_retry"] is True
    assert out["plan_used_fallback_targets"] is False


def test_node_plan_uses_deterministic_fallback_after_retry_failure(tmp_path: Path, monkeypatch):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "demo.c").write_text(
        "int parse_yaml_stream(const char* s) { return s ? 1 : 0; }\n",
        encoding="utf-8",
    )

    class _Patcher:
        def __init__(self):
            self.calls = 0

        def run_codex_command(self, _prompt: str, **kwargs):
            self.calls += 1
            (fuzz_dir / "PLAN.md").write_text("# plan\n", encoding="utf-8")
            (fuzz_dir / "targets.json").write_text("{}\n", encoding="utf-8")
            return None

    patcher = _Patcher()
    gen = SimpleNamespace(repo_root=tmp_path, patcher=patcher, _pass_plan_targets=lambda timeout: None)
    monkeypatch.setattr(workflow_graph, "_has_codex_key", lambda: True)
    monkeypatch.setattr(workflow_graph, "_make_plan_hint", lambda _repo_root: "base plan hint")
    monkeypatch.setenv("SHERPA_PLAN_STRICT_TARGETS_SCHEMA", "1")

    out = workflow_graph._node_plan({"generator": gen, "codex_hint": ""})

    assert out["last_error"] == ""
    assert patcher.calls == 2
    assert out["plan_retry_reason"] == "targets-schema"
    assert out["plan_targets_schema_valid_before_retry"] is False
    assert out["plan_targets_schema_valid_after_retry"] is True
    assert out["plan_used_fallback_targets"] is True
    targets = (fuzz_dir / "targets.json").read_text(encoding="utf-8")
    assert "parse_yaml_stream" in targets
