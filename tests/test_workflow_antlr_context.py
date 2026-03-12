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


def test_collect_target_analysis_context_prefers_runtime_fmt_targets(tmp_path: Path):
    include_fmt = tmp_path / "include" / "fmt"
    include_fmt.mkdir(parents=True, exist_ok=True)
    (include_fmt / "compile.h").write_text(
        "namespace fmt { namespace compile {\n"
        "int parse_replacement_field_then_tail(const char* s) { return s ? 1 : 0; }\n"
        "} }\n",
        encoding="utf-8",
    )
    (include_fmt / "format.h").write_text(
        "namespace fmt {\n"
        "int println(const char* s) { return s ? 1 : 0; }\n"
        "int format_to(const char* s) { return s ? 1 : 0; }\n"
        "}\n",
        encoding="utf-8",
    )
    test_fuzzing = tmp_path / "test" / "fuzzing"
    test_fuzzing.mkdir(parents=True, exist_ok=True)
    (test_fuzzing / "one-arg.cc").write_text(
        "int fuzz_entry() { return fmt::println(\"{}\"); }\n",
        encoding="utf-8",
    )

    doc = workflow_graph._collect_target_analysis_context(tmp_path)
    recommended = list(doc.get("recommended_targets") or [])
    names = [str(item.get("name") or "") for item in recommended[:4]]

    assert "println" in names[:2] or "format_to" in names[:2]
    assert recommended[0]["runtime_viability"] in {"high", "medium"}
    assert "selection_rationale" in recommended[0]
    assert "runtime_replacement_candidates" in recommended[0]


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
        (fuzz_dir / "targets.json").write_text(
            '[{"name":"a","api":"b","lang":"c-cpp","target_type":"parser","seed_profile":"parser-structure"}]\n',
            encoding="utf-8",
        )

    gen = SimpleNamespace(repo_root=tmp_path, _pass_plan_targets=_pass_plan_targets, patcher=_Patcher())
    monkeypatch.setattr(workflow_graph, "_has_codex_key", lambda: True)
    monkeypatch.setattr(workflow_graph, "_make_plan_hint", lambda _repo_root: "base plan hint")
    monkeypatch.setenv("SHERPA_PLAN_STRICT_TARGETS_SCHEMA", "0")

    out = workflow_graph._node_plan({"generator": gen, "codex_hint": ""})
    assert out["last_error"] == ""
    assert "antlr_context_path" in out
    antlr_ctx = Path(str(out.get("antlr_context_path") or ""))
    assert antlr_ctx.is_file()
    selected_targets = tmp_path / "fuzz" / "selected_targets.json"
    assert selected_targets.is_file()
    assert "antlr_plan_context.json" in str(out.get("codex_hint") or "")


def test_node_synthesize_injects_antlr_context_into_additional_context(tmp_path: Path, monkeypatch):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "PLAN.md").write_text("# plan\n", encoding="utf-8")
    (fuzz_dir / "targets.json").write_text(
        '[{"name":"a","api":"b","lang":"c-cpp","target_type":"parser","seed_profile":"parser-structure"}]\n',
        encoding="utf-8",
    )
    antlr_ctx = fuzz_dir / "antlr_plan_context.json"
    antlr_ctx.write_text('{"entrypoint_candidates":[{"name":"parse_zip"}]}\n', encoding="utf-8")
    target_ctx = fuzz_dir / "target_analysis.json"
    target_ctx.write_text('{"recommended_targets":[{"name":"a","seed_profile":"parser-structure"}]}\n', encoding="utf-8")
    selected_targets = fuzz_dir / "selected_targets.json"
    selected_targets.write_text(
        '[{"target_name":"a","api":"a","target_type":"parser","seed_profile":"parser-structure","seed_families_required":["document_markers"],"seed_families_optional":[]}]',
        encoding="utf-8",
    )

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
            "target_analysis_path": str(target_ctx),
            "target_analysis_summary": "target_analysis_file=fuzz/target_analysis.json",
            "selected_targets_path": str(selected_targets),
        }
    )
    assert out["last_error"] == ""
    assert "fuzz/antlr_plan_context.json" in captured.get("additional_context", "")
    assert "fuzz/target_analysis.json" in captured.get("additional_context", "")
    assert "fuzz/selected_targets.json" in captured.get("additional_context", "")


def test_node_synthesize_accepts_soft_target_drift_and_records_it(tmp_path: Path, monkeypatch):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "PLAN.md").write_text("# plan\n", encoding="utf-8")
    (fuzz_dir / "targets.json").write_text(
        '[{"name":"yaml_parser_parse","api":"yaml_parser_parse","lang":"c-cpp","target_type":"parser","seed_profile":"parser-structure"}]\n',
        encoding="utf-8",
    )
    selected_targets = fuzz_dir / "selected_targets.json"
    selected_targets.write_text(
        '[{"target_name":"yaml_parser_parse","api":"yaml_parser_parse","target_type":"parser","seed_profile":"parser-structure","seed_families_required":["document_markers"],"seed_families_optional":[]}]',
        encoding="utf-8",
    )

    calls = {"n": 0}
    captured: dict[str, str] = {}

    class _Patcher:
        def run_codex_command(self, _prompt: str, **kwargs):
            calls["n"] += 1
            captured["prompt"] = _prompt
            (fuzz_dir / "yaml_parser_fuzz.cc").write_text(
                "extern \"C\" int LLVMFuzzerTestOneInput(const unsigned char* data, unsigned long size) { return yaml_parser_load_document(0, 0); }\n",
                encoding="utf-8",
            )
            (fuzz_dir / "build.py").write_text("print('ok')\n", encoding="utf-8")
            (fuzz_dir / "README.md").write_text(
                "# fuzz\n\n"
                "Selected target: yaml_parser_parse\n"
                "Final target: yaml_parser_load_document\n"
                "Technical reason: runtime parser entrypoint is deeper.\n"
                "Relation: final target is a runtime replacement for the selected parser API.\n",
                encoding="utf-8",
            )
            return None

    gen = SimpleNamespace(repo_root=tmp_path, patcher=_Patcher(), _pass_synthesize_harness=lambda timeout: None)
    monkeypatch.setattr(workflow_graph, "_has_codex_key", lambda: True)
    monkeypatch.setenv("SHERPA_SYNTHESIZE_GRACE_SEC", "0")
    out = workflow_graph._node_synthesize(
        {
            "generator": gen,
            "codex_hint": "keep target",
            "selected_targets_path": str(selected_targets),
            "selected_target_api": "yaml_parser_parse",
        }
    )
    assert out["last_error"] == ""
    assert calls["n"] == 1
    assert out["synthesize_target_drifted"] is True
    assert out["synthesize_selected_target_api"] == "yaml_parser_parse"
    assert out["synthesize_observed_target_api"] == "yaml_parser_load_document"
    assert out["synthesize_target_relation"].startswith("final target")
    assert "runtime-executable replacement target" in captured["prompt"]


def test_node_synthesize_repairs_readme_for_target_drift(tmp_path: Path, monkeypatch):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "PLAN.md").write_text("# plan\n", encoding="utf-8")
    (fuzz_dir / "targets.json").write_text(
        '[{"name":"parse_replacement_field_then_tail","api":"parse_replacement_field_then_tail","lang":"c-cpp","target_type":"parser","seed_profile":"parser-format"}]\n',
        encoding="utf-8",
    )
    (fuzz_dir / "selected_targets.json").write_text(
        '[{"target_name":"parse_replacement_field_then_tail","api":"parse_replacement_field_then_tail","target_type":"parser","seed_profile":"parser-format","runtime_viability":"low","seed_families_required":["replacement_fields"],"seed_families_optional":[]}]',
        encoding="utf-8",
    )
    prompts: list[str] = []

    class _Patcher:
        def run_codex_command(self, prompt: str, **kwargs):
            prompts.append(prompt)
            (fuzz_dir / "println_fuzz.cc").write_text(
                "extern \"C\" int LLVMFuzzerTestOneInput(const unsigned char* data, unsigned long size) { return fmt::println(\"{}\"); }\n",
                encoding="utf-8",
            )
            (fuzz_dir / "build.py").write_text("print('ok')\n", encoding="utf-8")
            if len(prompts) == 1:
                (fuzz_dir / "README.md").write_text("# fuzz\n", encoding="utf-8")
            else:
                (fuzz_dir / "README.md").write_text(
                    "# fuzz\n\n"
                    "Selected target: parse_replacement_field_then_tail\n"
                    "Final target: fmt::println\n"
                    "Technical reason: selected target is not a practical runtime entrypoint.\n"
                    "Relation: fmt::println exercises the same formatting pipeline from a runtime API.\n",
                    encoding="utf-8",
                )
            return None

    gen = SimpleNamespace(repo_root=tmp_path, patcher=_Patcher(), _pass_synthesize_harness=lambda timeout: None)
    monkeypatch.setattr(workflow_graph, "_has_codex_key", lambda: True)
    monkeypatch.setenv("SHERPA_SYNTHESIZE_GRACE_SEC", "0")
    out = workflow_graph._node_synthesize(
        {
            "generator": gen,
            "codex_hint": "prefer runtime target",
            "selected_targets_path": str(fuzz_dir / "selected_targets.json"),
            "selected_target_api": "parse_replacement_field_then_tail",
            "selected_target_runtime_viability": "low",
        }
    )
    assert out["last_error"] == ""
    assert len(prompts) == 2
    assert "Update `fuzz/README.md` only" in prompts[1]
    assert out["synthesize_target_runtime_viability"] == "low"


def test_node_synthesize_completes_partial_scaffold_after_idle_like_partial_output(tmp_path: Path, monkeypatch):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "PLAN.md").write_text("# plan\n", encoding="utf-8")
    (fuzz_dir / "targets.json").write_text(
        '[{"name":"yaml_parser_parse","api":"yaml_parser_parse","lang":"c-cpp","target_type":"parser","seed_profile":"parser-structure"}]\n',
        encoding="utf-8",
    )

    calls: list[str] = []

    class _Patcher:
        def run_codex_command(self, prompt: str, **kwargs):
            calls.append(prompt)
            if len(calls) == 1:
                (fuzz_dir / "yaml_parser_parse_fuzz.cc").write_text(
                    "int LLVMFuzzerTestOneInput(const unsigned char*, unsigned long){return 0;}\n",
                    encoding="utf-8",
                )
                return None
            (fuzz_dir / "build.py").write_text("print('ok')\n", encoding="utf-8")
            (fuzz_dir / "README.md").write_text("# fuzz\n", encoding="utf-8")
            return None

    gen = SimpleNamespace(repo_root=tmp_path, patcher=_Patcher(), _pass_synthesize_harness=lambda timeout: None)
    monkeypatch.setattr(workflow_graph, "_has_codex_key", lambda: True)
    monkeypatch.setenv("SHERPA_SYNTHESIZE_GRACE_SEC", "0")

    out = workflow_graph._node_synthesize(
        {
            "generator": gen,
            "codex_hint": "use target hints",
            "antlr_context_path": "",
            "antlr_context_summary": "",
        }
    )

    assert out["last_error"] == ""
    assert len(calls) == 2
    assert "partial scaffold under `fuzz/`" in calls[1]
    assert (fuzz_dir / "build.py").is_file()


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
                    '[{"name":"parse_yaml","api":"parse_yaml","lang":"c-cpp","target_type":"parser","seed_profile":"parser-structure"}]\n',
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
    assert '"seed_profile": "parser-structure"' in targets


def test_collect_target_analysis_prefers_deeper_targets_over_shallow_utilities(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "z.c").write_text(
        "unsigned long adler32(unsigned long adler, const unsigned char* buf, unsigned int len) { return adler + len; }\n"
        "int decode_stream(const unsigned char* data, unsigned long size) { return (data && size) ? 1 : 0; }\n",
        encoding="utf-8",
    )

    doc = workflow_graph._collect_target_analysis_context(tmp_path)
    recommended = doc.get("recommended_targets") or []
    assert recommended
    names = [str(item.get("name") or "") for item in recommended]
    assert "decode_stream" in names
    assert "adler32" not in names
    decode_entry = next(item for item in recommended if item.get("name") == "decode_stream")
    assert decode_entry.get("depth_class") in {"medium", "deep"}
    assert int(decode_entry.get("depth_score") or 0) > 0


def test_node_plan_marks_replan_ineffective_when_outputs_do_not_materially_change(tmp_path: Path, monkeypatch):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "PLAN.md").write_text("# plan\nsame\n", encoding="utf-8")
    (fuzz_dir / "targets.json").write_text(
        '[{"name":"yaml_parser_parse","api":"yaml_parser_parse","lang":"c-cpp","target_type":"parser","seed_profile":"parser-structure"}]\n',
        encoding="utf-8",
    )

    class _Patcher:
        def run_codex_command(self, _prompt: str, **kwargs):
            (fuzz_dir / "PLAN.md").write_text("# plan\nsame\n", encoding="utf-8")
            (fuzz_dir / "targets.json").write_text(
                '[{"name":"yaml_parser_parse","api":"yaml_parser_parse","lang":"c-cpp","target_type":"parser","seed_profile":"parser-structure"}]\n',
                encoding="utf-8",
            )
            return None

    gen = SimpleNamespace(repo_root=tmp_path, patcher=_Patcher(), _pass_plan_targets=lambda timeout: None)
    monkeypatch.setattr(workflow_graph, "_has_codex_key", lambda: True)
    monkeypatch.setattr(workflow_graph, "_make_plan_hint", lambda _repo_root: "base plan hint")
    monkeypatch.setenv("SHERPA_PLAN_STRICT_TARGETS_SCHEMA", "1")

    out = workflow_graph._node_plan(
        {
            "generator": gen,
            "codex_hint": "",
            "coverage_improve_mode": "replan",
            "coverage_replan_required": True,
            "coverage_target_name": "yaml_parser_parse",
            "coverage_target_depth_score": 10,
            "coverage_target_depth_class": "medium",
        }
    )

    assert out["last_error"] == ""
    assert out["replan_effective"] is False
    assert out["replan_stop_reason"] == "no_material_change"
    assert out["coverage_replan_effective"] is False
    assert out["coverage_round_budget_exhausted"] is True
    assert out["coverage_stop_reason"] == "no_material_change"
