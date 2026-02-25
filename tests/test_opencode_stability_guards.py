from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import types

# Optional runtime deps are not required for these unit-level helper tests.
if "langchain_openai" not in sys.modules:
    mod = types.ModuleType("langchain_openai")
    class _DummyChatOpenAI:  # pragma: no cover
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    mod.ChatOpenAI = _DummyChatOpenAI
    sys.modules["langchain_openai"] = mod

if "langgraph.graph" not in sys.modules:
    pkg = types.ModuleType("langgraph")
    graph = types.ModuleType("langgraph.graph")
    graph.END = object()
    class _DummyStateGraph:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            pass
    graph.StateGraph = _DummyStateGraph
    sys.modules["langgraph"] = pkg
    sys.modules["langgraph.graph"] = graph

if "persistent_config" not in sys.modules:
    pmod = types.ModuleType("persistent_config")
    pmod.load_config = lambda: None
    sys.modules["persistent_config"] = pmod

if "fuzz_unharnessed_repo" not in sys.modules:
    fmod = types.ModuleType("fuzz_unharnessed_repo")
    class _Dummy:  # pragma: no cover
        pass
    fmod.FuzzerRunResult = _Dummy
    fmod.HarnessGeneratorError = RuntimeError
    fmod.NonOssFuzzHarnessGenerator = _Dummy
    fmod.RepoSpec = _Dummy
    fmod.parse_libfuzzer_final_stats = lambda *_args, **_kwargs: {}
    fmod.snapshot_repo_text = lambda *a, **k: ""
    fmod.write_patch_from_snapshot = lambda *a, **k: None
    sys.modules["fuzz_unharnessed_repo"] = fmod

import workflow_graph


def test_validate_targets_json_rejects_missing_required_fields(tmp_path: Path):
    fuzz = tmp_path / "fuzz"
    fuzz.mkdir(parents=True, exist_ok=True)
    (fuzz / "targets.json").write_text(json.dumps([{"name": "f", "lang": "c-cpp"}]), encoding="utf-8")

    ok, err = workflow_graph._validate_targets_json(tmp_path)

    assert not ok
    assert "api" in err


def test_validate_targets_json_accepts_valid_minimal_schema(tmp_path: Path):
    fuzz = tmp_path / "fuzz"
    fuzz.mkdir(parents=True, exist_ok=True)
    (fuzz / "targets.json").write_text(
        json.dumps([{"name": "f", "api": "LLVMFuzzerTestOneInput", "lang": "c-cpp"}]),
        encoding="utf-8",
    )

    ok, err = workflow_graph._validate_targets_json(tmp_path)

    assert ok
    assert err == ""


def test_summarize_build_error_classifies_linker_issue():
    out = workflow_graph._summarize_build_error(
        "",
        "",
        "ld: undefined reference to foo\ncollect2: error: ld returned 1 exit status",
    )

    assert out["error_type"] == "link_error"
    assert "undefined reference" in out["evidence"]


def test_collect_key_artifact_hashes_only_returns_existing(tmp_path: Path):
    fuzz = tmp_path / "fuzz"
    fuzz.mkdir(parents=True, exist_ok=True)
    (fuzz / "targets.json").write_text("[]\n", encoding="utf-8")

    hashes = workflow_graph._collect_key_artifact_hashes(tmp_path)

    assert sorted(hashes.keys()) == ["fuzz/targets.json"]
    assert len(hashes["fuzz/targets.json"]) == 64


def test_classify_build_failure_marks_buildx_issue_as_infra():
    kind, code = workflow_graph._classify_build_failure(
        "",
        "",
        "BuildKit is enabled but the buildx component is missing or broken.",
        build_rc=1,
        has_fuzzer_binaries=False,
    )

    assert kind == "infra"
    assert code == "buildkit_unavailable"


def test_classify_build_failure_marks_compile_error_as_source():
    kind, code = workflow_graph._classify_build_failure(
        "",
        "",
        "fatal error: zlib.h: No such file or directory",
        build_rc=1,
        has_fuzzer_binaries=False,
    )

    assert kind == "source"
    assert code in {"missing_source_file", "compile_error"}


def test_classify_build_failure_marks_dns_issue_as_infra():
    kind, code = workflow_graph._classify_build_failure(
        "",
        "",
        "failed to do request: Head https://registry-1.docker.io/v2/: dial tcp: lookup registry-1.docker.io: no such host",
        build_rc=1,
        has_fuzzer_binaries=False,
    )

    assert kind == "infra"
    assert code == "registry_dns_resolution_failed"


def test_build_failure_recovery_advice_mentions_dns_for_registry_resolution():
    advice = workflow_graph._build_failure_recovery_advice("infra", "registry_dns_resolution_failed")

    assert "DNS" in advice
    assert "SHERPA_DOCKER_BUILD_RETRIES" in advice
