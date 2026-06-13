from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "harness_generator" / "src" / "langchain_agent"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import promefuzz_companion as pc  # noqa: E402


def _mk_repo(tmp_path: Path) -> Path:
    (tmp_path / "include").mkdir(parents=True)
    (tmp_path / "include" / "toml.h").write_text("int toml_parse(const char*);\n", encoding="utf-8")
    (tmp_path / "toml.c").write_text(
        '#include "toml.h"\nint toml_parse(const char* s){return 0;}\n', encoding="utf-8"
    )
    return tmp_path


def test_synthesizes_db_for_makefile_project(tmp_path, monkeypatch):
    monkeypatch.delenv("SHERPA_SYNTH_COMPILE_COMMANDS", raising=False)
    repo = _mk_repo(tmp_path)
    out = pc._synthesize_compile_commands(repo)
    assert out is not None and out.is_file()
    db = json.loads(out.read_text(encoding="utf-8"))
    assert len(db) >= 1
    entry = next(e for e in db if e["file"].endswith("toml.c"))
    args = entry["arguments"]
    assert args[0] == "clang"
    # repo root + the include/ dir are on the include path
    inc = [a for a in args if a.startswith("-I")]
    assert any(a.endswith("/include") for a in inc)
    assert any(a == "-I" + str(repo.resolve()) for a in inc)
    assert entry["file"].endswith("toml.c")


def test_kill_switch_disables_synth(tmp_path, monkeypatch):
    monkeypatch.setenv("SHERPA_SYNTH_COMPILE_COMMANDS", "0")
    repo = _mk_repo(tmp_path)
    assert pc._synthesize_compile_commands(repo) is None
    assert not (repo / "compile_commands.json").is_file()


def test_no_sources_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("SHERPA_SYNTH_COMPILE_COMMANDS", raising=False)
    (tmp_path / "README.md").write_text("no code", encoding="utf-8")
    assert pc._synthesize_compile_commands(tmp_path) is None


def test_generate_falls_through_to_synth(tmp_path, monkeypatch):
    # no CMakeLists, no bear -> _try_generate_compile_commands uses the synth path
    monkeypatch.delenv("SHERPA_SYNTH_COMPILE_COMMANDS", raising=False)
    repo = _mk_repo(tmp_path)
    out = pc._try_generate_compile_commands(repo)
    assert out is not None and out.name == "compile_commands.json"
