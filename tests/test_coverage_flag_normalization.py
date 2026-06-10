from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
SRC_DIR = ROOT / "harness_generator" / "src"
for p in (APP_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import workflow_graph  # noqa: E402

MODERN = "-fsanitize-coverage=inline-8bit-counters,pc-table"


def test_inject_normalizes_deprecated_trace_pc_guard(tmp_path: Path) -> None:
    # Modern libFuzzer (clang>=14) refuses trace-pc-guard binaries at runtime.
    # A build.py the agent wrote with the deprecated flag must be rewritten.
    bp = tmp_path / "build.py"
    bp.write_text(
        "cmd = ['clang', '-fsanitize=fuzzer,address,undefined', "
        "'-fsanitize-coverage=trace-pc-guard,inline-8bit-counters', 'h.c']\n",
        encoding="utf-8",
    )
    workflow_graph._inject_coverage_instrumentation(str(bp), {})
    out = bp.read_text(encoding="utf-8")
    assert "trace-pc-guard" not in out
    assert MODERN in out


def test_inject_adds_modern_flag_when_absent(tmp_path: Path) -> None:
    bp = tmp_path / "build.py"
    bp.write_text(
        "cmd = ['clang', '-fsanitize=fuzzer,address,undefined', 'h.c']\n",
        encoding="utf-8",
    )
    workflow_graph._inject_coverage_instrumentation(str(bp), {})
    out = bp.read_text(encoding="utf-8")
    assert MODERN in out
    assert "trace-pc-guard" not in out


def test_inject_leaves_replay_lines_alone(tmp_path: Path) -> None:
    # Replay/coverage builds use -fprofile-instr-generate, not libFuzzer
    # coverage; the injector must not touch them.
    bp = tmp_path / "build.py"
    line = "cmd = ['clang', '-fprofile-instr-generate', '-fcoverage-mapping', 'h.c']\n"
    bp.write_text(line, encoding="utf-8")
    workflow_graph._inject_coverage_instrumentation(str(bp), {})
    assert bp.read_text(encoding="utf-8") == line


def test_cc_wrapper_is_valid_bash_and_rewrites(tmp_path: Path) -> None:
    wdir = workflow_graph._install_coverage_cc_wrapper(tmp_path)
    sh = wdir / "sherpa-cc-wrapper.sh"
    assert sh.is_file()
    # bash syntax check
    r = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    body = sh.read_text(encoding="utf-8")
    assert "inline-8bit-counters,pc-table" in body
    # the only mention of trace-pc-guard is the rewrite match pattern
    assert "*trace-pc-guard*)" in body
