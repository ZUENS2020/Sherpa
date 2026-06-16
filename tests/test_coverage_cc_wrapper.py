from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "harness_generator" / "src" / "langchain_agent"
SRC = ROOT / "harness_generator" / "src"
for p in (APP, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import workflow_graph as wg  # noqa: E402


def test_wrapper_env_installs_and_prepends_path(tmp_path):
    env = wg._apply_coverage_cc_wrapper_env({"PATH": "/usr/bin"}, tmp_path)
    wdir = tmp_path / "fuzz" / ".sherpa-cc"
    assert (wdir / "sherpa-cc-wrapper.sh").is_file()
    assert (wdir / "clang").is_symlink() and (wdir / "clang++").is_symlink()
    assert env["PATH"].startswith(str(wdir))
    assert env["CC"] == "clang" and env["CXX"] == "clang++"


def test_wrapper_instruments_plain_library_compile(tmp_path):
    # The whole point: a bare `clang -c lib.c` (no -fsanitize=fuzzer) must still
    # get coverage instrumentation so libFuzzer sees the library, not just the
    # harness.
    wg._apply_coverage_cc_wrapper_env({"PATH": "/usr/bin"}, tmp_path)
    sh = tmp_path / "fuzz" / ".sherpa-cc" / "sherpa-cc-wrapper.sh"
    assert subprocess.run(["bash", "-n", str(sh)], capture_output=True).returncode == 0
    body = sh.read_text()
    # adds coverage when neither coverage nor replay-profile flags are present
    assert "HAS_COV -eq 0" in body and "HAS_REPLAY -eq 0" in body
    assert "inline-8bit-counters,pc-table" in body
    # and normalizes deprecated trace-pc-guard
    assert "*trace-pc-guard*)" in body


def test_wrapper_env_degrades_safely(tmp_path, monkeypatch):
    # never raise even if install fails
    monkeypatch.setattr(wg, "_install_coverage_cc_wrapper", lambda r: (_ for _ in ()).throw(OSError("x")))
    env = wg._apply_coverage_cc_wrapper_env({"PATH": "/usr/bin"}, tmp_path)
    assert env["PATH"] == "/usr/bin"  # unchanged, no crash
