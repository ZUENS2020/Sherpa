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


def test_inject_instruments_library_make_cflags(tmp_path: Path) -> None:
    # The real failure: the target library is compiled via `make` with a
    # hardcoded CFLAGS string that lacks coverage, so libFuzzer is blind to the
    # library and coverage flatlines at the few harness edges. The injector must
    # append coverage to that library compile-flags string.
    bp = tmp_path / "build.py"
    bp.write_text(
        "def build_make(repo_root, cc='clang', cflags=None):\n"
        "    if cflags is None:\n"
        "        cflags = \"-std=c99 -Wall -Wextra -fpic -O2 -DNDEBUG\"\n"
        "    env = os.environ.copy()\n"
        "    env['CFLAGS'] = cflags\n"
        "    run(['make', '-C', str(repo_root), 'libtoml.a'], env=env)\n",
        encoding="utf-8",
    )
    workflow_graph._inject_coverage_instrumentation(str(bp), {})
    out = bp.read_text(encoding="utf-8")
    # the library CFLAGS string now carries libFuzzer coverage
    assert 'cflags = "-std=c99 -Wall -Wextra -fpic -O2 -DNDEBUG ' + MODERN + '"' in out
    # and the file is still valid Python
    compile(out, str(bp), "exec")


def test_inject_skips_replay_cflags_string(tmp_path: Path) -> None:
    # A replay CFLAGS string (-fprofile-instr-generate) is for llvm-cov, not
    # libFuzzer; the library pass must leave it untouched.
    bp = tmp_path / "build.py"
    replay = (
        "    env['CFLAGS'] = "
        "\"-std=c99 -fpic -fprofile-instr-generate -fcoverage-mapping\"\n"
    )
    bp.write_text(replay, encoding="utf-8")
    workflow_graph._inject_coverage_instrumentation(str(bp), {})
    assert bp.read_text(encoding="utf-8") == replay


def test_inject_forces_cc_wrapper_onto_bare_make(tmp_path: Path) -> None:
    # The hard case: build.py builds the library via a BARE `make` with no
    # CFLAGS at all, relying on the project Makefile (which may hard-assign
    # CFLAGS and ignore the environment). The injector must force the library
    # compiler to the cc-wrapper via a command-line `make CC=...` override so
    # coverage is appended regardless of the Makefile's own flags.
    fuzz = tmp_path / "fuzz"
    fuzz.mkdir()
    bp = fuzz / "build.py"
    bp.write_text(
        "import subprocess\n"
        "def run(cmd, **kw):\n"
        "    return subprocess.run(cmd, check=True, **kw)\n"
        "def build_make_library():\n"
        "    run(['make', '-C', str(REPO_ROOT), 'libtoml.a', '-j', '4'])\n"
        "def build_make_library_coverage():\n"
        "    env = os.environ.copy()\n"
        "    env['CFLAGS'] = '-fprofile-instr-generate -fcoverage-mapping'\n"
        "    run(['make', '-C', str(REPO_ROOT), 'libtoml.a'], env=env)\n",
        encoding="utf-8",
    )
    workflow_graph._inject_coverage_instrumentation(str(bp), {})
    out = bp.read_text(encoding="utf-8")
    wdir = fuzz / ".sherpa-cc"
    # primary (no env=) make now forces CC/CXX to the wrapper
    assert f'"CC={wdir / "clang"}"' in out
    assert f'"CXX={wdir / "clang++"}"' in out
    # the replay make (env=env) is left untouched — its CC override count is 0
    assert out.count("'libtoml.a'], env=env)") == 1  # unchanged replay line
    # still valid Python
    compile(out, str(bp), "exec")


def test_inject_instruments_direct_clang_c_library_compile(tmp_path: Path) -> None:
    # build.py that skips make and compiles the library directly with clang -c.
    # The library object must get coverage; the harness link and a replay
    # compile must be left to the other passes / untouched.
    bp = tmp_path / "build.py"
    bp.write_text(
        "lib_cmd = [\"clang\", \"-c\", \"-std=c99\", \"-Wall\", \"-Wextra\"]\n"
        "lib_cmd += sources\n"
        "run(lib_cmd)\n"
        "link = [\"clang\", \"-fsanitize=fuzzer,address,undefined\", \"-o\", \"out\", \"h.c\", \"lib.a\"]\n"
        "run(link)\n"
        "replay = [\"clang\", \"-c\", \"-fprofile-instr-generate\", \"-fcoverage-mapping\", \"toml.c\"]\n"
        "run(replay)\n",
        encoding="utf-8",
    )
    workflow_graph._inject_coverage_instrumentation(str(bp), {})
    out = bp.read_text(encoding="utf-8")
    # the library compile got coverage appended
    assert 'lib_cmd = ["clang", "-c", "-std=c99", "-Wall", "-Wextra", "' + MODERN + '"]' in out
    # the replay clang -c (with -fprofile) was NOT touched
    assert '["clang", "-c", "-fprofile-instr-generate", "-fcoverage-mapping", "toml.c"]' in out
    # the fuzzer link got coverage via the harness-link pass (not double via Pass D)
    assert out.count(MODERN) == 2  # library compile + harness link
    compile(out, str(bp), "exec")


def test_inject_single_element_list_no_string_concat(tmp_path: Path) -> None:
    # Regression: a single-element flags list with NO trailing comma
    #   PRIMARY_SANITIZER_FLAGS = ["-fsanitize=fuzzer,address,undefined"]
    # must not produce two adjacent string literals (which Python silently
    # concatenates into one malformed `-fsanitize=...undefined-fsanitize-coverage`
    # flag that fails to compile — broke tinyexpr builds).
    import ast

    bp = tmp_path / "build.py"
    bp.write_text(
        'PRIMARY_SANITIZER_FLAGS = ["-fsanitize=fuzzer,address,undefined"]\n',
        encoding="utf-8",
    )
    workflow_graph._inject_coverage_instrumentation(str(bp), {})
    out = bp.read_text(encoding="utf-8")
    assert "undefined-fsanitize-coverage" not in out  # the concatenation bug
    tree = ast.parse(out)  # still valid Python
    elts = [e.value for e in tree.body[0].value.elts]
    assert elts == [
        "-fsanitize=fuzzer,address,undefined",
        MODERN,
    ], elts


def test_inject_multi_element_list_keeps_separate_elements(tmp_path: Path) -> None:
    import ast

    bp = tmp_path / "build.py"
    bp.write_text(
        'cmd = ["clang", "-fsanitize=fuzzer,address,undefined", "h.c"]\n',
        encoding="utf-8",
    )
    workflow_graph._inject_coverage_instrumentation(str(bp), {})
    out = bp.read_text(encoding="utf-8")
    assert "undefined-fsanitize-coverage" not in out
    elts = [e.value for e in ast.parse(out).body[0].value.elts]
    assert MODERN in elts and "-fsanitize=fuzzer,address,undefined" in elts


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
