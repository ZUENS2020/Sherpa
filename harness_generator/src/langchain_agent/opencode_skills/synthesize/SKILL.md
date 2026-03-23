# Stage Skill: synthesize

## Stage Goal
Create a complete external fuzz scaffold aligned with the selected runtime target.

## Required Inputs
- `fuzz/PLAN.md`
- `fuzz/targets.json`
- `fuzz/execution_plan.json` (if present)
- `fuzz/selected_targets.json` (if present)
- `fuzz/observed_target.json` (if present)

## Required Outputs
- at least one harness source file under `fuzz/` (`*.c`, `*.cc`, `*.cpp`, `*.cxx`, or `*.java`) before completing scaffold docs/json.
- `fuzz/build.py` or `fuzz/build.sh`
- `fuzz/README.md`
- `fuzz/repo_understanding.json`
- `fuzz/build_strategy.json`
- `fuzz/build_runtime_facts.json`
- when execution plan has multiple targets, scaffold should preserve multi-target buildability (not single-target-only by default)

## Key File Templates
- `fuzz/README.md` fields:
  - `Selected target: ...`
  - `Final target: ...`
  - `Technical reason: ...`
  - `Relation: ...`
  - `Harness file: ...`
- `fuzz/repo_understanding.json` must include non-empty:
  - `build_system`
  - `chosen_target_api`
  - `chosen_target_reason`
  - `fuzzer_entry_strategy`
  - `evidence` (non-empty array)
- semantic contract for `chosen_target_api`:
  - must be a target API identifier (function/method/entrypoint), not a harness file path
  - forbidden examples: `fuzz/xxx_fuzz.cc`, `fuzz/xxx.c`, `xxx_fuzz.cpp`, `target_fuzz.java`
  - forbidden fallback values: empty string, filename-only values ending with `.c|.cc|.cpp|.cxx|.java`
- hard rules:
  - `build_system` must not be `unknown`
  - `evidence` must be a non-empty string array (not null/object/string)
- minimal valid template:
```json
{
  "build_system": "cmake",
  "chosen_target_api": "archive_read_open1",
  "chosen_target_reason": "runtime-reachable parser entrypoint",
  "fuzzer_entry_strategy": "sanitizer_fuzzer",
  "evidence": [
    "CMakeLists.txt defines a library target",
    "selected target API appears in repository source"
  ]
}
```
- `fuzz/build.py` must include:
  - `DEFAULT_CMAKE_ARGS = ["-DENABLE_TEST=OFF", "-DENABLE_INSTALL=OFF"]`
  - never use shell substitutions like `$(nproc)` or backticks inside Python command lists
  - for parallel build, use Python-native argument form: `["-j", str(os.cpu_count() or 1)]`
  - run CMake configure with real args (example: `-DENABLE_TEST=OFF`, not malformed quoting)
  - exact static-lib discovery block:
```python
def find_static_lib(repo_root):
    import subprocess
    result = subprocess.run(
        ["find", str(repo_root), "-name", "*.a", "-type", "f"],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        return None
    for p in result.stdout.strip().split("\n"):
        p = Path(p)
        if "test" not in p.name.lower() and p.exists():
            return p
    return None
```
  - and a complete compile flow that actually uses `find_static_lib()` and builds the fuzzer binary:
```python
from pathlib import Path
import os
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
FUZZ_DIR = REPO_ROOT / "fuzz"
OUT_DIR = FUZZ_DIR / "out"
BUILD_DIR = FUZZ_DIR / "build-work"

DEFAULT_CMAKE_ARGS = ["-DENABLE_TEST=OFF", "-DENABLE_INSTALL=OFF"]

def run_cmd(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)

def build_fuzzers():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    cmake_cmd = ["cmake", "-S", str(REPO_ROOT), "-B", str(BUILD_DIR)] + DEFAULT_CMAKE_ARGS
    run_cmd(cmake_cmd)
    run_cmd(["cmake", "--build", str(BUILD_DIR), "-j", str(os.cpu_count() or 1)])

    static_lib = find_static_lib(BUILD_DIR) or find_static_lib(REPO_ROOT)
    if static_lib is None:
        raise RuntimeError("static library not found")

    harness_src = FUZZ_DIR / "target_fuzz.cc"
    fuzz_bin = OUT_DIR / "target_fuzz"
    run_cmd(
        [
            "clang++",
            "-std=c++17",
            "-O1",
            "-g",
            "-fno-omit-frame-pointer",
            "-fsanitize=address,undefined,fuzzer",
            str(harness_src),
            str(static_lib),
            "-o",
            str(fuzz_bin),
        ]
    )

if __name__ == "__main__":
    build_fuzzers()
```
  - `build_fuzzers()` must call `find_static_lib()` directly; defining it without calling it is invalid.
  - `build_fuzzers()` must produce a concrete fuzzer executable under `fuzz/out/`.
  - if `fuzz/execution_plan.json` lists multiple execution targets, `build_fuzzers()` should compile/link multiple fuzzers accordingly.
- prefer public/stable APIs in generated harness code; avoid internal/private namespaces (for example `detail`, `_internal`, `impl`) unless concrete repository evidence proves no public entrypoint exists.
- soft-block rule for internal/private APIs:
  - default: do not use `detail::`, `internal::`, `_internal`, `impl::`, `private::` patterns in harness files.
  - exception path (only when no public alternative exists): add `api_surface_exception` to `fuzz/repo_understanding.json` with:
    - `reason` (non-empty string)
    - `evidence` (non-empty string array)
    - optional `approved_symbols` (specific symbols allowed to remain)

## Acceptance Criteria
- harness-first contract: create harness source file before completing build/json/readme scaffold.
- all required scaffold files exist.
- scaffold target alignment is explicit and consistent across README/harness/strategy files.
- harness code uses public/stable APIs by default; internal/private-only API usage requires explicit evidence in `fuzz/repo_understanding.json` `evidence`.
- if internal/private APIs remain, `fuzz/repo_understanding.json` must include valid `api_surface_exception.reason` and non-empty `api_surface_exception.evidence`.
- scaffold aligns with `fuzz/execution_plan.json` when present.
- build script does not hardcode a single guessed static-library path.
- when diagnostics/context include concrete file paths, issue explicit actions as `Read and fix <path>[:line]` before broader edits.
- `fuzz/README.md` field `Harness file:` points to an existing harness file under `fuzz/`.
- `fuzz/repo_understanding.json` contains all required non-empty keys:
  `build_system`, `chosen_target_api`, `chosen_target_reason`, `fuzzer_entry_strategy`, and non-empty `evidence`.
- perform explicit self-check before completion:
  - confirm harness file count is >= 1;
  - confirm `fuzz/build.py|build.sh`, `fuzz/README.md`, `fuzz/repo_understanding.json`, `fuzz/build_strategy.json`, and `fuzz/build_runtime_facts.json` all exist.
  - confirm `fuzz/repo_understanding.json` has non-empty `build_system/chosen_target_api/chosen_target_reason/fuzzer_entry_strategy/evidence`.
  - confirm `chosen_target_api` does not match harness file path patterns like `^fuzz/.*\\.(c|cc|cpp|cxx|java)$` and is not a filename-only suffix value.
  - confirm `build_system.lower() != \"unknown\"`.
  - confirm `evidence` is a non-empty array of non-empty strings.
  - confirm README consistency: `Final target` describes the same semantic API tracked by `chosen_target_api`, while `Harness file` points to an actual harness file path.
  - confirm `fuzz/build.py` contains a full `build_fuzzers()` flow that compiles and links a runnable fuzzer binary.

## Command Policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done Sentinel Contract
- write `fuzz/out/` into `./done`.
