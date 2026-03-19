# Stage Skill: synthesize

## Stage Goal
Create a complete external fuzz scaffold aligned with the selected runtime target.

## Required Inputs
- `fuzz/PLAN.md`
- `fuzz/targets.json`
- `fuzz/selected_targets.json` (if present)
- `fuzz/observed_target.json` (if present)

## Required Outputs
- at least one harness source file under `fuzz/` (`*.c`, `*.cc`, `*.cpp`, `*.cxx`, or `*.java`) before completing scaffold docs/json.
- `fuzz/build.py` or `fuzz/build.sh`
- `fuzz/README.md`
- `fuzz/repo_understanding.json`
- `fuzz/build_strategy.json`
- `fuzz/build_runtime_facts.json`

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
- minimal valid template:
```json
{
  "build_system": "cmake",
  "chosen_target_api": "target_fuzz.cc",
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

## Acceptance Criteria
- harness-first contract: create harness source file before completing build/json/readme scaffold.
- all required scaffold files exist.
- scaffold target alignment is explicit and consistent across README/harness/strategy files.
- build script does not hardcode a single guessed static-library path.
- `fuzz/README.md` field `Harness file:` points to an existing harness file under `fuzz/`.
- `fuzz/repo_understanding.json` contains all required non-empty keys:
  `build_system`, `chosen_target_api`, `chosen_target_reason`, `fuzzer_entry_strategy`, and non-empty `evidence`.
- perform explicit self-check before completion:
  - confirm harness file count is >= 1;
  - confirm `fuzz/build.py|build.sh`, `fuzz/README.md`, `fuzz/repo_understanding.json`, `fuzz/build_strategy.json`, and `fuzz/build_runtime_facts.json` all exist.
  - confirm `fuzz/repo_understanding.json` has non-empty `build_system/chosen_target_api/chosen_target_reason/fuzzer_entry_strategy/evidence`.
  - confirm `fuzz/build.py` contains a full `build_fuzzers()` flow that compiles and links a runnable fuzzer binary.

## Command Policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done Sentinel Contract
- write `fuzz/out/` into `./done`.
