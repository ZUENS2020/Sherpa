---
name: synthesize
description: Generate complete fuzz scaffold artifacts aligned to selected targets and execution plan.
compatibility: opencode
metadata:
  stage: synthesize
  owner: sherpa
---

## What this skill does
Builds a complete `fuzz/` scaffold from planning artifacts, including harness source, build script, and runtime facts.

## When to use this skill
Use this skill in the primary `synthesize` stage after `plan`.

## Required inputs
- `fuzz/PLAN.md`
- `fuzz/targets.json`
- `fuzz/execution_plan.json` (if present)
- `fuzz/selected_targets.json` (if present)
- `fuzz/observed_target.json` (if present)

## Required outputs
- at least one harness source file under `fuzz/` (`*.c`, `*.cc`, `*.cpp`, `*.cxx`, or `*.java`) before docs/json completion
- `fuzz/build.py` or `fuzz/build.sh`
- `fuzz/README.md`
- `fuzz/repo_understanding.json`
- `fuzz/build_strategy.json`
- `fuzz/build_runtime_facts.json`
- `fuzz/harness_index.json` aligned to `fuzz/execution_plan.json`

## Workflow
1. Read planning artifacts and lock target alignment first.
2. Create harness source(s) before scaffold documentation (`harness-first contract`).
3. Create build glue with runtime artifact discovery and compiler-by-suffix behavior.
4. Create README/JSON strategy files with consistent selected/final target semantics.
5. Validate execution plan and harness index mappings.

## Key template contracts
### `fuzz/repo_understanding.json`
- Must include non-empty:
  - `build_system`
  - `chosen_target_api`
  - `chosen_target_reason`
  - `fuzzer_entry_strategy`
  - `evidence` (non-empty array)
- `chosen_target_api` must be a target API identifier, not a harness path.
- Forbidden examples: `fuzz/xxx_fuzz.cc`, `fuzz/xxx.c`, `xxx_fuzz.cpp`, `target_fuzz.java`.
- `build_system` must not be `unknown`.
- `evidence` must be a non-empty string array.

Minimal valid template:
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

### `fuzz/build.py`
- Must include:
  - `DEFAULT_CMAKE_ARGS = ["-DENABLE_TEST=OFF", "-DENABLE_INSTALL=OFF"]`
  - Python-native parallel args: `["-j", str(os.cpu_count() or 1)]`
  - never use `$(nproc)` or shell substitutions
  - runtime artifact discovery (do not hardcode a single static library path)

Exact static-lib discovery block:
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

Compiler-by-suffix rule:
- use `clang` for `.c` sources
- use `clang++` for `.cc`, `.cpp`, `.cxx` sources
- never compile all harness sources with `clang++` by default

### API surface rule
- Prefer public/stable APIs in harness code.
- Avoid internal/private namespaces (`detail`, `_internal`, `impl`, `private`) by default.
- If no public alternative exists, add `api_surface_exception` in `fuzz/repo_understanding.json` with non-empty `reason` and `evidence` (optional `approved_symbols`).

## Constraints
- Multi-target buildability is required when execution plan has multiple targets.
- Do not leave stale or missing execution target mappings in `fuzz/harness_index.json`.
- When diagnostics include concrete file paths, use `Read and fix <path>[:line]` before broader edits.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Acceptance checklist
- Harness file count is >= 1 before marking completion.
- Required scaffold files exist.
- `fuzz/harness_index.json` maps each execution target to an existing harness source file.
- README field `Harness file:` points to a real harness file.
- `fuzz/repo_understanding.json` is semantically valid and complete.
- Build script follows compiler-by-suffix and static-lib-discovery contracts.

## Done contract
- Write `fuzz/out/` into `./done`.
