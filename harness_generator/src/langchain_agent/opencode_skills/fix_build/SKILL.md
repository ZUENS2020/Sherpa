---
name: fix_build
description: Apply minimal evidence-driven build fixes in fuzz scaffold files for next build attempt.
compatibility: opencode
metadata:
  stage: fix-build
  owner: tianheng
---

## What this skill does
Repairs `fuzz/` build glue and related scaffold metadata after build failures.

## When to use this skill
Use this skill when build diagnostics exist and coordinator requests targeted build recovery.

## Required inputs
- build diagnostics (`fuzz/build_full.log` or coordinator context)
- current `fuzz/build.py` and harness files
- strategy/understanding files under `fuzz/`
- `fuzz/execution_plan.json` (if present)
- `previous_failed_attempts` from context (if provided)

## Required outputs
- minimal build fix under `fuzz/`
- consistent updates to strategy/understanding/runtime-facts when required

## Key template contract (`fuzz/build.py`)
- Keep:
  - `DEFAULT_CMAKE_ARGS = ["-DENABLE_TEST=OFF", "-DENABLE_INSTALL=OFF"]`
  - invoke CMake with the system `cmake` binary (for example `["cmake", "-S", ...]` and `["cmake", "--build", ...]`); do not use `sys.executable, "-m", "cmake"` or `python -m cmake`
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
- Compiler-by-suffix is mandatory:
  - compile `.c` with `clang`
  - compile `.cc/.cpp/.cxx` with `clang++`
  - never compile C files with `clang++` by default

## Workflow
1. Read diagnostics first; identify smallest root-cause edit.
2. Apply minimal patch tied to concrete symbol/file/line errors. When multiple `undefined reference` errors point at internal library symbols, prefer adding the entire owning source directory (e.g. `apps/lib/*.c`) to the build SOURCES list in one edit, rather than chasing one symbol at a time.
3. Only when the build fails on a genuinely missing *external* library, add its canonical vcpkg port name to `fuzz/system_packages.txt`. Do not declare ports speculatively or for libraries the repo vendors/builds itself — a self-contained `make`/single-file project needs no entries, and spurious ports trigger an unnecessary vcpkg bootstrap.
4. Keep execution-plan coverage intent (do not silently collapse multi-target plans).

## Constraints
- Canonical vcpkg examples: `zlib`, `bzip2`, `liblzma`, `lz4`, `zstd`, `openssl`, `expat`, `libxml2` (never `z`, `bz2`, `lzma`).
- If editing `fuzz/repo_understanding.json`, keep:
  - `chosen_target_api` as API identifier (not `fuzz/*.cc` path)
  - `build_system != unknown`
  - `evidence` as non-empty string array
- Prefer public/stable APIs in harness code.
- For `non_public_api_usage`, replace offending symbols first.
- If no public alternative exists, record `api_surface_exception` with non-empty `reason` and `evidence`.
- Must produce textual code changes; pure no-op is invalid.
- Stale `./done` without fresh diff is invalid.
- No edits outside `fuzz/` except `./done`.
- Do not bypass workflow acceptance by weakening `repo_understanding` semantics.
- Use explicit path actions: `Read and fix <path>[:line]`.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Acceptance checklist
- Fix is evidence-driven and minimal.
- Repeated signatures trigger a changed strategy.
- Build/scaffold behavior remains aligned with execution-plan expectations.

## Done contract
- Write one key modified path under `fuzz/` as the sole text of `./done` (a relative path string like `fuzz/foo.c`; do **not** copy file contents).
