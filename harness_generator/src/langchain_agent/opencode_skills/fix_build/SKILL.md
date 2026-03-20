# Stage Skill: fix_build

## Stage Goal
Fix `fuzz/` build glue so the next workflow build attempt can pass.

## Required Inputs
- build diagnostics (`fuzz/build_full.log` or coordinator context)
- current `fuzz/build.py` and harness files
- current strategy/understanding files under `fuzz/`
- `fuzz/execution_plan.json` (if present)

## Required Outputs
- minimal build fix under `fuzz/`
- consistent updates to strategy/understanding/runtime-facts files when needed

## Key File Templates
- `fuzz/build.py` must preserve:
  - `DEFAULT_CMAKE_ARGS = ["-DENABLE_TEST=OFF", "-DENABLE_INSTALL=OFF"]`
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
- if dependency evidence exists, update `fuzz/system_packages.txt` with canonical vcpkg names only.
- canonical vcpkg examples: `zlib`, `bzip2`, `liblzma`, `lz4`, `zstd`, `openssl`, `expat`, `libxml2` (never `z`, `bz2`, `lzma`).
- if editing `fuzz/repo_understanding.json`, keep `chosen_target_api` as API identifier (not `fuzz/*.cc`-style path), keep `build_system != unknown`, and keep `evidence` as non-empty string array.
- when execution plan requires multiple targets, do not "fix" build by dropping to single-target-only output.

## Acceptance Criteria
- fix is evidence-driven and minimal.
- must produce textual code changes when current diagnostics are still failing; pure no-op is invalid.
- no edits outside `fuzz/` (except `./done`).
- strategy/understanding files remain aligned with build behavior.
- build result remains aligned with execution-plan target coverage constraints.
- do not bypass workflow acceptance by weakening or corrupting `repo_understanding` semantics.
- when the same error repeats, change strategy instead of repeating the same patch.

## Command Policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done Sentinel Contract
- write `fuzz/build.py` into `./done`.
