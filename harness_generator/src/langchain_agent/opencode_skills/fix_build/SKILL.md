# Stage Skill: fix_build

## Stage Goal
Fix `fuzz/` build glue so the next workflow build attempt can pass.

## Required Inputs
- build diagnostics (`fuzz/build_full.log` or coordinator context)
- current `fuzz/build.py` and harness files
- current strategy/understanding files under `fuzz/`

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

## Acceptance Criteria
- fix is evidence-driven and minimal.
- no edits outside `fuzz/` (except `./done`).
- strategy/understanding files remain aligned with build behavior.

## Command Policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done Sentinel Contract
- write `fuzz/build.py` into `./done`.
