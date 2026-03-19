# Stage Skill: synthesize

## Stage Goal
Create a complete external fuzz scaffold aligned with the selected runtime target.

## Required Inputs
- `fuzz/PLAN.md`
- `fuzz/targets.json`
- `fuzz/selected_targets.json` (if present)
- `fuzz/observed_target.json` (if present)

## Required Outputs
- at least one harness source file under `fuzz/` (`*.c`, `*.cc`, `*.cpp`, `*.cxx`, or `*.java`)
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
- `fuzz/build.py` must include:
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

## Acceptance Criteria
- harness-first contract: create harness source file before completing build/json/readme scaffold.
- all required scaffold files exist.
- scaffold target alignment is explicit and consistent across README/harness/strategy files.
- build script does not hardcode a single guessed static-library path.
- `fuzz/README.md` field `Harness file:` points to an existing harness file under `fuzz/`.
- perform explicit self-check before completion:
  - confirm harness file count is >= 1;
  - confirm `fuzz/build.py|build.sh`, `fuzz/README.md`, `fuzz/repo_understanding.json`, `fuzz/build_strategy.json`, and `fuzz/build_runtime_facts.json` all exist.

## Command Policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done Sentinel Contract
- write `fuzz/out/` into `./done`.
