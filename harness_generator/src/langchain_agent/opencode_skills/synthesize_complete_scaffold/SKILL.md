# Stage Skill: synthesize_complete_scaffold

## Stage Goal
Complete only missing required scaffold items without rewriting unrelated files.

## Required Inputs
- current `fuzz/` scaffold
- missing items list from coordinator

## Required Outputs
- missing required files completed under `fuzz/`
- if harness source is missing, create at least one harness source file under `fuzz/` before only-doc/json fixes

## Key File Templates
- `fuzz/README.md` required fields:
  - `Selected target: ...`
  - `Final target: ...`
  - `Technical reason: ...`
  - `Relation: ...`
  - `Harness file: ...`
- `fuzz/repo_understanding.json` repair contract:
  - if file exists but keys are incomplete, repair it in place
  - ensure non-empty `build_system`, `chosen_target_api`, `chosen_target_reason`, `fuzzer_entry_strategy`
  - ensure `evidence` is a non-empty array
  - minimal valid shape example:
```json
{
  "build_system": "cmake",
  "chosen_target_api": "target_fuzz.cc",
  "chosen_target_reason": "runtime-reachable entrypoint",
  "fuzzer_entry_strategy": "sanitizer_fuzzer",
  "evidence": ["concrete repo build facts"]
}
```
- `fuzz/build_strategy.json`
  - explicit `fuzzer_entry_strategy`
  - consistent with existing harness/build path
- `fuzz/repo_understanding.json` and `fuzz/build_runtime_facts.json`
  - concise, evidence-backed, scaffold-consistent
- if `fuzz/build.py` exists and uses invalid parallel style (for example `$(nproc)`), repair it to Python-native args such as `["-j", str(os.cpu_count() or 1)]`.

## Acceptance Criteria
- all required scaffold files exist after this step.
- if harness was missing before this step, harness exists after this step.
- if `fuzz/repo_understanding.json` existed but was incomplete, required keys are repaired and non-empty.
- existing harness/build assets are preserved unless minimal changes are required.
- no guessed paths/targets are introduced.

## Command Policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done Sentinel Contract
- write `fuzz/out/` into `./done`.
