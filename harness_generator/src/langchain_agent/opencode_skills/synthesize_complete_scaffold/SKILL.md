# Stage Skill: synthesize_complete_scaffold

## Stage Goal
Complete only missing required scaffold items without rewriting unrelated files.

## Required Inputs
- current `fuzz/` scaffold
- missing items list from coordinator
- `fuzz/execution_plan.json` (if present)

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
  - if fields are present but semantically invalid, repair them before any cosmetic/doc-only edits
  - ensure non-empty `build_system`, `chosen_target_api`, `chosen_target_reason`, `fuzzer_entry_strategy`
  - ensure `evidence` is a non-empty array
  - ensure `chosen_target_api` is an API identifier, not a harness file path/value ending in `.c|.cc|.cpp|.cxx|.java`
  - ensure `build_system.lower() != "unknown"`
  - minimal valid shape example:
```json
{
  "build_system": "cmake",
  "chosen_target_api": "archive_read_open1",
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
- if `fuzz/execution_plan.json` has multiple execution targets, repair scaffold to keep multi-target buildability.
- if `fuzz/build.py` exists and uses invalid parallel style (for example `$(nproc)`), repair it to Python-native args such as `["-j", str(os.cpu_count() or 1)]`.

## Acceptance Criteria
- all required scaffold files exist after this step.
- if harness was missing before this step, harness exists after this step.
- if `fuzz/repo_understanding.json` existed but was incomplete or semantically invalid, required keys are repaired and non-empty.
- `chosen_target_api` is not a harness file path pattern and not a filename-only suffix value.
- `build_system` is concrete (not `unknown`) and `evidence` is a non-empty string array.
- existing harness/build assets are preserved unless minimal changes are required.
- no guessed paths/targets are introduced.
- repaired scaffold remains consistent with `fuzz/execution_plan.json` when that file exists.

## Command Policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done Sentinel Contract
- write `fuzz/out/` into `./done`.
