# Stage Skill: synthesize_complete_scaffold

## Stage Goal
Complete only missing required scaffold items without rewriting unrelated files.

## Required Inputs
- current `fuzz/` scaffold
- missing items list from coordinator

## Required Outputs
- missing required files completed under `fuzz/`
- if harness source is missing, create at least one harness source file under `fuzz/`

## Key File Templates
- `fuzz/README.md` required fields:
  - `Selected target: ...`
  - `Final target: ...`
  - `Technical reason: ...`
  - `Relation: ...`
  - `Harness file: ...`
- `fuzz/build_strategy.json`
  - explicit `fuzzer_entry_strategy`
  - consistent with existing harness/build path
- `fuzz/repo_understanding.json` and `fuzz/build_runtime_facts.json`
  - concise, evidence-backed, scaffold-consistent

## Acceptance Criteria
- all required scaffold files exist after this step.
- if harness was missing before this step, harness exists after this step.
- existing harness/build assets are preserved unless minimal changes are required.
- no guessed paths/targets are introduced.

## Command Policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done Sentinel Contract
- write `fuzz/out/` into `./done`.
