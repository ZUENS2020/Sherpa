# Stage Skill: plan

## Stage Goal
Produce a realistic, runtime-viable target plan for harness generation.

## Required Inputs
- `fuzz/target_analysis.json` (if present)
- `fuzz/antlr_plan_context.json` (if present)
- repository source/build metadata

## Required Outputs
- `fuzz/PLAN.md`
- `fuzz/targets.json`
- `fuzz/execution_plan.json`

## Key File Templates
- `fuzz/PLAN.md`
  - selected best target (first)
  - short rationale
  - implementation hints for synthesize/build
- `fuzz/targets.json`
  - non-empty JSON array
  - each item has non-empty: `name`, `api`, `lang`, `target_type`, `seed_profile`
  - `api` must describe a target API identifier, not a generated harness source path
  - forbidden `api` examples: `fuzz/*.c`, `fuzz/*.cc`, `fuzz/*.cpp`, `fuzz/*.cxx`, `fuzz/*.java`
  - forbidden: `name = LLVMFuzzerTestOneInput`
  - rank runtime-executable/public targets first
- `fuzz/execution_plan.json`
  - include top execution targets (default top 3)
  - each target keeps `execution_priority`, `must_run`, `target_name`, `expected_fuzzer_name`, `seed_profile`
  - include `min_required_built_targets` (default >=2 when multiple execution targets exist)

## Acceptance Criteria
- `fuzz/PLAN.md` exists and references a concrete primary target.
- `fuzz/targets.json` is strict-schema valid and non-empty.
- top-ranked target is runtime-viable and not a helper-only target.
- `fuzz/execution_plan.json` exists and is consistent with selected top runtime targets.

## Command Policy
- Allowed: read-only commands only (`find`, `grep`, `rg`, `cat`, `ls`, `head`, `tail`, read-only `sed`).
- Forbidden: build/execute commands.

## Done Sentinel Contract
- write `fuzz/PLAN.md` into `./done`.
