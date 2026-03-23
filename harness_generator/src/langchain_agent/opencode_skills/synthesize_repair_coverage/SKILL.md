# Stage Skill: synthesize_repair_coverage

## Stage Goal
Repair scaffold files under `fuzz/` for coverage-improvement replan cycles.

## Required Inputs
- coverage diagnostics from coordinator context (`coverage_*`, `repair_*`)
- current scaffold files under `fuzz/`
- `fuzz/execution_plan.json` (if present)
- `fuzz/harness_index.json` (if present)

## Required Outputs
- harness source under `fuzz/`
- `fuzz/build.py` or `fuzz/build.sh`
- `fuzz/README.md`
- `fuzz/repo_understanding.json`
- `fuzz/build_strategy.json`
- `fuzz/build_runtime_facts.json`
- `fuzz/harness_index.json` aligned to `fuzz/execution_plan.json`

## Acceptance Criteria
- edits are coverage-repair-driven (seed/modeling/call-path/depth), not cosmetic.
- this cycle must include a strategy change from the previous failed coverage cycle.
- no doc-only no-op patches; scaffold must materially change where needed.
- `fuzz/execution_plan.json`, harness files, and `fuzz/harness_index.json` stay consistent.
- preserve runtime viability for the next build/run cycle.

## Command Policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done Sentinel Contract
- write `fuzz/out/` into `./done`.
