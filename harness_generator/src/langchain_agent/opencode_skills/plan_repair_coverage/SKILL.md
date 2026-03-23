# Stage Skill: plan_repair_coverage

## Stage Goal
Repair planning artifacts after a coverage plateau / replan trigger.

## Required Inputs
- coverage diagnostics from coordinator context (`coverage_*`, `repair_*`)
- `SeedFeedback` and `HarnessFeedback` blocks from coordinator context (when provided)
- `fuzz/PLAN.md` (if present)
- `fuzz/targets.json` (if present)
- `fuzz/execution_plan.json` (if present)
- `fuzz/harness_index.json` (if present)

## Required Outputs
- updated `fuzz/PLAN.md`
- schema-valid `fuzz/targets.json`
- updated `fuzz/execution_plan.json`
- explicit strategy-diff note from previous failed coverage cycle

## Acceptance Criteria
- plan consumes coverage diagnostics first (plateau reason, seed family gaps, quality flags).
- plan consumes `SeedFeedback` and `HarnessFeedback` first and maps each chosen action to one of these signals.
- plan describes at least one material strategy change, not a cosmetic rewrite.
- target choices remain runtime-viable and increase depth potential.
- plan keeps `fuzz/execution_plan.json` mappable to `fuzz/harness_index.json`.
- no doc-only update disconnected from next build/run outcomes.

## Command Policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done Sentinel Contract
- write `fuzz/PLAN.md` into `./done`.
