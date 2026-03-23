# Stage Skill: plan_repair_build

## Stage Goal
Repair planning artifacts after a build-stage failure.

## Required Inputs
- `repair_*` diagnostics from coordinator context
- `fuzz/PLAN.md` (if present)
- `fuzz/targets.json` (if present)
- `fuzz/execution_plan.json` (if present)

## Required Outputs
- updated `fuzz/PLAN.md`
- schema-valid `fuzz/targets.json`
- updated `fuzz/execution_plan.json`

## Acceptance Criteria
- plan explicitly addresses current build failure kind/code/signature.
- plan includes at least one strategy change from the latest failed attempt.
- targets remain runtime-viable and executable-first.
- do not produce doc-only updates disconnected from build recovery.
- default to public/stable APIs for harness logic.
- if non-public/internal API is unavoidable, require `api_surface_exception` in `fuzz/repo_understanding.json` with non-empty `reason` and `evidence`.
- when diagnostics contain `non_public_api_usage`, plan must prioritize replacing offending symbols first.

## Command Policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done Sentinel Contract
- write `fuzz/PLAN.md` into `./done`.
