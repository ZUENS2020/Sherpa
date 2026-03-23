# Stage Skill: plan_repair_crash

## Stage Goal
Repair planning artifacts after a crash/repro-stage failure.

## Required Inputs
- `repair_*` diagnostics from coordinator context
- crash/repro summaries and report tails (if provided)
- `fuzz/PLAN.md`, `fuzz/targets.json`, `fuzz/execution_plan.json` (if present)

## Required Outputs
- updated `fuzz/PLAN.md`
- schema-valid `fuzz/targets.json`
- updated `fuzz/execution_plan.json`

## Acceptance Criteria
- plan explicitly addresses crash-path failure diagnostics.
- plan includes a strategy change when the same crash/repro signature repeats.
- target relation is explicit: selected target vs observed runtime/crash target.
- avoid “repair” strategies that only disable behavior instead of preserving crash-path reachability.
- default to public/stable APIs for harness logic.
- if non-public/internal API is unavoidable, require `api_surface_exception` in `fuzz/repo_understanding.json` with non-empty `reason` and `evidence`.
- when diagnostics contain `non_public_api_usage`, plan must prioritize replacing offending symbols first.

## Command Policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done Sentinel Contract
- write `fuzz/PLAN.md` into `./done`.
