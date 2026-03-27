---
name: plan_repair_coverage
description: Re-plan target and strategy after coverage plateau using seed/harness feedback first.
compatibility: opencode
metadata:
  stage: plan-repair-coverage
  owner: sherpa
---

## What this skill does
Repairs planning artifacts when coverage has plateaued and a replan decision is needed.

## When to use this skill
Use this skill when `coverage-analysis` selects replan mode.

## Required inputs
- coverage diagnostics (`coverage_*`, `repair_*`)
- `SeedFeedback` and `HarnessFeedback` blocks (if provided)
- `fuzz/PLAN.md`, `fuzz/targets.json`, `fuzz/execution_plan.json`, `fuzz/harness_index.json` (if present)

## Required outputs
- updated `fuzz/PLAN.md`
- schema-valid `fuzz/targets.json`
- updated `fuzz/execution_plan.json`
- explicit strategy-diff note from previous failed coverage cycle
- `Known Issues` section in `fuzz/PLAN.md`

## Workflow
1. Read coverage diagnostics first.
2. Map seed/harness quality gaps to concrete actions.
3. Produce at least one material strategy change.
4. Keep execution plan mappable to harness index.

## Constraints
- Consume `SeedFeedback` and `HarnessFeedback` before proposing changes.
- `fuzz/PLAN.md` must include `Known Issues` with unresolved coverage blockers, suspected cause, and planned corrective action.
- Avoid cosmetic rewrites.
- Keep target choices runtime-viable and depth-oriented.
- Naming contract to reduce execution drift:
  - `target_name` stays suffix-free and API-oriented.
  - `expected_fuzzer_name` maps predictably to harness/binary stem (prefer `<target_name>_fuzz` / `<target_name>_fuzzer`).
  - Keep `execution_plan` and `fuzz/harness_index.json` naming aligned.
- No doc-only update disconnected from next build/run outcomes.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Acceptance checklist
- Plan includes strategy-diff with concrete changed actions.
- `fuzz/execution_plan.json` remains mappable to `fuzz/harness_index.json`.
- Coverage diagnostics are explicitly addressed.
- `Known Issues` exists and lists concrete unresolved plateau/quality blockers.

## Done contract
- Write `fuzz/PLAN.md` into `./done`.
