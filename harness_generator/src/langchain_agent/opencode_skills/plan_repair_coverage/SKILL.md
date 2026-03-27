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

## Workflow
1. Read coverage diagnostics first.
2. Map seed/harness quality gaps to concrete actions.
3. Produce at least one material strategy change.
4. Keep execution plan mappable to harness index.

## Constraints
- Consume `SeedFeedback` and `HarnessFeedback` before proposing changes.
- Avoid cosmetic rewrites.
- Keep target choices runtime-viable and depth-oriented.
- No doc-only update disconnected from next build/run outcomes.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Acceptance checklist
- Plan includes strategy-diff with concrete changed actions.
- `fuzz/execution_plan.json` remains mappable to `fuzz/harness_index.json`.
- Coverage diagnostics are explicitly addressed.

## Done contract
- Write `fuzz/PLAN.md` into `./done`.
