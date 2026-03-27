---
name: synthesize_repair_coverage
description: Repair scaffold for coverage replan cycles using seed and harness feedback as primary signals.
compatibility: opencode
metadata:
  stage: synthesize-repair-coverage
  owner: sherpa
---

## What this skill does
Applies coverage-oriented scaffold updates after replan decisions.

## When to use this skill
Use this skill when `coverage-analysis` selected replan and returned coverage diagnostics.

## Required inputs
- coverage diagnostics (`coverage_*`, `repair_*`)
- `SeedFeedback` and `HarnessFeedback` blocks (if provided)
- current scaffold files under `fuzz/`
- `fuzz/execution_plan.json` and `fuzz/harness_index.json` (if present)

## Required outputs
- updated harness/scaffold files under `fuzz/`
- `fuzz/harness_index.json` aligned with `fuzz/execution_plan.json`

## Workflow
1. Consume `SeedFeedback` and `HarnessFeedback` first.
2. Identify coverage bottlenecks and propose concrete fixes.
3. Apply at least one strategy change from previous failed coverage cycle.
4. Keep execution plan and harness index consistent.

## Constraints
- Edits must be coverage-repair-driven (seed/modeling/call-path/depth).
- No doc-only no-op patch.
- Preserve runtime viability for next build/run cycle.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Acceptance checklist
- Changes map to coverage diagnostics.
- Strategy change is explicit.
- Execution-plan/harness-index consistency is preserved.

## Done contract
- Write `fuzz/out/` into `./done`.
