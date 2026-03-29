---
name: improve_harness_in_place
description: Improve coverage in place for the current target without switching target identity.
compatibility: opencode
metadata:
  stage: improve-harness-in-place
  owner: sherpa
---

## What this skill does
Apply concrete in-place harness/seed-modeling improvements for coverage growth while keeping the current target stable.

## When to use this skill
Use this skill after `coverage-analysis` selects `in_place` improvement mode.

## Required inputs
- coverage diagnostics from coordinator (`coverage_*`, `repair_*`, `codex_hint`)
- `SeedFeedback` and `HarnessFeedback` (if provided)
- `fuzz/coverage_report.txt` with uncovered functions (if provided)
- current `fuzz/` scaffold files
- `fuzz/execution_plan.json` and `fuzz/harness_index.json` (if present)
- dictionaries under `fuzz/dict/*.dict` (if present)

## Required outputs
- material code/scaffold updates under `fuzz/`
- consistent `fuzz/execution_plan.json`, harness sources, and `fuzz/harness_index.json`
- updated dictionary entries when token modeling is part of the strategy

## Workflow
1. Consume coverage diagnostics, `SeedFeedback`, and `HarnessFeedback` first.
2. Identify one or more concrete bottlenecks (input model, call path depth, tokenization, corpus quality).
3. Apply minimal code/scaffold updates for those bottlenecks.
4. Keep target identity unchanged for this stage.

## Constraints
- No target switching in this stage.
- No doc-only patch in this stage.
- Include at least one strategy change versus the previous failed in-place cycle.
- Keep scaffold runnable for next build/run.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Acceptance checklist
- At least one change is directly mapped to `SeedFeedback` or `HarnessFeedback`.
- If uncovered functions are listed, at least one edit attempts to exercise them.
- `fuzz/execution_plan.json` and `fuzz/harness_index.json` remain consistent.

## Done contract
- Write `fuzz/out/` into `./done`.
