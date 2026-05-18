---
name: plan_repair_fix_harness
description: Repair planning loop for crash-triaged harness bugs with evidence-first strategy changes.
compatibility: opencode
metadata:
  stage: plan-repair-fix-harness
  owner: tianheng
---

## What this skill does
Builds a harness-bug-focused repair plan that feeds directly into synthesize/build recovery.

## When to use this skill
Use this skill when `repair_origin_stage=fix-harness` and crash triage classified the issue as `harness_bug`.

## Required inputs
- repo-root `crash_info.md`
- repo-root `crash_triage.json`
- repo-root `crash_analysis.md` when present; if absent on the crash-triage repair path, continue with explicit degraded reasoning
- `repair_error_digest` and `repair_recent_attempts`
- `fuzz/PLAN.md`, `fuzz/targets.json`, `fuzz/execution_plan.json`
- MCP tools from task-scoped PromeFuzz companion (if available)

## Required outputs
- updated `fuzz/PLAN.md`
- schema-valid `fuzz/targets.json`
- updated `fuzz/execution_plan.json`
- `Known Issues` section in `fuzz/PLAN.md`

## Workflow
1. Consume repo-root crash evidence and `repair_error_digest` first.
2. Identify harness misuse root cause and impacted `fuzz/` files.
3. Define one material strategy change versus the previous failed cycle.
4. Keep target mapping executable (`execution_plan` must stay mappable to harness files).

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done contract
- Write the path string `fuzz/PLAN.md` as the sole text of `./done` (run `echo 'fuzz/PLAN.md' > ./done`; do **not** copy the file's contents).

## Constraints
- `fuzz/PLAN.md` must include `Known Issues` with concrete blockers and next action.
- Treat repo-root crash artifact paths as authoritative; do not guess `fuzz/crash_*` paths.
- If `crash_analysis.md` is unavailable on this route, record `crash_analysis_not_available_yet` in `Known Issues` and proceed from `crash_info.md` + `crash_triage.json`.
- Plan must prioritize harness/build glue edits under `fuzz/`; doc-only/no-op is invalid.
- Always prefer public/stable APIs; do not default to internal/private symbols.
- If non-public API is unavoidable, require `api_surface_exception` with non-empty `reason` and `evidence`.
- If diagnostics include `non_public_api_usage`, plan must fix offending symbols first.
- When MCP is unavailable, explicitly record degraded reasoning in `fuzz/PLAN.md`.
