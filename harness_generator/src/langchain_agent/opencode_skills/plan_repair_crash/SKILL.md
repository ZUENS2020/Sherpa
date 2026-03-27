---
name: plan_repair_crash
description: Re-plan after crash/repro failures while preserving crash-path reachability and target relation clarity.
compatibility: opencode
metadata:
  stage: plan-repair-crash
  owner: sherpa
---

## What this skill does
Repairs planning artifacts for crash/repro recovery cycles.

## When to use this skill
Use this skill when workflow is in repair mode with `repair_origin_stage` related to crash/repro stages.

## Required inputs
- `repair_*` diagnostics
- `repair_error_digest` (if provided)
- crash/repro summaries and report tails (if provided)
- `fuzz/PLAN.md`, `fuzz/targets.json`, `fuzz/execution_plan.json` (if present)

## Required outputs
- updated `fuzz/PLAN.md`
- schema-valid `fuzz/targets.json`
- updated `fuzz/execution_plan.json`
- strategy note ensuring `fuzz/harness_index.json` remains mappable

## Workflow
1. Read crash/repro diagnostics first.
2. Explain selected target vs observed runtime/crash target relation.
3. Propose a strategy change if signatures repeat.
4. Update planning artifacts consistently.

## Constraints
- Avoid “repair” plans that only disable behavior.
- Preserve crash-path reachability.
- Default to public/stable APIs for harness logic.
- If internal APIs are unavoidable, require `api_surface_exception` with non-empty `reason` and `evidence`.
- If diagnostics contain `non_public_api_usage`, prioritize replacing offending symbols first.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Acceptance checklist
- Plan addresses crash-path diagnostics explicitly.
- Strategy change is present on repeated signatures.
- Target relation is explicit and technically justified.

## Done contract
- Write `fuzz/PLAN.md` into `./done`.
