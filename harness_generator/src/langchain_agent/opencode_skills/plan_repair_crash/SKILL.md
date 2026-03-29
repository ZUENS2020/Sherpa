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
- MCP tools from task-scoped PromeFuzz companion (if available)

## Required outputs
- updated `fuzz/PLAN.md`
- schema-valid `fuzz/targets.json`
- updated `fuzz/execution_plan.json`
- strategy note ensuring `fuzz/harness_index.json` remains mappable
- `Known Issues` section in `fuzz/PLAN.md`

## Workflow
1. Query MCP evidence first when MCP is available.
2. Read crash/repro diagnostics.
3. Explain selected target vs observed runtime/crash target relation.
4. Propose a strategy change if signatures repeat.
5. Update planning artifacts consistently.

## Constraints
- Avoid “repair” plans that only disable behavior.
- `fuzz/PLAN.md` must include `Known Issues` with unresolved crash/repro blockers and concrete next-step checks.
- Preserve crash-path reachability.
- Naming contract to reduce crash-target mismatch:
  - `target_name` uses API-centric suffix-free naming.
  - `expected_fuzzer_name` maps to real harness/binary stem (prefer `<target_name>_fuzz` / `<target_name>_fuzzer`).
  - Keep consistency with `fuzz/harness_index.json`.
- Default to public/stable APIs for harness logic.
- If internal APIs are unavoidable, require `api_surface_exception` with non-empty `reason` and `evidence`.
- If diagnostics contain `non_public_api_usage`, prioritize replacing offending symbols first.
- If MCP is unavailable, continue in degraded mode and explicitly record that in `fuzz/PLAN.md`.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Acceptance checklist
- Plan addresses crash-path diagnostics explicitly.
- Strategy change is present on repeated signatures.
- Target relation is explicit and technically justified.
- `Known Issues` exists and lists concrete unresolved crash risks for this cycle.

## Done contract
- Write `fuzz/PLAN.md` into `./done`.
