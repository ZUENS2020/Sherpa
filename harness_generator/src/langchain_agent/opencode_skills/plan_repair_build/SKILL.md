---
name: plan_repair_build
description: Re-plan targets and scaffold strategy after build-stage failures using diagnostic-first reasoning.
compatibility: opencode
metadata:
  stage: plan-repair-build
  owner: sherpa
---

## What this skill does
Repairs planning artifacts for build recovery while keeping execution targets mappable and runtime-viable.

## When to use this skill
Use this skill when the workflow is in repair mode with `repair_origin_stage=build`.

## Required inputs
- `repair_*` diagnostics from coordinator context
- `repair_error_digest` (if provided)
- `fuzz/PLAN.md`, `fuzz/targets.json`, `fuzz/execution_plan.json` (if present)
- MCP tools from task-scoped PromeFuzz companion (if available), including preprocessor and semantic tools

## Required outputs
- updated `fuzz/PLAN.md`
- schema-valid `fuzz/targets.json`
- updated `fuzz/execution_plan.json`
- strategy note that keeps `fuzz/harness_index.json` mappable
- `Known Issues` section in `fuzz/PLAN.md`

## Workflow
1. Query MCP evidence first when MCP is available (preprocessor first, semantic evidence second).
2. Read repair diagnostics.
3. Identify root build failure pattern (compile/link/toolchain/path).
4. Produce planning changes with at least one strategy change.
5. Keep execution targets runtime-viable and mappable.

## Constraints
- Do not produce doc-only updates disconnected from build recovery.
- `fuzz/PLAN.md` must include `Known Issues` with current build blockers, suspected root cause, and next corrective action.
- Include compiler-selection strategy in plan notes: `.c -> clang`, `.cc/.cpp/.cxx -> clang++`.
- Explicitly reject universal `clang++` for mixed C/C++ harness builds unless evidence proves it is required.
- Naming contract to reduce undercoverage false negatives:
  - `target_name` remains suffix-free API name.
  - `expected_fuzzer_name` should map to real binary stem (prefer `<target_name>_fuzz` or `<target_name>_fuzzer`).
  - Ensure naming remains consistent with `fuzz/harness_index.json` mappings.
- Default to public/stable APIs for harness logic.
- If non-public/internal API is unavoidable, require `api_surface_exception` in `fuzz/repo_understanding.json` with non-empty `reason` and `evidence`.
- If diagnostics contain `non_public_api_usage`, prioritize replacing offending symbols first.
- If MCP is unavailable, continue in degraded mode and explicitly record that in `fuzz/PLAN.md`.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Acceptance checklist
- Plan explicitly addresses current build failure kind/code/signature.
- Plan includes at least one strategy change from latest failed attempt.
- Targets remain runtime-viable and executable-first.
- `Known Issues` exists and names concrete unresolved blockers for this build-repair cycle.

## Done contract
- Write `fuzz/PLAN.md` into `./done`.
