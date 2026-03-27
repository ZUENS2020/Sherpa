---
name: fix_crash_harness_error
description: Repair harness-side crash causes while preserving target behavior and repro semantics.
compatibility: opencode
metadata:
  stage: fix-crash-harness-error
  owner: sherpa
---

## What this skill does
Apply targeted harness/build-glue fixes for crashes classified as harness bugs.

## When to use this skill
Use this skill only when crash triage/analysis indicates `harness_bug`.

## Required inputs
- `crash_info.md`
- `crash_analysis.md` indicating harness-side fault
- crash artifact metadata from coordinator context

## Required outputs
- minimal harness/build-glue patch under `fuzz/`

## Workflow
1. Read crash diagnostics and identify harness misuse/precondition violations.
2. Apply a minimal patch in `fuzz/` files.
3. Keep crash-path semantics and target behavior intact.

## Constraints
- Focus on `fuzz/` harness/build glue files.
- No unrelated refactor.
- Avoid upstream project source edits unless strictly required.
- Must produce textual code changes; pure no-op is invalid.
- Do not bypass acceptance by tampering with `fuzz/repo_understanding.json`.
- Use explicit actions when diagnostics include concrete paths: `Read and fix <path>[:line]`.
- Prefer public/stable APIs; replace internal/private symbols first.
- If no public alternative exists, add `api_surface_exception` with non-empty `reason` and `evidence`.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Acceptance checklist
- Patch addresses harness-side root cause directly.
- Changes are minimal and evidence-driven.
- No semantic bypass and no doc-only/no-op patch.

## Done contract
- Write one key modified path into `./done`.
