---
name: synthesize_repair_crash
description: Repair scaffold for crash/repro failures while preserving crash-path reachability.
compatibility: opencode
metadata:
  stage: synthesize-repair-crash
  owner: sherpa
---

## What this skill does
Repairs harness/scaffold after crash/repro failures with crash evidence as the primary signal.

## When to use this skill
Use this skill in repair mode for crash/repro failures.

## Required inputs
- `repair_*` diagnostics from coordinator context
- crash/repro report tail and related error text (if provided)
- current scaffold files under `fuzz/`
- `fuzz/execution_plan.json` (if present)
- MCP tools from task-scoped PromeFuzz companion (if available), including preprocessor and semantic tools

## Required outputs
- updated harness/scaffold files under `fuzz/`
- `fuzz/harness_index.json` aligned to execution plan

## Workflow
1. Query MCP evidence first when MCP is available (preprocessor first, semantic evidence second).
2. Consume crash/repro evidence.
3. Apply focused scaffold/harness repair for crash-path stability.
4. Keep selected vs final runtime target relation explicit.
5. Ensure strategy change when repeated signatures occur.

## Constraints
- Do not “fix” by disabling harness behavior or deleting crash-relevant paths.
- Public/stable APIs are mandatory by default.
- If non-public API is unavoidable, require `api_surface_exception` with non-empty `reason` and `evidence`.
- If diagnostics contain `non_public_api_usage`, replace offending symbols first.
- LibFuzzer harness contract is mandatory:
  - do not define custom `main()` in harness source;
  - use `LLVMFuzzerTestOneInput` (or language-equivalent fuzz entrypoint) as the only fuzz entry.
- Forbid argv/file-driven harness entry logic in libFuzzer mode (`fopen(argv[1], ...)`, `read(argv[1], ...)`, manual corpus file loops).
- If MCP is unavailable, continue in degraded mode and record this in `fuzz/repo_understanding.json`.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Acceptance checklist
- Edits are crash-evidence-driven.
- Strategy changes on repeated signatures.
- Execution target mapping remains valid.

## Done contract
- Write `fuzz/out/` into `./done`.
