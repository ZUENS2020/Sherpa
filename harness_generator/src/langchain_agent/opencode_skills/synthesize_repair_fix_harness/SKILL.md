---
name: synthesize_repair_fix_harness
description: Repair harness/build glue for crash-triaged harness bugs using evidence-first synthesis.
compatibility: opencode
metadata:
  stage: synthesize-repair-fix-harness
  owner: tianheng
---

## What this skill does
Produces concrete `fuzz/` code changes to fix harness bugs while preserving execution-plan mapping.

## When to use this skill
Use this skill when `repair_origin_stage=fix-harness` after crash triage labeled the crash as `harness_bug`.

## Required inputs
- repo-root `crash_info.md`
- repo-root `crash_triage.json`
- repo-root `crash_analysis.md` when present; if absent on the crash-triage repair path, continue with explicit degraded reasoning
- `repair_error_digest` and recent repair attempts
- current `fuzz/` scaffold files
- MCP tools from task-scoped PromeFuzz companion (if available)

## Required outputs
- updated harness/build glue files under `fuzz/`
- `fuzz/harness_index.json` aligned to `fuzz/execution_plan.json`
- consistent `fuzz/README.md`, `fuzz/repo_understanding.json`, and `fuzz/build_strategy.json`

## Workflow
1. Read repo-root crash evidence first and identify the failing harness path.
2. Query MCP evidence first when available (preprocessor first, semantic evidence second).
3. Apply one material strategy change compared with the previous failed cycle.
4. Edit offending harness/build glue files and keep target mapping consistent.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done contract
- Write the path string `fuzz/out/` as the sole text of `./done` (run `echo 'fuzz/out/' > ./done`; do **not** copy the file's contents).

## Constraints
- Must modify executable `fuzz/` code paths; doc-only/no-op patches are invalid.
- Treat repo-root crash artifact paths as authoritative; do not guess `fuzz/crash_*` paths.
- If `crash_analysis.md` is unavailable on this route, proceed from `crash_info.md` + `crash_triage.json` and document `crash_analysis_not_available_yet` in `fuzz/repo_understanding.json`.
- Preserve libFuzzer entry contract:
  - no custom `main()` in harness source
  - use `LLVMFuzzerTestOneInput` (or language-equivalent fuzz entrypoint only)
- Forbid argv/file-driven harness entry logic (`fopen(argv[1], ...)`, `read(argv[1], ...)`, manual corpus file loops).
- Prefer public/stable APIs; internal/private APIs require explicit `api_surface_exception` evidence.
- If diagnostics include `non_public_api_usage`, replace offending symbols first.
- Keep `execution_plan` and `harness_index` naming consistent; avoid drift/mismatch.
- If MCP is unavailable, proceed in degraded mode and document degraded reason in `fuzz/repo_understanding.json`.
