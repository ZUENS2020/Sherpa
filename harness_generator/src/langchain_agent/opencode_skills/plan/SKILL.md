---
name: plan
description: Produce runtime-viable fuzz targets and execution plan artifacts for synthesize/build stages.
compatibility: opencode
metadata:
  stage: plan
  owner: sherpa
---

## What this skill does
Generate planning artifacts that choose practical targets and define execution priorities.

## When to use this skill
Use this skill in the `plan` stage for initial planning or re-planning.

## Required inputs
- `fuzz/target_analysis.json` (if present)
- `fuzz/antlr_plan_context.json` (if present)
- MCP tools from task-scoped PromeFuzz companion (if available), including preprocessor and semantic tools
  - code navigation: `list_definitions`, `read_definition`, `read_source`, `find_references`
  - preprocessor: `run_ast_preprocessor`, `extract_api_functions`, `build_library_callgraph`
  - semantic (if enabled): `init_knowledge_base`, `retrieve_documents`, `comprehend_*`
- repository source/build metadata

## Required outputs
- `fuzz/PLAN.md`
- `fuzz/targets.json`
- `fuzz/execution_plan.json`

## Workflow
1. Query MCP evidence first when MCP is available (code-navigation first, preprocessor second, semantic evidence third).
2. Read target analysis and identify runtime-viable public entrypoints.
3. Produce `fuzz/targets.json` as a strict non-empty array.
4. Produce `fuzz/execution_plan.json` with prioritized execution targets.
5. Write concise implementation guidance into `fuzz/PLAN.md`.

## Constraints
- In `fuzz/targets.json`, each item must include non-empty `name`, `api`, `lang`, `target_type`, `seed_profile`.
- `api` must describe an API identifier, not a harness path.
- Forbidden `api` examples: `fuzz/*.c`, `fuzz/*.cc`, `fuzz/*.cpp`, `fuzz/*.cxx`, `fuzz/*.java`.
- Forbidden: `name = LLVMFuzzerTestOneInput`.
- Rank runtime-executable/public targets first.
- `fuzz/execution_plan.json` must include `execution_priority`, `must_run`, `target_name`, `expected_fuzzer_name`, `seed_profile`.
- Naming contract to reduce target/binary mismatch:
  - `target_name` should be API-centric and suffix-free (for example: `decode`).
  - `expected_fuzzer_name` must map predictably to the harness/binary name (prefer `<target_name>_fuzz` or `<target_name>_fuzzer`).
  - Keep `expected_fuzzer_name` consistent with `fuzz/harness_index.json` and harness filename stem.
- Include `min_required_built_targets` (default >=2 when multiple execution targets exist).
- When diagnostics include concrete file paths, use `Read and fix <path>[:line]`.
- If MCP is unavailable, continue in degraded mode and explicitly note missing MCP evidence in `fuzz/PLAN.md`.

## Command policy
- Allowed: read-only commands only (`find`, `grep`, `rg`, `cat`, `ls`, `head`, `tail`, read-only `sed`).
- Forbidden: build/execute commands.

## Acceptance checklist
- `fuzz/PLAN.md` exists and references a concrete primary target.
- `fuzz/targets.json` is strict-schema valid and non-empty.
- `fuzz/execution_plan.json` is consistent with selected runtime targets.

## Done contract
- Write `fuzz/PLAN.md` into `./done`.
