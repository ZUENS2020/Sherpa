---
name: plan
description: Produce runtime-viable fuzz targets and execution plan artifacts for synthesize/build stages.
compatibility: opencode
metadata:
  stage: plan
  owner: tianheng
---

## What this skill does
Generate planning artifacts that choose practical targets and define execution priorities.

## When to use this skill
Use this skill in the `plan` stage for initial planning or re-planning.

## Required inputs
- `fuzz/vuln_candidates.json` (if present; read before target scoring)
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
2. Read `fuzz/vuln_candidates.json` first when present. Treat pending high-priority candidates as the primary target source; coverage and complexity are only tie-breakers.
3. Read target analysis and identify runtime-viable entrypoints that can validate the top vulnerability candidates.
4. Apply pure vulnerability-driven scoring when selecting targets:
   - `score_total = 0.50*vuln_likelihood + 0.30*exploitability + 0.20*reachability_confidence - recent_yield_penalty`.
   - Coverage, complexity, API relevance, and consumer-order-support are NOT scored; they default to 0.
   - `vuln_likelihood` and `exploitability` dominate the ranking; all other dimensions are tie-breakers only.
5. Produce `fuzz/targets.json` as a strict non-empty array.
6. Produce `fuzz/execution_plan.json` with prioritized execution targets.
7. Write concise implementation guidance into `fuzz/PLAN.md`.

## Constraints
- In `fuzz/targets.json`, each item must include non-empty `name`, `api`, `lang`, `target_type`, `seed_profile`.
- Treat `fuzz/targets.json` as advisory candidate input. The coordinator will normalize target identity, seed profile, and execution metadata before writing `fuzz/selected_targets.json`.
- `api` must describe an API identifier, not a harness path.
- Forbidden `api` examples: `fuzz/*.c`, `fuzz/*.cc`, `fuzz/*.cpp`, `fuzz/*.cxx`, `fuzz/*.java`.
- Forbidden: `name = LLVMFuzzerTestOneInput`.
- Rank runtime-executable/public targets first.
- Keep vulnerability candidates primary; coverage/complexity are secondary references.
- If `fuzz/vuln_candidates.json` has pending candidates, `fuzz/PLAN.md` must name the chosen `candidate_id`, target API, evidence IDs, and attack hint.
- `fuzz/execution_plan.json` must include `execution_priority`, `must_run`, `target_name`, `expected_fuzzer_name`, `seed_profile`.
- Naming contract to reduce target/binary mismatch:
  - `target_name` should be API-centric and suffix-free (for example: `decode`).
  - `expected_fuzzer_name` must map predictably to the harness/binary name (prefer `<target_name>_fuzz` or `<target_name>_fuzzer`).
  - Keep `expected_fuzzer_name` consistent with `fuzz/harness_index.json` and harness filename stem.
- Include `min_required_built_targets` (default >=2 when multiple execution targets exist).
- `fuzz/selected_targets.json` must include `security_score_breakdown`.
- `fuzz/selected_targets.json` is the normalized execution truth source. Keep target identity stable and API-centric.
- Empty `seed_families_suggested` is valid when no specific advisory family guidance is justified.
- Internal/private API selection requires explicit `api_surface_exception`:
  - allow only when `vuln_likelihood >= 0.75`
  - include non-empty `reason` and `evidence_ids`.
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
- Write the path string `fuzz/PLAN.md` as the sole text of `./done` (run `echo 'fuzz/PLAN.md' > ./done`; do **not** copy the file's contents).
