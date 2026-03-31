---
name: analysis
description: Generate pre-plan analysis context artifacts for target selection and repair-aware planning.
compatibility: opencode
metadata:
  stage: analysis
  owner: sherpa
---

## What this skill does
Builds analysis context files consumed by downstream `plan` and `synthesize` stages.

## When to use this skill
Use this skill in the dedicated `analysis` stage before `plan`.

## Required inputs
- repository source tree (read-only)
- MCP tools from task-scoped PromeFuzz companion (HTTP MCP), when available
- preferred MCP tools in this round:
  - preprocessor: `run_ast_preprocessor`, `extract_api_functions`, `build_library_callgraph`
  - semantic (if enabled): `init_knowledge_base`, `retrieve_documents`, `comprehend_*`
- optional companion outputs under `/shared/output/_k8s_jobs/<job-id>/promefuzz/` as fallback
- previous repair context from coordinator hint (if provided)

## Required inputs (additional)
- `fuzz/target_analysis.json` (preliminary target analysis from regex/tree-sitter heuristics)

## Required outputs
- `fuzz/analysis_context.json`
- `fuzz/antlr_plan_context.json` (if grammar/static context is available)
- `fuzz/target_analysis.json` (preliminary - do NOT reclassify target types here)

## Workflow
1. Query MCP evidence first when MCP is available.
   Use preprocessor tools first, then semantic tools for evidence-backed summaries.
2. Read existing analysis artifacts (if any) and companion file outputs as fallback.
3. Refresh static analysis summaries for grammar/target context.
4. Update `fuzz/analysis_context.json` with concise evidence from MCP and static analysis.
5. Do NOT reclassify `target_type` or `seed_profile` in this stage — that will be done by the seed generation stage with full function code context.
6. Ensure downstream plan can consume paths and summaries directly.

## Note on target_type classification
The preliminary target analysis in `fuzz/target_analysis.json` uses regex/tree-sitter heuristics.
**Do not reclassify target_type or seed_profile here.**
The seed generation stage has access to actual function code and will make final decisions on seed format.

## Command policy
- Allowed: read-only commands (`find`, `rg`, `grep`, `cat`, `head`, `tail`, read-only `sed`, `ls`).
- Forbidden: build/execute/mutation commands.

## Degraded mode
- If MCP is unavailable or returns invalid output, continue using local/static evidence.
- If semantic MCP tools are unavailable, continue with preprocessor evidence and mark degraded reason.
- Record degraded reason in `fuzz/analysis_context.json` instead of silently skipping MCP evidence.

## Done contract
- Write `fuzz/analysis_context.json` into `./done`.