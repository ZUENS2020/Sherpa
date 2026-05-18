---
name: analysis
description: Generate pre-plan analysis context artifacts for target selection and repair-aware planning.
compatibility: opencode
metadata:
  stage: analysis
  owner: tianheng
---

## What this skill does
Builds analysis context files consumed by downstream `plan` and `synthesize` stages.

## When to use this skill
Use this skill in the dedicated `analysis` stage before `plan`.

## Required inputs
- repository source tree (read-only)
- MCP tools from task-scoped PromeFuzz companion (HTTP MCP), when available
- preferred MCP tools in this round:
  - code navigation: `list_definitions`, `read_definition`, `read_source`, `find_references`
  - preprocessor: `run_ast_preprocessor`, `extract_api_functions`, `build_library_callgraph`
  - semantic (if enabled): `init_knowledge_base`, `retrieve_documents`, `comprehend_*`
- optional companion outputs under `/shared/output/_k8s_jobs/<job-id>/promefuzz/` as fallback
- previous repair context from coordinator hint (if provided)

## Required inputs (additional)
- `fuzz/target_analysis.json` (preliminary target analysis from regex/tree-sitter heuristics)

## Required outputs
- `fuzz/analysis_context.json` (system-generated input/output marker; read it, but do not rewrite it wholesale)
- `fuzz/vuln_hypotheses.md` (concise AI advisory notes)
- `fuzz/antlr_plan_context.json` (if grammar/static context is available)
- `fuzz/target_analysis.json` (preliminary - do NOT reclassify target types here)
- `fuzz/analysis_context.json.analysis_evidence.security_evidence`
- `fuzz/analysis_context.json.analysis_evidence.vuln_candidate_inventory`
- `VULN_HYPOTHESES` section with evidence-linked risk hypotheses

## Bounded analysis mode
- This stage must finish promptly. Do not perform open-ended vulnerability research.
- After reading required files, use at most 6 additional MCP/tool reads in the first pass.
- Prefer existing system-generated `security_evidence[]` and `vuln_candidates.json`; treat them as sufficient unless they are empty or corrupt.
- Do not call semantic/comprehension MCP tools in the first pass unless the coordinator explicitly asks for semantic enrichment.
- After one bounded evidence pass, write `fuzz/vuln_hypotheses.md` and `./done` immediately. Do not keep exploring after writing a coherent top 3-8 hypothesis set.
- Keep `fuzz/vuln_hypotheses.md` under 120 lines.

## Workflow
1. Query MCP evidence first when MCP is available.
   Use code-navigation tools first to locate concrete symbols/definitions, then preprocessor tools only if needed.
2. Read existing analysis artifacts (if any) and companion file outputs as fallback.
3. Refresh static analysis summaries for grammar/target context.
4. Do not rewrite the full `fuzz/analysis_context.json`; it is a system-generated fact artifact and can be large.
   Treat it as read-only input unless you need a tiny, surgical metadata correction.
5. Do not rewrite `fuzz/vuln_candidates.json`; it is a system-generated machine-readable worklist.
   Treat it as read-only input and summarize only the highest-signal candidate IDs in advisory prose.
6. Write concise AI advisory findings to `fuzz/vuln_hypotheses.md`.
   Include vulnerability evidence fields:
   - `security_evidence[]` entries with `evidence_id`, `signal_id`, `severity`, `confidence`, `source_path`, `line`, `summary`.
   - `vuln_candidate_inventory[]` entries with `candidate_id`, `api`, `file`, `target_type`, `vuln_likelihood`, `exploitability`, `reachability_confidence`, `evidence_ids`.
   - summary counters: `security_evidence_count`, `vuln_candidate_count`, `security_mode`, `vuln_focus_profile`, `target_surface_policy`.
   Keep this file small: prefer the top 3-8 high-signal hypotheses, not a full copy of `analysis_context.json`.
7. Add `VULN_HYPOTHESES` in analysis notes/output and ensure every hypothesis cites existing `evidence_id` values.
8. Do NOT reclassify `target_type` or `seed_profile` in this stage — that will be done by the seed generation stage with full function code context.
9. Ensure downstream plan can consume paths and summaries directly.

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
- Record degraded reason in `fuzz/vuln_hypotheses.md` instead of silently skipping MCP evidence.

## Companion state interpretation
The PromeFuzz companion's `status.json` may show `state: waiting_repo_root` during initialization. **This is NOT an error or degraded condition** — it means the companion is running normally and waiting for analysis to begin. Do NOT report this as a degraded reason. Only report degraded reason when MCP tools actually fail or return errors.

## Done contract
- Write the path string `fuzz/analysis_context.json` as the sole text of `./done` (run `echo 'fuzz/analysis_context.json' > ./done`; do **not** copy the file's contents).
