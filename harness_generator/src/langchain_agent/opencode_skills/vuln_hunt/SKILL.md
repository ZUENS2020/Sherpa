---
name: vuln_hunt
description: Discover, update, and rank vulnerability candidates before execution planning.
compatibility: opencode
metadata:
  stage: vuln_hunt
  owner: tianheng
---

## What this skill does
Generate vulnerability-first advisory candidates that `plan` can materialize into executable fuzz targets.

## When to use this skill
Use this skill in the internal hunt subphase before `plan` materializes `selected_targets.json` and `execution_plan.json`.

## Required inputs
- `fuzz/analysis_context.json`, especially `analysis_evidence.security_evidence[]`
- `fuzz/vuln_candidates.json` when present
- recent coverage, crash, repair, and seed feedback when provided

## Required outputs
- `fuzz/vuln_candidates.json`
- `fuzz/vuln_hunt_summary.md`

## Workflow
1. Read `fuzz/analysis_context.json` and identify evidence-backed vulnerability hypotheses.
2. Read existing `fuzz/vuln_candidates.json` and preserve useful candidate state such as `validation_status`, `attempt_count`, and `last_result`.
3. Rank candidates by vulnerability risk first; coverage and complexity are only supporting evidence.
4. Change focus when validation feedback shows plateau, false positive, exhausted candidate, or repeated harness failure.
5. Write a concise `fuzz/vuln_hunt_summary.md` explaining top candidates and strategy changes.

## Constraints
- Candidate output is advisory. Do not write workflow control fields directly.
- Do not change `workflow_context`, `selected_targets.json`, or `execution_plan.json`.
- Evidence references should point to `analysis_evidence.security_evidence[].evidence_id` when available.
- Do not reclassify `target_type` or `seed_profile`; those are normalized by the coordinator.
- Keep `validation_status` within `pending`, `validating`, `validated`, `inconclusive`, `exhausted`, or `cooling`.

## Command policy
- Allowed: read-only commands only (`find`, `grep`, `rg`, `cat`, `ls`, `head`, `tail`, read-only `sed`).
- Forbidden: build/execute commands.

## Acceptance checklist
- `fuzz/vuln_candidates.json` is valid JSON with a `candidates` array.
- Every candidate has `candidate_id`, target API/name, risk type, priority or risk scores, evidence references, attack hint, validation status, attempt count, and last result.
- `fuzz/vuln_hunt_summary.md` exists and describes the top candidate rationale.

## Done contract
- Write `fuzz/vuln_hunt_summary.md` into `./done`.
