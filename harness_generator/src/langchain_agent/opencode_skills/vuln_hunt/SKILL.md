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
3. Read per-iteration fuzz feedback from `fuzz/run_feedback.json` and `fuzz/vuln_hunt_events.jsonl` to refine confidence scores with actual runtime data.
4. Rank candidates **purely by vulnerability risk** (likelihood + exploitability + reachability). Coverage, complexity, and API surface metrics are NOT used in ranking.
5. Calibrate analysis depth per iteration (read `vuln_hunt_iteration` from workflow context):
   - Iteration 1-2: broad candidates from static analysis of all reachable APIs.
   - Iteration 3-4: narrow to candidates with crash/coverage evidence from fuzz runs; increase exploitability estimates.
   - Iteration 5+: re-evaluate exhausted candidates against fresh data; cross-reference with actual crash signatures.
6. **Increase confidence scores each iteration** as supporting evidence accumulates. A candidate that survives multiple iterations with consistent crash reproduction should have confidence approaching 1.0. A candidate with no crash after multiple iterations should be cooled.
7. Change focus aggressively when validation feedback shows plateau, false positive, exhausted candidate, or repeated harness failure — do not keep the same top candidate when evidence weakens.
8. Write a concise `fuzz/vuln_hunt_summary.md` explaining top candidates, strategy changes, and iteration-specific findings.

## Constraints
- Do not promote candidates from `test/`, `tests/`, `demo/`, `demos/`, `examples/`, `example/`, `deprecated/`, `legacy/`, or `contrib/` directories to high priority.
  Mark such candidates with `validation_status=degraded_test_code` (test/demo) or `validation_status=deprecated` (deprecated/legacy).
  Only raise their priority when crash evidence from a production build confirms reachability.
- Do not promote candidates whose target function name indicates cleanup / resource management.
  Mark with `validation_status=degraded_cleanup` when the function name matches:
  `Delete`, `Dealloc`, `Deallocate`, `Free`, `Destroy`, `Cleanup`, `Release`, `Dispose`, `Close`, `Uninit`.
  These are lifecycle helpers, not exploitable attack surface. Override only when crash evidence
  from production builds confirms a real vulnerability.
- Do not promote candidates whose target function name matches generic test-helper patterns:
  `MyRead`, `MyWrite`, `ReadFunc`, `WriteFunc`, `HelperFunc`, `TestFunc`, `Dummy*`, `Fake*`.
  Mark with `validation_status=degraded_test_helper`. When a candidate is found in a test/ directory
  AND the name matches a test-helper pattern, prefer `degraded_test_code`.
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
- Confidence scores (vuln_likelihood, exploitability) increase or hold steady across iterations; decreases without explicit evidence are invalid.
- Top candidates change across iterations when supporting evidence weakens or new candidates emerge.
- `validation_status` is updated for candidates that appeared in recent fuzz/coverage feedback.

## Done contract
- Write the path string `fuzz/vuln_hunt_summary.md` as the sole text of `./done` (run `echo 'fuzz/vuln_hunt_summary.md' > ./done`; do **not** copy the file's contents).
