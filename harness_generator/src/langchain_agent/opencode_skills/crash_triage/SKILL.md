---
name: crash_triage
description: Classify reproduced crashes into harness bug, upstream bug, or inconclusive using evidence only.
compatibility: opencode
metadata:
  stage: crash-triage
  owner: sherpa
---

## What this skill does
This skill performs classification-only crash triage and outputs a structured label for downstream routing.

## When to use this skill
Use this skill in the `crash-triage` stage after `run` or `re-run` crash evidence is available.

## Required inputs
- `crash_info.md` (if present)
- `crash_analysis.md` (if present)
- `re_build_report.md` / `re_run_report.md` tails (if present)
- runtime fields from coordinator: `last_fuzzer`, `last_crash_artifact`, `crash_signature`

## Required outputs
- `crash_triage.json` with non-empty fields:
  - `label` (`harness_bug|upstream_bug|inconclusive`)
  - `confidence` (0.0-1.0)
  - `reason` (short English sentence)
  - `evidence` (non-empty string array with concrete signals)

## Workflow
1. Read crash artifacts and report tails.
2. Identify whether root-cause evidence points to harness, upstream, or remains inconclusive.
3. Write `crash_triage.json` with concise reason and evidence.
4. Do not patch code in this stage.

## Constraints
- Classification-only; no source edits.
- Prefer conservative classification when uncertain.
- Keep reason/evidence grounded in observed logs and traces.

## Command policy
- Allowed: read-only commands only (`find`, `grep`, `rg`, `cat`, `ls`, `sed -n`, `head`, `tail`).
- Forbidden: build, run, execute, package install, or any mutating command.

## Acceptance checklist
- `label` is exactly one of `harness_bug`, `upstream_bug`, `inconclusive`.
- `reason` is English and tied to concrete signals.
- `evidence` is non-empty and traceable.
- No source files are modified.

## Done contract
- Create `./done`.
- Write exactly `crash_triage.json` into `./done`.
