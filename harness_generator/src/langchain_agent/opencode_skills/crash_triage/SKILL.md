# Stage Skill: crash_triage

## Stage Goal
Classify the crash root cause into exactly one label using crash evidence only:
- `harness_bug`
- `upstream_bug`
- `inconclusive`

This stage is classification-only. Do not patch source code here.

## Required Inputs
- `crash_info.md` (if present)
- `crash_analysis.md` (if present)
- `re_build_report.md` / `re_run_report.md` tails (if present)
- runtime fields from coordinator: `last_fuzzer`, `last_crash_artifact`, `crash_signature`

## Required Outputs
- `crash_triage.json` with non-empty fields:
  - `label` (`harness_bug|upstream_bug|inconclusive`)
  - `confidence` (0.0-1.0)
  - `reason` (short English sentence)
  - `evidence` (non-empty string array with concrete log/report signals)

## Acceptance Criteria
- label is exactly one of the three allowed values.
- reason is English and references observed signals.
- evidence is non-empty and points to concrete lines/patterns.
- classification is conservative when uncertain (`inconclusive`).
- no source files are modified.

## Command Policy
- Allowed: read-only commands only (`find`, `grep`, `rg`, `cat`, `ls`, `sed -n`, `head`, `tail`).
- Forbidden: build, run, execute, package install, or any mutating command.

## Done Sentinel Contract
- Must create `./done`.
- `./done` must contain exactly one line: `crash_triage.json`.
