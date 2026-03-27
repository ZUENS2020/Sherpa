---
name: crash_analysis
description: Analyze reproduced crash evidence and produce a structured verdict for workflow routing.
compatibility: opencode
metadata:
  stage: crash-analysis
  owner: sherpa
---

## What this skill does
This skill performs analysis-only crash verdicting after repro, without patching source code.

## When to use this skill
Use this skill in the `crash-analysis` stage after `re-run` has produced crash evidence.

## Required inputs
- `crash_info.md`
- `re_run_report.md`
- `crash_triage.json` (if present)
- coordinator hint/context

## Required outputs
- `crash_analysis.json`
- `crash_analysis.md`

`crash_analysis.json` must follow this minimal shape:
```json
{
  "verdict": "false_positive|real_bug|unknown",
  "reason": "short explanation",
  "confidence": 0.0,
  "signals": ["concrete log lines or findings"]
}
```

## Workflow
1. Read crash and repro artifacts first.
2. Correlate triage label with concrete log evidence.
3. Write `crash_analysis.json` and `crash_analysis.md`.
4. Do not modify source files in this stage.

## Constraints
- Analysis-only stage; no code edits.
- Prefer explicit evidence lines over generic conclusions.
- Use conservative verdict `unknown` when evidence is weak.

## Command policy
- Allowed: read-only commands (`find`, `grep`, `rg`, `cat`, `ls`, `sed`, `awk`, `head`, `tail`).
- Forbidden: build or execution commands.

## Acceptance checklist
- `verdict` is exactly one of `false_positive`, `real_bug`, `unknown`.
- `reason` is non-empty and grounded in observed evidence.
- `signals` is a non-empty string array with concrete findings.
- Output remains analysis-only.

## Done contract
- Create `./done`.
- Write exactly `crash_analysis.json` into `./done`.
