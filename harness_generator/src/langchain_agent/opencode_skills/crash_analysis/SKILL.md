# Stage Skill: crash_analysis

## Stage Goal

Analyze reproduced crash evidence and decide whether it is a false positive harness issue or a real upstream bug signal.

## Required Inputs

- `crash_info.md`
- `re_run_report.md`
- `crash_triage.json` (if present)
- coordinator hint/context

## Required Outputs

- `crash_analysis.json`
- `crash_analysis.md`
- `./done` with `crash_analysis.json`

`crash_analysis.json` minimal shape:

```json
{
  "verdict": "false_positive|real_bug|unknown",
  "reason": "short explanation",
  "confidence": 0.0,
  "signals": ["concrete log lines or findings"]
}
```

## Acceptance Criteria

- Verdict is exactly one of `false_positive`, `real_bug`, `unknown`.
- `reason` is non-empty and grounded in crash evidence.
- `signals` is a non-empty array of concrete evidence lines.
- Output is analysis-only; no source code changes in this stage.

## Command Policy

- Allowed: read-only commands (`find`, `grep`, `rg`, `cat`, `ls`, `sed`, `awk`, `head`, `tail`).
- Forbidden: build or execution commands.

## Done Sentinel Contract

- Create `./done`.
- Write exactly `crash_analysis.json` into `./done`.
