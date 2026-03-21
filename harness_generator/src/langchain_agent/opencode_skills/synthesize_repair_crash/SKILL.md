# Stage Skill: synthesize_repair_crash

## Stage Goal
Repair scaffold files under `fuzz/` for crash/repro-stage failures.

## Required Inputs
- `repair_*` diagnostics from coordinator context
- crash/repro report tail and related error text (if provided)
- current scaffold files under `fuzz/`

## Required Outputs
- harness source under `fuzz/`
- `fuzz/build.py` or `fuzz/build.sh`
- `fuzz/README.md`
- `fuzz/repo_understanding.json`
- `fuzz/build_strategy.json`
- `fuzz/build_runtime_facts.json`

## Acceptance Criteria
- edits are tied to crash/repro evidence and keep crash-path reachability.
- selected vs final runtime target relation is explicit and technically justified.
- when signatures repeat, this round must change strategy.
- no “fix” by disabling harness behavior or deleting crash-relevant code paths.

## Command Policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done Sentinel Contract
- write `fuzz/out/` into `./done`.
