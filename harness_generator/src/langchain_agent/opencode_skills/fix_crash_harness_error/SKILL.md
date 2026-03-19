# Stage Skill: fix_crash_harness_error

## Stage Goal
Fix harness/build glue errors so the same crashing input no longer fails due to harness misuse.

## Required Inputs
- `crash_info.md`
- `crash_analysis.md` indicating harness error
- crashing artifact metadata from coordinator

## Required Outputs
- targeted harness/build glue fix

## Key File Templates
- focus on `fuzz/` harness/build glue files only
- keep changes minimal and directly tied to misuse/precondition violations

## Acceptance Criteria
- patch addresses harness-side root cause.
- no unrelated refactor.
- no upstream/project source modifications unless strictly required.

## Command Policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done Sentinel Contract
- write the key modified path into `./done`.
