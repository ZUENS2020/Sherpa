# Stage Skill: fix_crash_upstream_bug

## Stage Goal
Fix the upstream bug so the same crashing input no longer reproduces the fault.

## Required Inputs
- `crash_info.md`
- `crash_analysis.md`
- crashing artifact metadata from coordinator

## Required Outputs
- minimal correctness/security fix for upstream cause

## Key File Templates
- patch should preserve harness behavior and keep crash reproduction semantics valid
- avoid disabling checks or bypassing parsing paths

## Acceptance Criteria
- fix addresses root cause in upstream code path.
- no broad refactor or behavior masking.
- clear, minimal code delta.
- must produce textual code changes; pure no-op is invalid.
- do not bypass acceptance by tampering with `fuzz/repo_understanding.json` semantics.
- when diagnostics include concrete file paths, issue explicit actions as `Read and fix <path>[:line]`.

## Command Policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done Sentinel Contract
- write the key modified path into `./done`.
