# Stage Skill: synthesize_repair_build

## Stage Goal
Repair scaffold files under `fuzz/` for build-stage failures.

## Required Inputs
- `repair_*` diagnostics from coordinator context
- current scaffold files under `fuzz/`
- `fuzz/execution_plan.json` (if present)

## Required Outputs
- harness source under `fuzz/`
- `fuzz/build.py` or `fuzz/build.sh`
- `fuzz/README.md`
- `fuzz/repo_understanding.json`
- `fuzz/build_strategy.json`
- `fuzz/build_runtime_facts.json`

## Acceptance Criteria
- edits are build-failure-driven (compile/link/toolchain/path), not cosmetic.
- when signatures repeat, this round must use a different repair strategy.
- no doc-only no-op patches; scaffold must materially change where needed.
- keep selected/final target and build strategy fields consistent across README and JSON files.
- public/stable APIs are mandatory by default in harness code.
- if non-public/internal API is unavoidable, require `api_surface_exception` in `fuzz/repo_understanding.json` with non-empty `reason` and `evidence` (optional `approved_symbols`).
- when diagnostics contain `non_public_api_usage`, replace offending symbols first before any unrelated edits.

## Command Policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done Sentinel Contract
- write `fuzz/out/` into `./done`.
