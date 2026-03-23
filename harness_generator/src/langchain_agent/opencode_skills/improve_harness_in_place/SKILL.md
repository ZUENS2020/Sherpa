# Stage Skill: improve_harness_in_place

## Stage Goal
Apply in-place coverage improvements for the current target without switching targets.

## Required Inputs
- coordinator coverage diagnostics (`coverage_*`, `repair_*`, `codex_hint`)
- `SeedFeedback` and `HarnessFeedback` blocks from coordinator context (when provided)
- current files under `fuzz/`
- `fuzz/execution_plan.json` (if present)
- `fuzz/harness_index.json` (if present)

## Required Outputs
- material code/scaffold updates under `fuzz/` for coverage improvement
- consistent `fuzz/execution_plan.json`, harness sources, and `fuzz/harness_index.json`
- no doc-only patch in this stage

## Acceptance Criteria
- edits address concrete coverage diagnostic gaps first (seed family gaps, input modeling, dictionary/corpus strategy, call ordering/path depth).
- edits consume `SeedFeedback`/`HarnessFeedback` first and include at least one directly mapped code/scaffold change.
- keep current target identity stable for this stage (no target replacement).
- include at least one strategy change versus the latest failed in-place cycle.
- resulting scaffold remains ready for the next workflow build/run.

## Command Policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done Sentinel Contract
- write `fuzz/out/` into `./done`.
