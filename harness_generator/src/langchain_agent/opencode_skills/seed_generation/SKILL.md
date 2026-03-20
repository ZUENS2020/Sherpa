# Stage Skill: seed_generation

## Stage Goal
Generate high-signal warm-up corpus files for the current harness, with real repository samples prioritized over synthetic noise.

## Required Inputs
- `fuzz/observed_target.json` (when present)
- `fuzz/selected_targets.json`
- `fuzz/target_analysis.json`
- harness source under `fuzz/`
- current corpus directory for the active fuzzer

## Required Outputs
- seed files under `fuzz/corpus/<fuzzer_name>/`
- `seed_exploration_<fuzzer>.json`
- `seed_check_<fuzzer>.json`

## Key File Templates
- For `archive-container` profiles:
  - import real archive samples first from:
    - `contrib/oss-fuzz/corpus.zip`
    - `contrib/oss-fuzz/**`
    - `test/**` or `tests/**`
  - Avoid hand-crafted magic-only files (header bytes without valid archive structure)
  - keep malformed/truncated seeds <= 30% of corpus
  - ensure at least one semantically valid archive sample exists
- `seed_exploration_*.json` and `seed_check_*.json` must follow coordinator-required JSON keys exactly.

## Acceptance Criteria
- required family buckets are covered or explicitly documented as missing with reason.
- corpus is not dominated by malformed archive samples.
- archive-focused corpus includes valid real samples before synthetic edge cases.
- when diagnostics/context include concrete file paths, issue explicit actions as `Read and fix <path>[:line]`.

## Command Policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done Sentinel Contract
- write one created/updated seed file path into `./done`.
