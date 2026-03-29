---
name: seed_generation
description: Generate high-signal seed corpus with real samples first and controlled synthetic expansion.
compatibility: opencode
metadata:
  stage: seed-generation
  owner: sherpa
---

## What this skill does
Creates warm-up corpus files and seed diagnostics for a target fuzzer, favoring valid and diverse inputs.

## When to use this skill
Use this skill during pre-run seed generation and seed repair cycles.

## Required inputs
- `fuzz/observed_target.json` (when present)
- `fuzz/selected_targets.json`
- `fuzz/target_analysis.json`
- harness source under `fuzz/`
- current corpus directory for active fuzzer

## Required outputs
- seed files under `fuzz/corpus/<fuzzer_name>/`
- `seed_exploration_<fuzzer>.json`
- `seed_check_<fuzzer>.json`

## Workflow
1. Explore target format and available repository examples.
2. Import real samples first, then add controlled synthetic variants.
3. Keep family coverage balanced and noise bounded.
4. Write required seed diagnostics JSON files.

## Constraints
- Global filtering defaults to `soft` mode:
  - preserve semantically distinct seeds
  - still reject oversized files and exact-content duplicates
  - avoid malformed-only growth when required families are missing
- For `archive-container`:
  - use real archive samples first (`contrib/oss-fuzz/corpus.zip`, `contrib/oss-fuzz/**`, `test/**`, `tests/**`)
  - avoid hand-crafted magic-only files
  - keep malformed/truncated seeds <= 30%
  - ensure at least one semantically valid archive sample exists
- `seed_exploration_*.json` and `seed_check_*.json` must follow coordinator-required schema.
- When diagnostics include concrete paths, use `Read and fix <path>[:line]`.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Acceptance checklist
- Required family buckets are covered or explicitly documented with reason.
- Corpus is not dominated by malformed archive samples.
- Valid real samples are present before synthetic edge cases.

## Done contract
- Write one created/updated seed file path into `./done`.
