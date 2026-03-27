---
name: plan_fix_targets_schema
description: Repair targets.json into strict schema-valid shape without losing planning semantics.
compatibility: opencode
metadata:
  stage: plan-fix-targets-schema
  owner: sherpa
---

## What this skill does
Fixes `fuzz/targets.json` schema violations while preserving target intent.

## When to use this skill
Use this skill when coordinator reports `targets.json` schema errors.

## Required inputs
- current `fuzz/targets.json`
- schema error text from coordinator

## Required outputs
- fixed `fuzz/targets.json`

## Workflow
1. Parse schema diagnostics and locate invalid fields.
2. Repair only the invalid fields while preserving valid planning data.
3. Re-check array shape and required enums.

## Constraints
- `fuzz/targets.json` must be a non-empty JSON array.
- Each item must include `name`, `api`, `lang`, `target_type`, `seed_profile`.
- `lang` must be in `c-cpp|cpp|c|c++|java`.
- `target_type` must be in `parser|decoder|archive|image|document|network|database|serializer|interpreter|generic`.
- `seed_profile` must be in `parser-structure|parser-token|parser-format|parser-numeric|decoder-binary|archive-container|serializer-structured|document-text|network-message|generic`.
- Forbidden: `name = LLVMFuzzerTestOneInput`.
- Semantic reminder: do not rewrite `api` to harness paths like `fuzz/*.cc`.
- When diagnostics include concrete file paths, use `Read and fix <path>[:line]`.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Acceptance checklist
- JSON parses successfully.
- Schema fields and enum values are valid for all entries.
- Array remains non-empty.

## Done contract
- Write `fuzz/targets.json` into `./done`.
