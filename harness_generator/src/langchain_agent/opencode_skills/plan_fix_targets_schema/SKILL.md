# Stage Skill: plan_fix_targets_schema

## Stage Goal
Repair `fuzz/targets.json` so it passes strict schema validation.

## Required Inputs
- current `fuzz/targets.json`
- schema error text from coordinator

## Required Outputs
- fixed `fuzz/targets.json`

## Key File Templates
- `fuzz/targets.json`
  - JSON array (not wrapped object)
  - non-empty array
  - each object contains:
    - `name` (string)
    - `api` (string)
    - `lang` in `c-cpp|cpp|c|c++|java`
    - `target_type` in `parser|decoder|archive|image|document|network|database|serializer|interpreter|generic`
  - `seed_profile` in `parser-structure|parser-token|parser-format|parser-numeric|decoder-binary|archive-container|serializer-structured|document-text|network-message|generic`
  - forbidden: `name = LLVMFuzzerTestOneInput`
  - semantic reminder: do not rewrite `api` to harness file paths like `fuzz/*.cc` even when schema only requires string type

## Acceptance Criteria
- JSON parses successfully.
- schema fields and enum values are valid for all entries.
- array remains non-empty.
- when schema diagnostics include concrete file/line references, surface explicit actions as `Read and fix <path>[:line]`.

## Command Policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Done Sentinel Contract
- write `fuzz/targets.json` into `./done`.
