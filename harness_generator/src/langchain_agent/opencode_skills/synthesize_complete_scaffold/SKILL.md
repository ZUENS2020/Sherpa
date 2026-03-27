---
name: synthesize_complete_scaffold
description: Complete missing scaffold artifacts with minimal changes while preserving target/build consistency.
compatibility: opencode
metadata:
  stage: synthesize-complete-scaffold
  owner: sherpa
---

## What this skill does
Repairs only missing or semantically invalid scaffold items without rewriting unrelated files.

## When to use this skill
Use this skill when coordinator reports missing scaffold files after synthesize.

## Required inputs
- current `fuzz/` scaffold
- missing items list from coordinator
- `fuzz/execution_plan.json` (if present)
- `fuzz/harness_index.json` (if present)

## Required outputs
- missing required files completed under `fuzz/`
- if harness source is missing, create at least one harness source file before doc/json-only fixes
- refreshed `fuzz/harness_index.json` mapping execution targets to real harness files

## Workflow
1. Repair missing harness source first if absent.
2. Repair/complete required scaffold files only.
3. Repair semantic invalid states in `repo_understanding.json`.
4. Reconcile execution plan and harness index mappings.

## Constraints
- Preserve existing harness/build assets unless minimal change is required.
- `fuzz/repo_understanding.json` must contain non-empty:
  - `build_system`, `chosen_target_api`, `chosen_target_reason`, `fuzzer_entry_strategy`, `evidence`
- `chosen_target_api` must not be harness file path-like.
- `build_system.lower() != "unknown"`.
- `evidence` must be non-empty string array.
- If `fuzz/build.py` uses invalid parallel style (`$(nproc)`), repair to `["-j", str(os.cpu_count() or 1)]`.
- Keep multi-target buildability when `fuzz/execution_plan.json` contains multiple targets.
- Use explicit path actions: `Read and fix <path>[:line]`.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Acceptance checklist
- All required scaffold files exist after this step.
- If harness was missing before this step, harness exists after this step.
- `fuzz/harness_index.json` contains no missing execution-target mappings.
- `repo_understanding.json` is semantically valid.

## Done contract
- Write `fuzz/out/` into `./done`.
