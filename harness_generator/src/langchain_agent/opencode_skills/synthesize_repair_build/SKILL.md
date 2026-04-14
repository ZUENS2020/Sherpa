---
name: synthesize_repair_build
description: Repair fuzz scaffold after build failures with strategy change and mapping consistency.
compatibility: opencode
metadata:
  stage: synthesize-repair-build
  owner: sherpa
---

## What this skill does
Repairs scaffold files under `fuzz/` for build-stage failures.

## When to use this skill
Use this skill in repair mode when the previous build failed.

## Required inputs
- `repair_*` diagnostics from coordinator context
- current scaffold files under `fuzz/`
- `fuzz/execution_plan.json` (if present)
- MCP tools from task-scoped PromeFuzz companion (if available), including preprocessor and semantic tools

## Required outputs
- harness source under `fuzz/`
- `fuzz/build.py` or `fuzz/build.sh`
- `fuzz/README.md`
- `fuzz/repo_understanding.json`
- `fuzz/build_strategy.json`
- `fuzz/build_runtime_facts.json`
- `fuzz/harness_index.json` aligned to `fuzz/execution_plan.json`
- `Known Issues` note in repair artifacts (README or repo_understanding) for unresolved blockers
- `Strategy Delta` note in repair artifacts that states what changed versus the previous failed attempt
- `Output Path Contract` note declaring executable fuzzer outputs must be written to `fuzz/out/`

## Workflow
1. Query MCP evidence first when MCP is available (preprocessor first, semantic evidence second).
2. Consume repair diagnostics.
3. Apply a build-failure-driven strategy update (not cosmetic edits).
4. Update scaffold/build glue and keep mappings consistent.
5. Ensure this round differs from the previous failed strategy.

## Constraints
- No doc-only no-op patches.
- Keep selected/final target and build strategy fields consistent across README and JSON files.
- Update `fuzz/harness_index.json` so each execution target maps to existing harness.
- Compiler-by-suffix in `fuzz/build.py`:
  - `.c` sources must use `clang`
  - `.cc/.cpp/.cxx` sources must use `clang++`
  - do not use `clang++` as universal compiler for mixed C/C++
- Public/stable APIs are mandatory by default.
- If non-public API is unavoidable, require `api_surface_exception` with non-empty `reason` and `evidence`.
- If diagnostics contain `non_public_api_usage`, replace offending symbols first.
- LibFuzzer harness contract is mandatory:
  - do not define custom `main()` in harness source;
  - use `LLVMFuzzerTestOneInput` (or language-equivalent fuzz entrypoint) as the only fuzz entry.
- Forbid argv/file-driven harness entry logic in libFuzzer mode (`fopen(argv[1], ...)`, `read(argv[1], ...)`, manual corpus file loops).
- If MCP is unavailable, continue in degraded mode and record this in `fuzz/repo_understanding.json`.
- Always keep output-path consistency explicit: build glue must place runnable fuzzers under `fuzz/out/` and avoid root-level `fuzz/` binary outputs.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Acceptance checklist
- Edits are build-failure-driven.
- Repeated signatures result in strategy change.
- Execution-plan and harness-index consistency is preserved.

## Done contract
- Write `fuzz/out/` into `./done`.
