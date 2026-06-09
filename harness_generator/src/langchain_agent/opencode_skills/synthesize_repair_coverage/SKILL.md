---
name: synthesize_repair_coverage
description: Repair scaffold for coverage replan cycles using seed and harness feedback as primary signals.
compatibility: opencode
metadata:
  stage: synthesize-repair-coverage
  owner: tianheng
---

## What this skill does
Applies coverage-oriented scaffold updates after replan decisions.

## When to use this skill
Use this skill when `coverage-analysis` selected replan and returned coverage diagnostics.

## Required inputs
- coverage diagnostics (`coverage_*`, `repair_*`)
- `SeedFeedback` and `HarnessFeedback` blocks (if provided)
- current scaffold files under `fuzz/`
- `fuzz/execution_plan.json` and `fuzz/harness_index.json` (if present)
- MCP tools from task-scoped PromeFuzz companion (if available), including preprocessor and semantic tools

## Required outputs
- updated harness/scaffold files under `fuzz/`
- `fuzz/harness_index.json` aligned with `fuzz/execution_plan.json`

## Workflow
1. Query MCP evidence first when MCP is available (preprocessor first, semantic evidence second).
2. Consume `SeedFeedback` and `HarnessFeedback`.
3. Identify coverage bottlenecks and propose concrete fixes.
4. Apply at least one strategy change from previous failed coverage cycle.
5. Keep execution plan and harness index consistent.

## Coverage-bottleneck patterns (diagnose before editing)
- **Uninitialized/empty context (top priority)**: coverage frozen at a trivial
  value (e.g. `cov`/`ft` stuck below ~15) across millions of executions, with a
  tiny corpus that never grows, almost always means the harness reaches the
  target API but the input object was never populated. Verify the harness runs
  the library's full setup sequence before the target call:
  - grammar/parser libraries (tree-sitter, ANTLR, PEG): the grammar/language
    MUST be set before parsing — e.g. `ts_parser_set_language(parser,
    tree_sitter_<grammar>())` before `ts_parser_parse_string(...)`. Without it
    the parse tree is empty and the node/cursor target tests nothing. Link a
    concrete grammar so the parse yields a non-trivial tree.
  - decoders/codecs: the decoder context must be initialized with the expected
    format/options before decoding.
  Fix by adding the missing initialization, not by tweaking seeds — seeds
  cannot help an empty-context harness.
- **Wrong/internal entrypoint**: if the harness drives an internal binding or
  transfer-buffer shim (e.g. `*_wasm`, marshal-buffer indirection) instead of
  the real public API, switch to the public API that performs the actual
  parse/decode the fuzzer needs to exercise.
- **Seed/model gaps**: only after the harness is confirmed to build a valid
  context, address seed structure/format and call-path depth.

## Constraints
- Edits must be coverage-repair-driven (seed/modeling/call-path/depth).
- No doc-only no-op patch.
- Preserve runtime viability for next build/run cycle.
- LibFuzzer harness contract is mandatory:
  - do not define custom `main()` in harness source;
  - use `LLVMFuzzerTestOneInput` (or language-equivalent fuzz entrypoint) as the only fuzz entry.
- Forbid argv/file-driven harness entry logic in libFuzzer mode (`fopen(argv[1], ...)`, `read(argv[1], ...)`, manual corpus file loops).
- If MCP is unavailable, continue in degraded mode and record this in `fuzz/repo_understanding.json`.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Acceptance checklist
- Changes map to coverage diagnostics.
- Strategy change is explicit.
- Execution-plan/harness-index consistency is preserved.

## Done contract
- Write the path string `fuzz/out/` as the sole text of `./done` (run `echo 'fuzz/out/' > ./done`; do **not** copy the file's contents).
