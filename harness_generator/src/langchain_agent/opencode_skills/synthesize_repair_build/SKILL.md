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
- coverage replay output contract: `fuzz/out/replay/<fuzzer>` must be built with `-fprofile-instr-generate -fcoverage-mapping` for every native fuzzer

## Workflow
1. Query MCP evidence first when MCP is available (preprocessor first, semantic evidence second).
2. Consume repair diagnostics.
3. Apply a build-failure-driven strategy update (not cosmetic edits).
4. Update scaffold/build glue and keep mappings consistent.
5. Ensure this round differs from the previous failed strategy.

## Constraints
- No doc-only no-op patches.
- Do not modify repository source files outside `fuzz/` and `./done`; upstream/demo/contrib/example code is read-only and build repair must adapt harness/build glue instead.
- Keep selected/final target and build strategy fields consistent across README and JSON files.
- Update `fuzz/harness_index.json` so each execution target maps to existing harness.
- Compiler-by-suffix in `fuzz/build.py`:
  - `.c` sources must use `clang`
  - `.cc/.cpp/.cxx` sources must use `clang++`
  - do not use `clang++` as universal compiler for mixed C/C++
- CMake invocation in `fuzz/build.py`:
  - use the system `cmake` binary directly, e.g. `["cmake", "-S", ...]` and `["cmake", "--build", ...]`
  - do not use `sys.executable, "-m", "cmake"` or `python -m cmake`; the runtime image does not guarantee the Python `cmake` package
- Public/stable APIs are mandatory by default.
- If non-public API is unavoidable, require `api_surface_exception` with non-empty `reason` and `evidence`.
- If diagnostics contain `non_public_api_usage`, replace offending symbols first.
- LibFuzzer harness contract is mandatory:
  - do not define custom `main()` in harness source;
  - use `LLVMFuzzerTestOneInput` (or language-equivalent fuzz entrypoint) as the only fuzz entry.
  - C/C++ harnesses that use `uint8_t` or `size_t` must include the standard headers that define them (`<stdint.h>` and `<stddef.h>` or C++ equivalents) in the harness source.
- LibFuzzer link contract is mandatory: every runnable executable under `fuzz/out/`, including `fuzz/out/replay/<name>`, must link a runnable entrypoint. Prefer `-fsanitize=fuzzer,address,undefined` for both primary and replay executables; `-fsanitize=fuzzer-no-link` alone causes `undefined reference to main` unless a separate replay `main()` wrapper is compiled and linked to call `LLVMFuzzerTestOneInput`.
- Forbid argv/file-driven harness entry logic in libFuzzer mode (`fopen(argv[1], ...)`, `read(argv[1], ...)`, manual corpus file loops).
- If MCP is unavailable, continue in degraded mode and record this in `fuzz/repo_understanding.json`.
- Always keep output-path consistency explicit: build glue must place runnable fuzzers under `fuzz/out/` and avoid root-level `fuzz/` binary outputs.
- Do not satisfy coverage replay by symlinking/copying primary fuzzers into `fuzz/out/replay/`; replay binaries must be separately linked with LLVM profile coverage instrumentation and must be runnable executables, not `fuzzer-no-link` objects without `main()`.
- Generated headers must be discoverable: when CMake/configure emits headers into a build directory, add that CMake build directory to every primary and replay compile command before retrying the same build strategy.
- Contrib/example/demo target calls must resolve their owning source: compile the owning source file alongside the harness, or switch the harness to a public library API if the source is not safe to link.
- Do not call a static example helper directly from a harness; static example helpers are not linkable API and example sources that define `main()` must not be linked into libFuzzer harnesses unless rewritten/guarded.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Acceptance checklist
- Edits are build-failure-driven.
- Repeated signatures result in strategy change.
- Execution-plan and harness-index consistency is preserved.
- Coverage replay siblings are profile-instrumented and located under `fuzz/out/replay/`.

## Done contract
- Write `fuzz/out/` into `./done`.
