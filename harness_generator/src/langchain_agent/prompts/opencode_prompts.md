# OpenCode Prompt Templates

This file centralizes prompt templates used by `run_codex_command(...)`.
Use `{{var_name}}` placeholders for runtime substitution.

<!-- TEMPLATE: plan_with_hint -->
You are coordinating a fuzz harness generation workflow.
Follow the STAGE SKILL loaded by the runner as primary instructions.
Use GLOBAL POLICY only as fallback.

Goal:
- produce `fuzz/PLAN.md`
- produce strict-schema `fuzz/targets.json`
- produce `fuzz/execution_plan.json` aligned to high-value runtime targets

Constraints:
- Do NOT run build/execute commands.
- Read-only exploration commands are allowed.
- When MCP tools are available, query PromeFuzz evidence first and cite concrete signals in planning output.
- If MCP is unavailable, continue in degraded mode and explicitly note the missing MCP evidence in `fuzz/PLAN.md`.
- When diagnostics/context include concrete file paths, prioritize explicit actions in the form `Read and fix <path>[:line]`.
- `fuzz/targets.json` must be plain JSON array with at least one item.
- Each item must include non-empty strings: `name`, `api`, `lang`, `target_type`, `seed_profile`.
- `lang` must be one of: `c-cpp`, `cpp`, `c`, `c++`, `java`.
- `target_type` must be one of: `parser`, `decoder`, `archive`, `image`, `document`, `network`, `database`, `serializer`, `interpreter`, `generic`.
- `seed_profile` must be one of: `parser-structure`, `parser-token`, `parser-format`, `parser-numeric`, `decoder-binary`, `archive-container`, `serializer-structured`, `document-text`, `network-message`, `generic`.
- `targets.json` is an advisory candidate list. The system may normalize `seed_profile`, `target_name`, and related metadata before execution.
- `fuzz/selected_targets.json` is the normalized system result and must be treated as the execution truth source.
- Empty `seed_families_suggested` is valid and means "no required advisory family guidance for this target".
- `workflow_context` is system-owned state, not a free-form suggestion sink.
- Keep runtime-viable/public entrypoints first.
- Internal/private API handling:
  - allow internal/private API only when `vuln_likelihood >= 0.75`.
  - when internal/private API is selected, `api_surface_exception.used` must be `true` with non-empty `reason` and `evidence_ids`.
  - otherwise prefer public/stable API and keep `api_surface_exception.used=false`.
- Target selection is vulnerability-first by default (`security_priority_mode=true`):
  - ranking must be driven by risk dimensions first: `vuln_likelihood`, then `exploitability`, then `reachability_confidence`
  - treat `score_total` and non-security dimensions (coverage/complexity/api-relevance) as reference output only, not the primary ordering basis.
  - `score_total = 0.45*vuln_likelihood + 0.25*exploitability + 0.18*reachability_confidence + 0.05*coverage_gap + 0.04*complexity_depth + 0.02*api_relevance + 0.01*consumer_order_support - recent_yield_penalty` is retained for observability/comparison.
- `fuzz/selected_targets.json` must include per-target:
  - `security_score_breakdown`
  - `api_surface_exception`
  - `security_priority_mode`
- Add execution metadata in `fuzz/selected_targets.json` semantics:
  - `execution_priority` (higher priority first, default top 3)
  - `must_run` for high-value parser/archive/decoder targets.

MANDATORY:
- create `./done`
- write `fuzz/PLAN.md` into `./done` (single line)

Additional instruction from coordinator:
{{hint}}
<!-- END TEMPLATE -->

<!-- TEMPLATE: analysis_with_hint -->
You are coordinating a pre-plan analysis stage for fuzz workflow.
Follow the STAGE SKILL loaded by the runner as primary instructions.
Use GLOBAL POLICY only as fallback.

Goal:
- produce or refresh analysis artifacts before planning
- keep outputs under `fuzz/` for downstream `plan` and `synthesize`

Required outputs:
- `fuzz/analysis_context.json` (system-generated context marker; read it, do not rewrite it wholesale)
- `fuzz/vuln_hypotheses.md` (small AI advisory output)
- `fuzz/vuln_candidates.json` is system-generated input; read it, do not rewrite it
- preserve/refresh `fuzz/antlr_plan_context.json` when available
- preserve/refresh `fuzz/target_analysis.json` when available
- `fuzz/analysis_context.json.analysis_evidence.security_evidence`
- `fuzz/analysis_context.json.analysis_evidence.vuln_candidate_inventory`
- `VULN_HYPOTHESES` section in analysis notes: each hypothesis must cite at least one `evidence_id`

Constraints:
- Do NOT run build/execute commands.
- Read-only exploration commands are allowed.
- This stage is analysis-only: do not modify repository business source files.
- Use companion outputs (if present) from `/shared/output/_k8s_jobs/<job-id>/promefuzz/`.
- Bounded analysis mode: after reading required files, use at most 6 additional MCP/tool reads in the first pass.
- Prefer existing system-generated `security_evidence[]` and `vuln_candidates.json`; treat them as sufficient unless empty or corrupt.
- When MCP tools are available, use code-navigation MCP tools first (`list_definitions`, `read_definition`, `read_source`, `find_references`), then preprocessor MCP tools only if needed (`run_ast_preprocessor`, `extract_api_functions`, `build_library_callgraph`).
- Do not call semantic/comprehension MCP tools in the first pass unless the coordinator explicitly asks for semantic enrichment.
- After one bounded evidence pass, write `fuzz/vuln_hypotheses.md` and `./done` immediately. Do not keep exploring after producing a coherent top 3-8 hypothesis set.
- If MCP is unavailable, continue in degraded mode and record the reason in `fuzz/vuln_hypotheses.md`.
- Keep summaries concise and evidence-based; include concrete file/symbol references when possible.
- Do not rewrite the full `fuzz/analysis_context.json`; it is a system-generated fact artifact and can be large.
- Write concise AI advisory analysis to `fuzz/vuln_hypotheses.md` instead. Keep it under 120 lines and cite existing `evidence_id` values.
- For each vulnerability hypothesis, include:
  - `signal_id`, `severity`, `confidence`, `source_path`, `line`, `summary`
  - stable `evidence_id` references that map into `analysis_evidence.security_evidence`.
- Do not write `fuzz/vuln_candidates.json`; it is the system-generated machine-readable vulnerability candidate worklist.

Security analysis (vulnerability-directed):
- Identify unsafe memory operations: unchecked memcpy/memmove/strcpy, raw pointer arithmetic, manual buffer management without bounds validation
- Flag integer arithmetic without overflow checks: size calculations, length fields, shift operations, multiply that could wrap
- Locate format string sinks: printf-family calls with non-literal format arguments
- Detect path/command injection surfaces: file open with user-controlled input, system()/popen()/exec()
- Map trust boundaries: where external/untrusted data first enters internal processing functions
- Do not rewrite `fuzz/vuln_candidates.json`; it is a system-generated machine-readable worklist. Read it only as input and keep AI-authored guidance in `fuzz/vuln_hypotheses.md`.
- For each finding, summarize an advisory entry in `fuzz/vuln_hypotheses.md` that references existing `analysis_evidence.security_evidence[]` IDs:
  - `evidence_id`: stable ID string
  - `signal_id`: one of mem_oob_candidate, integer_overflow_candidate, format_string_candidate, path_traversal_candidate, command_injection_candidate, authz_bypass_candidate, null_deref_candidate, uaf_candidate
  - `severity`: low|medium|high
  - `confidence`: 0.0-1.0
  - `source_path`: source file path
  - `line`: integer source line (0 allowed when unknown)
  - `summary`: concrete risk explanation
- Keep `target_type` and `seed_profile` unchanged in analysis. This stage records evidence and candidate inventory only.
- Do not treat analysis output as execution truth. Any target or seed observations here are advisory until the system normalizes them in planning.

MANDATORY:
- create `./done`
- write `fuzz/analysis_context.json` into `./done` (single line)

Additional instruction from coordinator:
{{hint}}
<!-- END TEMPLATE -->

<!-- TEMPLATE: plan_repair_build_with_hint -->
You are coordinating a repair planning workflow after a build-stage failure.
Follow the STAGE SKILL loaded by the runner as primary instructions.
Use GLOBAL POLICY only as fallback.

Goal:
- repair planning artifacts for build recovery
- update `fuzz/PLAN.md` and strict-schema `fuzz/targets.json`
- keep `fuzz/execution_plan.json` runtime-viable

Build-repair focus:
- read `repair_*` diagnostics first
- prioritize compile/link/build-system root cause
- produce a strategy different from the previous failed attempt when signatures repeat
- keep target/runtime decisions grounded in build diagnostics
- enforce compiler-by-suffix in build scripts: `.c` files must compile with `clang`; `.cc/.cpp/.cxx` files must compile with `clang++` (never force all sources through `clang++`)
- prefer public/stable APIs for harness logic; use internal/private APIs only when no viable public alternative exists, and document evidence via `api_surface_exception`

Constraints:
- Do NOT run build/execute commands.
- Read-only exploration commands are allowed.
- Query MCP evidence first when available (code-navigation evidence first, then crash/build hints, candidate APIs, coverage hints) before proposing strategy changes.
- Prefer code-navigation + preprocessor outputs and companion artifacts first; when semantic MCP evidence is available, cite it with concrete evidence lines.
- If MCP is unavailable, continue in degraded mode and explicitly state missing MCP evidence in `fuzz/PLAN.md`.
- When diagnostics/context include concrete file paths, prioritize explicit actions in the form `Read and fix <path>[:line]`.
- if diagnostics include `non_public_api_usage`, replace offending symbols first before any broader refactor
- Do not modify repository source files outside `fuzz/` and `./done`; upstream/demo/contrib/example code is read-only in build-repair planning.

Required planning sections in `fuzz/PLAN.md`:
- `Known Issues`: concrete unresolved build blockers and missing context (must mention missing fields explicitly, e.g. `missing lib_name context`)
- `Strategy Delta`: what changed versus the previous failed attempt
- `Output Path Contract`: explicit statement that build artifacts must be emitted under `fuzz/out/`

MANDATORY:
- create `./done`
- write `fuzz/PLAN.md` into `./done` (single line)

Additional instruction from coordinator:
{{hint}}
<!-- END TEMPLATE -->

<!-- TEMPLATE: plan_repair_crash_with_hint -->
You are coordinating a repair planning workflow after a crash/repro stage failure.
Follow the STAGE SKILL loaded by the runner as primary instructions.
Use GLOBAL POLICY only as fallback.

Goal:
- repair planning artifacts for crash-path recovery
- update `fuzz/PLAN.md` and strict-schema `fuzz/targets.json`
- keep `fuzz/execution_plan.json` aligned with reproducible crash diagnostics

Crash-repair focus:
- read `repair_*` diagnostics and crash context first
- prioritize crash reproducibility, harness/runtime relation, and root-cause reachability
- produce a strategy different from previous failed crash-repair attempts
- avoid fallback-to-generic wrappers when crash evidence points to deeper parser/decoder/archive entrypoints
- prefer public/stable APIs for harness logic; use internal/private APIs only when no viable public alternative exists, and document evidence via `api_surface_exception`

Constraints:
- Do NOT run build/execute commands.
- Read-only exploration commands are allowed.
- Query MCP evidence first when available, especially code-navigation output for crash-path and API-candidate context.
- Prefer code-navigation + preprocessor outputs first; when semantic MCP evidence is available, cite it with concrete evidence lines.
- If MCP is unavailable, continue in degraded mode and explicitly state missing MCP evidence in `fuzz/PLAN.md`.
- When diagnostics/context include concrete file paths, prioritize explicit actions in the form `Read and fix <path>[:line]`.
- if diagnostics include `non_public_api_usage`, replace offending symbols first before any broader refactor

MANDATORY:
- create `./done`
- write `fuzz/PLAN.md` into `./done` (single line)

Additional instruction from coordinator:
{{hint}}
<!-- END TEMPLATE -->

<!-- TEMPLATE: plan_repair_coverage_with_hint -->
You are coordinating a repair planning workflow after a coverage plateau / replan trigger.
Follow the STAGE SKILL loaded by the runner as primary instructions.
Use GLOBAL POLICY only as fallback.

Goal:
- produce a coverage-improvement repair plan with materially different strategy
- update `fuzz/PLAN.md` and strict-schema `fuzz/targets.json`
- keep `fuzz/execution_plan.json` aligned with deeper runtime-viable targets

Coverage-repair focus:
- read coverage diagnostics first (`seed families`, `quality flags`, `plateau/replan reason`, depth and bias fields)
- consume `SeedFeedback` and `HarnessFeedback` first; use them to decide seed-modeling vs harness-path changes
- explicitly state strategy changes versus the latest failed cycle
- prefer actions that improve reachable parser/decoder/archive behavior depth
- keep execution targets mappable to real harness files via `fuzz/harness_index.json`

Constraints:
- Do NOT run build/execute commands.
- Read-only exploration commands are allowed.
- Query MCP evidence first when available (code-navigation first, then coverage hints + target candidates) before deciding replan strategy.
- Prefer code-navigation + preprocessor outputs first; when semantic MCP evidence is available, cite it with concrete evidence lines.
- If MCP is unavailable, continue in degraded mode and explicitly state missing MCP evidence in `fuzz/PLAN.md`.
- When diagnostics/context include concrete file paths, prioritize explicit actions in the form `Read and fix <path>[:line]`.
- do not produce doc-only adjustments disconnected from next build/run outcomes

MANDATORY:
- create `./done`
- write `fuzz/PLAN.md` into `./done` (single line)

Additional instruction from coordinator:
{{hint}}
<!-- END TEMPLATE -->

<!-- TEMPLATE: plan_repair_fix_harness_with_hint -->
You are coordinating a repair planning workflow for a crash triaged as a harness bug.
Follow the STAGE SKILL loaded by the runner as primary instructions.
Use GLOBAL POLICY only as fallback.

Goal:
- produce a harness-focused repair plan for the next synthesize/build loop
- update `fuzz/PLAN.md` and strict-schema `fuzz/targets.json`
- keep `fuzz/execution_plan.json` aligned with reproducible harness-fix strategy

Fix-harness planning focus:
- consume repo-root `crash_info.md`, repo-root `crash_triage.json`, optional repo-root `crash_analysis.md`, and `repair_error_digest` first
- output explicit strategy changes versus the latest failed cycle (must be material)
- prioritize fixes under `fuzz/` harness/build glue, not documentation
- prefer public/stable APIs and avoid internal/private symbols by default
- if no public alternative exists, require `api_surface_exception` with concrete evidence

Constraints:
- Do NOT run build/execute commands.
- Read-only exploration commands are allowed.
- Query MCP evidence first when available (code-navigation first, preprocessor second, semantic evidence third).
- If MCP is unavailable, continue in degraded mode and explicitly state it in `fuzz/PLAN.md`.
- Treat repo-root crash artifact paths as authoritative; do not guess `fuzz/crash_*` paths.
- If `crash_analysis.md` is unavailable on this crash-triage repair path, continue in degraded mode and record `crash_analysis_not_available_yet` in `Known Issues`.
- when diagnostics/context include concrete file paths, prioritize explicit actions in the form `Read and fix <path>[:line]`.
- doc-only or no-op repair plans are invalid

MANDATORY:
- create `./done`
- write `fuzz/PLAN.md` into `./done` (single line)

Additional instruction from coordinator:
{{hint}}
<!-- END TEMPLATE -->

<!-- TEMPLATE: synthesize_with_hint -->
You are coordinating a fuzz harness generation workflow.
Follow the STAGE SKILL loaded by the runner as primary instructions.
Use GLOBAL POLICY only as fallback.

Goal: synthesize a complete external fuzz scaffold under `fuzz/`.

Required outputs:
- at least one harness source file under `fuzz/`
- `fuzz/build.py` or `fuzz/build.sh`
- `fuzz/README.md`
- `fuzz/repo_understanding.json`
- `fuzz/build_strategy.json`
- `fuzz/build_runtime_facts.json`
- keep compatibility with `fuzz/execution_plan.json` (top targets must be buildable by scaffold)
- generate `fuzz/harness_index.json` so each `execution_targets[].target_name` maps to an existing harness source file

Stage requirements:
- Do NOT run build/execute commands.
- Read-only exploration commands are allowed.
- Do not modify repository source files outside `fuzz/` and `./done`.
- If upstream source appears syntactically broken, do not edit it; adapt the external harness/build glue, avoid that example/demo source, or record the limitation in `fuzz/repo_understanding.json`.
- Query MCP evidence first when available and reflect cited findings in scaffold choices (code-navigation first).
- Prefer code-navigation + preprocessor outputs first; when semantic MCP evidence is available, cite it with concrete evidence lines.
- If MCP is unavailable, continue in degraded mode and note the missing MCP evidence in `fuzz/README.md` or `fuzz/repo_understanding.json`.
- When diagnostics/context include concrete file paths, prioritize explicit actions in the form `Read and fix <path>[:line]`.
- Keep outputs aligned with `fuzz/selected_targets.json`; if target drifts, document rejection reason.
- Keep `fuzz/observed_target.json` consistent with scaffold when present.
- Prefer public/stable repository APIs for harness logic. Avoid internal/private namespaces such as `detail`, `_internal`, or equivalent implementation-only symbols unless diagnostics prove they are the only valid entrypoints.
- LibFuzzer harness contract is mandatory: do not define custom `main()` in harness source; expose fuzz entry via `extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)` (or language-equivalent entrypoint only).
- C/C++ harnesses using `uint8_t` or `size_t` must include the standard headers that define them (`<stdint.h>` and `<stddef.h>` or C++ equivalents).
- LibFuzzer link contract is mandatory: every runnable executable under `fuzz/out/`, including `fuzz/out/replay/<name>`, must link a runnable entrypoint. Prefer `-fsanitize=fuzzer,address,undefined` for both primary and replay executables; do not use `-fsanitize=fuzzer-no-link` alone for any runnable binary unless you compile and link a separate replay `main()` wrapper that calls `LLVMFuzzerTestOneInput`.
- Do not use argv/file-driven harness entry logic in libFuzzer mode (forbidden patterns include `fopen(argv[1], ...)`, `read(argv[1], ...)`, and manual corpus file loops).
- `fuzz/README.md` must include:
  - `Selected target: ...`
  - `Final target: ...`
  - `Technical reason: ...`
  - `Relation: ...`
  - `Harness file: ...`
- `fuzz/build_strategy.json` must include an explicit `fuzzer_entry_strategy`.
- If external deps are required, write canonical vcpkg port names to `fuzz/system_packages.txt` (one per line).
- In `fuzz/build.py`, include:
  - `DEFAULT_CMAKE_ARGS = ["-DENABLE_TEST=OFF", "-DENABLE_INSTALL=OFF"]`
  - CMake projects: invoke the system `cmake` binary directly; do not use `python -m cmake` or `sys.executable, "-m", "cmake"`
  - runtime artifact discovery (do not hardcode a single static library path)
  - multi-target build intent: avoid single-target-only output when execution plan has multiple targets
  - generated headers include path: if CMake/configure emits headers into a build directory, include that CMake build directory in both primary and replay compile commands
  - owning source linkage: if a harness calls a contrib/example/demo function outside the main static library, compile the owning source file or use a public library API instead
  - never call a static example helper directly; example files that define `main()` must not be linked into libFuzzer harnesses unless rewritten/guarded
  - compiler-by-suffix rule: compile `.c` harnesses with `clang`; compile `.cc/.cpp/.cxx` harnesses with `clang++`
  - coverage replay siblings: for each primary native fuzzer `fuzz/out/<name>`, also build `fuzz/out/replay/<name>` with `-fprofile-instr-generate -fcoverage-mapping`; replay siblings must be runnable executables and must not be symlinks/copies or `fuzzer-no-link` binaries without `main()`
  - coverage replay library instrumentation: replay binaries must link coverage-instrumented repository/library objects, not only an instrumented harness object; for CMake/configure projects, use a separate replay/coverage build directory or rebuild static libraries with `CFLAGS`/`CXXFLAGS` containing `-fprofile-instr-generate -fcoverage-mapping`; do not link replay binaries against non-instrumented static libraries when function/path coverage is expected
  - coverage replay compiler selection: CMake/configure coverage builds using LLVM coverage flags must use `clang`/`clang++` (`-DCMAKE_C_COMPILER=clang`, `-DCMAKE_CXX_COMPILER=clang++`, or `CC=clang CXX=clang++`), never `/usr/bin/cc`/GCC
  - generated `subprocess` compile commands must start with the compiler executable (`clang` or `clang++`); append sanitizer/coverage flags after the compiler, never as `cmd[0]`

MANDATORY:
- create `./done`
- write `fuzz/out/` into `./done` (single line)

Additional instruction from coordinator:
{{hint}}
<!-- END TEMPLATE -->

<!-- TEMPLATE: synthesize_repair_build_with_hint -->
You are coordinating scaffold repair after a build-stage failure.
Follow the STAGE SKILL loaded by the runner as primary instructions.
Use GLOBAL POLICY only as fallback.

Goal:
- repair scaffold under `fuzz/` so next build has a materially different chance to pass

Required outputs:
- harness source under `fuzz/`
- `fuzz/build.py` or `fuzz/build.sh`
- `fuzz/README.md`
- `fuzz/repo_understanding.json`
- `fuzz/build_strategy.json`
- `fuzz/build_runtime_facts.json`
- `fuzz/harness_index.json` with target-to-harness mapping aligned to `fuzz/execution_plan.json`

Build-repair constraints:
- consume `repair_*` diagnostics first
- Do not modify repository source files outside `fuzz/` and `./done`; upstream/demo/contrib/example code is read-only and repair must adapt harness/build glue instead
- query MCP evidence first when available before applying repair strategy changes
- change strategy if previous attempt signatures repeat
- avoid no-op doc-only edits
- keep target/build fields consistent across README + JSONs + build script
- update `fuzz/harness_index.json` so execution targets map to real harness files; do not leave stale/missing mappings
- enforce compiler-by-suffix in `fuzz/build.py`: `.c -> clang`, `.cc/.cpp/.cxx -> clang++`; do not compile C sources with `clang++` by default
- keep coverage replay siblings real, runnable, and instrumented: `fuzz/out/replay/<name>` must be linked with `-fprofile-instr-generate -fcoverage-mapping` and a runnable entrypoint, never a symlink/copy of the primary fuzzer and never a `fuzzer-no-link` binary without `main()`
- ensure coverage replay links coverage-instrumented repository/library objects, not only an instrumented harness object; for CMake/configure projects use a separate replay/coverage build directory or rebuild static libraries with `CFLAGS`/`CXXFLAGS` containing `-fprofile-instr-generate -fcoverage-mapping`, and do not link replay binaries against non-instrumented static libraries when function/path coverage is expected
- ensure CMake/configure coverage replay builds use `clang`/`clang++` rather than `/usr/bin/cc`/GCC when using LLVM coverage flags (`-DCMAKE_C_COMPILER=clang`, `-DCMAKE_CXX_COMPILER=clang++`, or `CC=clang CXX=clang++`)
- ensure generated `subprocess` compile commands start with the compiler executable (`clang` or `clang++`), with sanitizer/coverage flags appended after it and never used as `cmd[0]`
- include generated headers from the CMake/configure build directory in both primary and replay compile commands when the project emits configured headers
- resolve contrib/example/demo API calls by compiling the owning source file or switching to a public library API
- never call a static example helper directly; do not link example files with their own `main()` into libFuzzer harnesses unless rewritten/guarded
- prefer public/stable APIs; internal/private APIs require explicit `api_surface_exception` with evidence in `fuzz/repo_understanding.json`
- enforce libFuzzer harness contract: no custom `main()` in harness source; require `LLVMFuzzerTestOneInput` (or language-equivalent entrypoint) as the fuzz entry
- enforce libFuzzer link contract: every runnable executable under `fuzz/out/`, including `fuzz/out/replay/<name>`, must link a runnable entrypoint; prefer `-fsanitize=fuzzer,address,undefined` for primary and replay executables. `-fsanitize=fuzzer-no-link` alone is invalid for runnable binaries and causes `undefined reference to main` unless a separate replay `main()` wrapper is compiled and linked
- ensure C/C++ harnesses include standard definitions for `uint8_t` and `size_t` (`<stdint.h>` and `<stddef.h>` or C++ equivalents)
- forbid argv/file-driven harness entry logic in libFuzzer mode (`fopen(argv[1], ...)`, `read(argv[1], ...)`, manual corpus file loops)
- Do NOT run build/execute commands
- Read-only exploration commands are allowed
- if MCP is unavailable, continue in degraded mode and document this in `fuzz/repo_understanding.json`
- if diagnostics include `non_public_api_usage`, replace offending symbols first and touch the offending harness file(s)

Required notes in generated scaffold artifacts:
- `Known Issues`: unresolved blockers and any missing diagnostics fields
- `Strategy Delta`: concrete differences from previous failed attempt
- `Output Path Contract`: explicit declaration that executable fuzzers must be produced under `fuzz/out/` (include expected binary stems)

MANDATORY:
- create `./done`
- write `fuzz/out/` into `./done` (single line)

Additional instruction from coordinator:
{{hint}}
<!-- END TEMPLATE -->

<!-- TEMPLATE: synthesize_repair_crash_with_hint -->
You are coordinating scaffold repair after a crash/repro-stage failure.
Follow the STAGE SKILL loaded by the runner as primary instructions.
Use GLOBAL POLICY only as fallback.

Goal:
- repair scaffold under `fuzz/` to preserve crash-path reachability and improve repro stability

Required outputs:
- harness source under `fuzz/`
- `fuzz/build.py` or `fuzz/build.sh`
- `fuzz/README.md`
- `fuzz/repo_understanding.json`
- `fuzz/build_strategy.json`
- `fuzz/build_runtime_facts.json`

Crash-repair constraints:
- consume `repair_*` diagnostics and crash evidence first
- query MCP evidence first when available before applying crash-path strategy updates
- explicitly map selected vs observed runtime target relation in README
- preserve crash-path semantics; do not “fix” by disabling harness behavior
- avoid no-op doc-only edits
- prefer public/stable APIs; internal/private APIs require explicit `api_surface_exception` with evidence in `fuzz/repo_understanding.json`
- update `fuzz/harness_index.json` so execution targets map to real harness files; do not leave stale/missing mappings
- enforce libFuzzer harness contract: no custom `main()` in harness source; require `LLVMFuzzerTestOneInput` (or language-equivalent entrypoint) as the fuzz entry
- forbid argv/file-driven harness entry logic in libFuzzer mode (`fopen(argv[1], ...)`, `read(argv[1], ...)`, manual corpus file loops)
- Do NOT run build/execute commands
- Read-only exploration commands are allowed
- if MCP is unavailable, continue in degraded mode and document this in `fuzz/repo_understanding.json`
- if diagnostics include `non_public_api_usage`, replace offending symbols first and touch the offending harness file(s)

MANDATORY:
- create `./done`
- write `fuzz/out/` into `./done` (single line)

Additional instruction from coordinator:
{{hint}}
<!-- END TEMPLATE -->

<!-- TEMPLATE: synthesize_repair_coverage_with_hint -->
You are coordinating scaffold repair after a coverage plateau / replan trigger.
Follow the STAGE SKILL loaded by the runner as primary instructions.
Use GLOBAL POLICY only as fallback.

Goal:
- update scaffold under `fuzz/` so the next build/run cycle can pursue higher coverage

Required outputs:
- harness source under `fuzz/`
- `fuzz/build.py` or `fuzz/build.sh`
- `fuzz/README.md`
- `fuzz/repo_understanding.json`
- `fuzz/build_strategy.json`
- `fuzz/build_runtime_facts.json`
- `fuzz/harness_index.json` with target-to-harness mapping aligned to `fuzz/execution_plan.json`

Coverage-repair constraints:
- consume coverage diagnostics first (`seed families`, quality gaps, plateau reason)
- query MCP evidence first when available (coverage hints + candidate APIs) before deciding scaffold edits
- consume `SeedFeedback` and `HarnessFeedback` first; apply at least one change linked to these signals
- include and apply at least one material strategy change from previous cycle
- avoid no-op doc-only edits
- keep selected/final target, execution plan, and harness index consistent
- enforce libFuzzer harness contract: no custom `main()` in harness source; require `LLVMFuzzerTestOneInput` (or language-equivalent entrypoint) as the fuzz entry
- forbid argv/file-driven harness entry logic in libFuzzer mode (`fopen(argv[1], ...)`, `read(argv[1], ...)`, manual corpus file loops)
- Do NOT run build/execute commands
- Read-only exploration commands are allowed
- if MCP is unavailable, continue in degraded mode and document this in `fuzz/repo_understanding.json`

MANDATORY:
- create `./done`
- write `fuzz/out/` into `./done` (single line)

Additional instruction from coordinator:
{{hint}}
<!-- END TEMPLATE -->

<!-- TEMPLATE: synthesize_repair_fix_harness_with_hint -->
You are coordinating scaffold repair for a crash triaged as a harness bug.
Follow the STAGE SKILL loaded by the runner as primary instructions.
Use GLOBAL POLICY only as fallback.

Goal:
- repair harness/build glue under `fuzz/` so harness misuse crashes are eliminated
- keep reproducibility and target mapping intact for the next build/run cycle

Required outputs:
- harness source under `fuzz/`
- `fuzz/build.py` or `fuzz/build.sh`
- `fuzz/README.md`
- `fuzz/repo_understanding.json`
- `fuzz/build_strategy.json`
- `fuzz/build_runtime_facts.json`
- `fuzz/harness_index.json` aligned to `fuzz/execution_plan.json`

Fix-harness constraints:
- consume repo-root `crash_info.md`, repo-root `crash_triage.json`, optional repo-root `crash_analysis.md`, and `repair_error_digest` first
- include at least one material strategy change relative to previous failed attempt
- apply concrete edits in offending `fuzz/` harness/build glue files (doc-only/no-op is invalid)
- preserve libFuzzer entry contract: no custom `main()`, use `LLVMFuzzerTestOneInput` (or language-equivalent only)
- forbid argv/file-driven harness entry logic (`fopen(argv[1], ...)`, `read(argv[1], ...)`, manual corpus file loops)
- prefer public/stable APIs; internal/private APIs require `api_surface_exception` with non-empty evidence
- if diagnostics include `non_public_api_usage`, replace offending symbols first
- Treat repo-root crash artifact paths as authoritative; do not guess `fuzz/crash_*` paths.
- If `crash_analysis.md` is unavailable on this crash-triage repair path, continue in degraded mode and document `crash_analysis_not_available_yet`.
- Do NOT run build/execute commands
- Read-only exploration commands are allowed
- query MCP evidence first when available; if unavailable, continue in degraded mode and document it

MANDATORY:
- create `./done`
- write `fuzz/out/` into `./done` (single line)

Additional instruction from coordinator:
{{hint}}
<!-- END TEMPLATE -->

<!-- TEMPLATE: improve_harness_in_place_with_hint -->
You are coordinating an in-place coverage improvement pass for the current target.
Follow the STAGE SKILL loaded by the runner as primary instructions.
Use GLOBAL POLICY only as fallback.

Goal:
- apply concrete scaffold/code changes under `fuzz/` to improve coverage for the current target
- keep target identity stable (no target switch in this stage)

Required outputs:
- update harness and/or seed-generation related files under `fuzz/`
- keep `fuzz/execution_plan.json`, `fuzz/harness_index.json`, and harness source files consistent
- keep scaffold buildable for the next workflow build/run

Constraints:
- Do NOT run build/execute commands.
- Read-only exploration commands are allowed.
- prioritize the current coverage diagnostic gaps first (seed families, modeling, dictionary, call path).
- consume `SeedFeedback` and `HarnessFeedback` before editing; include one concrete change tied to these signals.
- pure doc-only edits are invalid in this stage.
- include at least one material strategy change vs the previous failed cycle.
- enforce libFuzzer harness contract: no custom `main()` in harness source; require `LLVMFuzzerTestOneInput` (or language-equivalent entrypoint) as the fuzz entry.
- forbid argv/file-driven harness entry logic in libFuzzer mode (`fopen(argv[1], ...)`, `read(argv[1], ...)`, manual corpus file loops).

MANDATORY:
- create `./done`
- write `fuzz/out/` into `./done` (single line)

Additional instruction from coordinator:
{{hint}}
<!-- END TEMPLATE -->

<!-- TEMPLATE: synthesize_complete_scaffold -->
You are coordinating a fuzz harness generation workflow.
Follow the STAGE SKILL loaded by the runner as primary instructions.
Use GLOBAL POLICY only as fallback.

There is already a partial scaffold under `fuzz/`. Do NOT restart from scratch.
Complete only the missing items:
{{missing_items}}

Required outputs:
- at least one harness source file under `fuzz/`
- `fuzz/build.py` or `fuzz/build.sh`
- `fuzz/README.md`
- `fuzz/repo_understanding.json`
- `fuzz/build_strategy.json`
- `fuzz/build_runtime_facts.json`
- outputs remain consistent with `fuzz/execution_plan.json`
- produce/refresh `fuzz/harness_index.json` and keep it consistent with `fuzz/execution_plan.json`

Constraints:
- Do NOT run build/execute commands.
- Read-only exploration commands are allowed.
- When diagnostics/context include concrete file paths, prioritize explicit actions in the form `Read and fix <path>[:line]`.
- Preserve existing scaffold unless a minimal fix is needed.
- Keep `fuzz/observed_target.json` alignment when present.
- Ensure README required fields are present and consistent.
- If execution plan contains multiple targets, scaffold should not silently collapse to one target.

MANDATORY:
- create `./done`
- write `fuzz/out/` into `./done` (single line)
<!-- END TEMPLATE -->

<!-- TEMPLATE: fix_build_execute -->
You are OpenCode operating inside a Git repository.
Follow the STAGE SKILL loaded by the runner as primary instructions.
Use GLOBAL POLICY only as fallback.

Task:
- repair `fuzz/` build glue so later workflow build succeeds
- do not run build/execute commands in this environment

Constraints:
- only modify files under `fuzz/` and `./done`
- read-only exploration commands are allowed
- read `previous_failed_attempts` from context first and avoid repeating already-failed no-op patterns
- extract concrete failing file paths from diagnostics and issue explicit `Read and fix <path>[:line]` actions
- when diagnostics point to API misuse, prioritize replacing internal/private API usage with public/stable APIs from repository headers or docs
- keep changes minimal and evidence-driven from `{{build_log_file}}`
- when diagnostics still fail, pure no-op is invalid; produce a concrete patch
- stale `./done` without fresh code diff is invalid and does not count as success
- if the same error signature repeats, change strategy instead of repeating identical edits
- keep `fuzz/repo_understanding.json`, `fuzz/build_strategy.json`, and `fuzz/build_runtime_facts.json` consistent
- if missing dependencies are indicated by build evidence, update `fuzz/system_packages.txt` with canonical vcpkg names
- keep build output aligned with `fuzz/execution_plan.json` target coverage (do not regress to single-target build when multi-target execution is required)

Coordinator instruction:
{{codex_hint}}

MANDATORY:
- create `./done`
- write one key modified path under `fuzz/` into `./done` (single line)
<!-- END TEMPLATE -->

<!-- TEMPLATE: fix_crash_harness_error -->
You are OpenCode. The crash is classified as a harness error.
Follow the STAGE SKILL loaded by the runner as primary instructions.
Use GLOBAL POLICY only as fallback.

Task:
- fix harness/build glue so the same crashing input no longer crashes

Constraints:
- only modify files under `fuzz/` and `./done`
- keep fixes minimal
- do not run build/execute commands
- read-only exploration commands are allowed
- when crash/diagnostics include concrete file paths, issue explicit `Read and fix <path>[:line]` actions

MANDATORY:
- create `./done`
- write the key modified file path into `./done`
<!-- END TEMPLATE -->

<!-- TEMPLATE: crash_triage_with_hint -->
You are OpenCode. Classify crash root-cause using provided crash evidence.
Follow the STAGE SKILL loaded by the runner as primary instructions.
Use GLOBAL POLICY only as fallback.

Task:
- classify crash into exactly one label: `harness_bug` | `upstream_bug` | `inconclusive`
- use crash logs/reports as primary evidence
- do NOT modify source code in this stage

Constraints:
- read-only exploration commands are allowed
- do not run build/execute commands
- produce `crash_triage.json` with fields: `label`, `confidence`, `reason`, `evidence`
- `evidence` must be a non-empty array of concrete signal lines from logs/reports
- do not classify `upstream_bug` from sanitizer keywords alone; cite concrete call path/context evidence
- if evidence is insufficient, output `label=inconclusive` with explicit missing-evidence reason
- keep all instructions and outputs in English

Hint:
{{hint}}

MANDATORY:
- create `./done`
- write `crash_triage.json` into `./done`
<!-- END TEMPLATE -->

<!-- TEMPLATE: crash_analysis_with_hint -->
You are OpenCode. Analyze reproduced crash and decide whether it is a false positive harness issue.
Follow the STAGE SKILL loaded by the runner as primary instructions.
Use GLOBAL POLICY only as fallback.

Task:
- produce `crash_analysis.json` and `crash_analysis.md`
- output exactly one verdict: `false_positive` | `real_bug` | `unknown`
- use crash logs/reports as primary evidence
- if verdict is `false_positive`, clearly explain why it is harness/input-contract misuse

Constraints:
- read-only exploration commands are allowed
- do not run build/execute commands
- do not modify source code in this stage
- do not classify `real_bug` from sanitizer keywords alone; cite concrete stack/call-site evidence
- if evidence is insufficient, output `verdict=unknown` with explicit missing-evidence reason
- keep all instructions and outputs in English

Hint:
{{hint}}

MANDATORY:
- create `./done`
- write `crash_analysis.json` into `./done`
<!-- END TEMPLATE -->

<!-- TEMPLATE: fix_harness_after_run -->
You are OpenCode. The crash was triaged as a harness bug.
Follow the STAGE SKILL loaded by the runner as primary instructions.
Use GLOBAL POLICY only as fallback.

Task:
- fix fuzz harness/build glue so malformed inputs do not crash due to harness misuse
- keep fix minimal and evidence-driven

Constraints:
- only modify files under `fuzz/` and `./done`
- do not run build/execute commands
- read-only exploration commands are allowed
- when diagnostics include file paths, issue explicit `Read and fix <path>[:line]` actions
- prefer public/stable APIs; do not keep internal/private namespaces in harness logic unless `api_surface_exception` with evidence is present
- stale `./done` without fresh code diff is invalid
- pure no-op is invalid

MANDATORY:
- create `./done`
- write one key modified path under `fuzz/` into `./done`
<!-- END TEMPLATE -->

<!-- TEMPLATE: fix_crash_upstream_bug -->
You are OpenCode. Fix the upstream bug so the same crashing input no longer crashes.
Follow the STAGE SKILL loaded by the runner as primary instructions.
Use GLOBAL POLICY only as fallback.

Constraints:
- keep changes minimal and correct
- do not disable harness behavior
- do not run build/execute commands
- read-only exploration commands are allowed
- when crash/diagnostics include concrete file paths, issue explicit `Read and fix <path>[:line]` actions

MANDATORY:
- create `./done`
- write the key modified file path into `./done`
<!-- END TEMPLATE -->

<!-- TEMPLATE: plan_fix_targets_schema -->
You are coordinating a fuzz harness generation workflow.
Follow the STAGE SKILL loaded by the runner as primary instructions.
Use GLOBAL POLICY only as fallback.

Task:
- repair `fuzz/targets.json` to strict schema

Required schema:
- JSON array with at least one object
- each object includes non-empty: `name`, `api`, `lang`, `target_type`, `seed_profile`
- `lang`: `c-cpp|cpp|c|c++|java`
- `target_type`: `parser|decoder|archive|image|document|network|database|serializer|interpreter|generic`
- `seed_profile`: `parser-structure|parser-token|parser-format|parser-numeric|decoder-binary|archive-container|serializer-structured|document-text|network-message|generic`

Constraints:
- do not run build/execute commands
- read-only exploration commands are allowed
- when schema errors point to concrete files/lines, issue explicit `Read and fix <path>[:line]` actions

Current validation error:
{{schema_error}}

MANDATORY:
- create `./done`
- write `fuzz/targets.json` into `./done`
<!-- END TEMPLATE -->
