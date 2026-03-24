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
- When diagnostics/context include concrete file paths, prioritize explicit actions in the form `Read and fix <path>[:line]`.
- `fuzz/targets.json` must be plain JSON array with at least one item.
- Each item must include non-empty strings: `name`, `api`, `lang`, `target_type`, `seed_profile`.
- `lang` must be one of: `c-cpp`, `cpp`, `c`, `c++`, `java`.
- `target_type` must be one of: `parser`, `decoder`, `archive`, `image`, `document`, `network`, `database`, `serializer`, `interpreter`, `generic`.
- `seed_profile` must be one of: `parser-structure`, `parser-token`, `parser-format`, `parser-numeric`, `decoder-binary`, `archive-container`, `serializer-structured`, `document-text`, `network-message`, `generic`.
- Keep runtime-viable/public entrypoints first.
- Add execution metadata in `fuzz/selected_targets.json` semantics:
  - `execution_priority` (higher priority first, default top 3)
  - `must_run` for high-value parser/archive/decoder targets.

MANDATORY:
- create `./done`
- write `fuzz/PLAN.md` into `./done` (single line)

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
- prefer public/stable APIs for harness logic; use internal/private APIs only when no viable public alternative exists, and document evidence via `api_surface_exception`

Constraints:
- Do NOT run build/execute commands.
- Read-only exploration commands are allowed.
- When diagnostics/context include concrete file paths, prioritize explicit actions in the form `Read and fix <path>[:line]`.
- if diagnostics include `non_public_api_usage`, replace offending symbols first before any broader refactor

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
- When diagnostics/context include concrete file paths, prioritize explicit actions in the form `Read and fix <path>[:line]`.
- do not produce doc-only adjustments disconnected from next build/run outcomes

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
- When diagnostics/context include concrete file paths, prioritize explicit actions in the form `Read and fix <path>[:line]`.
- Keep outputs aligned with `fuzz/selected_targets.json`; if target drifts, document rejection reason.
- Keep `fuzz/observed_target.json` consistent with scaffold when present.
- Prefer public/stable repository APIs for harness logic. Avoid internal/private namespaces such as `detail`, `_internal`, or equivalent implementation-only symbols unless diagnostics prove they are the only valid entrypoints.
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
  - runtime artifact discovery (do not hardcode a single static library path)
  - multi-target build intent: avoid single-target-only output when execution plan has multiple targets

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
- change strategy if previous attempt signatures repeat
- avoid no-op doc-only edits
- keep target/build fields consistent across README + JSONs + build script
- update `fuzz/harness_index.json` so execution targets map to real harness files; do not leave stale/missing mappings
- prefer public/stable APIs; internal/private APIs require explicit `api_surface_exception` with evidence in `fuzz/repo_understanding.json`
- Do NOT run build/execute commands
- Read-only exploration commands are allowed
- if diagnostics include `non_public_api_usage`, replace offending symbols first and touch the offending harness file(s)

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
- explicitly map selected vs observed runtime target relation in README
- preserve crash-path semantics; do not “fix” by disabling harness behavior
- avoid no-op doc-only edits
- prefer public/stable APIs; internal/private APIs require explicit `api_surface_exception` with evidence in `fuzz/repo_understanding.json`
- update `fuzz/harness_index.json` so execution targets map to real harness files; do not leave stale/missing mappings
- Do NOT run build/execute commands
- Read-only exploration commands are allowed
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
- consume `SeedFeedback` and `HarnessFeedback` first; apply at least one change linked to these signals
- include and apply at least one material strategy change from previous cycle
- avoid no-op doc-only edits
- keep selected/final target, execution plan, and harness index consistent
- Do NOT run build/execute commands
- Read-only exploration commands are allowed

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
