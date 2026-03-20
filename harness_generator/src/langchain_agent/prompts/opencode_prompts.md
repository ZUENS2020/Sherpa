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

Stage requirements:
- Do NOT run build/execute commands.
- Read-only exploration commands are allowed.
- Keep outputs aligned with `fuzz/selected_targets.json`; if target drifts, document rejection reason.
- Keep `fuzz/observed_target.json` consistent with scaffold when present.
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

Constraints:
- Do NOT run build/execute commands.
- Read-only exploration commands are allowed.
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
- keep changes minimal and evidence-driven from `{{build_log_file}}`
- when diagnostics still fail, pure no-op is invalid; produce a concrete patch
- if the same error signature repeats, change strategy instead of repeating identical edits
- keep `fuzz/repo_understanding.json`, `fuzz/build_strategy.json`, and `fuzz/build_runtime_facts.json` consistent
- if missing dependencies are indicated by build evidence, update `fuzz/system_packages.txt` with canonical vcpkg names
- keep build output aligned with `fuzz/execution_plan.json` target coverage (do not regress to single-target build when multi-target execution is required)

Coordinator instruction:
{{codex_hint}}

MANDATORY:
- create `./done`
- write `fuzz/build.py` into `./done` (single line)
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

MANDATORY:
- create `./done`
- write the key modified file path into `./done`
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

Current validation error:
{{schema_error}}

MANDATORY:
- create `./done`
- write `fuzz/targets.json` into `./done`
<!-- END TEMPLATE -->
