# OpenCode Prompt Templates

This file centralizes prompt templates used when calling `run_codex_command(...)`.
Use `{{var_name}}` placeholders for runtime substitution.

<!-- TEMPLATE: plan_with_hint -->
You are coordinating a fuzz harness generation workflow.
Perform the planning step and produce fuzz/PLAN.md and fuzz/targets.json as required.
PLAN.md must include a short summary and concrete next-step implementation suggestions for synthesis/build.

IMPORTANT: Do NOT run any build, compile, or test commands. Only create/edit files.
MANDATORY: you MUST create `./done` before finishing this step.
Write `fuzz/PLAN.md` into `./done` (single line). Missing `./done` means this step fails.
If progress stalls, still deliver minimal valid artifacts now: create `fuzz/PLAN.md` and schema-valid `fuzz/targets.json`, then write `fuzz/PLAN.md` into `./done`.

Additional instruction from coordinator:
{{hint}}
<!-- END TEMPLATE -->

<!-- TEMPLATE: synthesize_with_hint -->
You are coordinating a fuzz harness generation workflow.
Perform the synthesis step: create harness + fuzz/build.py + build glue under fuzz/.

IMPORTANT: Do NOT run any build, compile, or test commands. Only create/edit files.
MANDATORY: you MUST create `./done` before finishing this step.
Write `fuzz/out/` into `./done` (single line). Missing `./done` means this step fails.
If progress stalls, still deliver minimal valid artifacts now: create at least one harness source file and `fuzz/build.py`, then write `fuzz/out/` into `./done`.

If external system dependencies are required, write package names (one per line) to fuzz/system_packages.txt.
Use package names only; no shell commands.
Avoid forcing C++ standard library selection flags (for example: do not add `-stdlib=libc++`).
If the upstream source contains a `main` symbol, handle symbol conflict in build flags (for example `-Dmain=vuln_main`) so libFuzzer link can succeed.

Additional instruction from coordinator:
{{hint}}
<!-- END TEMPLATE -->

<!-- TEMPLATE: fix_build_execute -->
You are OpenCode operating inside a Git repository.
Task: fix the fuzz harness/build source code so the build will pass when run later.
Environment note: build/fuzz runs in a separate runtime container (typically `sherpa-fuzz-cpp:latest` or `sherpa-fuzz-java:latest`), not this OpenCode environment.

Goal (will be verified by a separate automated system — do NOT run these yourself):
- `(cd fuzz && python build.py)` should complete successfully
- fuzz/out/ should contain at least one runnable fuzzer binary

CRITICAL: Do NOT run any commands (no cmake, make, python, bash, gcc, clang, etc.).
Only edit source files. The build will be executed by the workflow after you finish.

Constraints:
- Keep changes minimal; avoid refactors
- Only edit files under `fuzz/`
- The only allowed file outside `fuzz/` is `./done` (sentinel). Any other path change is rejected by the workflow.
- Do not modify repository source/build files outside `fuzz/` (for example: `*.c`, `*.cc`, `*.cpp`, `*.h`, `CMakeLists.txt`, `Makefile`, `configure`).
- If external system deps are required, declare package names in fuzz/system_packages.txt (one per line, comments allowed, no shell commands)
- Do not force C++ stdlib flags like `-stdlib=libc++` in this environment.
- If target sources define `main`, resolve libFuzzer main conflict (for example add `-Dmain=vuln_main` in compile flags).
- Full build output from the previous failed attempts is available in `{{build_log_file}}`.
- You MUST read `{{build_log_file}}` before editing, and base your fix on that full log (not only short tails).

Coordinator instruction:
{{codex_hint}}

If you are blocked, do not wait idly. Produce the smallest valid `fuzz/` fix set immediately and finish with `./done`.
When finished, write `fuzz/build.py` into `./done`.
If `./done` is missing, this step is treated as failed.
<!-- END TEMPLATE -->

<!-- TEMPLATE: fix_crash_harness_error -->
You are OpenCode. The crash was diagnosed as a HARNESS ERROR.
Task: fix the fuzz harness/build glue so the crash no longer happens for the same input.
Environment note: build/fuzz runs in a separate runtime container (typically `sherpa-fuzz-cpp:latest` or `sherpa-fuzz-java:latest`), not this OpenCode environment.

Constraints:
- Only modify files under fuzz/ or minimal build glue required for the harness.
- Do not change upstream/product code unless absolutely required.
- Keep changes minimal and targeted.

Goal (will be verified by a separate automated system — do NOT run these yourself):
- The fuzzer should build successfully.
- Running the fuzzer with the previous crashing input should no longer crash.

CRITICAL: Do NOT run any commands. Only edit source files.

When finished, write the key file you modified into ./done.
If `./done` is missing, this step is treated as failed.
<!-- END TEMPLATE -->

<!-- TEMPLATE: fix_crash_upstream_bug -->
You are OpenCode. Fix the underlying bug in the target repository so the crash no longer occurs.
Environment note: build/fuzz runs in a separate runtime container (typically `sherpa-fuzz-cpp:latest` or `sherpa-fuzz-java:latest`), not this OpenCode environment.

Constraints:
- Keep changes minimal and focused on correctness/security.
- Do NOT disable the harness or skip input processing.
- Avoid broad refactors.

Goal (will be verified by a separate automated system — do NOT run these yourself):
- The fuzzer should build successfully.
- The previous crashing input should no longer crash.

CRITICAL: Do NOT run any commands. Only edit source files.

When finished, write the key file you modified into ./done.
If `./done` is missing, this step is treated as failed.
<!-- END TEMPLATE -->

<!-- TEMPLATE: plan_fix_targets_schema -->
You are coordinating a fuzz harness generation workflow.
Repair `fuzz/targets.json` so it passes strict schema checks.

Required schema:
- JSON array with at least one object
- each object must include non-empty string keys: `name`, `api`, `lang`
- `lang` must be one of: c-cpp, cpp, c, c++, java

Constraints:
- Keep edits minimal
- Do NOT run any build, compile, or test commands
- Only edit files
- MANDATORY: create `./done` when finished, and write `fuzz/targets.json` into it.
- Missing `./done` means this step fails.

Current validation error:
{{schema_error}}

When finished, ensure `fuzz/targets.json` is valid JSON and matches the schema.
<!-- END TEMPLATE -->
