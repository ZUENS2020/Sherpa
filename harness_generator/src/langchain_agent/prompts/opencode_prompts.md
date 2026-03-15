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

Use both `fuzz/antlr_plan_context.json` and `fuzz/target_analysis.json` as grounding when available.
Planning policy:
- Prefer runtime-executable, buildable, fuzzable entrypoints first.
- Put the best runtime target first in `fuzz/targets.json`.
- If `fuzz/target_analysis.json` ranks a public/runtime API above compile-time/detail/helper APIs, preserve that ordering in `fuzz/targets.json`.
- Do not put compile-only, constexpr-only, detail/helper, local utility, or wrapper-only APIs ahead of a clearly viable runtime entrypoint.
- If no direct runtime entrypoint exists, still choose the closest executable target and keep likely runtime replacements near the top.

`fuzz/targets.json` format is STRICT:
- MUST be plain JSON, not Markdown and not fenced code blocks
- MUST be a JSON array, not an object
- MUST contain at least one target object
- Each target object MUST include non-empty string fields: `name`, `api`, `lang`, `target_type`, `seed_profile`
- `lang` MUST be one of: `c-cpp`, `cpp`, `c`, `c++`, `java`
- `target_type` MUST be one of: `parser`, `decoder`, `archive`, `image`, `document`, `network`, `database`, `serializer`, `interpreter`, `generic`
- `seed_profile` MUST be one of: `parser-structure`, `parser-token`, `parser-format`, `parser-numeric`, `decoder-binary`, `archive-container`, `serializer-structured`, `document-text`, `network-message`, `generic`
- Choose the most specific target type; do not invent new type names
- Choose the most specific seed profile from the fixed enum; do not invent new profile names
- If unsure, overwrite the whole file with the smallest valid array instead of leaving partial/invalid JSON
- Never emit an empty array
- Never wrap the array inside another object such as `{ "targets": [...] }`

Minimal valid example:
[
  {
    "name": "yaml_parser_parse",
    "api": "yaml_parser_parse",
    "lang": "c-cpp",
    "target_type": "parser",
    "seed_profile": "parser-structure"
  }
]

Additional instruction from coordinator:
{{hint}}
<!-- END TEMPLATE -->

<!-- TEMPLATE: synthesize_with_hint -->
You are coordinating a fuzz harness generation workflow.
Perform the synthesis step: create harness + fuzz/build.py + build glue under fuzz/.
Execution strategy requirement:
- Do not optimize for early artifact output.
- First read enough repository/build context to write `fuzz/repo_understanding.json` and a concrete `fuzz/build_strategy.json`.
- Only after those understanding artifacts are grounded in repository facts should you create or update the harness and `fuzz/build.py`.

IMPORTANT: Do NOT run any build, compile, or test commands. Only create/edit files.
MANDATORY: you MUST create `./done` before finishing this step.
Write `fuzz/out/` into `./done` (single line). Missing `./done` means this step fails.
If progress stalls, still deliver repository-understanding artifacts first, then the smallest scaffold consistent with them.

If external system dependencies are required, write package names (one per line) to fuzz/system_packages.txt.
Use package names only; no shell commands.
Avoid forcing C++ standard library selection flags (for example: do not add `-stdlib=libc++`).
If the upstream source contains a `main` symbol, handle symbol conflict in build flags (for example `-Dmain=vuln_main`) so libFuzzer link can succeed.
Target-selection policy:
- Treat `fuzz/selected_targets.json` as the preferred plan, but only use its first target if it is a viable runtime fuzz entrypoint.
- Prefer a directly callable public/runtime API over compile-time, detail/helper, constexpr-only, or local utility functions.
- If you must drift from the selected target, choose the nearest runtime-executable replacement target that exercises the same parsing/formatting/decoding path.
- The final target must be the actual external/library API exercised by the harness, not a local helper, checker, wrapper utility, or placeholder name.
- Your harness, `fuzz/README.md`, and build scaffold must all agree on the same final observed target and harness filename.
- If `fuzz/observed_target.json` already exists, treat it as the execution truth source and keep new outputs consistent with it unless the harness itself changes.
- You may invoke a repository-provided fuzz target only if you have first identified the exact real target name from repository files/build metadata and recorded it in both `fuzz/repo_understanding.json` and `fuzz/build_strategy.json`.
- Never guess target names such as `<name>-fuzzer`, `<name>_fuzzer`, or infer that `test/fuzzing`, `main.cc`, or `fuzzer-common.h` automatically means a buildable fuzz target exists.
- Create `fuzz/repo_understanding.json` and keep it limited to these fields: `build_system`, `candidate_library_inputs`, `chosen_target_api`, `chosen_target_reason`, `extra_sources`, `include_dirs`, `fuzzer_entry_strategy`, `constraints`, `evidence`.
- `fuzz/repo_understanding.json` must be grounded in actual repository files or build metadata; `evidence` must not be empty.
- If you choose a repository-provided fuzz target, add these optional fields to `fuzz/repo_understanding.json`: `repo_fuzz_targets`, `selected_repo_target`.
- Create `fuzz/build_strategy.json` and keep it limited to these fields: `build_system`, `build_mode`, `library_targets`, `library_artifacts`, `include_dirs`, `extra_sources`, `fuzzer_entry_strategy`, `reason`, `evidence`, `repo_fuzz_targets`, `selected_repo_target`.
- `build_mode` must be `repo_target`, `library_link`, or `custom_script`.
- `fuzz/build_strategy.json` must reflect the repository understanding above; do not leave `build_system` as `unknown` when repository facts support a more specific value.
- `fuzz/README.md` MUST contain these exact fields with values that match the actual harness:
  - `Selected target: ...`
  - `Final target: ...`
  - `Technical reason: ...`
  - `Relation: ...`
  - `Harness file: ...`

Additional instruction from coordinator:
{{hint}}
<!-- END TEMPLATE -->

<!-- TEMPLATE: synthesize_complete_scaffold -->
You are coordinating a fuzz harness generation workflow.
There is already a partial scaffold under `fuzz/`. Do NOT restart from scratch.

Task: complete the missing scaffold items only:
{{missing_items}}

Rules:
- Preserve existing harness/build files unless a minimal fix is required.
- If a harness source file already exists, keep it and add/fix the missing build glue around it.
- Prioritize creating `fuzz/build.py` first if it is missing.
- `README.md` and `.options` files should be added after the harness and build script are in place.
- Treat `fuzz/selected_targets.json` as a preferred plan, not an unconditional hard stop.
- Prefer the selected target when it is runtime-executable.
- If you switch to another target, prefer the nearest runtime-executable replacement target and record these exact fields in `fuzz/README.md`:
  - `Selected target: ...`
  - `Final target: ...`
  - `Technical reason: ...`
  - `Relation: ...`
- Also record:
  - `Harness file: ...`
- If `fuzz/observed_target.json` exists, treat it as the execution truth source and keep `README.md`, harness filenames, and build scaffold consistent with it.
- Do not describe a local helper/checker/wrapper as the final target when the harness actually calls an external/library API.
- Keep only real harness source files in `fuzz/build.py` / `fuzz/build.sh`; never reference missing scaffold files.
- Do not add guessed repository fuzz target invocations such as `--target xxx-fuzzer` or `make xxx_fuzzer` unless that exact target is already documented as real in `fuzz/repo_understanding.json` and `fuzz/build_strategy.json`.
- Ensure `fuzz/repo_understanding.json` exists and stays consistent with the actual external build path before considering the scaffold complete.
- Keep `fuzz/build_strategy.json` aligned with an external scaffold strategy and record an explicit `fuzzer_entry_strategy`.
- Do NOT run any build, compile, or test commands. Only create/edit files.
- If progress stalls, prioritize missing understanding files before writing fallback scaffold files, then write `fuzz/out/` into `./done`.

MANDATORY: you MUST create `./done` before finishing this step.
Write `fuzz/out/` into `./done` (single line). Missing `./done` means this step fails.
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
- If you change `fuzz/system_packages.txt`, still finish all other necessary `fuzz/` edits in the same attempt. Do not stop after only declaring packages if `fuzz/build.py` or harness glue also needs changes.
- Treat `fuzz/system_packages.txt` as “requires a fresh build job to validate”. Do not assume the current container can verify those package additions.
- Do not force C++ stdlib flags like `-stdlib=libc++` in this environment.
- If target sources define `main`, resolve libFuzzer main conflict (for example add `-Dmain=vuln_main` in compile flags).
- You may repair the build by switching to a repository-provided fuzz target only when the exact target is documented as real in `fuzz/repo_understanding.json` and `fuzz/build_strategy.json`; never guess target names.
- If `fuzz/repo_understanding.json` is missing or weak, repair it first and make the build scaffold match that understanding.
- Keep `fuzz/build_strategy.json` aligned with `library_link` or `custom_script`, and ensure it records an explicit `fuzzer_entry_strategy`.
- Full build output from the previous failed attempts is available in `{{build_log_file}}`.
- You MUST read `{{build_log_file}}` before editing, and base your fix on that full log (not only short tails).
- If this attempt cannot produce a valid fix, do NOT exit with sentinel only; you must provide the smallest verifiable patch under `fuzz/`.
- You must explicitly address the current error signature and avoid repeating previously rejected no-op patterns.
- If the failure is due to missing tools/packages (for example `aclocal`, `autoconf`, `automake`, `libtool`, missing `-dev` packages), prefer:
  1. declare the required packages in `fuzz/system_packages.txt`
  2. make any matching `fuzz/build.py` adjustments needed for the new environment
  3. avoid fake fixes that would still fail before packages are installed

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
- each object must include non-empty string keys: `name`, `api`, `lang`, `target_type`, `seed_profile`
- `lang` must be one of: c-cpp, cpp, c, c++, java
- `target_type` must be one of: parser, decoder, archive, image, document, network, database, serializer, interpreter, generic
- `seed_profile` must be one of: parser-structure, parser-token, parser-format, parser-numeric, decoder-binary, archive-container, serializer-structured, document-text, network-message, generic

Constraints:
- Keep edits minimal
- Do NOT run any build, compile, or test commands
- Only edit files
- `fuzz/targets.json` must be plain JSON only; no Markdown fences, no wrapper object, no empty array
- MANDATORY: create `./done` when finished, and write `fuzz/targets.json` into it.
- Missing `./done` means this step fails.

Current validation error:
{{schema_error}}

When finished, ensure `fuzz/targets.json` is valid JSON and matches the schema.
<!-- END TEMPLATE -->
