# OpenCode Prompt Templates

This file centralizes prompt templates used when calling `run_codex_command(...)`.
Use `{{var_name}}` placeholders for runtime substitution.

<!-- TEMPLATE: plan_with_hint -->
You are coordinating a fuzz harness generation workflow.
Perform the planning step and produce fuzz/PLAN.md and fuzz/targets.json as required.
PLAN.md must include a short summary and concrete next-step implementation suggestions for synthesis/build.

IMPORTANT: Do NOT run any build, compile, or test commands. Only create/edit files.
You MAY run any read-only commands for repository exploration (for example `find`, `grep`, `rg`, `cat`, `ls`).
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
MANDATORY OUTPUT CHECKLIST (must be true before writing `./done`):
- At least one harness source file under `fuzz/` (`*.c`/`*.cc`/`*.cpp`/`*.cxx`/`*.java`)
- `fuzz/build.py` or `fuzz/build.sh`
- `fuzz/README.md`
- `fuzz/repo_understanding.json`
- `fuzz/build_strategy.json`

If blocked, still create minimal valid versions of the missing required files now; do not exit with partial scaffold.
Execution strategy requirement:
- Do not optimize for early artifact output.
- First read enough repository/build context to write `fuzz/repo_understanding.json` and a concrete `fuzz/build_strategy.json`.
- Also write a concise `fuzz/build_runtime_facts.json` that captures the actual fuzzer entry/link strategy for this environment.
- Only after those understanding artifacts are grounded in repository facts should you create or update the harness and `fuzz/build.py`.

IMPORTANT: Do NOT run any build, compile, or test commands. Only create/edit files.
You MAY run any read-only commands for repository exploration (for example `find`, `grep`, `rg`, `cat`, `ls`).
MANDATORY: you MUST create `./done` before finishing this step.
Write `fuzz/out/` into `./done` (single line). Missing `./done` means this step fails.
If progress stalls, still deliver repository-understanding artifacts first, then the smallest scaffold consistent with them.

If external system dependencies are required, write vcpkg port names (one per line) to fuzz/system_packages.txt.
Use package names only; no shell commands.
Use canonical vcpkg port names only (not generic/apt names). Examples:
- `zlib` (NOT `z`)
- `bzip2` (NOT `bz2`)
- `liblzma` (NOT `lzma`)
- `lz4` (valid as-is)
Non-root runtime rule: do not enable install-to-system-dir flows in build scripts (for example avoid `-DENABLE_INSTALL=ON`, `cmake --install`, or `--target install`); link libraries directly from build tree artifacts.
Avoid forcing C++ standard library selection flags (for example: do not add `-stdlib=libc++`).
If the upstream source contains a `main` symbol, handle symbol conflict in build flags (for example `-Dmain=vuln_main`) so libFuzzer link can succeed.
Hard requirements:
- Default to the first runtime-viable target in `fuzz/selected_targets.json`; drift only when repository facts prove it is not directly fuzzable.
- Add an early input-size guard in the harness entrypoint before heavy parsing/work (for C/C++ style harnesses, prefer `if (size > 8192) return 0;` unless a smaller cap is clearly required).
- The harness, `fuzz/README.md`, and `fuzz/build_strategy.json` must agree on one final external/library API. Do not call a local helper/checker/wrapper the final target.
- In `fuzz/build.py`, do not hardcode a single static library path.
- Use runtime discovery with command execution (for example `subprocess.run(["find", str(REPO_ROOT), "-name", "*.a", "-type", "f"], ...)`) plus a helper like `find_static_lib(...)`.
- Exclude obvious test-only artifacts and verify the selected library path exists before linking.
- In `fuzz/build.py`, define default CMake args and apply them by default:
  - `DEFAULT_CMAKE_ARGS = [`
  - `    "-DENABLE_TEST=OFF",`
  - `    "-DENABLE_INSTALL=OFF",`
  - `]`
  Ensure these defaults are included in CMake configure command unless repository facts explicitly require overrides.
- Do not silently accept optional dependency downgrades. When CMake/build output indicates missing key libraries (for example zlib/bzip2/lzma/lz4/zstd/openssl/libxml2/expat), declare matching vcpkg ports in `fuzz/system_packages.txt` and keep build configuration aligned with those dependencies.
- If `fuzz/observed_target.json` exists, keep new outputs consistent with it unless the harness target actually changes.
- If you drift, record the rejected original target and the replacement rationale in `fuzz/repo_understanding.json`.
- You may use a repository-provided fuzz target only when its exact real target name is documented in both `fuzz/repo_understanding.json` and `fuzz/build_strategy.json`. Never guess names such as `<name>-fuzzer` or `<name>_fuzzer`.
- `fuzz/repo_understanding.json` must stay limited to: `build_system`, `candidate_library_inputs`, `chosen_target_api`, `chosen_target_reason`, `rejected_targets`, `extra_sources`, `include_dirs`, `fuzzer_entry_strategy`, `constraints`, `evidence`, plus optional `repo_fuzz_targets`, `selected_repo_target`.
- `fuzz/build_strategy.json` must stay limited to: `build_system`, `build_mode`, `library_targets`, `library_artifacts`, `include_dirs`, `extra_sources`, `fuzzer_entry_strategy`, `reason`, `evidence`, `repo_fuzz_targets`, `selected_repo_target`. `build_mode` must be `repo_target`, `library_link`, or `custom_script`.
- `fuzz/build_runtime_facts.json` must stay limited to: `compiler`, `fuzzer_entry_strategy`, `fuzzer_link_flags`, `forbidden_link_flags`, `sanitizers`, `reason`, `evidence`.
- `evidence` must not be empty. `chosen_target_reason` must explain why the chosen target is the best runtime entrypoint. `rejected_targets` must list the near-miss candidates and why they were rejected.
- Use `fuzz/build_runtime_facts.json` to record environment-specific facts such as “use `-fsanitize=fuzzer`” and “do not use `-lfuzzer`” when applicable.
- Seed design must be target-specific: in `fuzz/README.md`, enumerate at least 3 concrete seed families tied to target semantics, and each family must map to an actual corpus example or planned corpus file pattern.
- `fuzz/README.md` MUST contain these exact fields with values that match the actual harness:
  - `Selected target: ...`
  - `Final target: ...`
  - `Technical reason: ...`
  - `Relation: ...`
  - `Harness file: ...`
- FIRST-PASS QUALITY GATE (non-optional): before writing `./done`, verify `fuzz/build.py`, `fuzz/build_strategy.json`, and `fuzz/system_packages.txt` are internally consistent for the first build attempt.
- If `fuzz/build.py` links any of `-lz`, `-lbz2`, `-llzma`, `-llz4`, `-lzstd`, `-lcrypto`, `-lssl`, `-lxml2`, or `-lexpat`, you MUST declare matching vcpkg ports in `fuzz/system_packages.txt` in the same attempt.
- If the build strategy intentionally disables those features, remove their matching link flags from `fuzz/build.py` in the same attempt. Never keep contradictory "feature disabled but still linked" states.

Additional instruction from coordinator:
{{hint}}
<!-- END TEMPLATE -->

<!-- TEMPLATE: synthesize_complete_scaffold -->
You are coordinating a fuzz harness generation workflow.
There is already a partial scaffold under `fuzz/`. Do NOT restart from scratch.

Task: complete the missing scaffold items only:
{{missing_items}}

MANDATORY OUTPUT CHECKLIST (must be true before writing `./done`):
- At least one harness source file under `fuzz/`
- `fuzz/build.py` or `fuzz/build.sh`
- `fuzz/README.md`
- `fuzz/repo_understanding.json`
- `fuzz/build_strategy.json`

Rules:
- Preserve existing harness/build files unless a minimal fix is required.
- If a harness source file already exists, keep it and add/fix the missing build glue around it.
- Prioritize creating `fuzz/build.py` first if it is missing.
- `README.md` and `.options` files should be added after the harness and build script are in place.
- Prefer the selected runtime target. If you switch, record the rejected original target and repository-grounded reason in `fuzz/repo_understanding.json`.
- Keep these exact `fuzz/README.md` fields aligned with the actual harness: `Selected target: ...`, `Final target: ...`, `Technical reason: ...`, `Relation: ...`, `Harness file: ...`.
- If `fuzz/observed_target.json` exists, treat it as the execution truth source and keep `README.md`, harness filenames, and build scaffold consistent with it.
- Do not describe a local helper/checker/wrapper as the final target when the harness actually calls an external/library API.
- Keep only real harness source files in `fuzz/build.py` / `fuzz/build.sh`; never reference missing scaffold files.
- Do not add guessed repository fuzz target invocations such as `--target xxx-fuzzer` or `make xxx_fuzzer` unless that exact target is already documented as real in `fuzz/repo_understanding.json` and `fuzz/build_strategy.json`.
- Ensure `fuzz/repo_understanding.json` exists and stays consistent with the actual external build path before considering the scaffold complete.
- Ensure `fuzz/repo_understanding.json` explains both the chosen path and the rejected near-miss paths; avoid high-level repository summaries with no execution consequences.
- Keep `fuzz/build_strategy.json` aligned with an external scaffold strategy and record an explicit `fuzzer_entry_strategy`.
- Keep `fuzz/build_runtime_facts.json` aligned with the actual compiler/runtime assumptions used by `fuzz/build.py`.
- Do NOT run any build, compile, or test commands. Only create/edit files.
- You MAY run any read-only commands for repository exploration (for example `find`, `grep`, `rg`, `cat`, `ls`).
- If progress stalls, prioritize missing understanding files before writing fallback scaffold files, then write `fuzz/out/` into `./done`.
- If `fuzz/README.md` is missing, create it with required fields (`Selected target`, `Final target`, `Technical reason`, `Relation`, `Harness file`).
- If `fuzz/build_strategy.json` is missing, create a minimal valid JSON strategy aligned with current harness/build path.

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

CRITICAL: Do NOT run build/execute commands (no cmake, make, python build scripts, bash build wrappers, gcc/clang compile runs, etc.).
You MAY run any read-only commands for repository exploration (for example `find`, `grep`, `rg`, `cat`, `ls`).
Only edit source files for the fix. The build will be executed by the workflow after you finish.

Constraints:
- Keep changes minimal; avoid refactors
- Only edit files under `fuzz/`
- The only allowed file outside `fuzz/` is `./done` (sentinel). Any other path change is rejected by the workflow.
- Do not modify repository source/build files outside `fuzz/` (for example: `*.c`, `*.cc`, `*.cpp`, `*.h`, `CMakeLists.txt`, `Makefile`, `configure`).
- If external system deps are required, declare vcpkg port names in fuzz/system_packages.txt (one per line, comments allowed, no shell commands)
- Use canonical vcpkg port names only in `fuzz/system_packages.txt`. Never write generic aliases like `z`, `bz2`, or `lzma`; use `zlib`, `bzip2`, `liblzma`.
- Non-root runtime rule: never add install-to-system-dir build steps (`-DENABLE_INSTALL=ON`, `cmake --install`, `--target install`). Build and link from workspace artifacts only.
- If you change `fuzz/system_packages.txt`, still finish all other necessary `fuzz/` edits in the same attempt. Do not stop after only declaring packages if `fuzz/build.py` or harness glue also needs changes.
- Treat `fuzz/system_packages.txt` as “requires a fresh build job to validate”. Do not assume the current container can verify those package additions.
- Do not force C++ stdlib flags like `-stdlib=libc++` in this environment.
- Ensure the harness entrypoint contains an explicit early input-size guard (default 8192 bytes unless repository facts require stricter).
- If target sources define `main`, resolve libFuzzer main conflict (for example add `-Dmain=vuln_main` in compile flags).
- You may repair the build by switching to a repository-provided fuzz target only when the exact target is documented as real in `fuzz/repo_understanding.json` and `fuzz/build_strategy.json`; never guess target names.
- If `fuzz/repo_understanding.json` is missing or weak, repair it first.
- If the selected target and observed target disagree, repair that mismatch before incremental build tweaks.
- Keep `fuzz/repo_understanding.json` concrete: chosen target, rejected alternatives, required libraries/sources, exact fuzzer entry strategy.
- Keep `fuzz/build_strategy.json` aligned with `library_link` or `custom_script`, and ensure it records an explicit `fuzzer_entry_strategy`.
- If `fuzz/build_runtime_facts.json` is missing or weak, repair it so it states the real fuzzer entry strategy, required link flags, forbidden link flags, and sanitizer set for this environment.
- If the current corpus/seed design is too generic, tighten `fuzz/README.md` so it names concrete seed families tied to target semantics.
- Full build output from the previous failed attempts is available in `{{build_log_file}}`.
- You MUST read `{{build_log_file}}` before editing, and base your fix on that full log (not only short tails).
- If this attempt cannot produce a valid fix, do NOT exit with sentinel only; you must provide the smallest verifiable patch under `fuzz/`.
- You must explicitly address the current error signature and avoid repeating previously rejected no-op patterns.
- If the error indicates a built library cannot be found (for example `Could not find <lib> library`), treat it as an artifact-discovery bug first: repair `fuzz/build.py` to search nested build output directories and versioned shared libraries (`.so.*`) before failing.
- Avoid assumptions that libraries are emitted at build root; support common layouts like `build/<module>/lib<name>.a` and `build/<module>/lib<name>.so.*`.
- Prefer a reusable helper (`find_static_lib`) that uses runtime command discovery (for example `find`) over ad-hoc one-off hardcoded paths.
- If logs show `Could NOT find ...` for key optional libraries, do not treat that as acceptable completion; update `fuzz/system_packages.txt` with the matching vcpkg ports and make the build script consume them.
- If logs show linker errors like `cannot find -l...`, you MUST create or update `fuzz/system_packages.txt` in the same attempt (port mapping examples: `-lz`→`zlib`, `-lbz2`→`bzip2`, `-llzma`→`liblzma`, `-llz4`→`lz4`, `-lcrypto/-lssl`→`openssl`, `-lxml2`→`libxml2`, `-lexpat`→`expat`).
- First repair pass must be build-ready: do not defer dependency declaration to a later round when the current log already names missing link libraries.
- If the failure is due to missing tools/packages (for example `aclocal`, `autoconf`, `automake`, `libtool`, missing `-dev` packages), prefer:
  1. declare the required vcpkg ports in `fuzz/system_packages.txt`
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

CRITICAL: Do NOT run build/execute commands. You MAY run any read-only commands for repository exploration (for example `find`, `grep`, `rg`, `cat`, `ls`).
Only edit source files.

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

CRITICAL: Do NOT run build/execute commands. You MAY run any read-only commands for repository exploration (for example `find`, `grep`, `rg`, `cat`, `ls`).
Only edit source files.

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
- You MAY run any read-only commands for repository exploration (for example `find`, `grep`, `rg`, `cat`, `ls`).
- Only edit files
- `fuzz/targets.json` must be plain JSON only; no Markdown fences, no wrapper object, no empty array
- MANDATORY: create `./done` when finished, and write `fuzz/targets.json` into it.
- Missing `./done` means this step fails.

Current validation error:
{{schema_error}}

When finished, ensure `fuzz/targets.json` is valid JSON and matches the schema.
<!-- END TEMPLATE -->
