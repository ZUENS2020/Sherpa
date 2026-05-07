---
name: synthesize
description: Generate complete fuzz scaffold artifacts aligned to selected targets and execution plan.
compatibility: opencode
metadata:
  stage: synthesize
  owner: sherpa
---

## What this skill does
Builds a complete `fuzz/` scaffold from planning artifacts, including harness source, build script, and runtime facts.

## When to use this skill
Use this skill in the primary `synthesize` stage after `plan`.

## Required inputs
- `fuzz/PLAN.md`
- `fuzz/targets.json`
- `fuzz/execution_plan.json` (if present)
- `fuzz/selected_targets.json` (if present)
- `fuzz/analysis_context.json` (if present)
  - consume `analysis_evidence.vuln_candidate_inventory[]` when available
  - consume `attack_hint.trigger_condition`, `attack_hint.key_code_path`, `attack_hint.boundary_values`, `attack_hint.vuln_category`, `attack_hint.sanitizer_hint`
- `fuzz/observed_target.json` (if present)
- MCP tools from task-scoped PromeFuzz companion (if available), including preprocessor and semantic tools
  - code navigation: `list_definitions`, `read_definition`, `read_source`, `find_references`
  - preprocessor: `run_ast_preprocessor`, `extract_api_functions`, `build_library_callgraph`
  - semantic (if enabled): `init_knowledge_base`, `retrieve_documents`, `comprehend_*`

## Required outputs
- at least one harness source file under `fuzz/` (`*.c`, `*.cc`, `*.cpp`, `*.cxx`, or `*.java`) before docs/json completion
- `fuzz/build.py` or `fuzz/build.sh`
- `fuzz/README.md`
- `fuzz/repo_understanding.json`
- `fuzz/build_strategy.json`
- `fuzz/build_runtime_facts.json`
- `fuzz/harness_index.json` aligned to `fuzz/execution_plan.json`

## Workflow
1. Query MCP evidence first when MCP is available (code-navigation first, preprocessor second, semantic evidence third).
2. Read planning artifacts and lock target alignment first.
3. When vulnerability candidates exist, use the highest-priority `attack_hint` values to shape harness input flow and boundary-case seeds.
3. Create harness source(s) before scaffold documentation (`harness-first contract`).
4. Create build glue with runtime artifact discovery and compiler-by-suffix behavior.
5. Create README/JSON strategy files with consistent selected/final target semantics.
6. Validate execution plan and harness index mappings.

## Key template contracts
### `fuzz/repo_understanding.json`
- Must include non-empty:
  - `build_system`
  - `chosen_target_api`
  - `chosen_target_reason`
  - `fuzzer_entry_strategy`
  - `evidence` (non-empty array)
- `chosen_target_api` must be a target API identifier, not a harness path.
- Forbidden examples: `fuzz/xxx_fuzz.cc`, `fuzz/xxx.c`, `xxx_fuzz.cpp`, `target_fuzz.java`.
- `build_system` must not be `unknown`.
- `evidence` must be a non-empty string array.

Minimal valid template:
```json
{
  "build_system": "cmake",
  "chosen_target_api": "archive_read_open1",
  "chosen_target_reason": "runtime-reachable parser entrypoint",
  "fuzzer_entry_strategy": "sanitizer_fuzzer",
  "evidence": [
    "CMakeLists.txt defines a library target",
    "selected target API appears in repository source"
  ]
}
```

### `fuzz/build.py`
- Must include:
  - `DEFAULT_CMAKE_ARGS = ["-DENABLE_TEST=OFF", "-DENABLE_INSTALL=OFF"]`
  - invoke CMake with the system `cmake` binary (for example `["cmake", "-S", ...]` and `["cmake", "--build", ...]`); do not use `sys.executable, "-m", "cmake"` or `python -m cmake`
  - Python-native parallel args: `["-j", str(os.cpu_count() or 1)]`
  - never use `$(nproc)` or shell substitutions
  - runtime artifact discovery (do not hardcode a single static library path)
  - generated headers include path: when a CMake/configure build emits headers into a build directory, add that CMake build directory to every primary and replay compile command (for example `-I{BUILD_DIR}` for generated headers such as `pnglibconf.h`)
  - owning source linkage: when a harness calls APIs implemented in contrib/example/demo source files outside the main static library, compile the owning source file alongside the harness or switch to a public library API
  - never call a static example helper directly from a harness; static example helpers are not linkable API and example sources with their own `main()` must not be linked into libFuzzer harnesses unless rewritten/guarded
  - primary runnable fuzzers must link the libFuzzer runtime with `-fsanitize=fuzzer,address,undefined` (or equivalent sanitizer set including `fuzzer`); do not use `-fsanitize=fuzzer-no-link` for final binaries unless you also provide and link a valid fuzzer `main`
  - for every primary fuzzer `fuzz/out/<name>`, also build a coverage replay sibling at `fuzz/out/replay/<name>` with `-fprofile-instr-generate -fcoverage-mapping`
  - coverage replay must link coverage-instrumented repository/library objects, not only an instrumented harness object; for CMake/configure projects, use a separate replay/coverage build directory or rebuild the static libraries with `CFLAGS`/`CXXFLAGS` containing `-fprofile-instr-generate -fcoverage-mapping`
  - coverage/replay CMake builds that use LLVM coverage flags must configure with `clang`/`clang++` (for example `-DCMAKE_C_COMPILER=clang` and `-DCMAKE_CXX_COMPILER=clang++`, or `CC=clang CXX=clang++`); do not pass LLVM coverage flags to `/usr/bin/cc`/GCC
  - do not link replay binaries against non-instrumented static libraries when function/path coverage is expected
  - generated `subprocess` compile commands must start with the compiler executable (`clang` or `clang++`); append sanitizer/coverage flags after the compiler, never as `cmd[0]`
  - replay binaries must be real coverage-instrumented executables, not symlinks or copies of the primary fuzzer
  - replay executables must also link a runnable entrypoint: prefer `-fsanitize=fuzzer,address,undefined -fprofile-instr-generate -fcoverage-mapping`; if you use `-fsanitize=fuzzer-no-link`, you must compile and link a separate replay `main()` wrapper that calls `LLVMFuzzerTestOneInput`

Exact static-lib discovery block:
```python
def find_static_lib(repo_root):
    import subprocess
    result = subprocess.run(
        ["find", str(repo_root), "-name", "*.a", "-type", "f"],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        return None
    for p in result.stdout.strip().split("\n"):
        p = Path(p)
        if "test" not in p.name.lower() and p.exists():
            return p
    return None
```

Compiler-by-suffix rule:
- use `clang` for `.c` sources
- use `clang++` for `.cc`, `.cpp`, `.cxx` sources
- never compile all harness sources with `clang++` by default

### API surface rule
- Prefer public/stable APIs in harness code.
- Avoid internal/private namespaces (`detail`, `_internal`, `impl`, `private`) by default.
- If no public alternative exists, add `api_surface_exception` in `fuzz/repo_understanding.json` with non-empty `reason` and `evidence` (optional `approved_symbols`).

## Constraints
- Do not modify repository source files outside `fuzz/` and `./done`.
- If upstream source appears syntactically broken, do not edit it; adapt the external harness/build glue, avoid that example/demo source, or record the limitation in `fuzz/repo_understanding.json`.
- Multi-target buildability is required when execution plan has multiple targets.
- Do not leave stale or missing execution target mappings in `fuzz/harness_index.json`.
- If `attack_hint.key_code_path` is present, prefer harness call flow that reaches those functions instead of generic wrapper paths.
- If `attack_hint.boundary_values` is present, reflect them in corpus/bootstrap seed strategy and parser field choices.
- LibFuzzer harness contract is mandatory:
  - do not define custom `main()` in harness source;
  - use `LLVMFuzzerTestOneInput` (or language-equivalent fuzz entrypoint) as the only fuzz entry.
  - C/C++ harnesses that use `uint8_t` or `size_t` must include the standard headers that define them (`<stdint.h>` and `<stddef.h>` or C++ equivalents) in the harness source.
- LibFuzzer link contract is mandatory: every runnable executable under `fuzz/out/`, including `fuzz/out/replay/<name>`, must link a runnable entrypoint; prefer `-fsanitize=fuzzer,address,undefined` for both primary and replay executables. `-fsanitize=fuzzer-no-link` alone is only valid for objects/libraries, or for replay executables that compile and link a separate `main()` wrapper that calls `LLVMFuzzerTestOneInput`.
- Forbid argv/file-driven harness entry logic in libFuzzer mode (`fopen(argv[1], ...)`, `read(argv[1], ...)`, manual corpus file loops).
- When diagnostics include concrete file paths, use `Read and fix <path>[:line]` before broader edits.
- If MCP is unavailable, continue in degraded mode and record this in `fuzz/repo_understanding.json`.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Acceptance checklist
- Harness file count is >= 1 before marking completion.
- Required scaffold files exist.
- `fuzz/harness_index.json` maps each execution target to an existing harness source file.
- README field `Harness file:` points to a real harness file.
- `fuzz/repo_understanding.json` is semantically valid and complete.
- Build script follows compiler-by-suffix and static-lib-discovery contracts.
- `fuzz/out/replay/<name>` exists for each built native fuzzer and can emit `LLVM_PROFILE_FILE=...profraw` during single-input replay.

## Done contract
- Write `fuzz/out/` into `./done`.
