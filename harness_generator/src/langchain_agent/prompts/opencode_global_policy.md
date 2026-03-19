# Sherpa OpenCode Global Policy

This policy applies to every OpenCode stage unless explicitly overridden by stage instructions.

## 1) Build And Link Policy
- Default to minimal linking: start with `additional_libs = []`.
- Add external link libraries only when there is explicit build evidence (for example `undefined reference`, `cannot find -l...`, or clear CMake/link diagnostics).
- When adding libraries, do minimal incremental changes and keep them evidence-driven.

## 2) Artifact Discovery Policy
- Do not hardcode a single build artifact path.
- Discover real artifacts with read-only discovery commands and verify chosen paths exist before linking.
- Prefer robust discovery helpers (for example `find_static_lib(...)`) over one-off guessed paths.

## 3) Command Policy
- Allowed: read-only inspection commands (for example `find`, `grep`, `rg`, `cat`, `ls`, `head`, `tail`, `sed` in read mode).
- Forbidden: build/execute commands (for example `cmake`, `make`, `ninja`, compiler invocations, running scripts/binaries/tests/fuzzers).

## 4) targets.json Policy
- `targets.json` must be plain JSON array (not markdown, not wrapped object).
- `name` should be the source file stem when applicable.
- `api` should be the source filename or concrete callable API.
- Forbidden: `name = "LLVMFuzzerTestOneInput"`.
