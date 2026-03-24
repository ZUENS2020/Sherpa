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

## 5) Archive Seed Policy
- For `archive-container`, use real repository samples first (for example `contrib/oss-fuzz/corpus.zip`, `contrib/oss-fuzz/**`, `test/**`, `tests/**`).
- Avoid hand-crafted magic-only archive seeds.
- Keep malformed/truncated archive seeds as a minority of the corpus (<=30%).

## 6) Dictionary Policy
- The system auto-generates libFuzzer dictionaries (`fuzz/dict/*.dict`) from seed_profile tokens, existing .dict files, and harness string literals.
- When improving coverage, consider adding domain-specific tokens to the dictionary: format keywords, protocol delimiters, magic bytes, reserved words from the target API.
- Dictionary entries use libFuzzer format: `token_name="value"` (one per line).

## 7) Source Coverage Feedback Policy
- When `fuzz/coverage_report.txt` is available, it contains function-level coverage data from llvm-cov.
- Prioritize exercising uncovered functions listed in the coordinator hint's "Top uncovered functions" section.
- Consider adding new API call sequences, input patterns, or seed variants that target uncovered code paths.

## 8) Adaptive Input Length Policy
- The system automatically adapts `-max_len` based on seed_profile (e.g. archive-container=65536, parser-structure=4096, generic=1024).
- Harness code should handle variable-length inputs gracefully; do not assume a fixed input size.
