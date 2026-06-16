# Sherpa — Improvement Log & Known Issues

A living backlog of improvement points and problems found while developing/operating
the Sherpa fuzzing pipeline. Append new entries at the top of the relevant section.

**Format per entry:** `### [ID] short title` → *Symptom / Why it matters / Where / Status*.
Statuses: `OPEN` · `IN PROGRESS` · `SHIPPED (#PR)` · `WONTFIX`.

---

## 1. Shipped fixes

### [S-473] Harness-validity gate, robust origin by crashing function (Phase B)
- Phase B: before fuzzing, replay KNOWN-VALID seeds through a harness; if it
  crashes from its OWN code, skip + write a repair note instead of self-crashing.
  Wired at the single chokepoint `_run_fuzzer` (PR #471 fixed an earlier
  mis-wiring into the CLI-only Pass-E loop, which the k8s run node bypassed).
- **Origin classification by crashing FUNCTION, not just file** (PR #472): single-
  header libs compiled into the harness TU attribute library frames to `*_fuzz.c`;
  classifying by file alone could mis-skip a real library bug (e.g. tinyobj
  `parseLine`). Now harness-origin only when the crashing function is one the
  harness source defines (`dump`, `fuzz_file_reader`, `LLVMFuzzerTestOneInput`);
  library functions stay library-origin and are never skipped. **Resolves O-6.**
  `SHERPA_HARNESS_VALIDITY_GATE` (default on). Validated live on dev: tinyobjloader
  defective harnesses flagged (`harness_validity_*.md`), json.h/utf8.h real bugs
  still found (`upstream_bug`). *Known limit:* a harness bug that only triggers on
  deep fuzzer-found input (jsmn `dump`) isn't caught by valid-seed replay — the
  LLM triage safety net still classifies it `harness_bug`.

### [S-465] Contract-aware crash triage — out-of-contract crashes ≠ vulnerabilities
- Crashes only reachable by violating a *documented* precondition (NUL-termination,
  non-NULL, required init, length bounds) are harness/API-misuse, not library bugs.
- Added `contract_analysis.py` (heuristic precondition extraction from doc-comments),
  inject contract into crash-triage context, SKILL rule → classify as `harness_bug`.
  `SHERPA_CONTRACT_TRIAGE` (default on).
- **SHIPPED (#465).** Validated: json.h OOB correctly stayed `upstream_bug` (in-contract).

### [S-464] Coverage injection concatenated adjacent string literals
- `_inject_coverage_instrumentation` Case 2: a single-element flags list with no
  trailing comma (`["-fsanitize=fuzzer,address,undefined"]`) got the coverage flag
  inserted as an adjacent literal → Python implicit concat → malformed
  `-fsanitize=...undefined-fsanitize-coverage` → clang rejects → **build death-loop**
  (tinyexpr stuck at build-13 for ~3.3h).
- Fix: capture element + separator separately; insert `, "<cov>"` when no comma.
- **SHIPPED (#464).**

### [S-463] Sanitizer severity classification — benign UB must not block fuzzing
- `-fsanitize=undefined` flags pervasive-but-benign UB (e.g. `pointer-overflow` /
  `(char*)0+n` pointer-as-accumulator) on nearly every input → masquerades as a
  crash and aborts the fuzzer (corpus stuck at 0; observed on json-parser json.c:437).
- Fix: UBSan non-fatal in fuzz runs (`halt_on_error=0`, `SHERPA_UBSAN_HALT` to revert);
  `classify_sanitizer_severity()` → memory_safety / benign_ub / other; benign-UB-only
  logs are advisory, not crashes (`SHERPA_BENIGN_UB_NONFATAL`). ASan stays the hard gate.
- **SHIPPED (#463).** Validated: json.h heap-overflow still correctly `memory_safety`.

### [S-462] Instrument direct `clang -c` library compiles (Pass D)
- build.py that compiles the library object directly with `clang -c` (no make) escaped
  coverage. **SHIPPED (#462).**

### [S-460] Instrument the target library deterministically in build.py
- The fuzzed binary's target library was compiled WITHOUT coverage → libFuzzer blind →
  cov flatlined at the ~7 harness edges (verified: valid TOML moved cov 6→7).
  `_inject_coverage_instrumentation` was dead code; wired it up + added a library-CFLAGS
  pass (+ `make CC=<wrapper>` override) at both build sites. **SHIPPED (#460).**
  Validated: tomlc99 cov 7 → 1214.

---

## 2. Open improvement backlog (harness generation)

The recurring theme: **harnesses for non-trivial API shapes are generated incorrectly**,
producing false-positive crashes or ineffective (cov=0) fuzzers. Triage now classifies
these correctly (see S-463/S-465), but fixing them at the *source* (synthesize) would
stop wasting harness slots and fuzz budget.

### [H-1] Callback/reader-based APIs leak the callback-allocated buffer  — OPEN (task started)
- tinyobjloader-c: `tinyobj_parse_obj` takes a `file_reader` callback that `malloc`s the
  file buffer; the generated `fuzz_file_reader` never frees it → LeakSanitizer aborts on
  the first input → **corpus stuck at 0, library never fuzzed**.
- Fix: harness must free callback-allocated buffers per the API's ownership contract, or
  set `ASAN_OPTIONS=detect_leaks=0` for such targets. Where: synthesize SKILL + harness gen.

### [H-2] settings/allocator-struct APIs — alloc/free mismatch  — OPEN
- json-parser: harness zero-inits `json_settings` (so `mem_free == NULL`) then frees with
  `json_value_free_ex(&settings, …)` → SEGV. Should use the simple `json_value_free`, or
  set the allocator callbacks to defaults.
- Fix: synthesize must pair allocator/settings correctly (consume `api_contract` from S-465).

### [H-3] format-string APIs produce cov=0 (ineffective) harnesses  — OPEN
- cesanta/frozen: `json_scanf`/`json_vscanf` need a format string; generated harnesses
  fuzzed them without a meaningful format → **all fuzzers cov=0/exec/s=0** across 13 rounds.
- Fix: detect scanf-style format APIs; either synthesize a fixed sensible format and fuzz
  the data, or skip such entrypoints in favour of buffer-in ones.

### [H-4] Recursive token/tree-walker harness helpers lack bounds checks  — OPEN
- jsmn: generated `dump()` walks the token tree using `t->size` without validating
  `t+1+j < count` → harness self-crash (heap-overflow READ in `dump_fuzz.c`, not jsmn.h).
  Wastes a harness slot on a self-crash; jsmn itself bounds-checks correctly.
- Fix: harness-gen guidance for bounds-checked tree walkers, or avoid emitting walker
  helpers that can over-read.

### [H-5] Low coverage saturation on some libs  — OPEN
- iniparser: maxcov plateaued at 19 across ~17 rounds — harness only reaches a small slice
  of the parser. improve-harness couldn't push past it.
- Fix: investigate whether the entrypoint/seed selection under-drives the parser.

### [H-6] Contract Part 2 — make harnesses *respect* documented preconditions  — OPEN
- S-465 only gates triage. The stronger fix: synthesize should satisfy documented
  preconditions before calling (NUL-terminate, clamp lengths, init structs) so the fuzzer
  stays in-contract and crashes are real. Depends on `api_contract` extraction (S-465 Part 1).

### [H-7] Parser-combinator / tagged-union-result APIs — harness invalid-free  — OPEN
- orangeduck/mpc: every generated harness SEGVs on the first input, so the library never
  fuzzes (`cov=0 ft=0 corpus=0 exec/s=0`, run rc=76). mpc requires building a grammar first
  and returns a tagged union (`.output` = `mpc_ast_t*` on success, `.error` needs
  `mpc_err_delete` on failure); the harness calls `mpc_ast_delete()` unconditionally /
  on the wrong union member → invalid free → SEGV in `mpc_ast_delete` (mpc.c:2947).
- Same family as H-1/H-2/H-3: synthesize can't set up unusual APIs (grammar-building +
  result unions + type-specific destructors). Fix: detect tagged-union/result APIs and
  branch the cleanup on the success/error tag; skip targets whose setup can't be inferred
  rather than emitting a harness that self-crashes on every input.

---

## 3. Open improvement backlog (orchestration / robustness)

### [O-1] Build death-loop circuit breaker  — OPEN
- tinyexpr retried an identical failing build ~13 times over 3.3h (root cause was S-464).
  The loop should detect repeated *identical* build failures and bail / escalate early
  instead of burning hours and a node slot.

### [O-2] Auto-suppress + remember benign UB / known harness crashes  — OPEN
- Follow-on to S-463: when triage classifies a finding benign_ub or harness_bug, generate
  a suppression and store it in procedural memory so the same library starts suppressed and
  fuzzes deeper next time (turns "blocked" into "explores real surface").

### [O-3] vuln-hunt / LLM stage latency  — OPEN/observe
- deepseek-v4-pro stages take 25–50 min each; jobs commonly run 3+ hours through the
  coverage-improvement loop. Watch for stalls vs. genuine long work (heartbeat `updated_at`
  is the liveness signal). Consider per-stage budgets / faster model for cheap stages.

### [O-4] Seed/flag coverage gaps  — OPEN
- We did not exercise `json_parse_flags_allow_simplified_json` for json.h, so we missed the
  `json_parse_object` OOB (json.h:1695) that another researcher found. Consider fuzzing
  across the library's documented flag/mode combinations, not just one default mode.

### [O-5] Triage over-confident on unreproducible timeouts/hangs  — OPEN
- dr_libs: an idle-timeout/hang artifact that **re-runs clean in 8ms** (does not reproduce,
  filed under `unreproducible/`) was labeled `upstream_bug` confidence 0.85 — the evidence
  itself said "ASAN re-run completed in 8ms without error — timeout/hang, not a memory
  safety violation". An unreproducible timeout is most often a busy-node/idle-kill artifact,
  not a confirmed bug. Calibration fix: timeouts/hangs that do not reproduce (and any
  finding under `unreproducible/`) should be capped at `inconclusive` / low confidence and
  must NOT be presented as `upstream_bug` without a reproducing input. Code-review
  hypotheses (e.g. "unbounded `for(;;)` loop") are useful as evidence but are not, on their
  own, a confirmed finding.

### [O-6] Harness-validity origin mis-attributed for single-header libs  — RESOLVED (S-473)
- The Phase B gate classified harness- vs library-origin by the crashing frame's
  source *file*. Single-header libraries (tinyobjloader-c, json.h, utf8.h) compile
  library code into the harness TU, so library frames are attributed to `*_fuzz.c`
  — a file-only check could mis-skip a real library bug (e.g. tinyobj `parseLine`).
  Fixed by classifying on the crashing **function** (is it harness-defined?) with
  the file pattern only as fallback. See **S-473**.

---

## 4. Validation / findings log

### tinyobjloader-c — stack-buffer-overflow WRITE in `parseLine()` (tinyobj_loader_c.h)  — PR submitted
- Real CWE-787 stack **write** overflow (more serious than the read OOBs): an OBJ
  face (`f`) line with > `TINYOBJ_MAX_FACES_PER_F_LINE` (16) vertices overflows the
  fixed stack array `f[16]`; the guard `assert` is compiled out under `-DNDEBUG`.
  With `TINYOBJ_FLAG_TRIANGULATE`, `command->f[3*n+..]` also overflows for >7-vertex
  faces. Reachable via public `tinyobj_parse_obj(..., TINYOBJ_FLAG_TRIANGULATE)` on a
  valid OBJ (minimal PoC: one `f` line with 20 verts). Source-verified + independently
  reproduced on a clean clone (commit 0f8ea84).
- Not a duplicate (#55 = sscanf overflow; #60 = triangulation indexing correctness,
  doesn't address the unbounded write). Issues disabled upstream → fixed + filed as
  [syoyo/tinyobjloader-c#73](https://github.com/syoyo/tinyobjloader-c/pull/73)
  (bounds-check stop-gap; real fix = size buffers for the triangulated face).
  Materials at `~/Downloads/tinyobjloader-c-stack-overflow-2026-06-16/`.
- Surfaced + validated the S-473 origin-by-function fix (O-6): `parseLine` is
  attributed to `*_fuzz.c` (single-header lib) but is library code → correctly NOT
  skipped by the validity gate.

### dr_libs (dr_flac) — unreproducible timeout/hang — NOT confirmed
- Triage labeled `upstream_bug` (0.85): hypothesised unbounded `for(;;)` loops in dr_flac
  Ogg/frame decoding + no `totalPCMFrameCount` sanity check → CPU-exhaustion DoS. But the
  artifact **does not reproduce** (re-runs clean in 8ms; filed under `unreproducible/`) — it
  was an idle-timeout kill, not a memory-safety crash. Treated as **not a confirmed bug**;
  not disclosed. Surfaced calibration gap [O-5]. dr_flac is widely used / OSS-Fuzz'd, so a
  trivial hang would likely already be known.

### mpc — generated harness invalid-free → no fuzz data (harness bug)
- Every generated mpc harness SEGVs on the first input (`mpc_ast_delete` invalid free,
  mpc.c:2947), so the library never fuzzes (cov=0/corpus=0). Triage correctly returned
  `harness_bug` (false_positive). Root cause is harness-gen, not mpc — see [H-7].

### utf8.h — heap-buffer-overflow READ in `utf8nlen()` / `utf8len()` (utf8.h:550)  — PR submitted
- Real CWE-125 OOB read on a NUL-terminated string whose final byte is a multibyte lead
  byte with no continuation (PoC: 2 bytes `0x2c 0xdf`). The codepoint width is taken from
  the lead byte and `str` advanced unconditionally → steps past the NUL → next `'\0' != *str`
  reads OOB. Reachable via the **default** `utf8len()` — no flag needed (broader than the
  json.h one).
- **Not a duplicate of the exact function:** same root-cause class as open issue
  `sheredom/utf8.h#117` (multibyte lookahead past end, reported there for
  `utf8makevalid`/`utf8codepoint`), but #117 does not name `utf8nlen`/`utf8len`.
- **Fixed + PR'd:** [sheredom/utf8.h#136](https://github.com/sheredom/utf8.h/pull/136) —
  bounded one-byte-at-a-time advance + regression test; full suite passes under ASan (157).
- Value: pipeline found it; S-465 contract-aware triage correctly kept it `upstream_bug`
  (harness satisfies the documented `nul_terminated` precondition → in-contract → real bug).
  Materials archived at `~/Downloads/utf8h-oob-read-2026-06-14/`.

### json.h — heap-buffer-overflow READ in `json_parse_number()` (json.h:1925)
- Real CWE-125 OOB read (1 byte) on input `"0"` with `allow_json5`; `src[offset+1]` read
  without `offset+1 < size` (the sizing pass at :1187 has the guard, the data pass doesn't).
- **DUPLICATE** — already reported upstream by another researcher in `sheredom/json.h#113`
  (comment) and tracked via #116/#117. **Not disclosed/CVE'd by us.**
- Value: independent confirmation that the pipeline + S-463/S-465 classified a real,
  in-contract memory-safety bug correctly (`upstream_bug`). Materials archived locally at
  `~/Downloads/jsonh-oob-read-2026-06-14/`.

### json.h — heap-buffer-overflow READ in `json_parse_object()` (json.h:~1698)  — DUPLICATE
- Real CWE-125 OOB read: with `allow_global_object` (also implied by
  `allow_simplified_json`), `json_parse_object()` reads `src[state->offset]` without
  checking `offset < size` when `is_global_object` is true (whitespace-only / EOF
  input → offset == size). Same two-pass-asymmetry class as the json_parse_number
  one (the sizing counterpart `json_get_object_size()` guards via
  `json_skip_all_skippables()`; the data pass doesn't).
- **DUPLICATE** — this is matteoalba's "Bug 1" already reported in the
  `sheredom/json.h#113` comment thread (json_parse_object, same global-object path).
  **Not disclosed by us.**
- Value: re-confirmed the Phase B gate did NOT suppress it (no validity note,
  `upstream_bug`, cov 941) — library-origin crash on valid input preserved.

### json-parser — `pointer-overflow` UBSan at json.c:437
- Benign UB (pointer-as-accumulator idiom); **not a vulnerability** → informed S-463. Not reported.

### jsmn — heap-overflow in harness `dump()`
- Harness bug (see H-4), not a jsmn bug; triage correctly returned `harness_bug` (conf 1.0).

---

## 5. Process notes / gotchas

- **Branch hygiene:** PRs target `dev` only, never `main`; deploy via `deploy-dev.yml`
  (RESET_DEV_BEFORE_DEPLOY wipes in-progress jobs — don't redeploy mid-batch you care about).
- **Worktree drift:** a feature worktree was 157 commits behind `origin/dev`; all the
  #443–#465 work lives on `dev`. Rebase onto `origin/dev` before starting new fixes.
- **Repo URL accuracy:** wrong org → clone fails with `rc=128` on all mirrors (tinyobjloader
  was `tinyobjloader/…` 404; correct is `syoyo/tinyobjloader-c`). Verify the URL (curl 200) first.
- **Good fuzz targets for validation:** simple buffer-in APIs (`parse(buf, len)`) fuzz
  cleanly (jsmn/parson/json.h/pdjson). Avoid callback/format-string/settings APIs for clean
  validation runs — they hit H-1/H-2/H-3.
- **Monitoring:** `~/Documents/sherpa-tui/` — single-screen TUI dashboard of all tasks
  (panels per task: phase/cov/exec-s/round/vuln/crash/elapsed). `node index.js`.
- **Liveness vs stuck:** `/api/tasks` `updated_at` ticking = alive; a stage quiet for >10min
  with no artifact writes is suspect (check the run pod's codex `elapsed=` and job log).
