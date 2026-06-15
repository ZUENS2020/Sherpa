# Harness Synthesis Hardening Plan

**Status:** proposal (not implemented; not deployed)
**Motivation:** repeated harness-generation failures on non-trivial API shapes
(IMPROVEMENTS backlog H-1, H-2, H-3, H-7). The pipeline writes *correct* harnesses
for simple buffer-in APIs (jsmn / parson / json.h / pdjson all fuzzed cleanly and
found real bugs), but produces self-crashing or ineffective harnesses for APIs
whose correct use is a multi-step, partly-implicit protocol.

---

## 1. Problem

The failures are not "the model can't read the header." They cluster on a class
where the correct usage is a **cross-call protocol** that is not stated locally in
the function's doc-comment:

| Library | Shape | Failure |
|---|---|---|
| orangeduck/mpc | grammar-build + **tagged-union result** + type-specific destructor | harness `mpc_ast_delete()`s the wrong union member → SEGV on every input → cov=0 (H-7) |
| tinyobjloader-c | **reader callback** allocates a buffer the caller must free | callback buffer leaked → LSan aborts → corpus=0 (H-1) |
| json-parser | **settings/allocator struct** must be paired | zero-init settings → NULL `mem_free` → SEGV (H-2) |
| cesanta/frozen | **format-string** API (`json_scanf`) | fuzzed without a meaningful format → cov=0 (H-3) |

Root causes (see IMPROVEMENTS §"why AI can't just read it"):
1. The contract is **scattered/implicit** (README, examples, union discriminants,
   ownership semantics) — multi-hop synthesis, where LLMs are weakest.
2. The model defaults to the **statistically-common idiom** ("parse → delete AST")
   which is wrong for the uncommon API.
3. There is **no write-time grounding**: the harness isn't reliably run against
   valid input and *fixed* before being accepted; a self-crash is classified
   `harness_bug` and the run moves on instead of repairing the harness.

## 2. Approach — give synthesis the loop a human uses

Three complementary mechanisms. None requires the model to "read better"; they add
grounding, examples, and a validity gate.

### A. Learn correct usage from the library's own examples/tests
The canonical correct call sequence (build grammar, check the result tag, pair the
allocator, free with the right destructor) is almost always present in the repo's
`examples/`, `test/`, `demo/`, or README code blocks.

- New `usage_extractor` (companion/analysis stage): for each selected entrypoint,
  grep the repo for call sites in example/test/readme code and extract the
  surrounding setup + teardown lines into a `api_usage.json`:
  `{ func: [{ example_path, setup_snippet, call_snippet, teardown_snippet }] }`.
- Feed `api_usage.json` into the synthesize context so the harness mirrors a real,
  working call sequence rather than an invented one.
- Cheap, deterministic, high-signal — this alone would have fixed mpc (its README
  shows `mpc_parse` + `r.output`/`r.error` branching + `mpc_ast_delete`/`mpc_err_delete`).

### B. Harness-validity self-check BEFORE acceptance (the key gate)
A correct harness must not crash/leak on trivially-valid input. Make that an
acceptance gate, not a post-hoc triage outcome.

- After synthesize builds a harness, run it on a tiny set of **known-valid seeds**
  (repo sample inputs, or a couple of hand-trivial ones) under ASan/LSan/UBSan.
- If it crashes/leaks/UB on valid input → **this is a harness defect, not a
  finding** → route to `fix_harness` with the sanitizer output, loop (bounded
  retries), do **not** accept and do **not** count as a vuln.
- Only accept the harness once it survives the valid seeds. This directly closes
  the "write → run → observe → fix" loop the model is missing.
- Gate flag: `SHERPA_HARNESS_VALIDITY_GATE` (default on); bounded by existing
  `max_fix_rounds`.

### C. Detect hard API shapes → template or skip (don't emit a self-crasher)
- A `shape_classifier` flags entrypoints whose signature/usage indicates:
  reader/loader **callback**, **tagged-union/result** return, **settings/allocator
  struct** parameter, **grammar-builder** prerequisite, **format-string** parameter.
- For a recognised shape: use a known-correct harness **template** (parameterised
  by the extracted usage from A), or — if setup can't be inferred — **skip the
  entrypoint** in favour of a buffer-in one, rather than emitting a harness that
  self-crashes on every input and burns a fuzz slot.

## 3. Fit with the existing pipeline

- **A** extends the analysis/companion stage and the `api_contract` work
  (S-465 / H-6): contract = preconditions; usage = the correct call sequence.
- **B** reuses the existing `fix_harness_after_run` / `fix_crash_harness_error`
  loop and sanitizer-severity classification (S-463) — it just runs *before*
  acceptance, gated on valid seeds, so harness defects never reach crash-triage as
  findings.
- **C** is a new lightweight classifier consumed by the synthesize SKILL; pairs
  with the harness templates already implied by the SKILL contract.

## 4. Acceptance criteria / metrics

Measure on the known-hard set (mpc, tinyobjloader-c, json-parser `_ex`, frozen):
- harness no longer self-crashes on valid seeds (cov/corpus > 0, exec/s > 0);
- false-positive crash count (harness_bug / false_positive) drops materially;
- no regression on the easy set (jsmn / parson / json.h / pdjson still fuzz);
- real-bug finding preserved (json.h / utf8.h class still caught & classified).

## 5. Risks / mitigations

- **Over-skipping (C)** could drop fuzzable targets → only skip when both shape is
  hard *and* usage can't be extracted (A) *and* the validity gate (B) fails after
  retries.
- **Valid-seed scarcity** → fall back to a couple of trivial type-correct inputs;
  absence of a clean valid seed is itself a signal the harness setup is wrong.
- **Example/test extraction noise** → prefer compiling, self-contained example
  snippets; rank by "does this snippet reference the target function directly".
- **Latency** → A and C are cheap/deterministic; B adds a few short seed runs,
  far cheaper than a wasted multi-hour fuzz on a self-crashing binary.

## 6. Phasing

1. **B (validity gate)** — highest leverage, mostly reuses existing fix loop;
   stops self-crashing harnesses from being accepted at all.
2. **A (usage extraction)** — makes the agent write the right sequence the first
   time; biggest quality jump for the hard class.
3. **C (shape classifier + templates/skip)** — backstop for shapes that A+B still
   can't get right.

---

*Cross-refs:* IMPROVEMENTS.md H-1/H-2/H-3/H-7 (the failure instances), S-463
(sanitizer severity), S-465 + H-6 (contract extraction), O-1 (build death-loop
circuit breaker — same "bounded retry" discipline applies here).
