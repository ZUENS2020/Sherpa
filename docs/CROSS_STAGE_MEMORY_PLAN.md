# Cross-Stage / Cross-Job Procedural Memory — Design Proposal

Status: **Draft for review**
Author: (proposed)
Scope: `harness_generator/src/langchain_agent` orchestration + opencode SKILL injection
Related: `VULN_HUNT_FULL_REFACTOR_PLAN.md`, `CONTROL_PLANE_CONTRACT.md`

---

## 1. Problem

The fuzzing pipeline repeats the same mistakes — within a single job and, worse,
on every new repository — because the agents that drive each stage
(`analysis → vuln-hunt → plan → synthesize → build → run → coverage-analysis`)
have **no durable memory of lessons learned**.

Two observed, concrete instances:

1. **vcpkg over-declaration churn.** On a `tomlc99` dev run, the `synthesize`
   stage declared vcpkg ports in `fuzz/system_packages.txt` for a zero-dependency
   Makefile library. The native executor has no vcpkg toolchain, so the deps
   bootstrap hard-failed the stage → fallback to `plan` → replan → repeat. The
   job looped **~29 iterations over ~16h** re-committing the *same* mistake. (The
   immediate hard-fail was addressed in the vcpkg best-effort PR, but the agent
   still has no memory that *declaring those ports was wrong*.)

2. **Non-public / binding target selection.** Vuln-hunt repeatedly selected
   internal/`*_wasm` symbols that cannot link, wasting full
   plan→synthesize→build→repair cycles. We currently suppress this with hard
   gating rules, not learning.

Reflexion's own trigger heuristic — *"the same action yielding the same response
for >3 cycles"* — describes these loops exactly. The fix is a **procedural
memory**: distill each failure into a reusable lesson and feed it back into the
relevant stage so the agent stops repeating it.

## 2. What already exists (and why it is insufficient)

| Mechanism | File | Role | Limitation |
|---|---|---|---|
| Cross-stage **state** carrier | `workflow_context_store.py` | propagates scalar fields (cov/ft, companion status, counts) across stages | structural state, not "lessons" |
| **Constraint memory** | `fuzz/constraint_memory.json` via `workflow_graph._record_constraint_memory_observation` / `_constraint_memory_snapshot_from_state` (`workflow_graph.py:1231–1396`) | keys crash/repair/timeout **signatures** → triage classification (`harness_bug`, `upstream_bug`, …); records only after `repeats ≥ SHERPA_CONSTRAINT_MEMORY_REPEAT_THRESHOLD` (default 2); injected back into repair prompts as `fix_hint` | (a) stored under per-job `fuzz/` → **wiped on every new repo**; (b) covers **only crash triage**, not build/synthesize/plan failures |
| Decision trace | `fuzz/decision_trace.jsonl` via `workflow_observability.record_decision_trace` | observability log, capped | **not re-injected** into prompts — pure audit |

**Key insight:** the `constraint_memory → fix_hint` path already proves that
"distill failure → store → inject into prompt" works inside Sherpa. It is a
mini-Reflexion loop. This proposal **generalizes that proven path**; it does not
introduce a foreign paradigm.

## 3. Goals / Non-Goals

**Goals**
- G1. Persist lessons **across jobs** (survive repo/job boundaries), not just within `fuzz/`.
- G2. Cover **all stages** (build/synthesize/plan/vuln-hunt), not only crash triage.
- G3. **Inject** the most relevant lessons into the consuming stage's prompt/context (closed loop).
- G4. **Memory hygiene**: confidence, occurrence count, decay/expiry, scope — so bad lessons don't poison future runs.
- G5. Minimal new infrastructure; reuse LangGraph + the existing Postgres job store.
- G6. Gated + reversible (env kill-switch), defaulting to additive/no-regression.

**Non-Goals**
- Not adding an external memory service (Mem0/Zep/Letta) — see §8.
- Not giving opencode subprocess agents live `store.put/search` tools (architecture mismatch — see §4).
- Not semantic recall of arbitrary facts; the target is **procedural** memory (rules that reduce repeated errors).

## 4. Architectural constraint that shapes the design

Each stage runs as a **separate k8s job** executing an **opencode CLI agent**
driven by `SKILL.md` + on-disk context files. There is **no long-running
LangGraph process** in which an LLM makes `store.put()/search()` tool calls.

Therefore the memory loop is **orchestrator-mediated**, mirroring the existing
`constraint_memory` flow:

```
   (stage job fails / succeeds)
            │
            ▼
  orchestrator (workflow_graph)  ──reflect──▶  distill lesson
            │                                      │
            │                                      ▼
            │                            MEMORY STORE (cross-job, Postgres)
            │                                      │
            ▼                                      │ retrieve top-K by (stage, library_class)
   next stage job  ◀──inject lessons into context/ SKILL hint──┘
```

The agent never calls the store; the **orchestrator reads/writes it and injects
lessons into the context files the agent already consumes** (the same channel as
`fix_hint`, `attack_hint.key_code_path`, `coverage_hints`).

## 5. Design

### 5.1 Data model — a "lesson" (procedural memory entry)

```jsonc
{
  "lesson_id": "build/vcpkg-overdeclare/makefile-selfcontained",   // stable dedup key
  "stage": "synthesize",                 // producing/consuming stage
  "error_class": "vcpkg_overdeclare",    // coarse, enumerated category
  "scope": "library_class:makefile-selfcontained",  // or "global", "library:tomlc99"
  "signature": "synthesize failed; vcpkg unavailable; system_packages.txt non-empty",
  "lesson": "Do NOT write fuzz/system_packages.txt for self-contained make/single-file libs; declare vcpkg ports only on a concrete missing-external-lib build error.",
  "evidence": ["job=…", "log: vcpkg unavailable while required ports are declared"],
  "confidence": 0.8,
  "occurrence_count": 4,
  "first_seen": 0, "last_seen": 0,
  "decay_after_days": 90,
  "source_jobs": ["6b8e34c7…"],
  "schema_version": 1
}
```

- `error_class` is an **enumerated** set (e.g. `vcpkg_overdeclare`,
  `non_public_api_selection`, `missing_owning_source`, `coverage_trivial_context`,
  `harness_bad_free`, …) so retrieval is precise and prompts stay small.
- `scope` controls blast radius: prefer `library_class` (e.g. cmake / configure /
  makefile-selfcontained / header-only) over `global` to avoid over-generalizing.

### 5.2 Storage backend — Postgres (reuse the job store)

The web service already runs Postgres (`postgres-0`, `_job_store_database_url()`).
Add one table, namespaced by environment:

```sql
CREATE TABLE IF NOT EXISTS procedural_memory (
  lesson_id        TEXT PRIMARY KEY,
  stage            TEXT NOT NULL,
  error_class      TEXT NOT NULL,
  scope            TEXT NOT NULL,
  signature        TEXT,
  lesson           TEXT NOT NULL,
  evidence         JSONB,
  confidence       REAL  DEFAULT 0.5,
  occurrence_count INT   DEFAULT 1,
  first_seen       TIMESTAMPTZ DEFAULT now(),
  last_seen        TIMESTAMPTZ DEFAULT now(),
  decay_after_days INT   DEFAULT 90,
  source_jobs      JSONB
);
CREATE INDEX IF NOT EXISTS idx_pm_lookup ON procedural_memory (stage, error_class, scope);
```

Rationale vs alternatives: this maps cleanly onto **LangGraph's `BaseStore`
namespace/key/value contract** (namespace = `(stage, error_class)`, key =
`scope`), so we can later swap in a Postgres-backed `BaseStore` with no schema
change. A JSON file on the existing `/shared/<env>/memory/` hostPath PV is an
acceptable fallback if a DB dependency in the worker path is undesirable; the
access layer hides the choice.

### 5.3 Write path — reflection (Reflexion "distill")

Extend the existing observation hook. After a stage **fails** (or succeeds after
prior failures), the orchestrator:

1. computes `(stage, error_class, scope, signature)` from existing signals
   (`repair_error_code`, `error_kind`, build/synthesize diagnostics, library
   build-system detection already done in `repo_understanding`/`build_strategy`);
2. only promotes to a lesson when `occurrence_count ≥ threshold` (reuse the
   existing repeat-threshold idea, default 2) — this is the cheap hygiene gate;
3. optionally uses **one LLM call** to phrase the `lesson` text from the failure
   trajectory (the Reflexion self-reflection step). For well-known
   `error_class`es we can ship **templated lessons** (no LLM call) — e.g. the
   vcpkg one is deterministic.

`error_class` detection for the two known cases is rule-based and cheap:
`vcpkg_overdeclare` = (`system_packages.txt` non-empty) ∧ (build/synthesize failed
with `vcpkg unavailable|missing vcpkg toolchain`); `non_public_api_selection` =
selected target flagged internal/binding by the public-API oracle.

### 5.4 Read path — retrieval + injection (Reflexion "self-hint")

Before launching a stage job, the orchestrator:

1. retrieves lessons matching `(stage, library_class)` ordered by
   `confidence * recency`;
2. takes **top-K (default 3, à la Reflexion's sliding window)** to bound prompt size;
3. writes them into the stage's context (e.g. a `fuzz/lessons.json` +
   a short rendered block appended to the SKILL hint), reusing the existing
   context-injection channel that already carries `fix_hint` / `attack_hint`.

SKILL files gain a short "Known pitfalls for this library class" section that
consumes `fuzz/lessons.json` when present.

### 5.5 Memory hygiene (Reflexion's documented failure mode)

- **Confidence + occurrence**: lessons strengthen with repetition, decay with age (`decay_after_days`).
- **Contradiction handling**: a lesson that precedes a later *success* in the same `(stage, scope)` gets confidence-decayed (ADD/UPDATE/DELETE-style maintenance, cf. Mem0).
- **Scope discipline**: default to `library_class` scope; promotion to `global` requires N distinct libraries.
- **Caps**: top-K injection + max prompt budget so memory can never bloat context.
- **Kill-switch**: `SHERPA_PROCEDURAL_MEMORY=0` disables read+write; `_READONLY=1` injects but never writes (safe canary).

## 6. Phasing

- **Phase 0 (this doc).** Agree data model, backend, error_class enum, injection points.
- **Phase 1 — storage + access layer.** `procedural_memory.py`: Postgres-backed store with a JSON-file fallback; unit-tested in isolation. No behavior change yet.
- **Phase 2 — write path.** Hook reflection into the existing observation point; ship templated lessons for `vcpkg_overdeclare` + `non_public_api_selection`. Run in `_READONLY=0` but **injection still off** — only accumulate + observe via metrics.
- **Phase 3 — read/injection.** Inject top-K into synthesize/plan context + SKILL sections. Gated on `SHERPA_PROCEDURAL_MEMORY=1`.
- **Phase 4 — generalize + LLM reflection.** Add LLM-phrased lessons for open-ended classes; widen `error_class` coverage; add contradiction-decay maintenance.

## 7. Config & metrics

Config (`SHERPA_*` convention):
- `SHERPA_PROCEDURAL_MEMORY` (default `0` until Phase 3 validated) — master switch.
- `SHERPA_PROCEDURAL_MEMORY_READONLY` (default `0`) — accumulate without injecting.
- `SHERPA_PROCEDURAL_MEMORY_TOPK` (default `3`).
- `SHERPA_PROCEDURAL_MEMORY_MIN_OCCURRENCE` (default `2`).
- `SHERPA_PROCEDURAL_MEMORY_DECAY_DAYS` (default `90`).

Metrics (surface via the existing task API + monitor):
- `procedural_lessons_total`, `lessons_injected_per_stage`,
  `repeat_error_rate` (same `error_class` recurring after a lesson exists),
  `iterations_to_first_successful_build` (expected to drop),
  `vcpkg_overdeclare_recurrences` (expected → 0 after Phase 3).

**Success criterion:** on a re-run of `tomlc99` (and a second self-contained lib),
`vcpkg_overdeclare` does not recur once a lesson exists; mean replan iterations
per job decrease vs the pre-memory baseline.

## 8. Alternatives considered (mature projects)

| Option | Verdict |
|---|---|
| **Mem0** (48k★, AWS Agent SDK) | Strong generic fact memory (ADD/UPDATE/DELETE), but optimized for semantic personalization; would add an external service + redundant store. Borrow its UPDATE/DELETE maintenance *idea*, not the dependency. |
| **Letta / MemGPT** | OS-style tiered memory for long-horizon single agents; mismatched with our multi-job, SKILL-driven execution. |
| **Zep (Graphiti)** | Temporal knowledge graph — great for "stale fact" errors, overkill for procedural lessons; new service. |
| **LangMem** | The one with first-class **procedural memory (self-rewriting instructions)** — directly on-target. But it requires a live LangGraph `StateGraph` agent loop, which our subprocess-agent execution does not have; p95 search ~60s. We adopt its **pattern** (procedural memory over `BaseStore`) without the SDK. |
| **Reflexion** (arXiv 2303.11366) | The methodological basis (failure → verbal lesson → store → self-hint, top-3 window, repeat-cycle trigger). Adopted conceptually. |

**Decision:** implement the **Reflexion procedural-memory pattern on LangGraph
`BaseStore` semantics, backed by the existing Postgres**, extending the proven
`constraint_memory → fix_hint` loop. No external memory service. This is the
"mature pattern, minimal new infra" path for a system that is already
LangGraph + Postgres and runs batch (latency-insensitive) workloads.

## 9. Risks

- **Bad lessons poison future runs** → mitigated by confidence/decay, scope
  discipline, top-K caps, and a read-only canary phase.
- **Prompt bloat** → bounded by top-K + byte budget.
- **Over-generalization across unrelated libraries** → `library_class` scoping by default.
- **Postgres coupling in the worker path** → access layer supports a `/shared/<env>/memory/` JSON fallback.

## 10. References

- Reflexion: Language Agents with Verbal Reinforcement Learning — arXiv 2303.11366
- LangGraph long-term memory (`BaseStore`, cross-thread/cross-job store)
- LangMem procedural memory (self-updating instructions)
- Mem0 ADD/UPDATE/DELETE memory maintenance
- Existing in-repo precedent: `workflow_graph.py:1231–1396` (`constraint_memory`), `workflow_observability.py` (`decision_trace`), `workflow_context_store.py`
