# CLAUDE.md — TianHeng (Sherpa) Automated Fuzzing Orchestration

## Project Overview

AI-driven automated fuzzing harness generation and vulnerability discovery system. Takes a target C/C++ codebase → analyzes → generates fuzzing harnesses → builds with coverage instrumentation → runs LibFuzzer → analyzes coverage feedback → triages crashes. All orchestrated as a recoverable, observable multi-stage LangGraph workflow.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python FastAPI (port 8001) + LangGraph state machine |
| Frontend | Next.js 14 App Router + React + MUI v7 + TanStack Query |
| Code Analysis | PromeFuzz MCP Server (AST, callgraph, LLM comprehension) |
| AI Agent | OpenCode CLI (opencode-ai) via `codex_helper.py` |
| LLM | deepseek-v4-pro via litellm proxy (`192.168.1.79:4000`) |
| DB | PostgreSQL 16 (psycopg2, no ORM) |
| CI/CD | GitHub Actions (`deploy-dev.yml`, `deploy-prod.yml`) |

## Key Files

```
harness_generator/src/langchain_agent/
├── main.py                      # FastAPI app, all API routes, job lifecycle
├── workflow_graph.py            # 16K+ lines, LangGraph StateGraph, 14 nodes
├── codex_helper.py              # OpenCode CLI wrapper, retry logic, idle detection
├── job_store.py                 # InMemoryJobStore + PostgresJobStore
├── k8s_job_worker.py            # K8s Job executor for workflow stages
├── workflow_common.py           # Shared workflow utilities
├── workflow_coverage_decision.py
├── workflow_target_scoring.py
├── workflow_target_selection.py
├── workflow_observability.py
├── opencode_skills/             # AI agent contracts per stage (SKILL.md each)
│   ├── analysis/ plan/ synthesize/ synthesize_complete_scaffold/
│   ├── seed_generation/ fix_build/
│   ├── fix_harness_after_run/ improve_harness_in_place/
│   ├── crash_analysis/ crash_triage/
│   ├── fix_crash_harness_error/ fix_crash_upstream_bug/
│   └── vuln_hunt/
harness_generator/src/
├── harness_generator.py         # Core harness generation engine
├── fuzz_unharnessed_repo.py     # Fuzzer execution (build, run, crash extraction)
├── codex_helper.py              # OpenCode CLI wrapper
├── seed_families.py             # Fuzz seed generation
frontend-next/                   # Next.js 14 monitoring dashboard
promefuzz-mcp/                   # MCP code analysis server
docker/                          # Dockerfiles (web, frontend, gateway, opencode, fuzz)
k8s/                             # Kustomize overlays (base, dev, prod, cloudflare)
```

## Workflow Stages (LangGraph)

14 nodes in `workflow_graph.py:build_fuzz_workflow()` (line 16271). Init can resume from any stage.

### Main Happy Path
```
init → analysis → vuln-hunt → plan → synthesize → build → run
```

### After Run — Coverage Loop (no crash)
```
run → per-input-replay → coverage-analysis
                            ├─ should_improve + vuln → vuln-hunt → plan/synthesize/...
                            └─ should_improve        → improve-harness
                                                         ├─ in_place + vuln → vuln-hunt
                                                         ├─ in_place        → build
                                                         └─ replan          → plan
```

### After Run — Crash Path
```
run → crash-triage
        ├─ harness_bug  → plan
        ├─ upstream_bug → re-build → re-run → crash-analysis → plan
        └─ (other)      → plan
```

### Routing Rules (key thresholds)

| From | Condition | To |
|---|---|---|
| analysis | `vuln_hunting_enabled` | vuln-hunt |
| analysis | otherwise | plan |
| vuln-hunt | source=analysis | plan |
| vuln-hunt | source=coverage-analysis, priority ≥ 0.65 | plan |
| vuln-hunt | source=coverage-analysis, priority < 0.65 | improve-harness |
| vuln-hunt | source=improve-harness | build |
| build | ok | run |
| build | error or restart_to_plan | plan |
| run | crash_found | crash-triage |
| run | otherwise | per-input-replay |
| coverage-analysis | should_improve + vuln enabled | vuln-hunt |
| coverage-analysis | should_improve | improve-harness |
| coverage-analysis | loop_count ≥ max (env `SHERPA_MAX_CONTINUOUS_LOOP`, default 3) + vuln | vuln-hunt |
| coverage-analysis | loop_count ≥ max | plan |
| improve-harness | in_place + vuln | vuln-hunt |
| improve-harness | in_place | build |
| improve-harness | replan or loop_count ≥ max | plan |

## API Routes (port 8001)

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/task` | Submit fuzzing task |
| GET | `/api/task/{job_id}` | Get task detail |
| POST | `/api/task/{job_id}/resume` | Resume stopped task |
| POST | `/api/task/{job_id}/stop` | Stop running task |
| GET | `/api/tasks` | List all tasks |
| GET | `/api/system` | System health + metrics |
| GET | `/api/config` | Get runtime config |
| GET | `/healthz` | Liveness probe |

## Deployment

- **Dev**: Push to `dev` branch triggers `deploy-dev.yml` → K8s `sherpa-dev` namespace
- **Prod**: Manual workflow dispatch → K8s `sherpa-prod` namespace
- **Server**: `ssh -i ~/.ssh/id_ed25519 deploy@frp-jar.com -p 63893`
- **K8s**: `sudo kubectl --kubeconfig /etc/kubernetes/admin.conf <cmd> -n sherpa-dev`
- Branch strategy: PRs target `dev`, never `main`. Deploy via CI.

## Session State

<!-- SESSION-START -->
last_session: 2026-05-16
last_deploy: 6b3626b31 — self-evolving docs (journal, habits, auto sections)
active_jobs: cJSON (d0b0cb5c), uriparser (3c3c2c8e), libwebp (3504b418) — all in analysis
<!-- SESSION-END -->

## Key Env Vars

| Var | Default | Purpose |
|---|---|---|
| `SHERPA_OPENCODE_IDLE_TIMEOUT_SEC` | 600 | Fallback idle timeout for all stages |
| `SHERPA_OPENCODE_IDLE_TIMEOUT_VULN_HUNT_SEC` | 1800 | Idle timeout for vuln_hunt stage |
| `SHERPA_OPENCODE_IDLE_TIMEOUT_PLAN_SEC` | 1200 | Idle timeout for plan stage |
| `SHERPA_OPENCODE_IDLE_TIMEOUT_SYNTH_SEC` | 300 | Idle timeout for synthesize stage |
| `SHERPA_ANALYSIS_OPENCODE_IDLE_TIMEOUT_SEC` | 75 | Idle timeout for analysis stage |
| `SHERPA_VULN_HUNTING_ENABLED` | 1 | Enable vuln-hunt sub-phase |
| `SHERPA_VERIFY_STAGE_NO_AI` | 0 | Skip AI seed generation in run stage |

## Common Issues & Debugging

### Known Issues

<!-- ISSUES-START -->
| ID | Issue | Status | Mitigation |
|---|---|---|---|
| CK2 | vuln_hunt large JSON → 600s idle timeout | Mitigated | 1800s override (env var) |
<!-- ISSUES-END -->

### CK2: Agent Idle Timeout (600s too short for large file generation)
- **Symptom**: Agent reads files, outputs "Now I have a complete picture...", then 10 min of `running… elapsed=Xs` → `idle timeout after 600s without activity; terminating agent`
- **Root cause**: opencode CLI buffers Write tool calls internally; model takes 10+ min to generate 79KB vuln_candidates.json; zero stdout during generation means codex_helper idle detector fires
- **Fix**: Stage-specific idle timeout overrides (see env vars above)
- **Check**: `kubectl logs <pod> | grep "idle timeout"`

### CK4: Synthesize Harness Mismatch
- **Symptom**: `execution_plan_harness_mismatch` — synthesize reads old targets.json, not selected_targets.json
- **Fix**: Ensure synthesize context includes selected_targets.json + execution_plan.json

### CK6: Run Not Actually Fuzzing
- **Check**: `kubectl logs <pod> | grep -E 'cov:|INITED|pulse|exec/s|NEW'`
- Healthy output: `#8192 pulse cov: 35 ft: 35 corp: 6/549b exec/s: 744`

### General Debugging
```bash
# View fuzz pods
kubectl get pods -n sherpa-dev | grep fuzz
# Follow pod logs
kubectl logs -n sherpa-dev <pod> --tail=50
# Query API tasks
kubectl exec -n sherpa-dev deploy/sherpa-web -- curl -s http://localhost:8001/api/tasks
# Check API task detail
kubectl exec -n sherpa-dev deploy/sherpa-web -- curl -s http://localhost:8001/api/task/<job_id>
```

## Documentation Maintenance

This file and related docs (`.claude/project-journal.md`, `.claude/habits.md`, `.claude/skills/`) are self-evolving. Update them silently when these events occur:

1. **Bug fix deployed to dev** → Update `last_deploy` in Session State + add entry to project-journal.md
2. **New issue discovered** → Add/update row in Known Issues table (remove if fixed)
3. **New job submitted** → Update `active_jobs` in Session State (remove completed ones)
4. **Monitoring insight** → Append discovery to project-journal.md
5. **New user habit observed** → Update `.claude/habits.md` (only clear patterns, not one-offs)

Rules:
- Be silent — don't announce doc updates
- One line per event in journal
- Remove stale entries
- Don't update if nothing changed
