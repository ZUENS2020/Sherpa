---
name: vuln-hunt-loop
description: Monitor vuln-hunt pipeline on dev, find bugs, fix them, deploy, and re-test in a continuous cycle. Also use for one-off dev monitoring.
arguments: []
---

# Vuln-Hunt Dev Loop

Continuous cycle: monitor → find bugs → fix → PR → deploy → run jobs → monitor.

## Server Access

```bash
ssh -i ~/.ssh/id_ed25519 -o ConnectTimeout=15 deploy@frp-jar.com -p 63893
```

K8s access on server:
```bash
sudo kubectl --kubeconfig /etc/kubernetes/admin.conf <cmd> -n sherpa-dev
```

API port inside web pod: **8001**.

## Quick Status (one-liner)

```bash
ssh -i ~/.ssh/id_ed25519 deploy@frp-jar.com -p 63893 \
  "kubectl exec -n sherpa-dev \$(kubectl get pods -n sherpa-dev -l app=sherpa-web -o jsonpath='{.items[0].metadata.name}') -- curl -s http://localhost:8001/api/tasks" \
  | python3 -c "
import json, sys
for t in json.load(sys.stdin)['items']:
    c = t.get('children_status', {})
    print(f\"{t['job_id'][:12]} | {t['status']} | {t.get('stage','?')} | repo:{t.get('repo','?')} | child:{c.get('running',0)}R/{c.get('success',0)}S/{c.get('error',0)}E | cov:{t.get('fuzz_max_cov','?')}% ft:{t.get('fuzz_max_ft','?')}\")
"
```

## Fuzz Pod Monitoring

```bash
# List fuzz pods
kubectl get pods -n sherpa-dev | grep fuzz

# Follow active fuzz output (stdout = LibFuzzer stats)
kubectl logs -n sherpa-dev <pod> | grep -E '(cov:|INITED|pulse|exec/s|DONE|NEW)'

# Follow workflow stage logs
kubectl logs -n sherpa-dev <pod> | grep -E '(\[wf\]|\[k8s|step=|Entering)'

# Follow agent activity
kubectl logs -n sherpa-dev <pod> | grep -E '(Read|Write|Glob|running…|idle timeout)'
```

## 7 Critical Checkpoints

### CK1: TypedDict State Fields
After vuln-hunt stage completes, query the API:
```bash
kubectl exec -n sherpa-dev deploy/sherpa-web -- curl -s http://localhost:8001/api/task/<job_id> \
  | python3 -c "import json,sys; t=json.load(sys.stdin); print('iter:', t.get('vuln_hunt_iteration'), 'pri:', t.get('vuln_hunt_highest_priority'))"
```
- `vuln_hunt_iteration` > 0 (1, 2, 3...)
- `vuln_hunt_highest_priority` > 0 (~0.67)
- **Fixed in**: PR #405

### CK2: Vuln-Hunt Agent Completion
- **Symptom**: Agent reads files, outputs analysis, then idles — only `running… elapsed=Xs` for 10+ min
- **Root cause**: opencode CLI buffers large Write tool calls (e.g., 79KB vuln_candidates.json) internally. Model takes 10+ min to generate, zero stdout = codex_helper idle detector fires at 600s.
- **Fix**: Stage-specific idle timeout overrides. vuln_hunt → 1800s, plan → 1200s.
  - `SHERPA_OPENCODE_IDLE_TIMEOUT_VULN_HUNT_SEC` (default 1800)
  - `SHERPA_OPENCODE_IDLE_TIMEOUT_PLAN_SEC` (default 1200)
- **Check**: `kubectl logs <pod> | grep "idle timeout"`
- **Fixed in**: PR #402 (Write persistence) + `workflow_graph.py` idle_timeout_override

### CK3: Routing Decision
After vuln-hunt, routing depends on `_vuln_hunt_entry_source` (set before stage runs):

| Source | Priority ≥ 0.65 | Priority < 0.65 |
|---|---|---|
| `analysis` | → plan (always) | → plan (always) |
| `coverage-analysis` | → plan (replan with vuln targets) | → improve-harness |
| `improve-harness` | → build (always) | → build (always) |

Threshold: `_VULN_REPLAN_PRIORITY_THRESHOLD = 0.65` (env `SHERPA_VULN_REPLAN_PRIORITY_THRESHOLD`)

### CK4: Synthesize Harness Match
- **Symptom**: `execution_plan_harness_mismatch` — missing harness source
- **Root cause**: Synthesize reads outdated targets.json instead of selected_targets.json
- **Fixed in**: PR #406, #407, #408, #410

### CK5: Build Not Stalling
- Build pod should complete < 5 min
- **Fixed in**: PR #402

### CK6: Run Actually Fuzzing
```bash
kubectl logs <pod> | grep -E '(cov:|INITED|pulse|exec/s)'
```
- Should see LibFuzzer output: `cov: X ft: Y exec/s: Z`
- If coverage stuck at initial value (e.g., `cov: 35 ft: 35`), fuzzer may be hitting a trivial code path
- RSS growing rapidly (e.g., 100MB → 12GB) may indicate target memory leak
- **Fixed in**: PR #402

### CK7: Vuln-Hunt Second Trigger
After coverage-analysis completes → vuln-hunt should trigger again
- Check API for `vuln_hunt_iteration >= 2`
- **Fixed in**: PR #403

## Submitting Jobs

```bash
ssh -i ~/.ssh/id_ed25519 deploy@frp-jar.com -p 63893 \
  "POD=\$(kubectl get pods -n sherpa-dev -l app=sherpa-web -o jsonpath='{.items[0].metadata.name}') && \
   kubectl exec -n sherpa-dev \$POD -- curl -s -X POST http://localhost:8001/api/task \
   -H 'Content-Type: application/json' \
   -d '{\"jobs\":[{\"code_url\":\"https://github.com/<owner>/<repo>\",\"timeout\":10,\"time_budget\":1800,\"total_time_budget\":3600,\"docker\":true,\"docker_image\":\"auto\"}],\"auto_init\":true,\"build_images\":true}'"
```

## Fix → PR → Deploy Cycle

1. Fix code locally
2. Syntax check: `python3 -c "import py_compile; py_compile.compile('<file>', doraise=True)"`
3. Commit: `git add <files> && git commit -m "fix: short description"`
4. Push to dev: `git pull --rebase origin dev && git push origin dev`
5. Deploy auto-triggers on push to dev. Monitor: `gh run list --repo ZUENS2020/Sherpa --workflow=deploy-dev.yml --limit=1`
6. Wait for deploy (~5-7 min), then submit new jobs to verify

## Idle Timeout Reference

| Stage | Env Var | Default |
|---|---|---|
| All (fallback) | `SHERPA_OPENCODE_IDLE_TIMEOUT_SEC` | 600s |
| vuln_hunt | `SHERPA_OPENCODE_IDLE_TIMEOUT_VULN_HUNT_SEC` | 1800s |
| plan | `SHERPA_OPENCODE_IDLE_TIMEOUT_PLAN_SEC` | 1200s |
| synthesize | `SHERPA_OPENCODE_IDLE_TIMEOUT_SYNTH_SEC` | 300s |
| analysis | `SHERPA_ANALYSIS_OPENCODE_IDLE_TIMEOUT_SEC` | 75s |

## PR Reference

| PR | Fix | Layer |
|---|---|---|
| #402 | Write persistence + idle timeout | Infrastructure |
| #403 | Vuln-hunt every loop + pure vuln scoring | Routing + Scoring |
| #404 | Score formula in plan SKILL | Prompts |
| #405 | TypedDict state persistence | Infrastructure |
| #406 | Synthesize reads selected_targets | Integration |
| #407 | Prioritize must_run targets in synthesize | Integration |
| #408 | Rewrite targets.json with must_run | Integration |
| #409 | Vuln→plan priority bridge | Scoring |
| #410 | Ensure selected_targets.json exists | Integration |
| #411 | Fix NameError target_api→api | Bugfix |
| #413 | Revert synthetic target injection | Cleanup |

## Documentation Upkeep

After each monitoring session: if you discovered a new issue, behavior, or insight, check whether CLAUDE.md (Session State, Known Issues), project-journal.md, or habits.md need updating. Follow the self-update rules in CLAUDE.md.
