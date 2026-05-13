---
name: vuln-hunt-loop
description: Monitor vuln-hunt pipeline on dev, find bugs, fix them, deploy, and re-test in a continuous cycle.
arguments: []
---

# Vuln-Hunt Dev Loop

Continuous cycle: monitor → find bugs → fix → PR → deploy → run libpng → monitor.

## Server Access

```
ssh -i ~/.ssh/id_ed25519 -o ConnectTimeout=15 deploy@frp-jar.com -p 63893
```

K8s access on server:
```
sudo kubectl --kubeconfig /etc/kubernetes/admin.conf <cmd> -n sherpa-dev
```

API port inside web pod: **8001** (not 8000).

## Step 1: Check Deploy Status

```bash
gh run view <RUN_ID> --repo ZUENS2020/Sherpa --json status,conclusion
```

If not completed, wait (deploy takes 5-7 min typically).

## Step 2: Start New libpng Job

```bash
ssh ... "sudo kubectl exec -n sherpa-dev deploy/sherpa-web -- python3 -c \"
import urllib.request, json
payload = json.dumps({'jobs': [{'code_url': 'https://github.com/pnggroup/libpng.git', 'time_budget': 900, 'run_time_budget': 900, 'model': 'deepseek-v4-pro', 'timeout': 10}]}).encode()
req = urllib.request.Request('http://localhost:8001/api/task', data=payload, headers={'Content-Type': 'application/json'})
r = urllib.request.urlopen(req)
print('JOB_ID:', json.loads(r.read())['job_id'])
\""
```

## Step 3: Monitor Pipeline — 7 Critical Checkpoints

### Checkpoint 1: TypedDict State Fields
After vuln-hunt stage completes, verify:
```python
d = json.load(open('stage-*-vuln-hunt.json'))
r = d['result']
# Must be > 0:
r.get('vuln_hunt_iteration')  # Should be 1, 2, 3...
r.get('vuln_hunt_highest_priority')  # Should be ~0.67
# Must be non-empty:
r.get('vuln_hunt_active_candidate_id')  # e.g. "integer_overflow_017"
```
- **If broken**: Missing TypedDict fields → PR #405 fix
- **Check**: `grep 'vuln_hunt_iteration\|vuln_hunt_highest_priority' workflow_graph.py` at FuzzWorkflowState TypedDict

### Checkpoint 2: Vuln-Hunt Agent Completion
```bash
kubectl logs <pod> | grep -E 'Write|Edit|done'
```
- **Symptom**: No Write/Edit lines, only elapsed= counters
- **If broken**: Write persistence bug → PR #402 fix
- **Timeout behavior**: Agent idles 300s → restart → may eventually complete

### Checkpoint 3: Routing Decision
After vuln-hunt, next pod should be:
- `plan` if `vuln_hunt_highest_priority >= 0.65` (replan path)
- `improve-harness` if `< 0.65` (continue improving same target)
- `build` if entry source was `improve-harness`
- **If broken**: `_vuln_hunt_entry_source` not propagated → PR #405 fix

### Checkpoint 4: Synthesize Harness Match
Build stage should succeed. If not:
```
RuntimeError: synthesize incomplete: execution_plan_harness_mismatch
  missing harness source: safe_read, readpng2_decode_data
  extra_harnesses: fuzz/png_read_image_fuzz.c
```
- **Root cause**: Synthesize reads `targets.json` (old), not `selected_targets.json` (new vuln-driven)
- **Fix**: Add `selected_targets.json` + `execution_plan.json` to synthesize context → PR #406
- **Check**: `_pass_synthesize_harness` in `fuzz_unharnessed_repo.py`

### Checkpoint 5: Build Not Stalling
Build pod should complete in < 5 min. LibFuzzer compile output expected.
- **If broken**: Same as Checkpoint 2 (Write persistence)

### Checkpoint 6: Run Actually Fuzzing
```bash
kubectl logs <pod> | tail -3
# Should show LibFuzzer output: "cov: XXXX ft: XXXX exec/s: XXX"
```
- **If broken**: Write persistence + idle loop → PR #402

### Checkpoint 7: Vuln-Hunt Second Trigger
After coverage-analysis completes, count vuln-hunt files:
```bash
ls stage-*-vuln-hunt.json | wc -l  # Should be >= 2
```
- **If 1**: Routing not triggering vuln-hunt in coverage loop → PR #403 fix
- Also verify iteration increments: `vuln_hunt_iteration = 2`

## Step 4: Quick Status Command

Copy this script to server as `/tmp/metrics.py`:
```python
import json, glob, os
dirs = glob.glob('/home/deploy/output/dev/_k8s_jobs/<JOB_PREFIX>*')
if dirs:
    base = dirs[0]
    for f in sorted(glob.glob(base + '/stage-*-vuln-hunt.json')):
        d = json.load(open(f)); r = d['result']
        print(f'{os.path.basename(f)}: iter={r.get("vuln_hunt_iteration","?")} pri={r.get("vuln_hunt_highest_priority","?")} active={str(r.get("vuln_hunt_active_candidate_id","?"))[:40]}')
    for f in sorted(glob.glob(base + '/stage-*.json')):
        print(os.path.basename(f))
```

Run: `scp /tmp/metrics.py deploy@frp-jar.com:/tmp/ && ssh ... "python3 /tmp/metrics.py"`

## Step 5: Fix → PR → Deploy Cycle

When a bug is found:
1. Fix the code locally
2. Run syntax check + tests: `python3 -c "import ast; ast.parse(open('...').read())" && pytest tests/test_workflow_target_selection.py -x`
3. Commit and push to `codex/vuln-hunt-every-loop` branch
4. Create PR: `gh pr create --title "..." --body "..." --base dev --head codex/vuln-hunt-every-loop`
5. Merge: `gh pr merge <N> --merge --subject "..."`
6. Deploy: `gh workflow run "Deploy Dev" --repo ZUENS2020/Sherpa --ref dev`
7. Wait for deploy, then go to Step 2

## Key Server Paths
- Job outputs: `/home/deploy/output/dev/_k8s_jobs/<CHILD_JOB_ID>/stage-*.json`
- Repo root: `/home/deploy/output/dev/libpng-<shortid>/`
- Vuln candidates: `fuzz/vuln_candidates.json`
- Selected targets: `fuzz/selected_targets.json`
- Execution plan: `fuzz/execution_plan.json`

## PR Reference
| PR | Fix |
|---|---|
| #402 | Write persistence — done file flush grace period |
| #403 | Vuln-hunt every loop + pure vuln scoring |
| #404 | Stale score formula in plan SKILL |
| #405 | TypedDict fields + routing state mutation |
| #406 | Synthesize reads selected_targets.json |
