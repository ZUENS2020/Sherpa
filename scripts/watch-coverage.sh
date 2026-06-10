#!/usr/bin/env bash
#
# watch-coverage.sh — live coverage / crash monitor for a Sherpa fuzz job.
#
# Polls a job's stage, task-level coverage (fuzz_max_cov/ft), the active
# fuzz pod's live libFuzzer output (cov/ft/corp/exec-s), and crash signals.
#
# Usage:
#   scripts/watch-coverage.sh [-n NAMESPACE] [-j JOB_ID] [-i INTERVAL_SEC] [-1]
#
#   -n  k8s namespace            (default: sherpa-staging)
#   -j  job id                   (default: newest task in the namespace)
#   -i  poll interval seconds    (default: 30)
#   -1  print one snapshot and exit (no loop)
#
# Env:
#   SHERPA_SSH   ssh target+opts  (default: the frp-jar deploy host below)
#   KUBECTL      kubectl invocation on the server
#                (default: sudo kubectl --kubeconfig /etc/kubernetes/admin.conf)
#
# Examples:
#   scripts/watch-coverage.sh                          # newest job in staging
#   scripts/watch-coverage.sh -n sherpa-dev -i 60      # dev, every 60s
#   scripts/watch-coverage.sh -j 82885b03... -1        # one snapshot
#
set -euo pipefail

NAMESPACE="sherpa-staging"
JOB_ID=""
INTERVAL=30
ONCE=0

while getopts "n:j:i:1h" opt; do
  case "$opt" in
    n) NAMESPACE="$OPTARG" ;;
    j) JOB_ID="$OPTARG" ;;
    i) INTERVAL="$OPTARG" ;;
    1) ONCE=1 ;;
    h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "see -h for usage" >&2; exit 2 ;;
  esac
done

SSH="${SHERPA_SSH:-ssh -i $HOME/.ssh/id_ed25519 -o ConnectTimeout=10 deploy@frp-jar.com -p 63893}"
KUBECTL="${KUBECTL:-sudo kubectl --kubeconfig /etc/kubernetes/admin.conf}"

# Run a kubectl command on the server.
kc() { $SSH "$KUBECTL $*" 2>/dev/null; }
# exec curl against the web API inside the namespace.
api() { $SSH "$KUBECTL exec -c web -n $NAMESPACE deploy/sherpa-web -- curl -s http://localhost:8001$1" 2>/dev/null; }

resolve_job() {
  api "/api/tasks" | python3 -c "
import json,sys
try:
    items=json.load(sys.stdin).get('items',[])
    # /api/tasks returns newest-first; prefer the newest running task, else newest.
    items.sort(key=lambda t: float(t.get('created_at') or 0), reverse=True)
    running=[t for t in items if str(t.get('status')).lower()=='running']
    pick=(running or items)[0] if (running or items) else None
    print(pick['job_id'] if pick else '')
except Exception:
    print('')
"
}

snapshot() {
  local job="$1"
  local ts; ts="$(date '+%H:%M:%S')"

  # newest fuzz pod (stage indicator)
  local pod; pod="$(kc get pods -n "$NAMESPACE" --sort-by=.metadata.creationTimestamp -o name | grep fuzz | tail -1)"
  local stage="${pod##*/}"

  # task-level numbers
  api "/api/task/$job" | python3 -c "
import json,sys
try:
    t=json.load(sys.stdin)
except Exception:
    print('  (task api unavailable)'); sys.exit()
print('  status=%s stage=%s' % (t.get('status'), t.get('stage')))
print('  task cov=%s ft=%s  vuln_iter=%s  crash_candidates=%s  crash_repro_ok=%s' % (
    t.get('fuzz_max_cov'), t.get('fuzz_max_ft'),
    t.get('vuln_hunt_iteration'), t.get('crash_vuln_candidate_count'),
    t.get('crash_repro_ok')))
"

  # live libFuzzer line + crash signals from the run pod
  if [[ "$stage" == *run* ]]; then
    local live; live="$(kc logs -n "$NAMESPACE" "$stage" --tail=400 | grep -oE 'cov: [0-9]+ ft: [0-9]+ corp: [0-9]+/[0-9]+b exec/s: [0-9]+' | tail -1)"
    [[ -n "$live" ]] && echo "  LIVE  $live"
    local crashes; crashes="$(kc logs -n "$NAMESPACE" "$stage" --tail=400 | grep -ciE 'ERROR: AddressSanitizer|SUMMARY: |libFuzzer: deadly signal|crash-')"
    [[ "${crashes:-0}" != "0" ]] && echo "  ⚠️  crash signals in run log: $crashes"
  fi

  echo "[$ts] job=${job:0:12} stage=$stage"
}

# Resolve job id if not given.
if [[ -z "$JOB_ID" ]]; then
  JOB_ID="$(resolve_job)"
  [[ -z "$JOB_ID" ]] && { echo "no job found in $NAMESPACE" >&2; exit 1; }
fi

echo "watching ns=$NAMESPACE job=$JOB_ID interval=${INTERVAL}s"
echo "------------------------------------------------------------"

while :; do
  snapshot "$JOB_ID"
  # stop on terminal status
  st="$(api "/api/task/$JOB_ID" | python3 -c "import json,sys
try: print(json.load(sys.stdin).get('status',''))
except Exception: print('')" )"
  case "$st" in
    success|error|failed|completed) echo "== terminal: $st =="; break ;;
  esac
  [[ "$ONCE" == "1" ]] && break
  sleep "$INTERVAL"
done
