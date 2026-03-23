# Local Kubernetes Quickstart

This document is for a developer who wants to boot a minimal Sherpa stack on a
local Kubernetes cluster for smoke testing. It is not a production deployment
guide.

## Goal

Bring up the control plane, submit one task, and confirm that stage pods can
run end to end.

## Prerequisites

- A working local Kubernetes cluster such as OrbStack, kind, or minikube
- `kubectl`
- Images or image build access for:
  - `sherpa-web`
  - `frontend-next`
- A writable output location for stage artifacts

## Minimal Bring-Up

1. Build or import the required images into the local cluster runtime.
2. Apply the development overlay:

```bash
kubectl apply -k k8s/overlays/dev
```

3. Verify the core services:

```bash
kubectl get pods
kubectl get svc
kubectl get ingress
```

You should at least see healthy instances for:

- `sherpa-web`
- `frontend-next`
- `postgres`

4. Open the web UI and submit one smoke-test job.

## Recommended Smoke-Test Repositories

- `https://github.com/yaml/libyaml.git`
- `https://github.com/fmtlib/fmt.git`

These repositories are small enough for quick feedback but still exercise the
main planning, synthesis, build, run, and artifact paths.

## What To Check

After submitting a task, verify:

- A task record appears in `/api/tasks`
- Stage jobs are created in order
- Stage logs are visible
- The repository working directory is created under the shared output root
- `run_summary.json` or stage result files are written for the job

## Local Troubleshooting Order

1. Inspect cluster objects:

```bash
kubectl get pods
kubectl get jobs
kubectl get events --sort-by=.metadata.creationTimestamp
```

2. Inspect the web service:

```bash
kubectl logs deploy/sherpa-web
```

3. Inspect the active stage pod or job:

```bash
kubectl logs <pod-name>
```

4. Inspect the generated artifact directory:

```bash
ls -la /shared/output/<repo>-<id>
cat /shared/output/<repo>-<id>/run_summary.json
```

## Related Documents

- [Deploy](DEPLOY.md)
- [Deployment Detailed](DEPLOYMENT_DETAILED.md)
- [Runbook](RUNBOOK.md)
- [API Reference](../API_REFERENCE.md)
