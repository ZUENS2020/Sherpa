# Docker-to-K8s Handoff Notes

This file is historical context explaining the move away from older execution assumptions. It is not the current runtime manual.

## Current Runtime Reality

Sherpa now assumes:

- Kubernetes as the primary staged execution environment
- native worker execution inside stage pods
- shared output rooted at `/shared/output`
- long-lived control-plane services plus short-lived stage jobs

## Why This Historical Note Still Exists

Some older discussions and artifacts may still refer to:

- inner Docker assumptions
- migration checkpoints
- pre-Kubernetes execution patterns

This file remains only to explain that those assumptions are no longer the reference model.

## Use These Instead

- [README.md](README.md)
- [k8s/DEPLOY.md](k8s/DEPLOY.md)
- [k8s/DEPLOYMENT_DETAILED.md](k8s/DEPLOYMENT_DETAILED.md)
