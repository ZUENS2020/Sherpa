# Deployment Issues (Historical Notes)

This file is retained only as historical context for past deployment failures.
It is not an operational runbook.

## Historical Themes

Previous deployment incidents were usually caused by one of these classes:

- Runtime image missing required tools
- Worker assumptions drifting from the actual execution model
- Long-running stage loops hiding a workflow logic problem
- Artifact or output paths not matching what the control plane expected

## How To Use This File

If you are debugging a current deployment problem, start with:

- [Runbook](RUNBOOK.md)
- [Deploy](DEPLOY.md)
- [Deployment Detailed](DEPLOYMENT_DETAILED.md)

Only return to this file if you are comparing a new incident with an older one
for root-cause analysis.
