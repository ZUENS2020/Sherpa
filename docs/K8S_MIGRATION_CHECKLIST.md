# K8s Migration Checklist

This file is retained as historical migration context. It is not the current deployment manual.

Primary current docs:

- [k8s/DEPLOY.md](k8s/DEPLOY.md)
- [k8s/DEPLOYMENT_DETAILED.md](k8s/DEPLOYMENT_DETAILED.md)
- [k8s/RUNBOOK.md](k8s/RUNBOOK.md)

## Historical Migration Outcome

The migration objective has already been achieved at a high level:

- staged Kubernetes job execution exists
- output and logs are persisted outside pod lifetime
- frontend/backend services are separated from stage jobs
- coverage improvement and crash repro are part of the workflow model

## What To Validate After Changes

When making infrastructure or workflow changes, validate:

- stage jobs still produce `stage-*.json` and `stage-*.error.txt`
- task workspaces still land under `/shared/output`
- `/api/tasks` and `/api/system` still reflect live task state correctly
- at least one real repository task can complete the mainline path

## What This File No Longer Tries To Do

- it does not describe the current workflow stage graph in detail
- it does not document deployment commands
- it does not define release criteria
