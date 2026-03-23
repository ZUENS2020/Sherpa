# Kubernetes Deploy Guide

This is the current short-form deployment guide for Sherpa on Kubernetes.

## 1. What Gets Deployed

Long-lived services:

- backend API service
- frontend UI service
- Postgres

Short-lived workloads:

- one Kubernetes Job per workflow stage

## 2. Expected Runtime Shape

- executor mode should be `k8s_job`
- output root should be shared and visible to stage jobs and the backend
- backend and worker runtime images must be version-aligned
- non-root execution is the default assumption

## 3. Basic Deploy Flow

1. build or reference the target backend/frontend images
2. apply the appropriate overlay or manifests
3. wait for long-lived services to become ready
4. verify configuration and worker image references
5. submit a real repository task as smoke test

## 4. Smoke Test Repositories

Recommended:

- `https://github.com/fmtlib/fmt.git`
- `https://github.com/yaml/libyaml.git`
- `https://github.com/madler/zlib.git`
- `https://github.com/libarchive/libarchive.git`

## 5. Success Criteria

- task submission succeeds
- stage jobs are created and complete or fail with persisted reports
- `/api/tasks` and `/api/system` reflect live state
- task artifacts appear under `/shared/output`

## 6. Read Next

- [DEPLOYMENT_DETAILED.md](DEPLOYMENT_DETAILED.md)
- [RUNBOOK.md](RUNBOOK.md)
- [MAPPING.md](MAPPING.md)
