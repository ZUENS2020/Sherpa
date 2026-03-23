# Documentation Index

This directory documents Sherpa as it behaves today. It is organized for technical learning first, not for historical reconstruction.

## Read These First

1. [../README.md](../README.md)
   System overview, current workflow, and artifact model.

2. [CODEBASE_TECHNICAL_ANALYSIS.md](CODEBASE_TECHNICAL_ANALYSIS.md)
   Module boundaries, control plane vs execution plane, and state machine semantics.

3. [TECHNICAL_DEEP_DIVE.md](TECHNICAL_DEEP_DIVE.md)
   How to study the codebase efficiently and what each major capability is doing.

4. [API_REFERENCE.md](API_REFERENCE.md)
   Current backend API contract for frontend integration and task control.

5. [STANDARD_CHANGE_PROCESS.md](STANDARD_CHANGE_PROCESS.md)
   Branching, validation, doc-sync, and release expectations.

## Deployment and Operations

- [k8s/DEPLOY.md](k8s/DEPLOY.md)
- [k8s/DEPLOYMENT_DETAILED.md](k8s/DEPLOYMENT_DETAILED.md)
- [k8s/RUNBOOK.md](k8s/RUNBOOK.md)
- [k8s/MAPPING.md](k8s/MAPPING.md)
- [k8s/ORIGINAL_K8S_CLUSTER_DEPLOYMENT.md](k8s/ORIGINAL_K8S_CLUSTER_DEPLOYMENT.md)

## Historical / Legacy Context

These files are preserved as context, not as the primary operating manual:

- [PROJECT_HANDOFF_STATUS.md](PROJECT_HANDOFF_STATUS.md)
- [K8S_MIGRATION_CHECKLIST.md](K8S_MIGRATION_CHECKLIST.md)
- [DOCKER_TO_K8S_HANDOFF.md](DOCKER_TO_K8S_HANDOFF.md)
- [k8s/DEPLOY_ISSUES_NON_NETWORK.md](k8s/DEPLOY_ISSUES_NON_NETWORK.md)
- [k8s/E2E_ZLIB_REPORT.md](k8s/E2E_ZLIB_REPORT.md)

## Current Documentation Rules

- Workflow descriptions should follow the current mainline stages:
  - `plan`
  - `synthesize`
  - `build`
  - `run`
  - `crash-triage`
  - `fix-harness`
  - `coverage-analysis`
  - `improve-harness`
  - `re-build`
  - `re-run`
- API docs must match the actual FastAPI implementation in `harness_generator/src/langchain_agent/main.py`.
- Links in docs should be relative repository paths.
- Historical design notes must be clearly labeled as historical rather than operational truth.
