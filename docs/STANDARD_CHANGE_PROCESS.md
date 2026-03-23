# Standard Change Process

This document describes the current recommended change, validation, and release flow for Sherpa.

## 1. Branching Model

- feature work: `codex/*` or explicit topic branches
- integration branch: `dev`
- production branch: `main`

Expected path:

1. develop on a feature branch
2. open a PR into `dev`
3. validate through `dev`
4. open `dev -> main`
5. release through `main`

## 2. Required Project Process

For major project changes:

1. define goal and planned scope first
2. write that summary into a Linear issue
3. apply type-based labels
4. set the issue to `In Progress`
5. only then start implementation
6. update the same issue with progress comments as work evolves
7. finish with a `Done` status and a final Chinese summary comment

## 3. Validation Expectations

Minimum local validation depends on change type.

### Code changes

- `python3 -m py_compile` for touched Python modules
- relevant pytest subset

### Frontend changes

- build and/or tests for the touched frontend
- confirm API field usage still matches backend behavior

### Workflow changes

Prefer at least one real repository validation on `dev`, commonly:

- `fmt`
- `libyaml`
- `zlib`
- `libarchive`

## 4. Documentation Sync Rules

Any change that affects these behaviors must update docs in the same change:

- workflow stages or routing
- target selection or execution-plan behavior
- seed generation or seed-quality policy
- crash triage / repro / harness-fix behavior
- deployment model
- API contracts consumed by frontend

At minimum, review:

- [../README.md](../README.md)
- [API_REFERENCE.md](API_REFERENCE.md)
- [TECHNICAL_DEEP_DIVE.md](TECHNICAL_DEEP_DIVE.md)
- deployment docs under `docs/k8s/`

## 5. Release Guardrails

- do not push directly to `dev` or `main`
- `main` should accept changes from `dev`, not directly from feature branches
- required PRs should include:
  - change summary
  - risk / rollback note
  - reproducible validation result

## 6. Things to Avoid

- updating only README while leaving deeper docs stale
- describing idealized APIs instead of actual API behavior
- treating stage success alone as proof without checking artifacts
- keeping historical migration notes unlabeled so they look current
