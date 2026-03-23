# Release Gate

This document defines the minimum checks required before promoting a change from
development validation to production release.

## Required Checks

### Backend

- `python -m py_compile` for the modules touched by the change
- Relevant `pytest` subsets for the changed workflow, API, or quality logic

### Frontend

- `npm run build`
- Any test or lint step required by the changed frontend package

### Deployment Readiness

- Dev deployment uses the expected image tags
- Worker image pinning and config are aligned
- API fields consumed by the frontend are still present and documented
- Docs are updated when workflow or API behavior changes

## Recommended Smoke Tests

Before promoting `dev` to `main`, validate at least:

- One parser-focused repository, such as `libyaml` or `fmt`
- One build-sensitive repository, such as `zlib` or `libarchive`

The goal is to cover both the planning/synthesis path and the build/run/crash
artifact path.

## Reject Release If

- Stage jobs are looping without meaningful state change
- Coverage improvement keeps replanning without producing a strategy change
- Seed generation regresses into source-file ingestion or excessive noise
- The worker falls back to unsupported execution assumptions
- The deployed image version cannot be proven from the running cluster state

## Evidence To Keep

For each release candidate, retain:

- PR link
- Validation commands and result summary
- One successful dev task ID
- Any rollback note if the change touches workflow routing or deployment logic
