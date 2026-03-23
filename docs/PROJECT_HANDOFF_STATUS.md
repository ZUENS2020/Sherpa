# Project Handoff Status

This file is historical context, not the primary operating manual.

For current documentation, start here:

- [../README.md](../README.md)
- [README.md](README.md)
- [CODEBASE_TECHNICAL_ANALYSIS.md](CODEBASE_TECHNICAL_ANALYSIS.md)
- [TECHNICAL_DEEP_DIVE.md](TECHNICAL_DEEP_DIVE.md)

## Current High-Level Status

Sherpa currently has:

- a staged Kubernetes execution model
- a workflow centered on planning, synthesis, build, run, coverage improvement, crash triage, harness repair, and repro
- frontend/backend dynamic task and system views
- seed-quality-aware coverage improvement

## What This File Should Be Used For

Use this file only to communicate high-level handoff notes, such as:

- where the primary docs now live
- what broad subsystems exist
- what still requires deeper engineering work

## Current Ongoing Engineering Risks

- target depth selection can still drift toward shallow wrappers
- scaffold consistency can still be broken by weak synthesis rounds
- seed quality remains a major determinant of real coverage progress
- crash classification quality depends on repro and log quality

## Not the Source of Truth

Do not use this file as the source of truth for:

- stage routing
- API field contracts
- deployment steps
- validation criteria

Those now live in the rewritten main docs.
