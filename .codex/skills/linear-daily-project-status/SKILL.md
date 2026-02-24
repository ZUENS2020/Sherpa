---
name: linear-daily-project-status
description: "Maintain one daily project-level status issue in Linear in command-driven mode. Use only when the user explicitly triggers `$linear-daily-project-status` (or asks to run this skill exactly). Upsert today's status issue titled \"Sherpa 项目进度同步（YYYY-MM-DD）\": overwrite the existing issue body if today's issue exists, otherwise create a new issue. Always use label \"project status\", read first before write, and avoid comment-based updates."
---

# Linear Daily Project Status

## Overview

Keep project progress logging deterministic in Linear:
1. Write one authoritative status issue per day.
2. Overwrite today's issue body on every major update.
3. Create a new issue only when today's issue does not exist.

## Trigger (Mandatory)

Only execute Linear read/write when trigger is explicit:
1. `$linear-daily-project-status`
2. Equivalent explicit request: "执行 $linear-daily-project-status"

If trigger is not present:
1. Do not call Linear MCP read/write APIs.
2. Do not create/update status issues.
3. At most, prepare a local draft summary if the user asks.

## Change Scope Gate (Applied After Trigger)

After trigger is received, summarize cumulative same-day status with focus on material changes:
1. Workflow graph, node routing, retry/termination policy.
2. Docker/compose/container network/build/runtime path.
3. Memory/session semantics and cross-step context flow.
4. Build/fuzz failure handling and repair loop behavior.
5. API/Frontend behavior changes and runtime default changes.
6. Docs/tests that alter operational decisions.

## Required Targets

Use these defaults unless the user explicitly overrides:
1. Team: `Sherpa_XDU`
2. Project: `Sherpa Stabilization & Delivery`
3. Label: `project status`
4. Issue title format: `Sherpa 项目进度同步（YYYY-MM-DD）`
5. Status: `In Progress`

## MCP Access Policy (Mandatory)

Use Linear MCP for all reads and writes.
1. Read with MCP:
   1. `list_issue_labels` to confirm label.
   2. `list_projects` or `get_project` to confirm target project.
   3. `list_issues` and `get_issue` to find today's issue and inspect body.
2. Write with MCP:
   1. `update_issue` to overwrite today's `description`.
   2. `create_issue` only when today's issue does not exist.
3. Do not use comment logs (`create_comment`) for status snapshots unless the user explicitly asks.
4. Do not ask the user to update Linear manually when MCP is available.

## Pre-write Reconciliation (Mandatory)

Before every write, read today's full issue body first and reconcile.
1. If today's issue exists, always call MCP `get_issue` and load `description`.
2. Merge existing same-day content with new facts from current run.
3. Keep "今日关键进展" as cumulative same-day progress up to now, not just the newest delta.
4. Preserve already-confirmed entries unless they are explicitly corrected by newer evidence.
5. Deduplicate by commit hash, change topic, and test command/result pairs.
6. Write back a fully replaced body after reconciliation.

## Upsert Workflow (No Comments)

Follow this sequence strictly:
1. Collect snapshot data:
   1. Branch, head commit, remote sync state.
   2. `main` vs current branch divergence and unique commits.
   3. Latest validation result (tests/build/run command + pass/fail).
   4. Major changes, risks, and next actions.
2. If today's issue exists, read current `description` first with MCP `get_issue`.
3. Reconcile old body + new facts into one cumulative same-day snapshot.
4. Build full replacement markdown from `references/daily_status_template.md`.
5. Find today's status issue:
   1. Search team issues with label `project status`.
   2. Match exact title `Sherpa 项目进度同步（YYYY-MM-DD）` for local date.
6. Decide:
   1. If today's issue exists: call MCP `update_issue` and overwrite `description`.
   2. If today's issue does not exist: call MCP `create_issue` with full body.
7. Keep exactly one canonical daily issue:
   1. Do not append comments for status content.
   2. If duplicates exist for the same date, update the newest one and tell user to remove/archive the rest.

## Output Rules

Always enforce:
1. Description body is the source of truth.
2. One date, one status issue.
3. Every major update rewrites the full body, not incremental comments.
4. Include concrete values (commit hashes, dates, command results).
5. Use absolute dates (`YYYY-MM-DD`) instead of relative words.
6. Ensure the body represents cumulative same-day progress to current time.

## Fast Path Commands

Use this helper when available:
1. Run `scripts/build_status_snapshot.sh` to collect git snapshot text quickly.
2. Merge script output into template sections.
3. Upsert issue through Linear MCP read/write calls.

## References

1. Template: `references/daily_status_template.md`
2. Snapshot script: `scripts/build_status_snapshot.sh`
