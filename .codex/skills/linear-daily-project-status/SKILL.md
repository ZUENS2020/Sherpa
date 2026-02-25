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

After trigger is received, produce a same-day final-style summary (full and complete) with focus on material changes:
1. Workflow graph, node routing, retry/termination policy.
2. Docker/compose/container network/build/runtime path.
3. Memory/session semantics and cross-step context flow.
4. Build/fuzz failure handling and repair loop behavior.
5. API/Frontend behavior changes and runtime default changes.
6. Docs/tests that alter operational decisions.

## Required Targets

Use these defaults unless the user explicitly overrides:
1. Team: `Sherpa_XDU`
2. Label: `project status`
3. Issue title format: `Sherpa 项目进度同步（YYYY-MM-DD）`
4. Status: `In Progress`
5. Branch progress and branch-diff summary: mandatory in every write.
6. Project handling: **do not auto-attach project**. Only attach/update `project` when the user explicitly requests a project.

## MCP Access Policy (Mandatory)

Use Linear MCP for all reads and writes.
1. Read with MCP:
   1. `list_issue_labels` to confirm label.
   2. `list_issues` and `get_issue` to find today's issue and inspect body.
   3. Only when user explicitly asks to set/validate project: `list_projects` / `get_project`.
2. Write with MCP:
   1. `update_issue` to overwrite today's `description`.
   2. `create_issue` only when today's issue does not exist.
   3. Always include label `project status` in both `update_issue` and `create_issue`.
   4. Do not send `project` field in `create_issue` / `update_issue` unless user explicitly requested a project binding in this turn.
3. Do not use comment logs (`create_comment`) for status snapshots unless the user explicitly asks.
4. Do not ask the user to update Linear manually when MCP is available.

## Label Enforcement (Mandatory)

The daily status issue must always carry label `project status`.
1. Before write, call `list_issue_labels` and verify `project status` exists.
2. If missing, call `create_issue_label` to create `project status` for team `Sherpa_XDU`.
3. On create, call `create_issue` with `labels: [\"project status\"]`.
4. On update, call `update_issue` with `labels: [\"project status\"]` together with `description` to enforce label presence.

## Pre-write Reconciliation (Mandatory)

Before every write, read today's full issue body first and reconcile.
1. If today's issue exists, always call MCP `get_issue` and load `description`.
2. Use old body only as context, then regenerate today's full snapshot from current project inspection.
3. Keep the work section as "当日完成工作清单（全量）", covering all completed work up to now, not only the newest delta.
4. Drop stale items that cannot be verified in current repository state.
5. Deduplicate by change topic and test command/result pairs.
6. Write back a fully replaced body after regeneration.

## Upsert Workflow (No Comments)

Follow this sequence strictly:
1. Collect snapshot data:
   1. Current branch and working tree status.
   2. Branch progress and divergence (`main` vs current working branch), including unique change size on each side.
   3. Remote sync state for each compared branch.
   4. Convert commit-level facts into high-level themes and completion items.
   5. Key runtime/workflow/config checks from current repository files.
   6. Latest validation result (tests/build/run command + pass/fail).
   7. Major changes, risks, and next actions.
2. If today's issue exists, read current `description` first with MCP `get_issue`.
3. Reconcile old body + new facts, then regenerate one cumulative same-day snapshot.
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
4. Keep output highly summarized; use change themes, impact, and completion status.
5. Use absolute dates (`YYYY-MM-DD`) instead of relative words.
6. Ensure the body represents cumulative same-day progress to current time.
7. Summary style must look like a same-day final comprehensive summary, not phased notes.
8. Branch progress and branch-diff summary is mandatory.
9. Do not use heading text `今日关键进展（YYYY-MM-DD，累计到当前时点）`.
10. Do not output raw commit hashes, commit messages, or commit lists in the final description.

## Fast Path Commands

Use this helper when available:
1. Run `scripts/build_status_snapshot.sh` to collect git snapshot text quickly.
2. Merge script output into template sections.
3. Upsert issue through Linear MCP read/write calls.

## References

1. Template: `references/daily_status_template.md`
2. Snapshot script: `scripts/build_status_snapshot.sh`
