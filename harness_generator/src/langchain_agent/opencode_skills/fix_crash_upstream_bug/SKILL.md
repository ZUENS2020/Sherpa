---
name: fix_crash_upstream_bug
description: Apply minimal upstream source fixes for verified real crashes without masking behavior.
compatibility: opencode
metadata:
  stage: fix-crash-upstream-bug
  owner: sherpa
---

## What this skill does
Apply minimal correctness/security fixes in upstream code paths for reproducible real bugs.

## When to use this skill
Use this skill only when crash analysis indicates `real_bug` and fix should target upstream code.

## Required inputs
- `crash_info.md`
- `crash_analysis.md`
- crash artifact metadata from coordinator context

## Required outputs
- minimal upstream fix patch (not harness workaround)

## Workflow
1. Locate the upstream fault path from crash evidence.
2. Implement a minimal root-cause fix.
3. Preserve harness behavior and crash-path semantics.

## Constraints
- Avoid broad refactors and behavior masking.
- Do not "fix" by disabling checks or bypassing parser logic.
- Must produce textual code changes; pure no-op is invalid.
- Do not bypass acceptance by tampering with `fuzz/repo_understanding.json`.
- Use explicit actions when diagnostics include concrete paths: `Read and fix <path>[:line]`.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Acceptance checklist
- Fix addresses upstream root cause directly.
- Delta is minimal, targeted, and auditable.
- No doc-only/no-op patch.

## Done contract
- Write one key modified path into `./done`.
