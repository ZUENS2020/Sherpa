## Skills

No repository-local skill is currently enabled.

## Major Change Rule (Linear)

For every major project change or improvement, follow this workflow strictly:

1. Before writing code, first clarify the change goal and planned scope (what problem this change solves, and the rough implementation content).
2. Record that goal + scope summary in a new Linear issue via MCP, and auto-apply labels based on change type.
3. Assign the issue to yourself.
4. Set issue status to `In Progress`.
5. Ask the user to confirm whether execution should start.
6. Only after explicit user confirmation, complete implementation and verification.
7. Set issue status to `Done`.
8. Add one final comment summarizing:
   - what changed,
   - validation/test result,
   - impact/risk notes.

Constraints:

1. Do not start implementation before the change purpose and rough scope are written into Linear and the issue is set to `In Progress`.
2. Do not start implementation without explicit user confirmation after the Linear write is complete.
3. Do not mark `Done` without a completion summary comment.
4. Enforce type-based labels on the issue (create labels first if missing, then apply them).

Type-to-label mapping for major changes:

1. Frontend/UI changes -> `frontend`
2. Backend/API/Workflow logic -> `backend`
3. Container/Compose/Deploy/Runtime infra -> `infra`
4. CI/CD and GitHub Actions -> `ci/cd`
5. Docs/README/architecture docs -> `docs`
6. Tests/quality/coverage -> `test`
7. Bug fixes (root-cause correction) -> `bugfix`
8. Refactor without behavior change -> `refactor`
9. Security hardening -> `security`

Labeling rule:

1. Choose at least one primary type label from the mapping above.
2. If multiple types are involved, apply multiple labels (max 3, most impactful first).

Recommended issue body structure for major changes:

1. `Goal`: Why this change is needed.
2. `Planned Scope`: Main modules/files and rough implementation plan.
3. `Out of Scope`: What is intentionally excluded in this round.
