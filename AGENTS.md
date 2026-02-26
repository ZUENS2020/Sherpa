## Skills

A skill is a set of local instructions stored in a `SKILL.md` file.

### Available skills

- linear-daily-project-status: Keep one daily project-level status issue in Linear for major repository changes. If today's issue exists, overwrite its description; if not, create one. Linear read/write must use MCP. (file: `/Users/zuens2020/Documents/Sherpa/.codex/skills/linear-daily-project-status/SKILL.md`)

### How to use skills

- Trigger `linear-daily-project-status` only when user explicitly sends `$linear-daily-project-status` (or explicitly asks to execute this skill).
- Always read/write Linear through MCP tools, not manual web editing.
- Before each write, read today's existing issue content first and upsert a cumulative same-day summary.
- Prefer overwriting the same-day status issue body instead of writing comments.
- Use label `project status` and keep one canonical issue per date.
- Enforce `project status` label on every upsert (both create and update).

## Major Change Rule (Linear)

For every major project change or improvement, follow this workflow strictly:

1. Before writing code, create a new Linear issue via MCP.
2. Assign the issue to yourself.
3. Set issue status to `In Progress`.
4. Complete implementation and verification.
5. Set issue status to `Done`.
6. Add one final comment summarizing:
   - what changed,
   - validation/test result,
   - impact/risk notes.

Constraints:

1. Do not start implementation before the issue is created and set to `In Progress`.
2. Do not mark `Done` without a completion summary comment.
