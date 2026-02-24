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
