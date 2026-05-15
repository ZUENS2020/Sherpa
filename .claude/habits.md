# User Habits — Sherpa

Learned preferences. Auto-updated when clear patterns emerge (not one-offs).

<!-- HABITS-START -->

## Workflow
- Push directly to dev branch; deploy auto-triggers via `deploy-dev.yml`
- Fix bugs in-place → commit with `fix:` prefix → `git pull --rebase && git push origin dev`
- Branch strategy: PRs target dev only, never main. Deploy via CI.
- Verify fixes by SSH monitoring on dev, not unit tests
- Prefer practical results over process: direct dev pushes, skip PR overhead for fixes

## Code Style
- Commit format: `type: short description` (e.g., `fix: add idle timeout override`)
- Python: no comments unless the WHY is non-obvious; skip docstrings
- Config: env vars over hardcoded values; follow existing patterns (e.g., `_analysis_opencode_idle_timeout_sec()`)
- Env vars: use descriptive names, reasonable defaults, min/max clamping

## Target Preferences
- Prefer newer/less-mature C/C++ libraries (easier vulns than heavily-fuzzed codebases)
- Good targets: image parsers, compression libraries, protocol parsers, JSON parsers
- Avoid: libpng, zlib, lz4 — already heavily tested
- Accept: cJSON, uriparser, libwebp, stb-style single-header libs

## Debugging
- SSH into dev server: `ssh -i ~/.ssh/id_ed25519 deploy@frp-jar.com -p 63893`
- Monitor fuzz pods via `kubectl logs` + grep for fuzz stats
- Query API via `kubectl exec` into web pod (no port-forward)
- Check 7 CKs as pipeline health checklist
- Prefer terminal-based monitoring over web UI

## Communication
- Prefers Chinese for discussion, code/commands in English
- Terse responses, no fluff, no trailing summaries
- Direct answers: state findings, don't narrate thought process

<!-- HABITS-END -->
