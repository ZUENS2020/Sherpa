#!/usr/bin/env bash
set -euo pipefail

# Build a compact git snapshot for daily Linear project-status upserts.
# Usage: build_status_snapshot.sh [repo_root] [current_branch] [main_branch]

repo_root="${1:-.}"
current_branch="${2:-}"
main_branch="${3:-main}"

cd "$repo_root"

if [[ -z "${current_branch}" ]]; then
  current_branch="$(git branch --show-current)"
fi

today="$(date +%F)"
repo_url="$(git remote get-url origin 2>/dev/null || echo "N/A")"

main_head="$(git show -s --format='%h %ci %s' "${main_branch}" 2>/dev/null || echo "N/A")"
current_head="$(git show -s --format='%h %ci %s' "${current_branch}" 2>/dev/null || echo "N/A")"

main_remote_sync="$(git rev-list --left-right --count "${main_branch}...origin/${main_branch}" 2>/dev/null || echo "N/A")"
current_remote_sync="$(git rev-list --left-right --count "${current_branch}...origin/${current_branch}" 2>/dev/null || echo "N/A")"
divergence_count="$(git rev-list --left-right --count "${main_branch}...${current_branch}" 2>/dev/null || echo "N/A")"

main_unique_raw="$(git log --oneline "${current_branch}..${main_branch}" 2>/dev/null || true)"
current_unique_raw="$(git log --oneline "${main_branch}..${current_branch}" 2>/dev/null || true)"

if [[ -n "${main_unique_raw}" ]]; then
  main_unique="$(printf '%s\n' "${main_unique_raw}" | sed 's/^/- /')"
else
  main_unique="- (none)"
fi

if [[ -n "${current_unique_raw}" ]]; then
  current_unique="$(printf '%s\n' "${current_unique_raw}" | sed 's/^/- /')"
else
  current_unique="- (none)"
fi

worktree_status="clean"
if [[ -n "$(git status --porcelain 2>/dev/null || true)" ]]; then
  worktree_status="dirty"
fi

cat <<EOF
DATE=${today}
REPO_URL=${repo_url}
CURRENT_BRANCH=${current_branch}
MAIN_BRANCH=${main_branch}
WORKTREE_STATUS=${worktree_status}
MAIN_HEAD=${main_head}
CURRENT_HEAD=${current_head}
MAIN_REMOTE_SYNC=${main_remote_sync}
CURRENT_REMOTE_SYNC=${current_remote_sync}
DIVERGENCE_COUNT=${divergence_count}
MAIN_UNIQUE_COMMITS<<__BLOCK__
${main_unique}
__BLOCK__
CURRENT_UNIQUE_COMMITS<<__BLOCK__
${current_unique}
__BLOCK__
EOF
