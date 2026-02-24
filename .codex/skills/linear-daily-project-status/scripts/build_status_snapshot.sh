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

main_remote_sync="$(git rev-list --left-right --count "${main_branch}...origin/${main_branch}" 2>/dev/null || echo "N/A")"
current_remote_sync="$(git rev-list --left-right --count "${current_branch}...origin/${current_branch}" 2>/dev/null || echo "N/A")"
divergence_count="$(git rev-list --left-right --count "${main_branch}...${current_branch}" 2>/dev/null || echo "N/A")"

main_unique_size="$(git rev-list --count "${current_branch}..${main_branch}" 2>/dev/null || echo "N/A")"
current_unique_size="$(git rev-list --count "${main_branch}..${current_branch}" 2>/dev/null || echo "N/A")"

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
MAIN_REMOTE_SYNC=${main_remote_sync}
CURRENT_REMOTE_SYNC=${current_remote_sync}
DIVERGENCE_COUNT=${divergence_count}
MAIN_UNIQUE_SIZE=${main_unique_size}
CURRENT_BRANCH_UNIQUE_SIZE=${current_unique_size}
EOF
