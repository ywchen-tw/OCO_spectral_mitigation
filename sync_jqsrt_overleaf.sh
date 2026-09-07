#!/usr/bin/env bash
# Push manuscript/JQSRT_draft to the Overleaf project via its git bridge.
#
#   ./sync_jqsrt_overleaf.sh                 normal sync: commit local changes, pull, push
#   ./sync_jqsrt_overleaf.sh --adopt-remote  first run only: graft local content on top of
#                                            Overleaf's history (local files win outright)
#   ./sync_jqsrt_overleaf.sh -m "message"    custom commit message
#
# manuscript/ is ignored by the outer analysis repo, so JQSRT_draft carries its own
# talks only to Overleaf. Authentication is username "git" plus an Overleaf
# git token (Account Settings -> Git integration); osxkeychain caches it after the
# first push.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/manuscript/JQSRT_draft"
REMOTE=overleaf
BRANCH=main

ADOPT=0
MSG="Sync from local $(date '+%Y-%m-%d %H:%M')"
while [ $# -gt 0 ]; do
  case "$1" in
    --adopt-remote) ADOPT=1; shift ;;
    -m) MSG="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$REPO"
git rev-parse --git-dir >/dev/null 2>&1 || { echo "$REPO is not a git repository" >&2; exit 1; }
git remote get-url "$REMOTE" >/dev/null 2>&1 || { echo "no '$REMOTE' remote in $REPO" >&2; exit 1; }

git add -A

if [ "$ADOPT" -eq 1 ]; then
  git fetch "$REMOTE"
  # Adopt Overleaf's history, keep our tree: one commit, no merge conflicts.
  # Files that exist only on Overleaf are removed by this commit.
  git reset --soft "$REMOTE/$BRANCH"
  git commit -q -m "$MSG" || echo "nothing to commit (already identical to Overleaf)"
  git push "$REMOTE" "HEAD:$BRANCH"
  echo "adopted $REMOTE/$BRANCH and pushed local content"
  exit 0
fi

if git diff --cached --quiet; then
  echo "no local changes to commit"
else
  git commit -q -m "$MSG"
  echo "committed: $MSG"
fi

git fetch "$REMOTE"
if ! git pull --rebase "$REMOTE" "$BRANCH"; then
  echo >&2
  echo "Rebase onto $REMOTE/$BRANCH stopped (someone edited the same lines in Overleaf)." >&2
  echo "Resolve in $REPO, then 'git rebase --continue' and re-run this script." >&2
  exit 1
fi

git push "$REMOTE" "HEAD:$BRANCH"
echo "pushed to $(git remote get-url "$REMOTE")"
