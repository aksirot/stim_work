#!/usr/bin/env bash
# Tidy local git clutter: prune dead worktrees and delete branches already merged
# into the base branch. SAFE BY DEFAULT — prints what it *would* do; pass --apply
# to actually delete. Never touches the current branch, the base branch, or any
# branch still checked out in a worktree.
#
#   bash scripts/git_cleanup.sh                 # dry-run against 'main'
#   bash scripts/git_cleanup.sh --apply         # actually delete
#   bash scripts/git_cleanup.sh --base=develop  # use a different base branch
set -euo pipefail

APPLY=0
MAIN=main
for a in "$@"; do
  case "$a" in
    --apply)      APPLY=1 ;;
    --base=*)     MAIN="${a#--base=}" ;;
    -h|--help)    echo "usage: git_cleanup.sh [--apply] [--base=<branch>]"; exit 0 ;;
    *)            echo "unknown arg: $a (try --help)"; exit 2 ;;
  esac
done

cd "$(git rev-parse --show-toplevel)"
cur="$(git symbolic-ref --quiet --short HEAD || echo DETACHED)"
echo "repo : $(pwd)"
echo "base : $MAIN   current: $cur"
[ "$APPLY" -eq 1 ] && echo "mode : APPLY (deleting)" || echo "mode : dry-run (pass --apply to delete)"
echo

# 1) Drop admin records for worktrees whose directory is already gone.
echo "== prune worktrees with a missing directory =="
if [ "$APPLY" -eq 1 ]; then git worktree prune -v; else git worktree prune -n -v; fi
echo

# Branches currently checked out in some worktree must never be deleted.
checked_out="$(git worktree list --porcelain \
  | awk '/^branch /{sub("refs/heads/","",$2); print $2}')"

# 2) List present worktrees whose branch is already merged (you may want to remove these).
echo "== worktrees whose branch is merged into $MAIN (remove manually if done) =="
found_wt=0
while read -r path; do
  [ -z "$path" ] && continue
  b="$(git -C "$path" symbolic-ref --quiet --short HEAD 2>/dev/null || echo)"
  [ -z "$b" ] && continue
  if git merge-base --is-ancestor "$b" "$MAIN" 2>/dev/null; then
    echo "  $path  [$b]  ->  git worktree remove \"$path\""
    found_wt=1
  fi
done < <(git worktree list --porcelain | awk '/^worktree /{print $2}' | grep -F "/.claude/worktrees/" || true)
[ "$found_wt" -eq 0 ] && echo "  (none)"
echo

# 3) Delete local branches fully merged into the base branch.
echo "== local branches merged into $MAIN =="
any=0
while read -r b; do
  [ -z "$b" ] && continue
  case "$b" in "$MAIN"|"$cur") continue ;; esac
  if printf '%s\n' "$checked_out" | grep -qxF "$b"; then
    echo "  skip   $b (checked out in a worktree)"
    continue
  fi
  any=1
  if [ "$APPLY" -eq 1 ]; then
    git branch -d "$b" >/dev/null && echo "  delete $b"
  else
    echo "  would delete $b"
  fi
done < <(git branch --merged "$MAIN" --format '%(refname:short)')
[ "$any" -eq 0 ] && echo "  (none)"
echo
echo "done."
