#!/usr/bin/env bash

# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fetch the current local branch's diff against a base branch, for pre-push review.
#
# IMPORTANT — only the changes this branch is *ahead* of base are emitted.
# The diff is always computed from the merge-base (equivalent to `git diff base...HEAD`,
# three-dot), so commits that landed on `base` after this branch forked are NOT included.
# Reviewers should care only about what this branch proposes to merge into base.
#
# Usage:
#   bash fetch_local_diff.sh [base-branch]
#
# If [base-branch] is omitted, tries `origin/main` then `main` then `master`.
#
# Output (stdout) is structured for the agent to parse:
#   === BRANCH META ===
#   <current branch, base, ahead/behind, commit count>
#   === COMMITS ===
#   <git log --oneline merge-base..HEAD>      # ahead-only, oldest first
#   === CHANGED FILES ===
#   <file list with +/- numstat, ahead-only>
#   === DIFF ===
#   <git diff merge-base..HEAD>               # ahead-only, == base...HEAD>

set -euo pipefail

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "ERROR: not inside a git repository." >&2
    exit 1
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" == "HEAD" ]]; then
    echo "ERROR: detached HEAD; checkout a branch first." >&2
    exit 1
fi

BASE="${1:-}"
if [[ -z "$BASE" ]]; then
    for candidate in origin/main main origin/master master; do
        if git rev-parse --verify "$candidate" >/dev/null 2>&1; then
            BASE="$candidate"
            break
        fi
    done
fi

if [[ -z "$BASE" ]]; then
    echo "ERROR: could not auto-detect a base branch. Pass one explicitly: $0 <base-branch>" >&2
    exit 2
fi

if ! git rev-parse --verify "$BASE" >/dev/null 2>&1; then
    echo "ERROR: base '$BASE' does not exist." >&2
    exit 2
fi

MERGE_BASE=$(git merge-base "$BASE" HEAD)
AHEAD=$(git rev-list --count "${MERGE_BASE}..HEAD")
BEHIND=$(git rev-list --count "HEAD..${BASE}")

echo "=== BRANCH META ==="
echo "current branch : $CURRENT_BRANCH"
echo "base           : $BASE"
echo "merge-base     : $MERGE_BASE"
echo "ahead of base  : $AHEAD commit(s)   <-- review scope (these go INTO base)"
echo "behind base    : $BEHIND commit(s)  <-- informational only; NOT reviewed"
echo "diff range     : ${MERGE_BASE}..HEAD   (ahead-only; equivalent to ${BASE}...HEAD)"

echo ""
echo "=== COMMITS (oldest first) ==="
git log --reverse --oneline --no-decorate "${MERGE_BASE}..HEAD"

echo ""
echo "=== CHANGED FILES ==="
git diff --numstat "${MERGE_BASE}..HEAD" \
    | awk '{ printf "  +%-5s -%-5s  %s\n", $1, $2, $3 }'

echo ""
echo "=== UNCOMMITTED CHANGES (working tree + staged) ==="
DIRTY=$(git status --porcelain)
if [[ -n "$DIRTY" ]]; then
    echo "$DIRTY" | sed 's/^/  /'
    echo ""
    echo "  NOTE: uncommitted changes exist. Diff below covers committed changes ONLY."
    echo "        Re-run after committing if you want them reviewed."
else
    echo "  (clean)"
fi

echo ""
echo "=== DIFF (ahead-only: ${MERGE_BASE}..HEAD) ==="
git diff "${MERGE_BASE}..HEAD"
