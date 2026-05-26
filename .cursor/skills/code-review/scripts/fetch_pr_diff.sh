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

# Fetch a GitHub PR's diff, description, and changed files for review.
#
# Usage:
#   bash fetch_pr_diff.sh <pr-url-or-number> [repo]
#
# Examples:
#   bash fetch_pr_diff.sh 123
#   bash fetch_pr_diff.sh https://github.com/owner/repo/pull/123
#   bash fetch_pr_diff.sh 123 owner/repo
#
# Output (stdout) is structured for the agent to parse:
#   === PR META ===
#   <title, author, base, head, files-changed summary>
#   === PR DESCRIPTION ===
#   <body>
#   === CHANGED FILES ===
#   <file list with +/- counts>
#   === DIFF ===
#   <unified diff>

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <pr-url-or-number> [repo]" >&2
    exit 2
fi

if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: gh CLI is not installed. Install it from https://cli.github.com/." >&2
    exit 1
fi

INPUT="$1"
REPO_FLAG=()

if [[ "$INPUT" =~ ^https?:// ]]; then
    PR_REF="$INPUT"
elif [[ "$INPUT" =~ ^[0-9]+$ ]]; then
    PR_REF="$INPUT"
    if [[ $# -ge 2 ]]; then
        REPO_FLAG=(--repo "$2")
    fi
else
    echo "ERROR: '$INPUT' is neither a PR URL nor a number." >&2
    exit 2
fi

echo "=== PR META ==="
gh pr view "$PR_REF" "${REPO_FLAG[@]}" \
    --json number,title,author,baseRefName,headRefName,additions,deletions,changedFiles,state,isDraft \
    --template '#{{.number}}  {{.title}}
author : {{.author.login}}
base   : {{.baseRefName}}
head   : {{.headRefName}}
state  : {{.state}}{{if .isDraft}} (DRAFT){{end}}
changes: +{{.additions}} / -{{.deletions}} across {{.changedFiles}} files
'

echo ""
echo "=== PR DESCRIPTION ==="
gh pr view "$PR_REF" "${REPO_FLAG[@]}" --json body --jq '.body // "(no description)"'

echo ""
echo "=== CHANGED FILES ==="
gh pr view "$PR_REF" "${REPO_FLAG[@]}" --json files \
    --jq '.files[] | "  \(.additions | tostring | (" " * (5 - length)) + .)+/\(.deletions | tostring | (" " * (5 - length)) + .)-  \(.path)"'

echo ""
echo "=== EXISTING REVIEW COMMENTS ==="
PR_NUM=$(gh pr view "$PR_REF" "${REPO_FLAG[@]}" --json number --jq '.number')
REPO_FULL=$(gh pr view "$PR_REF" "${REPO_FLAG[@]}" --json headRepository,headRepositoryOwner --jq '"\(.headRepositoryOwner.login)/\(.headRepository.name)"' 2>/dev/null \
            || gh repo view --json nameWithOwner --jq '.nameWithOwner')
gh api "repos/${REPO_FULL}/pulls/${PR_NUM}/comments" \
    --jq '.[] | "  [\(.user.login) on \(.path):\(.line // .original_line // "?")] \(.body | split("\n")[0])"' \
    2>/dev/null || echo "  (none or API failed)"

echo ""
echo "=== DIFF ==="
gh pr diff "$PR_REF" "${REPO_FLAG[@]}"
