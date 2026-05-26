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

# Save a code-review report to a markdown file under the current repo's
# .cursor/code-reviews/ directory.
#
# Usage:
#   bash save_report.sh <mode> <identifier> <report-file>
#
# Arguments:
#   mode        : "pr"  or  "branch"
#   identifier  : for mode=pr     -> the PR number (e.g. "123")
#                 for mode=branch -> the branch name (e.g. "feat/cuda-fusion")
#   report-file : path to a file containing the rendered markdown report body
#                 (typically /tmp/review.md written by the agent before calling
#                 this script). Pass "-" to read the body from stdin.
#
# Output (stdout): the absolute path of the saved file. Print this back to the
# user so they can open it.
#
# Examples:
#   bash save_report.sh pr 123 /tmp/review.md
#   bash save_report.sh branch "$(git rev-parse --abbrev-ref HEAD)" /tmp/review.md
#   echo "$REPORT" | bash save_report.sh branch my-feat -

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <mode:pr|branch> <identifier> <report-file|->" >&2
    exit 2
fi

MODE="$1"
IDENT="$2"
SRC="$3"

if [[ "$MODE" != "pr" && "$MODE" != "branch" ]]; then
    echo "ERROR: mode must be 'pr' or 'branch', got '$MODE'." >&2
    exit 2
fi

if ! REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null); then
    echo "ERROR: save_report.sh must be run inside a git repository." >&2
    exit 1
fi

SAVE_DIR="${REPO_ROOT}/.cursor/code-reviews"
mkdir -p "$SAVE_DIR"

# Build a filesystem-safe identifier (strip everything that isn't alnum/dash/dot/underscore).
SAFE_IDENT=$(printf '%s' "$IDENT" | tr '/' '-' | sed 's/[^A-Za-z0-9._-]/_/g')

# Capture short SHA of HEAD for traceability (best-effort).
SHORT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")

TIMESTAMP=$(date '+%Y%m%d-%H%M%S')
ISO_NOW=$(date '+%Y-%m-%d %H:%M:%S %Z')

case "$MODE" in
    pr)     FNAME="pr-${SAFE_IDENT}-${SHORT_SHA}-${TIMESTAMP}.md" ;;
    branch) FNAME="branch-${SAFE_IDENT}-${SHORT_SHA}-${TIMESTAMP}.md" ;;
esac

OUT="${SAVE_DIR}/${FNAME}"

# Read body from file or stdin.
if [[ "$SRC" == "-" ]]; then
    BODY=$(cat)
else
    if [[ ! -f "$SRC" ]]; then
        echo "ERROR: report file '$SRC' not found." >&2
        exit 1
    fi
    BODY=$(cat "$SRC")
fi

# Determine git remote / repo URL for the metadata block (best-effort).
REMOTE_URL=$(git config --get remote.origin.url 2>/dev/null || echo "(not a git repo)")
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "(detached or not a git repo)")

yaml_quote() {
    local value="$1"
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    printf '"%s"' "$value"
}

# Write metadata header + body.
{
    printf -- '---\n'
    printf 'mode: %s\n' "$(yaml_quote "$MODE")"
    printf 'identifier: %s\n' "$(yaml_quote "$IDENT")"
    printf 'reviewed_at: %s\n' "$(yaml_quote "$ISO_NOW")"
    printf 'repo_remote: %s\n' "$(yaml_quote "$REMOTE_URL")"
    printf 'current_branch: %s\n' "$(yaml_quote "$CURRENT_BRANCH")"
    printf 'head_sha: %s\n' "$(yaml_quote "$SHORT_SHA")"
    printf 'reviewer: %s\n' "$(yaml_quote "cursor-agent (code-review skill)")"
    printf -- '---\n\n'
    printf '%s\n' "$BODY"
} > "$OUT"

# Add to .gitignore so local review reports stay out of commits unless the user
# explicitly opts in.
GITIGNORE="${REPO_ROOT}/.cursor/.gitignore"
if [[ ! -f "$GITIGNORE" ]] || ! grep -qxF 'code-reviews/' "$GITIGNORE" 2>/dev/null; then
    mkdir -p "$(dirname "$GITIGNORE")"
    printf 'code-reviews/\n' >> "$GITIGNORE"
fi

echo "$OUT"
