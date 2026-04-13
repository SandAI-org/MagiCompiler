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

# Robust PR checkout for self-hosted runners.
#
# Strategy:
#   1. Download HEAD tarball via GitHub API (fast, single-file HTTP)
#   2. Fallback to git-fetch if API download fails
#
# Required env vars:
#   REPO_URL     – https clone URL          (e.g. https://github.com/org/repo.git)
#   BASE_SHA     – PR base commit SHA
#   HEAD_SHA     – PR head commit SHA
#   GITHUB_TOKEN – GitHub token for API auth (set automatically by Actions)
#
# Optional env vars:
#   GITHUB_REPOSITORY – owner/repo (for API URL construction)

set -euo pipefail

MAX_ATTEMPTS=10
SLEEP_SECS=5

# ── Derive repo slug from REPO_URL if GITHUB_REPOSITORY is not set ──
if [ -z "${GITHUB_REPOSITORY:-}" ]; then
    GITHUB_REPOSITORY=$(echo "$REPO_URL" | sed -E 's|.*github\.com/||; s|\.git$||')
fi

# ── Method 1: tarball via GitHub API ─────────────────────────────────
download_tarball() {
    local tarball_url="https://api.github.com/repos/${GITHUB_REPOSITORY}/tarball/${HEAD_SHA}"
    local tarball="/tmp/checkout_${HEAD_SHA}.tar.gz"

    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        echo "[checkout] tarball download attempt ${attempt}/${MAX_ATTEMPTS}"
        if curl -fsSL --retry 3 --retry-delay 2 \
            -H "Authorization: Bearer ${GITHUB_TOKEN}" \
            -H "Accept: application/vnd.github+json" \
            -L "$tarball_url" -o "$tarball"; then

            echo "[checkout] tarball downloaded, extracting..."
            # tarball contains a top-level directory like org-repo-<sha>/
            # strip it so files land in current directory
            tar xzf "$tarball" --strip-components=1
            rm -f "$tarball"
            return 0
        fi

        if [ "$attempt" -eq "$MAX_ATTEMPTS" ]; then
            echo "[checkout] tarball download failed after ${MAX_ATTEMPTS} attempts"
            return 1
        fi
        sleep "$SLEEP_SECS"
    done
}

# ── Method 2: git fetch (fallback) ──────────────────────────────────
git_fetch_fallback() {
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git remote set-url origin "$REPO_URL" 2>/dev/null || git remote add origin "$REPO_URL"
    else
        git init .
        git remote add origin "$REPO_URL"
    fi

    cleanup_git_locks() {
        rm -f .git/shallow.lock .git/index.lock .git/packed-refs.lock \
              .git/FETCH_HEAD.lock .git/config.lock
    }

    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        cleanup_git_locks
        echo "[checkout] git-fetch attempt ${attempt}/${MAX_ATTEMPTS}"

        if [ $((attempt % 2)) -eq 1 ]; then
            fetch_cmd=(timeout 2m git -c http.lowSpeedLimit=100 -c http.lowSpeedTime=10
                fetch --no-tags --prune --depth=1 origin "$BASE_SHA" "$HEAD_SHA")
        else
            fetch_cmd=(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY
                timeout 2m git -c http.proxy= -c https.proxy= -c http.lowSpeedLimit=1 -c http.lowSpeedTime=30
                fetch --no-tags --prune --depth=1 origin "$BASE_SHA" "$HEAD_SHA")
        fi

        if "${fetch_cmd[@]}"; then
            cleanup_git_locks
            git checkout --force "$HEAD_SHA"
            git clean -fdx
            git reset --hard "$HEAD_SHA"
            return 0
        fi

        if [ "$attempt" -eq "$MAX_ATTEMPTS" ]; then
            echo "[checkout] git-fetch failed after ${MAX_ATTEMPTS} attempts"
            return 1
        fi
        cleanup_git_locks
        sleep "$SLEEP_SECS"
    done
}

# ── Main ─────────────────────────────────────────────────────────────
echo "[checkout] HEAD_SHA=${HEAD_SHA}"
echo "[checkout] BASE_SHA=${BASE_SHA}"

if download_tarball; then
    echo "[checkout] tarball checkout succeeded"
else
    echo "[checkout] tarball failed, falling back to git-fetch"
    git_fetch_fallback
fi

echo "[checkout] done"
