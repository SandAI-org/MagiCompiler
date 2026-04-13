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
#   1. Download HEAD & BASE tarballs via GitHub API  (fast, curl-based)
#      Then synthesize a local git repo with two commits so that
#      downstream steps (git diff, pre-commit, etc.) work normally.
#   2. Fallback to traditional git-fetch if tarball download fails.
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

# ── Helper: download a tarball for a given SHA ───────────────────────
download_sha_tarball() {
    local sha="$1"
    local dest="$2"
    local tarball_url="https://api.github.com/repos/${GITHUB_REPOSITORY}/tarball/${sha}"

    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        echo "[checkout] tarball(${sha:0:8}) attempt ${attempt}/${MAX_ATTEMPTS}"
        if curl -fsSL --retry 3 --retry-delay 2 --max-time 120 \
            -H "Authorization: Bearer ${GITHUB_TOKEN}" \
            -H "Accept: application/vnd.github+json" \
            -L "$tarball_url" -o "$dest"; then
            return 0
        fi
        if [ "$attempt" -eq "$MAX_ATTEMPTS" ]; then
            return 1
        fi
        sleep "$SLEEP_SECS"
    done
}

# ── Method 1: tarball + synthetic git history ────────────────────────
tarball_checkout() {
    local head_tar="/tmp/checkout_head_${HEAD_SHA}.tar.gz"
    local base_tar="/tmp/checkout_base_${BASE_SHA}.tar.gz"

    echo "[checkout] downloading HEAD tarball..."
    if ! download_sha_tarball "$HEAD_SHA" "$head_tar"; then
        echo "[checkout] HEAD tarball download failed"
        return 1
    fi

    echo "[checkout] downloading BASE tarball..."
    if ! download_sha_tarball "$BASE_SHA" "$base_tar"; then
        echo "[checkout] BASE tarball download failed"
        rm -f "$head_tar"
        return 1
    fi

    # Clean working directory (keep .git if it exists, we'll reinit)
    rm -rf .git
    git init .
    git config user.email "ci@sandai.org"
    git config user.name "CI"

    # Commit 1: BASE
    echo "[checkout] extracting BASE tarball..."
    tar xzf "$base_tar" --strip-components=1
    rm -f "$base_tar"
    git add -A
    GIT_COMMITTER_DATE="2000-01-01T00:00:00Z" \
    GIT_AUTHOR_DATE="2000-01-01T00:00:00Z" \
    git commit --allow-empty -m "base ${BASE_SHA}"
    # Tag the commit so we can reference it by the original SHA
    git tag "sha-base"

    # Commit 2: HEAD (clear everything, then extract head tarball)
    git rm -rf . > /dev/null 2>&1 || true
    echo "[checkout] extracting HEAD tarball..."
    tar xzf "$head_tar" --strip-components=1
    rm -f "$head_tar"
    git add -A
    GIT_COMMITTER_DATE="2000-01-02T00:00:00Z" \
    GIT_AUTHOR_DATE="2000-01-02T00:00:00Z" \
    git commit --allow-empty -m "head ${HEAD_SHA}"
    git tag "sha-head"

    # Create replace refs so that `git rev-parse <real-sha>` resolves
    local base_local head_local
    base_local=$(git rev-parse sha-base)
    head_local=$(git rev-parse sha-head)
    git replace "$BASE_SHA" "$base_local" 2>/dev/null || true
    git replace "$HEAD_SHA" "$head_local" 2>/dev/null || true

    echo "[checkout] synthetic git history created"
    echo "[checkout]   BASE ${BASE_SHA} -> ${base_local}"
    echo "[checkout]   HEAD ${HEAD_SHA} -> ${head_local}"
    return 0
}

# ── Method 2: git fetch (fallback) ──────────────────────────────────
git_fetch_fallback() {
    rm -rf .git
    git init .
    git remote add origin "$REPO_URL"

    cleanup_git_locks() {
        rm -f .git/shallow.lock .git/index.lock .git/packed-refs.lock \
              .git/FETCH_HEAD.lock .git/config.lock
    }

    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        cleanup_git_locks
        echo "[checkout] git-fetch attempt ${attempt}/${MAX_ATTEMPTS}"

        if [ $((attempt % 2)) -eq 1 ]; then
            echo "[checkout] mode=proxy strict(lowSpeed=100/10)"
            fetch_cmd=(timeout 2m git -c http.lowSpeedLimit=100 -c http.lowSpeedTime=10
                fetch --no-tags --prune --depth=1 origin "$BASE_SHA" "$HEAD_SHA")
        else
            echo "[checkout] mode=direct relaxed(lowSpeed=1/30)"
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

if tarball_checkout; then
    echo "[checkout] tarball checkout succeeded"
else
    echo "[checkout] tarball failed, falling back to git-fetch"
    git_fetch_fallback
fi

echo "[checkout] verifying..."
git rev-parse "$BASE_SHA"
git rev-parse "$HEAD_SHA"
git log --oneline --all | head -5
git diff --stat "$BASE_SHA" "$HEAD_SHA" | tail -3
echo "[checkout] done"
