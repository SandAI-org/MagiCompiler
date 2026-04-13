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

# Robust PR checkout with retry for self-hosted runners.
#
# Required env vars (set by the caller / GitHub Actions):
#   REPO_URL   – https clone URL of the repository
#   BASE_SHA   – base commit SHA of the pull request
#   HEAD_SHA   – head commit SHA of the pull request
#
# Usage (in a workflow step):
#   env:
#     REPO_URL: https://github.com/${{ github.repository }}.git
#     BASE_SHA: ${{ github.event.pull_request.base.sha }}
#     HEAD_SHA: ${{ github.event.pull_request.head.sha }}
#   run: bash .github/scripts/checkout_pr.sh

set -euo pipefail

MAX_ATTEMPTS=15
SLEEP_SECS=5

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git remote set-url origin "$REPO_URL" 2>/dev/null || git remote add origin "$REPO_URL"
else
    git init .
    git remote add origin "$REPO_URL"
fi

cleanup_git_locks() {
    rm -f .git/shallow.lock \
          .git/index.lock \
          .git/packed-refs.lock \
          .git/FETCH_HEAD.lock \
          .git/config.lock
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    cleanup_git_locks
    echo "[checkout] attempt ${attempt}/${MAX_ATTEMPTS}"

    if [ $((attempt % 2)) -eq 1 ]; then
        echo "[checkout] mode=proxy strict(lowSpeed=100/10)"
        fetch_cmd=(
            timeout 1m
            git -c http.lowSpeedLimit=100 -c http.lowSpeedTime=10
            fetch --no-tags --prune --depth=1 origin
            "$BASE_SHA" "$HEAD_SHA"
        )
    else
        echo "[checkout] mode=direct relaxed(lowSpeed=1/30)"
        fetch_cmd=(
            env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY
            timeout 1m
            git -c http.proxy= -c https.proxy= -c http.lowSpeedLimit=1 -c http.lowSpeedTime=30
            fetch --no-tags --prune --depth=1 origin
            "$BASE_SHA" "$HEAD_SHA"
        )
    fi

    if "${fetch_cmd[@]}"; then
        break
    fi

    if [ "$attempt" -eq "$MAX_ATTEMPTS" ]; then
        echo "[checkout] failed after ${MAX_ATTEMPTS} attempts"
        exit 1
    fi

    cleanup_git_locks
    echo "[checkout] fetch timeout/failure, retry in ${SLEEP_SECS}s"
    sleep "$SLEEP_SECS"
done

cleanup_git_locks
git checkout --force "$HEAD_SHA"
git clean -fdx
git reset --hard "$HEAD_SHA"
git rev-parse "$BASE_SHA"
git rev-parse "$HEAD_SHA"
git log -1 --oneline
git diff --name-only "${BASE_SHA}..${HEAD_SHA}" | head -n 20
echo "[checkout] done"
