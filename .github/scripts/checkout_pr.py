#!/usr/bin/env python3

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

"""Robust PR checkout for self-hosted runners.

Strategy: tarball via GitHub API → git-fetch fallback.

Env vars: REPO_URL, BASE_SHA, HEAD_SHA, GITHUB_TOKEN, GITHUB_REPOSITORY.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time

RETRIES = 10
SLEEP = 5

REPO_URL = os.environ["REPO_URL"]
BASE_SHA = os.environ["BASE_SHA"]
HEAD_SHA = os.environ["HEAD_SHA"]
TOKEN = os.environ["GITHUB_TOKEN"]
REPO = os.environ.get("GITHUB_REPOSITORY") or re.sub(r".*github\.com/", "", REPO_URL).removesuffix(".git")


def log(msg: str) -> None:
    print(f"[checkout] {msg}", flush=True)


def sh(cmd: str, *, check: bool = True, env: dict | None = None) -> int:
    """Run a shell command, return exit code."""
    log(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check, env=env).returncode


def cleanup_locks() -> None:
    for f in glob.glob(".git/*.lock"):
        os.remove(f)


def no_proxy_env() -> dict[str, str]:
    env = os.environ.copy()
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(k, None)
    return env


# ── Method 1: git fetch ─────────────────────────────────────────────


def git_fetch_checkout() -> bool:
    if sh("git rev-parse --is-inside-work-tree", check=False) != 0:
        sh("git init .")
    if sh(f"git remote set-url origin {REPO_URL}", check=False) != 0:
        sh(f"git remote add origin {REPO_URL}")

    for i in range(1, RETRIES + 1):
        cleanup_locks()
        log(f"fetch attempt {i}/{RETRIES}")

        if i % 2 == 1:
            log("mode=proxy strict")
            cmd = (
                f"timeout 120 git -c http.lowSpeedLimit=100 -c http.lowSpeedTime=10"
                f" fetch --no-tags --prune --depth=1 origin {BASE_SHA} {HEAD_SHA}"
            )
            env = None
        else:
            log("mode=direct relaxed")
            cmd = (
                f"timeout 120 git -c http.proxy= -c https.proxy="
                f" -c http.lowSpeedLimit=1 -c http.lowSpeedTime=30"
                f" fetch --no-tags --prune --depth=1 origin {BASE_SHA} {HEAD_SHA}"
            )
            env = no_proxy_env()

        if sh(cmd, check=False, env=env) == 0:
            cleanup_locks()
            sh(f"git checkout --force {HEAD_SHA}")
            sh("git clean -fdx")
            sh(f"git reset --hard {HEAD_SHA}")
            return True

        if i < RETRIES:
            cleanup_locks()
            log(f"retry in {SLEEP}s")
            time.sleep(SLEEP)

    log("git-fetch failed")
    return False


# ── Method 2: tarball + synthetic git history ────────────────────────


def _download(sha: str, dest: str) -> bool:
    url = f"https://api.github.com/repos/{REPO}/tarball/{sha}"
    for i in range(1, RETRIES + 1):
        via = "proxy" if i % 2 == 1 else "direct"
        log(f"tarball({sha[:8]}) attempt {i}/{RETRIES} via {via}")
        curl = [
            "curl",
            "-fSL",
            "--retry",
            "2",
            "--retry-delay",
            "3",
            "--connect-timeout",
            "15",
            "--max-time",
            "180",
            "-H",
            f"Authorization: Bearer {TOKEN}",
            "-H",
            "Accept: application/vnd.github+json",
            "-o",
            dest,
            url,
        ]
        env = no_proxy_env() if via == "direct" else None
        if subprocess.run(curl, check=False, env=env).returncode == 0:
            return True
        if i < RETRIES:
            time.sleep(SLEEP)
    return False


def _extract(tar_path: str) -> None:
    with tarfile.open(tar_path, "r:gz") as tf:
        prefix = os.path.commonprefix(tf.getnames()).rstrip("/")
        for m in tf.getmembers():
            if prefix:
                m.name = m.name[len(prefix) :].lstrip("/")
            if m.name:
                tf.extract(m, ".", filter="data")


def _commit(msg: str, date: str) -> None:
    sh("git add -A")
    env = {**os.environ, "GIT_COMMITTER_DATE": date, "GIT_AUTHOR_DATE": date}
    sh(f'git commit --allow-empty -m "{msg}"', env=env)


def tarball_checkout() -> bool:
    head_tar, base_tar = "/tmp/_head.tar.gz", "/tmp/_base.tar.gz"
    try:
        if not _download(HEAD_SHA, head_tar) or not _download(BASE_SHA, base_tar):
            return False

        shutil.rmtree(".git", ignore_errors=True)
        sh("git init . && git config user.email ci@sandai.org && git config user.name CI")

        # commit base
        _extract(base_tar)
        _commit(f"base {BASE_SHA}", "2000-01-01T00:00:00Z")

        # commit head
        sh("git rm -rf .", check=False)
        _extract(head_tar)
        _commit(f"head {HEAD_SHA}", "2000-01-02T00:00:00Z")

        # make real SHAs resolvable via git-replace
        base_local = subprocess.check_output(["git", "rev-parse", "HEAD~1"], text=True).strip()
        head_local = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        sh(f"git replace {BASE_SHA} {base_local}", check=False)
        sh(f"git replace {HEAD_SHA} {head_local}", check=False)
        log(f"synthetic: BASE {BASE_SHA[:8]}→{base_local[:8]}, HEAD {HEAD_SHA[:8]}→{head_local[:8]}")
        return True
    finally:
        for f in (head_tar, base_tar):
            try:
                os.remove(f)
            except FileNotFoundError:
                pass


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    log(f"HEAD={HEAD_SHA}, BASE={BASE_SHA}")

    if tarball_checkout():
        log("tarball succeeded")
    elif git_fetch_checkout():
        log("git-fetch fallback succeeded")
    else:
        log("all methods failed")
        sys.exit(1)

    sh(f"git diff --stat {BASE_SHA} {HEAD_SHA} | tail -3")
    log("done")


if __name__ == "__main__":
    main()
