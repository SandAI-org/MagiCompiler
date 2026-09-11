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

"""Robust PR checkout: tarball (GitHub API) → git-fetch fallback.

Env: REPO_URL, BASE_SHA, HEAD_SHA, GITHUB_TOKEN, GITHUB_REPOSITORY.
"""

from __future__ import annotations

import glob  # noqa: E401
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time

RETRIES, SLEEP = 10, 5
REPO_URL = os.environ["REPO_URL"]
BASE_SHA = os.environ["BASE_SHA"]
HEAD_SHA = os.environ["HEAD_SHA"]
TOKEN = os.environ["GITHUB_TOKEN"]
REPO = os.environ.get("GITHUB_REPOSITORY") or re.sub(r".*github\.com/", "", REPO_URL).removesuffix(".git")


def _parse_fork_ref(ref: str) -> tuple[str, str] | None:
    """Parse ``user:branch`` fork reference into ``(fork_repo, branch)``.

    Returns ``None`` when *ref* is a plain SHA, tag, or branch name.
    """
    if ":" not in ref:
        return None
    user, branch = ref.split(":", 1)
    if not user or not branch:
        return None
    repo_name = REPO.split("/")[-1]
    return f"{user}/{repo_name}", branch


def log(msg: str) -> None:
    print(f"[checkout] {msg}", flush=True)


def sh(cmd: str, **kw) -> int:
    kw.setdefault("check", True)
    return subprocess.run(cmd, shell=True, **kw).returncode


def env_no_proxy() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k.lower() not in ("http_proxy", "https_proxy")}


# ── tarball checkout ─────────────────────────────────────────────────


def _curl(sha: str, dest: str, repo: str | None = None) -> bool:
    url = f"https://api.github.com/repos/{repo or REPO}/tarball/{sha}"
    for i in range(1, RETRIES + 1):
        via = "proxy" if i % 2 else "direct"
        log(f"tarball({sha[:8]}) attempt {i}/{RETRIES} via {via}")
        rc = subprocess.run(
            [
                "curl",
                "-fSL",
                "--retry",
                "2",
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
            ],
            env=env_no_proxy() if via == "direct" else None,
        ).returncode
        if rc == 0:
            return True
        if i < RETRIES:
            time.sleep(SLEEP)
    return False


def _extract(tar_path: str) -> None:
    with tarfile.open(tar_path, "r:gz") as tf:
        prefix = os.path.commonprefix(tf.getnames()).rstrip("/")
        for m in tf.getmembers():
            m.name = m.name[len(prefix) :].lstrip("/") if prefix else m.name
            if m.name:
                tf.extract(m, ".", filter="data")


def tarball_checkout() -> tuple[str, str] | None:
    """Download HEAD & BASE tarballs, build synthetic two-commit repo.

    Returns ``(local_base_sha, local_head_sha)`` or ``None``.
    """
    h, b = "/tmp/_head.tar.gz", "/tmp/_base.tar.gz"
    try:
        head_ref, head_repo = HEAD_SHA, None
        fork = _parse_fork_ref(HEAD_SHA)
        if fork:
            head_repo, head_ref = fork
            log(f"fork detected: repo={head_repo}, branch={head_ref}")

        if not _curl(head_ref, h, repo=head_repo):
            return None
        if not BASE_SHA:
            log("no BASE_SHA, skipping base tarball")
        elif not _curl(BASE_SHA, b):
            return None

        shutil.rmtree(".git", ignore_errors=True)
        sh("git init . && git config user.email ci@sandai.org && git config user.name CI")

        tarballs = []
        if BASE_SHA:
            tarballs.append((b, f"base {BASE_SHA}", "2000-01-01T00:00:00Z"))
        tarballs.append((h, f"head {HEAD_SHA}", "2000-01-02T00:00:00Z"))

        for tar, msg, date in tarballs:
            if tar == h:
                sh("git rm -rf . 2>/dev/null || true", check=False)
            _extract(tar)
            sh("git add -A")
            env = {**os.environ, "GIT_COMMITTER_DATE": date, "GIT_AUTHOR_DATE": date}
            sh(f'git commit --allow-empty -m "{msg}"', env=env)

        head_l = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if BASE_SHA:
            base_l = subprocess.check_output(["git", "rev-parse", "HEAD~1"], text=True).strip()
        else:
            base_l = head_l
        log(f"synthetic: base→{base_l[:8]}, head→{head_l[:8]}")
        return base_l, head_l
    finally:
        for f in (h, b):
            try:
                os.remove(f)
            except FileNotFoundError:
                pass


# ── git-fetch fallback ───────────────────────────────────────────────


def git_fetch_checkout() -> bool:
    if sh("git rev-parse --is-inside-work-tree", check=False) != 0:
        sh("git init .")
    if sh(f"git remote set-url origin {REPO_URL}", check=False) != 0:
        sh(f"git remote add origin {REPO_URL}")

    fork = _parse_fork_ref(HEAD_SHA)
    if fork:
        fork_repo, fork_branch = fork
        fork_url = f"https://github.com/{fork_repo}.git"
        log(f"fork fetch: {fork_url} branch={fork_branch}")
        sh(f"git remote remove fork 2>/dev/null || true", check=False)
        sh(f"git remote add fork {fork_url}")
        for i in range(1, RETRIES + 1):
            for f in glob.glob(".git/*.lock"):
                os.remove(f)
            log(f"fork-fetch attempt {i}/{RETRIES}")
            proxy_flags = "" if i % 2 else "-c http.proxy= -c https.proxy="
            env = None if i % 2 else env_no_proxy()
            cmd = f"timeout 120 git {proxy_flags} fetch --no-tags --depth=1 fork {fork_branch}"
            if sh(cmd, check=False, env=env) == 0:
                sh(f"git checkout --force FETCH_HEAD && git clean -fdx")
                return True
            if i < RETRIES:
                time.sleep(SLEEP)
        return False

    for i in range(1, RETRIES + 1):
        for f in glob.glob(".git/*.lock"):
            os.remove(f)
        log(f"fetch attempt {i}/{RETRIES}")
        refs_to_fetch = f"{BASE_SHA} {HEAD_SHA}" if BASE_SHA else HEAD_SHA
        if i % 2:
            cmd = f"timeout 120 git -c http.lowSpeedLimit=100 -c http.lowSpeedTime=10 fetch --no-tags --prune --depth=1 origin {refs_to_fetch}"
            env = None
        else:
            cmd = f"timeout 120 git -c http.proxy= -c https.proxy= -c http.lowSpeedLimit=1 -c http.lowSpeedTime=30 fetch --no-tags --prune --depth=1 origin {refs_to_fetch}"
            env = env_no_proxy()
        if sh(cmd, check=False, env=env) == 0:
            sh(f"git checkout --force {HEAD_SHA} && git clean -fdx && git reset --hard {HEAD_SHA}")
            return True
        if i < RETRIES:
            time.sleep(SLEEP)
    return False


# ── main ─────────────────────────────────────────────────────────────


def main() -> None:
    log(f"HEAD={HEAD_SHA}, BASE={BASE_SHA}")
    result = tarball_checkout()
    if result:
        base_ref, head_ref = result
        log("tarball succeeded")
    elif git_fetch_checkout():
        head_ref = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        base_ref = BASE_SHA or head_ref
        log("git-fetch fallback succeeded")
    else:
        log("all methods failed")
        sys.exit(1)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"base_ref={base_ref}\nhead_ref={head_ref}\n")
    if base_ref != head_ref:
        sh(f"git diff --stat {base_ref} {head_ref} | tail -3")
    log("done")


if __name__ == "__main__":
    main()
