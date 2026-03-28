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

"""
Check for Chinese characters in source code.

Two modes:
  - CI mode   : set env vars BASE_REF and HEAD_REF to only check the PR diff.
  - Local mode: run without those env vars to scan every tracked file in the repo.

Usage:
  python3 .github/workflows/check_chinese_chars.py          # scan entire repo
  BASE_REF=main HEAD_REF=HEAD python3 ...                   # scan diff only
"""

import os
import re
import subprocess
import sys
from typing import List, Tuple

CHINESE_CHAR_PATTERN = re.compile(
    "["
    "\u4e00-\u9fff"  # CJK Unified Ideographs
    "\u3400-\u4dbf"  # CJK Unified Ideographs Extension A
    "\uf900-\ufaff"  # CJK Compatibility Ideographs
    "\u3000-\u303f"  # CJK Symbols and Punctuation
    "\uff01-\uff5e"  # Fullwidth ASCII variants
    "]"
)

BINARY_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".svg",
        ".webp",
        ".mp3",
        ".mp4",
        ".wav",
        ".avi",
        ".mov",
        ".mkv",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".bin",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".pt",
        ".pth",
        ".onnx",
        ".safetensors",
        ".pickle",
        ".pkl",
        ".pdf",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".eot",
        ".pyc",
        ".o",
        ".a",
        ".nsys-rep",
        ".npz",
        ".npy",
    }
)


def _is_binary(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in BINARY_EXTENSIONS


# ---------------------------------------------------------------------------
# CI mode: only check newly added / modified lines in the PR diff
# ---------------------------------------------------------------------------


def _check_diff(base_sha: str, head_sha: str) -> List[Tuple[str, int, str]]:
    base_sha = subprocess.check_output(["git", "rev-parse", base_sha], text=True).strip()
    head_sha = subprocess.check_output(["git", "rev-parse", head_sha], text=True).strip()

    print(f"[CI mode] Checking diff between {base_sha[:8]} and {head_sha[:8]} ...")

    result = subprocess.run(
        ["git", "diff", "-U0", "--diff-filter=ACM", base_sha, head_sha], capture_output=True, text=True, check=True
    )

    findings: List[Tuple[str, int, str]] = []
    current_file = None
    line_num = 0

    for line in result.stdout.split("\n"):
        if line.startswith("diff --git"):
            parts = line.split(" b/")
            current_file = parts[-1] if len(parts) >= 2 else None
            continue
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            if match:
                line_num = int(match.group(1)) - 1
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            line_num += 1
            content = line[1:]
            if current_file and not _is_binary(current_file) and CHINESE_CHAR_PATTERN.search(content):
                findings.append((current_file, line_num, content))
        elif not line.startswith("-"):
            line_num += 1

    return findings


# ---------------------------------------------------------------------------
# Local mode: scan every tracked file in the repo
# ---------------------------------------------------------------------------


def _check_all_files() -> List[Tuple[str, int, str]]:
    print("[Local mode] Scanning all tracked files for Chinese characters ...")

    tracked = subprocess.check_output(["git", "ls-files"], text=True).strip().split("\n")

    findings: List[Tuple[str, int, str]] = []
    for filepath in tracked:
        if not filepath or _is_binary(filepath) or not os.path.isfile(filepath):
            continue
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as fh:
                for line_num, line in enumerate(fh, start=1):
                    if CHINESE_CHAR_PATTERN.search(line):
                        findings.append((filepath, line_num, line.rstrip("\n")))
        except (OSError, UnicodeDecodeError):
            continue

    return findings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _report(findings: List[Tuple[str, int, str]], is_ci: bool) -> None:
    if not findings:
        print("\nNo Chinese characters found.")
        return

    print(f"\nFound {len(findings)} line(s) containing Chinese characters:\n")
    for filepath, line_no, content in findings:
        stripped = content.strip()
        print(f"  {filepath}:{line_no}: {stripped}")
        if is_ci:
            print(f"::error file={filepath},line={line_no}::Chinese character detected: {stripped}")

    print(f"\n{len(findings)} occurrence(s) total. Please remove Chinese characters from your code.")


def main():
    base_ref = os.environ.get("BASE_REF")
    head_ref = os.environ.get("HEAD_REF")
    is_ci = bool(base_ref and head_ref)

    if is_ci:
        findings = _check_diff(base_ref, head_ref)
    else:
        findings = _check_all_files()

    _report(findings, is_ci)

    if findings:
        sys.exit(1)


if __name__ == "__main__":
    main()
