# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_assert_cache_hit_succeeds_with_restart_analysis(tmp_path: Path):
    """Regression test: assert_cache_hit must not false-alarm on restart_analysis_count.

    When a subgraph triggers RestartAnalysis during bake, the cache records
    restart_analysis_count > 0.  In normal (non-verify) mode the loader
    replays those skips before returning the real artifact.  But under
    assert_cache_hit=True (verify mode), that replay returns None which the
    caller interprets as a cache miss → RuntimeError.

    This test:
      process-1  — compile (bake): warms the cache, encounters RestartAnalysis
      process-2  — load with MAGI_COMPILE_ASSERT_CACHE_HIT=1 (verify):
                    must succeed after the fix (skip restart replay in verify mode)
    """
    helper_path = Path(__file__).parent / "cache_reuse_helper" / "restart_analysis_cache_helper.py"
    cache_root = tmp_path / "cache"
    out_bake = tmp_path / "bake.json"
    out_verify = tmp_path / "verify.json"

    env = os.environ.copy()
    env["MAGI_LOGGING_LEVEL"] = "info"
    env["MAGI_COMPILE_CACHE_ROOT_DIR"] = str(cache_root)

    # ── Process 1: bake (warm cache) ──────────────────────────────────
    cmd_bake = [sys.executable, str(helper_path), "--output", str(out_bake)]
    p_bake = subprocess.run(cmd_bake, env=env, capture_output=True, text=True)
    assert p_bake.returncode == 0, f"bake process failed\nstdout:\n{p_bake.stdout}\nstderr:\n{p_bake.stderr}"
    assert (
        "standalone_compile raised RestartAnalysis" in p_bake.stderr
    ), "bake process did not encounter RestartAnalysis — test precondition violated"

    # Verify at least one cache handle has restart_analysis_count > 0
    cache_files = list(cache_root.rglob("subgraph_indices.py"))
    assert cache_files, "no cache file generated during bake"
    any_marked = False
    for cache_file in cache_files:
        raw = ast.literal_eval(cache_file.read_text())
        for _, handle in raw.items():
            if len(handle) >= 3 and int(handle[2]) > 0:
                any_marked = True
                break
        if any_marked:
            break
    assert any_marked, "expected at least one cache handle with restart_analysis_count > 0"

    # ── Process 2: verify (assert_cache_hit) ──────────────────────────
    verify_env = {**env, "MAGI_COMPILE_ASSERT_CACHE_HIT": "1"}
    cmd_verify = [sys.executable, str(helper_path), "--output", str(out_verify)]
    p_verify = subprocess.run(cmd_verify, env=verify_env, capture_output=True, text=True)

    assert p_verify.returncode == 0, (
        f"verify (assert_cache_hit) failed with restart_analysis_count > 0.\n"
        f"This is the bug: restart-replay returns None → false cache miss.\n"
        f"stderr:\n{p_verify.stderr}"
    )

    # Sanity-check: outputs should be numerically close
    payload_bake = json.loads(out_bake.read_text())
    payload_verify = json.loads(out_verify.read_text())
    assert payload_verify["shape"] == payload_bake["shape"]
    assert abs(payload_bake["sum"] - payload_verify["sum"]) < 1e-2
