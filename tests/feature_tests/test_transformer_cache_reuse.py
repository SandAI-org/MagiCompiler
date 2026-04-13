# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("run_mode", ["jit", "aot"])
def test_transformer_cache_reuse_across_processes(tmp_path: Path, run_mode: str):
    helper_path = Path(__file__).parent / "cache_reuse_helper" / "transformer_cache_reuse_helper.py"
    cache_root = tmp_path / "cache_shared"

    baseline_out = tmp_path / f"baseline_{run_mode}.json"
    cache_out1 = tmp_path / f"cache1_{run_mode}.json"
    cache_out2 = tmp_path / f"cache2_{run_mode}.json"

    env = os.environ.copy()
    env["MAGI_LOGGING_LEVEL"] = "info"
    env["MAGI_COMPILE_AOT"] = "true" if run_mode == "aot" else "false"

    baseline_env = {**env, "MAGI_COMPILE_CACHE_ROOT_DIR": str(tmp_path / f"baseline_cache_{run_mode}")}
    baseline = subprocess.run(
        [sys.executable, str(helper_path), "--output", str(baseline_out), "--run-kind", "baseline"],
        capture_output=True,
        text=True,
        env=baseline_env,
    )
    assert baseline.returncode == 0, f"baseline process failed (mode={run_mode})\n{baseline.stderr}"

    cache_env = {**env, "MAGI_COMPILE_CACHE_ROOT_DIR": str(cache_root)}
    run1 = subprocess.run(
        [sys.executable, str(helper_path), "--output", str(cache_out1), "--run-kind", "cache"],
        capture_output=True,
        text=True,
        env=cache_env,
    )
    assert run1.returncode == 0, f"cache run1 failed (mode={run_mode})\n{run1.stderr}"

    run2 = subprocess.run(
        [sys.executable, str(helper_path), "--output", str(cache_out2), "--run-kind", "cache"],
        capture_output=True,
        text=True,
        env=cache_env,
    )
    assert run2.returncode == 0, f"cache run2 failed (mode={run_mode})\n{run2.stderr}"

    baseline_payload = json.loads(baseline_out.read_text())
    payload1 = json.loads(cache_out1.read_text())
    payload2 = json.loads(cache_out2.read_text())

    expected_shape = baseline_payload["shape"]
    assert payload1["shape"] == expected_shape
    assert payload2["shape"] == expected_shape

    assert abs(payload1["sum"] - payload2["sum"]) < 1e-2
    assert abs(payload1["mean"] - payload2["mean"]) < 1e-4
