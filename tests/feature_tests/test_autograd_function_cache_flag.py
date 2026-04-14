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

"""Tests verifying that InductorStandaloneAdaptor.compile() correctly patches
autograd_cache_allow_custom_autograd_functions=True around standalone_compile().

Two-process integration test (mirrors test_transformer_cache_reuse.py):

  run 1 (warm)  – compiles a training model containing autograd_function_apply,
                  saves the artifact to a shared cache directory, and verifies
                  that autograd_cache_allow_custom_autograd_functions was True
                  inside every standalone_compile() call.
  run 2 (cache) – starts fresh, loads the artifact from disk, and verifies
                  that no recompilation occurred.

Assertions:
  - "Failed to save compiled artifact" must NOT appear in run 1 stderr.
  - flag_during_compile entries are all True on run 1.
  - num_compiled_artifacts_saved > 0 on run 1.
  - num_inductor_compiles == 0 on run 2.
  - loss values are numerically consistent between runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_autograd_cache_flag_and_cache_reuse(tmp_path: Path):
    """Training model with autograd.Function: artifact saved on run 1, loaded
    on run 2, and autograd_cache_allow_custom_autograd_functions is True during
    every standalone_compile() call."""

    helper_path = Path(__file__).parent / "cache_reuse_helper" / "autograd_cache_flag_helper.py"
    cache_root = tmp_path / "cache"
    out1 = tmp_path / "run1.json"
    out2 = tmp_path / "run2.json"

    env = os.environ.copy()
    env["MAGI_LOGGING_LEVEL"] = "info"
    env["MAGI_COMPILE_CACHE_ROOT_DIR"] = str(cache_root)

    def _run(output: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(helper_path), "--output", str(output)], env=env, capture_output=True, text=True
        )

    # ── Run 1: warm cache ────────────────────────────────────────────────────
    p1 = _run(out1)
    assert p1.returncode == 0, f"run 1 failed\nstdout:\n{p1.stdout}\nstderr:\n{p1.stderr}"

    # The fix must prevent "Failed to save compiled artifact" from appearing.
    assert "Failed to save compiled artifact" not in p1.stderr, (
        "CompiledArtifact.save() still failing — autograd_function_apply bypass not fixed.\n" f"stderr:\n{p1.stderr}"
    )

    payload1 = json.loads(out1.read_text())

    # Flag check: every standalone_compile() call must see the flag as True.
    assert payload1["num_standalone_compile_calls"] > 0, "Spy was never called — standalone_compile was not intercepted."
    assert payload1["all_flags_true"], (
        "autograd_cache_allow_custom_autograd_functions was NOT True during "
        f"standalone_compile(); observed per-call values: {payload1['flag_during_compile']}"
    )

    # Artifact must have been saved.
    assert payload1["num_compiled_artifacts_saved"] > 0, (
        "Expected at least one artifact to be saved on the warm run, "
        f"got num_compiled_artifacts_saved={payload1['num_compiled_artifacts_saved']}"
    )

    # ── Run 2: cache hit ─────────────────────────────────────────────────────
    p2 = _run(out2)
    assert p2.returncode == 0, f"run 2 failed\nstdout:\n{p2.stdout}\nstderr:\n{p2.stderr}"

    payload2 = json.loads(out2.read_text())

    # No recompilation: PiecewiseCompiler.load() returns early before compile().
    assert payload2["num_inductor_compiles"] == 0, (
        "Expected 0 inductor compiles on the cache-hit run — artifact was not loaded.\n"
        f"num_inductor_compiles={payload2['num_inductor_compiles']}\n"
        f"stderr:\n{p2.stderr}"
    )

    # Numerical consistency.
    assert (
        abs(payload1["loss"] - payload2["loss"]) < 1e-2
    ), f"Loss mismatch between runs: run1={payload1['loss']}, run2={payload2['loss']}"
