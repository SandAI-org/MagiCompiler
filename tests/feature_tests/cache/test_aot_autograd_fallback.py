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

"""Tests for the eager backward lowering fix (backport of PyTorch PR #185635).

Root cause (PyTorch Issue #152022):
standalone_compile() calls save_cache_artifacts() immediately after compile_fx(),
but for training graphs AOTAutograd lazily defers backward lowering to the first
.backward() call.  This means aot_autograd_artifacts is empty when save() runs,
causing: ``AssertionError: CacheInfo(..., aot_autograd_artifacts=[], ...)``.

Our fix: ``_force_eager_backward_lowering()`` context manager in
``piecewise_compiler.py`` monkey-patches ``AOTConfig.__post_init__`` to set
``force_non_lazy_backward_lowering=True``.

Tests
-----
test_context_manager_patches_and_restores
    Unit test: verifies the context manager patches __post_init__ correctly
    and restores it on exit.

test_eager_backward_artifact_saved_and_reused
    Integration test (subprocess, requires CUDA): a real training model is
    compiled through magi_compile with a spy on standalone_compile that
    records force_non_lazy_backward_lowering at each call.  Verifies:
    - the flag is True during every standalone_compile call
    - artifacts are saved without "Failed to save" errors
    - a second run hits cache (0 recompilations)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from magi_compiler.magi_backend.piecewise_compiler import _force_eager_backward_lowering


class TestContextManagerUnit:
    """_force_eager_backward_lowering patches AOTConfig.__post_init__ correctly."""

    def test_patches_and_restores(self):
        from torch._functorch._aot_autograd.schemas import AOTConfig

        orig = AOTConfig.__post_init__

        # Inside: patched
        with _force_eager_backward_lowering():
            assert AOTConfig.__post_init__ is not orig
            config = AOTConfig(
                fw_compiler=lambda *a, **k: None,
                bw_compiler=lambda *a, **k: None,
                partition_fn=lambda *a, **k: None,
                decompositions={},
                num_params_buffers=0,
                aot_id=0,
                keep_inference_input_mutations=False,
            )
            assert config.force_non_lazy_backward_lowering is True

        # Outside: restored
        assert AOTConfig.__post_init__ is orig
        config2 = AOTConfig(
            fw_compiler=lambda *a, **k: None,
            bw_compiler=lambda *a, **k: None,
            partition_fn=lambda *a, **k: None,
            decompositions={},
            num_params_buffers=0,
            aot_id=0,
            keep_inference_input_mutations=False,
        )
        assert config2.force_non_lazy_backward_lowering is False

    def test_default_aotconfig_is_lazy(self):
        """Without the fix, AOTConfig defaults to lazy backward (the bug condition)."""
        from torch._functorch._aot_autograd.schemas import AOTConfig

        config = AOTConfig(
            fw_compiler=lambda *a, **k: None,
            bw_compiler=lambda *a, **k: None,
            partition_fn=lambda *a, **k: None,
            decompositions={},
            num_params_buffers=0,
            aot_id=0,
            keep_inference_input_mutations=False,
        )
        assert config.force_non_lazy_backward_lowering is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_eager_backward_artifact_saved_and_reused(tmp_path: Path):
    """Real training model: backward flag is True → artifact saved → cache reused.

    Two-process integration test (same pattern as test_autograd_function_cache_flag):
      run 1 (warm)  — compile + save, verify flag and artifact count
      run 2 (cache) — load from cache, verify no recompilation
    """
    helper_path = Path(__file__).parent / "cache_reuse_helper" / "eager_backward_helper.py"
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

    # ── Run 1: warm cache ────────────────────────────────────────────────
    p1 = _run(out1)
    assert p1.returncode == 0, f"run 1 failed\nstdout:\n{p1.stdout}\nstderr:\n{p1.stderr}"

    assert "Failed to save compiled artifact" not in p1.stderr, (
        "Artifact save still failing — eager backward lowering not effective.\n" f"stderr:\n{p1.stderr}"
    )

    r1 = json.loads(out1.read_text())

    assert r1["num_standalone_compile_calls"] > 0, "Spy never called — standalone_compile not intercepted"
    assert r1["all_flags_true"], (
        f"force_non_lazy_backward_lowering was NOT True during standalone_compile; "
        f"per-call values: {r1['backward_flag_during_compile']}"
    )
    assert r1["num_compiled_artifacts_saved"] > 0, f"No artifacts saved on warm run (got {r1['num_compiled_artifacts_saved']})"

    # ── Run 2: cache hit ─────────────────────────────────────────────────
    p2 = _run(out2)
    assert p2.returncode == 0, f"run 2 failed\nstdout:\n{p2.stdout}\nstderr:\n{p2.stderr}"

    r2 = json.loads(out2.read_text())

    assert r2["num_inductor_compiles"] == 0, (
        f"Expected 0 recompiles on cache-hit run, got {r2['num_inductor_compiles']}\n" f"stderr:\n{p2.stderr}"
    )

    assert abs(r1["loss"] - r2["loss"]) < 1e-2, f"Loss mismatch: run1={r1['loss']}, run2={r2['loss']}"
