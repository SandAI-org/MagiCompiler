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

Test structure (bug-reproduce-test-first)
-----------------------------------------
Unit (no GPU, fast):
    test_bug_reproduced   — without CM, AOTConfig defaults to lazy backward (the bug)
    test_bug_fixed        — with CM, AOTConfig is forced to eager (the fix)

Integration (GPU, subprocess):
    test_bug_training_cache_save_fails_without_fix
        — training model + fix disabled → flag is False, cache save fails
    test_fix_inference_cache_saved_and_reused
        — inference model (real scenario) + fix enabled → cache saved and reused
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
from magi_compiler.utils.envs import IS_PT_212

HELPER = Path(__file__).parent / "cache_reuse_helper" / "eager_backward_helper.py"


class TestContextManagerUnit:
    """Fast unit tests (no GPU): verify _force_eager_backward_lowering patches AOTConfig."""

    def test_bug_reproduced(self):
        """Without fix: AOTConfig defaults to lazy backward lowering (the bug condition).

        This is the root cause of PyTorch Issue #152022: save_cache_artifacts()
        finds empty aot_autograd_artifacts because backward hasn't been lowered yet.
        """
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

    def test_bug_fixed(self):
        """With fix: _force_eager_backward_lowering forces flag to True and restores on exit."""
        from torch._functorch._aot_autograd.schemas import AOTConfig

        orig = AOTConfig.__post_init__

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


def _make_env(cache_root: Path) -> dict:
    env = os.environ.copy()
    env["MAGI_LOGGING_LEVEL"] = "info"
    env["MAGI_COMPILE_CACHE_ROOT_DIR"] = str(cache_root)
    return env


def _run_helper(output: Path, env: dict, *, mode: str = "infer", disable_fix: bool = False) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(HELPER), "--output", str(output), "--mode", mode]
    if disable_fix:
        cmd.append("--disable-fix")
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(IS_PT_212, reason="PT 2.12 includes upstream fix; bug cannot be reproduced")
def test_bug_training_cache_save_fails_without_fix(tmp_path: Path):
    """Bug reproduction (integration): training model WITHOUT fix.

    Disables _force_eager_backward_lowering via --disable-fix, then compiles
    a training model through magi_compile. This reproduces the exact condition
    from PyTorch Issue #152022:
    - AOTConfig.force_non_lazy_backward_lowering remains False
    - save_cache_artifacts() fails because backward hasn't been lowered

    The subprocess may crash (AssertionError inside standalone_compile) or
    succeed with 0 artifacts saved — both confirm the bug.
    """
    env = _make_env(tmp_path / "cache")
    out = tmp_path / "result.json"

    p = _run_helper(out, env, mode="train", disable_fix=True)

    if p.returncode != 0:
        assert (
            "AssertionError" in p.stderr or "aot_autograd" in p.stderr.lower()
        ), f"Subprocess crashed but not with expected assertion.\nstderr:\n{p.stderr}"
        return

    result = json.loads(out.read_text())
    assert not result["all_flags_true"], (
        f"Without fix, force_non_lazy_backward_lowering should be False; "
        f"got per-call values: {result['backward_flag_during_compile']}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(IS_PT_212, reason="PT 2.12 includes upstream fix; cache ABI differs")
def test_fix_inference_cache_saved_and_reused(tmp_path: Path):
    """Fix verification (integration): inference model WITH fix → cache saved and reused.

    Uses inference mode (forward only under torch.no_grad) — the real deployment
    scenario. Two subprocess runs with shared cache directory:
      run 1 (warm)  — compile + save, verify fix is active and artifacts saved
      run 2 (cache) — load from cache, verify no recompilation
    """
    cache_root = tmp_path / "cache"
    env = _make_env(cache_root)
    out1 = tmp_path / "run1.json"
    out2 = tmp_path / "run2.json"

    # ── Run 1: warm cache ────────────────────────────────────────────────
    p1 = _run_helper(out1, env, mode="infer")
    assert p1.returncode == 0, f"run 1 failed\nstdout:\n{p1.stdout}\nstderr:\n{p1.stderr}"
    assert "Failed to save compiled artifact" not in p1.stderr, f"Artifact save still failing.\nstderr:\n{p1.stderr}"

    r1 = json.loads(out1.read_text())
    assert r1["num_standalone_compile_calls"] > 0, "Spy never called"
    assert r1["all_flags_true"], (
        f"force_non_lazy_backward_lowering was NOT True; " f"per-call values: {r1['backward_flag_during_compile']}"
    )
    assert r1["num_compiled_artifacts_saved"] > 0, f"No artifacts saved (got {r1['num_compiled_artifacts_saved']})"

    # ── Run 2: cache hit ─────────────────────────────────────────────────
    p2 = _run_helper(out2, env, mode="infer")
    assert p2.returncode == 0, f"run 2 failed\nstdout:\n{p2.stdout}\nstderr:\n{p2.stderr}"

    r2 = json.loads(out2.read_text())
    assert r2["num_inductor_compiles"] == 0, f"Expected 0 recompiles, got {r2['num_inductor_compiles']}\nstderr:\n{p2.stderr}"
    assert (
        abs(r1["output_value"] - r2["output_value"]) < 1e-2
    ), f"Output mismatch: run1={r1['output_value']}, run2={r2['output_value']}"
