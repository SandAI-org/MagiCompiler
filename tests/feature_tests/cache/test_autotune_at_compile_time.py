# Copyright (c) 2025 SandAI. All Rights Reserved.
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

"""Verify runtime-autotune configuration and cross-process persistence of autotune selections."""

import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from magi_compiler.config import CompileConfig

# ──────────────────────────────────────────────────────────────────────
# Part 1: piecewise_compiler must override autotune_at_compile_time=False
# ──────────────────────────────────────────────────────────────────────


class TestAutotuneOverriddenToFalse:
    """Verify config_patches contain autotune_at_compile_time=False."""

    @staticmethod
    def _capture_config_patches(runtime_shape=None):
        import torch.fx as fx

        from magi_compiler.magi_backend.piecewise_compiler import InductorStandaloneAdaptor

        captured = {}

        def fake_standalone_compile(graph, example_inputs, *, dynamic_shapes, options):
            captured.update(options.get("config_patches", {}))
            mock_artifact = MagicMock()
            mock_artifact.compiled_fn = lambda *a: None
            mock_artifact.artifacts = None
            return mock_artifact

        adaptor = InductorStandaloneAdaptor(CompileConfig())
        adaptor.initialize_cache(Path(CompileConfig().cache_root_dir) / "test")

        graph = fx.Graph()
        graph.output(None)
        gm = fx.GraphModule({}, graph)

        with patch("torch._inductor.standalone_compile", fake_standalone_compile):
            try:
                adaptor.compile(gm, [], {}, runtime_shape=runtime_shape, key="test")
            except Exception:
                pass

        return captured

    def test_dynamic_shape_autotune_is_false(self):
        config = self._capture_config_patches(runtime_shape=None)
        assert config.get("triton.autotune_at_compile_time") is False

    def test_static_shape_autotune_is_false(self):
        config = self._capture_config_patches(runtime_shape=128)
        assert config.get("triton.autotune_at_compile_time") is False


# ──────────────────────────────────────────────────────────────────────
# Part 2: _compilation_context must set TRITON_CACHE_AUTOTUNING=1
# ──────────────────────────────────────────────────────────────────────


class TestTritonCacheAutotuningEnvVar:
    """Verify _compilation_context sets TRITON_CACHE_AUTOTUNING=1."""

    @staticmethod
    def _make_state():
        from magi_compiler.magi_backend.magi_compiler_base import MagiCompileState

        def _dummy_fn(x):
            return x

        return MagiCompileState(obj=_dummy_fn, compile_config=CompileConfig(), model_tag="test", dynamic_arg_dims={})

    def test_compile_context_enables_triton_cache_autotuning(self):
        from magi_compiler._api import _compilation_context

        state = self._make_state()
        with _compilation_context(state):
            assert os.environ.get("TRITON_CACHE_AUTOTUNING") == "1", (
                "TRITON_CACHE_AUTOTUNING must be '1' during compilation " "to persist autotune results for cold start reuse"
            )

    def test_compile_context_sets_triton_cache_dir(self):
        from magi_compiler._api import _compilation_context

        state = self._make_state()
        with _compilation_context(state):
            cache_dir = os.environ.get("TRITON_CACHE_DIR", "")
            assert cache_dir, "TRITON_CACHE_DIR must be set"
            assert "triton_cache" in cache_dir, "TRITON_CACHE_DIR should point to persistent triton_cache dir"


# ──────────────────────────────────────────────────────────────────────
# Part 3: Reproduce the bug + verify the fix (2-process simulation)
# ──────────────────────────────────────────────────────────────────────

_WORKER_SCRIPT = Path(__file__).parent / "cache_reuse_helper" / "autotune_cache_worker.py"


def _run_worker(cache_dir: str, enable_cache: bool) -> float:
    """Run the worker in a subprocess, return elapsed time for the kernel call."""
    env = os.environ.copy()
    env["TRITON_CACHE_DIR"] = cache_dir
    if enable_cache:
        env["TRITON_CACHE_AUTOTUNING"] = "1"
    else:
        env.pop("TRITON_CACHE_AUTOTUNING", None)

    result = subprocess.run([sys.executable, str(_WORKER_SCRIPT)], env=env, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Worker failed (rc={result.returncode}):\n{result.stderr[-500:]}")
    for line in result.stdout.strip().split("\n"):
        if line.startswith("ELAPSED="):
            return float(line.split("=")[1])
    raise RuntimeError(f"Worker did not print ELAPSED line: {result.stdout}")


@pytest.mark.skipif(not os.path.exists("/dev/nvidia0"), reason="GPU required")
class TestAutotuneCachePersistence:
    """Autotune selections persist across worker processes when disk caching is enabled."""

    def test_bug_reproduced_no_cache_without_env(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            _run_worker(cache_dir, enable_cache=False)

            jsons = glob.glob(os.path.join(cache_dir, "**/*.autotune.json"), recursive=True)
            assert len(jsons) == 0, (
                f"Bug not reproduced: expected 0 .autotune.json files "
                f"without TRITON_CACHE_AUTOTUNING=1, but found {len(jsons)}: {jsons}"
            )

    def test_bug_fixed_cache_persisted_with_env(self):
        """Compare autotune cache hits and misses using the same warmed compilation cache."""
        with tempfile.TemporaryDirectory() as cache_dir:
            # Process 1: warm both compilation and autotune caches
            _run_worker(cache_dir, enable_cache=True)

            # ── Primary evidence: .autotune.json must exist ──
            jsons = glob.glob(os.path.join(cache_dir, "**/*.autotune.json"), recursive=True)
            assert len(jsons) >= 1, (
                f"Fix not working: expected ≥1 .autotune.json file " f"with TRITON_CACHE_AUTOTUNING=1, but found {len(jsons)}"
            )

            # ── Supplementary evidence: timing comparison ──
            # Process 2: compilation cache warm + autotune cache HIT
            t_cached = _run_worker(cache_dir, enable_cache=True)

            # Process 3: compilation cache warm + autotune cache MISS
            t_uncached = _run_worker(cache_dir, enable_cache=False)

            assert t_uncached > t_cached, (
                f"Autotune cache miss should be slower than cache hit "
                f"(same compilation cache): cached={t_cached:.3f}s, "
                f"uncached={t_uncached:.3f}s, ratio={t_uncached / t_cached:.1f}x"
            )
