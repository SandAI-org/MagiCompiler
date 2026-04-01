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

"""Performance test: pointwise operator chain.

Covers all supported compilation paths (class, instance, instance+TC, method).

Measured baseline (H100):
  torch.compile  ~5.9x vs eager
  magi_compile   ~3.5x vs eager (all paths)

TODO(perf-fusion-gap): magi_compile still trails torch.compile in fusion-heavy workloads;
investigate graph partitioning/fusion opportunities and reduce the gap.
"""

import pytest
import torch
import torch.nn as nn

from magi_compiler import magi_compile
from magi_compiler.config import CompileMode
from tests.perf_tests import cuda_benchmark, print_perf_comparison

HIDDEN_SIZE = 4096
NUM_TOKENS = 16384
SPEEDUP_VS_EAGER_THRESHOLD = 3.15


class PointwiseFusionChain(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 0.5
        x = x + 1.0
        x = torch.relu(x)
        x = x * x
        x = x - 0.5
        x = torch.sigmoid(x)
        return x


# ── Shared baselines (computed once per module) ────────────────────────


@pytest.fixture(scope="module")
def pointwise_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def pointwise_input(pointwise_device):
    return torch.randn(NUM_TOKENS, HIDDEN_SIZE, device=pointwise_device, dtype=torch.float32)


@pytest.fixture(scope="module")
def pointwise_baselines(pointwise_device, pointwise_input):
    """Eager and torch.compile baselines, benchmarked once for the whole module."""
    x = pointwise_input
    eager_model = PointwiseFusionChain().to(pointwise_device).eval()
    torch_compiled = torch.compile(
        PointwiseFusionChain().to(pointwise_device).eval(), backend="inductor"
    )
    with torch.no_grad():
        eager_result = cuda_benchmark(lambda: eager_model(x))
        torch_result = cuda_benchmark(lambda: torch_compiled(x), compilation_warmup=3)
    return eager_result, torch_result


# ── Helpers ────────────────────────────────────────────────────────────


def _assert_speedup(magi_vs_eager, eager_result, magi_result, label):
    assert magi_vs_eager >= SPEEDUP_VS_EAGER_THRESHOLD, (
        f"[{label}] magi_compile must achieve >= {SPEEDUP_VS_EAGER_THRESHOLD:.2f}x over eager. "
        f"Got {magi_vs_eager:.2f}x "
        f"(eager={eager_result.median:.3f}ms, magi={magi_result.median:.3f}ms)"
    )


# ── Tests ──────────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_pointwise_class_decoration(pointwise_device, pointwise_input, pointwise_baselines):
    """Pointwise chain: @magi_compile class decoration."""
    eager_result, torch_result = pointwise_baselines

    @magi_compile(dynamic_arg_dims={"x": 0})
    class CompiledPointwise(PointwiseFusionChain):
        pass

    magi_compiled = CompiledPointwise().to(pointwise_device).eval()

    with torch.no_grad():
        magi_result = cuda_benchmark(lambda: magi_compiled(pointwise_input), compilation_warmup=3)

    magi_vs_eager, _ = print_perf_comparison(
        "Pointwise - class decoration",
        eager_result, magi_result, torch_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})",
    )
    _assert_speedup(magi_vs_eager, eager_result, magi_result, "class")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_pointwise_instance_decoration(pointwise_device, pointwise_input, pointwise_baselines):
    """Pointwise chain: magi_compile(instance) decoration."""
    eager_result, torch_result = pointwise_baselines

    magi_compiled = magi_compile(
        PointwiseFusionChain().to(pointwise_device), dynamic_arg_dims={"x": 0}
    )
    magi_compiled.eval()

    with torch.no_grad():
        magi_result = cuda_benchmark(lambda: magi_compiled(pointwise_input), compilation_warmup=3)

    magi_vs_eager, _ = print_perf_comparison(
        "Pointwise - instance decoration",
        eager_result, magi_result, torch_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})",
    )
    _assert_speedup(magi_vs_eager, eager_result, magi_result, "instance")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_pointwise_instance_torch_compile_mode(pointwise_device, pointwise_input, pointwise_baselines):
    """Pointwise chain: magi_compile(instance, mode=TORCH_COMPILE)."""
    eager_result, torch_result = pointwise_baselines

    def _tc_mode(cfg):
        cfg.compile_mode = CompileMode.TORCH_COMPILE
        return cfg

    magi_compiled = magi_compile(
        PointwiseFusionChain().to(pointwise_device),
        dynamic_arg_dims={"x": 0},
        config_patch=_tc_mode,
    )
    magi_compiled.eval()

    with torch.no_grad():
        magi_result = cuda_benchmark(lambda: magi_compiled(pointwise_input), compilation_warmup=3)

    magi_vs_eager, _ = print_perf_comparison(
        "Pointwise - instance (TORCH_COMPILE mode)",
        eager_result, magi_result, torch_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})",
    )
    _assert_speedup(magi_vs_eager, eager_result, magi_result, "instance_tc")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_pointwise_function_decoration(pointwise_device, pointwise_input, pointwise_baselines):
    """Pointwise chain: @magi_compile function-level entry."""
    eager_result, torch_result = pointwise_baselines

    model = PointwiseFusionChain().to(pointwise_device).eval()

    @magi_compile(dynamic_arg_dims={"x": 0})
    def compiled_entry(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    with torch.no_grad():
        magi_result = cuda_benchmark(lambda: compiled_entry(pointwise_input), compilation_warmup=3)

    magi_vs_eager, _ = print_perf_comparison(
        "Pointwise - function decoration",
        eager_result, magi_result, torch_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})",
    )
    _assert_speedup(magi_vs_eager, eager_result, magi_result, "function")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_pointwise_method_decoration(pointwise_device, pointwise_input, pointwise_baselines):
    """Pointwise chain: magi_compile(model.forward) method decoration."""
    eager_result, torch_result = pointwise_baselines

    magi_compiled = PointwiseFusionChain().to(pointwise_device).eval()
    magi_compiled.forward = magi_compile(
        magi_compiled.forward, dynamic_arg_dims={"x": 0}
    )

    with torch.no_grad():
        magi_result = cuda_benchmark(lambda: magi_compiled(pointwise_input), compilation_warmup=3)

    magi_vs_eager, _ = print_perf_comparison(
        "Pointwise - method decoration",
        eager_result, magi_result, torch_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})",
    )
    _assert_speedup(magi_vs_eager, eager_result, magi_result, "method")
