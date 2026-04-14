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

"""Performance test: norm + residual + activation fusion.

Covers all supported compilation paths (class, instance, instance+TC, method).

Measured baseline (H100):
  torch.compile  ~10.0x vs eager
  magi_compile   ~4.5x vs eager (all paths)

TODO(perf-fusion-gap): magi_compile still trails torch.compile in fusion-heavy workloads;
investigate graph partitioning/fusion opportunities and reduce the gap.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from magi_compiler import magi_compile
from magi_compiler.config import CompileMode
from tests.model_definition import RawNonModuleNormResidualActivation, RMSNorm
from tests.perf_tests import cuda_benchmark, print_perf_comparison
from tests.perf_tests.utils import assert_magi_vs_torch, assert_speedup

HIDDEN_SIZE = 4096
NUM_TOKENS = 16384
SPEEDUP_VS_EAGER_THRESHOLD = 3.7


class NormResidualActivation(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(x) + residual)


# ── Shared baselines (computed once per module) ────────────────────────


@pytest.fixture(scope="module")
def nra_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def nra_inputs(nra_device):
    x = torch.randn(NUM_TOKENS, HIDDEN_SIZE, device=nra_device, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    return x, residual


@pytest.fixture(scope="module")
def nra_baselines(nra_device, nra_inputs):
    """Eager and torch.compile baselines, benchmarked once for the whole module."""
    x, residual = nra_inputs
    eager_model = NormResidualActivation(HIDDEN_SIZE).to(nra_device).eval()
    torch_compiled = torch.compile(
        NormResidualActivation(HIDDEN_SIZE).to(nra_device).eval(), fullgraph=True, dynamic=True, backend="inductor"
    )
    with torch.no_grad():
        eager_result = cuda_benchmark(lambda: eager_model(x, residual))
        torch_result = cuda_benchmark(lambda: torch_compiled(x, residual), compilation_warmup=3)
    return eager_result, torch_result


# ── Tests ──────────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_norm_residual_class_decoration(nra_device, nra_inputs, nra_baselines):
    """Norm+residual+SiLU: @magi_compile class decoration."""
    eager_result, torch_result = nra_baselines
    x, residual = nra_inputs

    @magi_compile(dynamic_arg_dims={"x": 0, "residual": 0})
    class CompiledNRA(NormResidualActivation):
        pass

    magi_compiled = CompiledNRA(HIDDEN_SIZE).to(nra_device).eval()

    with torch.no_grad():
        magi_result = cuda_benchmark(lambda: magi_compiled(x, residual), compilation_warmup=3)

    magi_vs_eager, magi_vs_torch = print_perf_comparison(
        "Norm+Residual+SiLU - class decoration",
        eager_result,
        magi_result,
        torch_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})  dtype=bf16",
    )
    assert_speedup(magi_vs_eager, eager_result, magi_result, "class", SPEEDUP_VS_EAGER_THRESHOLD)
    assert_magi_vs_torch(magi_vs_torch, torch_result, magi_result, "class")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_norm_residual_instance_decoration(nra_device, nra_inputs, nra_baselines):
    """Norm+residual+SiLU: magi_compile(instance) decoration."""
    eager_result, torch_result = nra_baselines
    x, residual = nra_inputs

    magi_compiled = magi_compile(NormResidualActivation(HIDDEN_SIZE).to(nra_device), dynamic_arg_dims={"x": 0, "residual": 0})
    magi_compiled.eval()

    with torch.no_grad():
        magi_result = cuda_benchmark(lambda: magi_compiled(x, residual), compilation_warmup=3)

    magi_vs_eager, magi_vs_torch = print_perf_comparison(
        "Norm+Residual+SiLU - instance decoration",
        eager_result,
        magi_result,
        torch_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})  dtype=bf16",
    )
    assert_speedup(magi_vs_eager, eager_result, magi_result, "instance", SPEEDUP_VS_EAGER_THRESHOLD)
    assert_magi_vs_torch(magi_vs_torch, torch_result, magi_result, "instance")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_norm_residual_instance_torch_compile_mode(nra_device, nra_inputs, nra_baselines):
    """Norm+residual+SiLU: magi_compile(instance, mode=TORCH_COMPILE)."""
    eager_result, torch_result = nra_baselines
    x, residual = nra_inputs

    def _tc_mode(cfg):
        cfg.compile_mode = CompileMode.TORCH_COMPILE
        return cfg

    magi_compiled = magi_compile(
        NormResidualActivation(HIDDEN_SIZE).to(nra_device), dynamic_arg_dims={"x": 0, "residual": 0}, config_patch=_tc_mode
    )
    magi_compiled.eval()

    with torch.no_grad():
        magi_result = cuda_benchmark(lambda: magi_compiled(x, residual), compilation_warmup=3)

    magi_vs_eager, magi_vs_torch = print_perf_comparison(
        "Norm+Residual+SiLU - instance (TORCH_COMPILE mode)",
        eager_result,
        magi_result,
        torch_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})  dtype=bf16",
    )
    assert_speedup(magi_vs_eager, eager_result, magi_result, "instance_tc", SPEEDUP_VS_EAGER_THRESHOLD)
    assert_magi_vs_torch(magi_vs_torch, torch_result, magi_result, "instance_tc")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_norm_residual_function_decoration(nra_device, nra_inputs, nra_baselines):
    """Norm+residual+SiLU: @magi_compile function-level entry."""
    eager_result, torch_result = nra_baselines
    x, residual = nra_inputs

    model = NormResidualActivation(HIDDEN_SIZE).to(nra_device).eval()

    @magi_compile(dynamic_arg_dims={"x": 0, "residual": 0})
    def compiled_entry(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return model(x, residual)

    with torch.no_grad():
        magi_result = cuda_benchmark(lambda: compiled_entry(x, residual), compilation_warmup=3)

    magi_vs_eager, magi_vs_torch = print_perf_comparison(
        "Norm+Residual+SiLU - function decoration",
        eager_result,
        magi_result,
        torch_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})  dtype=bf16",
    )
    assert_speedup(magi_vs_eager, eager_result, magi_result, "function", SPEEDUP_VS_EAGER_THRESHOLD)
    assert_magi_vs_torch(magi_vs_torch, torch_result, magi_result, "function")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_norm_residual_method_decoration(nra_device, nra_inputs, nra_baselines):
    """Norm+residual+SiLU: magi_compile(model.forward) method decoration."""
    eager_result, torch_result = nra_baselines
    x, residual = nra_inputs

    magi_compiled = NormResidualActivation(HIDDEN_SIZE).to(nra_device).eval()
    magi_compiled.forward = magi_compile(magi_compiled.forward, dynamic_arg_dims={"x": 0, "residual": 0})

    with torch.no_grad():
        magi_result = cuda_benchmark(lambda: magi_compiled(x, residual), compilation_warmup=3)

    magi_vs_eager, magi_vs_torch = print_perf_comparison(
        "Norm+Residual+SiLU - method decoration",
        eager_result,
        magi_result,
        torch_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})  dtype=bf16",
    )
    assert_speedup(magi_vs_eager, eager_result, magi_result, "method", SPEEDUP_VS_EAGER_THRESHOLD)
    assert_magi_vs_torch(magi_vs_torch, torch_result, magi_result, "method")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_norm_residual_non_module_class_decoration_speedup(nra_device, nra_inputs):
    x, residual = nra_inputs

    @magi_compile(dynamic_arg_dims={"x": 0, "residual": 0})
    class CompiledNonModuleNRA(RawNonModuleNormResidualActivation):
        def __call__(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
            return super().__call__(x, residual)

    base = RawNonModuleNormResidualActivation(HIDDEN_SIZE)

    eager_obj = RawNonModuleNormResidualActivation(HIDDEN_SIZE)
    eager_obj.copy_from(base)

    compiled_obj = CompiledNonModuleNRA(HIDDEN_SIZE)
    compiled_obj.copy_from(base)

    with torch.no_grad():
        eager_result = cuda_benchmark(lambda: eager_obj(x, residual))
        compiled_result = cuda_benchmark(lambda: compiled_obj(x, residual), compilation_warmup=3)

    speedup, _ = print_perf_comparison(
        "Norm+Residual non-module - class decoration",
        eager_result,
        compiled_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})  dtype=bf16",
    )
    assert_speedup(speedup, eager_result, compiled_result, "norm_residual_non_module_class", SPEEDUP_VS_EAGER_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_norm_residual_non_module_instance_decoration_speedup(nra_device, nra_inputs):
    x, residual = nra_inputs

    base = RawNonModuleNormResidualActivation(HIDDEN_SIZE)

    eager_obj = RawNonModuleNormResidualActivation(HIDDEN_SIZE)
    eager_obj.copy_from(base)

    inst_obj = RawNonModuleNormResidualActivation(HIDDEN_SIZE)
    inst_obj.copy_from(base)
    compiled_obj = magi_compile(inst_obj, dynamic_arg_dims={"x": 0, "residual": 0})

    with torch.no_grad():
        eager_result = cuda_benchmark(lambda: eager_obj(x, residual))
        compiled_result = cuda_benchmark(lambda: compiled_obj(x, residual), compilation_warmup=3)

    speedup, _ = print_perf_comparison(
        "Norm+Residual non-module - instance decoration",
        eager_result,
        compiled_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})  dtype=bf16",
    )
    assert_speedup(speedup, eager_result, compiled_result, "norm_residual_non_module_instance", SPEEDUP_VS_EAGER_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_norm_residual_non_module_method_decoration_speedup(nra_device, nra_inputs):
    x, residual = nra_inputs

    base = RawNonModuleNormResidualActivation(HIDDEN_SIZE)

    eager_obj = RawNonModuleNormResidualActivation(HIDDEN_SIZE)
    eager_obj.copy_from(base)

    mtd_obj = RawNonModuleNormResidualActivation(HIDDEN_SIZE)
    mtd_obj.copy_from(base)
    mtd_obj.step = magi_compile(mtd_obj.step, dynamic_arg_dims={"x": 0, "residual": 0})

    with torch.no_grad():
        eager_result = cuda_benchmark(lambda: eager_obj.step(x, residual))
        compiled_result = cuda_benchmark(lambda: mtd_obj.step(x, residual), compilation_warmup=3)

    speedup, _ = print_perf_comparison(
        "Norm+Residual non-module - method decoration",
        eager_result,
        compiled_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})  dtype=bf16",
    )
    assert_speedup(speedup, eager_result, compiled_result, "norm_residual_non_module_method", SPEEDUP_VS_EAGER_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_norm_residual_non_module_eager_vs_module_eager_speed(nra_device, nra_inputs):
    x, residual = nra_inputs

    module_eager = NormResidualActivation(HIDDEN_SIZE).to(nra_device).eval()
    non_module_eager = RawNonModuleNormResidualActivation(HIDDEN_SIZE)

    with torch.no_grad():
        module_eager_result = cuda_benchmark(lambda: module_eager(x, residual))
        non_module_eager_result = cuda_benchmark(lambda: non_module_eager(x, residual))

    print_perf_comparison(
        "Norm+Residual eager baseline check: module vs non-module",
        module_eager_result,
        non_module_eager_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})  dtype=bf16",
    )
