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

"""Performance test: end-to-end MLP block.

Covers all supported compilation paths (class, instance, instance+TC, method).

Measured baseline (H100):
  torch.compile  ~1.8x vs eager
  magi_compile   ~1.8x vs eager (all paths)
"""

import pytest
import torch

from magi_compiler import magi_compile
from magi_compiler.config import CompileMode
from tests.model_definition import MLPConfig, RawMLP, RawNonModuleMLP
from tests.perf_tests import cuda_benchmark, print_perf_comparison
from tests.perf_tests.utils import assert_speedup

HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 8192
NUM_TOKENS = 8192
SPEEDUP_VS_EAGER_THRESHOLD = 1.5


def _build_config():
    return MLPConfig(hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE, params_dtype=torch.bfloat16)


# ── Shared baselines (computed once per module) ────────────────────────


@pytest.fixture(scope="module")
def mlp_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def mlp_input(mlp_device):
    return torch.randn(NUM_TOKENS, HIDDEN_SIZE, device=mlp_device, dtype=torch.bfloat16)


@pytest.fixture(scope="module")
def mlp_baselines(mlp_device, mlp_input):
    """Eager and torch.compile baselines, benchmarked once for the whole module."""
    config = _build_config()
    eager_model = RawMLP(config).to(mlp_device).eval()
    torch_compiled = torch.compile(RawMLP(config).to(mlp_device).eval(), backend="inductor")
    with torch.no_grad():
        eager_result = cuda_benchmark(lambda: eager_model(mlp_input))
        torch_result = cuda_benchmark(lambda: torch_compiled(mlp_input), compilation_warmup=3)
    return eager_result, torch_result


# ── Tests ──────────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_mlp_class_decoration(mlp_device, mlp_input, mlp_baselines):
    """MLP block: @magi_compile class decoration."""
    eager_result, torch_result = mlp_baselines
    config = _build_config()

    @magi_compile(dynamic_arg_dims={"x": 0})
    class CompiledMLP(RawMLP):
        pass

    magi_compiled = CompiledMLP(config).to(mlp_device).eval()

    with torch.no_grad():
        magi_result = cuda_benchmark(lambda: magi_compiled(mlp_input), compilation_warmup=3)

    magi_vs_eager, _ = print_perf_comparison(
        "MLP - class decoration",
        eager_result,
        magi_result,
        torch_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})  intermediate={INTERMEDIATE_SIZE}  dtype=bf16",
    )
    assert_speedup(magi_vs_eager, eager_result, magi_result, "class", SPEEDUP_VS_EAGER_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_mlp_instance_decoration(mlp_device, mlp_input, mlp_baselines):
    """MLP block: magi_compile(instance) decoration."""
    eager_result, torch_result = mlp_baselines
    config = _build_config()

    magi_compiled = magi_compile(RawMLP(config).to(mlp_device), dynamic_arg_dims={"x": 0})
    magi_compiled.eval()

    with torch.no_grad():
        magi_result = cuda_benchmark(lambda: magi_compiled(mlp_input), compilation_warmup=3)

    magi_vs_eager, _ = print_perf_comparison(
        "MLP - instance decoration",
        eager_result,
        magi_result,
        torch_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})  intermediate={INTERMEDIATE_SIZE}  dtype=bf16",
    )
    assert_speedup(magi_vs_eager, eager_result, magi_result, "instance", SPEEDUP_VS_EAGER_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_mlp_instance_torch_compile_mode(mlp_device, mlp_input, mlp_baselines):
    """MLP block: magi_compile(instance, mode=TORCH_COMPILE)."""
    eager_result, torch_result = mlp_baselines
    config = _build_config()

    def _tc_mode(cfg):
        cfg.compile_mode = CompileMode.TORCH_COMPILE
        return cfg

    magi_compiled = magi_compile(RawMLP(config).to(mlp_device), dynamic_arg_dims={"x": 0}, config_patch=_tc_mode)
    magi_compiled.eval()

    with torch.no_grad():
        magi_result = cuda_benchmark(lambda: magi_compiled(mlp_input), compilation_warmup=3)

    magi_vs_eager, _ = print_perf_comparison(
        "MLP - instance (TORCH_COMPILE mode)",
        eager_result,
        magi_result,
        torch_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})  intermediate={INTERMEDIATE_SIZE}  dtype=bf16",
    )
    assert_speedup(magi_vs_eager, eager_result, magi_result, "instance_tc", SPEEDUP_VS_EAGER_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_mlp_function_decoration(mlp_device, mlp_input, mlp_baselines):
    """MLP block: @magi_compile function-level entry."""
    eager_result, torch_result = mlp_baselines
    config = _build_config()

    model = RawMLP(config).to(mlp_device).eval()

    @magi_compile(dynamic_arg_dims={"x": 0})
    def compiled_entry(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    with torch.no_grad():
        magi_result = cuda_benchmark(lambda: compiled_entry(mlp_input), compilation_warmup=3)

    magi_vs_eager, _ = print_perf_comparison(
        "MLP - function decoration",
        eager_result,
        magi_result,
        torch_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})  intermediate={INTERMEDIATE_SIZE}  dtype=bf16",
    )
    assert_speedup(magi_vs_eager, eager_result, magi_result, "function", SPEEDUP_VS_EAGER_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_mlp_method_decoration(mlp_device, mlp_input, mlp_baselines):
    """MLP block: magi_compile(model.forward) method decoration."""
    eager_result, torch_result = mlp_baselines
    config = _build_config()

    magi_compiled = RawMLP(config).to(mlp_device).eval()
    magi_compiled.forward = magi_compile(magi_compiled.forward, dynamic_arg_dims={"x": 0})

    with torch.no_grad():
        magi_result = cuda_benchmark(lambda: magi_compiled(mlp_input), compilation_warmup=3)

    magi_vs_eager, _ = print_perf_comparison(
        "MLP - method decoration",
        eager_result,
        magi_result,
        torch_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})  intermediate={INTERMEDIATE_SIZE}  dtype=bf16",
    )
    assert_speedup(magi_vs_eager, eager_result, magi_result, "method", SPEEDUP_VS_EAGER_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_non_module_mlp_class_decoration_speedup(mlp_device, mlp_input):
    base = RawNonModuleMLP(HIDDEN_SIZE, INTERMEDIATE_SIZE, mlp_device)

    @magi_compile(dynamic_arg_dims={"inp": 0})
    class CompiledNonModuleMLP(RawNonModuleMLP):
        def __call__(self, inp: torch.Tensor) -> torch.Tensor:
            return super().__call__(inp)

    eager_obj = RawNonModuleMLP(HIDDEN_SIZE, INTERMEDIATE_SIZE, mlp_device)
    eager_obj.copy_from(base)

    compiled_obj = CompiledNonModuleMLP(HIDDEN_SIZE, INTERMEDIATE_SIZE, mlp_device)
    compiled_obj.copy_from(base)

    with torch.no_grad():
        eager_result = cuda_benchmark(lambda: eager_obj(mlp_input))
        compiled_result = cuda_benchmark(lambda: compiled_obj(mlp_input), compilation_warmup=3)

    speedup, _ = print_perf_comparison(
        "Non-module MLP - class decoration",
        eager_result,
        compiled_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})  intermediate={INTERMEDIATE_SIZE}  dtype=bf16",
    )
    assert_speedup(speedup, eager_result, compiled_result, "non_module_class", SPEEDUP_VS_EAGER_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_non_module_mlp_instance_decoration_speedup(mlp_device, mlp_input):
    base = RawNonModuleMLP(HIDDEN_SIZE, INTERMEDIATE_SIZE, mlp_device)

    eager_obj = RawNonModuleMLP(HIDDEN_SIZE, INTERMEDIATE_SIZE, mlp_device)
    eager_obj.copy_from(base)

    inst_obj = RawNonModuleMLP(HIDDEN_SIZE, INTERMEDIATE_SIZE, mlp_device)
    inst_obj.copy_from(base)
    compiled_obj = magi_compile(inst_obj, dynamic_arg_dims={"inp": 0})

    with torch.no_grad():
        eager_result = cuda_benchmark(lambda: eager_obj(mlp_input))
        compiled_result = cuda_benchmark(lambda: compiled_obj(mlp_input), compilation_warmup=3)

    speedup, _ = print_perf_comparison(
        "Non-module MLP - instance decoration",
        eager_result,
        compiled_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})  intermediate={INTERMEDIATE_SIZE}  dtype=bf16",
    )
    assert_speedup(speedup, eager_result, compiled_result, "non_module_instance", SPEEDUP_VS_EAGER_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_non_module_mlp_method_decoration_speedup(mlp_device, mlp_input):
    base = RawNonModuleMLP(HIDDEN_SIZE, INTERMEDIATE_SIZE, mlp_device)

    eager_obj = RawNonModuleMLP(HIDDEN_SIZE, INTERMEDIATE_SIZE, mlp_device)
    eager_obj.copy_from(base)

    mtd_obj = RawNonModuleMLP(HIDDEN_SIZE, INTERMEDIATE_SIZE, mlp_device)
    mtd_obj.copy_from(base)
    mtd_obj.step = magi_compile(mtd_obj.step, dynamic_arg_dims={"inp": 0})

    with torch.no_grad():
        eager_result = cuda_benchmark(lambda: eager_obj.step(mlp_input))
        compiled_result = cuda_benchmark(lambda: mtd_obj.step(mlp_input), compilation_warmup=3)

    speedup, _ = print_perf_comparison(
        "Non-module MLP - method decoration",
        eager_result,
        compiled_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})  intermediate={INTERMEDIATE_SIZE}  dtype=bf16",
    )
    assert_speedup(speedup, eager_result, compiled_result, "non_module_method", SPEEDUP_VS_EAGER_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_non_module_mlp_eager_vs_module_eager_speed(mlp_device, mlp_input):
    config = _build_config()
    module_eager = RawMLP(config).to(mlp_device).eval()

    non_module = RawNonModuleMLP(HIDDEN_SIZE, INTERMEDIATE_SIZE, mlp_device)
    with torch.no_grad():
        module_eager_result = cuda_benchmark(lambda: module_eager(mlp_input))
        non_module_eager_result = cuda_benchmark(lambda: non_module(mlp_input))

    print_perf_comparison(
        "MLP eager baseline check: module vs non-module",
        module_eager_result,
        non_module_eager_result,
        extra_info=f"shape=({NUM_TOKENS}, {HIDDEN_SIZE})  intermediate={INTERMEDIATE_SIZE}  dtype=bf16",
    )
