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

from typing import Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from magi_compiler.api import magi_compile
from magi_compiler.config import get_compile_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------


def high_precision_silu(x, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    return F.silu(x).to(out_dtype)


def high_precision_sigmoid(x, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    return F.sigmoid(x).to(out_dtype)


def high_precision_gelu(x, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    return F.gelu(x).to(out_dtype)


def swiglu7(x, alpha: float = 1.702, limit: float = 7.0, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return (out_glu * (x_linear + 1)).to(out_dtype)


def gelu7(x, alpha: float = 1.702, limit: float = 7.0, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    x_glu = x.clamp(min=None, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu.to(out_dtype)


def relu_square(x, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    return torch.square(F.relu(x)).to(out_dtype)


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------


class SiluModel(nn.Module):
    def forward(self, a, b):
        return high_precision_silu(torch.mm(a, b), out_dtype=torch.bfloat16)


class SigmoidModel(nn.Module):
    def forward(self, a, b):
        return high_precision_sigmoid(torch.mm(a, b), out_dtype=torch.bfloat16)


class GeluModel(nn.Module):
    def forward(self, a, b):
        return high_precision_gelu(torch.mm(a, b), out_dtype=torch.bfloat16)


class Swiglu7Model(nn.Module):
    def forward(self, a, b):
        return swiglu7(torch.mm(a, b), out_dtype=torch.bfloat16)


class Gelu7Model(nn.Module):
    def forward(self, a, b):
        return gelu7(torch.mm(a, b), out_dtype=torch.bfloat16)


class ReluSquareModel(nn.Module):
    def forward(self, a, b):
        return relu_square(torch.mm(a, b), out_dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_fusion_test(model: nn.Module, a: torch.Tensor, b: torch.Tensor, atol: float = 0.5, rtol: float = 0.0):
    """Run a matmul-epilogue fusion test.

    Checks that the fused result satisfies:  |actual - expected| < atol + rtol * |expected|

    atol=0.5 covers the bf16 → fp32 accumulation difference for element-wise
    activations whose output magnitude is O(1).  For activations that amplify
    magnitude (e.g. relu_square), pass a non-zero rtol instead.
    """
    model = model.cuda().bfloat16()
    with torch.no_grad():
        expected = model(a, b)

    get_compile_config().disable_cache = True
    compiled_model = magi_compile(model, dynamic_arg_dims={"a": 0})
    with torch.no_grad():
        actual = compiled_model(a, b)

    abs_diff = (actual - expected).abs()
    tol = atol + rtol * expected.abs()
    max_violation = (abs_diff - tol).max().item()
    assert max_violation <= 0, (
        f"Fused result too far from reference: "
        f"max(|diff| - tol) = {max_violation:.4f}, "
        f"max |diff| = {abs_diff.max().item():.4f}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_matmul_epilogue_fusion_silu():
    M, K, N = 128, 256, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    _run_fusion_test(SiluModel(), a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_matmul_epilogue_fusion_sigmoid():
    M, K, N = 128, 256, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    _run_fusion_test(SigmoidModel(), a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_matmul_epilogue_fusion_gelu():
    M, K, N = 128, 256, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    _run_fusion_test(GeluModel(), a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_matmul_epilogue_fusion_swiglu7():
    M, K, N = 128, 256, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    _run_fusion_test(Swiglu7Model(), a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_matmul_epilogue_fusion_gelu7():
    M, K, N = 128, 256, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    _run_fusion_test(Gelu7Model(), a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_matmul_epilogue_fusion_relu_square():
    M, K, N = 128, 256, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    # relu_square amplifies values quadratically (output ~ x^2, up to ~256),
    # so use relative tolerance instead of a fixed absolute bound.
    _run_fusion_test(ReluSquareModel(), a, b, atol=0.0, rtol=0.2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
