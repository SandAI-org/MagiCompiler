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

"""Inductor's graph partition walks free symbols through chains of extern kernels, unmemoized before PT 2.10.

An extern kernel (a custom op, a GEMM) walks each of its inputs, and a MultiOutput forwards the walk to the kernel
that produced it, so the walk of a chain of multi-output custom ops makes calls exponential in the chain's depth.
The partition only serves Inductor's own CUDA graphs, which MagiCompiler never enables, so piecewise compiles turn it
off: one graph holding a whole model of such ops never finished compiling.
"""

from unittest.mock import patch

import pytest
import torch
import torch._inductor.config as inductor_config
from torch._inductor import ir

from magi_compiler import magi_compile, magi_register_custom_op
from magi_compiler.utils.envs import IS_PT_212

DEPTH = 12


def _pair_meta(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(a), torch.empty_like(a)


@magi_register_custom_op("test_graph_partition::pair", infer_output_meta_fn=_pair_meta)
def pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (a + b) * 0.5, (a - b) * 0.5


class _Chain(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, y = x, x
        for _ in range(DEPTH):
            s, y = pair(s, y)
        return s + y


def _free_symbol_walks(compile_model) -> int:
    calls = 0
    walk = ir.InputsKernel.get_free_symbol_uses

    def counted(self, unbacked_only=False):
        nonlocal calls
        calls += 1
        return walk(self, unbacked_only)

    torch._dynamo.reset()
    x = torch.randn(64, 8)
    with patch.object(ir.InputsKernel, "get_free_symbol_uses", counted), inductor_config.patch(fx_graph_cache=False):
        out = compile_model(_Chain())(x)
    torch.testing.assert_close(out, _Chain()(x))
    return calls


@pytest.mark.skipif(IS_PT_212, reason="PT 2.10+ memoizes InputsKernel.get_free_symbol_uses")
def test_graph_partition_walk_is_exponential_in_chain_depth():
    with inductor_config.patch(graph_partition=True):
        calls = _free_symbol_walks(lambda m: torch.compile(m, backend="inductor", dynamic=True, fullgraph=True))
    assert calls > 2**DEPTH


def test_magi_compile_skips_the_graph_partition_walk():
    calls = _free_symbol_walks(lambda m: magi_compile(m, dynamic_arg_dims={"x": 0}))
    assert calls < 2**DEPTH
