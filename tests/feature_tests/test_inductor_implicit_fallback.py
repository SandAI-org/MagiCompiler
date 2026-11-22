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

"""Inductor's implicit fallback for a custom op formats the op's IR arguments, eagerly, up to PT 2.12.

Inductor lowers a custom op it has no lowering for to a fallback it creates on meeting the op, after logging the op's
arguments at INFO, a string built whatever the log level. Each argument prints the extern kernels upstream of it, a
shared one once per path, so an op first met at the end of a chain of custom ops formats a string exponential in the
chain's depth: one graph holding a whole model of such ops never finished compiling. MagiCompiler registers the same
fallbacks before Inductor lowers a graph, so Inductor never formats that string.
"""

from unittest.mock import patch

import torch
import torch._inductor.config as inductor_config
from torch._inductor import exc
from torch._inductor.lowering import lowerings

from magi_compiler import magi_compile, magi_register_custom_op

DEPTH = 8


def _pair_meta(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(a), torch.empty_like(a)


@magi_register_custom_op("test_magi_fallback::pair", infer_output_meta_fn=_pair_meta)
def magi_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (a + b) * 0.5, (a - b) * 0.5


@magi_register_custom_op("test_magi_fallback::tail")
def magi_tail(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b


@torch.library.custom_op("test_magi_fallback::bump_", mutates_args=("x",))
def bump_(x: torch.Tensor) -> None:
    x.add_(1)


@bump_.register_fake
def _(x):
    return None


@torch.library.custom_op("test_magi_fallback::scale", mutates_args=())
def scale(x: torch.Tensor) -> torch.Tensor:
    return x * 2


@scale.register_fake
def _(x):
    return torch.empty_like(x)


class _Chain(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, y = x, x
        for _ in range(DEPTH):
            s, y = magi_pair(s, y)
        out = magi_tail(s, y)
        bump_(out)
        # Through its packet: the Dynamo graph holds an OpOverloadPacket, not an overload.
        return torch.ops.test_magi_fallback.scale(out)


def test_magi_compile_never_takes_the_implicit_fallback():
    def formatted(*_):
        raise AssertionError("Inductor took the implicit fallback path")

    torch._dynamo.reset()
    x = torch.randn(64, 8)
    with patch.object(exc.OperatorIssue, "operator_str", staticmethod(formatted)), inductor_config.patch(fx_graph_cache=False):
        out = magi_compile(_Chain(), dynamic_arg_dims={"x": 0})(x)
    torch.testing.assert_close(out, _Chain()(x))
    ops = torch.ops.test_magi_fallback
    for op in (ops.pair, ops.tail, ops.bump_, ops.scale):
        assert op.default in lowerings
