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
fallbacks before Inductor lowers a graph.
"""

from unittest.mock import patch

import pytest
import torch
import torch._inductor.config as inductor_config
from torch._inductor import exc

from magi_compiler import magi_compile, magi_register_custom_op

DEPTH = 8


def _pair_meta(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(a), torch.empty_like(a)


def _chain(pair, tail):
    def run(x: torch.Tensor, depth: int) -> torch.Tensor:
        s, y = x, x
        for _ in range(depth):
            s, y = pair(s, y)
        return tail(s, y)

    return run


@magi_register_custom_op("test_implicit_fallback::pair", infer_output_meta_fn=_pair_meta)
def pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (a + b) * 0.5, (a - b) * 0.5


@magi_register_custom_op("test_implicit_fallback::tail")
def tail(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b


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


def test_implicit_fallback_formats_a_string_exponential_in_chain_depth():
    formatted: list[int] = []
    operator_str = exc.OperatorIssue.operator_str

    def measured(target, args, kwargs):
        text = operator_str(target, args, kwargs)
        if target == torch.ops.test_implicit_fallback.tail.default:
            formatted.append(len(text))
        return text

    run = _chain(pair, tail)
    x = torch.randn(64, 8)
    with patch.object(exc.OperatorIssue, "operator_str", staticmethod(measured)), inductor_config.patch(fx_graph_cache=False):
        for depth in (DEPTH, DEPTH + 2):
            torch._dynamo.reset()
            torch.compile(lambda t, d=depth: run(t, d), backend="inductor", fullgraph=True)(x)
            # The first compile registered the fallback; meet the op again for the first time.
            torch._inductor.lowering.lowerings.pop(torch.ops.test_implicit_fallback.tail.default, None)
    if not formatted:
        pytest.skip("this PyTorch formats the implicit fallback's log line lazily")
    shallow, deep = formatted
    assert deep > 3 * shallow


class _Chain(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = _chain(magi_pair, magi_tail)(x, DEPTH)
        bump_(out)
        return out


def test_magi_compile_never_takes_the_implicit_fallback():
    def formatted(*_):
        raise AssertionError("Inductor took the implicit fallback path")

    torch._dynamo.reset()
    x = torch.randn(64, 8)
    with patch.object(exc.OperatorIssue, "operator_str", staticmethod(formatted)), inductor_config.patch(fx_graph_cache=False):
        out = magi_compile(_Chain(), dynamic_arg_dims={"x": 0})(x)
    torch.testing.assert_close(out, _Chain()(x))
    for op in (torch.ops.test_magi_fallback.pair, torch.ops.test_magi_fallback.tail, torch.ops.test_magi_fallback.bump_):
        assert op.default in torch._inductor.lowering.lowerings
