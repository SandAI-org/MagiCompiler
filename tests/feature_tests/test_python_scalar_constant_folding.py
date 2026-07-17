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

"""Tests for folding constant Python scalar nodes before graph splitting."""

import operator
import tempfile

import pytest
import torch
import torch.nn as nn

from magi_compiler import magi_compile, magi_register_custom_op
from magi_compiler.config import CompileMode, get_compile_config
from magi_compiler.magi_backend import magi_backend as magi_backend_module
from magi_compiler.passes.full_graph.fold_python_scalar_constants import FoldPythonScalarConstantsPass

_FP8_MAX_VALUE = 448.0


def _output_values(graph: torch.fx.Graph) -> tuple:
    output = next(node for node in graph.nodes if node.op == "output")
    assert isinstance(output.args[0], tuple)
    return output.args[0]


def test_folds_constant_scalar_expression_chain_to_literals():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    neg = graph.call_function(operator.neg, (448.0,))
    low = graph.call_function(operator.sub, (neg, 1.0))
    graph.output((x, neg, low))

    fold = FoldPythonScalarConstantsPass()
    assert fold.is_applicable(graph)
    fold(graph)

    assert _output_values(graph) == (x, -448.0, -449.0)
    assert not any(node.op == "call_function" and node.target in {operator.neg, operator.sub} for node in graph.nodes)


def test_does_not_fold_tensor_or_symbolic_dependent_expression():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    neg = graph.call_function(operator.neg, (x,))
    graph.output((neg,))

    FoldPythonScalarConstantsPass()(graph)

    assert _output_values(graph) == (neg,)
    assert neg in graph.nodes


def test_does_not_execute_unlisted_call_function():
    called = False

    def unlisted(value):
        nonlocal called
        called = True
        return -value

    graph = torch.fx.Graph()
    custom = graph.call_function(unlisted, (448.0,))
    graph.output((custom,))

    FoldPythonScalarConstantsPass()(graph)

    assert not called
    assert _output_values(graph) == (custom,)


def _boundary_meta(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@magi_register_custom_op("test_scalar_fold::boundary", infer_output_meta_fn=_boundary_meta, is_subgraph_boundary=True)
def _boundary(x: torch.Tensor) -> torch.Tensor:
    return x + 1.0


class _NegatedConstantAcrossBoundaryBlock(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().clamp(-_FP8_MAX_VALUE, _FP8_MAX_VALUE)
        x = _boundary(x)
        return x.float().clamp(-_FP8_MAX_VALUE, _FP8_MAX_VALUE)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_recaptured_negated_float_constant_is_folded_before_split(monkeypatch):
    torch._dynamo.reset()
    config = get_compile_config()
    monkeypatch.setattr(config, "compile_mode", CompileMode.MAGI_COMPILE)
    monkeypatch.setattr(config, "backend", "inductor")
    monkeypatch.setattr(config, "cache_root_dir", tempfile.mkdtemp())

    split_calls: list[torch.fx.GraphModule] = []
    original_split = magi_backend_module.MagiBackend._split_graph

    def capture_split(self, graph):
        split_calls.append(graph)
        return original_split(self, graph)

    monkeypatch.setattr(magi_backend_module.MagiBackend, "_split_graph", capture_split)

    model = magi_compile(_NegatedConstantAcrossBoundaryBlock().cuda().eval(), dynamic_arg_dims={"x": 0})
    # A value above the FP8 max so clamping is observable in the output.
    x = torch.full((8, 16), 1000.0, device="cuda")
    with torch.no_grad():
        output = model(x).cpu()

    # boundary adds 1.0 to the clamped 448.0, and the second clamp caps it back.
    torch.testing.assert_close(output, torch.full_like(output, _FP8_MAX_VALUE))

    assert split_calls
    for full_graph in split_calls:
        assert not any(
            node.op == "call_function"
            and node.target is operator.neg
            and all(not isinstance(arg, torch.fx.Node) for arg in node.args)
            for node in full_graph.graph.nodes
        )
