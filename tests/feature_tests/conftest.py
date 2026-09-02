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

"""Shared fixtures for Inductor-pass feature tests.

``fake_mode`` plus the tensor/graph builders below are reused by both
``test_magi_inductor_pass.py`` (base-class utilities) and
``test_dynamic_nd_tiling.py`` (ND_TilingWorkaroundPass decision logic).
"""

import pytest
import torch
import torch.fx as fx


@pytest.fixture
def fake_mode():
    """A FakeTensorMode backed by a fresh ShapeEnv for symbolic shapes."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    return FakeTensorMode(shape_env=ShapeEnv())


def static_tensor(fake_mode):
    """A FakeTensor with fully concrete (non-symbolic) dims."""
    with fake_mode:
        return torch.empty(4, 8)


def dynamic_tensor(fake_mode):
    """A FakeTensor whose first dim is a free symbol (mimics a dynamic batch)."""
    sym = fake_mode.shape_env.create_unbacked_symint()
    with fake_mode:
        return torch.empty(sym, 8)


def build_graph_module(fake_mode, *, placeholder_vals=(), placeholder_meta_key="example_value", n_conv=0, n_filler=0):
    """Build a tiny fx.GraphModule for exercising pass decision logic.

    ``placeholder_vals`` populate each placeholder's ``meta[placeholder_meta_key]``
    (drives ``is_dynamic``; ``MagiInductorPass.is_dynamic`` reads ``meta["val"]``
    first and falls back to ``meta["example_value"]``). ``n_conv`` adds
    ``aten.convolution.default`` call nodes and ``n_filler`` adds plain call
    nodes to inflate the node count (drives the ``is_conv_heavy`` heuristic).
    """
    graph = fx.Graph()
    inputs = []
    for i, ev in enumerate(placeholder_vals):
        node = graph.placeholder(f"arg_{i}")
        node.meta[placeholder_meta_key] = ev
        inputs.append(node)
    if not inputs:
        inputs.append(graph.placeholder("arg_0"))

    x = inputs[0]
    for c in range(n_conv):
        weight = graph.placeholder(f"weight_{c}")
        x = graph.call_function(torch.ops.aten.convolution.default, args=(x, weight))
    for _ in range(n_filler):
        x = graph.call_function(torch.ops.aten.relu.default, args=(x,))
    graph.output((x,))
    return fx.GraphModule(torch.nn.Module(), graph)
