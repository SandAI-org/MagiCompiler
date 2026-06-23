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

"""Decision-logic tests for ``ConvChannelsLastPass``.

When applicable, the pass rewrites static, conv-heavy graphs (by inserting
``aten.clone`` layout-changing nodes) to trigger channels-last convolutions.
The binary ``enable_conv_channels_last`` config controls registration:

  * ``True`` (default) -> register the pass; its internal heuristics then decide:
    apply iff static shapes AND conv-heavy.
  * ``False`` -> pass not registered at all.

These tests assert the registered pass's heuristic decision. The shared
base-class utilities (``is_dynamic``, ``is_conv_heavy``, config
snapshot/anti-leakage) are tested in ``test_magi_inductor_pass.py``; the
end-to-end speedup in ``tests/perf_tests/test_conv_channels_last_perf.py``.
"""

import pytest
import torch
import torch.fx as fx

from magi_compiler.config import PassConfig
from magi_compiler.passes.piecewise_graph.conv_channels_last import ConvChannelsLastPass

aten = torch.ops.aten


def _build_conv_graph(fake_mode, *, dynamic: bool, n_conv: int = 1, n_filler: int = 0) -> fx.Graph:
    """Build a tiny conv3d graph for the channels-last decision logic.

    ``dynamic`` makes the input placeholder's first dim a free symbol (drives
    ``is_dynamic``). ``n_conv`` adds ``aten.convolution.default`` nodes (5-D
    inputs/weights => conv3d, so the pass targets them). ``n_filler`` adds plain
    ``relu`` nodes to inflate the node count (drives ``nnodes vs 300 * nconv``).

    Inputs are kept contiguous (NCDHW) so the pass has a real layout change to
    make: a successful rewrite inserts ``aten.clone`` (channels_last_3d) nodes.
    """
    graph = fx.Graph()
    if dynamic:
        sym = fake_mode.shape_env.create_unbacked_symint()
        with fake_mode:
            x_val = torch.empty(sym, 8, 4, 4, 4)
    else:
        with fake_mode:
            x_val = torch.empty(2, 8, 4, 4, 4)

    x = graph.placeholder("x")
    x.meta["val"] = x_val

    with fake_mode:
        weight_val = torch.empty(8, 8, 3, 3, 3)
        out_val = torch.empty(2, 8, 4, 4, 4)

    node = x
    for c in range(n_conv):
        weight = graph.placeholder(f"weight_{c}")
        weight.meta["val"] = weight_val
        node = graph.call_function(aten.convolution.default, args=(node, weight))
        node.meta["val"] = out_val
    for _ in range(n_filler):
        node = graph.call_function(aten.relu.default, args=(node,))
        node.meta["val"] = out_val
    graph.output((node,))
    return graph


def _num_channels_last_clones(graph: fx.Graph) -> int:
    """Count ``aten.clone`` nodes the pass inserts to force channels-last."""
    return sum(1 for n in graph.nodes if n.op == "call_function" and n.target == aten.clone.default)


def _run_pass(graph: fx.Graph) -> int:
    """Run the pass on ``graph`` and return how many channels-last clones it inserted."""
    ConvChannelsLastPass()(graph)
    return _num_channels_last_clones(graph)


@pytest.mark.parametrize("value", [True, False])
def test_config_field_binary(value):
    """enable_conv_channels_last accepts True/False (default True covered in test_magi_inductor_pass)."""
    assert PassConfig(enable_conv_channels_last=value).enable_conv_channels_last is value


def test_auto_skips_dynamic(fake_mode):
    """auto: a dynamic graph is skipped (no clones inserted)."""
    graph = _build_conv_graph(fake_mode, dynamic=True, n_conv=1, n_filler=0)
    assert _run_pass(graph) == 0


def test_auto_skips_static_conv_sparse(fake_mode):
    """auto: a static but conv-sparse graph (nnodes >= 300 * nconv, i.e. not is_conv_heavy) is skipped."""
    graph = _build_conv_graph(fake_mode, dynamic=False, n_conv=1, n_filler=320)
    assert _run_pass(graph) == 0


def test_auto_rewrites_static_conv_heavy(fake_mode):
    """auto: a static, conv-heavy graph (nnodes < 300 * nconv, i.e. is_conv_heavy) gets rewritten."""
    graph = _build_conv_graph(fake_mode, dynamic=False, n_conv=1, n_filler=0)
    assert _run_pass(graph) > 0


def test_auto_skips_dynamic_conv_heavy(fake_mode):
    """auto: dynamic dominates -- even a conv-heavy dynamic graph is skipped."""
    graph = _build_conv_graph(fake_mode, dynamic=True, n_conv=1, n_filler=0)
    assert _run_pass(graph) == 0
