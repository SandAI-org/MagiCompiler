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

"""Logic tests for the conv channels-last switch (``enable_conv_channels_last``).

The switch is tri-state (``magi_compiler/config.py``):

  * ``True``  -> force on
  * ``False`` -> force off
  * ``None``  -> auto

Its behaviour is split across two layers:

1. Registration (``PostGradPassManager.configure``):
     - ``False``       -> the pass is **not** registered at all.
     - ``True``/``None`` -> the pass is registered; ``force_on`` is set to
       ``enable_conv_channels_last == True`` (i.e. only ``True`` forces on).

2. Runtime decision (``ConvChannelsLastPass.__call__``):
     - ``force_on=True``  -> rewrite unconditionally.
     - ``force_on=False`` (auto) -> **skip** (``return False``) when the graph
       ``is_dynamic`` OR is conv-sparse (``nnodes < 300 * nconv``); only a
       *static, conv-dense* graph gets rewritten.

The end-to-end speedup is validated separately in
``tests/perf_tests/test_conv_channels_last_perf.py``.

NOTE: ``__call__`` is wrapped by ``@emit_pass_lifecycle`` so its boolean return
value is not a reliable "did it rewrite?" signal. We instead assert on whether
``aten.clone`` (channels-last) nodes were inserted into the graph.
"""

import pytest
import torch
import torch.fx as fx

from magi_compiler.config import PassConfig
from magi_compiler.passes.piecewise_graph.conv_channels_last import ConvChannelsLastPass
from magi_compiler.passes.piecewise_graph.post_grad_pass_manager import PostGradPassManager

aten = torch.ops.aten


@pytest.fixture
def fake_mode():
    """A FakeTensorMode backed by a fresh ShapeEnv for symbolic shapes."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    return FakeTensorMode(shape_env=ShapeEnv())


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


def _run_pass(graph: fx.Graph, *, force_on: bool) -> int:
    """Run the pass on ``graph`` and return how many channels-last clones it inserted."""
    ConvChannelsLastPass(force_on=force_on)(graph)
    return _num_channels_last_clones(graph)


# ── Layer 1: registration + force_on tri-state ───────────────────────────


@pytest.mark.parametrize(
    "enable, expect_registered, expect_force_on",
    [
        (None, True, False),  # auto: registered, decides at runtime
        (True, True, True),  # force on: registered, unconditional
        (False, False, None),  # force off: not registered at all
    ],
)
def test_registration_tri_state(enable, expect_registered, expect_force_on):
    pm = PostGradPassManager()
    pm.configure(PassConfig(enable_conv_channels_last=enable))

    conv_passes = [p for p in pm.passes if isinstance(p, ConvChannelsLastPass)]
    assert len(conv_passes) == (1 if expect_registered else 0)
    if expect_registered:
        assert conv_passes[0].force_on is expect_force_on


# ── Layer 2: runtime decision in __call__ ────────────────────────────────


def test_force_on_rewrites_even_dynamic(fake_mode):
    """force_on=True applies channels-last regardless of dynamic/density."""
    graph = _build_conv_graph(fake_mode, dynamic=True, n_conv=1, n_filler=0)
    assert _run_pass(graph, force_on=True) > 0


def test_force_on_rewrites_static(fake_mode):
    """force_on=True applies channels-last on a static graph too."""
    graph = _build_conv_graph(fake_mode, dynamic=False, n_conv=1, n_filler=0)
    assert _run_pass(graph, force_on=True) > 0


def test_auto_skips_dynamic(fake_mode):
    """auto: a dynamic graph is skipped (no clones inserted)."""
    graph = _build_conv_graph(fake_mode, dynamic=True, n_conv=1, n_filler=320)
    assert _run_pass(graph, force_on=False) == 0


def test_auto_skips_static_conv_sparse(fake_mode):
    """auto: a static but conv-sparse graph (nnodes < 300 * nconv) is skipped."""
    graph = _build_conv_graph(fake_mode, dynamic=False, n_conv=1, n_filler=0)
    assert _run_pass(graph, force_on=False) == 0


def test_auto_rewrites_static_conv_dense(fake_mode):
    """auto: a static, conv-dense graph (nnodes >= 300 * nconv) gets rewritten."""
    graph = _build_conv_graph(fake_mode, dynamic=False, n_conv=1, n_filler=320)
    assert _run_pass(graph, force_on=False) > 0


def test_auto_skips_dynamic_conv_dense(fake_mode):
    """auto: dynamic dominates -- even a conv-dense dynamic graph is skipped."""
    graph = _build_conv_graph(fake_mode, dynamic=True, n_conv=1, n_filler=320)
    assert _run_pass(graph, force_on=False) == 0


# ── End-to-end through the pass manager (registration + run) ─────────────


def test_force_off_pass_manager_makes_no_change(fake_mode):
    """enable=False: pass not registered, so the manager never touches conv layout."""
    pm = PostGradPassManager()
    pm.configure(PassConfig(enable_conv_channels_last=False))
    graph = _build_conv_graph(fake_mode, dynamic=False, n_conv=1, n_filler=320)
    for pass_ in [p for p in pm.passes if isinstance(p, ConvChannelsLastPass)]:
        pass_(graph)
    assert _num_channels_last_clones(graph) == 0
