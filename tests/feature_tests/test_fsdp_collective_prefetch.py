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

"""
Unit test for apply_fsdp_collective_prefetch.

Builds a tiny FX graph that mirrors the gaga4 split structure:

    submod 0 (compute):   a = x @ w0
    submod 1 (boundary):  b = relu(a)              # stands in for a custom-op boundary
    submod 2 (compute):   g  = all_gather(w2)
                          wt = wait_tensor(g)
                          out = b @ wt             # weight `w2` is used here

The weight all-gather for `w2` lives in submod 2 alongside its use.  With
distance=2 the pass should move ONLY the `all_gather_into_tensor` launch into
submod 0 (the previous *compute* submod, skipping the odd boundary submod),
while leaving `wait_tensor` and the matmul in submod 2.  After split_module that
makes the collective launch during submod 0's compute and wait right before use.
"""

import operator

import torch
import torch.fx as fx

from magi_compiler.passes.graph_split import (
    apply_fsdp_collective_prefetch,
    bucket_weight_all_gather_coalesced_per_submod,
)

_ALL_GATHER = torch.ops._c10d_functional.all_gather_into_tensor.default
_ALL_GATHER_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default
_WAIT = torch.ops._c10d_functional.wait_tensor.default
_CAT = torch.ops.aten.cat.default
_SPLIT = torch.ops.aten.split_with_sizes.default


def _build_graph():
    g = fx.Graph()
    x = g.placeholder("x")
    w0 = g.placeholder("l_self_modules_proj0_parameters_weight_")
    w2 = g.placeholder("l_self_modules_proj2_parameters_weight_")

    # submod 0: compute
    a = g.call_function(torch.ops.aten.mm.default, (x, w0))
    # submod 1: boundary op (semantics irrelevant to the pass)
    b = g.call_function(torch.ops.aten.relu.default, (a,))
    # submod 2: weight gather + wait + use
    ag = g.call_function(_ALL_GATHER, (w2, 8, "0"))
    wt = g.call_function(_WAIT, (ag,))
    out = g.call_function(torch.ops.aten.mm.default, (b, wt))
    g.output((out,))

    gm = fx.GraphModule(torch.nn.Module(), g)
    node_to_subgraph_id = {a: 0, b: 1, ag: 2, wt: 2, out: 2}
    return gm, node_to_subgraph_id, {"a": a, "b": b, "ag": ag, "wt": wt, "out": out}


def test_launch_moves_wait_stays():
    gm, mapping, nodes = _build_graph()

    moved = apply_fsdp_collective_prefetch(gm, mapping, distance=2)

    assert moved == 1, f"expected exactly one launch moved, got {moved}"
    # Launch relocated to the previous compute submod ...
    assert mapping[nodes["ag"]] == 0, mapping[nodes["ag"]]
    # ... but the wait and the real use stay at the consumer site.
    assert mapping[nodes["wt"]] == 2, mapping[nodes["wt"]]
    assert mapping[nodes["out"]] == 2, mapping[nodes["out"]]

    # Launch is tagged with provenance metadata.
    assert nodes["ag"].meta.get("magi_fsdp_prefetch_to_subgraph") == 0
    assert nodes["ag"].meta.get("magi_fsdp_prefetch_for_consumer_subgraph") == 2

    # The all_gather now precedes the first node of submod 0 in graph order.
    order = list(gm.graph.nodes)
    assert order.index(nodes["ag"]) < order.index(nodes["a"]), "launch must come before submod-0 compute"
    assert order.index(nodes["wt"]) > order.index(nodes["b"]), "wait must remain after the boundary"

    # Graph stays valid.
    gm.graph.lint()


def test_no_move_when_already_early():
    """If the consumer is in submod 0 there is no previous submod to move to."""
    gm, mapping, nodes = _build_graph()
    # Pretend everything is in submod 0.
    for n in nodes.values():
        mapping[n] = 0
    moved = apply_fsdp_collective_prefetch(gm, mapping, distance=2)
    assert moved == 0
    assert mapping[nodes["ag"]] == 0


def test_non_weight_all_gather_is_ignored():
    """An all_gather of a non-weight (activation) tensor must not be moved."""
    g = fx.Graph()
    x = g.placeholder("x")  # activation, not a weight
    w0 = g.placeholder("l_self_modules_proj0_parameters_weight_")
    a = g.call_function(torch.ops.aten.mm.default, (x, w0))
    b = g.call_function(torch.ops.aten.relu.default, (a,))
    ag = g.call_function(_ALL_GATHER, (b, 8, "0"))  # gathers an activation
    wt = g.call_function(_WAIT, (ag,))
    out = g.call_function(torch.ops.aten.relu.default, (wt,))
    g.output((out,))
    gm = fx.GraphModule(torch.nn.Module(), g)
    mapping = {a: 0, b: 1, ag: 2, wt: 2, out: 2}

    moved = apply_fsdp_collective_prefetch(gm, mapping, distance=2)
    assert moved == 0
    assert mapping[ag] == 2


# --------------------------------------------------------------------------- #
# Coalesced bucketing: N same-submod weight gathers -> ONE
# all_gather_into_tensor_coalesced unpacked with operator.getitem (no cat, no
# split_with_sizes clone), one wait per member.
# --------------------------------------------------------------------------- #
def _build_coalesced_graph(world: int = 4):
    """Two weight gathers in submod 2 (a compute submod after a boundary),
    each off its own Shard(0) weight placeholder, plus a use of each."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    fake = FakeTensorMode()
    g = fx.Graph()
    x = g.placeholder("x")
    w0 = g.placeholder("l_self_modules_proj0_parameters_weight_")
    wa = g.placeholder("l_self_modules_mlp_parameters_weight_")
    wb = g.placeholder("l_self_modules_mlp_o_parameters_weight_")

    a = g.call_function(torch.ops.aten.mm.default, (x, w0))      # submod 0 compute
    b = g.call_function(torch.ops.aten.relu.default, (a,))       # submod 1 boundary
    # submod 2: two weight gathers + waits + uses
    ag_a = g.call_function(_ALL_GATHER, (wa, world, "0"))
    wt_a = g.call_function(_WAIT, (ag_a,))
    ag_b = g.call_function(_ALL_GATHER, (wb, world, "0"))
    wt_b = g.call_function(_WAIT, (ag_b,))
    use_a = g.call_function(torch.ops.aten.mm.default, (b, wt_a))
    out = g.call_function(torch.ops.aten.mm.default, (use_a, wt_b))
    g.output((out,))

    # Attach meta example_values (the bucket pass reads dtype + shapes).
    with fake:
        # local shards (chunk, rest) and gathered (world*chunk, rest)
        loc_a = torch.empty((8, 16), dtype=torch.bfloat16)
        loc_b = torch.empty((16, 8), dtype=torch.bfloat16)
    wa.meta["example_value"] = loc_a
    wb.meta["example_value"] = loc_b
    ag_a.meta["example_value"] = loc_a.new_empty((world * 8, 16))
    ag_b.meta["example_value"] = loc_b.new_empty((world * 16, 8))
    wt_a.meta["example_value"] = ag_a.meta["example_value"]
    wt_b.meta["example_value"] = ag_b.meta["example_value"]

    gm = fx.GraphModule(torch.nn.Module(), g)
    node_to_subgraph_id = {a: 0, b: 1, ag_a: 2, wt_a: 2, ag_b: 2, wt_b: 2, use_a: 2, out: 2}
    nodes = {"a": a, "b": b, "ag_a": ag_a, "ag_b": ag_b, "wt_a": wt_a, "wt_b": wt_b, "use_a": use_a, "out": out}
    return gm, node_to_subgraph_id, nodes


def _count(gm, target):
    return sum(1 for n in gm.graph.nodes if n.op == "call_function" and n.target is target)


def test_coalesced_bucket_unpacks_with_getitem():
    gm, mapping, _ = _build_coalesced_graph()
    n = bucket_weight_all_gather_coalesced_per_submod(gm, mapping)
    assert n == 1, f"expected one coalesced bucket, got {n}"
    # exactly one coalesced launch, two getitem, two waits ...
    assert _count(gm, _ALL_GATHER_COALESCED) == 1
    assert _count(gm, _WAIT) == 2
    assert sum(1 for x in gm.graph.nodes if x.op == "call_function" and x.target is operator.getitem) == 2
    # ... and NONE of the concat-path machinery (no cat, no split_with_sizes, no
    # leftover per-member single all_gather).
    assert _count(gm, _CAT) == 0
    assert _count(gm, _SPLIT) == 0
    assert _count(gm, _ALL_GATHER) == 0
    # launch + getitems all tagged / in submod 2; waits in submod 2 too.
    coal = next(x for x in gm.graph.nodes if x.target is _ALL_GATHER_COALESCED)
    assert coal.meta.get("magi_fsdp_weight_ag_coalesced") is True
    assert mapping[coal] == 2
    gm.graph.lint()


def test_coalesced_launch_and_getitems_move_together():
    gm, mapping, _ = _build_coalesced_graph()
    bucket_weight_all_gather_coalesced_per_submod(gm, mapping)
    moved = apply_fsdp_collective_prefetch(gm, mapping, distance=2)
    assert moved == 1, f"expected the coalesced launch moved once, got {moved}"

    coal = next(x for x in gm.graph.nodes if x.target is _ALL_GATHER_COALESCED)
    gis = [x for x in gm.graph.nodes if x.op == "call_function" and x.target is operator.getitem]
    waits = [x for x in gm.graph.nodes if x.op == "call_function" and x.target is _WAIT]

    # launch + both getitems hoisted into submod 0 (the previous compute submod) ...
    assert mapping[coal] == 0
    for gi in gis:
        assert mapping[gi] == 0, f"getitem must move with the coalesced launch, got {mapping[gi]}"
    # ... but the waits stay at the use site in submod 2.
    for wt in waits:
        assert mapping[wt] == 2, f"wait must stay at consumer, got {mapping[wt]}"

    # The list (coalesced launch) and its getitems never become a submod boundary
    # edge: only the getitem outputs cross.  Order check: launch precedes submod-0
    # compute.
    order = list(gm.graph.nodes)
    a = next(x for x in gm.graph.nodes if x.op == "call_function" and x.target is torch.ops.aten.mm.default)
    assert order.index(coal) < order.index(a)
    gm.graph.lint()


if __name__ == "__main__":
    test_launch_moves_wait_stays()
    test_no_move_when_already_early()
    test_non_weight_all_gather_is_ignored()
    test_coalesced_bucket_unpacks_with_getitem()
    test_coalesced_launch_and_getitems_move_together()
    print("PASS: fsdp_collective_prefetch unit tests")
