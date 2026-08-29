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

"""Unit tests for the copy-engine retarget pass (step 3 of the landing order).

The pass is deliberately tiny -- swap one node's target -- so what
is worth testing is what it *refuses* to touch.  A gather whose input is a cast
or a pad reads a tensor the caching allocator produced, not the symmetric window,
and retargeting it would make the operator reject it at run time.  A gather that
was never marked as a SimpleFSDP weight gather (CP, TP, MoE) must stay on NCCL
even in copy-engine mode.

Built on the real lowering pass so the node shapes are the ones production
produces, with a 1-rank mesh (nothing here touches the transport).
"""

from __future__ import annotations

import operator
import os

import pytest
import torch
import torch.fx as fx

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_AG = torch.ops._c10d_functional.all_gather_into_tensor.default
_AG_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default
_WAIT = torch.ops._c10d_functional.wait_tensor.default
_TO_COPY = torch.ops.aten._to_copy.default
_PAD = torch.ops.aten.constant_pad_nd.default


@pytest.fixture(scope="module")
def mesh_1rank():
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29673")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    created = False
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group("gloo")
        created = True
    from torch.distributed.device_mesh import init_device_mesh

    yield init_device_mesh("cuda", (1,))
    if created:
        dist.destroy_process_group()


def _symm_op():
    from magi_compiler.symm_mem.all_gather import SYMM_ALL_GATHER

    return SYMM_ALL_GATHER


def _graph_with_gathers(mesh, n: int, *, derive: str | None = None, marked: bool = True, shapes: list[tuple] | None = None):
    """``n`` weight gathers reading their shards, in program order.

    ``derive`` inserts a cast or a pad between the shard and the gather, the two
    shapes the lowering pass emits for mixed precision and uneven sharding.
    ``shapes`` gives the per-gather local shard shape.
    """
    from torch.distributed.tensor import Shard, distribute_tensor

    shapes = shapes or [(8, 4)] * n

    g = fx.Graph()
    weights = []
    for i in range(n):
        local = distribute_tensor(torch.randn(*shapes[i], device="cuda", dtype=torch.bfloat16), mesh, [Shard(0)])
        w = g.placeholder(f"layer_{i}_weight")
        w.meta["example_value"] = local
        weights.append((w, local))
    outs = []
    for w, local in weights:
        cur = g.call_method("to_local", (w,))
        cur.meta["example_value"] = local._local_tensor
        if derive == "cast":
            cur = g.call_function(_TO_COPY, (cur,), {"dtype": torch.float32})
            cur.meta["example_value"] = local._local_tensor.to(torch.float32)
        elif derive == "pad":
            cur = g.call_function(_PAD, (cur, [0, 0, 0, 2], 0.0))
            cur.meta["example_value"] = local._local_tensor.new_empty((10, 4))
        ag = g.call_function(_AG, (cur, 1, "dummy_group"))
        ag.meta["example_value"] = local._local_tensor.new_empty(local._local_tensor.shape)
        if marked:
            ag.meta["magi_fsdp_weight_ag"] = True
        outs.append(g.call_function(_WAIT, (ag,)))
    g.output(tuple(outs))
    return fx.GraphModule(torch.nn.Module(), g)


def _gathers(gm, target):
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target is target]


@requires_cuda
def test_marked_gathers_are_retargeted_and_waits_untouched(mesh_1rank):
    from magi_compiler.passes.fsdp_overlap import rewrite_weight_ag_to_copy_engine

    gm = _graph_with_gathers(mesh_1rank, 4)
    assert rewrite_weight_ag_to_copy_engine(gm) == 4

    assert not _gathers(gm, _AG)
    symm = _gathers(gm, _symm_op())
    assert len(symm) == 4
    # The wait is the load-bearing part of the design: it must be the same stock
    # node, still reading the gather.
    waits = _gathers(gm, _WAIT)
    assert len(waits) == 4
    assert [w.args[0] for w in waits] == symm


@requires_cuda
def test_group_args_are_preserved(mesh_1rank):
    """group_size / group_name stay in place, so the cost model reads them the
    same way for either transport."""
    from magi_compiler.passes.fsdp_overlap import rewrite_weight_ag_to_copy_engine

    gm = _graph_with_gathers(mesh_1rank, 3)
    rewrite_weight_ag_to_copy_engine(gm)
    assert all(n.args[1:3] == (1, "dummy_group") for n in _gathers(gm, _symm_op()))


@requires_cuda
@pytest.mark.parametrize("derive", ["cast", "pad"])
def test_derived_shards_stay_on_nccl(mesh_1rank, derive):
    """A cast or pad output is not in the symmetric window; retargeting it would
    be rejected at run time, so it must stay on NCCL."""
    from magi_compiler.passes.fsdp_overlap import rewrite_weight_ag_to_copy_engine

    gm = _graph_with_gathers(mesh_1rank, 3, derive=derive)
    assert rewrite_weight_ag_to_copy_engine(gm) == 0
    assert len(_gathers(gm, _AG)) == 3
    assert not _gathers(gm, _symm_op())


@requires_cuda
def test_unmarked_gathers_are_left_alone(mesh_1rank):
    """CP / TP / MoE gathers are never marked, and must survive copy-engine mode
    untouched -- the transports coexist in one graph."""
    from magi_compiler.passes.fsdp_overlap import rewrite_weight_ag_to_copy_engine

    gm = _graph_with_gathers(mesh_1rank, 3, marked=False)
    assert rewrite_weight_ag_to_copy_engine(gm) == 0
    assert len(_gathers(gm, _AG)) == 3


@requires_cuda
def test_mixed_graph_splits_by_transport(mesh_1rank):
    from magi_compiler.passes.fsdp_overlap import rewrite_weight_ag_to_copy_engine

    gm = _graph_with_gathers(mesh_1rank, 2)
    # Splice in one unmarked gather reading a graph input directly.
    node = _gathers(gm, _AG)[0]
    with gm.graph.inserting_before(node):
        other = gm.graph.placeholder("activation")
        other.meta["example_value"] = torch.randn(4, 4, device="cuda")
        extra = gm.graph.call_function(_AG, (other, 1, "dummy_group"))
        extra.meta["example_value"] = torch.randn(4, 4, device="cuda")
    gm.graph.lint()

    assert rewrite_weight_ag_to_copy_engine(gm) == 2
    assert len(_gathers(gm, _AG)) == 1
    assert len(_gathers(gm, _symm_op())) == 2


@requires_cuda
def test_end_to_end_lowering_then_rewrite(mesh_1rank):
    """The pass has to match what the lowering pass really emits, not what this
    file thinks it emits."""
    from test_fsdp_overlap_lowering import _build_redistribute_graph

    from magi_compiler.passes.fsdp_overlap import lower_and_bucket_full_graph

    gm = _build_redistribute_graph(mesh_1rank, "model_fc1_weight_parameter")
    lower_and_bucket_full_graph(gm, "none", transport="copy_engine")

    assert not _gathers(gm, _AG)
    symm = _gathers(gm, _symm_op())
    assert len(symm) == 1
    assert _gathers(gm, _WAIT)[0].args[0] is symm[0]


@requires_cuda
def test_copy_engine_buckets_then_rewrites_coalesced(mesh_1rank):
    """Phase-1 wrap: arena gathers are bucketed first, then the coalesced
    launch is retargeted.  Members stay separate dests underneath."""
    from magi_compiler.passes.fsdp_overlap import lower_and_bucket_full_graph
    from magi_compiler.symm_mem.all_gather import SYMM_ALL_GATHER, SYMM_ALL_GATHER_COALESCED

    gm = _graph_with_gathers(mesh_1rank, 4)
    n = lower_and_bucket_full_graph(gm, "coalesced", bucket_size_bytes=0, transport="copy_engine")
    assert n == 1
    assert not _gathers(gm, _AG)
    assert not _gathers(gm, _AG_COALESCED)
    assert not _gathers(gm, SYMM_ALL_GATHER)
    assert len(_gathers(gm, SYMM_ALL_GATHER_COALESCED)) == 1
    waits = _gathers(gm, _WAIT)
    assert len(waits) == 4


@requires_cuda
def test_copy_engine_does_not_bucket_cast_gathers(mesh_1rank):
    """Cast outputs are not arena shards; they must stay on NCCL and not join a CE bucket."""
    from magi_compiler.passes.fsdp_overlap import lower_and_bucket_full_graph
    from magi_compiler.symm_mem.all_gather import SYMM_ALL_GATHER_COALESCED

    gm = _graph_with_gathers(mesh_1rank, 3, derive="cast")
    n = lower_and_bucket_full_graph(gm, "coalesced", bucket_size_bytes=0, transport="copy_engine")
    assert n == 0
    assert len(_gathers(gm, _AG)) == 3
    assert not _gathers(gm, SYMM_ALL_GATHER_COALESCED)


def _coalesced_graph(mesh, n: int, *, derive: str | None = None, marked: bool = True):
    """One ``all_gather_into_tensor_coalesced`` over ``n`` shards.

    Bucketing normally builds this node, but it is also what a pre-bucketed graph
    hands the rewrite, so the membership check gets its own graph rather than
    being reached only through the bucket pass.
    """
    from torch.distributed.tensor import Shard, distribute_tensor

    g = fx.Graph()
    locals_ = []
    for i in range(n):
        local = distribute_tensor(torch.randn(8, 4, device="cuda", dtype=torch.bfloat16), mesh, [Shard(0)])
        w = g.placeholder(f"layer_{i}_weight")
        w.meta["example_value"] = local
        cur = g.call_method("to_local", (w,))
        cur.meta["example_value"] = local._local_tensor
        if derive == "cast":
            cur = g.call_function(_TO_COPY, (cur,), {"dtype": torch.float32})
            cur.meta["example_value"] = local._local_tensor.to(torch.float32)
        locals_.append(cur)

    ag = g.call_function(_AG_COALESCED, (locals_, 1, "dummy_group"))
    ag.meta["example_value"] = [torch.empty(8, 4, device="cuda", dtype=torch.bfloat16) for _ in range(n)]
    if marked:
        ag.meta["magi_fsdp_weight_ag"] = True
    outs = []
    for i in range(n):
        item = g.call_function(operator.getitem, (ag, i))
        outs.append(g.call_function(_WAIT, (item,)))
    g.output(tuple(outs))
    return fx.GraphModule(torch.nn.Module(), g)


@requires_cuda
def test_coalesced_of_arena_shards_is_retargeted(mesh_1rank):
    from magi_compiler.passes.fsdp_overlap import rewrite_weight_ag_to_copy_engine
    from magi_compiler.symm_mem.all_gather import SYMM_ALL_GATHER_COALESCED

    gm = _coalesced_graph(mesh_1rank, 3)
    assert rewrite_weight_ag_to_copy_engine(gm) == 1
    assert not _gathers(gm, _AG_COALESCED)
    assert len(_gathers(gm, SYMM_ALL_GATHER_COALESCED)) == 1


@requires_cuda
def test_coalesced_is_all_or_nothing(mesh_1rank):
    """A bucket is one submission, so it can only go to the copy engine if
    *every* member is an arena shard -- one cast member has to keep the whole
    bucket on NCCL rather than being gathered from an address with no peers."""
    from magi_compiler.passes.fsdp_overlap import rewrite_weight_ag_to_copy_engine
    from magi_compiler.symm_mem.all_gather import SYMM_ALL_GATHER_COALESCED

    gm = _coalesced_graph(mesh_1rank, 3, derive="cast")
    assert rewrite_weight_ag_to_copy_engine(gm) == 0
    assert len(_gathers(gm, _AG_COALESCED)) == 1
    assert not _gathers(gm, SYMM_ALL_GATHER_COALESCED)


@requires_cuda
def test_unmarked_coalesced_stays_on_nccl(mesh_1rank):
    from magi_compiler.passes.fsdp_overlap import rewrite_weight_ag_to_copy_engine

    gm = _coalesced_graph(mesh_1rank, 2, marked=False)
    assert rewrite_weight_ag_to_copy_engine(gm) == 0
    assert len(_gathers(gm, _AG_COALESCED)) == 1


@requires_cuda
def test_rewriting_nothing_leaves_the_graph_untouched(mesh_1rank):
    """An NCCL-only graph passed through copy-engine mode must not be recompiled
    into a different graph; the transports share this pass."""
    from magi_compiler.passes.fsdp_overlap import rewrite_weight_ag_to_copy_engine

    gm = _graph_with_gathers(mesh_1rank, 2, marked=False)
    before = gm.print_readable(print_output=False)
    assert rewrite_weight_ag_to_copy_engine(gm) == 0
    assert gm.print_readable(print_output=False) == before
