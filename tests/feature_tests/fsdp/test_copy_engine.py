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

"""Which gathers the copy engine may serve, and the retarget that acts on it.

Two passes, one question each.  ``copy_engine_weight_candidates`` decides what is
*possible* from the graph alone -- a gather whose input is a cast or a pad reads a
tensor the caching allocator produced, not a symmetric window.  The retarget
decides nothing: it acts on ``node_meta.CE_BOUND``, which only binding sets, so a
weight whose allocation never happened cannot be retargeted into an operator that
would reject it at run time.

Deliberately allocation-free, on a 1-rank gloo mesh: the allocation half is
``test_symm_bind.py``'s job, and keeping it out of here is what lets these run
anywhere.
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
    from magi_compiler.symm_mem.all_gather import CE_ALL_GATHER

    return CE_ALL_GATHER


def _graph_with_gathers(
    mesh, n: int, *, derive: str | None = None, marked: bool = True, shapes: list[tuple] | None = None, uneven: bool = False
):
    """``n`` weight gathers reading their shards, in program order.

    ``derive`` inserts a cast or a pad between the shard and the gather, the two
    shapes the lowering pass emits for mixed precision and uneven sharding.
    ``shapes`` gives the per-gather local shard shape.  ``uneven`` sets the
    rank-identical flag the lowering pass puts on a gather whose ``Shard(0)`` does
    not divide evenly -- independently of ``derive``, because the ranks that own a
    full chunk of an uneven weight get no pad at all.
    """
    from torch.distributed.tensor import Shard, distribute_tensor

    from magi_compiler.passes.fsdp_overlap.node_meta import mark_weight_ag

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
            mark_weight_ag(ag, uneven=uneven)
        outs.append(g.call_function(_WAIT, (ag,)))
    g.output(tuple(outs))
    return fx.GraphModule(torch.nn.Module(), g)


def _gathers(gm, target):
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target is target]


def _candidates(gm):
    from magi_compiler.passes.fsdp_overlap import copy_engine_weight_candidates

    return copy_engine_weight_candidates(gm)


def _pretend_bound(gm) -> int:
    """Mark every candidate, standing in for a ``bind_graph_weights`` that
    succeeded.  Keeps this file allocation-free while still deriving the mark from
    the real selection pass rather than hand-placing it."""
    from magi_compiler.passes.fsdp_overlap.node_meta import mark_ce_bound

    marked = 0
    for gather, _holder in _candidates(gm):
        mark_ce_bound(gather)
        marked += 1
    return marked


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
@requires_cuda
def test_plain_weight_gathers_are_candidates(mesh_1rank):
    gm = _graph_with_gathers(mesh_1rank, 3)
    candidates = _candidates(gm)

    assert [g for g, _h in candidates] == _gathers(gm, _AG)
    assert all(h.op == "placeholder" for _g, h in candidates)


@requires_cuda
@pytest.mark.parametrize("derive", ["cast", "pad"])
def test_derived_shards_are_not_candidates(mesh_1rank, derive):
    """A cast or pad output is not in any symmetric window, so there would be no
    peer views to read it from.

    Both graphs here are the same on every rank: ``forward_dtype`` is a model-wide
    setting, and the pad is spliced into every gather.  The case where only *some*
    ranks see the pad is a different property, covered by
    ``test_uneven_shards_are_not_candidates``.
    """
    gm = _graph_with_gathers(mesh_1rank, 3, derive=derive)
    assert _candidates(gm) == []


@requires_cuda
def test_uneven_shards_are_not_candidates(mesh_1rank):
    """The half of an uneven ``Shard(0)`` that the pad does not mark.

    ``ceil(F / world)`` rows go to the leading ranks and the remainder to the trailing
    ones, so only the trailing ranks get a pad.  A rank that owns a full chunk sees a
    clean ``to_local`` and nothing in its own graph says the weight is unevenly split.
    If it decides from the input shape alone it moves to the copy engine while its
    peers stay on NCCL, and their all-gather waits on a rank that will never join it.
    So the decision comes off the flag, which is derived from F and world and is
    therefore the same everywhere.
    """
    gm = _graph_with_gathers(mesh_1rank, 3, uneven=True)
    assert _candidates(gm) == []


@requires_cuda
def test_unmarked_gathers_are_not_candidates(mesh_1rank):
    """CP / TP / MoE gathers are never marked as SimpleFSDP weight gathers, and
    must survive copy-engine mode untouched -- the transports coexist in one graph."""
    gm = _graph_with_gathers(mesh_1rank, 3, marked=False)
    assert _candidates(gm) == []


# ---------------------------------------------------------------------------
# Retarget
# ---------------------------------------------------------------------------
@requires_cuda
def test_bound_gathers_are_retargeted_and_waits_untouched(mesh_1rank):
    from magi_compiler.passes.fsdp_overlap import rewrite_weight_ag_to_copy_engine

    gm = _graph_with_gathers(mesh_1rank, 4)
    assert _pretend_bound(gm) == 4
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
    _pretend_bound(gm)
    rewrite_weight_ag_to_copy_engine(gm)
    assert all(n.args[1:3] == (1, "dummy_group") for n in _gathers(gm, _symm_op()))


@requires_cuda
def test_an_unbound_gather_is_never_retargeted(mesh_1rank):
    """The retarget holds no opinion of its own: only binding knows whether the
    allocation actually happened, and a second opinion here is a second chance to
    disagree with the other ranks."""
    from magi_compiler.passes.fsdp_overlap import rewrite_weight_ag_to_copy_engine

    gm = _graph_with_gathers(mesh_1rank, 3)  # candidates, but nothing bound them
    assert rewrite_weight_ag_to_copy_engine(gm) == 0
    assert len(_gathers(gm, _AG)) == 3
    assert not _gathers(gm, _symm_op())


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

    assert _pretend_bound(gm) == 2
    assert rewrite_weight_ag_to_copy_engine(gm) == 2
    assert len(_gathers(gm, _AG)) == 1
    assert len(_gathers(gm, _symm_op())) == 2


@requires_cuda
def test_rewriting_nothing_leaves_the_graph_untouched(mesh_1rank):
    """An NCCL-only graph passed through copy-engine mode must not be recompiled
    into a different graph; the transports share this pass."""
    from magi_compiler.passes.fsdp_overlap import rewrite_weight_ag_to_copy_engine

    gm = _graph_with_gathers(mesh_1rank, 2, marked=False)
    before = gm.print_readable(print_output=False)
    assert rewrite_weight_ag_to_copy_engine(gm) == 0
    assert gm.print_readable(print_output=False) == before


@requires_cuda
def test_selection_matches_what_the_lowering_pass_emits(mesh_1rank):
    """The candidate shape has to match what lowering really produces, not what
    this file thinks it produces."""
    from magi_compiler.passes.fsdp_overlap import lower_prim_redistribute_to_collectives, rewrite_weight_ag_to_copy_engine

    from .test_fsdp_overlap_lowering import _build_redistribute_graph

    gm = _build_redistribute_graph(mesh_1rank, "model_fc1_weight_parameter")
    lower_prim_redistribute_to_collectives(gm)

    assert _pretend_bound(gm) == 1
    assert rewrite_weight_ag_to_copy_engine(gm) == 1
    assert not _gathers(gm, _AG)
    symm = _gathers(gm, _symm_op())
    assert _gathers(gm, _WAIT)[0].args[0] is symm[0]


# ---------------------------------------------------------------------------
# Bucketing carries the mark
# ---------------------------------------------------------------------------
@requires_cuda
def test_a_bucket_of_bound_gathers_is_retargeted(mesh_1rank):
    """Bucketing runs after binding, so the coalesced node it builds has to inherit
    the mark from its members or the whole bucket falls back."""
    from magi_compiler.passes.fsdp_overlap import bucket_weight_all_gather_coalesced, rewrite_weight_ag_to_copy_engine
    from magi_compiler.symm_mem.all_gather import CE_ALL_GATHER_COALESCED

    gm = _graph_with_gathers(mesh_1rank, 4)
    _pretend_bound(gm)
    assert bucket_weight_all_gather_coalesced(gm, bucket_size_bytes=0) == 1
    assert rewrite_weight_ag_to_copy_engine(gm) == 1

    assert not _gathers(gm, _AG_COALESCED)
    assert len(_gathers(gm, CE_ALL_GATHER_COALESCED)) == 1
    assert len(_gathers(gm, _WAIT)) == 4


@requires_cuda
def test_a_bucket_is_all_or_nothing(mesh_1rank):
    """A bucket is one submission, so it can only go to the copy engine if *every*
    member is bound -- one unbound member has to keep the whole bucket on NCCL
    rather than being gathered from an address with no peers."""
    from magi_compiler.passes.fsdp_overlap import bucket_weight_all_gather_coalesced, rewrite_weight_ag_to_copy_engine
    from magi_compiler.passes.fsdp_overlap.node_meta import CE_BOUND
    from magi_compiler.symm_mem.all_gather import CE_ALL_GATHER_COALESCED

    gm = _graph_with_gathers(mesh_1rank, 3)
    _pretend_bound(gm)
    _gathers(gm, _AG)[1].meta[CE_BOUND] = False

    # No eligibility filter, so the unbound member lands in the bucket anyway --
    # the case the propagation exists to catch.
    assert bucket_weight_all_gather_coalesced(gm, bucket_size_bytes=0) == 1
    assert rewrite_weight_ag_to_copy_engine(gm) == 0
    assert len(_gathers(gm, _AG_COALESCED)) == 1
    assert not _gathers(gm, CE_ALL_GATHER_COALESCED)


def _coalesced_graph(mesh, n: int, *, marked: bool = True, bound: bool = True):
    """One ``all_gather_into_tensor_coalesced`` over ``n`` shards.

    Bucketing normally builds this node, but it is also what a pre-bucketed graph
    hands the rewrite, so the membership check gets its own graph rather than being
    reached only through the bucket pass.
    """
    from torch.distributed.tensor import Shard, distribute_tensor

    from magi_compiler.passes.fsdp_overlap.node_meta import mark_ce_bound, mark_weight_ag

    g = fx.Graph()
    locals_ = []
    for i in range(n):
        local = distribute_tensor(torch.randn(8, 4, device="cuda", dtype=torch.bfloat16), mesh, [Shard(0)])
        w = g.placeholder(f"layer_{i}_weight")
        w.meta["example_value"] = local
        cur = g.call_method("to_local", (w,))
        cur.meta["example_value"] = local._local_tensor
        locals_.append(cur)

    ag = g.call_function(_AG_COALESCED, (locals_, 1, "dummy_group"))
    ag.meta["example_value"] = [torch.empty(8, 4, device="cuda", dtype=torch.bfloat16) for _ in range(n)]
    if marked:
        mark_weight_ag(ag, uneven=False)
        if bound:
            mark_ce_bound(ag)
    outs = []
    for i in range(n):
        item = g.call_function(operator.getitem, (ag, i))
        outs.append(g.call_function(_WAIT, (item,)))
    g.output(tuple(outs))
    return fx.GraphModule(torch.nn.Module(), g)


@requires_cuda
def test_a_pre_bucketed_bound_gather_is_retargeted(mesh_1rank):
    from magi_compiler.passes.fsdp_overlap import rewrite_weight_ag_to_copy_engine
    from magi_compiler.symm_mem.all_gather import CE_ALL_GATHER_COALESCED

    gm = _coalesced_graph(mesh_1rank, 3)
    assert rewrite_weight_ag_to_copy_engine(gm) == 1
    assert not _gathers(gm, _AG_COALESCED)
    assert len(_gathers(gm, CE_ALL_GATHER_COALESCED)) == 1


@requires_cuda
def test_unmarked_coalesced_stays_on_nccl(mesh_1rank):
    from magi_compiler.passes.fsdp_overlap import rewrite_weight_ag_to_copy_engine

    gm = _coalesced_graph(mesh_1rank, 2, marked=False)
    assert rewrite_weight_ag_to_copy_engine(gm) == 0
    assert len(_gathers(gm, _AG_COALESCED)) == 1
