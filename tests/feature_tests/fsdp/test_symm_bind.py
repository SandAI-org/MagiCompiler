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

"""Graph-driven binding of weights into symmetric memory.

Two properties matter here and neither is about bytes moving.  The first is
*scope*: exactly the weights the graph gathers get an allocation, and a weight
sitting next to them in the same model does not.  The second is *identity*: the
parameter object Dynamo already guarded on has to survive the move, because the
call that triggered this compilation is going to run against those same objects.

Single rank on purpose -- selection, identity, layout and every fallback are
rank-independent, and a 1-rank window exercises the same allocation path.  The
cross-rank half (peer reads, plan agreement) is covered by ``test_symm_e2e.py``
and ``test_uneven_shard_transport.py``.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.fx as fx

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_AG = torch.ops._c10d_functional.all_gather_into_tensor.default
_WAIT = torch.ops._c10d_functional.wait_tensor.default
_TO_COPY = torch.ops.aten._to_copy.default


@pytest.fixture(scope="module")
def mesh_1rank():
    """A single-rank NCCL group + cuda device mesh (symmetric memory needs both)."""
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29671")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    created = False
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=torch.device("cuda", 0))
        created = True
    from torch.distributed.device_mesh import init_device_mesh

    yield init_device_mesh("cuda", (1,), mesh_dim_names=("dp",))
    if created:
        dist.destroy_process_group()


@pytest.fixture
def group_name(mesh_1rank) -> str:
    return mesh_1rank.get_group("dp").group_name


@pytest.fixture(autouse=True)
def _clean_registry():
    from magi_compiler.symm_mem import reset_registry

    reset_registry()
    yield
    reset_registry()


def _param(mesh, shape=(8, 4), dtype=torch.bfloat16, placement=None):
    from torch.distributed.tensor import Shard, distribute_tensor

    full = torch.randn(*shape, device="cuda", dtype=dtype)
    return distribute_tensor(full, mesh, [placement or Shard(0)])


def _graph(params: list, *, derive: str | None = None, uneven: bool = False, extra_placeholder=None):
    """One marked weight all-gather per parameter, in program order.

    ``derive`` splices the mixed-precision cast between the shard and the gather;
    ``uneven`` sets the rank-identical flag the lowering pass puts on a weight
    whose ``Shard(0)`` does not divide evenly.  Returns the module and the
    ``example_inputs`` Dynamo would hand the backend -- the live parameters, in
    placeholder order.
    """
    from magi_compiler.passes.fsdp_overlap.node_meta import mark_weight_ag

    g = fx.Graph()
    examples: list[object] = []

    # Every placeholder first, as Dynamo emits them: bucketing hoists a member's
    # shard prep above the first gather, which is only legal if the weights it
    # reads are already defined up there.
    weights = []
    for i, p in enumerate(params):
        w = g.placeholder(f"layer_{i}_weight")
        w.meta["example_value"] = p
        examples.append(p)
        weights.append((w, p._local_tensor if hasattr(p, "_local_tensor") else p))
    if extra_placeholder is not None:
        # A weight the graph never gathers: it must come out of binding untouched.
        g.placeholder("spare_weight")
        examples.append(extra_placeholder)

    outs = []
    for w, local in weights:
        cur = g.call_method("to_local", (w,))
        cur.meta["example_value"] = local
        if derive == "cast":
            cur = g.call_function(_TO_COPY, (cur,), {"dtype": torch.float32})
            cur.meta["example_value"] = local.to(torch.float32)
        ag = g.call_function(_AG, (cur, 1, "dummy_group"))
        ag.meta["example_value"] = local.new_empty(local.shape)  # world size 1
        mark_weight_ag(ag, uneven=uneven)
        outs.append(g.call_function(_WAIT, (ag,)))
    g.output(tuple(outs))
    return fx.GraphModule(torch.nn.Module(), g), examples


def _bind(gm, examples, **kw):
    """Binding on its own, standing in for what the backend does around it."""
    from magi_compiler.passes.fsdp_overlap import copy_engine_weight_candidates
    from magi_compiler.symm_mem import bind_graph_weights

    names = (n.name for n in gm.graph.find_nodes(op="placeholder"))
    return bind_graph_weights(gm, copy_engine_weight_candidates(gm), dict(zip(names, examples)), **kw)


# ---------------------------------------------------------------------------
# The allocation itself
# ---------------------------------------------------------------------------
@requires_cuda
def test_a_shard_starts_at_its_own_storage(mesh_1rank, group_name):
    """Every shard owns its whole window.

    ``storage_offset`` 0 is not cosmetic: it is what Dynamo memoized for these
    parameters before the backend ran, and a shard part-way into a shared window
    makes the next fakification of the same source disagree with itself.  The
    16B alignment is what Inductor assumes of every graph input.
    """
    from magi_compiler.symm_mem import alloc_shard

    shard = alloc_shard((8, 4), torch.bfloat16, torch.device("cuda", 0), group_name)
    assert shard.storage_offset() == 0
    assert shard.shape == (8, 4)
    assert shard.is_contiguous()
    assert shard.data_ptr() % 16 == 0


@requires_cuda
def test_peer_view_round_trips_to_the_shard_itself(mesh_1rank, group_name):
    """At one rank the only peer view is this rank's, so it must alias the shard --
    the cheapest check that the window really is what got registered."""
    from magi_compiler.symm_mem import alloc_shard, lookup_shard

    shard = alloc_shard((8, 4), torch.bfloat16, torch.device("cuda", 0), group_name)
    shard.fill_(7.0)

    entry = lookup_shard(shard.data_ptr())
    assert entry is not None
    assert len(entry.peer_views) == 1
    assert torch.equal(entry.peer_views[0], shard)


@requires_cuda
def test_contains_rejects_memory_outside_the_allocation(mesh_1rank, group_name):
    """A caching-allocator tensor must never look like a symmetric shard."""
    from magi_compiler.symm_mem import alloc_shard, registered_buffers

    shard = alloc_shard((8, 8), torch.bfloat16, torch.device("cuda", 0), group_name)
    (buffer,) = registered_buffers()

    assert buffer.contains(shard)
    assert not buffer.contains(torch.empty(8, 8, device="cuda", dtype=torch.bfloat16))


@requires_cuda
def test_find_shard_by_layout_matches_on_shape_and_dtype(mesh_1rank, group_name):
    """The cost model replays a gather on *some* registered shard with the right
    layout -- it cannot use a generic ``empty``, which has no peers.  A miss has to
    be a None it can degrade on, not a wrong-dtype shard it would gather."""
    from magi_compiler.symm_mem import alloc_shard, find_shard_by_layout

    dev = torch.device("cuda", 0)
    small = alloc_shard((8, 4), torch.bfloat16, dev, group_name)
    large = alloc_shard((16, 4), torch.bfloat16, dev, group_name)

    assert find_shard_by_layout((8, 4), torch.bfloat16) is small
    assert find_shard_by_layout((16, 4), torch.bfloat16) is large
    assert find_shard_by_layout((8, 5), torch.bfloat16) is None
    assert find_shard_by_layout((8, 4), torch.float32) is None


@requires_cuda
def test_reset_registry_drops_buffers_and_shards(mesh_1rank, group_name):
    """Tests and the multi-model path rebuild in-process; a stale entry would let a
    freed shard's address answer a lookup."""
    from magi_compiler.symm_mem import alloc_shard, find_shard_by_layout, lookup_shard, registered_buffers, reset_registry

    shard = alloc_shard((8, 4), torch.bfloat16, torch.device("cuda", 0), group_name)
    assert lookup_shard(shard.data_ptr()) is not None

    reset_registry()
    assert registered_buffers() == []
    assert lookup_shard(shard.data_ptr()) is None
    assert find_shard_by_layout((8, 4), torch.bfloat16) is None


@requires_cuda
def test_group_name_of_refuses_a_mesh_it_cannot_resolve(mesh_1rank):
    """Defaulting to WORLD would open the window on the wrong group and read peers
    holding somebody else's rows, which is silent corruption rather than a crash."""
    from magi_compiler.symm_mem import group_name_of

    assert group_name_of(_param(mesh_1rank)) == mesh_1rank.get_group("dp").group_name
    with pytest.raises(RuntimeError, match="cannot resolve the process group"):
        group_name_of(torch.empty(4, 4, device="cuda"))


# ---------------------------------------------------------------------------
# Selection: what the graph says gets bound, and nothing else
# ---------------------------------------------------------------------------
@requires_cuda
def test_only_gathered_weights_are_bound(mesh_1rank):
    """The whole point of reading the graph: a weight the model owns but never
    all-gathers gets no window.  The patch this replaced took the entire subtree."""
    from magi_compiler.symm_mem import lookup_shard, registered_buffers

    gathered = [_param(mesh_1rank), _param(mesh_1rank)]
    spare = _param(mesh_1rank)
    gm, examples = _graph(gathered, extra_placeholder=spare)

    assert len(_bind(gm, examples)) == 2
    # Same dtype and group, so the two land in one pooled window.
    (buffer,) = registered_buffers()
    assert all(buffer.contains(p._local_tensor) for p in gathered)
    assert lookup_shard(spare._local_tensor.data_ptr()) is None


@requires_cuda
def test_binding_keeps_the_parameter_object_and_its_values(mesh_1rank):
    """Dynamo has already read this call's inputs and installed guards against
    these exact objects, so the move has to be a pointer swap under them: a
    replacement parameter would leave *this* invocation gathering the old,
    unregistered storage."""
    from magi_compiler.symm_mem import lookup_shard

    param = _param(mesh_1rank)
    local = param._local_tensor
    before = local.clone()

    gm, examples = _graph([param])
    assert len(_bind(gm, examples)) == 1

    assert param._local_tensor is local
    assert torch.equal(local, before)
    assert lookup_shard(local.data_ptr()) is not None


@requires_cuda
def test_binding_preserves_layout(mesh_1rank):
    from magi_compiler.symm_mem import registered_buffers

    param = _param(mesh_1rank, shape=(16, 8))
    local = param._local_tensor
    stride = local.stride()

    gm, examples = _graph([param])
    _bind(gm, examples)

    assert local.storage_offset() == 0
    assert local.stride() == stride
    assert local.data_ptr() % 16 == 0
    assert registered_buffers()[0].contains(local)


@requires_cuda
def test_a_cast_before_the_gather_binds_nothing(mesh_1rank):
    """Under mixed precision the gather reads the cast's output, which no window
    backs.  Binding the shard anyway would spend symmetric memory on a weight that
    still goes over NCCL -- the failure mode of deciding this at build time."""
    from magi_compiler.symm_mem import registered_buffers

    gm, examples = _graph([_param(mesh_1rank), _param(mesh_1rank)], derive="cast")

    assert _bind(gm, examples) == set()
    assert registered_buffers() == []


@requires_cuda
def test_an_uneven_shard_binds_nothing(mesh_1rank):
    """Read off the flag, never the graph shape: the pad appears only on the ranks
    that own fewer rows, so a shape-derived answer splits one collective across two
    transports and never completes."""
    from magi_compiler.symm_mem import registered_buffers

    gm, examples = _graph([_param(mesh_1rank)], uneven=True)

    assert _bind(gm, examples) == set()
    assert registered_buffers() == []


@requires_cuda
def test_a_replicated_weight_is_not_bound(mesh_1rank):
    from torch.distributed.tensor import Replicate

    from magi_compiler.symm_mem import registered_buffers

    gm, examples = _graph([_param(mesh_1rank, placement=Replicate())])

    assert _bind(gm, examples) == set()
    assert registered_buffers() == []


@requires_cuda
def test_a_meta_weight_is_not_bound(mesh_1rank):
    """Ahead-of-time compilation hands the backend fake inputs. There is no
    allocated weight to move, so binding declines and the graph stays on NCCL --
    those callers reach for ``bind_parameters`` instead."""
    from torch.distributed.tensor import DTensor, Shard

    from magi_compiler.symm_mem import registered_buffers

    fake = DTensor.from_local(torch.empty(8, 4, device="meta"), mesh_1rank, [Shard(0)], run_check=False)
    gm, examples = _graph([fake])

    assert _bind(gm, examples) == set()
    assert registered_buffers() == []


@requires_cuda
def test_an_unresolvable_graph_input_is_not_bound(mesh_1rank):
    """No ``example_inputs`` means no live parameter behind the placeholder."""
    from magi_compiler.symm_mem import registered_buffers

    gm, _examples = _graph([_param(mesh_1rank)])

    assert _bind(gm, {}) == set()
    assert registered_buffers() == []


@requires_cuda
def test_shards_below_the_size_floor_are_not_bound(mesh_1rank):
    """Each shard costs an allocation and a rendezvous, and a small gather is
    launch-bound anyway."""
    from magi_compiler.symm_mem import registered_buffers

    gm, examples = _graph([_param(mesh_1rank, shape=(8, 4))])

    assert _bind(gm, examples, min_shard_bytes=1 << 20) == set()
    assert registered_buffers() == []


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------
@requires_cuda
def test_many_shards_share_few_windows(mesh_1rank):
    """Windows are a capped driver resource -- roughly 128 per process, however
    small.  A window per shard looks fine at this scale and dies at shard 92 of
    gaga4's 808, in ``rendezvous``, with tens of GiB of the card still free.  So
    the invariant worth holding is a count that does not track the shard count."""
    from magi_compiler.symm_mem import registered_buffers

    params = [_param(mesh_1rank) for _ in range(16)]
    gm, examples = _graph(params)

    assert len(_bind(gm, examples)) == 16
    # One dtype, one group -- and 16 shards nowhere near the size cap.
    (buffer,) = registered_buffers()
    assert all(buffer.contains(p._local_tensor) for p in params)


@requires_cuda
def test_a_window_holds_one_dtype_per_group(mesh_1rank):
    """A window is a single ``symm_mem.empty``, so it has exactly one dtype; the
    shards of a second dtype need a second one."""
    from magi_compiler.symm_mem import registered_buffers

    params = [_param(mesh_1rank), _param(mesh_1rank, dtype=torch.float32)]
    gm, examples = _graph(params)

    assert len(_bind(gm, examples)) == 2
    buffers = registered_buffers()
    assert len(buffers) == 2
    assert {b.dtype for b in buffers} == {torch.bfloat16, torch.float32}


@requires_cuda
def test_a_pooled_shard_still_starts_its_own_storage(mesh_1rank):
    """Dynamo memoized ``storage_offset == 0`` for these parameters before the
    backend ran, so a slot handed out as ``window[off:off+n]`` makes the next
    fakification of the same source contradict the shape env -- an assert inside
    ``create_symintnode``, nowhere near here."""
    params = [_param(mesh_1rank) for _ in range(3)]
    gm, examples = _graph(params)

    assert len(_bind(gm, examples)) == 3
    assert [p._local_tensor.storage_offset() for p in params] == [0, 0, 0]
    assert len({p._local_tensor.data_ptr() for p in params}) == 3


@requires_cuda
def test_pooled_shards_do_not_overlap(mesh_1rank):
    """Suballocation is only safe if the slots are disjoint: a bad offset would let
    one weight silently overwrite the next, and the gather would still 'work'."""
    params = [_param(mesh_1rank) for _ in range(4)]
    for i, p in enumerate(params):
        p._local_tensor.fill_(i + 1)
    gm, examples = _graph(params)

    assert len(_bind(gm, examples)) == 4
    for i, p in enumerate(params):
        assert torch.equal(p._local_tensor, torch.full_like(p._local_tensor, i + 1))


# ---------------------------------------------------------------------------
# Aliasing and repetition
# ---------------------------------------------------------------------------
@requires_cuda
def test_a_tied_weight_is_bound_once_and_serves_both_gathers(mesh_1rank):
    """Two placeholders, one tensor. Allocating twice would leave the second
    gather reading a window nobody wrote into."""
    from magi_compiler.symm_mem import registered_buffers

    param = _param(mesh_1rank)
    gm, examples = _graph([param, param])

    assert len(_bind(gm, examples)) == 2
    assert len(registered_buffers()) == 1


@requires_cuda
def test_binding_twice_reuses_the_first_allocation(mesh_1rank):
    """A model can be compiled more than once (a second entry point, a recompile).
    A second allocation per rank would also desynchronize the rendezvous count."""
    from magi_compiler.symm_mem import registered_buffers

    param = _param(mesh_1rank)
    gm, examples = _graph([param])

    assert len(_bind(gm, examples)) == 1
    ptr = param._local_tensor.data_ptr()

    gm2, examples2 = _graph([param])
    assert len(_bind(gm2, examples2)) == 1
    assert len(registered_buffers()) == 1
    assert param._local_tensor.data_ptr() == ptr


# ---------------------------------------------------------------------------
# The explicit entry point, and the pipeline as a whole
# ---------------------------------------------------------------------------
@requires_cuda
def test_bind_parameters_moves_an_explicit_list(mesh_1rank):
    """The escape hatch for callers with no graph: op-level benchmarks and AOT."""
    from magi_compiler.symm_mem import bind_parameters, lookup_shard

    params = [_param(mesh_1rank), _param(mesh_1rank)]
    assert bind_parameters(params) == 2
    assert all(lookup_shard(p._local_tensor.data_ptr()) is not None for p in params)


@requires_cuda
def test_bind_parameters_skips_what_it_cannot_gather(mesh_1rank):
    from torch.distributed.tensor import Replicate

    from magi_compiler.symm_mem import bind_parameters, registered_buffers

    assert bind_parameters([_param(mesh_1rank, placement=Replicate())]) == 0
    assert bind_parameters([torch.empty(8, 4, device="cuda")]) == 0
    assert registered_buffers() == []


@requires_cuda
def test_bound_weights_are_bucketed_and_retargeted(mesh_1rank):
    """The pipeline end to end: bind, then bucket only what bound, then retarget.
    Binding before bucketing is what keeps a bucket homogeneous."""
    from magi_compiler.passes.fsdp_overlap import lower_and_bucket_full_graph
    from magi_compiler.symm_mem.all_gather import CE_ALL_GATHER_COALESCED

    gm, examples = _graph([_param(mesh_1rank) for _ in range(4)])
    n = lower_and_bucket_full_graph(gm, "coalesced", bucket_size_bytes=0, transport="copy_engine", example_inputs=examples)

    assert n == 1
    coalesced = [x for x in gm.graph.nodes if x.target is CE_ALL_GATHER_COALESCED]
    assert len(coalesced) == 1
    assert len(coalesced[0].args[0]) == 4


@requires_cuda
def test_an_unbound_weight_keeps_its_neighbours_on_the_copy_engine(mesh_1rank):
    """Only the weight that could not be bound loses the copy engine. Excluding it
    must not split the bucket around it, or one odd weight would cost throughput
    across the whole model."""
    from magi_compiler.passes.fsdp_overlap import lower_and_bucket_full_graph
    from magi_compiler.passes.fsdp_overlap.node_meta import UNEVEN_SHARD
    from magi_compiler.symm_mem.all_gather import CE_ALL_GATHER_COALESCED

    gm, examples = _graph([_param(mesh_1rank) for _ in range(4)])
    gathers = [n for n in gm.graph.nodes if n.target is _AG]
    gathers[1].meta[UNEVEN_SHARD] = True

    lower_and_bucket_full_graph(gm, "coalesced", bucket_size_bytes=0, transport="copy_engine", example_inputs=examples)

    coalesced = [n for n in gm.graph.nodes if n.target is CE_ALL_GATHER_COALESCED]
    assert len(coalesced) == 1
    assert len(coalesced[0].args[0]) == 3  # the three bound weights, in one bucket
    assert len([n for n in gm.graph.nodes if n.target is _AG]) == 1  # the uneven one, on NCCL


@requires_cuda
def test_unbound_weights_are_still_bucketed_as_nccl(mesh_1rank):
    """Losing the copy engine must not also lose bucketing.

    Bucketing partitions by transport rather than filtering on it: if unbound
    gathers were simply dropped out of the pass, a model that declines binding
    wholesale would go from a handful of coalesced launches to one launch per
    weight, which costs far more memory than the copy engine ever saved.
    """
    from magi_compiler.passes.fsdp_overlap import lower_and_bucket_full_graph
    from magi_compiler.passes.fsdp_overlap.node_meta import UNEVEN_SHARD
    from magi_compiler.symm_mem.all_gather import CE_ALL_GATHER_COALESCED

    _AG_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default

    gm, examples = _graph([_param(mesh_1rank) for _ in range(4)])
    for node in [n for n in gm.graph.nodes if n.target is _AG][2:]:
        node.meta[UNEVEN_SHARD] = True  # two weights the copy engine cannot serve

    assert (
        lower_and_bucket_full_graph(gm, "coalesced", bucket_size_bytes=0, transport="copy_engine", example_inputs=examples)
        == 2
    )

    (ce,) = [n for n in gm.graph.nodes if n.target is CE_ALL_GATHER_COALESCED]
    (nccl,) = [n for n in gm.graph.nodes if n.target is _AG_COALESCED]
    assert len(ce.args[0]) == 2
    assert len(nccl.args[0]) == 2
    assert not [n for n in gm.graph.nodes if n.target is _AG]


@requires_cuda
def test_nccl_transport_binds_nothing(mesh_1rank):
    """Symmetric memory is a copy-engine cost; the default transport must not pay it."""
    from magi_compiler.passes.fsdp_overlap import lower_and_bucket_full_graph
    from magi_compiler.symm_mem import registered_buffers

    gm, examples = _graph([_param(mesh_1rank) for _ in range(2)])
    lower_and_bucket_full_graph(gm, "coalesced", transport="nccl", example_inputs=examples)

    assert registered_buffers() == []
