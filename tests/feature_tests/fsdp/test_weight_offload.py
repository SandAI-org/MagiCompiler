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

"""Compile-time host offload of SimpleFSDP weight shards.

Three layers, tested separately because they fail in different ways:

* ``host_pool`` -- the bytes actually leave the device, and come back identical.
* ``magi::h2d_load`` -- the copy runs on its own stream and is published as a
  ``Work``, so ``wait_tensor`` is what makes it visible.  An op that silently
  synchronized instead would pass every correctness check and overlap nothing.
* the FX pass -- the load lands ABOVE the dtype cast and BELOW nothing, and the
  gather ends up reading the loaded copy rather than the freed shard.

Uses a 1-rank process group + device mesh (GPU required).
"""

import os

import pytest
import torch
import torch.fx as fx

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_AG = torch.ops._c10d_functional.all_gather_into_tensor.default
_WAIT = torch.ops._c10d_functional.wait_tensor.default
_TO_COPY = torch.ops.aten._to_copy.default


@pytest.fixture(scope="module")
def dist_1rank():
    """A single-rank process group + cuda device mesh (module-scoped)."""
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29663")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    created = False
    if not dist.is_initialized():
        dist.init_process_group("gloo")
        created = True
    torch.cuda.set_device(0)
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cuda", (1,))
    yield mesh
    if created:
        dist.destroy_process_group()


@pytest.fixture(autouse=True)
def clean_pool():
    from magi_compiler.passes.weight_offload import host_pool

    host_pool.reset()
    yield
    host_pool.reset()


def _park(tensor: torch.Tensor, name: str = "") -> int:
    """Put ``tensor`` in the host pool the production way: reserve, fill, empty, adopt."""
    from magi_compiler.passes.weight_offload import host_pool

    host = host_pool.reserve(tuple(tensor.shape), tensor.dtype, name=name)
    host.copy_(tensor.detach())
    tensor.untyped_storage().resize_(0)
    return host_pool.adopt(host, tensor, name=name)


# ------------------------------------------------ cross-rank agreement


def _cand(name, holder_name, group, shape=(4, 4)):
    from magi_compiler.passes.weight_offload.graph.weight_source import OffloadCandidate

    holder = type("Node", (), {"name": holder_name})()
    local = torch.empty(shape, dtype=torch.float32)
    return OffloadCandidate(holder=holder, local=local, name=name, nbytes=local.numel() * local.element_size(), group=group)


def _patch_dist(monkeypatch, *, world, replies):
    """``replies[group]`` is the list-of-ranks payload ``all_gather_object`` should write."""
    from magi_compiler.passes.weight_offload.graph import bind

    monkeypatch.setattr(bind.dist, "is_available", lambda: True)
    monkeypatch.setattr(bind.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(bind.dist, "get_world_size", lambda group=None: world)

    seen = []

    def fake_all_gather_object(out, obj, group=None):
        seen.append(group)
        payload = replies[group]
        assert len(payload) == len(out)
        for i, item in enumerate(payload):
            out[i] = item

    monkeypatch.setattr(bind.dist, "all_gather_object", fake_all_gather_object)
    return seen


def test_align_keeps_only_candidates_every_rank_in_the_group_planned(monkeypatch):
    """A weight only some ranks want is dropped; the rest of the plan stays."""
    from collections import Counter

    from magi_compiler.passes.weight_offload.graph import bind

    group = object()
    shared = _cand("w_shared", "tl_shared", group)
    mine_only = _cand("w_mine", "tl_mine", group)
    plan = [shared, mine_only]
    mine_keys = [bind._candidate_key(c) for c in plan]
    peer_keys = [mine_keys[0]]
    _patch_dist(monkeypatch, world=2, replies={group: [mine_keys, peer_keys]})

    skipped = Counter()
    kept = bind._align_across_ranks(plan, {}, skipped)
    assert [c.name for c in kept] == ["w_shared"]
    assert skipped["not planned by every rank in the shard group"] == 1


def test_align_votes_per_group_not_on_world(monkeypatch):
    """Expert-mesh disagreement must not take a dense-mesh weight with it."""
    from collections import Counter

    from magi_compiler.passes.weight_offload.graph import bind

    dense, expert = object(), object()
    dense_w = _cand("dense.w", "tl_dense", dense)
    expert_w = _cand("expert.w", "tl_expert", expert)
    plan = [dense_w, expert_w]
    dense_key = bind._candidate_key(dense_w)
    expert_key = bind._candidate_key(expert_w)
    seen = _patch_dist(monkeypatch, world=2, replies={dense: [[dense_key], [dense_key]], expert: [[expert_key], []]})

    skipped = Counter()
    kept = bind._align_across_ranks(plan, {}, skipped)
    assert [c.name for c in kept] == ["dense.w"]
    assert skipped["not planned by every rank in the shard group"] == 1
    assert set(seen) == {dense, expert}, "each mesh must vote on its own group, never WORLD"


def test_align_keeps_ungrouped_candidates_without_a_collective(monkeypatch):
    """An unsharded Parameter has no mesh and does not enter a vote."""
    from collections import Counter

    from magi_compiler.passes.weight_offload.graph import bind

    plain = _cand("linear.weight", "l_self_weight", group=None)
    seen = _patch_dist(monkeypatch, world=2, replies={})
    kept = bind._align_across_ranks([plain], {}, Counter())
    assert kept == [plain]
    assert seen == []


def test_align_still_enters_a_group_found_only_on_graph_inputs(monkeypatch):
    """A rank that collected nothing on a mesh must still join that mesh's vote."""
    from collections import Counter

    from magi_compiler.passes.weight_offload.graph import bind

    group = object()
    fake_param = type("P", (), {"_spec": type("S", (), {"mesh": object()})()})()
    monkeypatch.setattr(bind, "mesh_group", lambda obj: group if obj is fake_param else None)

    seen = _patch_dist(monkeypatch, world=2, replies={group: [[], [("peer", "h", (4, 4), "torch.float32")]]})
    skipped = Counter()
    kept = bind._align_across_ranks([], {"w": fake_param}, skipped)
    assert kept == []
    assert seen == [group]


def test_align_failed_group_does_not_abort_the_others(monkeypatch):
    """A collective failing on one mesh drops that mesh, not the whole plan."""
    from collections import Counter

    from magi_compiler.passes.weight_offload.graph import bind

    dense, expert = object(), object()
    dense_w = _cand("dense.w", "tl_dense", dense)
    expert_w = _cand("expert.w", "tl_expert", expert)
    plan = [dense_w, expert_w]
    dense_key = bind._candidate_key(dense_w)

    monkeypatch.setattr(bind.dist, "is_available", lambda: True)
    monkeypatch.setattr(bind.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(bind.dist, "get_world_size", lambda group=None: 2)

    def fake_all_gather_object(out, obj, group=None):
        if group is expert:
            raise RuntimeError("expert group is down")
        out[0] = [dense_key]
        out[1] = [dense_key]

    monkeypatch.setattr(bind.dist, "all_gather_object", fake_all_gather_object)

    skipped = Counter()
    kept = bind._align_across_ranks(plan, {}, skipped)
    assert [c.name for c in kept] == ["dense.w"]
    assert skipped["process-group agreement failed"] == 1


# ---------------------------------------------------------------- host pool


@requires_cuda
def test_many_shards_share_one_pinned_slab():
    """One slab per dtype, not one pinned allocation per weight: cudaHostAlloc is a
    driver round trip that stalls every stream, and a model has thousands of these."""
    from magi_compiler.passes.weight_offload import host_pool

    slots = [_park(torch.randn(128, 32, device="cuda", dtype=torch.bfloat16)) for _ in range(16)]

    storages = {host_pool.get(s).untyped_storage().data_ptr() for s in slots}
    assert len(storages) == 1, f"expected one backing slab, got {len(storages)}"


@requires_cuda
def test_restore_all_puts_the_shards_back():
    from magi_compiler.passes.weight_offload import host_pool

    local = torch.randn(64, 8, device="cuda")
    expected = local.clone()
    slot = _park(local)
    assert local.untyped_storage().nbytes() == 0

    host_pool.restore_all()
    torch.testing.assert_close(local, expected)
    assert host_pool.is_resident(slot), "restore_all must mark the slot resident so source() agrees"
    assert host_pool.source(slot).data_ptr() == local.data_ptr()


@requires_cuda
def test_reserve_and_adopt_park_a_shard_that_was_never_on_the_device():
    """The only way into the pool, and the reason peak memory drops.

    ``reserve`` hands out the host buffer up front, the loader fills it, and
    ``adopt`` pairs it with a device tensor that carries nothing but shape --
    so the shard is offloaded without ever having been resident.
    """
    from magi_compiler.passes.weight_offload import host_pool

    host = host_pool.reserve((256, 64), torch.bfloat16, name="layers.0.w")
    assert host.is_pinned(), "a pageable source would make the load synchronous"
    host.copy_(torch.arange(256 * 64, dtype=torch.bfloat16).reshape(256, 64))

    stand_in = torch.empty(256, 64, device="cuda", dtype=torch.bfloat16)
    stand_in.untyped_storage().resize_(0)
    slot = host_pool.adopt(host, stand_in, name="layers.0.w")

    assert host_pool.slot_of(stand_in) == slot, "the source looks the slot up by the device tensor's identity"
    assert host_pool.name_of(slot) == "layers.0.w"
    assert host_pool.slot_bytes(slot) == 256 * 64 * 2, "a storage-free tensor still has to report its real size"
    assert host_pool.source(slot).data_ptr() == host.data_ptr()

    # And the shard can still be handed back, which is what the placement pass
    # does for a load it cannot hide.
    host_pool.make_resident(slot)
    torch.testing.assert_close(stand_in, host.cuda())


@requires_cuda
def test_adopt_refuses_a_device_tensor_that_still_holds_bytes():
    """Adopting one would leak it: the pool would believe the shard is parked
    while its device storage stays allocated for the life of the process."""
    from magi_compiler.passes.weight_offload import host_pool

    host = host_pool.reserve((32, 8), torch.bfloat16)
    with pytest.raises(ValueError, match="no storage behind it"):
        host_pool.adopt(host, torch.empty(32, 8, device="cuda", dtype=torch.bfloat16))


@requires_cuda
def test_unclaimed_shards_are_handed_back_to_the_device():
    """The fail-safe: a parked shard nothing loads is a kernel reading freed storage.

    Parking is decided from a weight's placements while the model is built;
    whether a graph loads it is only known once the lowering has run. Everything
    that falls in that gap has to come back, and everything that does not must
    be left alone -- putting a loaded shard back is a silent regression to no
    offload at all.
    """
    from magi_compiler.passes.weight_offload import host_pool

    loaded = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    orphan = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    expected = orphan.clone()
    loaded_slot = _park(loaded, name="kept")
    orphan_slot = _park(orphan, name="orphan")
    host_pool.mark_claimed(loaded_slot)

    assert host_pool.restore_unclaimed() == ["orphan"]
    assert not host_pool.is_resident(loaded_slot), "a shard the graph loads must stay in host memory"
    assert host_pool.is_resident(orphan_slot)
    torch.testing.assert_close(orphan, expected)
    assert host_pool.restore_unclaimed() == [], "already-restored shards must not be reported twice"


@requires_cuda
def test_adopt_is_idempotent_on_the_same_stand_in():
    from magi_compiler.passes.weight_offload import host_pool

    host = host_pool.reserve((16, 8), torch.bfloat16, name="w")
    stand_in = torch.empty(16, 8, device="cuda", dtype=torch.bfloat16)
    stand_in.untyped_storage().resize_(0)

    first = host_pool.adopt(host, stand_in, name="w")
    second = host_pool.adopt(host, stand_in, name="w")
    assert first == second
    assert host_pool.num_bound() == 1


@requires_cuda
def test_make_resident_many_is_one_batch():
    """Promotion of several slots must not synchronize per slot."""
    from magi_compiler.passes.weight_offload import host_pool

    shards = [torch.randn(32, 16, device="cuda") for _ in range(4)]
    expected = [s.clone() for s in shards]
    slots = [_park(s) for s in shards]
    assert host_pool.make_resident_many(slots) == sum(s.numel() * s.element_size() for s in expected)
    assert host_pool.make_resident_many(slots) == 0
    for slot, exp in zip(slots, expected):
        assert host_pool.is_resident(slot)
        torch.testing.assert_close(host_pool.source(slot), exp)


@requires_cuda
def test_bandwidth_probe_is_plausible():
    from magi_compiler.passes.weight_offload import host_pool

    bw = host_pool.h2d_bandwidth_bytes_per_ns()
    # bytes/ns == GB/s.  Anything outside this is a broken probe, not a slow bus.
    assert 1.0 < bw < 500.0, f"implausible H2D bandwidth {bw} GB/s"


def test_bandwidth_override_is_taken_verbatim():
    """GB/s and bytes/ns are the same number; a stray unit conversion here would
    mis-size every overlap window by 1e9."""
    from magi_compiler.passes.weight_offload import host_pool

    assert host_pool.h2d_bandwidth_bytes_per_ns(override_gbps=42.0) == 42.0


@requires_cuda
def test_reset_clears_the_cached_bandwidth():
    from magi_compiler.passes.weight_offload import host_pool

    host_pool.h2d_bandwidth_bytes_per_ns()
    assert host_pool.default_pool()._bandwidth_bytes_per_ns is not None
    host_pool.reset()
    assert host_pool.default_pool()._bandwidth_bytes_per_ns is None


# ------------------------------------------------------------------- the op


@requires_cuda
def test_h2d_load_returns_the_offloaded_bytes():
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    local = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
    expected = local.clone()
    slot = _park(local)

    out = H2D_LOAD(local, slot)
    _WAIT(out)
    torch.cuda.synchronize()

    assert out.is_cuda and out.shape == expected.shape and out.dtype == expected.dtype
    torch.testing.assert_close(out, expected)


@requires_cuda
def test_h2d_load_runs_off_the_compute_stream():
    """The copy must be issued on the load stream, not the caller's.

    This is the property the whole design rests on: an op that copied on the
    current stream would be correct, would pass the check above, and would
    overlap exactly nothing no matter where the reorder pass put it.
    """
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD, h2d_stream

    local = torch.randn(4096, 1024, device="cuda", dtype=torch.bfloat16)
    slot = _park(local)
    torch.cuda.synchronize()

    before = torch.cuda.Event()
    out = H2D_LOAD(local, slot)
    before.record(torch.cuda.current_stream())
    # The compute stream reaches its own marker without the transfer having
    # landed; only the load stream is holding the copy.
    torch.cuda.current_stream().synchronize()
    assert before.query(), "the compute stream should not have been blocked by the load"
    h2d_stream().synchronize()
    _WAIT(out)  # retire the registered Work, so teardown has nothing to complain about


@requires_cuda
def test_wait_tensor_is_what_synchronizes_the_load():
    """``wait_tensor`` has to find a registered Work for the load's output.

    Without the registration it is a silent no-op -- the gather would then read a
    half-filled buffer, on a schedule that only misbehaves once the pass starts
    hoisting loads far enough for the race to open.
    """
    import torch._C._distributed_c10d as _c10d

    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    local = torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16)
    expected = local.clone()
    slot = _park(local)

    out = H2D_LOAD(local, slot)
    assert _c10d._get_work_registry_size() > 0, "h2d_load must publish its event as a c10d Work"
    _WAIT(out)
    # No device sync: wait_tensor alone must have ordered the compute stream
    # behind the copy, so reading `out` here is already safe.
    torch.testing.assert_close(out.float().sum().cpu(), expected.float().sum().cpu())


@requires_cuda
def test_h2d_load_meta_kernel_preserves_layout():
    """Inductor traces the op with fake tensors; a wrong meta shape shows up as a
    lowering error far away from here."""
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    with torch._subclasses.FakeTensorMode():
        shard = torch.empty(64, 16, device="cuda", dtype=torch.bfloat16)
        out = H2D_LOAD(shard, 0)
    assert out.shape == shard.shape and out.dtype == shard.dtype and out.device == shard.device


# ----------------------------------------------------------------- FX pass


def _lowered_weight_graph(mesh, name="model_fc1_weight_parameter", *, forward_dtype=None, rows=8, cols=4):
    """A lowered SimpleFSDP weight gather, plus the live DTensor behind it."""
    from torch.distributed.tensor import Partial, Replicate, Shard, distribute_tensor

    from magi_compiler.passes.fsdp_overlap import lower_prim_redistribute_to_collectives

    full = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    sharded = distribute_tensor(full, mesh, [Shard(0)])
    replicated = distribute_tensor(full, mesh, [Replicate()])

    g = fx.Graph()
    w = g.placeholder(name)
    w.meta["example_value"] = sharded
    rd = g.call_method(
        "redistribute", (w,), {"placements": [Replicate()], "forward_dtype": forward_dtype, "backward_dtype": None}
    )
    rd.meta["example_value"] = replicated
    tl = g.call_method("to_local", (rd,), {"grad_placements": [Partial()]})
    tl.meta["example_value"] = replicated._local_tensor
    g.output((tl,))
    gm = fx.GraphModule(torch.nn.Module(), g)

    assert lower_prim_redistribute_to_collectives(gm) == 1
    return gm, sharded


def _nodes(gm, target):
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target is target]


@requires_cuda
def test_bind_and_insert_puts_the_load_between_the_shard_and_the_gather(dist_1rank):
    from magi_compiler.passes.fsdp_overlap import FsdpShardSource
    from magi_compiler.passes.weight_offload import bind_weights_to_host, insert_h2d_loads, is_host_offloaded
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    gm, param = _lowered_weight_graph(dist_1rank)
    _park(param._local_tensor)
    assert bind_weights_to_host(gm, [param], FsdpShardSource(), min_bytes=0) == 1
    assert insert_h2d_loads(gm, FsdpShardSource()) == 1

    loads = _nodes(gm, H2D_LOAD)
    assert len(loads) == 1
    load = loads[0]
    assert load.args[0].op == "call_method" and load.args[0].target == "to_local"

    # The gather reads the loaded copy, through the load's wait -- not the shard,
    # whose storage no longer exists.
    (ag,) = _nodes(gm, _AG)
    assert is_host_offloaded(ag)
    reachable = {load, *(n for n in gm.graph.nodes if n.op == "call_function" and n.target is _WAIT)}
    assert any(a in reachable for a in ag.all_input_nodes)
    assert param._local_tensor.untyped_storage().nbytes() == 0


@requires_cuda
def test_load_sits_above_the_dtype_cast(dist_1rank):
    """The cast must run on the device, after the load.

    Casting on the host would burn CPU and, for a fp32-master / bf16-forward
    weight, double the bytes crossing PCIe -- the transfer this whole pass exists
    to hide.
    """
    from magi_compiler.passes.fsdp_overlap import FsdpShardSource
    from magi_compiler.passes.weight_offload import bind_weights_to_host, insert_h2d_loads
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    gm, param = _lowered_weight_graph(dist_1rank, forward_dtype=torch.float32)
    _park(param._local_tensor)
    assert bind_weights_to_host(gm, [param], FsdpShardSource(), min_bytes=0) == 1
    assert insert_h2d_loads(gm, FsdpShardSource()) == 1

    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    (load,) = _nodes(gm, H2D_LOAD)
    (cast,) = _nodes(gm, _TO_COPY)
    assert order[load] < order[cast]
    # and the cast reads the loaded shard, not the freed one
    assert any(isinstance(a, fx.Node) and a.target is _WAIT for a in cast.all_input_nodes)


@requires_cuda
def test_second_graph_over_the_same_parameters_still_gets_its_loads(dist_1rank):
    """A model compiled for several shapes produces several graphs over ONE set
    of parameters.

    The shard was adopted once; every later graph still has to load it back.
    Treating "already adopted" as "nothing to do" leaves the second graph
    all-gathering an empty stand-in -- which surfaces as an illegal memory
    access inside NCCL, on every rank, with nothing pointing back here.
    """
    from magi_compiler.passes.fsdp_overlap import FsdpShardSource
    from magi_compiler.passes.weight_offload import bind_weights_to_host, insert_h2d_loads
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    gm1, param = _lowered_weight_graph(dist_1rank)
    _park(param._local_tensor)
    assert bind_weights_to_host(gm1, [param], FsdpShardSource(), min_bytes=0) == 1
    assert insert_h2d_loads(gm1, FsdpShardSource()) == 1
    assert param._local_tensor.untyped_storage().nbytes() == 0

    # A second graph over the same live parameter, as a second shape would give.
    gm2, _ = _lowered_weight_graph(dist_1rank)
    assert bind_weights_to_host(gm2, [param], FsdpShardSource(), min_bytes=0) == 1, "the adopted shard is still a candidate"
    assert insert_h2d_loads(gm2, FsdpShardSource()) == 1, "the second graph needs its own load"

    slots = {n.args[1] for n in _nodes(gm2, H2D_LOAD)}
    assert slots == {n.args[1] for n in _nodes(gm1, H2D_LOAD)}, "both graphs must read the same slot"


def _twice_gathered_weight_graph(mesh, rows=8, cols=4):
    """One parameter, two independent gathers -- as a weight read by two branches."""
    from torch.distributed.tensor import Partial, Replicate, Shard, distribute_tensor

    from magi_compiler.passes.fsdp_overlap import lower_prim_redistribute_to_collectives

    full = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    sharded = distribute_tensor(full, mesh, [Shard(0)])
    replicated = distribute_tensor(full, mesh, [Replicate()])

    g = fx.Graph()
    w = g.placeholder("model_shared_weight_parameter")
    w.meta["example_value"] = sharded
    outs = []
    for _ in range(2):
        rd = g.call_method("redistribute", (w,), {"placements": [Replicate()], "forward_dtype": None, "backward_dtype": None})
        rd.meta["example_value"] = replicated
        tl = g.call_method("to_local", (rd,), {"grad_placements": [Partial()]})
        tl.meta["example_value"] = replicated._local_tensor
        outs.append(tl)
    g.output(tuple(outs))
    gm = fx.GraphModule(torch.nn.Module(), g)

    assert lower_prim_redistribute_to_collectives(gm) == 2
    return gm, sharded


@requires_cuda
def test_a_weight_two_gathers_read_gets_a_load_for_each(dist_1rank):
    """The shard is adopted once; loading is per gather.

    Each gather reaches the shard through its own ``to_local``, and the splice
    only repoints the readers of the holder it was handed. Treating the second
    one as a duplicate of the first leaves it gathering the empty stand-in --
    an illegal access inside NCCL, on every rank, with nothing pointing back
    here.
    """
    from magi_compiler.passes.fsdp_overlap import FsdpShardSource
    from magi_compiler.passes.weight_offload import bind_weights_to_host, host_pool, insert_h2d_loads
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    gm, param = _twice_gathered_weight_graph(dist_1rank)
    expected = param._local_tensor.clone()
    _park(param._local_tensor)

    assert bind_weights_to_host(gm, [param], FsdpShardSource(), min_bytes=0) == 2, "both gathers need a candidate"
    assert host_pool.num_bound() == 1, "one shard, one set of host bytes"
    torch.testing.assert_close(host_pool.get(0).cuda(), expected)

    assert insert_h2d_loads(gm, FsdpShardSource()) == 2
    loads = _nodes(gm, H2D_LOAD)
    assert {n.args[1] for n in loads} == {0}, "both loads read the one slot"

    # No gather may still be reading the holder whose storage is gone.
    for ag in _nodes(gm, _AG):
        assert any(isinstance(a, fx.Node) and a.target is _WAIT for a in ag.all_input_nodes), ag.format_node()


# Named to match what the passes look for: through torch 2.9 Dynamo captures a
# DTensor ``redistribute`` / ``to_local`` in an on-the-fly function of this name,
# and the predicates key off ``__name__``.
def prim_redistribute(x):
    return x


def prim_to_local(x):
    return x


def _replicated_weight_graph(mesh, rows=9, cols=64):
    """A weight SimpleFSDP replicated instead of sharding, as the lowering leaves it.

    ``rows`` is deliberately indivisible: athena replicates a ``Shard(0)`` whose
    dim0 does not divide its mesh, because padding only the trailing ranks makes
    the graph differ per rank and deadlocks NCCL. The lowering then declines it
    -- it only handles ``Shard(0)`` -- so the prim redistribute/to_local pair
    stays in the graph and there is no all-gather to key off.
    """
    from torch.distributed.tensor import Replicate, distribute_tensor

    from magi_compiler.passes.fsdp_overlap import lower_prim_redistribute_to_collectives

    full = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    replicated = distribute_tensor(full, mesh, [Replicate()])

    g = fx.Graph()
    w = g.placeholder("model_odd_weight_parameter")
    w.meta["example_value"] = replicated
    rd = g.call_function(prim_redistribute, (w,))
    rd.meta["example_value"] = replicated
    tl = g.call_function(prim_to_local, (rd,))
    tl.meta["example_value"] = replicated._local_tensor
    reader = g.call_function(torch.ops.aten.relu.default, (tl,))
    reader.meta["example_value"] = replicated._local_tensor
    g.output((reader,))
    gm = fx.GraphModule(torch.nn.Module(), g)

    assert lower_prim_redistribute_to_collectives(gm) == 0, "a replicated weight has nothing to lower"
    return gm, replicated


def _mixed_weight_graph(mesh, rows=8, cols=64, odd_rows=9):
    """One sharded weight and one replicated one, as a real model has them."""
    from torch.distributed.tensor import Partial, Replicate, Shard, distribute_tensor

    from magi_compiler.passes.fsdp_overlap import lower_prim_redistribute_to_collectives

    full = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    shard = distribute_tensor(full, mesh, [Shard(0)])
    gathered = distribute_tensor(full, mesh, [Replicate()])
    repl = distribute_tensor(torch.randn(odd_rows, cols, device="cuda", dtype=torch.bfloat16), mesh, [Replicate()])

    g = fx.Graph()
    w = g.placeholder("model_fc1_weight_parameter")
    w.meta["example_value"] = shard
    rd = g.call_method("redistribute", (w,), {"placements": [Replicate()], "forward_dtype": None, "backward_dtype": None})
    rd.meta["example_value"] = gathered
    tl = g.call_method("to_local", (rd,), {"grad_placements": [Partial()]})
    tl.meta["example_value"] = gathered._local_tensor

    ow = g.placeholder("model_odd_weight_parameter")
    ow.meta["example_value"] = repl
    ord_ = g.call_function(prim_redistribute, (ow,))
    ord_.meta["example_value"] = repl
    otl = g.call_function(prim_to_local, (ord_,))
    otl.meta["example_value"] = repl._local_tensor
    oread = g.call_function(torch.ops.aten.relu.default, (otl,))
    oread.meta["example_value"] = repl._local_tensor

    g.output((tl, oread))
    gm = fx.GraphModule(torch.nn.Module(), g)

    assert lower_prim_redistribute_to_collectives(gm) == 1, "only the sharded weight has anything to lower"
    return gm, shard, repl


@requires_cuda
def test_a_replicated_weight_is_offloaded_even_though_nothing_gathers_it(dist_1rank):
    """The weights that cost the most per GPU were the ones being skipped.

    SimpleFSDP replicates a shard whose dim0 does not divide its mesh, so every
    rank holds the whole tensor rather than 1/N -- world_size times the device
    memory of the same weight sharded. Keying offload off all-gathers missed
    exactly those, and the safety net then dutifully put back anything host-first
    had parked for them.
    """
    from magi_compiler.passes.fsdp_overlap import FsdpShardSource
    from magi_compiler.passes.weight_offload import bind_weights_to_host, host_pool, insert_h2d_loads
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    gm, param = _replicated_weight_graph(dist_1rank)
    expected = param._local_tensor.clone()
    _park(param._local_tensor)

    assert bind_weights_to_host(gm, [param], FsdpShardSource(), min_bytes=0) == 1
    assert param._local_tensor.untyped_storage().nbytes() == 0, "the full copy must leave the device"
    torch.testing.assert_close(host_pool.get(0).cuda(), expected)

    assert insert_h2d_loads(gm, FsdpShardSource()) == 1
    (load,) = _nodes(gm, H2D_LOAD)
    # The load reads the to_local, and the reader reads the load's wait -- not
    # the to_local, whose storage no longer exists.
    assert getattr(load.args[0].target, "__name__", "") == "prim_to_local"
    (reader,) = _nodes(gm, torch.ops.aten.relu.default)
    assert any(isinstance(a, fx.Node) and a.target is _WAIT for a in reader.all_input_nodes)


@requires_cuda
def test_an_ungathered_weight_gets_a_load_to_itself(dist_1rank):
    """It has no bucket to mirror, and merging it into one would be wrong twice.

    Its bytes live to the last matmul that reads them rather than dying at a
    gather, so a shared bucket would hold a full-size buffer open for as long as
    its shortest-lived member needs. And the splice hoists a group's holders to
    their earliest common reader, which is only legal while a holder reads
    nothing but a placeholder -- this one reads a redistribute.
    """
    from magi_compiler.passes.fsdp_overlap import FsdpShardSource
    from magi_compiler.passes.weight_offload import bind_weights_to_host, insert_h2d_loads
    from magi_compiler.passes.weight_offload.node_meta import host_slot
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD, H2D_LOAD_COALESCED

    gm, shard, repl = _mixed_weight_graph(dist_1rank)
    _park(shard._local_tensor)
    _park(repl._local_tensor)

    examples = {"model_fc1_weight_parameter": shard, "model_odd_weight_parameter": repl}
    plan, skipped = FsdpShardSource().collect(gm, examples, 0)
    assert len(plan) == 2, ([c.name for c in plan], skipped)

    assert bind_weights_to_host(gm, list(examples.values()), FsdpShardSource(), min_bytes=0) == 2
    groups = FsdpShardSource().group(gm, {n for n in gm.graph.nodes if host_slot(n) is not None})
    assert [len(g) for g in groups] == [1, 1], "the replicated weight may not share the shard's load"

    assert insert_h2d_loads(gm, FsdpShardSource()) == 2
    assert len(_nodes(gm, H2D_LOAD)) == 2
    assert not _nodes(gm, H2D_LOAD_COALESCED), "two separate submissions, not one merged"


@requires_cuda
def test_binding_after_bucketing_is_reported_rather_than_silently_empty(dist_1rank):
    """The one ordering ``FsdpShardSource`` depends on, and how it fails.

    Binding has to precede bucketing, because bucketing splits offloaded from
    resident gathers on the tag binding sets. Running them the other way round
    leaves ``collect`` looking at coalesced gathers it does not match, every
    weight drops out, and the only symptom is a "nothing to offload" line --
    followed much later by the OOM offload was enabled to prevent. So the
    mismatch is counted and named.
    """
    from magi_compiler.passes.fsdp_overlap import bucket_weight_all_gather_coalesced
    from magi_compiler.passes.weight_offload import bind_weights_to_host
    from magi_compiler.passes.weight_offload.graph.weight_source import FsdpShardSource

    # Two gathers, because a bucket of one is left as its own all_gather.
    gm, param = _twice_gathered_weight_graph(dist_1rank)
    assert bucket_weight_all_gather_coalesced(gm, bucket_size_bytes=0) == 1

    _, skipped = FsdpShardSource().collect(gm, {"model_shared_weight_parameter": param}, 0)
    assert any("must run before bucketing" in why for why in skipped), skipped
    assert bind_weights_to_host(gm, [param], FsdpShardSource(), min_bytes=0) == 0
    assert param._local_tensor.untyped_storage().nbytes() > 0, "nothing may be parked on the failed path"


@requires_cuda
def test_the_pool_remembers_which_parameter_each_shard_came_from(dist_1rank):
    """The placement log is unreadable without it.

    A bucket of MoE experts and a bucket of attention projections are
    indistinguishable as snode ids and behave nothing alike, so "why is that
    load there" can only be answered with the parameter names next to it.
    """
    from magi_compiler.passes.fsdp_overlap import FsdpShardSource
    from magi_compiler.passes.weight_offload import bind_weights_to_host, host_pool

    gm, param = _lowered_weight_graph(dist_1rank, name="L_self_modules_layers_3_modules_mlp_parameters_w1_")
    _park(param._local_tensor, name="layers.3.mlp.w1")
    assert bind_weights_to_host(gm, [param], FsdpShardSource(), min_bytes=0) == 1

    names = [host_pool.name_of(s) for s in range(host_pool.num_bound())]
    assert names == ["layers.3.mlp.w1"], names


@requires_cuda
def test_an_unparked_shard_is_not_offloaded(dist_1rank):
    """Collect never copies a resident shard off the device."""
    from magi_compiler.passes.fsdp_overlap import FsdpShardSource
    from magi_compiler.passes.weight_offload import bind_weights_to_host

    gm, param = _lowered_weight_graph(dist_1rank)
    assert bind_weights_to_host(gm, [param], FsdpShardSource(), min_bytes=0) == 0
    assert param._local_tensor.untyped_storage().nbytes() > 0


@requires_cuda
def test_shard_below_the_size_floor_is_left_unloaded(dist_1rank):
    """A small shard is the worst trade on both axes: fixed DMA overhead dominates
    the transfer, and it frees almost nothing."""
    from magi_compiler.passes.fsdp_overlap import FsdpShardSource
    from magi_compiler.passes.weight_offload import bind_weights_to_host

    gm, param = _lowered_weight_graph(dist_1rank)
    _park(param._local_tensor)
    assert bind_weights_to_host(gm, [param], FsdpShardSource(), min_bytes=4 << 20) == 0


@requires_cuda
def test_promoting_a_slot_switches_the_load_to_a_device_source():
    """Residency is a host-pool state, not a graph change.

    The load node stays exactly where it is and still runs -- it just copies from
    the device now.  That is what lets the placement pass revise the decision
    during scheduling without invalidating the artifact it is scheduling.
    """
    from magi_compiler.passes.weight_offload import host_pool
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    w = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    expected = w.clone()
    slot = _park(w)
    assert host_pool.source(slot).device.type == "cpu"

    assert host_pool.make_resident(slot) == host_pool.slot_bytes(slot)
    assert host_pool.is_resident(slot)
    assert host_pool.source(slot).is_cuda
    assert host_pool.make_resident(slot) == 0, "promotion must be idempotent"

    out = H2D_LOAD(w, slot)
    _WAIT(out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected)


@requires_cuda
def test_a_promoted_load_costs_no_stream_machinery():
    """A resident shard has nothing to overlap, so it pays for nothing.

    The stream hop, the event and the Work exist to hide a PCIe transfer behind
    compute. Once the placement pass has put a shard back on the device there is
    no transfer to hide, and leaving the machinery in place costs two
    cross-stream synchronizations per bucket every single forward.
    """
    import torch._C._distributed_c10d as _c10d

    from magi_compiler.passes.weight_offload import host_pool
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    w = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    expected = w.clone()
    slot = _park(w)
    host_pool.make_resident(slot)

    before = _c10d._get_work_registry_size()
    out = H2D_LOAD(w, slot)
    assert _c10d._get_work_registry_size() == before, "a resident load has nothing to publish a Work for"

    # And the wait stays harmless: it finds no Work and passes the tensor through.
    torch.cuda.synchronize()
    torch.testing.assert_close(_WAIT(out), expected)


@requires_cuda
def test_a_promoted_load_still_returns_its_own_buffer():
    """``h2d_load`` may never return a tensor that aliases its input.

    For a promoted slot the bytes the op is asked for are already in the shard
    it was handed, so handing that shard straight back looks free. It is not:
    Inductor does not allocate a fallback kernel's output, but it does put it in
    the reuse pool, so the next same-sized allocation takes the buffer over and
    the kernel writing into it writes into the weight. That is silent numerical
    corruption on the *second* call, with nothing pointing back here -- hence a
    test on the op's contract rather than on any one graph that trips it.
    """
    from magi_compiler.passes.weight_offload import host_pool
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD, H2D_LOAD_COALESCED

    shards = [torch.randn(128, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    expected = [s.clone() for s in shards]
    slots = [_park(s) for s in shards]
    for slot in slots:
        host_pool.make_resident(slot)

    single = H2D_LOAD(shards[0], slots[0])
    coalesced = H2D_LOAD_COALESCED(shards, slots)
    torch.cuda.synchronize()

    assert single.data_ptr() != shards[0].data_ptr(), "the output must own its storage"
    for out, shard in zip(coalesced, shards):
        assert out.data_ptr() != shard.data_ptr(), "the output must own its storage"

    torch.testing.assert_close(single, expected[0])
    for out, want in zip(coalesced, expected):
        torch.testing.assert_close(out, want)

    # Writing into the outputs, as a reusing kernel would, must leave the
    # weights alone.
    single.zero_()
    for out in coalesced:
        out.zero_()
    torch.cuda.synchronize()
    for shard, want in zip(shards, expected):
        torch.testing.assert_close(shard, want)


@requires_cuda
def test_bound_bytes_excludes_what_was_promoted_back():
    """The reported saving has to be the saving actually realized, or the budget
    the pass spends against is fiction."""
    from magi_compiler.passes.weight_offload import host_pool

    a = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    slots = [_park(a), _park(b)]
    total = host_pool.total_bound_bytes()
    assert host_pool.bound_bytes() == total

    host_pool.make_resident(slots[0])
    assert host_pool.total_bound_bytes() == total, "the shard is still managed, just resident"
    assert host_pool.bound_bytes() == total - host_pool.slot_bytes(slots[0])
    assert host_pool.resident_bytes() == host_pool.slot_bytes(slots[0])


# --------------------------------------------------- host-first materialize


class _Compiled(torch.nn.Module):
    """Stands in for a ``@magi_compile``'d submodule of a bigger model."""

    def __init__(self, rows: int, cols: int):
        super().__init__()
        self.big = torch.nn.Linear(cols, rows, bias=False)
        self.small = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.big(x)


def _shard_on_meta(mesh, rows: int, cols: int) -> torch.nn.Module:
    """A meta-built ``_Compiled`` sharded the way SimpleFSDP shards a model."""
    from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel

    with torch.device("meta"):
        inner = _Compiled(rows, cols).to(torch.bfloat16)
    root = torch.nn.Module()
    root.eager = torch.nn.Linear(4, 4, bias=False, device="meta", dtype=torch.bfloat16)
    root.inner = data_parallel(inner, mesh, mode="fully_shard", ac_mode="full")
    return root


def _raw(module: torch.nn.Module, name: str = "weight"):
    """The registered DTensor, not what the attribute returns.

    SimpleFSDP replaces ``weight`` with a property that runs the all-gather, so
    reading it would both hide the shard under its gathered form and, after the
    handoff, gather storage that no longer exists.  ``named_parameters`` reads
    ``_parameters`` directly for the same reason, which is what host-first sees.
    """
    return module._parameters[name]


def _patched(instance, *, min_shard_mib=0.0):
    from magi_compiler.config import get_compile_config
    from magi_compiler.passes.weight_offload import patch_materialize

    conf = get_compile_config().model_copy(deep=True)
    conf.offload_config.graph_weight_offload = True
    conf.offload_config.offload_min_shard_mib = min_shard_mib
    patch_materialize(instance, conf)
    return instance


class _Tied(torch.nn.Module):
    """Two projections over one weight, the way a model ties its output head to an embedding."""

    def __init__(self, rows: int, cols: int):
        super().__init__()
        self.a = torch.nn.Linear(cols, rows, bias=False)
        self.b = torch.nn.Linear(cols, rows, bias=False)

    def forward(self, x):
        return self.a(x) + self.b(x)


def _tied_on_meta(mesh, rows: int, cols: int) -> torch.nn.Module:
    """A meta-built, sharded ``_Tied`` whose two names really are one Parameter.

    Tied after the wrap because that is the only place it survives one:
    ``data_parallel`` walks the parameters and gives each its own DTensor, so a
    tie made in ``__init__`` comes out the other side as two shards.
    """
    from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel

    with torch.device("meta"):
        inner = _Tied(rows, cols).to(torch.bfloat16)
    root = torch.nn.Module()
    root.inner = data_parallel(inner, mesh, mode="fully_shard", ac_mode="full")
    root.inner.b._parameters["weight"] = _raw(root.inner.a)
    return root


def _tied_across_the_boundary(mesh, rows: int, cols: int, *, eager_first: bool) -> torch.nn.Module:
    """A model whose eager sibling shares one Parameter with the compiled subtree.

    ``_modules`` order decides which side's ``_apply`` reaches the shared object
    first, and the two orders arrive at host-first differently -- the eager side
    going first hands materialize a weight that already has storage, the compiled
    side going first has its host buffer reclaimed afterwards -- so the caller
    picks the order it means to test.
    """
    from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel

    with torch.device("meta"):
        inner = _Compiled(rows, cols).to(torch.bfloat16)
    wrapped = data_parallel(inner, mesh, mode="fully_shard", ac_mode="full")
    eager = torch.nn.Linear(cols, rows, bias=False, device="meta", dtype=torch.bfloat16)

    root = torch.nn.Module()
    if eager_first:
        root.eager, root.inner = eager, wrapped
    else:
        root.inner, root.eager = wrapped, eager
    eager._parameters["weight"] = _raw(wrapped.big)  # one object, both sides of the boundary
    return root


def _assert_both_names_share_one_slot(root: torch.nn.Module) -> None:
    """Every name of a tied weight must reach the same slot, on CUDA, with no bytes."""
    from magi_compiler.passes.weight_offload import host_pool

    slots = set()
    for name in ("a", "b"):
        local = _raw(getattr(root.inner, name))._local_tensor
        assert local.device.type == "cuda", f"{name} must be lowered against a CUDA parameter"
        assert local.untyped_storage().nbytes() == 0, f"{name} must carry no device bytes"
        slot = host_pool.slot_of(local)
        assert slot is not None, f"{name} must resolve to a slot, or its reader gathers the empty stand-in"
        slots.add(slot)
    assert len(slots) == 1, "one shard means one slot, whichever name the graph arrives by"
    assert host_pool.num_bound() == 1, "a tied weight must not be adopted twice"


@requires_cuda
def test_to_empty_materializes_only_the_compiled_subtree_in_host_memory(dist_1rank):
    """Scope comes from the module tree, and it has to.

    Weights outside the compiled subtree are read by the eager prologue --
    embeddings, final projections -- so freeing their storage is an illegal
    access on the very first forward. Patching the compiled module's ``_apply``
    draws that line for free, because ``_apply`` recurses into children and
    nothing else.
    """
    root = _shard_on_meta(dist_1rank, rows=256, cols=64)
    _patched(root.inner)

    root.to_empty(device=torch.device("cuda"))

    assert _raw(root.inner.big)._local_tensor.device.type == "cpu"
    assert _raw(root.inner.big)._local_tensor.is_pinned()
    assert root.eager.weight.device.type == "cuda", "a weight the eager path reads must keep its storage"


@requires_cuda
def test_a_shard_below_the_floor_is_materialized_on_the_device(dist_1rank):
    """The same size floor collect applies, applied early.

    Parking a weight the source will then refuse to load is the one way this can
    corrupt a run, so the two predicates have to ask the same question.
    """
    root = _shard_on_meta(dist_1rank, rows=256, cols=64)
    _patched(root.inner, min_shard_mib=1.0)  # 256x64 bf16 = 32 KiB, well under the floor

    root.to_empty(device=torch.device("cuda"))

    assert _raw(root.inner.big)._local_tensor.device.type == "cuda"
    assert _raw(root.inner.small)._local_tensor.device.type == "cuda"


@requires_cuda
def test_ordinary_device_moves_are_left_alone(dist_1rank):
    """``_apply`` is how a module does everything to its tensors.

    Only ``to_empty`` means "this model has no storage yet"; a dtype cast or a
    device move is a request to put bytes somewhere, and quietly redirecting one
    into host memory would be a very confusing way to lose a model.
    """
    root = _shard_on_meta(dist_1rank, rows=256, cols=64)
    _patched(root.inner)

    root.to_empty(device=torch.device("cuda"))
    before = _raw(root.inner.big)._local_tensor.data_ptr()
    root.inner.float()

    local = _raw(root.inner.big)._local_tensor
    assert local.dtype is torch.float32, "the cast must have happened"
    assert local.data_ptr() != before, "and produced its own storage rather than another reservation"


@requires_cuda
def test_handoff_swaps_in_a_storage_free_device_stand_in(dist_1rank):
    """What the graph is traced against, and what the loader wrote into.

    Between them sits the one moment this can be done: after the checkpoint and
    the post-load hooks, before Dynamo. The bytes must survive it, the parameter
    must come out on CUDA so the graph lowers to CUDA kernels, and the pool must
    be able to find the slot from the tensor the graph will carry.
    """
    from magi_compiler.passes.weight_offload import host_pool
    from magi_compiler.passes.weight_offload.host_first import handoff_if_pending

    root = _shard_on_meta(dist_1rank, rows=256, cols=64)
    _patched(root.inner)
    root.to_empty(device=torch.device("cuda"))

    weight = _raw(root.inner.big)
    expected = torch.arange(weight._local_tensor.numel(), dtype=torch.bfloat16).reshape(weight._local_tensor.shape)
    weight._local_tensor.copy_(expected)  # the loader

    assert handoff_if_pending(root.inner) == 2
    assert handoff_if_pending(root.inner) == 0, "handoff must not run twice"

    local = _raw(root.inner.big)._local_tensor
    assert local.device.type == "cuda", "the graph is lowered against the parameter's device"
    assert local.untyped_storage().nbytes() == 0, "no device bytes until the graph loads them"
    assert local.shape == expected.shape and local.dtype == expected.dtype

    slot = host_pool.slot_of(local)
    assert slot is not None, "the source finds a pre-parked shard by its device tensor's identity"
    torch.testing.assert_close(host_pool.get(slot), expected)
    host_pool.make_resident(slot)
    torch.testing.assert_close(local.cpu(), expected)


@requires_cuda
def test_a_pre_parked_shard_needs_no_binding(dist_1rank):
    """The join between host-first adopt and the source.

    After handoff the shard is CUDA with empty storage.  ``parked_slot`` has
    to recognize it as already adopted -- otherwise the graph would gather
    the empty stand-in.
    """
    from magi_compiler.passes.fsdp_overlap import FsdpShardSource
    from magi_compiler.passes.weight_offload import bind_weights_to_host, host_pool, insert_h2d_loads
    from magi_compiler.passes.weight_offload.host_first import handoff_if_pending
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    root = _shard_on_meta(dist_1rank, rows=256, cols=64)
    _patched(root.inner)
    root.to_empty(device=torch.device("cuda"))
    handoff_if_pending(root.inner)

    param = _raw(root.inner.big)
    gm, _ = _lowered_weight_graph(dist_1rank, rows=256, cols=64)
    # Re-point the graph's placeholder at the handed-off parameter.
    assert bind_weights_to_host(gm, [param], FsdpShardSource(), min_bytes=0) == 1
    assert host_pool.num_bound() == 2, "tagging a pre-parked shard must not adopt it a second time"
    assert insert_h2d_loads(gm, FsdpShardSource()) == 1

    (load,) = _nodes(gm, H2D_LOAD)
    assert load.args[1] == host_pool.slot_of(param._local_tensor)


@requires_cuda
def test_a_tied_shard_lands_on_one_slot(dist_1rank):
    """One Parameter under two names is one shard, and both names have to find it.

    The two halves of that come from opposite directions: ``named_parameters``
    dedupes, so the handoff sees the weight once and mints one slot, and
    ``swap_tensors`` rewrites the object rather than the registration, so that
    single pass lands on both names at no extra cost.
    """
    from magi_compiler.passes.weight_offload.host_first import handoff_if_pending

    root = _tied_on_meta(dist_1rank, rows=256, cols=64)
    _patched(root.inner)
    root.to_empty(device=torch.device("cuda"))
    assert _raw(root.inner.a) is _raw(root.inner.b), "to_empty must leave the tie intact"

    assert handoff_if_pending(root.inner) == 1, "a tied weight is one shard, not two"
    _assert_both_names_share_one_slot(root)


@requires_cuda
def test_a_tied_shard_survives_a_swap_tensors_refusal(dist_1rank):
    """The fallback has to be equivalent to the swap it stands in for.

    ``swap_tensors`` refuses a tensor anything holds a weakref to, and then the
    stand-in goes in by re-registration -- which writes one entry of one module's
    ``_parameters``.  A tied weight has more than one, and a name left behind
    keeps the host tensor and enters compilation as a CPU weight.  Nothing
    downstream can report that: the pool knows the shard by the stand-in it
    minted, so ``parked_slot`` would call the weight never-materialized while it
    sits in a pinned slab.
    """
    import weakref

    from magi_compiler.passes.weight_offload.host_first import handoff_if_pending

    root = _tied_on_meta(dist_1rank, rows=256, cols=64)
    _patched(root.inner)
    root.to_empty(device=torch.device("cuda"))

    # The mere existence of a weakref is what makes swap_tensors refuse, so this
    # reference has to outlive the handoff.
    witness = weakref.ref(_raw(root.inner.a))
    assert witness() is not None

    assert handoff_if_pending(root.inner) == 1
    _assert_both_names_share_one_slot(root)


@requires_cuda
def test_a_weight_tied_outside_the_compiled_subtree_is_never_parked(dist_1rank):
    """The compile boundary is drawn around modules, and a tie crosses it.

    ``_apply`` swaps the object, so the eager sibling's pass over the shared
    weight -- first, in this order -- materializes it for both sides.  Parking it
    anyway would leave the handoff to empty a weight only the eager prologue
    reads, and no graph would put the bytes back: an illegal access in a module
    that was never compiled.
    """
    from magi_compiler.passes.weight_offload import host_pool
    from magi_compiler.passes.weight_offload.host_first import handoff_if_pending

    root = _tied_across_the_boundary(dist_1rank, rows=256, cols=64, eager_first=True)
    _patched(root.inner)
    root.to_empty(device=torch.device("cuda"))

    shared = _raw(root.eager)._local_tensor
    assert shared.device.type == "cuda", "materialize must leave the shared weight where the sibling put it"
    assert shared.untyped_storage().nbytes() > 0, "a weight the eager prologue reads must keep its storage"

    # ``small`` is not shared with anything, so declining ``big`` must not cost it its offload.
    assert handoff_if_pending(root.inner) == 1, "declining one weight must not disable the rest"
    shared = _raw(root.eager)._local_tensor
    assert shared.untyped_storage().nbytes() > 0, "the handoff must not empty a weight it does not own"
    assert host_pool.slot_of(shared) is None, "a weight the compiled graph does not own must never be adopted"


@requires_cuda
def test_a_weight_reclaimed_before_the_handoff_is_not_offloaded(dist_1rank):
    """The other order, where materialize gets there first and the sibling undoes it.

    ``to_empty`` on the eager side gives the shared object device storage again,
    which orphans the host buffer.  What is left is correct and not free -- the
    reservation is never loaded, the device memory is still spent -- so the
    handoff has to find the weight gone rather than reason about a count of zero.
    """
    from magi_compiler.passes.weight_offload import host_pool
    from magi_compiler.passes.weight_offload.host_first import handoff_if_pending

    root = _tied_across_the_boundary(dist_1rank, rows=256, cols=64, eager_first=False)
    _patched(root.inner)
    root.to_empty(device=torch.device("cuda"))

    shared = _raw(root.eager)._local_tensor
    assert shared.device.type == "cuda", "the sibling's to_empty ran last and took the weight back"
    assert shared.untyped_storage().nbytes() > 0, "whoever ran last gave it real storage; it must keep it"

    assert handoff_if_pending(root.inner) == 1, "the weight that was not reclaimed is handed over as usual"
    shared = _raw(root.eager)._local_tensor
    assert shared.untyped_storage().nbytes() > 0, "the handoff must not empty a weight it no longer owns"
    assert host_pool.slot_of(shared) is None, "the orphaned reservation must not become a slot"


# ------------------------------------------------------------ real Inductor


@requires_cuda
def test_storage_freed_shard_survives_a_real_inductor_compile():
    """The premise the whole design rests on.

    The graph keeps carrying the shard -- for its shape, and for the data edge
    back to the parameter -- but there are no bytes behind it until the load puts
    some there.  Nothing downstream of Dynamo is supposed to have an opinion
    about that; Inductor's input handling (guards, ``assert_size_stride``, memory
    planning) is where it would go wrong if anything did.
    """
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    w = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    ref = x @ w.clone()
    slot = _park(w)
    assert w.untyped_storage().nbytes() == 0

    def f(shard, inp):
        return inp @ _WAIT(H2D_LOAD(shard, slot))

    compiled = torch.compile(f, backend="inductor", fullgraph=True)
    out = compiled(w, x)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)

    # The host copy is not consumed by the first load: the artifact is replayable.
    torch.testing.assert_close(compiled(w, x), out)


@requires_cuda
def test_inductor_lowers_the_load_to_a_snode_the_reorder_recognizes():
    """``h2d_load`` has to arrive at the scheduler as its own movable snode.

    If Inductor inlined it, or if ``_is_h2d_load`` failed to recognize the
    lowered form, both reorder passes would quietly do nothing: phase 1 would
    count the transfer as compute that hides an all-gather, and phase 2 would
    find nothing to hoist.  Neither shows up as an error, only as an overlap that
    never materializes -- so it is asserted here, inside a real compile.
    """
    from magi_compiler.passes.fsdp_overlap.reorder import _is_h2d_load
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    seen = {"loads": 0, "compute_misclassified": 0}

    def probe(snodes):
        from magi_compiler.passes.fsdp_overlap.reorder import FsdpOverlapReorder

        for s in snodes:
            if _is_h2d_load(s):
                seen["loads"] += 1
                if FsdpOverlapReorder._is_compute(s):
                    seen["compute_misclassified"] += 1
        return snodes

    w = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    slot = _park(w)

    def f(shard, inp):
        return torch.nn.functional.gelu(inp @ _WAIT(H2D_LOAD(shard, slot)))

    with torch._inductor.config.patch(reorder_for_compute_comm_overlap=True, reorder_for_compute_comm_overlap_passes=[probe]):
        torch.compile(f, backend="inductor", fullgraph=True)(w, x)
    torch.cuda.synchronize()

    assert seen["loads"] == 1, "the load must reach the scheduler as its own snode"
    assert seen["compute_misclassified"] == 0, "a load is a PCIe transfer, never compute that hides a gather"


@requires_cuda
def test_resolve_slot_is_identity_without_a_remap():
    from magi_compiler.passes.weight_offload import host_pool
    from magi_compiler.passes.weight_offload.runtime import slot_remap

    local = torch.randn(32, 16, device="cuda")
    slot = _park(local, name="w")
    assert slot_remap.resolve_slot(slot) == slot
    assert host_pool.find_slot("w", tuple(local.shape), str(local.dtype)) == slot


@requires_cuda
def test_h2d_load_follows_a_baked_to_current_remap():
    """A cached kernel calls h2d_load with another process's slot integers."""
    from magi_compiler.passes.weight_offload.runtime import slot_remap
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    decoy = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    real = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    expected = real.clone()
    decoy_slot = _park(decoy, name="decoy")
    real_slot = _park(real, name="real")
    assert decoy_slot != real_slot

    out = H2D_LOAD(real, decoy_slot)
    _WAIT(out)
    torch.cuda.synchronize()
    with slot_remap.using_slot_remap({decoy_slot: real_slot}):
        remapped = H2D_LOAD(real, decoy_slot)
        _WAIT(remapped)
    torch.cuda.synchronize()

    assert not torch.equal(out, expected), "without a remap the baked slot reads the decoy"
    torch.testing.assert_close(remapped, expected)
    assert slot_remap.resolve_slot(decoy_slot) == decoy_slot, "remap must not leak past the context"
