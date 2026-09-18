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
    from magi_compiler.offload import host_pool

    host_pool.reset()
    yield
    host_pool.reset()


# ---------------------------------------------------------------- host pool


@requires_cuda
def test_bind_frees_device_storage_and_keeps_the_bytes():
    from magi_compiler.offload import host_pool

    local = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
    expected = local.clone()
    nbytes = local.untyped_storage().nbytes()

    (slot,) = host_pool.bind_many([local])

    assert local.untyped_storage().nbytes() == 0, "the device storage must actually be released"
    assert local.shape == expected.shape, "freeing storage must not disturb the shape the graph was traced with"
    assert host_pool.get(slot).is_pinned(), "a pageable source would make the load synchronous"
    torch.testing.assert_close(host_pool.get(slot).cuda(), expected)
    assert host_pool.slot_bytes(slot) == nbytes


@requires_cuda
def test_many_shards_share_one_pinned_slab():
    """One slab per dtype, not one pinned allocation per weight: cudaHostAlloc is a
    driver round trip that stalls every stream, and a model has thousands of these."""
    from magi_compiler.offload import host_pool

    shards = [torch.randn(128, 32, device="cuda", dtype=torch.bfloat16) for _ in range(16)]
    slots = host_pool.bind_many(shards)

    storages = {host_pool.get(s).untyped_storage().data_ptr() for s in slots}
    assert len(storages) == 1, f"expected one backing slab, got {len(storages)}"


@requires_cuda
def test_restore_all_puts_the_shards_back():
    from magi_compiler.offload import host_pool

    local = torch.randn(64, 8, device="cuda")
    expected = local.clone()
    host_pool.bind_many([local])
    assert local.untyped_storage().nbytes() == 0

    host_pool.restore_all()
    torch.testing.assert_close(local, expected)


@requires_cuda
def test_bandwidth_probe_is_plausible():
    from magi_compiler.offload import host_pool

    bw = host_pool.h2d_bandwidth_bytes_per_ns()
    # bytes/ns == GB/s.  Anything outside this is a broken probe, not a slow bus.
    assert 1.0 < bw < 500.0, f"implausible H2D bandwidth {bw} GB/s"


def test_bandwidth_override_is_taken_verbatim():
    """GB/s and bytes/ns are the same number; a stray unit conversion here would
    mis-size every overlap window by 1e9."""
    from magi_compiler.offload import host_pool

    assert host_pool.h2d_bandwidth_bytes_per_ns(override_gbps=42.0) == 42.0


# ------------------------------------------------------------------- the op


@requires_cuda
def test_h2d_load_returns_the_offloaded_bytes():
    from magi_compiler.offload import host_pool
    from magi_compiler.offload.h2d_op import H2D_LOAD

    local = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
    expected = local.clone()
    (slot,) = host_pool.bind_many([local])

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
    from magi_compiler.offload import host_pool
    from magi_compiler.offload.h2d_op import H2D_LOAD, h2d_stream

    local = torch.randn(4096, 1024, device="cuda", dtype=torch.bfloat16)
    (slot,) = host_pool.bind_many([local])
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

    from magi_compiler.offload import host_pool
    from magi_compiler.offload.h2d_op import H2D_LOAD

    local = torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16)
    expected = local.clone()
    (slot,) = host_pool.bind_many([local])

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
    from magi_compiler.offload.h2d_op import H2D_LOAD

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
    from magi_compiler.offload.h2d_op import H2D_LOAD
    from magi_compiler.passes.fsdp_overlap import bind_weights_to_host, insert_h2d_loads, is_host_offloaded

    gm, param = _lowered_weight_graph(dist_1rank)
    assert bind_weights_to_host(gm, [param], min_shard_bytes=0) == 1
    assert insert_h2d_loads(gm) == 1

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
    from magi_compiler.offload.h2d_op import H2D_LOAD
    from magi_compiler.passes.fsdp_overlap import bind_weights_to_host, insert_h2d_loads

    gm, param = _lowered_weight_graph(dist_1rank, forward_dtype=torch.float32)
    assert bind_weights_to_host(gm, [param], min_shard_bytes=0) == 1
    assert insert_h2d_loads(gm) == 1

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

    The first compile frees the shards; every later graph still has to load them
    back.  Treating "already bound" as "nothing to do" leaves the second graph
    all-gathering freed storage -- which surfaces as an illegal memory access
    inside NCCL, on every rank, with nothing pointing back here.
    """
    from magi_compiler.offload.h2d_op import H2D_LOAD
    from magi_compiler.passes.fsdp_overlap import bind_weights_to_host, insert_h2d_loads

    gm1, param = _lowered_weight_graph(dist_1rank)
    assert bind_weights_to_host(gm1, [param], min_shard_bytes=0) == 1
    assert insert_h2d_loads(gm1) == 1
    assert param._local_tensor.untyped_storage().nbytes() == 0

    # A second graph over the same live parameter, as a second shape would give.
    gm2, _ = _lowered_weight_graph(dist_1rank)
    assert bind_weights_to_host(gm2, [param], min_shard_bytes=0) == 1, "the bound shard is still a candidate"
    assert insert_h2d_loads(gm2) == 1, "the second graph needs its own load"

    slots = {n.args[1] for n in _nodes(gm2, H2D_LOAD)}
    assert slots == {n.args[1] for n in _nodes(gm1, H2D_LOAD)}, "both graphs must read the same slot"


@requires_cuda
def test_the_pool_remembers_which_parameter_each_shard_came_from(dist_1rank):
    """The placement log is unreadable without it.

    A bucket of MoE experts and a bucket of attention projections are
    indistinguishable as snode ids and behave nothing alike, so "why is that
    load there" can only be answered with the parameter names next to it.
    """
    from magi_compiler.offload import host_pool
    from magi_compiler.passes.fsdp_overlap import bind_weights_to_host

    gm, param = _lowered_weight_graph(dist_1rank, name="L_self_modules_layers_3_modules_mlp_parameters_w1_")
    assert bind_weights_to_host(gm, [param], min_shard_bytes=0) == 1

    names = [host_pool.name_of(s) for s in range(host_pool.num_bound())]
    assert names == ["layers.3.mlp.w1"], names


@requires_cuda
def test_shard_below_the_size_floor_is_left_resident(dist_1rank):
    """A small shard is the worst trade on both axes: fixed DMA overhead dominates
    the transfer, and it frees almost nothing."""
    from magi_compiler.passes.fsdp_overlap import bind_weights_to_host

    gm, param = _lowered_weight_graph(dist_1rank)
    assert bind_weights_to_host(gm, [param], min_shard_bytes=4 << 20) == 0
    assert param._local_tensor.untyped_storage().nbytes() > 0


@requires_cuda
def test_promoting_a_slot_switches_the_load_to_a_device_source():
    """Residency is a host-pool state, not a graph change.

    The load node stays exactly where it is and still runs -- it just copies from
    the device now.  That is what lets the placement pass revise the decision
    during scheduling without invalidating the artifact it is scheduling.
    """
    from magi_compiler.offload import host_pool
    from magi_compiler.offload.h2d_op import H2D_LOAD

    w = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    expected = w.clone()
    (slot,) = host_pool.bind_many([w])
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
def test_bound_bytes_excludes_what_was_promoted_back():
    """The reported saving has to be the saving actually realized, or the budget
    the pass spends against is fiction."""
    from magi_compiler.offload import host_pool

    a = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    slots = host_pool.bind_many([a, b])
    total = host_pool.total_bound_bytes()
    assert host_pool.bound_bytes() == total

    host_pool.make_resident(slots[0])
    assert host_pool.total_bound_bytes() == total, "the shard is still managed, just resident"
    assert host_pool.bound_bytes() == total - host_pool.slot_bytes(slots[0])
    assert host_pool.resident_bytes() == host_pool.slot_bytes(slots[0])


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
    from magi_compiler.offload import host_pool
    from magi_compiler.offload.h2d_op import H2D_LOAD

    w = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    ref = x @ w.clone()
    (slot,) = host_pool.bind_many([w])
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
    from magi_compiler.offload import host_pool
    from magi_compiler.offload.h2d_op import H2D_LOAD
    from magi_compiler.passes.fsdp_overlap.reorder import _is_h2d_load

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
    (slot,) = host_pool.bind_many([w])

    def f(shard, inp):
        return torch.nn.functional.gelu(inp @ _WAIT(H2D_LOAD(shard, slot)))

    with torch._inductor.config.patch(reorder_for_compute_comm_overlap=True, reorder_for_compute_comm_overlap_passes=[probe]):
        torch.compile(f, backend="inductor", fullgraph=True)(w, x)
    torch.cuda.synchronize()

    assert seen["loads"] == 1, "the load must reach the scheduler as its own snode"
    assert seen["compute_misclassified"] == 0, "a load is a PCIe transfer, never compute that hides a gather"
