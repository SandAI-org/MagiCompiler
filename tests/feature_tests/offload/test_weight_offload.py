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

"""Compile-time host offload of FSDP shards and ordinary model parameters.

Three layers, tested separately because they fail in different ways:

* ``host_weights`` -- the bytes actually leave the device, and come back identical.
* ``magi::h2d_load`` -- the copy runs on its own stream and is published as a
  ``Work``, so ``wait_tensor`` is what makes it visible.  An op that silently
  synchronized instead would pass every correctness check and overlap nothing.
* the FX pass -- the load lands ABOVE the dtype cast and BELOW nothing, and the
  gather ends up reading the loaded copy rather than the freed shard.

FSDP cases use a per-test 1-rank process group and device mesh. Plain-parameter
cases explicitly require no process group. Runtime and integration cases need CUDA.
"""

import os

import pytest
import torch
import torch.fx as fx

from magi_compiler.passes.weight_offload import PlainParamSource, bind_host_slots_to_graph, insert_h2d_loads

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_ALL_GATHER_OP = torch.ops._c10d_functional.all_gather_into_tensor.default
_WAIT_TENSOR_OP = torch.ops._c10d_functional.wait_tensor.default
_TO_COPY = torch.ops.aten._to_copy.default


@pytest.fixture
def dist_1rank():
    """A per-test process group, so later plain-parameter cases stay standalone."""
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29663")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    created = False
    if not dist.is_initialized():
        dist.init_process_group("gloo")
        created = True
    try:
        torch.cuda.set_device(0)
        from torch.distributed.device_mesh import init_device_mesh

        yield init_device_mesh("cuda", (1,))
    finally:
        if created:
            dist.destroy_process_group()


@pytest.fixture(autouse=True)
def reset_host_weight_store():
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    get_global_host_weight_store().reset()
    yield
    get_global_host_weight_store().reset()


def _park_weight_in_host_store(tensor: torch.Tensor, name: str = "") -> int:
    """Put ``tensor`` in the host pool the production way: reserve, fill, empty, adopt."""
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    host_buffer = get_global_host_weight_store().reserve_host_buffer(tuple(tensor.shape), tensor.dtype, parameter_name=name)
    host_buffer.copy_(tensor.detach())
    tensor.untyped_storage().resize_(0)
    return get_global_host_weight_store().bind_host_buffer(host_buffer, tensor, parameter_name=name)


# ------------------------------------------------ cross-rank agreement


def _scheduler_passes_worker(rank: int, world_size: int, port: str, out_dir: str) -> None:
    """Build one rank's scheduler passes, with a load on rank 0 only, and record their names."""
    import json

    import torch.distributed as dist

    from magi_compiler.config import FSDPConfig, OffloadConfig
    from magi_compiler.magi_backend.weight_transfer_pipeline import build_transfer_overlap_passes

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        passes = build_transfer_overlap_passes(
            fsdp_config=FSDPConfig(),
            offload_config=OffloadConfig(graph_weight_offload=True, offload_h2d_bandwidth_gbps=20.0),
            loads_inserted=rank == 0,
            default_passes=None,
        )
        names = [p if isinstance(p, str) else type(p).__name__ for p in passes or ()]
        with open(os.path.join(out_dir, f"{rank}.json"), "w") as f:
            json.dump(names, f)
    finally:
        dist.destroy_process_group()


def test_every_rank_profiles_once_any_rank_inserted_a_load(tmp_path):
    """Synchronized profiling is a collective: a rank without loads must still install it."""
    import json
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = str(s.getsockname()[1])
    mp.spawn(_scheduler_passes_worker, args=(2, port, str(tmp_path)), nprocs=2, join=True)

    with_loads, without_loads = (json.loads((tmp_path / f"{rank}.json").read_text()) for rank in range(2))
    assert with_loads[:1] == without_loads[:1] == ["SnodeCostProfile"]
    assert "H2dLoadReorder" in with_loads and "H2dLoadReorder" not in without_loads


@requires_cuda
@pytest.mark.parametrize("ndim", [1, 2])
def test_alignment_resolves_mesh_groups_without_swallowing_errors(dist_1rank, monkeypatch, ndim):
    """Even an empty plan must resolve its DTensor groups; lookup failure must not bypass voting."""
    from collections import Counter

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate

    from magi_compiler.passes.weight_offload.graph import bind

    mesh = init_device_mesh(device_type="cpu", mesh_shape=(1,) * ndim)
    weight = torch.nn.Parameter(DTensor.from_local(torch.ones(4), device_mesh=mesh, placements=[Replicate()] * ndim))
    examples = {"weight": weight, "plain": torch.nn.Parameter(torch.ones(4)), "activation": torch.ones(4)}
    group = mesh.get_group(mesh_dim=0)
    assert bind._collect_candidate_sync_groups(candidates=[], example_value_by_placeholder_name=examples) == {id(group): group}
    assert (
        bind._intersect_rank_candidates(candidates=[], example_value_by_placeholder_name=examples, skip_counts=Counter()) == []
    )

    failure = RuntimeError("mesh group lookup failed")

    def fail_group(mesh_dim=None):
        raise failure

    monkeypatch.setattr(mesh, "get_group", fail_group)
    with pytest.raises(RuntimeError, match="mesh group lookup failed") as caught:
        bind._intersect_rank_candidates(candidates=[], example_value_by_placeholder_name=examples, skip_counts=Counter())
    assert caught.value is failure


def _make_offload_candidate(parameter_name, weight_node_name, process_group, shape=(4, 4)):
    from magi_compiler.passes.weight_offload.graph.weight_source import OffloadCandidate

    weight_node = type("Node", (), {"name": weight_node_name})()
    device_standin = torch.empty(shape, dtype=torch.float32)
    return OffloadCandidate(
        weight_node=weight_node,
        device_standin=device_standin,
        parameter_name=parameter_name,
        weight_bytes=device_standin.numel() * device_standin.element_size(),
        host_slot=0,
        process_group=process_group,
    )


def _stub_rank_exchange(monkeypatch, *, group_size, replies):
    """``replies[group]`` is the list-of-ranks payload ``all_gather_object`` should write."""
    from magi_compiler.passes.weight_offload.graph import bind

    monkeypatch.setattr(bind.dist, "is_available", lambda: True)
    monkeypatch.setattr(bind.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(bind.dist, "get_world_size", lambda group=None: group_size)

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
    shared = _make_offload_candidate("w_shared", "tl_shared", group)
    mine_only = _make_offload_candidate("w_mine", "tl_mine", group)
    plan = [shared, mine_only]
    mine_keys = [bind._candidate_key(c) for c in plan]
    peer_keys = [mine_keys[0]]
    _stub_rank_exchange(monkeypatch, group_size=2, replies={group: [mine_keys, peer_keys]})

    skipped = Counter()
    kept = bind._intersect_rank_candidates(plan, {}, skipped)
    assert [c.parameter_name for c in kept] == ["w_shared"]
    assert skipped["not planned by every rank in the shard group"] == 1


def test_align_votes_per_group_not_on_world(monkeypatch):
    """Expert-mesh disagreement must not take a dense-mesh weight with it."""
    from collections import Counter

    from magi_compiler.passes.weight_offload.graph import bind

    dense, expert = object(), object()
    dense_w = _make_offload_candidate("dense.w", "tl_dense", dense)
    expert_w = _make_offload_candidate("expert.w", "tl_expert", expert)
    plan = [dense_w, expert_w]
    dense_key = bind._candidate_key(dense_w)
    expert_key = bind._candidate_key(expert_w)
    seen = _stub_rank_exchange(
        monkeypatch, group_size=2, replies={dense: [[dense_key], [dense_key]], expert: [[expert_key], []]}
    )

    skipped = Counter()
    kept = bind._intersect_rank_candidates(plan, {}, skipped)
    assert [c.parameter_name for c in kept] == ["dense.w"]
    assert skipped["not planned by every rank in the shard group"] == 1
    assert set(seen) == {dense, expert}, "each mesh must vote on its own group, never WORLD"


def test_align_still_enters_a_group_found_only_on_graph_inputs(monkeypatch):
    """A rank that collected nothing on a mesh must still join that mesh's vote."""
    from collections import Counter

    from magi_compiler.passes.weight_offload.graph import bind

    group = object()
    fake_param = type("P", (), {"_spec": type("S", (), {"mesh": object()})()})()
    monkeypatch.setattr(bind, "parameter_mesh_group", lambda obj: group if obj is fake_param else None)

    seen = _stub_rank_exchange(monkeypatch, group_size=2, replies={group: [[], [("peer", "h", (4, 4), "torch.float32")]]})
    skipped = Counter()
    kept = bind._intersect_rank_candidates([], {"w": fake_param}, skipped)
    assert kept == []
    assert seen == [group]


def test_align_propagates_a_failed_group_collective(monkeypatch):
    """A failed collective propagates: a rank that dropped the group and moved on would desync its peers."""
    from collections import Counter

    from magi_compiler.passes.weight_offload.graph import bind

    dense, expert = object(), object()
    dense_w = _make_offload_candidate("dense.w", "tl_dense", dense)
    expert_w = _make_offload_candidate("expert.w", "tl_expert", expert)
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

    with pytest.raises(RuntimeError, match="expert group is down"):
        bind._intersect_rank_candidates(plan, {}, Counter())


# ---------------------------------------------------------------- host pool


def test_pinned_reservation_failure_is_not_downgraded(monkeypatch):
    """A failed pinned reservation must raise instead of returning pageable storage."""
    from magi_compiler.passes.weight_offload.runtime.host_memory_pool import HostMemPool
    from magi_compiler.passes.weight_offload.runtime.host_weights import HostWeightStore

    original_empty = torch.empty
    failure = RuntimeError("pinned allocation failed")

    def fail_pinned(*args, pin_memory=False, **kwargs):
        if pin_memory:
            raise failure
        return original_empty(*args, pin_memory=False, **kwargs)

    monkeypatch.setattr(torch, "empty", fail_pinned)
    store = HostWeightStore(pool=HostMemPool(pinned=True, slab_bytes=1024))
    with pytest.raises(RuntimeError, match="pinned allocation failed") as caught:
        store.reserve_host_buffer(shape=(8,), dtype=torch.float32, parameter_name="weight")
    assert caught.value is failure

    pageable = HostMemPool(pinned=False, slab_bytes=1024).alloc(shape=(8,), dtype=torch.float32)
    assert not pageable.is_pinned()
    assert pageable.shape == (8,)


@requires_cuda
def test_many_shards_share_one_pinned_slab():
    """One slab per dtype, not one pinned allocation per weight: cudaHostAlloc is a
    driver round trip that stalls every stream, and a model has thousands of these."""
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    slots = [_park_weight_in_host_store(torch.randn(128, 32, device="cuda", dtype=torch.bfloat16)) for _ in range(16)]

    storages = {get_global_host_weight_store().host_buffer(s).untyped_storage().data_ptr() for s in slots}
    assert len(storages) == 1, f"expected one backing slab, got {len(storages)}"


@requires_cuda
def test_restore_all_puts_the_shards_back():
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    local = torch.randn(64, 8, device="cuda")
    expected = local.clone()
    slot = _park_weight_in_host_store(local)
    assert local.untyped_storage().nbytes() == 0

    get_global_host_weight_store().restore_all_weights()
    torch.testing.assert_close(local, expected)
    assert get_global_host_weight_store().is_resident(
        slot
    ), "restore_all_weights must mark the slot resident so source() agrees"
    assert get_global_host_weight_store().copy_source(slot).data_ptr() == local.data_ptr()


@requires_cuda
def test_reserve_and_adopt_park_a_shard_that_was_never_on_the_device():
    """The only way into the pool, and the reason peak memory drops.

    ``reserve`` hands out the host buffer up front, the loader fills it, and
    ``adopt`` pairs it with a device tensor that carries nothing but shape --
    so the shard is offloaded without ever having been resident.
    """
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    host = get_global_host_weight_store().reserve_host_buffer((256, 64), torch.bfloat16, parameter_name="layers.0.w")
    assert host.is_pinned(), "a pageable source would make the load synchronous"
    host.copy_(torch.arange(256 * 64, dtype=torch.bfloat16).reshape(256, 64))

    stand_in = torch.empty(256, 64, device="cuda", dtype=torch.bfloat16)
    stand_in.untyped_storage().resize_(0)
    slot = get_global_host_weight_store().bind_host_buffer(host, stand_in, parameter_name="layers.0.w")

    assert (
        get_global_host_weight_store().slot_for_tensor(stand_in) == slot
    ), "the source looks the slot up by the device tensor's identity"
    assert get_global_host_weight_store().parameter_name(slot) == "layers.0.w"
    assert (
        get_global_host_weight_store().slot_bytes(slot) == 256 * 64 * 2
    ), "a storage-free tensor still has to report its real size"
    assert get_global_host_weight_store().copy_source(slot).data_ptr() == host.data_ptr()

    assert get_global_host_weight_store().bind_host_buffer(host, stand_in, parameter_name="layers.0.w") == slot
    assert get_global_host_weight_store().weight_count() == 1, "adopting the same stand-in must be idempotent"

    # And the shard can still be handed back, which is what the placement pass
    # does for a load it cannot hide.
    get_global_host_weight_store().make_weight_resident(slot)
    torch.testing.assert_close(stand_in, host.cuda())


@requires_cuda
def test_adopt_refuses_a_device_tensor_that_still_holds_bytes():
    """Adopting one would leak it: the pool would believe the shard is parked
    while its device storage stays allocated for the life of the process."""
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    host = get_global_host_weight_store().reserve_host_buffer((32, 8), torch.bfloat16)
    with pytest.raises(ValueError, match="no storage behind it"):
        get_global_host_weight_store().bind_host_buffer(host, torch.empty(32, 8, device="cuda", dtype=torch.bfloat16))


@requires_cuda
def test_weights_without_h2d_load_are_restored_to_device():
    """Restore only this graph's weights without H2D loads; keep weights with loads offloaded.
    Other models' weights must stay parked until their graphs compile.
    """
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    loaded = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    orphan = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    other_model = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    expected = orphan.clone()
    loaded_slot = _park_weight_in_host_store(loaded, name="kept")
    orphan_slot = _park_weight_in_host_store(orphan, name="orphan")
    other_slot = _park_weight_in_host_store(other_model, name="other model")
    get_global_host_weight_store().mark_slot_with_h2d_load(loaded_slot)

    assert get_global_host_weight_store().restore_weights_without_h2d_load([loaded_slot, orphan_slot]) == ["orphan"]
    assert not get_global_host_weight_store().is_resident(loaded_slot), "a shard the graph loads must stay in host memory"
    assert get_global_host_weight_store().is_resident(orphan_slot)
    assert not get_global_host_weight_store().is_resident(other_slot), "another model's weight is not this graph's to restore"
    torch.testing.assert_close(orphan, expected)
    assert (
        get_global_host_weight_store().restore_weights_without_h2d_load([orphan_slot]) == []
    ), "already-restored shards must not be reported twice"


@requires_cuda
def test_make_resident_many_restores_values_and_is_idempotent():
    """Batch promotion restores every slot and counts its bytes only once."""
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    shards = [torch.randn(32, 16, device="cuda") for _ in range(4)]
    expected = [s.clone() for s in shards]
    slots = [_park_weight_in_host_store(s) for s in shards]
    assert get_global_host_weight_store().make_weights_resident(slots) == sum(s.numel() * s.element_size() for s in expected)
    assert get_global_host_weight_store().make_weights_resident(slots) == 0
    for slot, exp in zip(slots, expected):
        assert get_global_host_weight_store().is_resident(slot)
        torch.testing.assert_close(get_global_host_weight_store().copy_source(slot), exp)


@requires_cuda
def test_release_resident_many_falls_back_to_host_and_can_be_promoted_again():
    """A released slot must stop reading its freed device copy, and promotion must restore it."""
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    shard = torch.randn(32, 16, device="cuda")
    expected = shard.clone()
    slot = _park_weight_in_host_store(shard)
    get_global_host_weight_store().make_weight_resident(slot)

    assert get_global_host_weight_store().release_resident_weights([slot]) == expected.numel() * expected.element_size()
    assert shard.untyped_storage().nbytes() == 0
    assert not get_global_host_weight_store().is_resident(slot), "h2d_load would D2D-copy from freed storage"
    assert get_global_host_weight_store().copy_source(slot).device.type == "cpu"
    assert get_global_host_weight_store().release_resident_weights([slot]) == 0

    get_global_host_weight_store().make_weights_resident([slot])
    assert get_global_host_weight_store().is_resident(slot)
    torch.testing.assert_close(shard, expected)


@requires_cuda
def test_bandwidth_probe_is_plausible_and_reset_clears_its_cache():
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    bw = get_global_host_weight_store().h2d_bandwidth_bytes_per_ns()
    # bytes/ns == GB/s.  Anything outside this is a broken probe, not a slow bus.
    assert 1.0 < bw < 500.0, f"implausible H2D bandwidth {bw} GB/s"

    assert get_global_host_weight_store()._bandwidth_bytes_per_ns == bw
    get_global_host_weight_store().reset()
    assert get_global_host_weight_store()._bandwidth_bytes_per_ns is None


@pytest.mark.parametrize(
    "setup_fails", [pytest.param(False, marks=requires_cuda, id="probed"), pytest.param(True, id="setup-failed")]
)
def test_bandwidth_probe_issues_no_collective(monkeypatch, setup_fails):
    """Only ranks that compile a load probe, so a probe collective would hang the ranks that do not."""
    import torch.distributed as dist

    from magi_compiler.passes.weight_offload.runtime import bandwidth

    collectives = []
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    for name in ("barrier", "all_reduce", "broadcast", "all_gather_object"):
        monkeypatch.setattr(dist, name, lambda *args, _name=name, **kwargs: collectives.append(_name))
    if setup_fails:
        monkeypatch.setattr(bandwidth.torch.cuda, "is_available", lambda: True)

        def failing_empty(*args, **kwargs):
            raise RuntimeError("pinned allocation failed")

        monkeypatch.setattr(bandwidth.torch, "empty", failing_empty)

    assert bandwidth.measure_h2d_bandwidth_bytes_per_ns() > 0
    assert collectives == []


# ------------------------------------------------------------------- the op


@requires_cuda
def test_h2d_load_runs_off_the_compute_stream(monkeypatch):
    """Observe the stream submitting the real copy, without timing assumptions."""
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD, get_h2d_stream

    local = torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16)
    slot = _park_weight_in_host_store(local)
    compute_stream = torch.cuda.current_stream()
    load_stream = get_h2d_stream()
    copies = []
    original_copy = torch.Tensor.copy_

    def observe_copy(dst, src, non_blocking=False):
        if dst.is_cuda and src.device.type == "cpu":
            copies.append((torch.cuda.current_stream(), non_blocking))
        return original_copy(dst, src, non_blocking=non_blocking)

    # Keep the actual copy and stream handling intact; observe at submission.
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "copy_", observe_copy)
        out = H2D_LOAD(local, slot)
    _WAIT_TENSOR_OP(out)
    torch.cuda.synchronize()

    assert load_stream != compute_stream, "H2D must have a stream of its own"
    assert copies == [(load_stream, True)], "the copy must be asynchronous and submitted on the load stream"


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
    slot = _park_weight_in_host_store(local)

    before = _c10d._get_work_registry_size()
    out = H2D_LOAD(local, slot)
    assert out.is_cuda and out.shape == expected.shape and out.dtype == expected.dtype
    assert _c10d._get_work_registry_size() == before + 1, "h2d_load must publish its event as a c10d Work"
    _WAIT_TENSOR_OP(out)
    # No device sync: wait_tensor alone must have ordered the compute stream
    # behind the copy, so reading `out` here is already safe.
    torch.testing.assert_close(out.float().cpu(), expected.float().cpu())


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
    sharded = torch.nn.Parameter(distribute_tensor(full, mesh, [Shard(0)]), requires_grad=False)
    replicated = distribute_tensor(full, mesh, [Replicate()])

    graph = fx.Graph()
    weight_node = graph.placeholder(name)
    weight_node.meta["example_value"] = sharded
    redistribute_node = graph.call_method(
        "redistribute", (weight_node,), {"placements": [Replicate()], "forward_dtype": forward_dtype, "backward_dtype": None}
    )
    redistribute_node.meta["example_value"] = replicated
    local_node = graph.call_method("to_local", (redistribute_node,), {"grad_placements": [Partial()]})
    local_node.meta["example_value"] = replicated._local_tensor
    graph.output((local_node,))
    graph_module = fx.GraphModule(torch.nn.Module(), graph)

    assert lower_prim_redistribute_to_collectives(graph_module) == 1
    return graph_module, sharded


def _find_nodes_by_target(graph_module, target):
    return [n for n in graph_module.graph.nodes if n.op == "call_function" and n.target is target]


@requires_cuda
def test_bind_and_insert_puts_the_load_between_the_shard_and_the_gather(dist_1rank):
    from magi_compiler.passes.weight_offload import (
        FsdpShardSource,
        bind_host_slots_to_graph,
        get_global_host_weight_store,
        insert_h2d_loads,
        is_host_offloaded,
    )
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    gm, param = _lowered_weight_graph(dist_1rank)
    slot = _park_weight_in_host_store(param._local_tensor, name="layers.3.mlp.w1")
    assert bind_host_slots_to_graph(gm, [param], FsdpShardSource()) == 1
    assert get_global_host_weight_store().parameter_name(slot) == "layers.3.mlp.w1"
    assert insert_h2d_loads(gm) == 1

    loads = _find_nodes_by_target(gm, H2D_LOAD)
    assert len(loads) == 1
    load = loads[0]
    assert load.args[0].op == "call_method" and load.args[0].target == "to_local"

    # The gather reads the loaded copy, through the load's wait -- not the shard,
    # whose storage no longer exists.
    (ag,) = _find_nodes_by_target(gm, _ALL_GATHER_OP)
    assert is_host_offloaded(ag)
    reachable = {load, *(n for n in gm.graph.nodes if n.op == "call_function" and n.target is _WAIT_TENSOR_OP)}
    assert any(a in reachable for a in ag.all_input_nodes)
    assert param._local_tensor.untyped_storage().nbytes() == 0


@requires_cuda
def test_load_sits_above_the_dtype_cast(dist_1rank):
    """Load the stored dtype before applying the graph's device-side cast."""
    from magi_compiler.passes.weight_offload import FsdpShardSource, bind_host_slots_to_graph, insert_h2d_loads
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    gm, param = _lowered_weight_graph(dist_1rank, forward_dtype=torch.float32)
    _park_weight_in_host_store(param._local_tensor)
    assert bind_host_slots_to_graph(gm, [param], FsdpShardSource()) == 1
    assert insert_h2d_loads(gm) == 1

    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    (load,) = _find_nodes_by_target(gm, H2D_LOAD)
    (cast,) = _find_nodes_by_target(gm, _TO_COPY)
    assert order[load] < order[cast]
    # and the cast reads the loaded shard, not the freed one
    assert any(isinstance(a, fx.Node) and a.target is _WAIT_TENSOR_OP for a in cast.all_input_nodes)


@requires_cuda
def test_second_graph_over_the_same_parameters_still_gets_its_loads(dist_1rank):
    """Every graph over the same parameters must insert loads while reusing adopted slots."""
    from magi_compiler.passes.weight_offload import FsdpShardSource, bind_host_slots_to_graph, insert_h2d_loads
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    gm1, param = _lowered_weight_graph(dist_1rank)
    _park_weight_in_host_store(param._local_tensor)
    assert bind_host_slots_to_graph(gm1, [param], FsdpShardSource()) == 1
    assert insert_h2d_loads(gm1) == 1
    assert param._local_tensor.untyped_storage().nbytes() == 0

    # A second graph over the same live parameter, as a second shape would give.
    gm2, _ = _lowered_weight_graph(dist_1rank)
    assert bind_host_slots_to_graph(gm2, [param], FsdpShardSource()) == 1, "the adopted shard is still a candidate"
    assert insert_h2d_loads(gm2) == 1, "the second graph needs its own load"

    slots = {n.args[1] for n in _find_nodes_by_target(gm2, H2D_LOAD)}
    assert slots == {n.args[1] for n in _find_nodes_by_target(gm1, H2D_LOAD)}, "both graphs must read the same slot"


def _twice_gathered_weight_graph(mesh, rows=8, cols=4):
    """One parameter, two independent gathers -- as a weight read by two branches."""
    from torch.distributed.tensor import Partial, Replicate, Shard, distribute_tensor

    from magi_compiler.passes.fsdp_overlap import lower_prim_redistribute_to_collectives

    full = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    sharded = torch.nn.Parameter(distribute_tensor(full, mesh, [Shard(0)]), requires_grad=False)
    replicated = distribute_tensor(full, mesh, [Replicate()])

    graph = fx.Graph()
    weight_node = graph.placeholder("model_shared_weight_parameter")
    weight_node.meta["example_value"] = sharded
    outs = []
    for _ in range(2):
        redistribute_node = graph.call_method(
            "redistribute", (weight_node,), {"placements": [Replicate()], "forward_dtype": None, "backward_dtype": None}
        )
        redistribute_node.meta["example_value"] = replicated
        local_node = graph.call_method("to_local", (redistribute_node,), {"grad_placements": [Partial()]})
        local_node.meta["example_value"] = replicated._local_tensor
        outs.append(local_node)
    graph.output(tuple(outs))
    graph_module = fx.GraphModule(torch.nn.Module(), graph)

    assert lower_prim_redistribute_to_collectives(graph_module) == 2
    return graph_module, sharded


@requires_cuda
def test_a_weight_two_gathers_read_gets_a_load_for_each(dist_1rank):
    """Distinct to_local holders of one shard must each load the shared host slot."""
    from magi_compiler.passes.weight_offload import (
        FsdpShardSource,
        bind_host_slots_to_graph,
        get_global_host_weight_store,
        insert_h2d_loads,
    )
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    gm, param = _twice_gathered_weight_graph(dist_1rank)
    expected = param._local_tensor.clone()
    _park_weight_in_host_store(param._local_tensor)

    assert bind_host_slots_to_graph(gm, [param], FsdpShardSource()) == 2, "both gathers need a candidate"
    assert get_global_host_weight_store().weight_count() == 1, "one shard, one set of host bytes"
    torch.testing.assert_close(get_global_host_weight_store().host_buffer(0).cuda(), expected)

    assert insert_h2d_loads(gm) == 2
    loads = _find_nodes_by_target(gm, H2D_LOAD)
    assert {n.args[1] for n in loads} == {0}, "both loads read the one slot"

    # No gather may still be reading the holder whose storage is gone.
    for ag in _find_nodes_by_target(gm, _ALL_GATHER_OP):
        assert any(isinstance(a, fx.Node) and a.target is _WAIT_TENSOR_OP for a in ag.all_input_nodes), ag.format_node()


# Named to match what the passes look for: through torch 2.9 Dynamo captures a
# DTensor ``redistribute`` / ``to_local`` in an on-the-fly function of this name,
# and the predicates key off ``__name__``.
def prim_redistribute(x):
    return x


def prim_to_local(x):
    return x


def _replicated_weight_graph(mesh, rows=9, cols=64):
    """A replicated DTensor whose prim redistribute/to_local pair survives lowering.
    Offload must discover its weight holder without an all-gather to key off."""
    from torch.distributed.tensor import Replicate, distribute_tensor

    from magi_compiler.passes.fsdp_overlap import lower_prim_redistribute_to_collectives

    full = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    replicated = torch.nn.Parameter(distribute_tensor(full, mesh, [Replicate()]), requires_grad=False)

    graph = fx.Graph()
    weight_node = graph.placeholder("model_odd_weight_parameter")
    weight_node.meta["example_value"] = replicated
    redistribute_node = graph.call_function(prim_redistribute, (weight_node,))
    redistribute_node.meta["example_value"] = replicated
    local_node = graph.call_function(prim_to_local, (redistribute_node,))
    local_node.meta["example_value"] = replicated._local_tensor
    reader = graph.call_function(torch.ops.aten.relu.default, (local_node,))
    reader.meta["example_value"] = replicated._local_tensor
    graph.output((reader,))
    graph_module = fx.GraphModule(torch.nn.Module(), graph)

    assert lower_prim_redistribute_to_collectives(graph_module) == 0, "a replicated weight has nothing to lower"
    return graph_module, replicated


def _mixed_weight_graph(mesh, rows=8, cols=64, odd_rows=9):
    """One sharded weight and one replicated one, as a real model has them."""
    from torch.distributed.tensor import Partial, Replicate, Shard, distribute_tensor

    from magi_compiler.passes.fsdp_overlap import lower_prim_redistribute_to_collectives

    full = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    shard = torch.nn.Parameter(distribute_tensor(full, mesh, [Shard(0)]), requires_grad=False)
    gathered = distribute_tensor(full, mesh, [Replicate()])
    repl = torch.nn.Parameter(
        distribute_tensor(torch.randn(odd_rows, cols, device="cuda", dtype=torch.bfloat16), mesh, [Replicate()]),
        requires_grad=False,
    )

    graph = fx.Graph()
    weight_node = graph.placeholder("model_fc1_weight_parameter")
    weight_node.meta["example_value"] = shard
    redistribute_node = graph.call_method(
        "redistribute", (weight_node,), {"placements": [Replicate()], "forward_dtype": None, "backward_dtype": None}
    )
    redistribute_node.meta["example_value"] = gathered
    local_node = graph.call_method("to_local", (redistribute_node,), {"grad_placements": [Partial()]})
    local_node.meta["example_value"] = gathered._local_tensor

    ungathered_weight_node = graph.placeholder("model_odd_weight_parameter")
    ungathered_weight_node.meta["example_value"] = repl
    ungathered_redistribute = graph.call_function(prim_redistribute, (ungathered_weight_node,))
    ungathered_redistribute.meta["example_value"] = repl
    ungathered_local = graph.call_function(prim_to_local, (ungathered_redistribute,))
    ungathered_local.meta["example_value"] = repl._local_tensor
    ungathered_reader = graph.call_function(torch.ops.aten.relu.default, (ungathered_local,))
    ungathered_reader.meta["example_value"] = repl._local_tensor

    graph.output((local_node, ungathered_reader))
    graph_module = fx.GraphModule(torch.nn.Module(), graph)

    assert lower_prim_redistribute_to_collectives(graph_module) == 1, "only the sharded weight has anything to lower"
    return graph_module, shard, repl


@requires_cuda
def test_a_replicated_weight_is_offloaded_even_though_nothing_gathers_it(dist_1rank):
    """The weights that cost the most per GPU were the ones being skipped.

    SimpleFSDP replicates a shard whose dim0 does not divide its mesh, so every
    rank holds the whole tensor rather than 1/N -- world_size times the device
    memory of the same weight sharded. Keying offload off all-gathers missed
    exactly those, and the safety net then dutifully put back anything host-first
    had parked for them.
    """
    from magi_compiler.passes.weight_offload import (
        FsdpShardSource,
        bind_host_slots_to_graph,
        get_global_host_weight_store,
        insert_h2d_loads,
    )
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    gm, param = _replicated_weight_graph(dist_1rank)
    expected = param._local_tensor.clone()
    _park_weight_in_host_store(param._local_tensor)

    assert bind_host_slots_to_graph(gm, [param], FsdpShardSource()) == 1
    assert param._local_tensor.untyped_storage().nbytes() == 0, "the full copy must leave the device"
    torch.testing.assert_close(get_global_host_weight_store().host_buffer(0).cuda(), expected)

    assert insert_h2d_loads(gm) == 1
    (load,) = _find_nodes_by_target(gm, H2D_LOAD)
    # The load reads the to_local, and the reader reads the load's wait -- not
    # the to_local, whose storage no longer exists.
    assert getattr(load.args[0].target, "__name__", "") == "prim_to_local"
    (reader,) = _find_nodes_by_target(gm, torch.ops.aten.relu.default)
    assert any(isinstance(a, fx.Node) and a.target is _WAIT_TENSOR_OP for a in reader.all_input_nodes)


@requires_cuda
def test_an_ungathered_weight_gets_a_load_to_itself(dist_1rank):
    """A replicated weight has no gather, but still one load, like every other weight.

    Its load sits in front of its own first reader -- a redistribute, not a
    placeholder -- and its readers read the loaded copy.
    """
    from magi_compiler.passes.weight_offload import FsdpShardSource, bind_host_slots_to_graph, insert_h2d_loads
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    gm, shard, repl = _mixed_weight_graph(dist_1rank)
    _park_weight_in_host_store(shard._local_tensor)
    _park_weight_in_host_store(repl._local_tensor)

    examples = {"model_fc1_weight_parameter": shard, "model_odd_weight_parameter": repl}
    plan, skipped = FsdpShardSource().collect_candidates(gm, examples)
    assert len(plan) == 2, ([c.parameter_name for c in plan], skipped)

    assert bind_host_slots_to_graph(gm, list(examples.values()), FsdpShardSource()) == 2
    assert insert_h2d_loads(gm) == 2
    loads = _find_nodes_by_target(gm, H2D_LOAD)
    assert len(loads) == 2
    assert {load.args[0].name for load in loads} == {c.weight_node.name for c in plan}


@requires_cuda
def test_an_unparked_shard_is_not_offloaded(dist_1rank):
    """Collect never copies a resident shard off the device."""
    from magi_compiler.passes.weight_offload import FsdpShardSource, bind_host_slots_to_graph

    gm, param = _lowered_weight_graph(dist_1rank)
    assert bind_host_slots_to_graph(gm, [param], FsdpShardSource()) == 0
    assert param._local_tensor.untyped_storage().nbytes() > 0


@requires_cuda
def test_a_dtensor_activation_named_like_a_weight_is_not_offloaded(dist_1rank):
    """Weight-ness comes from the live object, not the placeholder name: an activation changes every forward."""
    from magi_compiler.passes.weight_offload import FsdpShardSource

    gm, param = _replicated_weight_graph(dist_1rank)
    activation = param.detach()
    assert not isinstance(activation, torch.nn.Parameter)
    _park_weight_in_host_store(activation._local_tensor)

    plan, skipped = FsdpShardSource().collect_candidates(gm, {"model_odd_weight_parameter": activation})
    assert plan == [] and skipped["graph input is not a parameter"] == 1, skipped


@requires_cuda
def test_promotion_switches_the_source_and_updates_the_byte_accounting():
    """Promotion changes the source and savings, but not the managed byte total."""
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    shards = [torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    expected = shards[0].clone()
    slots = [_park_weight_in_host_store(shard) for shard in shards]
    slot = slots[0]
    total = get_global_host_weight_store().total_weight_bytes()
    assert get_global_host_weight_store().offloaded_bytes() == total
    assert get_global_host_weight_store().resident_bytes() == 0
    assert get_global_host_weight_store().copy_source(slot).device.type == "cpu"

    assert get_global_host_weight_store().make_weight_resident(slot) == get_global_host_weight_store().slot_bytes(slot)
    assert get_global_host_weight_store().is_resident(slot)
    assert get_global_host_weight_store().copy_source(slot).is_cuda
    torch.testing.assert_close(get_global_host_weight_store().copy_source(slot), expected)
    assert not get_global_host_weight_store().is_resident(slots[1])
    assert get_global_host_weight_store().total_weight_bytes() == total
    assert get_global_host_weight_store().offloaded_bytes() == total - get_global_host_weight_store().slot_bytes(slot)
    assert get_global_host_weight_store().resident_bytes() == get_global_host_weight_store().slot_bytes(slot)
    assert get_global_host_weight_store().make_weight_resident(slot) == 0, "promotion must be idempotent"


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
    import torch._C._distributed_c10d as _c10d

    from magi_compiler.passes.weight_offload import get_global_host_weight_store
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    shards = [torch.randn(128, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    expected = [s.clone() for s in shards]
    slots = [_park_weight_in_host_store(s) for s in shards]
    for slot in slots:
        get_global_host_weight_store().make_weight_resident(slot)

    before = _c10d._get_work_registry_size()
    outs = [H2D_LOAD(shard, slot) for shard, slot in zip(shards, slots)]
    assert _c10d._get_work_registry_size() == before, "resident loads must not register asynchronous Work"
    outs = [_WAIT_TENSOR_OP(out) for out in outs]
    torch.cuda.synchronize()

    for out, shard, want in zip(outs, shards, expected):
        assert out.data_ptr() != shard.data_ptr(), "the output must own its storage"
        torch.testing.assert_close(out, want)

    # Writing into the outputs, as a reusing kernel would, must leave the
    # weights alone.
    for out in outs:
        out.zero_()
    torch.cuda.synchronize()
    for shard, want in zip(shards, expected):
        torch.testing.assert_close(shard, want)


# --------------------------------------------------- host-first materialize


class _CompiledWeightModule(torch.nn.Module):
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
        inner = _CompiledWeightModule(rows, cols).to(torch.bfloat16)
    root = torch.nn.Module()
    root.eager = torch.nn.Linear(4, 4, bias=False, device="meta", dtype=torch.bfloat16)
    root.inner = data_parallel(inner, mesh, mode="fully_shard", ac_mode="full")
    return root


def _registered_parameter(module: torch.nn.Module, name: str = "weight"):
    """The registered DTensor, not what the attribute returns.

    SimpleFSDP replaces ``weight`` with a property that runs the all-gather, so
    reading it would both hide the shard under its gathered form and, after the
    handoff, gather storage that no longer exists.  ``named_parameters`` reads
    ``_parameters`` directly for the same reason, which is what host-first sees.
    """
    return module._parameters[name]


def _install_host_first_hooks(instance, *, min_shard_mib=0.0):
    from magi_compiler.passes.weight_offload import install_host_first_hooks

    install_host_first_hooks(instance=instance, min_weight_bytes=int(min_shard_mib * 1024 * 1024))
    return instance


def test_host_first_uses_the_current_class_apply_after_installation():
    """A sharding owner may replace the module class after hooks are installed."""

    class UpdatedLinear(torch.nn.Linear):
        def _apply(self, fn, recurse=True):
            self.class_apply_calls += 1
            return super()._apply(fn, recurse=recurse)

    module = _install_host_first_hooks(torch.nn.Linear(4, 4, bias=False))
    module.__class__ = UpdatedLinear
    module.class_apply_calls = 0
    module.double()

    assert module.class_apply_calls == 1
    assert module.weight.dtype is torch.float64


@requires_cuda
def test_host_first_installation_and_parking_decision_are_not_repeated():
    """Reinstallation must preserve the threshold, and a later CUDA move must not reselect weights."""
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    module = _install_host_first_hooks(torch.nn.Linear(4, 4, bias=False), min_shard_mib=1.0)
    _install_host_first_hooks(module, min_shard_mib=0.0)
    module.cuda()
    assert get_global_host_weight_store().slot_for_tensor(module.weight) is None

    module.cpu()
    module.weight = torch.nn.Parameter(torch.ones(1024, 1024))
    module.cuda()
    assert get_global_host_weight_store().slot_for_tensor(module.weight) is None
    torch.testing.assert_close(module.weight, torch.ones_like(module.weight))


@requires_cuda
def test_empty_handoff_completes_after_all_parked_weights_are_reclaimed():
    """An external reader may reclaim every parked weight; handoff must still finish exactly once."""
    from magi_compiler.passes.weight_offload.weight_lifecycle import (
        get_offload_state,
        handoff_host_weights_if_pending,
        is_handoff_complete,
    )

    module = _install_host_first_hooks(torch.nn.Linear(4, 4, bias=False, device="meta"))
    assert not is_handoff_complete(module)
    module.to_empty(device="cuda")
    assert get_offload_state(module).handoff_status == "pending"

    # Bypass the instance hook, as an eager sibling does for a shared parameter.
    type(module)._apply(module, lambda tensor: torch.empty_like(tensor, device="cuda"))
    assert handoff_host_weights_if_pending(module) == 0
    assert is_handoff_complete(module)
    assert get_offload_state(module).host_slots == ()
    assert handoff_host_weights_if_pending(module) == 0
    assert module.weight.untyped_storage().nbytes() > 0


class _TiedWeightModule(torch.nn.Module):
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
        inner = _TiedWeightModule(rows, cols).to(torch.bfloat16)
    root = torch.nn.Module()
    root.inner = data_parallel(inner, mesh, mode="fully_shard", ac_mode="full")
    root.inner.b._parameters["weight"] = _registered_parameter(root.inner.a)
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
        inner = _CompiledWeightModule(rows, cols).to(torch.bfloat16)
    wrapped = data_parallel(inner, mesh, mode="fully_shard", ac_mode="full")
    eager = torch.nn.Linear(cols, rows, bias=False, device="meta", dtype=torch.bfloat16)

    root = torch.nn.Module()
    if eager_first:
        root.eager, root.inner = eager, wrapped
    else:
        root.inner, root.eager = wrapped, eager
    eager._parameters["weight"] = _registered_parameter(wrapped.big)  # one object, both sides of the boundary
    return root


def _assert_both_names_share_one_slot(root: torch.nn.Module) -> None:
    """Every name of a tied weight must reach the same slot, on CUDA, with no bytes."""
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    slots = set()
    for name in ("a", "b"):
        local = _registered_parameter(getattr(root.inner, name))._local_tensor
        assert local.device.type == "cuda", f"{name} must be lowered against a CUDA parameter"
        assert local.untyped_storage().nbytes() == 0, f"{name} must carry no device bytes"
        slot = get_global_host_weight_store().slot_for_tensor(local)
        assert slot is not None, f"{name} must resolve to a slot, or its reader gathers the empty stand-in"
        slots.add(slot)
    assert len(slots) == 1, "one shard means one slot, whichever name the graph arrives by"
    assert get_global_host_weight_store().weight_count() == 1, "a tied weight must not be adopted twice"


@requires_cuda
def test_a_shard_below_the_floor_is_materialized_on_the_device(dist_1rank):
    """The size floor is applied once, here: a shard below it keeps its device
    storage, so collect never finds a host slot for it."""
    root = _shard_on_meta(dist_1rank, rows=256, cols=64)
    _install_host_first_hooks(root.inner, min_shard_mib=1.0)  # 256x64 bf16 = 32 KiB, well under the floor

    root.to_empty(device=torch.device("cuda"))

    assert _registered_parameter(root.inner.big)._local_tensor.device.type == "cuda"
    assert _registered_parameter(root.inner.small)._local_tensor.device.type == "cuda"


@requires_cuda
def test_ordinary_device_moves_are_left_alone(dist_1rank):
    """A dtype cast after host-first materialization must not reserve another host buffer."""
    root = _shard_on_meta(dist_1rank, rows=256, cols=64)
    _install_host_first_hooks(root.inner)

    root.to_empty(device=torch.device("cuda"))
    before = _registered_parameter(root.inner.big)._local_tensor.data_ptr()
    root.inner.float()

    local = _registered_parameter(root.inner.big)._local_tensor
    assert local.dtype is torch.float32, "the cast must have happened"
    assert local.data_ptr() != before, "and produced its own storage rather than another reservation"


@requires_cuda
def test_device_moves_of_a_sharded_module_change_residency_not_tensors(dist_1rank):
    """torch's ``_apply`` swaps every DTensor parameter, which a traced module refuses.

    A move of a module with parked shards must complete the handoff and leave the
    stand-ins where the graph expects them instead.
    """
    root = _shard_on_meta(dist_1rank, rows=256, cols=64)
    _install_host_first_hooks(root.inner)
    root.to_empty(device=torch.device("cuda"))

    root.inner.to("cpu")

    local = _registered_parameter(root.inner.big)._local_tensor
    assert local.device.type == "cuda" and local.untyped_storage().nbytes() == 0


@requires_cuda
def test_a_host_built_module_parks_on_its_first_move_and_moves_the_rest_with_torch():
    """An unsharded model built and loaded on the host never calls ``to_empty``.

    Its first move to CUDA must park the offloadable weights instead of moving them,
    while small weights and buffers still follow every move, both ways. That move
    may come under inference mode; placement still promotes the stand-ins outside it.
    """
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    module = _install_host_first_hooks(
        _CompiledWeightModule(rows=256, cols=64), min_shard_mib=0.01
    )  # big: 64 KiB, small: 64 B
    module.register_buffer("scale", torch.ones(4))
    expected = module.big.weight.detach().clone()

    with torch.inference_mode():
        module.to("cuda")
    slot = get_global_host_weight_store().slot_for_tensor(module.big.weight)
    assert slot is not None and module.big.weight.untyped_storage().nbytes() == 0
    torch.testing.assert_close(get_global_host_weight_store().host_buffer(slot), expected)
    assert module.small.weight.device.type == "cuda" and module.scale.device.type == "cuda"

    get_global_host_weight_store().make_weight_resident(slot)
    module.to("cpu")
    assert not get_global_host_weight_store().is_resident(slot) and module.big.weight.device.type == "cuda"
    assert module.small.weight.device.type == "cpu" and module.scale.device.type == "cpu"

    module.to("cuda")
    assert get_global_host_weight_store().is_resident(slot) and module.small.weight.device.type == "cuda"
    torch.testing.assert_close(module.big.weight.detach().cpu(), expected)


def test_graph_weight_offload_refuses_aot_before_parking_anything():
    """AOT replay cannot remap baked host slots, so the combination must fail at compile setup."""
    from magi_compiler import magi_compile
    from magi_compiler.config import CompileMode
    from magi_compiler.passes.weight_offload.weight_lifecycle import get_offload_state

    def patch(cfg):
        cfg.compile_mode = CompileMode.MAGI_COMPILE
        cfg.aot = True
        cfg.offload_config.graph_weight_offload = True
        return cfg

    module = _CompiledWeightModule(rows=256, cols=64)
    with pytest.raises(ValueError, match="does not support aot=True"):
        magi_compile(module, config_patch=patch, dynamic_arg_dims={"x": 0})
    assert get_offload_state(module) is None


def test_graph_weight_offload_refuses_split_graphs_before_parking_anything():
    """Each piecewise subgraph spends the residency budget again, so a split compile must fail at setup."""
    from magi_compiler import magi_compile
    from magi_compiler.config import CompileMode
    from magi_compiler.passes.weight_offload.weight_lifecycle import get_offload_state

    def patch(cfg):
        cfg.compile_mode = CompileMode.MAGI_COMPILE
        cfg.offload_config.graph_weight_offload = True
        cfg.disable_graph_split = False
        return cfg

    module = _CompiledWeightModule(rows=256, cols=64)
    with pytest.raises(ValueError, match="requires disable_graph_split=True"):
        magi_compile(module, config_patch=patch, dynamic_arg_dims={"x": 0})
    assert get_offload_state(module) is None


@requires_cuda
@pytest.mark.usefixtures("no_process_group")
def test_a_training_graph_over_an_offloaded_weight_fails_to_compile():
    """A grad-requiring offloaded weight must not compile into a graph that silently drops its gradient."""
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    w = torch.nn.Parameter(torch.randn(512, 256, device="cuda", dtype=torch.bfloat16))
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    slot = _park_weight_in_host_store(w, name="w")

    def f(weight, inp):
        return inp @ _WAIT_TENSOR_OP(H2D_LOAD(weight, slot))

    with pytest.raises(Exception, match="h2d_load has no backward"):
        torch.compile(f, backend="inductor", fullgraph=True)(w, x)


@requires_cuda
@pytest.mark.parametrize(
    "cast",
    [lambda m: m.half(), lambda m: m.to(torch.bfloat16), lambda m: m.to("cuda", torch.float16)],
    ids=["half", "to-dtype", "to-device-and-dtype"],
)
def test_a_dtype_cast_after_handoff_is_refused_before_it_reads_a_stand_in(cast):
    """``model.cuda().half()`` must fail loudly rather than cast a storage-free stand-in."""
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    module = _install_host_first_hooks(_CompiledWeightModule(rows=256, cols=64), min_shard_mib=0.01)
    expected = module.big.weight.detach().clone()
    module.cuda()
    slot = get_global_host_weight_store().slot_for_tensor(module.big.weight)
    assert slot is not None

    with pytest.raises(RuntimeError, match="already handed to host memory"):
        cast(module)

    assert module.big.weight.dtype is torch.float32 and module.small.weight.dtype is torch.float32
    assert get_global_host_weight_store().slot_for_tensor(module.big.weight) == slot
    torch.testing.assert_close(get_global_host_weight_store().host_buffer(slot), expected)
    module.float()  # a cast that changes no stand-in's dtype stays allowed


@requires_cuda
def test_device_moves_free_and_repromote_only_this_instances_promoted_shards():
    """To the host frees this instance's promoted copies; back promotes exactly those.

    The pool is shared by every compiled model in the process, so another instance's
    residency must not move, and a shard the graph loads must stay in host memory.
    """
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    def build():
        with torch.device("meta"):
            module = _CompiledWeightModule(rows=256, cols=64)
        _install_host_first_hooks(module)
        module.to_empty(device=torch.device("cuda"))
        return module.to("cuda")

    model_a, model_b = build(), build()
    resident_slot_a, offloaded_slot_a, resident_slot_b = (
        get_global_host_weight_store().slot_for_tensor(w)
        for w in (model_a.big.weight, model_a.small.weight, model_b.big.weight)
    )
    expected = get_global_host_weight_store().host_buffer(resident_slot_a).clone()
    get_global_host_weight_store().make_weights_resident([resident_slot_a, resident_slot_b])

    model_a.to("cpu")
    assert model_a.big.weight.untyped_storage().nbytes() == 0 and not get_global_host_weight_store().is_resident(
        resident_slot_a
    )
    assert get_global_host_weight_store().is_resident(resident_slot_b), "another model's residency must not move"

    model_a.to("cuda")
    assert get_global_host_weight_store().is_resident(resident_slot_a)
    assert not get_global_host_weight_store().is_resident(offloaded_slot_a)
    torch.testing.assert_close(model_a.big.weight.detach().cpu(), expected)


@requires_cuda
def test_host_first_materializes_hands_off_and_binds_the_compiled_subtree(dist_1rank):
    """Handoff must preserve loaded bytes and register CUDA stand-ins before tracing.
    Binding must resolve the slots through the parameters installed on the module."""
    from magi_compiler.passes.weight_offload import (
        FsdpShardSource,
        bind_host_slots_to_graph,
        get_global_host_weight_store,
        insert_h2d_loads,
    )
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD
    from magi_compiler.passes.weight_offload.weight_lifecycle import handoff_host_weights_if_pending

    root = _shard_on_meta(dist_1rank, rows=256, cols=64)
    _install_host_first_hooks(root.inner)
    root.to_empty(device=torch.device("cuda"))

    assert _registered_parameter(root.inner.big)._local_tensor.device.type == "cpu"
    assert _registered_parameter(root.inner.big)._local_tensor.is_pinned()
    assert root.eager.weight.device.type == "cuda", "the eager sibling must keep its device storage"

    weight = _registered_parameter(root.inner.big)
    expected = torch.arange(weight._local_tensor.numel(), dtype=torch.bfloat16).reshape(weight._local_tensor.shape)
    weight._local_tensor.copy_(expected)  # the loader

    assert handoff_host_weights_if_pending(root.inner) == 2
    assert handoff_host_weights_if_pending(root.inner) == 0, "handoff must not run twice"

    local = _registered_parameter(root.inner.big)._local_tensor
    assert local.device.type == "cuda", "the graph is lowered against the parameter's device"
    assert local.untyped_storage().nbytes() == 0, "no device bytes until the graph loads them"
    assert local.shape == expected.shape and local.dtype == expected.dtype

    slot = get_global_host_weight_store().slot_for_tensor(local)
    assert slot is not None, "the source finds a pre-parked shard by its device tensor's identity"
    torch.testing.assert_close(get_global_host_weight_store().host_buffer(slot), expected)
    # Bind while still offloaded, before the restoration check changes the source.
    param = _registered_parameter(root.inner.big)
    gm, _ = _lowered_weight_graph(dist_1rank, rows=256, cols=64)
    assert bind_host_slots_to_graph(gm, [param], FsdpShardSource()) == 1
    assert get_global_host_weight_store().weight_count() == 2, "binding must reuse the slots adopted by handoff"
    assert insert_h2d_loads(gm) == 1
    (load,) = _find_nodes_by_target(gm, H2D_LOAD)
    assert load.args[1] == slot

    get_global_host_weight_store().make_weight_resident(slot)
    torch.testing.assert_close(local.cpu(), expected)


@requires_cuda
def test_a_tied_shard_lands_on_one_slot(dist_1rank):
    """One Parameter under two names is one shard, and both names have to find it.

    The two halves of that come from opposite directions: ``named_parameters``
    dedupes, so the handoff sees the weight once and mints one slot, and
    ``swap_tensors`` rewrites the object rather than the registration, so that
    single pass lands on both names at no extra cost.
    """
    from magi_compiler.passes.weight_offload.weight_lifecycle import handoff_host_weights_if_pending

    root = _tied_on_meta(dist_1rank, rows=256, cols=64)
    _install_host_first_hooks(root.inner)
    root.to_empty(device=torch.device("cuda"))
    assert _registered_parameter(root.inner.a) is _registered_parameter(root.inner.b), "to_empty must leave the tie intact"

    assert handoff_host_weights_if_pending(root.inner) == 1, "a tied weight is one shard, not two"
    _assert_both_names_share_one_slot(root)


@requires_cuda
def test_handoff_fails_rather_than_re_register_a_weight_it_cannot_swap(dist_1rank):
    """A weight swap_tensors refuses must stop the handoff: re-registering it would break its identity."""
    import weakref

    from magi_compiler.passes.weight_offload.weight_lifecycle import handoff_host_weights_if_pending

    root = _tied_on_meta(dist_1rank, rows=256, cols=64)
    _install_host_first_hooks(root.inner)
    root.to_empty(device=torch.device("cuda"))
    # swap_tensors refuses a tensor that a live weakref points at.
    witness = weakref.ref(_registered_parameter(root.inner.a))
    with pytest.raises(RuntimeError, match="weakref"):
        handoff_host_weights_if_pending(root.inner)
    assert witness() is _registered_parameter(root.inner.a), "the registered Parameter keeps its identity"


@requires_cuda
@pytest.mark.parametrize("eager_first", [True, False], ids=["eager-first", "compiled-first"])
def test_handoff_preserves_a_weight_shared_with_the_eager_subtree(dist_1rank, eager_first):
    """A shared weight stays on the device in either module traversal order.

    Eager-first materializes it before host-first can park it; compiled-first
    reserves host bytes which eager then replaces before handoff.
    """
    from magi_compiler.passes.weight_offload import get_global_host_weight_store
    from magi_compiler.passes.weight_offload.weight_lifecycle import handoff_host_weights_if_pending

    root = _tied_across_the_boundary(dist_1rank, rows=256, cols=64, eager_first=eager_first)
    _install_host_first_hooks(root.inner)
    root.to_empty(device=torch.device("cuda"))

    shared = _registered_parameter(root.eager)._local_tensor
    assert shared.device.type == "cuda", "materialize must leave the shared weight where the sibling put it"
    assert shared.untyped_storage().nbytes() > 0, "a weight the eager prologue reads must keep its storage"

    # ``small`` is not shared with anything, so declining ``big`` must not cost it its offload.
    assert handoff_host_weights_if_pending(root.inner) == 1, "declining one weight must not disable the rest"
    shared = _registered_parameter(root.eager)._local_tensor
    assert shared.untyped_storage().nbytes() > 0, "the handoff must not empty a weight it does not own"
    assert (
        get_global_host_weight_store().slot_for_tensor(shared) is None
    ), "a weight the compiled graph does not own must never be adopted"


# ------------------------------------------------------------ real Inductor


@requires_cuda
def test_storage_freed_shard_survives_a_real_inductor_compile(dist_1rank):
    """An empty CUDA stand-in must survive Inductor input handling and repeated execution.
    Each load returns a separate filled buffer; it does not fill the stand-in."""
    from magi_compiler.passes.snode_utils import is_compute
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD
    from magi_compiler.passes.weight_offload.schedule.h2d_snode import is_h2d_load

    seen = {"loads": 0, "compute_misclassified": 0}

    def probe(snodes):
        for snode in snodes:
            if is_h2d_load(snode):
                seen["loads"] += 1
                seen["compute_misclassified"] += int(is_compute(snode))
        return snodes

    w = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.gelu(x @ w.clone())
    slot = _park_weight_in_host_store(w)
    assert w.untyped_storage().nbytes() == 0

    def f(shard, inp):
        return torch.nn.functional.gelu(inp @ _WAIT_TENSOR_OP(H2D_LOAD(shard, slot)))

    # A cache hit would bypass the scheduler probe. Inductor's overlap scheduler
    # queries get_rank(), so this test explicitly requests dist_1rank.
    with torch._inductor.config.patch(
        force_disable_caches=True, reorder_for_compute_comm_overlap=True, reorder_for_compute_comm_overlap_passes=[probe]
    ):
        compiled = torch.compile(f, backend="inductor", fullgraph=True)
        out = compiled(w, x)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)

    # The host copy is not consumed by the first load: the artifact is replayable.
    torch.testing.assert_close(compiled(w, x), out)

    assert seen["loads"] == 1, "the load must reach the scheduler as its own snode"
    assert seen["compute_misclassified"] == 0, "a load must not be counted as compute"


@requires_cuda
def test_h2d_load_follows_a_baked_to_current_remap():
    """A cached kernel calls h2d_load with another process's slot integers."""
    from magi_compiler.passes.weight_offload.runtime import slot_remap
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    decoy = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    real = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    expected = real.clone()
    decoy_slot = _park_weight_in_host_store(decoy, name="decoy")
    real_slot = _park_weight_in_host_store(real, name="real")
    assert decoy_slot != real_slot
    assert slot_remap.resolve_runtime_host_slot(decoy_slot) == decoy_slot
    assert slot_remap.resolve_runtime_host_slot(real_slot) == real_slot

    out = H2D_LOAD(real, decoy_slot)
    _WAIT_TENSOR_OP(out)
    torch.cuda.synchronize()
    with slot_remap.using_slot_remap({decoy_slot: real_slot}):
        remapped = H2D_LOAD(real, decoy_slot)
        _WAIT_TENSOR_OP(remapped)
    torch.cuda.synchronize()

    assert not torch.equal(out, expected), "without a remap the baked slot reads the decoy"
    torch.testing.assert_close(remapped, expected)
    assert slot_remap.resolve_runtime_host_slot(decoy_slot) == decoy_slot, "remap must not leak past the context"


# --------------------------------------------------- plain parameters (no FSDP)


@pytest.fixture
def no_process_group():
    import torch.distributed as dist

    assert not dist.is_initialized(), "plain-parameter offload must work without a process group"


def _linear_graph(*params):
    """``x`` and some weights in, one matmul per weight out.

    Shaped like what Dynamo hands the backend: every parameter is lifted to a
    placeholder at the top, which is exactly why grouping cannot go by
    declaration order.
    """
    graph = fx.Graph()
    x = graph.placeholder("l_x_")
    x.meta["example_value"] = torch.empty(8, params[0].shape[0], device="meta", dtype=params[0].dtype)

    holders = []
    for i, p in enumerate(params):
        n = graph.placeholder(f"L_self_modules_layers_{i}_parameters_weight_")
        n.meta["example_value"] = p
        holders.append(n)

    out = x
    for n, p in zip(holders, params):
        out = graph.call_function(torch.matmul, (out, n))
        out.meta["example_value"] = torch.empty(8, p.shape[1], device="meta", dtype=p.dtype)
    graph.output((out,))
    gm = fx.GraphModule(torch.nn.Module(), graph)
    return gm, [None, *params]  # example_inputs: x has no live parameter behind it


@requires_cuda
@pytest.mark.usefixtures("no_process_group")
def test_a_plain_parameter_is_offloaded_without_any_fsdp():
    from magi_compiler.passes.weight_offload import get_global_host_weight_store
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    w = torch.nn.Parameter(torch.randn(256, 256, device="cuda", dtype=torch.bfloat16))
    _park_weight_in_host_store(w, name="layers.0.weight")
    gm, examples = _linear_graph(w)

    source = PlainParamSource()
    assert bind_host_slots_to_graph(gm, examples, source) == 1
    assert w.untyped_storage().nbytes() == 0, "the stand-in must stay empty"
    assert insert_h2d_loads(gm) == 1

    (load,) = _find_nodes_by_target(gm, H2D_LOAD)
    assert load.args[1] == 0, "the load carries the host-pool slot"
    assert get_global_host_weight_store().parameter_name(0) == "layers.0.weight"

    # The matmul reads the loaded copy, not the freed placeholder.
    (mm,) = _find_nodes_by_target(gm, torch.matmul)
    assert any(getattr(a, "target", None) is _WAIT_TENSOR_OP for a in mm.all_input_nodes)


@requires_cuda
@pytest.mark.usefixtures("no_process_group")
def test_only_parameters_are_taken_not_activations():
    """Every graph input looks alike; only the ones that are the same every
    forward are worth moving."""
    w = torch.nn.Parameter(torch.randn(256, 256, device="cuda", dtype=torch.bfloat16))
    _park_weight_in_host_store(w)
    gm, examples = _linear_graph(w)
    # x's "live value" is a plain tensor, not a Parameter.
    examples[0] = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)

    assert bind_host_slots_to_graph(gm, examples, PlainParamSource()) == 1


@requires_cuda
@pytest.mark.usefixtures("no_process_group")
def test_each_load_sits_at_its_own_first_reader_not_at_the_declaration():
    """Dynamo lifts every parameter to the top of the graph, so their placeholder
    order says nothing about when they run.  A load placed by it would land layer
    N's weight before layer 0 runs."""
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    params = [torch.nn.Parameter(torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)) for _ in range(4)]
    for p in params:
        _park_weight_in_host_store(p)
    gm, examples = _linear_graph(*params)

    assert bind_host_slots_to_graph(gm, examples, PlainParamSource()) == 4
    assert insert_h2d_loads(gm) == 4

    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    loads = sorted(_find_nodes_by_target(gm, H2D_LOAD), key=order.__getitem__)
    assert len(loads) == len(params)
    layers = [int(load.args[0].name.split("layers_")[1].split("_")[0]) for load in loads]
    assert layers == list(range(len(params))), f"loads out of layer order: {layers}"
    first_readers = [min(next(iter(load.users)).users, key=order.__getitem__) for load in loads]
    for k in range(1, len(loads)):
        assert order[loads[k]] > order[first_readers[k - 1]], "a later layer's load went ahead of an earlier layer's compute"


@requires_cuda
@pytest.mark.usefixtures("no_process_group")
def test_unparked_plain_parameter_is_not_offloaded():
    small = torch.nn.Parameter(torch.randn(16, 16, device="cuda", dtype=torch.bfloat16))
    gm, examples = _linear_graph(small)

    assert bind_host_slots_to_graph(gm, examples, PlainParamSource()) == 0
    assert small.untyped_storage().nbytes() > 0


@requires_cuda
@pytest.mark.usefixtures("no_process_group")
def test_offload_survives_a_real_inductor_compile_without_fsdp():
    """The whole point of the decoupling, end to end in one process.

    No process group is initialized here -- if anything in the offload path still
    reached for a mesh, a rank or a collective, this would raise rather than run.
    """
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    w = torch.nn.Parameter(torch.randn(512, 256, device="cuda", dtype=torch.bfloat16), requires_grad=False)
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    ref = x @ w.detach().clone()

    slot = _park_weight_in_host_store(w, name="w")

    def f(weight, inp):
        return inp @ _WAIT_TENSOR_OP(H2D_LOAD(weight, slot))

    # ``requires_grad=False`` is not incidental: ``magi::h2d_load``'s backward
    # raises, so a grad-requiring weight under grad mode fails to compile.
    with torch.no_grad():
        out = torch.compile(f, backend="inductor", fullgraph=True)(w, x)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


# --------------------------------------------------- host-first materialize


def _make_meta_module_tree() -> torch.nn.Module:
    """A meta-built model: compiled inner, eager sibling the prologue still reads."""
    with torch.device("meta"):
        root = torch.nn.Module()
        root.eager = torch.nn.Linear(4, 4, bias=False)
        root.inner = _CompiledWeightModule(rows=256, cols=256)
        root.to(torch.bfloat16)
    return root


@requires_cuda
@pytest.mark.usefixtures("no_process_group")
def test_plain_parameter_device_moves_are_left_alone():
    """A dtype cast after to_empty must not quietly re-reserve host memory."""
    root = _make_meta_module_tree()
    _install_host_first_hooks(root.inner)
    root.to_empty(device=torch.device("cuda"))
    before = root.inner.big.weight.data_ptr()
    root.inner.float()

    weight = root.inner.big.weight
    assert weight.dtype is torch.float32
    assert weight.data_ptr() != before


@requires_cuda
@pytest.mark.usefixtures("no_process_group")
def test_a_plain_parameter_below_the_floor_stays_on_the_device():
    root = _make_meta_module_tree()
    _install_host_first_hooks(root.inner, min_shard_mib=1.0)  # 256x256 bf16 = 128 KiB
    root.to_empty(device=torch.device("cuda"))

    assert root.inner.big.weight.device.type == "cuda"
    assert root.inner.small.weight.device.type == "cuda"


class _TransposedWeightModule(torch.nn.Module):
    """A weight held transposed, so its dense layout is not the contiguous one."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty(1024, 256, dtype=torch.bfloat16).t())

    def forward(self, x):
        return x @ self.w


class _ConvWeightModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(256, 256, 3, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.conv(x)


def _reference_materialized_layout(cls, memory_format=None) -> tuple[int, ...]:
    """The stride an unpatched ``to_empty`` produces, which host-first must match.

    ``to_empty`` is ``empty_like`` with ``preserve_format``, so a dense
    non-contiguous weight keeps its stride.  Asserting against a measurement
    rather than a literal is what makes these tests about the two paths agreeing.
    """
    with torch.device("meta"):
        root = torch.nn.Module()
        root.inner = cls()
        if memory_format is not None:
            root.inner = root.inner.to(memory_format=memory_format)
    root.to_empty(device=torch.device("cuda"))
    (param,) = root.inner.parameters()
    return param.stride()


@requires_cuda
@pytest.mark.usefixtures("no_process_group")
@pytest.mark.parametrize(
    "model_cls,memory_format",
    [
        pytest.param(_TransposedWeightModule, None, id="transposed"),
        pytest.param(_ConvWeightModule, torch.channels_last, id="channels-last"),
    ],
)
def test_plain_parameter_keeps_the_layout_to_empty_would_give_it(model_cls, memory_format):
    """Decline non-contiguous weights: flat host reservations cannot preserve their layout."""
    from magi_compiler.passes.weight_offload import get_global_host_weight_store
    from magi_compiler.passes.weight_offload.weight_lifecycle import handoff_host_weights_if_pending

    expected = _reference_materialized_layout(model_cls, memory_format=memory_format)
    with torch.device("meta"):
        root = torch.nn.Module()
        root.inner = model_cls()
        if memory_format is not None:
            root.inner.to(memory_format=memory_format)
    _install_host_first_hooks(root.inner)
    root.to_empty(device=torch.device("cuda"))

    (weight,) = root.inner.parameters()
    assert not weight.is_contiguous(), "the fixture must exercise a non-contiguous layout"
    assert weight.device.type == "cuda", "a declined weight must keep its device storage"
    assert weight.stride() == expected, "host-first must preserve the layout produced by to_empty"
    if memory_format is not None:
        assert weight.is_contiguous(memory_format=memory_format)
    assert handoff_host_weights_if_pending(root.inner) == 0, "nothing was parked, so nothing is handed over"
    assert get_global_host_weight_store().weight_count() == 0


@requires_cuda
@pytest.mark.usefixtures("no_process_group")
def test_plain_host_first_materializes_hands_off_and_binds_the_compiled_subtree():
    """Materialize on the host, hand off once, reuse the slots in FX, then restore."""
    from magi_compiler.passes.weight_offload import get_global_host_weight_store
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD
    from magi_compiler.passes.weight_offload.weight_lifecycle import handoff_host_weights_if_pending

    root = _make_meta_module_tree()
    _install_host_first_hooks(root.inner)
    root.to_empty(device=torch.device("cuda"))
    assert root.inner.big.weight.device.type == "cpu"
    assert root.inner.big.weight.is_pinned()
    assert root.eager.weight.device.type == "cuda", "the eager path must keep its device storage"

    expected = torch.arange(root.inner.big.weight.numel(), dtype=torch.bfloat16).reshape(root.inner.big.weight.shape)
    with torch.no_grad():
        root.inner.big.weight.copy_(expected)

    assert handoff_host_weights_if_pending(root.inner) == 2
    assert handoff_host_weights_if_pending(root.inner) == 0, "handoff must not run twice"

    weight = root.inner.big.weight
    assert weight.device.type == "cuda", "the graph is lowered against the parameter's device"
    assert weight.untyped_storage().nbytes() == 0, "no device bytes until the graph loads them"
    assert weight.shape == expected.shape and weight.dtype == expected.dtype
    slot = get_global_host_weight_store().slot_for_tensor(weight)
    assert slot is not None, "PlainParamSource finds the pre-parked Parameter by identity"
    torch.testing.assert_close(get_global_host_weight_store().host_buffer(slot), expected)

    # Bind before restoring device storage, while the parameter is still offloaded.
    gm, examples = _linear_graph(weight)
    assert bind_host_slots_to_graph(gm, examples, PlainParamSource()) == 1
    assert get_global_host_weight_store().weight_count() == 2, "binding must reuse the slots adopted by handoff"
    assert insert_h2d_loads(gm) == 1
    (load,) = _find_nodes_by_target(gm, H2D_LOAD)
    assert load.args[1] == slot

    get_global_host_weight_store().make_weight_resident(slot)
    torch.testing.assert_close(weight.cpu(), expected)
