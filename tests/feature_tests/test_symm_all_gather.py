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

"""Mechanics of ``magi::symm_all_gather`` (step 2 of the landing order).

Single rank, so the copies are device-to-self and the values are trivially
checkable -- what is under test here is the plumbing that is easy to get subtly
wrong and hard to see: that the untouched ``wait_tensor`` really does pick up our
event through the work registry, that two in-flight gathers do not alias, and
that a shard the arena never saw is rejected loudly rather than gathering
garbage.

The transport itself (peer reads, overlap, ordering under load) needs several
NVLink-connected ranks and lives in
``example/inference/fsdp_overlap/verify_symm_ag_op.py``.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_WAIT = torch.ops._c10d_functional.wait_tensor


@pytest.fixture(scope="module")
def mesh_1rank():
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29672")
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


@pytest.fixture(autouse=True)
def _clean_state():
    # Importing the module is what defines ``magi::symm_all_gather``.  Nothing
    # else here pulls it in, so without this the ops only exist when some other
    # test in the same session happened to import them first.
    import magi_compiler.symm_mem.all_gather  # noqa: F401
    from magi_compiler.symm_mem import reset_registry

    reset_registry()
    yield
    reset_registry()


def _arena_model(mesh, hidden: int = 64, n_layers: int = 3, dtype=torch.bfloat16):
    """A meta-built, Shard(0)-sharded model materialized into a symmetric arena,
    exactly the shape step 1 produces."""
    from torch.distributed.tensor import Shard, distribute_tensor

    from magi_compiler.symm_mem import materialize_into_arenas

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(hidden, hidden, bias=False, dtype=dtype) for _ in range(n_layers)])

    with torch.device("meta"):
        model = Block()
    for mod in model.modules():
        for name, p in list(mod.named_parameters(recurse=False)):
            mod.register_parameter(name, nn.Parameter(distribute_tensor(p, mesh, [Shard(0)])))

    device = torch.device("cuda", 0)
    from torch.distributed.tensor import DTensor

    from magi_compiler.symm_mem.arena import _arena_key, register_shard

    arenas = materialize_into_arenas(model, device)
    views: dict[int, torch.Tensor] = {}

    def materialize(t):
        if isinstance(t, DTensor):
            arena = arenas[_arena_key(t)]
            local = arena.take(t._local_tensor.shape)
            register_shard(local, arena)
            views[id(t)] = local
            return DTensor.from_local(local, t.device_mesh, t.placements, run_check=False)
        return torch.empty_like(t, device=device)

    materialize.__qualname__ = "Module.to_empty.<locals>.<lambda>"
    nn.Module._apply(model, materialize)

    shards = [p._local_tensor for p in model.parameters()]
    for i, s in enumerate(shards):
        s.copy_(torch.full_like(s, float(i + 1)))
    torch.cuda.synchronize()
    return model, shards


@requires_cuda
def test_gather_matches_nccl_bitwise(mesh_1rank):
    _, shards = _arena_model(mesh_1rank)

    for shard in shards:
        got = _WAIT(torch.ops.magi.symm_all_gather(shard, 1, ""))
        ref = torch.empty_like(got)
        torch.distributed.all_gather_into_tensor(ref, shard)
        torch.cuda.synchronize()
        assert torch.equal(got, ref), "disagrees with all_gather_into_tensor"


@requires_cuda
def test_coalesced_wrap_matches_per_member_gather(mesh_1rank):
    """The thin wrap is per-member dests; each wait must match a single gather."""
    _, shards = _arena_model(mesh_1rank)
    outs = torch.ops.magi.symm_all_gather_coalesced(list(shards), 1, "")
    assert len(outs) == len(shards)
    for out, shard in zip(outs, shards):
        got = _WAIT(out)
        ref = _WAIT(torch.ops.magi.symm_all_gather(shard, 1, ""))
        assert torch.equal(got, ref)


@requires_cuda
def test_wait_tensor_picks_up_the_registered_event(mesh_1rank):
    """The launch registers a Work; the *stock* wait_tensor must consume it."""
    _, shards = _arena_model(mesh_1rank)
    shard = shards[0]

    out = _WAIT(torch.ops.magi.symm_all_gather(shard, 1, ""))
    # No synchronize: the value must be correct because of the wait alone.
    assert torch.equal(out, shard.expand_as(out)), "wait_tensor did not order the copies"


@requires_cuda
def test_each_gather_returns_a_fresh_buffer(mesh_1rank):
    """Two live gathers must land in different buffers, or a prefetched weight
    would be overwritten before its consumer ran."""
    _, shards = _arena_model(mesh_1rank)
    a = torch.ops.magi.symm_all_gather(shards[0], 1, "")
    b = torch.ops.magi.symm_all_gather(shards[1], 1, "")
    _WAIT(a)
    _WAIT(b)
    torch.cuda.synchronize()
    assert a.data_ptr() != b.data_ptr()
    assert torch.equal(a, shards[0].expand_as(a))
    assert torch.equal(b, shards[1].expand_as(b))


@requires_cuda
def test_unregistered_shard_is_rejected(mesh_1rank):
    """A weight the arena never claimed has no peer views, so gathering it would
    read whatever happens to be at that address.  Fail instead."""
    ordinary = torch.ones(8, 4, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="not a registered symmetric-memory shard"):
        torch.ops.magi.symm_all_gather(ordinary, 1, "")


@requires_cuda
def test_group_size_mismatch_is_rejected(mesh_1rank):
    _, shards = _arena_model(mesh_1rank)
    with pytest.raises(RuntimeError, match="peers but the gather asks for"):
        torch.ops.magi.symm_all_gather(shards[0], 4, "")


@requires_cuda
def test_coalesced_validates_every_member_before_allocating(mesh_1rank):
    """One bad member must fail the whole call.  Allocating the dests first and
    discovering it halfway through would leave the earlier members' copies in
    flight against buffers nobody waits on."""
    _, shards = _arena_model(mesh_1rank)
    ordinary = torch.ones(8, 4, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="not a registered symmetric-memory shard"):
        torch.ops.magi.symm_all_gather_coalesced([shards[0], ordinary], 1, "")


@requires_cuda
def test_meta_kernel_shape(mesh_1rank):
    with torch.device("meta"):
        local = torch.empty(4, 6, dtype=torch.bfloat16)
    out = torch.ops.magi.symm_all_gather(local, 8, "")
    assert out.shape == (32, 6) and out.is_meta


@requires_cuda
def test_coalesced_meta_kernel_shapes(mesh_1rank):
    """Bucket members have different row counts, so the meta kernel cannot
    broadcast one shape across the list -- Dynamo would trace the wrong dest."""
    with torch.device("meta"):
        shards = [torch.empty(4, 6, dtype=torch.bfloat16), torch.empty(2, 6, dtype=torch.bfloat16)]
    outs = torch.ops.magi.symm_all_gather_coalesced(shards, 8, "")
    assert [tuple(o.shape) for o in outs] == [(32, 6), (16, 6)]
    assert all(o.is_meta for o in outs)


# ---------------------------------------------------------------------------
# The copy plan -- raw pointer arithmetic, so nothing downstream can catch a
# wrong offset: it reads whatever is at that address.
# ---------------------------------------------------------------------------
@requires_cuda
def test_copy_plan_lands_each_rank_at_its_own_offset():
    """Rank r's shard belongs at row ``r * rows``.  An off-by-one here is a
    silently wrong weight, not a crash."""
    from magi_compiler.symm_mem.all_gather import _copy_triples, _gather_dest

    world = 4
    local = torch.ones(8, 4, device="cuda", dtype=torch.bfloat16)
    peers = tuple(torch.full_like(local, r) for r in range(world))
    out = _gather_dest(local, world)
    assert out.shape == (8 * world, 4)

    triples = _copy_triples([(out, local, peers)])
    nbytes = local.numel() * local.element_size()
    assert triples == [(out.data_ptr() + r * nbytes, peers[r].data_ptr(), nbytes) for r in range(world)]


@requires_cuda
def test_copy_plan_concatenates_members_of_one_batch():
    """A coalesced gather is one submission covering every member; the members
    must not share a destination."""
    from magi_compiler.symm_mem.all_gather import _copy_triples, _gather_dest

    world = 2
    a = torch.ones(8, 4, device="cuda", dtype=torch.bfloat16)
    b = torch.ones(2, 4, device="cuda", dtype=torch.bfloat16)
    gathers = [(_gather_dest(t, world), t, tuple(torch.empty_like(t) for _ in range(world))) for t in (a, b)]

    triples = _copy_triples(gathers)
    assert len(triples) == 2 * world
    assert {b for _d, _s, b in triples} == {a.numel() * a.element_size(), b.numel() * b.element_size()}
    assert len({d for d, _s, _b in triples}) == 2 * world, "two copies target the same address"


@requires_cuda
def test_batch_plan_preserves_order_and_count():
    """``cudaMemcpyBatchAsync`` reads three parallel arrays; a reordering between
    them pairs a destination with the wrong source."""
    from magi_compiler.symm_mem.all_gather import BatchMemcpy

    triples = [(0x1000, 0x2000, 64), (0x3000, 0x4000, 128)]
    dsts, srcs, sizes, n = BatchMemcpy.plan(triples)
    assert n == 2
    assert [dsts[i] for i in range(n)] == [0x1000, 0x3000]
    assert [srcs[i] for i in range(n)] == [0x2000, 0x4000]
    assert [sizes[i] for i in range(n)] == [64, 128]


@requires_cuda
def test_per_peer_fallback_matches_the_batched_path(mesh_1rank, monkeypatch):
    """Older CUDA runtimes have no ``cudaMemcpyBatchAsync``; the fallback is the
    only path there and is never exercised on the boxes we develop on."""
    from magi_compiler.symm_mem import all_gather as ag_mod

    _, shards = _arena_model(mesh_1rank)
    batched = _WAIT(torch.ops.magi.symm_all_gather(shards[0], 1, ""))
    torch.cuda.synchronize()

    monkeypatch.setattr(ag_mod, "_batcher", lambda: None)
    fallback = _WAIT(torch.ops.magi.symm_all_gather(shards[0], 1, ""))
    torch.cuda.synchronize()

    assert torch.equal(fallback, batched)
