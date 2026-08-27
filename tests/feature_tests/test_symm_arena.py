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

"""Materialization tests for ``fsdp_config.transport="copy_engine"``.

These cover step 1 of the landing order: the weights move into symmetric memory
and nothing else changes.  The interesting property is *scope* -- the decorated
class must claim exactly its own subtree, from a ``to_empty`` issued on the root
model, without the builder participating.

Single rank on purpose: the placement, scoping, aliasing and fallback logic is
rank-independent, and a 1-rank symmetric window exercises the same allocation
path.  The multi-rank peer reads are covered by
``example/inference/fsdp_overlap/verify_symm_param_hook.py``.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


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


@pytest.fixture(autouse=True)
def _clean_registry():
    from magi_compiler.symm_mem import reset_registry

    reset_registry()
    yield
    reset_registry()


def _decorate(cls: type, transport: str) -> type:
    """Run the real decorator, so the config wiring is under test too."""
    from magi_compiler import magi_compile
    from magi_compiler.config import CompileMode

    def patch(cfg):
        cfg.compile_mode = CompileMode.MAGI_COMPILE
        cfg.fsdp_config.enable_fsdp = True
        cfg.fsdp_config.transport = transport
        return cfg

    return magi_compile(cls, config_patch=patch, dynamic_arg_dims={"x": 0})


def _shard(model: nn.Module, mesh, placement=None) -> None:
    """Wrap every parameter as a DTensor, like torchtitan ``data_parallel``."""
    from torch.distributed.tensor import Shard, distribute_tensor

    placements = [placement or Shard(0)]
    for mod in model.modules():
        for name, p in list(mod.named_parameters(recurse=False)):
            mod.register_parameter(name, nn.Parameter(distribute_tensor(p, mesh, placements)))


class Layer(nn.Module):
    def __init__(self, hidden: int, dtype: torch.dtype):
        super().__init__()
        self.wq = nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.wo = nn.Linear(hidden, hidden, bias=False, dtype=dtype)

    def forward(self, x):
        return self.wo(self.wq(x))


def _make_root(block_cls: type, hidden: int, n_layers: int, dtype: torch.dtype) -> nn.Module:
    class Root(nn.Module):
        """``to_empty`` is called on this, never on the decorated block."""

        def __init__(self):
            super().__init__()
            self.block = block_cls(hidden, n_layers, dtype)
            self.head = nn.Linear(hidden, hidden, bias=False, dtype=dtype)

        def forward(self, x):
            return self.head(self.block(x))

    with torch.device("meta"):
        return Root()


def _make_block_cls(transport: str) -> type:
    class Block(nn.Module):
        def __init__(self, hidden: int, n_layers: int, dtype: torch.dtype):
            super().__init__()
            self.layers = nn.ModuleList([Layer(hidden, dtype) for _ in range(n_layers)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return _decorate(Block, transport)


@requires_cuda
def test_root_to_empty_puts_block_shards_in_arena(mesh_1rank):
    """The decorated block claims its own subtree; the rest stays ordinary."""
    from magi_compiler.symm_mem import lookup_shard, registered_arenas

    hidden, n_layers = 64, 3
    model = _make_root(_make_block_cls("copy_engine"), hidden, n_layers, torch.bfloat16)
    _shard(model, mesh_1rank)
    model.to_empty(device=torch.device("cuda", 0))

    arenas = registered_arenas()
    assert len(arenas) == 1, "one window per (dtype, group), not one per weight"
    arena = arenas[0]

    block_shards = list(model.block.parameters())
    assert len(block_shards) == 2 * n_layers
    assert all(arena.contains(p._local_tensor) for p in block_shards)
    assert all(lookup_shard(p._local_tensor.data_ptr()) is not None for p in block_shards)

    # Outside the decorated class: untouched by the interception.
    assert not arena.contains(model.head.weight._local_tensor)
    assert lookup_shard(model.head.weight._local_tensor.data_ptr()) is None

    # Still ordinary, working DTensors on real storage.
    assert all(p._local_tensor.device.type == "cuda" for p in block_shards)
    assert not any(p.is_meta for p in block_shards)


@requires_cuda
def test_nccl_transport_leaves_allocation_alone(mesh_1rank):
    """The default transport must not install the interception at all."""
    from magi_compiler.symm_mem import registered_arenas

    model = _make_root(_make_block_cls("nccl"), 64, 2, torch.bfloat16)
    _shard(model, mesh_1rank)
    model.to_empty(device=torch.device("cuda", 0))

    assert registered_arenas() == []
    assert all(p._local_tensor.device.type == "cuda" for p in model.parameters())


@requires_cuda
def test_peer_view_round_trips_through_arena(mesh_1rank):
    """A shard written locally must be visible through its own peer view: this is
    the addressing the copy-engine gather depends on."""
    from magi_compiler.symm_mem import lookup_shard

    model = _make_root(_make_block_cls("copy_engine"), 64, 2, torch.bfloat16)
    _shard(model, mesh_1rank)
    model.to_empty(device=torch.device("cuda", 0))

    for i, p in enumerate(model.block.parameters()):
        p._local_tensor.fill_(i + 1)
    torch.cuda.synchronize()

    for i, p in enumerate(model.block.parameters()):
        entry = lookup_shard(p._local_tensor.data_ptr())
        assert len(entry.peer_views) == 1
        assert torch.equal(entry.peer_views[0], p._local_tensor), f"shard {i} aliases the wrong bytes"


@requires_cuda
def test_tied_weights_share_one_slot(mesh_1rank):
    """A shared parameter is visited once per referencing module, so it must get
    one slot and keep its identity -- two allocations would silently untie it and
    overflow a window sized by a deduped walk.

    Note the tie has to be (re)established *after* sharding: torchtitan's
    ``data_parallel`` calls ``distribute_tensor`` per module, which replaces each
    entry with its own DTensor and unties them on its own.
    """
    from magi_compiler.symm_mem import registered_arenas

    hidden = 64

    class TiedBlock(nn.Module):
        def __init__(self, hidden: int, n_layers: int, dtype: torch.dtype):
            super().__init__()
            self.a = nn.Linear(hidden, hidden, bias=False, dtype=dtype)
            self.b = nn.Linear(hidden, hidden, bias=False, dtype=dtype)

        def forward(self, x):
            return self.b(self.a(x))

    model = _make_root(_decorate(TiedBlock, "copy_engine"), hidden, 1, torch.bfloat16)
    _shard(model, mesh_1rank)
    model.block.b.weight = model.block.a.weight
    model.to_empty(device=torch.device("cuda", 0))

    arena = registered_arenas()[0]
    assert model.block.a.weight is model.block.b.weight, "tying must survive materialization"
    assert arena.contains(model.block.a.weight._local_tensor)
    # One slot in the window, not two.
    slot = arena.ALIGN * ((hidden * hidden + arena.ALIGN - 1) // arena.ALIGN)
    assert arena.nbytes == slot * torch.bfloat16.itemsize


@requires_cuda
def test_nested_decorated_block_shares_the_outer_arena(mesh_1rank):
    """A decorated block inside a decorated block must not open a second window:
    the inner one has to fail the lambda check and delegate."""
    from magi_compiler.symm_mem import registered_arenas

    inner_cls = _decorate(
        type(
            "Inner",
            (nn.Module,),
            {
                "__init__": lambda self, hidden, dtype: (
                    nn.Module.__init__(self),
                    setattr(self, "lin", nn.Linear(hidden, hidden, bias=False, dtype=dtype)),
                )[0],
                "forward": lambda self, x: self.lin(x),
            },
        ),
        "copy_engine",
    )

    class Outer(nn.Module):
        def __init__(self, hidden: int, n_layers: int, dtype: torch.dtype):
            super().__init__()
            self.own = nn.Linear(hidden, hidden, bias=False, dtype=dtype)
            self.inner = inner_cls(hidden, dtype)

        def forward(self, x):
            return self.inner(self.own(x))

    model = _make_root(_decorate(Outer, "copy_engine"), 64, 1, torch.bfloat16)
    _shard(model, mesh_1rank)
    model.to_empty(device=torch.device("cuda", 0))

    arenas = registered_arenas()
    assert len(arenas) == 1, f"nested decoration opened {len(arenas)} windows"
    assert arenas[0].contains(model.block.own.weight._local_tensor)
    assert arenas[0].contains(model.block.inner.lin.weight._local_tensor)


@requires_cuda
def test_non_shard0_placement_falls_back(mesh_1rank):
    """Replicate weights are not gatherable, so they must be allocated normally
    rather than silently placed in the arena."""
    from torch.distributed.tensor import Replicate

    from magi_compiler.symm_mem import registered_arenas

    model = _make_root(_make_block_cls("copy_engine"), 64, 2, torch.bfloat16)
    _shard(model, mesh_1rank, placement=Replicate())
    model.to_empty(device=torch.device("cuda", 0))

    assert registered_arenas() == []
    assert all(p._local_tensor.device.type == "cuda" for p in model.block.parameters())


@requires_cuda
def test_two_process_groups_same_dtype_get_two_windows(mesh_1rank, monkeypatch):
    """gaga4 shards bf16 dense weights on the FSDP mesh and bf16 experts on the
    orthogonal edp mesh.  One window per dtype would either rendezvous on the
    wrong group or mix offsets that are only meaningful inside one group.

    Fake group names cannot rendezvous, so ``commit`` is stubbed; the assertion
    is that planning opens two windows keyed by (dtype, group).
    """
    from torch.distributed.tensor import Shard, distribute_tensor

    from magi_compiler.symm_mem import arena as sa

    hidden = 64
    a = nn.Parameter(distribute_tensor(torch.empty(hidden, hidden, dtype=torch.bfloat16), mesh_1rank, [Shard(0)]))
    b = nn.Parameter(distribute_tensor(torch.empty(hidden, hidden, dtype=torch.bfloat16), mesh_1rank, [Shard(0)]))

    monkeypatch.setattr(sa, "_group_name_of", lambda p, _a=a: "dense_fsdp" if p is _a else "edp")
    monkeypatch.setattr(sa.SymmArena, "commit", lambda self: None)
    arenas = sa._plan_arenas([a, b], torch.device("cuda", 0))
    assert set(arenas) == {(torch.bfloat16, "dense_fsdp"), (torch.bfloat16, "edp")}
    assert arenas[(torch.bfloat16, "dense_fsdp")].group_name == "dense_fsdp"
    assert arenas[(torch.bfloat16, "edp")].group_name == "edp"


@requires_cuda
def test_barrier_after_load_is_idempotent(mesh_1rank):
    from magi_compiler.symm_mem import barrier_after_load

    model = _make_root(_make_block_cls("copy_engine"), 64, 2, torch.bfloat16)
    _shard(model, mesh_1rank)
    model.to_empty(device=torch.device("cuda", 0))

    barrier_after_load()
    barrier_after_load()  # a second call must not issue a second collective


# ---------------------------------------------------------------------------
# Window suballocation.  Every rank must agree on which bytes are which shard,
# and nothing at run time re-derives that -- the gather trusts the offsets.
# ---------------------------------------------------------------------------
@pytest.fixture
def group_name():
    import torch.distributed as dist

    return dist.group.WORLD.group_name


def _committed_arena(group_name, numels, dtype=torch.bfloat16):
    from magi_compiler.symm_mem import SymmArena

    arena = SymmArena(dtype, torch.device("cuda", 0), group_name)
    for n in numels:
        arena.reserve(n)
    arena.commit()
    return arena


@requires_cuda
def test_shards_are_dispensed_at_aligned_offsets(mesh_1rank, group_name):
    """Slots are padded to ``ALIGN`` for copy-engine throughput, so the second
    shard does not start where the first one ends.  ``offset_of`` is what the
    peer views are built from, so it has to agree with what ``take`` handed out."""
    from magi_compiler.symm_mem import SymmArena

    rows, cols = 3, 5  # 15 elems: deliberately not a multiple of ALIGN
    arena = _committed_arena(group_name, [rows * cols, rows * cols])
    first = arena.take((rows, cols))
    second = arena.take((rows, cols))

    assert arena.offset_of(first) == 0
    assert arena.offset_of(second) == SymmArena.ALIGN
    assert arena.contains(first) and arena.contains(second)
    assert first.shape == (rows, cols)


@requires_cuda
def test_dispensing_more_than_was_reserved_is_an_error(mesh_1rank, group_name):
    """The sizing walk and the dispensing walk are two separate traversals; if
    they ever disagree the shards silently overlap, so the window must run out
    rather than hand back memory reserved for someone else."""
    arena = _committed_arena(group_name, [64])
    arena.take((8, 8))
    with pytest.raises(RuntimeError, match="symmetric arena overflow"):
        arena.take((8, 8))


@requires_cuda
def test_contains_rejects_memory_outside_the_window(mesh_1rank, group_name):
    """``contains`` is how the rewrite decides a weight is gatherable; a caching
    allocator tensor must never pass."""
    arena = _committed_arena(group_name, [64])
    arena.take((8, 8))
    assert not arena.contains(torch.empty(8, 8, device="cuda", dtype=torch.bfloat16))


@requires_cuda
def test_find_shard_by_layout_matches_on_shape_and_dtype(mesh_1rank, group_name):
    """The cost model replays a gather on *some* registered shard with the right
    layout -- it cannot use a generic ``empty``, which has no peers.  A miss must
    be a None it can degrade on, not a wrong-dtype shard it would gather."""
    from magi_compiler.symm_mem import find_shard_by_layout, register_shard

    arena = _committed_arena(group_name, [8 * 4, 16 * 4])
    small = arena.take((8, 4))
    large = arena.take((16, 4))
    register_shard(small, arena)
    register_shard(large, arena)

    assert find_shard_by_layout((8, 4), torch.bfloat16) is small
    assert find_shard_by_layout((16, 4), torch.bfloat16) is large
    assert find_shard_by_layout((8, 5), torch.bfloat16) is None
    assert find_shard_by_layout((8, 4), torch.float32) is None


@requires_cuda
def test_reset_registry_drops_arenas_and_shards(mesh_1rank, group_name):
    """Tests and the multi-model path rebuild in-process; a stale entry would let
    a freed shard's address answer a lookup."""
    from magi_compiler.symm_mem import find_shard_by_layout, lookup_shard, register_shard, registered_arenas, reset_registry

    arena = _committed_arena(group_name, [8 * 4])
    shard = arena.take((8, 4))
    register_shard(shard, arena)
    assert lookup_shard(shard.data_ptr()) is not None

    reset_registry()
    assert registered_arenas() == []
    assert lookup_shard(shard.data_ptr()) is None
    assert find_shard_by_layout((8, 4), torch.bfloat16) is None


# ---------------------------------------------------------------------------
# migrate_to_arenas -- the live-model path (magi_compile on an already
# materialized model, where there is no to_empty to intercept).
# ---------------------------------------------------------------------------
@requires_cuda
def test_migrate_moves_live_shards_and_keeps_their_values(mesh_1rank):
    """Unlike ``to_empty``, this runs on weights that already hold data, so the
    copy is load-bearing: dropping it would gather uninitialized memory."""
    from magi_compiler.symm_mem import lookup_shard, migrate_to_arenas

    hidden = 64

    class Live(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(hidden, hidden, bias=False, dtype=torch.bfloat16)
            self.b = nn.Linear(hidden, hidden, bias=False, dtype=torch.bfloat16)

    model = Live().to("cuda")
    _shard(model, mesh_1rank)
    before = {n: p._local_tensor.clone() for n, p in model.named_parameters()}

    arenas = migrate_to_arenas(model)
    assert len(arenas) == 1
    arena = next(iter(arenas.values()))

    for name, p in model.named_parameters():
        local = p._local_tensor
        assert arena.contains(local), f"{name} was not migrated"
        assert lookup_shard(local.data_ptr()) is not None
        assert torch.equal(local, before[name]), f"{name} lost its values"


@requires_cuda
def test_migrate_gives_a_tied_weight_one_slot(mesh_1rank):
    """Migration rebuilds each parameter, so python identity does not survive --
    what must survive is the storage, or the tie is gone and the window is
    overflowed by a walk that sized it once."""
    from magi_compiler.symm_mem import migrate_to_arenas

    hidden = 64

    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(hidden, hidden, bias=False, dtype=torch.bfloat16)
            self.b = nn.Linear(hidden, hidden, bias=False, dtype=torch.bfloat16)

    model = Tied().to("cuda")
    _shard(model, mesh_1rank)
    model.b.weight = model.a.weight

    arenas = migrate_to_arenas(model)
    arena = next(iter(arenas.values()))
    assert model.a.weight._local_tensor.data_ptr() == model.b.weight._local_tensor.data_ptr()
    slot = arena.ALIGN * ((hidden * hidden + arena.ALIGN - 1) // arena.ALIGN)
    assert arena.nbytes == slot * torch.bfloat16.itemsize


@requires_cuda
def test_migrate_refuses_a_model_that_was_never_materialized(mesh_1rank):
    """``migrate_to_arenas`` is the live-model entry point, so being handed a
    still-on-meta model is the way it gets misused.  Copying from meta silently
    produces a window of uninitialized weights, so it has to fail instead."""
    from torch.distributed.tensor import DTensor, Shard

    from magi_compiler.symm_mem import migrate_to_arenas

    with torch.device("meta"):
        model = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    local = DTensor.from_local(model.weight.data, mesh_1rank, [Shard(0)], run_check=False)
    model.register_parameter("weight", nn.Parameter(local))
    assert model.weight._local_tensor.is_meta  # the state under test

    with pytest.raises(RuntimeError, match="needs the shards on cuda"):
        migrate_to_arenas(model)


@requires_cuda
def test_migrate_leaves_a_model_with_no_gatherable_shards_alone(mesh_1rank):
    """A plain (unsharded) model must not open an empty window."""
    from magi_compiler.symm_mem import migrate_to_arenas, registered_arenas

    model = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16).to("cuda")
    assert migrate_to_arenas(model) == {}
    assert registered_arenas() == []
