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

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

from magi_compiler.utils import magi_logger

# Only hijack ``to_empty``'s lambda.  ``.cuda()`` / ``.to()`` / ``.float()`` also
# go through ``_apply``; intercepting those would change builder semantics.
_TO_EMPTY_LAMBDA = "Module.to_empty.<locals>.<lambda>"


class SymmArena:
    """One symmetric-memory window, suballocated to many weight shards.

    One window per (decorated block, dtype, process group): a single rendezvous,
    and because every rank walks the module tree in the same order, offset ``k``
    is the same shard on every peer.  Two meshes sharing a dtype get two windows.
    """

    # 256 bf16 elems = 512B, which the copy engine wants for peak throughput.
    ALIGN = 256

    def __init__(self, dtype: torch.dtype, device: torch.device, group_name: str) -> None:
        self.dtype = dtype
        self.device = device
        self.group_name = group_name
        self.buf: torch.Tensor | None = None
        self.handle = None
        self.peers: list[torch.Tensor] = []
        self._reserved = 0
        self._cursor = 0

    # -- build phase ------------------------------------------------------
    def reserve(self, numel: int) -> None:
        self._reserved += self._round(numel)

    def commit(self) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        symm_mem.enable_symm_mem_for_group(self.group_name)
        self.buf = symm_mem.empty(self._reserved, dtype=self.dtype, device=self.device)
        self.handle = symm_mem.rendezvous(self.buf, self.group_name)
        # Slice each peer's whole window once.  VMM maps every window into one
        # VA space so these are directly copyable; NCCL-backend peer pointers are not.
        self.peers = [self.handle.get_buffer(r, (self._reserved,), self.dtype) for r in range(self.handle.world_size)]

    def take(self, shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
        numel = 1
        for s in shape:
            numel *= int(s)
        off = self._cursor
        self._cursor += self._round(numel)
        if self._cursor > self._reserved:
            raise RuntimeError(
                f"symmetric arena overflow: wanted {self._cursor} elems, reserved {self._reserved}. "
                "The sizing walk and the dispensing walk must visit the same shards in the same order."
            )
        return self.buf[off : off + numel].view(shape)

    # -- query ------------------------------------------------------------
    @property
    def nbytes(self) -> int:
        return self._reserved * self.dtype.itemsize

    def offset_of(self, t: torch.Tensor) -> int:
        return (t.data_ptr() - self.buf.data_ptr()) // self.buf.element_size()

    def peer_views(self, t: torch.Tensor) -> list[torch.Tensor]:
        """``world_size`` views of the same shard, one per rank, in rank order."""
        off, numel = self.offset_of(t), t.numel()
        return [p[off : off + numel].view(t.shape) for p in self.peers]

    def contains(self, t: torch.Tensor) -> bool:
        if self.buf is None:
            return False
        base = self.buf.data_ptr()
        return base <= t.data_ptr() < base + self.nbytes

    @classmethod
    def _round(cls, numel: int) -> int:
        return (numel + cls.ALIGN - 1) // cls.ALIGN * cls.ALIGN


@dataclass(frozen=True)
class ShardEntry:
    """What the run-time gather needs to know about one local shard."""

    arena: SymmArena
    offset: int
    local: torch.Tensor
    peer_views: tuple[torch.Tensor, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.local.shape)


# Keyed by ``data_ptr()``: the gather op only sees a plain tensor.
_SHARD_REGISTRY: dict[int, ShardEntry] = {}
_ARENAS: list[SymmArena] = []
_BARRIER_DONE = False


def register_shard(local: torch.Tensor, arena: SymmArena) -> ShardEntry:
    entry = ShardEntry(arena=arena, offset=arena.offset_of(local), local=local, peer_views=tuple(arena.peer_views(local)))
    _SHARD_REGISTRY[local.data_ptr()] = entry
    return entry


def lookup_shard(data_ptr: int) -> ShardEntry | None:
    return _SHARD_REGISTRY.get(data_ptr)


def registered_arenas() -> list[SymmArena]:
    return list(_ARENAS)


def find_shard_by_layout(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor | None:
    """Any registered shard with this layout -- the cost model cannot replay a gather on a generic ``empty`` (no peers)."""
    want = tuple(int(s) for s in shape)
    for entry in _SHARD_REGISTRY.values():
        if entry.shape == want and entry.local.dtype == dtype:
            return entry.local
    return None


def reset_registry() -> None:
    """Test-only: drop every arena so a new model can be built in-process."""
    global _BARRIER_DONE
    _SHARD_REGISTRY.clear()
    _ARENAS.clear()
    _BARRIER_DONE = False


def barrier_after_load() -> None:
    """Publish every rank's freshly written shards, once per process.

    Must run after weights are loaded and before the first peer read.
    """
    global _BARRIER_DONE
    if _BARRIER_DONE or not _ARENAS:
        return
    if dist.is_available() and dist.is_initialized():
        torch.cuda.synchronize()
        dist.barrier()
    _BARRIER_DONE = True
    magi_logger.info(
        "Symmetric arena: published %d arena(s), %.1f MiB, %d shards; steady state is barrier-free",
        len(_ARENAS),
        sum(a.nbytes for a in _ARENAS) / 2**20,
        len(_SHARD_REGISTRY),
    )


def _is_gatherable_shard(t: object) -> bool:
    """A Shard(0) DTensor on a 1-D mesh -- the only placement the copy-engine gather handles."""
    from torch.distributed.tensor import DTensor, Shard

    if not isinstance(t, DTensor):
        return False
    placements = t.placements
    return len(placements) == 1 and isinstance(placements[0], Shard) and placements[0].dim == 0


def _group_name_of(t) -> str | None:
    try:
        return t.device_mesh._dim_group_names[0]
    except Exception:  # noqa: BLE001
        return None


def _arena_key(t) -> tuple[torch.dtype, str]:
    """One window per (dtype, process group). Same dtype on two meshes (gaga4 FSDP + edp) must not share a window."""
    group_name = _group_name_of(t)
    if group_name is None:
        raise RuntimeError(f"cannot resolve the process group of a Shard(0) parameter on mesh {t.device_mesh}")
    return (t.dtype, group_name)


def _apply_order_entries(mod: nn.Module):
    """``(owner, name, param)`` in ``_apply`` order: post-order, every ``_parameters`` entry.

    Not ``named_parameters()`` (pre-order, dedups shared tensors).  A different
    walk would break cross-rank offset symmetry.  Walking ``_parameters`` also
    finds SimpleFSDP weights in ``parametrizations.weight.original``.
    """
    for child in mod.children():
        yield from _apply_order_entries(child)
    for name, p in mod._parameters.items():
        if p is not None:
            yield mod, name, p


def _plan_arenas(shards: list, device: torch.device) -> dict[tuple[torch.dtype, str], SymmArena]:
    """Size and commit one window per (dtype, group). Dedup by identity so a tied weight reserves a single slot."""
    arenas: dict[tuple[torch.dtype, str], SymmArena] = {}
    seen: set[int] = set()
    for p in shards:
        if id(p) in seen:
            continue
        seen.add(id(p))
        key = _arena_key(p)
        arena = arenas.get(key)
        if arena is None:
            arena = arenas[key] = SymmArena(p.dtype, device, key[1])
        arena.reserve(p._local_tensor.numel())

    for arena in arenas.values():
        arena.commit()  # the only collective, once per window
    _ARENAS.extend(arenas.values())
    return arenas


def materialize_into_arenas(mod: nn.Module, device: torch.device) -> dict[tuple[torch.dtype, str], SymmArena]:
    """Size windows for ``mod``'s Shard(0) shards while they are still on meta. Non-gatherable params are left to the caller."""
    shards = [p for _, _, p in _apply_order_entries(mod) if _is_gatherable_shard(p)]
    if not shards:
        return {}
    return _plan_arenas(shards, device)


def migrate_to_arenas(root: nn.Module) -> dict[tuple[torch.dtype, str], SymmArena]:
    """Copy already-allocated Shard(0) shards into symmetric memory.

    Used when ``magi_compile(model, ...)`` is given a live model rather than a
    meta + ``to_empty`` path.  ``load_state_dict(assign=True)`` after this would
    replace arena views with ordinary tensors; the gather then rejects them.
    """
    entries = [(m, n, p) for m, n, p in _apply_order_entries(root) if _is_gatherable_shard(p)]
    if not entries:
        return {}

    device = entries[0][2]._local_tensor.device
    if device.type != "cuda":
        raise RuntimeError(f"symmetric memory needs the shards on cuda, found {device}")
    arenas = _plan_arenas([p for _, _, p in entries], device)

    from torch.distributed.tensor import DTensor

    views: dict[int, torch.Tensor] = {}
    for owner, name, p in entries:
        local = views.get(id(p))
        if local is None:
            arena = arenas[_arena_key(p)]
            local = views[id(p)] = arena.take(p._local_tensor.shape)
            local.copy_(p._local_tensor)
            register_shard(local, arena)
        moved = DTensor.from_local(local, p.device_mesh, p.placements, run_check=False)
        owner.register_parameter(name, nn.Parameter(moved, requires_grad=p.requires_grad))

    magi_logger.info(
        "Symmetric arena: migrated %d shard(s) into %.1f MiB across %d window(s)",
        len(views),
        sum(a.nbytes for a in arenas.values()) / 2**20,
        len(arenas),
    )
    return arenas


def patch_symm_arena_apply(cls: type[nn.Module]) -> None:
    """Install the ``_apply`` interception on a decorated class.

    Mirrors ``_patch_cpu_offload_apply``: take over for ``to_empty``'s lambda, delegate everything else.
    """
    if getattr(cls, "_magi_symm_apply_patched", False):
        return
    orig_apply = cls._apply
    magi_logger.info("Symmetric arena: intercepting %s._apply for copy-engine FSDP", cls.__name__)

    def _symm_apply(self, fn, recurse: bool = True):
        if getattr(fn, "__qualname__", "") != _TO_EMPTY_LAMBDA:
            return orig_apply(self, fn, recurse)
        if getattr(self, "_magi_symm_arenas", None) is not None:
            return orig_apply(self, fn, recurse)

        device = torch.device(inspect.getclosurevars(fn).nonlocals["device"])
        from torch.distributed.tensor import DTensor

        arenas = materialize_into_arenas(self, device)
        if not arenas:
            return orig_apply(self, fn, recurse)

        views: dict[int, torch.Tensor] = {}

        def materialize(t: torch.Tensor) -> torch.Tensor:
            if not _is_gatherable_shard(t):
                return torch.empty_like(t, device=device)
            # Tied weight: same view so tying survives materialization.
            local = views.get(id(t))
            if local is None:
                arena = arenas[_arena_key(t)]
                local = views[id(t)] = arena.take(t._local_tensor.shape)
                register_shard(local, arena)
            return DTensor.from_local(local, t.device_mesh, t.placements, run_check=False)

        # Do not forge to_empty's qualname: a nested decorated block must fail
        # the check above and delegate, so its params land in *this* arena.
        out = orig_apply(self, materialize, recurse)
        self._magi_symm_arenas = arenas
        magi_logger.info(
            "Symmetric arena: %s materialized %d shard(s) into %.1f MiB across %d window(s)",
            cls.__name__,
            len(views),
            sum(a.nbytes for a in arenas.values()) / 2**20,
            len(arenas),
        )
        return out

    cls._apply = _symm_apply
    cls._magi_symm_apply_patched = True
