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

"""Symmetric-memory windows for copy-engine weight all-gather.

A window is a plain addressable region: open one, then read any slot in it by
``(offset, shape)``. Deciding what goes where -- and keeping that decision
identical on every rank -- belongs to the caller; for weights that is ``bind``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class SymmBuffer:
    """One symmetric-memory window, addressed by element offset.

    The driver caps windows at 128 per process regardless of size, so callers
    pool many shards into one window. An offset names the same bytes on every
    rank, which is what makes a peer read meaningful -- so a caller that hands
    out slots must hand out the same ones on every rank.
    """

    def __init__(
        self, dtype: torch.dtype, device: torch.device, group_name: str, numel: int, buf: torch.Tensor, handle
    ) -> None:
        """Wrap an already-rendezvous'd window.  Callers want ``open``."""
        self.dtype = dtype
        self.device = device
        self.group_name = group_name
        self.numel = numel
        self.buf = buf
        self.handle = handle

    @classmethod
    def open(cls, dtype: torch.dtype, device: torch.device, group_name: str, numel: int) -> SymmBuffer:
        """Allocate and rendezvous a window of ``numel`` elements.

        Collective: every rank must open the same windows in the same order.
        """
        import torch.distributed._symmetric_memory as symm_mem

        symm_mem.enable_symm_mem_for_group(group_name)
        buf = symm_mem.empty(numel, dtype=dtype, device=device)
        handle = symm_mem.rendezvous(buf, group_name)
        return cls(dtype, device, group_name, numel, buf, handle)

    def get_tensor(self, offset: int, shape: torch.Size | tuple[int, ...], *, rank: int | None = None) -> torch.Tensor:
        """The slot at ``offset`` elements in, on ``rank`` (default: this rank).

        Not a slice of ``buf``: the returned tensor's storage starts at the slot.
        Dynamo memoized ``storage_offset == 0`` for bound parameters, and a
        mid-window slice would contradict the shape env.
        """
        shape = tuple(int(s) for s in shape)
        numel = 1
        for s in shape:
            numel *= s
        if offset < 0 or offset + numel > self.numel:
            raise RuntimeError(
                f"symmetric window overflow: {shape} at offset {offset} needs {offset + numel} "
                f"elems of a {self.numel}-elem window"
            )
        return self.handle.get_buffer(self.handle.rank if rank is None else rank, shape, self.dtype, offset)

    def peer_tensors(self, offset: int, shape: torch.Size | tuple[int, ...]) -> list[torch.Tensor]:
        """``world_size`` views of one slot, one per rank.

        They borrow the window mapping; ``self.buf`` keeps it alive.
        """
        return [self.get_tensor(offset, shape, rank=r) for r in range(self.handle.world_size)]

    @property
    def nbytes(self) -> int:
        return self.numel * self.dtype.itemsize

    def contains(self, t: torch.Tensor) -> bool:
        base = self.buf.data_ptr()
        return base <= t.data_ptr() < base + self.nbytes


@dataclass(frozen=True)
class ShardEntry:
    """What a run-time gather needs to know about one local shard."""

    buffer: SymmBuffer
    offset: int
    local: torch.Tensor
    peer_views: tuple[torch.Tensor, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.local.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self.local.dtype


_SHARD_REGISTRY: dict[int, ShardEntry] = {}
_BUFFERS: list[SymmBuffer] = []


def open_buffer(dtype: torch.dtype, device: torch.device, group_name: str, numel: int) -> SymmBuffer:
    """Open one window of ``numel`` elements, and keep it alive for the process.

    Whoever writes the window owes its peers a barrier before they read it; that
    is the writer's business, not the registry's.
    """
    buffer = SymmBuffer.open(dtype, device, group_name, numel)
    _BUFFERS.append(buffer)
    return buffer


def register_shard(local: torch.Tensor, buffer: SymmBuffer, offset: int) -> ShardEntry:
    """Record a slot so the run-time gather can find its peer views."""
    entry = ShardEntry(buffer=buffer, offset=offset, local=local, peer_views=tuple(buffer.peer_tensors(offset, local.shape)))
    _SHARD_REGISTRY[local.data_ptr()] = entry
    return entry


def lookup_shard(data_ptr: int) -> ShardEntry | None:
    """The registered shard starting at ``data_ptr``, or None if it is not one."""
    return _SHARD_REGISTRY.get(data_ptr)


def registered_buffers() -> list[SymmBuffer]:
    return list(_BUFFERS)


def find_shard_by_layout(shape, dtype: torch.dtype) -> torch.Tensor | None:
    """Any registered shard with this exact layout, for the runtime estimator's stand-in."""
    want = tuple(int(s) for s in shape)
    for entry in _SHARD_REGISTRY.values():
        if entry.shape == want and entry.dtype == dtype:
            return entry.local
    return None


def reset_registry() -> None:
    """Drop every window and shard.  Tests only -- frees the symmetric allocations."""
    _SHARD_REGISTRY.clear()
    _BUFFERS.clear()


def alloc_shard(shape, dtype: torch.dtype, device: torch.device, group_name: str) -> torch.Tensor:
    """One shard in a window of its own. Tests and the cost model only -- binding a whole model must pool."""
    shape = tuple(int(s) for s in shape)
    numel = 1
    for s in shape:
        numel *= s

    buffer = open_buffer(dtype, device, group_name, numel)
    shard = buffer.get_tensor(0, shape)
    register_shard(shard, buffer, 0)
    return shard
