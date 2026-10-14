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

"""Slot table of the weight shards the compiled graph loads from host on demand.

The bytes live in a pinned :class:`~magi_compiler.host_mem.HostMemPool`; this
module decides which shard each region holds and whether the device copy is
back.  Pinned, because a paged source would make every load synchronous and
leave nothing for the reorder pass to overlap.

A shard is addressed by an integer ``slot``, not by a pointer.  Adopt pairs a
filled host buffer with a storage-free CUDA stand-in; the stand-in's address is
not a usable key, so the slot rides along in the graph instead.

Slots are minted in this process at adopt time (0, 1, 2, …).  A cached
kernel bakes those integers, so a later process remaps them through
``slot_remap.using_slot_remap`` (see the ``host_slots.py`` sidecar next to
the piecewise cache) rather than compiling the loads again.

The only way in is ``reserve`` + ``adopt``: the loader reads the checkpoint
straight into a pinned reservation, and the shard never occupies a device byte.
That is what keeps the device high-water mark at the resident set instead of
the whole model.

The process holds one :class:`HostWeightStore`.  ``magi::h2d_load`` reads it at
runtime by the slot integers baked into the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from magi_compiler.host_mem import HostMemPool
from magi_compiler.utils import magi_logger

from .bandwidth import PROBE_BYTES, measure_h2d_bandwidth, probe_world


@dataclass(frozen=True)
class HostWeight:
    """One shard parked in host memory, plus the tensor it came out of.

    ``device_tensor`` is kept so :func:`restore_all` can put the shard back
    without the caller having to remember which parameter a slot belonged to.
    """

    host: torch.Tensor
    device_tensor: torch.Tensor
    nbytes: int
    name: str = ""  # the parameter this shard came from, for the placement logs


class HostWeightStore:
    """Slot table + residency state for one process's host-offloaded weights.

    Adopt is idempotent on tensor identity: a second compile of the same
    stand-in must reuse the slot the first one minted.  ``_resident`` and the
    device tensor's storage are the same question -- ``restore_all`` /
    ``make_resident`` always update both.
    """

    def __init__(self, pool: HostMemPool | None = None) -> None:
        self._pool = pool if pool is not None else HostMemPool(pinned=True)
        self._slots: list[HostWeight] = []
        # ``id(local shard) -> slot``.  The store holds a reference to each
        # shard, so the object stays alive and its id stays unique even after
        # its CUDA storage is gone.
        self._by_tensor: dict[int, int] = {}
        self._claimed: set[int] = set()
        self._resident: set[int] = set()
        self._bandwidth_bytes_per_ns: float | None = None

    def reset(self) -> None:
        """Drop every slot, slab and cached probe.  Tests only."""
        self._slots.clear()
        self._pool.reset()
        self._resident.clear()
        self._claimed.clear()
        self._by_tensor.clear()
        self._bandwidth_bytes_per_ns = None

    def _register(self, host: torch.Tensor, device_tensor: torch.Tensor, nbytes: int, name: str) -> int:
        slot = len(self._slots)
        self._slots.append(HostWeight(host=host, device_tensor=device_tensor, nbytes=nbytes, name=name))
        self._by_tensor[id(device_tensor)] = slot
        return slot

    def _entry(self, slot: int) -> HostWeight:
        try:
            return self._slots[slot]
        except IndexError:
            raise RuntimeError(
                f"magi::h2d_load got slot {slot} but only {len(self._slots)} shard(s) are bound. "
                "The compiled graph outlived the host weight store it was built against -- a cached "
                "artifact replayed in a fresh process, or reset() between compile and run."
            ) from None

    def reserve(self, shape: Sequence[int], dtype: torch.dtype, name: str = "") -> torch.Tensor:
        """Pinned host storage for a shard that has no device copy yet.

        A checkpoint can be read straight into the reservation, so the shard
        never occupies a device byte.  There is no slot yet: a slot pairs a
        host buffer with the device tensor that stands in for it in the graph,
        and that tensor does not exist until :meth:`adopt`.
        """
        host = self._pool.alloc(shape, dtype)
        magi_logger.debug(
            "host offload: reserved %.1f MiB of pinned host memory for %s", host.numel() * host.element_size() / 2**20, name
        )
        return host

    def adopt(self, host: torch.Tensor, device_tensor: torch.Tensor, name: str = "") -> int:
        """Pair a filled host buffer with a storage-free device tensor.  Returns its slot.

        No copy: the bytes are already where they belong.  No ``resize_(0)``:
        ``device_tensor`` is expected to arrive empty -- it exists only to carry
        shape, dtype and device into the graph.

        Re-adopting the same device tensor returns the slot already minted for
        it.  A second compile must not create a second slot over the same
        stand-in.
        """
        existing = self._by_tensor.get(id(device_tensor))
        if existing is not None:
            return existing
        if device_tensor.untyped_storage().nbytes() != 0:
            raise ValueError(
                f"host offload: adopt({name!r}) wants a device tensor with no storage behind it, but got "
                f"{device_tensor.untyped_storage().nbytes()} byte(s).  A tensor that still holds device "
                "bytes is not an offload candidate -- materialize it in host memory instead."
            )
        return self._register(host, device_tensor, host.numel() * host.element_size(), name)

    def slot_of(self, local: torch.Tensor) -> int | None:
        """The slot this stand-in was adopted under, or None."""
        return self._by_tensor.get(id(local))

    def name_of(self, slot: int) -> str:
        """The parameter behind a slot, for the placement logs."""
        return self._entry(slot).name

    def get(self, slot: int) -> torch.Tensor:
        """The pinned host shard behind ``slot``."""
        return self._entry(slot).host

    def source(self, slot: int) -> torch.Tensor:
        """Where a load should copy this shard from.

        The device copy for a promoted slot, the host copy otherwise.  The load
        reads this instead of ``get`` so promotion needs no second code path:
        same op, same stream, same event, just a much shorter wire.
        """
        entry = self._entry(slot)
        return entry.device_tensor if slot in self._resident else entry.host

    def make_resident(self, slot: int) -> int:
        """Put a shard back on the device.  Returns the bytes this cost (0 if already there)."""
        return self.make_resident_many([slot])

    def make_resident_many(self, slots: Sequence[int]) -> int:
        """Put every listed shard back on the device.  One sync for the whole batch.

        Called from the placement pass for loads whose transfer no amount of
        hoisting could hide.  Such a load is pure exposed latency: it stalls
        the compute stream in front of its own all-gather and delays the
        launch, so paying its bytes in device memory buys back more than it
        costs.

        Per-slot ``cuda.synchronize`` used to serialize this: a model that
        hands back dozens of weights paid a host round-trip for each one.
        """
        nbytes = 0
        copied = False
        for slot in slots:
            if slot in self._resident:
                continue
            entry = self._entry(slot)
            if entry.device_tensor.untyped_storage().nbytes() == 0:
                entry.device_tensor.untyped_storage().resize_(entry.nbytes)
                # An unsharded nn.Parameter is itself the stand-in the slot is
                # keyed on.  Promotion runs at compile time, not under autograd.
                with torch.no_grad():
                    entry.device_tensor.copy_(entry.host, non_blocking=True)
                copied = True
            self._resident.add(slot)
            nbytes += entry.nbytes
        if copied:
            torch.cuda.synchronize()
        return nbytes

    def is_resident(self, slot: int) -> bool:
        return slot in self._resident

    def mark_claimed(self, slot: int) -> None:
        """Record that a graph now carries a load for this slot."""
        self._claimed.add(slot)

    def restore_unclaimed(self) -> list[str]:
        """Hand back every parked shard no graph loads.  Returns what was restored.

        The fail-safe for the whole offload path.  A shard is parked on the
        strength of a prediction -- that the graph will turn out to contain an
        all-gather reading it -- and a prediction that misses leaves a kernel
        reading freed storage, which surfaces as an illegal access with nothing
        pointing back here.  Restoring costs device memory and nothing else:
        the shard's graph, if one ever appears, still loads it, just
        device-to-device.
        """
        pending = [slot for slot in range(len(self._slots)) if slot not in self._claimed and slot not in self._resident]
        if not pending:
            return []
        self.make_resident_many(pending)
        return [self._slots[slot].name or f"slot {slot}" for slot in pending]

    def slot_bytes(self, slot: int) -> int:
        return self._entry(slot).nbytes

    def resident_bytes(self) -> int:
        """Bytes of promoted shards, i.e. device memory offload gave back."""
        return sum(self._slots[s].nbytes for s in self._resident)

    def bound_bytes(self) -> int:
        """Device bytes actually freed: everything bound, less what was promoted back."""
        return sum(s.nbytes for s in self._slots) - self.resident_bytes()

    def total_bound_bytes(self) -> int:
        """Every bound shard, promoted or not -- the base the residency budget is a fraction of."""
        return sum(s.nbytes for s in self._slots)

    def num_bound(self) -> int:
        return len(self._slots)

    def restore_all(self) -> None:
        """Put every bound shard back on the device.

        Escape hatch for teardown and for running an offloaded model through
        an un-offloaded code path.  Marks every slot resident so ``source``
        and the device storage agree afterwards -- leaving them unmarked made
        the next ``h2d_load`` copy from host while the bytes were already on
        the device.
        """
        self.make_resident_many(range(len(self._slots)))

    def h2d_bandwidth_bytes_per_ns(self, override_gbps: float = 0.0) -> float:
        """Measured host-to-device bandwidth, in bytes per nanosecond.

        Measured once per process and reused: the reorder pass sizes thousands
        of loads, and re-timing each one would cost more than the placement it
        informs while telling us nothing new -- a pinned DMA of a contiguous
        shard is the same transfer at every size above the fixed overhead.

        The probe runs on every rank AT ONCE, which is the whole point.
        Measured alone, one H100 reads pinned host memory at ~55 GB/s; measured
        with its seven neighbours doing the same, it gets ~27 GB/s, because
        what runs out is host memory bandwidth (~207 GB/s aggregate), not the
        PCIe link.  Calibrating in isolation hands the placement pass a number
        that is twice the truth, and it then sizes every overlap window at
        half the transfer it has to hide -- which does not fail, it just
        quietly stops hiding things.

        The result is deliberately NOT reduced across ranks.  Load placement
        moves no collective, so ranks are free to disagree (see
        ``H2dLoadReorder``), and a rank on a slower root complex should hoist
        further, not adopt a peer's number.
        """
        if override_gbps > 0:
            # 1 GB/s == 1e9 bytes / 1e9 ns == 1 byte/ns, so the units coincide.
            return override_gbps
        if self._bandwidth_bytes_per_ns is not None:
            return self._bandwidth_bytes_per_ns

        self._bandwidth_bytes_per_ns = measure_h2d_bandwidth()
        magi_logger.info(
            "host offload: measured H2D bandwidth %.1f GB/s (pinned, %d MiB probe, %d rank(s) probing together)",
            self._bandwidth_bytes_per_ns,
            PROBE_BYTES // 2**20,
            probe_world(),
        )
        return self._bandwidth_bytes_per_ns


# Process-wide store.  ``magi::h2d_load`` looks slots up here at runtime; a
# per-backend instance would desync from the integers already in the graph.
_STORE = HostWeightStore()


def default_store() -> HostWeightStore:
    return _STORE


def reserve(shape: Sequence[int], dtype: torch.dtype, name: str = "") -> torch.Tensor:
    return _STORE.reserve(shape, dtype, name)


def adopt(host: torch.Tensor, device_tensor: torch.Tensor, name: str = "") -> int:
    return _STORE.adopt(host, device_tensor, name)


def slot_of(local: torch.Tensor) -> int | None:
    return _STORE.slot_of(local)


def name_of(slot: int) -> str:
    return _STORE.name_of(slot)


def get(slot: int) -> torch.Tensor:
    return _STORE.get(slot)


def source(slot: int) -> torch.Tensor:
    return _STORE.source(slot)


def make_resident(slot: int) -> int:
    return _STORE.make_resident(slot)


def make_resident_many(slots: Sequence[int]) -> int:
    return _STORE.make_resident_many(slots)


def is_resident(slot: int) -> bool:
    return _STORE.is_resident(slot)


def mark_claimed(slot: int) -> None:
    _STORE.mark_claimed(slot)


def restore_unclaimed() -> list[str]:
    return _STORE.restore_unclaimed()


def slot_bytes(slot: int) -> int:
    return _STORE.slot_bytes(slot)


def resident_bytes() -> int:
    return _STORE.resident_bytes()


def bound_bytes() -> int:
    return _STORE.bound_bytes()


def total_bound_bytes() -> int:
    return _STORE.total_bound_bytes()


def num_bound() -> int:
    return _STORE.num_bound()


def restore_all() -> None:
    _STORE.restore_all()


def reset() -> None:
    _STORE.reset()


def h2d_bandwidth_bytes_per_ns(override_gbps: float = 0.0) -> float:
    return _STORE.h2d_bandwidth_bytes_per_ns(override_gbps)
