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

"""Process-wide host-weight slots and device residency state.

Pinned buffers support asynchronous H2D copies. ``reserve`` supplies storage;
``adopt`` pairs a filled host buffer with a storage-free CUDA stand-in by identity.
Host-first handoff temporarily allocates one full CUDA shard before freeing its storage.

Compiled loads carry process-local slot integers rather than storage pointers.
Cached artifacts remap these integers through ``using_slot_remap`` and ``host_slots.py``.
Promotion restores a slot's device copy without discarding its host buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from magi_compiler.utils import magi_logger

from .bandwidth import PROBE_BYTES, measure_h2d_bandwidth
from .pool import HostMemPool


@dataclass(frozen=True)
class HostWeight:
    """One shard parked in host memory, plus the tensor it came out of.

    ``device_tensor`` is kept so :meth:`HostWeightStore.restore_all` can put the shard back
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
    ``make_resident`` / ``release_resident_many`` always update both.
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
        """Reserve pinned host storage that a checkpoint loader can fill directly.
        No slot is created until ``adopt`` pairs it with a CUDA stand-in."""
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
        """Return the device copy for a resident slot, otherwise the pinned host copy."""
        entry = self._entry(slot)
        return entry.device_tensor if slot in self._resident else entry.host

    def make_resident(self, slot: int) -> int:
        """Put a shard back on the device.  Returns the bytes this cost (0 if already there)."""
        return self.make_resident_many([slot])

    def make_resident_many(self, slots: Sequence[int]) -> int:
        """Restore listed shards, synchronizing once if any copies were submitted.
        Return newly resident bytes; already resident slots contribute zero."""
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

    def release_resident_many(self, slots: Sequence[int]) -> int:
        """Free the device copy of every listed promoted shard.  Returns the bytes freed.

        The inverse of :meth:`make_resident_many`, for a caller that parks a whole
        model between uses.  The host copy stays, so a released slot loads over
        H2D again and ``make_resident_many`` brings the device copy back.  The
        caller must have synchronized every stream still reading these shards.
        """
        nbytes = 0
        for slot in slots:
            if slot not in self._resident:
                continue
            entry = self._entry(slot)
            entry.device_tensor.untyped_storage().resize_(0)
            self._resident.discard(slot)
            nbytes += entry.nbytes
        return nbytes

    def is_resident(self, slot: int) -> bool:
        return slot in self._resident

    def mark_claimed(self, slot: int) -> None:
        """Record that a graph now carries a load for this slot."""
        self._claimed.add(slot)

    def restore_unclaimed(self, slots: Sequence[int]) -> list[str]:
        """Restore the unclaimed, nonresident ``slots`` before graph execution; return their names.
        This prevents readers without an inserted load from accessing empty storage. Only the
        compiling graph's slots: another model's are unclaimed until its own compile binds them."""
        pending = [slot for slot in dict.fromkeys(slots) if slot not in self._claimed and slot not in self._resident]
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
        """Restore all slots for teardown or execution outside the offloaded graph.
        Mark them resident so later loads read their device copies."""
        self.make_resident_many(range(len(self._slots)))

    def h2d_bandwidth_bytes_per_ns(self, override_gbps: float = 0.0) -> float:
        """Return H2D bandwidth in bytes/ns, using an override or a process-cached probe.

        Rank-local: H2D placement moves no collectives, so ranks may disagree."""
        if override_gbps > 0:
            # 1 GB/s == 1e9 bytes / 1e9 ns == 1 byte/ns, so the units coincide.
            self._bandwidth_bytes_per_ns = override_gbps
            return override_gbps
        if self._bandwidth_bytes_per_ns is not None:
            return self._bandwidth_bytes_per_ns

        self._bandwidth_bytes_per_ns = measure_h2d_bandwidth()
        magi_logger.info(
            "host offload: measured H2D bandwidth %.1f GB/s (pinned, %d MiB probe, this rank alone)",
            self._bandwidth_bytes_per_ns,
            PROBE_BYTES // 2**20,
        )
        return self._bandwidth_bytes_per_ns


# Process-wide store.  ``magi::h2d_load`` looks slots up here at runtime; a
# per-backend instance would desync from the integers already in the graph.
_STORE = HostWeightStore()


def default_store() -> HostWeightStore:
    return _STORE
