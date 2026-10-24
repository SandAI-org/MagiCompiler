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

Pinned buffers support asynchronous H2D copies. ``reserve_host_buffer`` supplies storage;
``bind_host_buffer`` pairs a filled host buffer with a storage-free CUDA stand-in by identity.
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

from .bandwidth import H2D_PROBE_BYTES, measure_h2d_bandwidth_bytes_per_ns
from .host_memory_pool import HostMemPool


@dataclass(frozen=True)
class HostWeight:
    """One shard parked in host memory, plus the tensor it came out of.

    ``device_standin`` is kept so :meth:`HostWeightStore.restore_all_weights` can put the shard back
    without the caller having to remember which parameter a slot belonged to.
    """

    host_buffer: torch.Tensor
    device_standin: torch.Tensor
    nbytes: int
    # The parameter this shard came from; with shape and dtype it is the identity
    # a cached artifact's slots are matched by in another process (slot_table.py).
    parameter_name: str = ""


class HostWeightStore:
    """Slot table + residency state for one process's host-offloaded weights.

    Binding is idempotent on tensor identity: a second compile of the same
    stand-in must reuse the slot the first one minted.  ``_resident_slots`` and the
    device tensor's storage are the same question -- ``restore_all_weights`` /
    ``make_weight_resident`` / ``release_resident_weights`` always update both.
    """

    def __init__(self, pool: HostMemPool | None = None) -> None:
        self._pool = pool if pool is not None else HostMemPool(pinned=True)
        self._weights_by_slot: list[HostWeight] = []
        # ``id(local shard) -> slot``.  The store holds a reference to each
        # shard, so the object stays alive and its id stays unique even after
        # its CUDA storage is gone.
        self._slot_by_tensor_id: dict[int, int] = {}
        self._slots_with_h2d_load: set[int] = set()
        self._resident_slots: set[int] = set()
        self._bandwidth_bytes_per_ns: float | None = None

    def reset(self) -> None:
        """Drop every slot, slab and cached probe.  Tests only."""
        self._weights_by_slot.clear()
        self._pool.reset()
        self._resident_slots.clear()
        self._slots_with_h2d_load.clear()
        self._slot_by_tensor_id.clear()
        self._bandwidth_bytes_per_ns = None

    def _register_weight(
        self, host_buffer: torch.Tensor, device_standin: torch.Tensor, nbytes: int, parameter_name: str
    ) -> int:
        slot = len(self._weights_by_slot)
        self._weights_by_slot.append(
            HostWeight(host_buffer=host_buffer, device_standin=device_standin, nbytes=nbytes, parameter_name=parameter_name)
        )
        self._slot_by_tensor_id[id(device_standin)] = slot
        return slot

    def _weight_for_slot(self, slot: int) -> HostWeight:
        try:
            return self._weights_by_slot[slot]
        except IndexError:
            raise RuntimeError(
                f"magi::h2d_load got slot {slot} but only {len(self._weights_by_slot)} shard(s) are bound. "
                "The compiled graph outlived the host weight store it was built against -- a cached "
                "artifact replayed in a fresh process, or reset() between compile and run."
            ) from None

    def reserve_host_buffer(self, shape: Sequence[int], dtype: torch.dtype, parameter_name: str = "") -> torch.Tensor:
        """Reserve pinned host storage that a checkpoint loader can fill directly.
        No slot is created until ``bind_host_buffer`` pairs it with a CUDA stand-in."""
        host_buffer = self._pool.alloc(shape, dtype)
        magi_logger.debug(
            "host offload: reserved %.1f MiB of pinned host memory for %s",
            host_buffer.numel() * host_buffer.element_size() / 2**20,
            parameter_name,
        )
        return host_buffer

    def bind_host_buffer(self, host_buffer: torch.Tensor, device_standin: torch.Tensor, parameter_name: str = "") -> int:
        """Pair a filled host buffer with a storage-free device tensor.  Returns its slot.

        No copy: the bytes are already where they belong.  No ``resize_(0)``:
        ``device_standin`` is expected to arrive empty -- it exists only to carry
        shape, dtype and device into the graph.

        Rebinding the same device tensor returns the slot already minted for
        it.  A second compile must not create a second slot over the same
        stand-in.
        """
        existing = self._slot_by_tensor_id.get(id(device_standin))
        if existing is not None:
            return existing
        if device_standin.untyped_storage().nbytes() != 0:
            raise ValueError(
                f"host offload: bind_host_buffer({parameter_name!r}) wants a device tensor with no storage behind it, but got "
                f"{device_standin.untyped_storage().nbytes()} byte(s).  A tensor that still holds device "
                "bytes is not an offload candidate -- materialize it in host memory instead."
            )
        return self._register_weight(
            host_buffer=host_buffer,
            device_standin=device_standin,
            nbytes=host_buffer.numel() * host_buffer.element_size(),
            parameter_name=parameter_name,
        )

    def slot_for_tensor(self, device_standin: torch.Tensor) -> int | None:
        """The slot this stand-in was adopted under, or None."""
        return self._slot_by_tensor_id.get(id(device_standin))

    def parameter_name(self, slot: int) -> str:
        """The parameter behind a slot; part of its cross-process cache identity."""
        return self._weight_for_slot(slot).parameter_name

    def host_buffer(self, slot: int) -> torch.Tensor:
        """The pinned host shard behind ``slot``."""
        return self._weight_for_slot(slot).host_buffer

    def copy_source(self, slot: int) -> torch.Tensor:
        """Return the device copy for a resident slot, otherwise the pinned host copy."""
        entry = self._weight_for_slot(slot)
        return entry.device_standin if slot in self._resident_slots else entry.host_buffer

    def validated_copy_source(self, *, shard: torch.Tensor, slot: int) -> torch.Tensor:
        copy_source = self.copy_source(slot)
        if tuple(shard.shape) != tuple(copy_source.shape) or shard.dtype != copy_source.dtype:
            raise RuntimeError(
                f"magi::h2d_load slot {slot} ({self.parameter_name(slot)!r}) is "
                f"{tuple(copy_source.shape)} {copy_source.dtype}, but the graph asked for "
                f"{tuple(shard.shape)} {shard.dtype}. The compiled artifact's slot "
                "ids do not match this process's host weight store."
            )
        return copy_source

    def make_weight_resident(self, slot: int) -> int:
        """Put a shard back on the device.  Returns the bytes this cost (0 if already there)."""
        return self.make_weights_resident([slot])

    def make_weights_resident(self, slots: Sequence[int]) -> int:
        """Restore listed shards, synchronizing once if any copies were submitted.
        Return newly resident bytes; already resident slots contribute zero."""
        promoted_bytes = 0
        copied_any_weight = False
        for slot in slots:
            if slot in self._resident_slots:
                continue
            entry = self._weight_for_slot(slot)
            if entry.device_standin.untyped_storage().nbytes() == 0:
                entry.device_standin.untyped_storage().resize_(entry.nbytes)
                # An unsharded nn.Parameter is itself the stand-in the slot is
                # keyed on.  Promotion runs at compile time, not under autograd.
                with torch.no_grad():
                    entry.device_standin.copy_(entry.host_buffer, non_blocking=True)
                copied_any_weight = True
            self._resident_slots.add(slot)
            promoted_bytes += entry.nbytes
        if copied_any_weight:
            torch.cuda.synchronize()
        return promoted_bytes

    def release_resident_weights(self, slots: Sequence[int]) -> int:
        """Free the device copy of every listed promoted shard.  Returns the bytes freed.

        The inverse of :meth:`make_weights_resident`, for a caller that parks a whole
        model between uses.  The host copy stays, so a released slot loads over
        H2D again and ``make_weights_resident`` brings the device copy back.  The
        caller must have synchronized every stream still reading these shards.
        """
        released_bytes = 0
        for slot in slots:
            if slot not in self._resident_slots:
                continue
            entry = self._weight_for_slot(slot)
            entry.device_standin.untyped_storage().resize_(0)
            self._resident_slots.discard(slot)
            released_bytes += entry.nbytes
        return released_bytes

    def is_resident(self, slot: int) -> bool:
        return slot in self._resident_slots

    def mark_slot_with_h2d_load(self, slot: int) -> None:
        """Record that a graph now carries an H2D load for this slot."""
        self._slots_with_h2d_load.add(slot)

    def restore_weights_without_h2d_load(self, slots: Sequence[int]) -> list[str]:
        """Restore nonresident weights without H2D loads before execution; return their names.
        Only pass the compiling graph's slots: other models may not have compiled yet."""
        slots_to_restore = [
            slot for slot in dict.fromkeys(slots) if slot not in self._slots_with_h2d_load and slot not in self._resident_slots
        ]
        if not slots_to_restore:
            return []
        self.make_weights_resident(slots_to_restore)
        return [self._weights_by_slot[slot].parameter_name or f"slot {slot}" for slot in slots_to_restore]

    def slot_bytes(self, slot: int) -> int:
        return self._weight_for_slot(slot).nbytes

    def resident_bytes(self) -> int:
        """Bytes of promoted shards, i.e. device memory offload gave back."""
        return sum(self._weights_by_slot[s].nbytes for s in self._resident_slots)

    def offloaded_bytes(self) -> int:
        """Device bytes actually freed: everything bound, less what was promoted back."""
        return sum(s.nbytes for s in self._weights_by_slot) - self.resident_bytes()

    def total_weight_bytes(self) -> int:
        """Every bound shard, promoted or not -- the base the residency budget is a fraction of."""
        return sum(s.nbytes for s in self._weights_by_slot)

    def weight_count(self) -> int:
        return len(self._weights_by_slot)

    def restore_all_weights(self) -> None:
        """Restore all slots for teardown or execution outside the offloaded graph.
        Mark them resident so later loads read their device copies."""
        self.make_weights_resident(range(len(self._weights_by_slot)))

    def h2d_bandwidth_bytes_per_ns(self, override_gbps: float = 0.0) -> float:
        """Return H2D bandwidth in bytes/ns, using an override or a process-cached probe.

        Rank-local: H2D placement moves no collectives, so ranks may disagree."""
        if override_gbps > 0:
            # 1 GB/s == 1e9 bytes / 1e9 ns == 1 byte/ns, so the units coincide.
            self._bandwidth_bytes_per_ns = override_gbps
            return override_gbps
        if self._bandwidth_bytes_per_ns is not None:
            return self._bandwidth_bytes_per_ns

        self._bandwidth_bytes_per_ns = measure_h2d_bandwidth_bytes_per_ns()
        magi_logger.info(
            "host offload: measured H2D bandwidth %.1f GB/s (pinned, %d MiB probe, this rank alone)",
            self._bandwidth_bytes_per_ns,
            H2D_PROBE_BYTES // 2**20,
        )
        return self._bandwidth_bytes_per_ns


# Process-wide store.  ``magi::h2d_load`` looks slots up here at runtime; a
# per-backend instance would desync from the integers already in the graph.
_GLOBAL_HOST_WEIGHT_STORE = HostWeightStore()


def get_global_host_weight_store() -> HostWeightStore:
    return _GLOBAL_HOST_WEIGHT_STORE
