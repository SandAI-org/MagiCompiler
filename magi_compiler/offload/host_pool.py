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

"""Pinned host storage for weight shards the compiled graph loads on demand.

One slab per dtype holds many shards: ``cudaHostAlloc`` is a driver round trip
that also pauses every other stream, and a sharded model has thousands of
weights.  Pinning is what makes the load asynchronous at all -- a pageable source
forces ``cudaMemcpyAsync`` to stage through a driver bounce buffer synchronously,
which would leave nothing for the reorder pass to overlap.

A shard is addressed by an integer ``slot``, not by a pointer.  The symmetric
memory registry can key off ``data_ptr`` because the shard stays live; binding
here *frees* the shard's CUDA storage, so its address stops being a usable key
and the slot has to ride along in the graph instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from magi_compiler.utils import magi_logger

_SLAB_BYTES = 1 << 30
"""Cap on one pinned slab (1 GiB).

Pinned memory is not swappable, so a slab is a hard reservation of host RAM; a
smaller cap bounds the over-allocation of the last, partly used slab.  A shard
larger than this gets a slab of its own.
"""

_ALIGN_BYTES = 512
"""Every shard starts on a multiple of this many bytes: the alignment the DMA
engine wants for peak host-to-device throughput."""


@dataclass(frozen=True)
class HostShard:
    """One shard parked in host memory, plus the tensor it came out of.

    ``device_tensor`` is kept so :func:`restore_all` can put the shard back
    without the caller having to remember which parameter a slot belonged to.
    """

    host: torch.Tensor
    device_tensor: torch.Tensor
    nbytes: int
    name: str = ""  # the parameter this shard came from, for the placement logs


_SLOTS: list[HostShard] = []
_SLABS: list[torch.Tensor] = []
_SLAB_USED: dict[int, int] = {}
_BY_TENSOR: dict[int, int] = {}
"""``id(local shard) -> slot``, so a second compile can find a shard it already bound.

A model compiled for several shapes produces several graphs over the SAME
parameters.  Binding frees the shard's storage the first time, and every later
graph still has to load it back -- looking the shard up by its storage address
would not work, because that is exactly what binding took away.  The pool holds
a reference to each shard, so the object stays alive and its id stays unique."""
_RESIDENT: set[int] = set()
"""Slots put back on the device because their transfer could not be hidden.

Promotion happens during scheduling, once the placement pass knows how much
compute each load actually has to hide behind -- which is the first moment
anything in the compiler is in a position to know.  The graph does not change:
``h2d_load`` still runs, it just copies device-to-device, which is ~50x the
bandwidth and effectively free.  Keeping the graph identical is deliberate, so
the compiled artifact stays cacheable regardless of what was promoted.
"""


def _slab_for(dtype: torch.dtype, numel: int) -> torch.Tensor:
    """A pinned region of ``numel`` elements, bump-allocated out of a slab.

    Returns a *view* into the slab.  Views keep the slab alive, so the slab list
    is only there to make the reservation explicit and resettable.
    """
    align = max(1, _ALIGN_BYTES // dtype.itemsize)
    want = (numel + align - 1) // align * align

    for slab in _SLABS:
        if slab.dtype is not dtype:
            continue
        used = _SLAB_USED[id(slab)]
        if used + want <= slab.numel():
            _SLAB_USED[id(slab)] = used + want
            return slab[used : used + numel]

    slab_numel = max(want, _SLAB_BYTES // dtype.itemsize)
    try:
        slab = torch.empty(slab_numel, dtype=dtype, device="cpu", pin_memory=True)
    except RuntimeError as exc:  # pinning failed -- pageable still works, just synchronously
        magi_logger.warning(
            "host offload: could not pin a %.1f MiB slab (%s); falling back to pageable host memory, "
            "which makes the load synchronous and unhideable",
            slab_numel * dtype.itemsize / 2**20,
            exc,
        )
        slab = torch.empty(slab_numel, dtype=dtype, device="cpu")
    _SLABS.append(slab)
    _SLAB_USED[id(slab)] = want
    return slab[:numel]


def bind_many(locals_: Sequence[torch.Tensor], names: Sequence[str] | None = None) -> list[int]:
    """Park every shard in pinned host memory and free its CUDA storage.

    Batched rather than per-shard because the device storage can only be dropped
    once the copy has landed, and a model has thousands of weights: one sync for
    the whole batch instead of one per weight.

    The shard tensors themselves are not replaced -- only their storage is
    resized to zero, the same in-place trick ``simple_fsdp.offload`` uses -- so
    Dynamo's guards on the parameter objects stay valid and the shape metadata
    the graph was traced with survives.  Returns one slot per input, in order.
    """
    staged: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    for local in locals_:
        host = _slab_for(local.dtype, local.numel()).view(local.shape)
        host.copy_(local, non_blocking=True)
        staged.append((host, local, local.untyped_storage().nbytes()))

    torch.cuda.synchronize()

    names = list(names or []) + [""] * max(0, len(staged) - len(names or []))
    slots: list[int] = []
    for (host, local, nbytes), name in zip(staged, names):
        local.untyped_storage().resize_(0)
        _SLOTS.append(HostShard(host=host, device_tensor=local, nbytes=nbytes, name=name))
        _BY_TENSOR[id(local)] = len(_SLOTS) - 1
        slots.append(len(_SLOTS) - 1)
    torch.cuda.empty_cache()
    return slots


def slot_of(local: torch.Tensor) -> int | None:
    """The slot this shard was bound to by an earlier compile, or None."""
    return _BY_TENSOR.get(id(local))


def name_of(slot: int) -> str:
    """The parameter behind a slot, for the placement logs."""
    return _entry(slot).name


def get(slot: int) -> torch.Tensor:
    """The pinned host shard behind ``slot``."""
    return _entry(slot).host


def source(slot: int) -> torch.Tensor:
    """Where a load should copy this shard from.

    The device copy for a promoted slot, the host copy otherwise.  The load reads
    this instead of ``get`` so promotion needs no second code path: same op, same
    stream, same event, just a much shorter wire.
    """
    entry = _entry(slot)
    return entry.device_tensor if slot in _RESIDENT else entry.host


def _entry(slot: int) -> HostShard:
    try:
        return _SLOTS[slot]
    except IndexError:
        raise RuntimeError(
            f"magi::h2d_load got slot {slot} but only {len(_SLOTS)} shard(s) are bound. "
            "The compiled graph outlived the host pool it was built against -- a cached "
            "artifact replayed in a fresh process, or reset() between compile and run."
        ) from None


def make_resident(slot: int) -> int:
    """Put a shard back on the device.  Returns the bytes this cost (0 if already there).

    Called from the placement pass for loads whose transfer no amount of hoisting
    could hide.  Such a load is pure exposed latency: it stalls the compute stream
    in front of its own all-gather and delays the launch, so paying its bytes in
    device memory buys back more than it costs.
    """
    if slot in _RESIDENT:
        return 0
    entry = _entry(slot)
    if entry.device_tensor.untyped_storage().nbytes() == 0:
        entry.device_tensor.untyped_storage().resize_(entry.nbytes)
        entry.device_tensor.copy_(entry.host)
        torch.cuda.synchronize()
    _RESIDENT.add(slot)
    return entry.nbytes


def is_resident(slot: int) -> bool:
    return slot in _RESIDENT


def slot_bytes(slot: int) -> int:
    return _entry(slot).nbytes


def resident_bytes() -> int:
    """Bytes of promoted shards, i.e. device memory offload gave back."""
    return sum(_SLOTS[s].nbytes for s in _RESIDENT)


def bound_bytes() -> int:
    """Device bytes actually freed: everything bound, less what was promoted back."""
    return sum(s.nbytes for s in _SLOTS) - resident_bytes()


def total_bound_bytes() -> int:
    """Every bound shard, promoted or not -- the base the residency budget is a fraction of."""
    return sum(s.nbytes for s in _SLOTS)


def num_bound() -> int:
    return len(_SLOTS)


def restore_all() -> None:
    """Put every bound shard back on the device.  Escape hatch for teardown and
    for running an offloaded model through an un-offloaded code path."""
    for entry in _SLOTS:
        if entry.device_tensor.untyped_storage().nbytes() == 0:
            entry.device_tensor.untyped_storage().resize_(entry.nbytes)
            entry.device_tensor.copy_(entry.host)
    torch.cuda.synchronize()


def reset() -> None:
    """Drop every slot and slab.  Tests only."""
    _SLOTS.clear()
    _SLABS.clear()
    _SLAB_USED.clear()
    _RESIDENT.clear()
    _BY_TENSOR.clear()


_BANDWIDTH_BYTES_PER_NS: float | None = None


def h2d_bandwidth_bytes_per_ns(override_gbps: float = 0.0) -> float:
    """Measured host-to-device bandwidth, in bytes per nanosecond.

    Measured once per process and reused: the reorder pass sizes thousands of
    loads, and re-timing each one would cost more than the placement it informs
    while telling us nothing new -- a pinned DMA of a contiguous shard is the
    same transfer at every size above the fixed overhead.

    The probe runs on every rank AT ONCE, which is the whole point.  Measured
    alone, one H100 reads pinned host memory at ~55 GB/s; measured with its
    seven neighbours doing the same, it gets ~27 GB/s, because what runs out is
    host memory bandwidth (~207 GB/s aggregate), not the PCIe link.  Calibrating
    in isolation hands the placement pass a number that is twice the truth, and
    it then sizes every overlap window at half the transfer it has to hide --
    which does not fail, it just quietly stops hiding things.

    The result is deliberately NOT reduced across ranks.  Load placement moves no
    collective, so ranks are free to disagree (see ``H2dLoadReorder``), and a
    rank on a slower root complex should hoist further, not adopt a peer's number.
    """
    global _BANDWIDTH_BYTES_PER_NS
    if override_gbps > 0:
        # 1 GB/s == 1e9 bytes / 1e9 ns == 1 byte/ns, so the units coincide.
        return override_gbps
    if _BANDWIDTH_BYTES_PER_NS is not None:
        return _BANDWIDTH_BYTES_PER_NS

    _BANDWIDTH_BYTES_PER_NS = _measure_h2d_bandwidth()
    magi_logger.info(
        "host offload: measured H2D bandwidth %.1f GB/s (pinned, %d MiB probe, %d rank(s) probing together)",
        _BANDWIDTH_BYTES_PER_NS,
        _PROBE_BYTES // 2**20,
        _probe_world(),
    )
    return _BANDWIDTH_BYTES_PER_NS


def _probe_world() -> int:
    import torch.distributed as dist

    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _probe_barrier() -> None:
    """Line every rank up on the probe, so it measures the contended bandwidth.

    Best-effort: a failure here costs accuracy, not correctness, and must never
    be the thing that hangs a compile.
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception as exc:  # noqa: BLE001
            magi_logger.warning("host offload: bandwidth probe barrier failed (%s); measuring unsynchronized", exc)


_PROBE_BYTES = 64 << 20
_FALLBACK_BYTES_PER_NS = 20.0
"""~20 GB/s: a conservative PCIe Gen4 x16 pinned transfer, used when the probe
cannot run.  Under-estimating bandwidth over-estimates the load, which makes the
reorder pass hoist further than needed -- slower but never incorrect."""


def _measure_h2d_bandwidth() -> float:
    if not torch.cuda.is_available():
        return _FALLBACK_BYTES_PER_NS
    try:
        src = torch.empty(_PROBE_BYTES, dtype=torch.uint8, device="cpu", pin_memory=True)
        dst = torch.empty(_PROBE_BYTES, dtype=torch.uint8, device="cuda")
        stream = torch.cuda.current_stream()
        for _ in range(2):  # warm the driver's mapping before timing
            dst.copy_(src, non_blocking=True)
        stream.synchronize()

        # Enough iterations that the ranks stay overlapped for the whole window
        # rather than straggling apart after the barrier.
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        iters = 20
        _probe_barrier()
        start.record(stream)
        for _ in range(iters):
            dst.copy_(src, non_blocking=True)
        end.record(stream)
        stream.synchronize()
        _probe_barrier()
        elapsed_ns = start.elapsed_time(end) * 1e6
        if elapsed_ns <= 0:
            return _FALLBACK_BYTES_PER_NS
        return _PROBE_BYTES * iters / elapsed_ns
    except RuntimeError as exc:  # noqa: BLE001
        magi_logger.warning("host offload: H2D bandwidth probe failed (%s); assuming %.1f GB/s", exc, _FALLBACK_BYTES_PER_NS)
        return _FALLBACK_BYTES_PER_NS
