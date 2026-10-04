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

"""``magi::h2d_load``: pull an offloaded weight shard back onto the device.

Modelled on ``magi::ce_all_gather``: the copy runs on a stream of its own and
publishes a CUDA event as a c10d ``Work``, so the ordinary
``_c10d_functional::wait_tensor`` is what synchronizes it.  That choice is what
makes the load schedulable -- Inductor lowers the op to a ``FallbackKernel`` and
the wait to a ``_WaitKernel``, which are two snodes the reorder passes can move
independently, rather than one opaque blocking copy.

``shard`` is the parameter's local stand-in -- a CUDA tensor whose storage was
never filled, after host-first adopt.  It is here to carry shape/dtype/device
and to give the graph a real data edge from the weight placeholder.  The bytes
come from ``slot``.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch._C._distributed_c10d as _c10d

from magi_compiler.cuda.event_work import EventWork

from . import host_pool
from .slot_remap import resolve_slot

_LIB = torch.library.Library("magi", "FRAGMENT")
_SCHEMA = "h2d_load(Tensor shard, int slot) -> Tensor"


@lru_cache(maxsize=1)
def h2d_stream() -> torch.cuda.Stream:
    """The one stream every offloaded weight load is submitted on.

    A single stream serializes the loads, which is what the reorder pass assumes
    when it hands each load a disjoint run of compute to hide behind: the DMA
    engines would not go faster for being asked twice at once anyway.
    """
    return torch.cuda.Stream()


def _check_shard(shard: torch.Tensor, slot: int) -> None:
    host = host_pool.source(slot)
    if tuple(shard.shape) != tuple(host.shape) or shard.dtype != host.dtype:
        raise RuntimeError(
            f"magi::h2d_load slot {slot} ({host_pool.name_of(slot)!r}) is "
            f"{tuple(host.shape)} {host.dtype}, but the graph asked for "
            f"{tuple(shard.shape)} {shard.dtype}. The compiled artifact's slot "
            "ids do not match this process's host pool."
        )


def _h2d_load(shard: torch.Tensor, slot: int) -> torch.Tensor:
    # ``source``, not ``get``: a slot the placement pass promoted back onto the
    # device copies from there instead, which turns this into a D2D copy without
    # any other part of the op, the graph or the schedule having to know.
    slot = resolve_slot(slot)
    _check_shard(shard, slot)
    host = host_pool.source(slot)
    # Allocated on the COMPUTE stream, deliberately: the caching allocator ties a
    # block to the stream it was allocated on, and this buffer is consumed by
    # compute.  ``record_stream`` below is what tells it the load stream wrote
    # it, so a freed block is not handed out before the copy lands.
    out = torch.empty(host.shape, dtype=host.dtype, device=shard.device)

    if host_pool.is_resident(slot):
        # Nothing to hide and nothing to wait for: the shard is already on the
        # device, so the transfer is a short D2D hop rather than a trip across
        # PCIe.  Doing it inline on the compute stream skips two cross-stream
        # synchronizations, the event and its Work registration -- all of which
        # exist to overlap a transfer that no longer happens.  The
        # ``wait_tensor`` downstream then finds no Work and is a no-op.
        #
        # The buffer is still a real copy, and that is not an oversight: a
        # promoted slot's source IS the shard the graph handed us, so returning
        # it would make this op's output alias a graph input.  Inductor does not
        # allocate a fallback kernel's output but does put it in the reuse pool,
        # so the next same-sized allocation would take over the parameter's
        # storage and the following kernel would write into the weight.
        out.copy_(host)
        return out

    stream = h2d_stream()
    # The shard's own producers are on the compute stream; ordering after them
    # costs nothing here and keeps the op correct if a caller ever writes it.
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out.copy_(host, non_blocking=True)
        event = torch.cuda.Event()
        event.record(stream)
    out.record_stream(stream)
    # The registry takes ownership of the Work.
    _c10d._register_work(out, EventWork(event))
    return out


def _h2d_load_meta(shard: torch.Tensor, slot: int) -> torch.Tensor:
    return torch.empty_like(shard)


def _register() -> None:
    _LIB.define(_SCHEMA)
    _LIB.impl("h2d_load", _h2d_load, "CUDA")
    _LIB.impl("h2d_load", _h2d_load_meta, "Meta")


_register()

# Importing this module is what makes the op exist, so this is always bound.
H2D_LOAD = torch.ops.magi.h2d_load.default
