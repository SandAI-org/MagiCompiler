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

_LIB = torch.library.Library("magi", "FRAGMENT")
_SCHEMA = "h2d_load(Tensor shard, int slot) -> Tensor"
_SCHEMA_COALESCED = "h2d_load_coalesced(Tensor[] shards, int[] slots) -> Tensor[]"


@lru_cache(maxsize=1)
def h2d_stream() -> torch.cuda.Stream:
    """The one stream every offloaded weight load is submitted on.

    A single stream serializes the loads, which is what the reorder pass assumes
    when it hands each load a disjoint run of compute to hide behind: the DMA
    engines would not go faster for being asked twice at once anyway.
    """
    return torch.cuda.Stream()


def _issue_loads(shards: list[torch.Tensor], slots: list[int]) -> list[torch.Tensor]:
    """Submit every load as one batch; publish one event for all of them.

    The stream sync, the event and its ``Work`` are paid once for the whole
    bucket rather than once per weight.  That fixed cost is ~10us of CPU per
    load, which on a model with hundreds of weights is milliseconds of launch
    overhead sitting directly in front of the first all-gather -- and it inflates
    the very window the reorder pass is trying to size.
    """
    # ``source``, not ``get``: a slot the placement pass promoted back onto the
    # device copies from there instead, which turns this into a D2D copy without
    # any other part of the op, the graph or the schedule having to know.
    hosts = [host_pool.source(slot) for slot in slots]
    for shard, host, slot in zip(shards, hosts, slots):
        if tuple(shard.shape) != tuple(host.shape) or shard.dtype != host.dtype:
            raise RuntimeError(
                f"magi::h2d_load slot {slot} ({host_pool.name_of(slot)!r}) is "
                f"{tuple(host.shape)} {host.dtype}, but the graph asked for "
                f"{tuple(shard.shape)} {shard.dtype}. The compiled artifact's slot "
                "ids do not match this process's host pool."
            )
    # Allocated on the COMPUTE stream, deliberately: the caching allocator ties a
    # block to the stream it was allocated on, and these buffers are consumed by
    # compute.  ``record_stream`` below is what tells it the load stream wrote
    # them, so a freed block is not handed out before the copy lands.
    outs = [torch.empty(h.shape, dtype=h.dtype, device=s.device) for s, h in zip(shards, hosts)]

    if all(host_pool.is_resident(slot) for slot in slots):
        # Nothing to hide and nothing to wait for: every shard in this group is
        # already on the device, so the transfer is a short D2D hop rather than
        # a trip across PCIe.  Doing it inline on the compute stream skips two
        # cross-stream synchronizations, the event and its Work registration --
        # all of which exist to overlap a transfer that no longer happens.  The
        # ``wait_tensor`` downstream then finds no Work and is a no-op.
        #
        # The buffer is still a real copy, and that is not an oversight: a
        # promoted slot's source IS the shard the graph handed us, so returning
        # it would make this op's output alias a graph input.  Inductor does not
        # allocate a fallback kernel's output but does put it in the reuse pool,
        # so the next same-sized allocation would take over the parameter's
        # storage and the following kernel would write into the weight.
        #
        # Whole group or nothing.  Promotion is per load and a load is a bucket,
        # so a mixed group does not arise.
        for out, host in zip(outs, hosts):
            out.copy_(host)
        return outs

    stream = h2d_stream()
    # The shards' own producers are on the compute stream; ordering after them
    # costs nothing here and keeps the op correct if a caller ever writes them.
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for out, host in zip(outs, hosts):
            out.copy_(host, non_blocking=True)
        event = torch.cuda.Event()
        event.record(stream)

    for out in outs:
        out.record_stream(stream)
        # The registry takes ownership of each Work; members share the event.
        _c10d._register_work(out, EventWork(event))
    return outs


def _h2d_load(shard: torch.Tensor, slot: int) -> torch.Tensor:
    return _issue_loads([shard], [slot])[0]


def _h2d_load_meta(shard: torch.Tensor, slot: int) -> torch.Tensor:
    return torch.empty_like(shard)


def _h2d_load_coalesced(shards: list[torch.Tensor], slots: list[int]) -> list[torch.Tensor]:
    return _issue_loads(list(shards), list(slots))


def _h2d_load_coalesced_meta(shards: list[torch.Tensor], slots: list[int]) -> list[torch.Tensor]:
    return [torch.empty_like(s) for s in shards]


def _register() -> None:
    _LIB.define(_SCHEMA)
    _LIB.impl("h2d_load", _h2d_load, "CUDA")
    _LIB.impl("h2d_load", _h2d_load_meta, "Meta")

    _LIB.define(_SCHEMA_COALESCED)
    _LIB.impl("h2d_load_coalesced", _h2d_load_coalesced, "CUDA")
    _LIB.impl("h2d_load_coalesced", _h2d_load_coalesced_meta, "Meta")


_register()

# Importing this module is what makes the ops exist, so these are always bound.
H2D_LOAD = torch.ops.magi.h2d_load.default
H2D_LOAD_COALESCED = torch.ops.magi.h2d_load_coalesced.default
