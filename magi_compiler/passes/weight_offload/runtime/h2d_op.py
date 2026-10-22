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

from .host_weights import get_global_host_weight_store
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
    store = get_global_host_weight_store()
    host = store.source(slot)
    if tuple(shard.shape) != tuple(host.shape) or shard.dtype != host.dtype:
        raise RuntimeError(
            f"magi::h2d_load slot {slot} ({store.name_of(slot)!r}) is "
            f"{tuple(host.shape)} {host.dtype}, but the graph asked for "
            f"{tuple(shard.shape)} {shard.dtype}. The compiled artifact's slot "
            "ids do not match this process's host weight store."
        )


def _h2d_load(shard: torch.Tensor, slot: int) -> torch.Tensor:
    # Resident slots copy from the device; other slots copy from pinned host storage.
    slot = resolve_slot(slot)
    _check_shard(shard, slot)
    store = get_global_host_weight_store()
    host = store.source(slot)
    # Allocate on the consuming compute stream;
    # record_stream below prevents reuse while the H2D stream is still writing.
    out = torch.empty(host.shape, dtype=host.dtype, device=shard.device)

    if store.is_resident(slot):
        # D2D runs on the compute stream without a Work; wait_tensor is then a no-op.
        # Keep a separate output: Inductor may reuse a fallback output's storage,
        # so aliasing the parameter would let a later kernel overwrite its weights.
        out.copy_(host)
        return out

    stream = h2d_stream()
    # out may reuse a block the compute stream freed before its kernel finishes;
    # the H2D stream must wait for the compute stream before writing it.
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


def _h2d_load_bwd(ctx, grad: torch.Tensor):
    # AOTAutograd traces the joint graph at compile time, so this fires before the first step runs.
    raise RuntimeError(
        "magi::h2d_load has no backward: offload_config.graph_weight_offload is inference-only, "
        "and an offloaded weight would silently get no gradient. Run the compiled module under "
        "torch.no_grad() or torch.inference_mode(), set requires_grad=False on its weights, "
        "or disable offload_config.graph_weight_offload for training."
    )


def _register() -> None:
    _LIB.define(_SCHEMA)
    _LIB.impl("h2d_load", _h2d_load, "CUDA")
    _LIB.impl("h2d_load", _h2d_load_meta, "Meta")
    torch.library.register_autograd("magi::h2d_load", _h2d_load_bwd, lib=_LIB)


_register()

# Importing this module is what makes the op exist, so this is always bound.
H2D_LOAD = torch.ops.magi.h2d_load.default
